import os
from absl import logging
import tensorflow as tf

from official.core import base_task
from official.core import task_factory
from official.core import train_utils
from typing import Any, Dict, List, Optional, Tuple

from official.vision.dataloaders import input_reader_factory
from official.common import dataset_fn

from official.projects.maskformer.configs import maskformer as maskformer_cfg
from official.projects.maskformer.modeling.maskformer import MaskFormer
from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.maskformer.dataloaders import panoptic_input

from official.projects.detr.ops.matchers import hungarian_matching
from official.projects.maskformer.losses.maskformer_losses import Loss

from official.vision.evaluation import panoptic_quality
from official.projects.maskformer.losses.inference import PanopticInference
from official.vision.modeling import backbones

from official.projects.maskformer.losses.mapper import _get_contigious_to_original, _get_original_to_contigious, _get_original_is_thing
from official.core.train_utils import try_count_params
import numpy as np


@task_factory.register_task_cls(maskformer_cfg.MaskFormerTask)
class PanopticTask(base_task.Task):
	"""A single-replica view of training procedure.

	PanopticTask task provides artifacts for training/evalution procedures, including
	loading/iterating over Datasets, initializing the model, calculating the loss,
	post-processing, and customized metrics with reduction.
	"""
	
	def build_model(self):
		"""Builds MaskFormer Model."""
		logging.info('Building MaskFormer model.')
		input_specs = tf.keras.layers.InputSpec(shape=[None] + self._task_config.model.input_size)
		
		backbone = backbones.factory.build_backbone(input_specs=input_specs,
					backbone_config=self._task_config.model.backbone,
					norm_activation_config=self._task_config.model.norm_activation)
		logging.info('Backbone build successful.')
		model = MaskFormer(backbone=backbone,input_specs= input_specs,
							num_queries=self._task_config.model.num_queries,
							hidden_size=self._task_config.model.hidden_size,
							backbone_endpoint_name=self._task_config.model.backbone_endpoint_name,
							fpn_encoder_layers=self._task_config.model.fpn_encoder_layers,
							detr_encoder_layers=self._task_config.model.detr_encoder_layers,
							num_decoder_layers=self._task_config.model.num_decoder_layers,
							num_classes=self._task_config.model.num_classes,
							bfloat16=self._task_config.bfloat16, 
							which_pixel_decoder=self._task_config.model.which_pixel_decoder,
							deep_supervision=self._task_config.model.deep_supervision,
							)
		logging.info('Maskformer model build successful.')
		self.DATA_IDX = 0
		return model

	def initialize(self, model: tf.keras.Model) -> None:
		"""
		Used to initialize the models with checkpoint
		"""
		
		logging.info('Initializing model from checkpoint: %s', self._task_config.init_checkpoint)
		if not self._task_config.init_checkpoint:
			return
		ckpt_dir_or_file = self._task_config.init_checkpoint

		# Restoring ckpt
		if tf.io.gfile.isdir(ckpt_dir_or_file):
			ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

		if self._task_config.init_checkpoint_modules == 'all':
			ckpt = tf.train.Checkpoint(model=model)
			status = ckpt.restore(ckpt_dir_or_file)
			status.expect_partial().assert_existing_objects_matched()
			logging.info('Loaded whole model from %s',ckpt_dir_or_file)
			
		elif self._task_config.init_checkpoint_modules == 'backbone':
			ckpt = tf.train.Checkpoint(backbone=model.backbone)
			status = ckpt.restore(ckpt_dir_or_file)
			status.expect_partial().assert_existing_objects_matched()
			logging.info('Finished loading backbone checkpoint from %s',
					ckpt_dir_or_file)
		else:
			raise ValueError('Not a valid module to initialize from: {}'.format(
				self._task_config.init_checkpoint_modules))

	def build_inputs(self, params, input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
		""" 
		Build panoptic segmentation dataset.

		"""
		logging.info('Building panoptic segmentation dataset.')
		if params.decoder.type == 'simple_decoder':
			decoder = panoptic_input.TfExampleDecoder(regenerate_source_id = params.regenerate_source_id)
		else:
			raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))
		
		parser = panoptic_input.mask_former_parser(params.parser, is_training = params.is_training, decoder_fn=decoder.decode)
	
		reader = input_reader_factory.input_reader_generator(
		  params,
		  dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
		  decoder_fn=decoder.decode,
		  parser_fn=parser.parse_fn(params.is_training))
		
		dataset = reader.read(input_context=input_context)
		return dataset

	def build_losses(self, output, labels, aux_outputs=None):
		outputs = {"pred_logits": output["class_prob_predictions"],
				   "pred_masks": output["mask_prob_predictions"]}

		if aux_outputs:
			outputs["pred_masks"] = output["mask_prob_predictions"][-1]
			outputs["pred_logits"] = output["class_prob_predictions"][-1]

			formatted_aux_output = [
					{"pred_logits": a, "pred_masks": b}
					for a, b in zip(output["class_prob_predictions"][:-1], output["mask_prob_predictions"][:-1])
				]

			outputs.update({"aux_outputs": formatted_aux_output})

		if os.environ.get('PRINT_OUTPUTS') == 'True':
			if aux_outputs:
				for i, aux_output in enumerate(outputs["aux_outputs"]):
					print(f'LOGGING LOSSES: {i}: {[(x, aux_output[x].shape) for x in aux_output]}')

		targets = labels

		matcher = hungarian_matching

		no_object_weight = self._task_config.losses.background_cls_weight
		loss = Loss(num_classes = self._task_config.model.num_classes,
					matcher = matcher,
					eos_coef = no_object_weight,
					cost_class= self._task_config.losses.cost_class,
					cost_dice= self._task_config.losses.cost_dice,
					cost_focal= self._task_config.losses.cost_focal, ignore_label=self._task_config.train_data.parser.ignore_label)

		calculated_losses = loss(outputs, targets)
		
		if os.environ.get('PRINT_OUTPUTS') == 'True':
			print('LOGGING LOSSES: ', {(x, calculated_losses[x].numpy()) for x in calculated_losses})
		
		# Losses are returned as weighted sum of individual losses
		total_loss = calculated_losses['loss_ce'] + calculated_losses['loss_dice'] + calculated_losses['loss_focal']

		weighted_ce = calculated_losses['loss_ce']
		weighted_dice = calculated_losses['loss_dice']
		weighted_focal = calculated_losses['loss_focal']

		aux_outputs = output.get('aux_outputs')
	
		if aux_outputs is not None:
			total_aux_loss = 0.0
			for i in range(len(aux_outputs)): 
				total_aux_loss += calculated_losses['loss_ce_'+str(i)] + calculated_losses['loss_dice_'+str(i)] + calculated_losses['loss_focal_'+str(i)]
			total_loss = total_loss + total_aux_loss

		return total_loss, weighted_ce, weighted_focal, weighted_dice
		
	def build_metrics(self, training=True):
		"""Builds panoptic metrics."""
		metrics = []
		metric_names = ['cls_loss', 'focal_loss', 'dice_loss']
		for name in metric_names:
			metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
		
		if not training:
			print("[INFO] Building panoptic quality metric ")
			_, _, thing_tensor_bool = _get_contigious_to_original()
			self.is_thing_dict_bool = thing_tensor_bool
			pq_config = self._task_config.panoptic_quality_evaluator

			self.map_original_to_contigious = _get_original_to_contigious()

			if self._task_config.model.on_tpu: 
				self.panoptic_quality_metric = panoptic_quality.PanopticQualityV2(
					num_categories=pq_config.num_categories,
					is_thing=self.is_thing_dict_bool,
					ignored_label=pq_config.ignored_label,
					max_num_instances=pq_config.max_num_instances,
					rescale_predictions=pq_config.rescale_predictions,
				)
			else: 
				self.panoptic_quality_metric = panoptic_quality.PanopticQuality(
					num_categories=pq_config.num_categories,
					ignored_label=pq_config.ignored_label,
					max_instances_per_category=pq_config.max_num_instances,
					offset=pq_config.max_num_instances**3,
				)

			self.panoptic_inference = PanopticInference(
				num_classes=self._task_config.model.num_classes, 
				background_class_id=pq_config.ignored_label
			)

		return metrics
		
		
	def train_step(self, inputs: Tuple[Any, Any], model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, metrics: Optional[List[Any]] = None) -> Dict[str, Any]:
		"""
		Does forward and backward.

		Args:
		inputs: a dictionary of input tensors.
		model: the model, forward pass definition.
		optimizer: the optimizer for this training step.
		metrics: a nested structure of metrics objects.

		Returns:
		A dictionary of logs.
		"""
						
		features, labels = inputs
		
		
		with tf.GradientTape() as tape:
			if os.environ.get('PRINT_OUTPUTS') == 'True':
				print('------------------------------')
				print('LOGGING STEP: Starting forward pass')
			outputs = model(features, training=True)

			if os.environ.get('PRINT_OUTPUTS') == 'True':
				print('LOGGING STEP: Starting losses')
				print(f'LOGGING STEP: DEEP_SUPERVISION: {model._deep_supervision}')
			total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels,  aux_outputs=model._deep_supervision)
			scaled_loss = total_loss
			
			if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
				total_loss = optimizer.get_scaled_loss(scaled_loss)
				
		tvars = model.trainable_variables	
		grads = tape.gradient(scaled_loss,tvars)
	
		if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
			grads = optimizer.get_unscaled_gradients(grads)		
		optimizer.apply_gradients(list(zip(grads, tvars)))
	
		if os.environ.get('PRINT_OUTPUTS') == 'True':
			probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
			pred_labels = tf.argmax(probs, axis=-1)
			unique_elements = []
			for i in range(pred_labels.shape[0]):
					unique_elements.append(pred_labels[i, 0, 0].numpy())
			print('LOGGING STEP: Pred Labels: ', unique_elements)
		
		# # Multiply for logging.
		# # Since we expect the gradient replica sum to happen in the optimizer,
		# # the loss is scaled with global num_boxes and weights.
		# # To have it more interpretable/comparable we scale it back when logging.
		num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
		total_loss *= num_replicas_in_sync
		cls_loss *= num_replicas_in_sync
		focal_loss *= num_replicas_in_sync
		dice_loss *= num_replicas_in_sync
		
		logs = {self.loss: total_loss}

		all_losses = {
			'cls_loss': cls_loss,
			'focal_loss': focal_loss,
			'dice_loss': dice_loss,}
		
		if metrics:
			for m in metrics:
				m.update_state(all_losses[m.name])

		if os.environ.get('PRINT_OUTPUTS') == 'True':
			print('LOGGING STEP: Finished forward pass')
			print('LOGGING STEP: ------------------------------')
		
		return logs
		
	def _postprocess_outputs(self, outputs: Dict[str, Any], image_shapes, deep_supervision: bool):
		""" 
		Implements postprocessing using the output binary masks and labels to produce
		1. Output Category Mask
		2. Output Instance Mask

		"""
		pred_binary_masks = outputs["mask_prob_predictions"]
		pred_labels = outputs["class_prob_predictions"]
		if deep_supervision:
			pred_binary_masks = pred_binary_masks[-1]
			pred_labels = pred_labels[-1]

		if os.environ.get('PRINT_OUTPUTS') == 'True':
			print(f"LOGGING STEP: Pred_Binary_Masks: {pred_binary_masks.shape} || Pred_Labels: {pred_labels.shape}")

		ouput_instance_mask, output_category_mask = self.panoptic_inference(pred_labels, pred_binary_masks, image_shapes)
		return ouput_instance_mask, output_category_mask

	def validation_step(self, inputs, model, metrics=None):
		if os.environ.get('PRINT_OUTPUTS') == 'True':
			print('LOGGING STEP: ------------------------------')
			print('LOGGING STEP: Starting forward pass')
		
		features, labels = inputs

		outputs = model(features, training=False)

		if os.environ.get('PRINT_OUTPUTS') == 'True':
			print('LOGGING STEP: Starting losses')
			print(f'LOGGING STEP: DEEP_SUPERVISION: {model._deep_supervision}')

		total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels, aux_outputs=model._deep_supervision)

		num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
		total_loss *= num_replicas_in_sync
		cls_loss *= num_replicas_in_sync
		focal_loss *= num_replicas_in_sync
		dice_loss *= num_replicas_in_sync
		logs = {self.loss: total_loss}

		all_losses = {
				'cls_loss': cls_loss,
				'focal_loss': focal_loss,
				'dice_loss': dice_loss,
			}

		if os.environ.get('PRINT_OUTPUTS') == 'True':
			probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
			pred_labels = tf.argmax(probs, axis=-1)
			unique_elements = []
			for i in range(pred_labels.shape[0]):
					unique_elements.append(pred_labels[i, 0, 0].numpy())
			print('LOGGING STEP: Pred Labels: ', unique_elements)

		
		pq_metric_labels = {    
		'category_mask': labels['category_mask'],
		'instance_mask': labels['instance_mask']
		}

		output_instance_mask, output_category_mask = self._postprocess_outputs(outputs, [640, 640], model._deep_supervision)

		if output_instance_mask.shape[0] == 0 or output_category_mask.shape[0] == 0:
			raise ValueError('Post Processed Predictions are empty.')

		pq_metric_outputs = {
		'category_mask': output_category_mask,
		'instance_mask': output_instance_mask
		}

		if os.environ.get('PRINT_OUTPUTS') == 'True':
			print('LOGGING STEP: Starting PQ Metric Logging')

		results = self._generate_panoptic_metrics(pq_metric_labels, pq_metric_outputs)

		for key, value in results.items():
			logs[key] = value

		if metrics:
			for m in metrics:
				m.update_state(all_losses[m.name])

		if os.environ.get('PRINT_OUTPUTS') == 'True':
			print('LOGGING STEP: Finished forward pass')
			print('LOGGING STEP: LOGS: ', logs)
			print('LOGGING STEP: METRICS: ', metrics)
			print('LOGGING STEP: ------------------------------')

		self.DATA_IDX += 1

		exit()

		if self.DATA_IDX > 5: 
			exit() 
		
		return logs

	
	def _generate_panoptic_metrics(self, pq_metric_labels, pq_metric_outputs):
		# Mapping labels from original to contigious 
		# Processed outputs are converted to original ids in _postprocess_outputs, and here we will convert them back to contigious ids
		pq_metric_labels = {key:self.map_original_to_contigious(value.numpy()) for key, value in pq_metric_labels.items()}
		pq_metric_outputs = {key:self.map_original_to_contigious(value.numpy()) for key, value in pq_metric_outputs.items()}

		# There are different PQ implementations for TPU and GPU/CPU
		# TPU implementation requires a lot of memory bandwidth 
		if self._task_config.model.on_tpu:
			self.panoptic_quality_metric.update_state(
				{key:tf.convert_to_tensor(value) for key, value in pq_metric_labels.items()}, 
				{key:tf.convert_to_tensor(value) for key, value in pq_metric_outputs.items()}
			)
			results = self.panoptic_quality_metric.result()
		else: 
			self.panoptic_quality_metric.compare_and_accumulate(
				pq_metric_labels, 
				pq_metric_outputs
			)
			results = self.panoptic_quality_metric.result(_get_original_is_thing())

		if os.environ.get('PRINT_OUTPUTS') == 'True':
			print('LOGGING PQ: PQ Metrics: ', results)
		
		return self.reduce_aggregated_results(results)


	def reduce_aggregated_results(self, aggregated_results: Dict[str, Any]):
		reduced_metrics = self._reduce_metrics(aggregated_results)

		if self._task_config.model.on_tpu:
			self.panoptic_quality_metric.reset_state()	
		else: 
			self.panoptic_quality_metric.reset()

		return reduced_metrics

	def _reduce_metrics(self, results: Dict[str, Any]):
		if self._task_config.model.on_tpu:
			return self._reduce_panoptic_metrics_v2(results)
		else:
			return self._reduce_panoptic_metrics(results)
	
	def _reduce_panoptic_metrics(self, results: Dict[str, Any]):
		reduced_metrics = {} 

		categories = ['All', 'Things', 'Stuff']
		for category in categories:
				key = f'panoptic_quality/{category}_num_categories'
				reduced_metrics[key] = results[f'{category}_num_categories']

		categories = ['All', 'Things', 'Stuff']
		metrics = ['pq', 'rq', 'sq']
		for category in categories:
				for metric in metrics:
						key = f'panoptic_quality/{category}_{metric}'
						reduced_metrics[key] = results[f'{category}_{metric}']

		metrics = ['pq', 'rq', 'sq']
		for metric in metrics:
				for i in range(1, self._task_config.panoptic_quality_evaluator.num_categories+1):  
						key = f'panoptic_quality/{metric}/class_{i}'
						reduced_metrics[key] = results[f'{metric}_per_class'][i-1]  
						

		return reduced_metrics 	
		
	def _reduce_panoptic_metrics_v2(self, results: Dict[str, Any]):
		"""
		Updates the per class and mean panoptic metrics in the reduced_metrics.
		"""
		reduced_metrics = {} 

		valid_thing_classes = results['valid_thing_classes']
		valid_stuff_classes = results['valid_stuff_classes']
		valid_classes = valid_stuff_classes | valid_thing_classes
		num_categories = tf.math.count_nonzero(valid_classes, dtype=tf.float32)
		num_thing_categories = tf.math.count_nonzero(
						valid_thing_classes, dtype=tf.float32
		)
		num_stuff_categories = tf.math.count_nonzero(
						valid_stuff_classes, dtype=tf.float32
		)
		valid_thing_classes = tf.cast(valid_thing_classes, dtype=tf.float32)
		valid_stuff_classes = tf.cast(valid_stuff_classes, dtype=tf.float32)

		reduced_metrics['panoptic_quality/All_num_categories'] = num_categories
		reduced_metrics['panoptic_quality/Things_num_categories'] = num_thing_categories
		reduced_metrics['panoptic_quality/Stuff_num_categories'] = num_stuff_categories

		for metric in ['pq', 'sq', 'rq']:
			metric_per_class = results[f'{metric}_per_class']
			reduced_metrics[f'panoptic_quality/All_{metric}'] = tf.math.divide_no_nan(
							tf.reduce_sum(metric_per_class), num_categories
			)
			reduced_metrics[f'panoptic_quality/Things_{metric}'] = tf.math.divide_no_nan(
							tf.reduce_sum(metric_per_class * valid_thing_classes),
							num_thing_categories,
			)
			reduced_metrics[f'panoptic_quality/Stuff_{metric}'] = tf.math.divide_no_nan(
							tf.reduce_sum(metric_per_class * valid_stuff_classes),
							num_stuff_categories,
			)
			if self.task_config.panoptic_quality_evaluator.report_per_class_metrics:
							for i, is_valid in enumerate(valid_classes.numpy()):
											if is_valid:
															reduced_metrics[f'panoptic_quality/{metric}/class_{i}'] = metric_per_class[i]
		
		return reduced_metrics 