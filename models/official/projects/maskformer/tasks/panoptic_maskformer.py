from absl import logging
import tensorflow as tf

from official.core import base_task
from official.core import task_factory
from typing import Any, Dict, List, Mapping, Optional, Tuple

from official.projects.maskformer.dataloaders import input_reader
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

import numpy as np


@task_factory.register_task_cls(maskformer_cfg.MaskFormerTask)
class PanopticTask(base_task.Task):
	"""A single-replica view of training procedure.

	PanopticTask task provides artifacts for training/evalution procedures, including
	loading/iterating over Datasets, initializing the model, calculating the loss,
	post-processing, and customized metrics with reduction.
	"""
	def build_model(self)-> tf.keras.Model:
		"""Builds MaskFormer Model."""
		# TODO : Remove hardcoded values, Verify the number of classes 
		input_specs = tf.keras.layers.InputSpec(shape=[None] + self._task_config.model.input_size)
		
		model = MaskFormer(input_specs= input_specs,
							num_queries=self._task_config.model.num_queries,
							hidden_size=self._task_config.model.hidden_size,
							backbone_endpoint_name=self._task_config.model.backbone_endpoint_name,
							fpn_encoder_layers=self._task_config.model.fpn_encoder_layers,
							detr_encoder_layers=self._task_config.model.detr_encoder_layers,
							num_decoder_layers=self._task_config.model.num_decoder_layers,
							num_classes=self._task_config.model.num_classes,
							bfloat16=self._task_config.bfloat16, 
							which_pixel_decoder=self._task_config.model.which_pixel_decoder,)
		return model

	def initialize(self, model: tf.keras.Model) -> None:
		"""
		Used to initialize the models with checkpoint
		"""
		
		if not self._task_config.init_checkpoint:
			return
		
		def _get_checkpoint_path(checkpoint_dir_or_file):
			checkpoint_path = checkpoint_dir_or_file

			if tf.io.gfile.isdir(checkpoint_dir_or_file):
				checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir_or_file)
			return checkpoint_path
		
		ckpt_dir_or_file = self._task_config.init_checkpoint
		
		if self._task_config.init_checkpoint_modules == 'all':
			checkpoint_path = _get_checkpoint_path(ckpt_dir_or_file)
			ckpt = tf.train.Checkpoint(model)
			status = ckpt.read(checkpoint_path)
			status.expect_partial()
			logging.info('Loaded whole model from %s',
				 ckpt_dir_or_file)
			
			
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
		# for sample in dataset.take(1):
		# 	# print(f"unique idsin dataset take : {sample[1]['unique_ids']}")
		# 	print("individual masks :", sample[1]["individual_masks"].shape)
		# 	# np.save("contigious_mask.npy", sample[1]["contigious_mask"].numpy())
		# 	print(f"image shape : {sample[0].shape}")
		# 	# np.save("individual_masks.npy", sample[1]["individual_masks"].numpy())
		# 	# np.save("unique_ids.npy", sample[1]["unique_ids"].numpy())
		# 	# np.save("image.npy", sample[0].numpy())
		# 	# exit()
		return dataset


	def build_losses(self, output, labels, aux_outputs=None):
		# TODO : Auxilary outputs
		# NOTE : Loss calculation using Bfloat16 hampers the convergence of the model
		outputs = {"pred_logits": output["class_prob_predictions"], "pred_masks": output["mask_prob_predictions"]}
		targets = labels

		matcher = hungarian_matching
		no_object_weight = self._task_config.losses.background_cls_weight
		# TODO : Remove hardcoded values, number of classes
		loss = Loss(num_classes = self._task_config.model.num_classes,
					matcher = matcher,
					eos_coef = no_object_weight,
					cost_class= self._task_config.losses.cost_class,
					cost_dice= self._task_config.losses.cost_dice,
					cost_focal= self._task_config.losses.cost_focal,)

		calculated_losses = loss(outputs, targets)
		
		# Losses are returned as weighted sum of individual losses
		total_loss = calculated_losses['loss_ce'] + calculated_losses['loss_dice'] + calculated_losses['loss_focal']

		weighted_ce = calculated_losses['loss_ce']
		weighted_focal = calculated_losses['loss_dice']
		weighted_dice = calculated_losses['loss_focal']

		# Not implemented auxilary outputs
		# if aux_outputs is not None:
		#       total_aux_loss = 0.0
		#       for i in range(4): #4 number of auxilary outputs
		#               total_aux_loss += calculated_losses['loss_ce_'+str(i)] + calculated_losses['loss_dice_'+str(i)] + calculated_losses['loss_focal_'+str(i)]
		#       total_loss = total_loss + total_aux_loss
		

		return total_loss, weighted_ce, weighted_focal, weighted_dice
		
	def build_metrics(self, training=True):
		"""Builds panoptic metrics."""
		metrics = []
		metric_names = ['cls_loss', 'focal_loss', 'dice_loss']
		for name in metric_names:
			metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
		# TODO : Need panoptic quality metric for evaluation
		if not training:
			print("[INFO] Building panoptic quality metric ")
			pq_config = self._task_config.panoptic_quality_evaluator
			self.panoptic_quality_metric = panoptic_quality.PanopticQualityV2(
				num_categories=pq_config.num_categories,
				is_thing=pq_config.is_thing,
				ignored_label=pq_config.ignored_label,
				rescale_predictions=pq_config.rescale_predictions,
			)
			self.panoptic_inference = PanopticInference(
				num_classes=self._task_config.model.num_classes, 
				background_class_id=pq_config.ignored_label
			)
		return metrics
		
		
		
	def train_step(self, inputs: Tuple[Any, Any],model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, metrics: Optional[List[Any]] = None) -> Dict[str, Any]:
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
			outputs = model(features, training=True)
			##########################################################
			# FIXME : This loop must be used for auxilary outputs
			loss = 0.0
			cls_loss = 0.0
			focal_loss = 0.0
			dice_loss = 0.0
			
			# for output in outputs:
			#       # Computes per-replica loss.	
			#       total_loss, cls_loss_, focal_loss_, dice_loss_ = self.build_losses(
			#               output=output, labels=labels)
			#       loss += total_loss
			#       cls_loss += cls_loss_
			#       focal_loss += focal_loss_
			#       dice_loss += dice_loss_
			
			#       scaled_loss = loss
			#       # For mixed_precision policy, when LossScaleOptimizer is used, loss is
			#       # scaled for numerical stability.
			
			##########################################################################
			
			# TODO : Add auxiallary losses
			total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels)
			
			if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
				total_loss = optimizer.get_scaled_loss(total_loss)
					
		tvars = model.trainable_variables
		
		grads = tape.gradient(total_loss, tvars)

		optimizer.apply_gradients(list(zip(grads, tvars)))
		
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

		# pred_logits = outputs["class_prob_predictions"]
		# probs = tf.keras.activations.softmax(pred_logits, axis=-1) # (batch, num_predictions, num_classes) (2,100,134)
		# scores = tf.reduce_max(probs, axis=-1) 
		# print("gt labels :", labels["unique_ids"])
		# predicted_labels = tf.argmax(probs, axis=-1)
		# print("predicted labels :", predicted_labels)
	
		all_losses = {
			'cls_loss': cls_loss,
			'focal_loss': focal_loss,
			'dice_loss': dice_loss,}

		
		# Metric results will be added to logs for you.
		if metrics:
			for m in metrics:
				m.update_state(all_losses[m.name])

		return logs
		
	def _postprocess_outputs(self, outputs: Dict[str, Any], image_shapes):
		""" 
		Implements postprocessing using the output binary masks and labels to produce
		1. Output Category Mask
		2. Output Instance Mask

		"""
		pred_binary_masks = outputs["mask_prob_predictions"]
		pred_labels = outputs["class_prob_predictions"]
		ouput_instance_mask, output_category_mask = self.panoptic_inference( pred_labels,pred_binary_masks, image_shapes)
		return ouput_instance_mask, output_category_mask

	def validation_step(self, inputs, model, metrics=None):
		features, labels = inputs
		outputs = model(features, training=False)
		# np.save("individual_masks.npy", labels["individual_masks"])
		print("Unique_ids in validation step : ", labels["unique_ids"])
		# np.save("targets.npy", labels["unique_ids"])
		# np.save("image.npy", features)
		total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels)
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
		
		if self.panoptic_quality_metric is not None:
			
			pq_metric_labels = {
			'category_mask': labels['category_mask'],
			'instance_mask': labels['instance_mask'],
			# 'image_info': labels['image_info'],
			}
			# FIXME : The image shape must not be fixed
			output_category_mask, output_instance_mask = self._postprocess_outputs(outputs, [640, 640])
			pq_metric_outputs = {
			'category_mask': output_category_mask,
			'instance_mask': output_instance_mask,
			}
			
			self.panoptic_quality_metric.update_state(
		  	pq_metric_labels, pq_metric_outputs
	  		)

	def aggregate_logs(self, state=None, step_outputs=None):
		is_first_step = not state
		if state is None:
			state = self.panoptic_quality_metric
		state.update_state(
			step_outputs['ground_truths'],
			step_outputs['predictions'])
		
		return state
	
	def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
		if self.panoptic_quality_metric is not None:
			self._reduce_panoptic_metrics(aggregated_logs)
			self.panoptic_quality_metric.reset_state()	
		return aggregated_logs
	
	def _reduce_panoptic_metrics(self, logs: Dict[str, Any]):
		"""
		Updates the per class and mean panoptic metrics in the logs.
		
		"""
		result = self.panoptic_quality_metric.result()
		valid_thing_classes = result['valid_thing_classes']
		valid_stuff_classes = result['valid_stuff_classes']
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

		logs['panoptic_quality/All_num_categories'] = num_categories
		logs['panoptic_quality/Things_num_categories'] = num_thing_categories
		logs['panoptic_quality/Stuff_num_categories'] = num_stuff_categories
		for metric in ['pq', 'sq', 'rq']:
			metric_per_class = result[f'{metric}_per_class']
			logs[f'panoptic_quality/All_{metric}'] = tf.math.divide_no_nan(
				tf.reduce_sum(metric_per_class), num_categories
			)
			logs[f'panoptic_quality/Things_{metric}'] = tf.math.divide_no_nan(
				tf.reduce_sum(metric_per_class * valid_thing_classes),
				num_thing_categories,
			)
			logs[f'panoptic_quality/Stuff_{metric}'] = tf.math.divide_no_nan(
				tf.reduce_sum(metric_per_class * valid_stuff_classes),
				num_stuff_categories,
			)
			if self.task_config.panoptic_quality_evaluator.report_per_class_metrics:
				for i, is_valid in enumerate(valid_classes.numpy()):
					if is_valid:
						logs[f'panoptic_quality/{metric}/class_{i}'] = metric_per_class[i]
		
	