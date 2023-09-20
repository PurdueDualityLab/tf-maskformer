import os
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
from official.vision.modeling import backbones
import numpy as np
import pysnooper
import os

@task_factory.register_task_cls(maskformer_cfg.MaskFormerTask)
class PanopticTask(base_task.Task):
	"""A single-replica view of training procedure.

	PanopticTask task provides artifacts for training/evalution procedures, including
	loading/iterating over Datasets, initializing the model, calculating the loss,
	post-processing, and customized metrics with reduction.
	"""
	def build_model(self)-> tf.keras.Model:
		"""Builds MaskFormer Model."""
		logging.info('Building MaskFormer model.')
		
		input_specs = tf.keras.layers.InputSpec(shape=[None] + self._task_config.model.input_size)
		
		backbone = backbones.factory.build_backbone(input_specs=input_specs,
					backbone_config=self._task_config.model.backbone,
					norm_activation_config=self._task_config.model.norm_activation)
		logging.info('Backbone build successful.')
		model = MaskFormer(backbone=backbone, input_specs= input_specs,
							num_queries=self._task_config.model.num_queries,
							hidden_size=self._task_config.model.hidden_size,
							backbone_endpoint_name=self._task_config.model.backbone_endpoint_name,
							fpn_encoder_layers=self._task_config.model.fpn_encoder_layers,
							detr_encoder_layers=self._task_config.model.detr_encoder_layers,
							num_decoder_layers=self._task_config.model.num_decoder_layers,
							num_classes=self._task_config.model.num_classes,
							bfloat16=self._task_config.bfloat16, 
							which_pixel_decoder=self._task_config.model.which_pixel_decoder,)
		logging.info('Maskformer model build successful.')
		return model

	def initialize(self, model: tf.keras.Model) -> None:
		"""
		Used to initialize the models with checkpoint
		"""

		# Akshath
		self.num_images = 0
		self.counts = {} 	
		with open('/depot/davisjam/data/akshath/exps/tf/instance_counting/counts.txt', 'w') as f: 
			pass
		with open('/depot/davisjam/data/akshath/exps/tf/instance_counting/num_images.txt', 'w') as f: 
			pass
		with open('/depot/davisjam/data/akshath/exps/tf/instance_counting/check_cat_cont.txt', 'w') as f: 
			pass
		with open('/depot/davisjam/data/akshath/exps/tf/instance_counting/mismatched_background_masks.txt', 'w') as f: 
			pass
			
		# pass
		logging.info('Initializing model from checkpoint: %s', self._task_config.init_checkpoint)
		if not self._task_config.init_checkpoint:
			return
		ckpt_dir_or_file = self._task_config.init_checkpoint

		# Restoring ckpt
		if tf.io.gfile.isdir(ckpt_dir_or_file):
			ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

		if self._task_config.init_checkpoint_modules == 'all':
			ckpt = tf.train.Checkpoint(model)
			status = ckpt.restore(ckpt_dir_or_file)
			status.assert_consumed()
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
		# TODO : Auxilary outputs
		outputs = {"pred_logits": output["class_prob_predictions"], "pred_masks": output["mask_prob_predictions"]}
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
		
		# Losses are returned as weighted sum of individual losses
		total_loss = calculated_losses['loss_ce'] + calculated_losses['loss_dice'] + calculated_losses['loss_focal']

		weighted_ce = calculated_losses['loss_ce']
		weighted_dice = calculated_losses['loss_dice']
		weighted_focal = calculated_losses['loss_focal']

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
		
		
	# @pysnooper.snoop('/depot/davisjam/data/akshath/exps/tf/traces/sample.log')
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
		
		# Akshath
		mapping_dict  = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80, 92: 81, 93: 82, 95: 83, 100: 84, 107: 85, 109: 86, 112: 87, 118: 88, 119: 89, 122: 90, 125: 91, 128: 92, 130: 93, 133: 94, 138: 95, 141: 96, 144: 97, 145: 98, 147: 99, 148: 100, 149: 101, 151: 102, 154: 103, 155: 104, 156: 105, 159: 106, 161: 107, 166: 108, 168: 109, 171: 110, 175: 111, 176: 112, 177: 113, 178: 114, 180: 115, 181: 116, 184: 117, 185: 118, 186: 119, 187: 120, 188: 121, 189: 122, 190: 123, 191: 124, 192: 125, 193: 126, 194: 127, 195: 128, 196: 129, 197: 130, 198: 131, 199: 132, 200: 133}
		count_backgrounds_in_labels = 0
		# Images 
		self.num_images += 1
		with open('/depot/davisjam/data/akshath/exps/tf/instance_counting/num_images.txt', 'w') as f: 
				f.write(str(self.num_images)) 
		# IDs 
		labels_list = labels["unique_ids"]._numpy().tolist()
		for la in labels_list[0]:
			if la == 0: 
					count_backgrounds_in_labels += 1
			if la in self.counts: 
				self.counts[la] += 1
			else: 
				self.counts[la] = 1
			
		with open('/depot/davisjam/data/akshath/exps/tf/instance_counting/counts.txt', 'w') as f: 
			f.write(str(self.counts))

	  # Check masks
		cat = labels["category_mask"]._numpy()[0]
		cont = labels["contigious_mask"]._numpy()[0, :, :, 0]
		mapped_cat = np.array([[mapping_dict.get(int(x), int(x)) for x in row] for row in cat])
		are_equal = np.array_equal(mapped_cat, cont)
		if not are_equal: 
			np.save(f'/depot/davisjam/data/akshath/exps/tf/testing/mask_compare/cat{self.num_images}.npy', cat) 
			np.save(f'/depot/davisjam/data/akshath/exps/tf/testing/mask_compare/mapped_cat{self.num_images}.npy', mapped_cat) 
			np.save(f'/depot/davisjam/data/akshath/exps/tf/testing/mask_compare/cont{self.num_images}.npy', cont) 
		with open('/depot/davisjam/data/akshath/exps/tf/instance_counting/check_cat_cont.txt', 'a') as f: 
			f.write(str(are_equal) + '\n') 
		# Checking number of background masks
		count_total_background_masks = 0
		induvidual_masks = labels["individual_masks"]._numpy()
		subset1k = induvidual_masks[0, :, :, :]
		result = np.any(subset1k != 0, axis=(1, 2))
		np.save(f'/depot/davisjam/data/akshath/exps/tf/testing/mask_compare/subset1k.npy', subset1k) 

		zero_mask = np.zeros((640, 640), dtype=subset1k.dtype)
		for i, has_non_zero in enumerate(result):
				if not has_non_zero:
					count_total_background_masks += 1
				else: 
					if i > (len(labels_list[0]) - 1): 
						subset1k[i] = zero_mask
		del zero_mask
		labels["individual_masks"] = np.expand_dims(subset1k, axis=0)

		num_extra = count_total_background_masks - count_backgrounds_in_labels
		with open('/depot/davisjam/data/akshath/exps/tf/instance_counting/mismatched_background_masks.txt', 'a') as f: 
				f.write(str(num_extra) + '\n') 		
		
		print('--------------------------')
		print(self.num_images)
		print('Are Equal:', are_equal)	
		print('Extra:', num_extra)
		print('--------------------------')

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
			scaled_loss = total_loss
			if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
				total_loss = optimizer.get_scaled_loss(scaled_loss)
		grads = tape.gradient(scaled_loss, model.trainable_variables)

		if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
			grads = optimizer.get_unscaled_gradients(grads)
		optimizer.apply_gradients(list(zip(grads,  model.trainable_variables)))
		
		if os.environ.get('PRINT_OUTPUTS') == 'True':
			probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
			pred_labels = tf.argmax(probs, axis=-1)
			# print("Target labels :", labels["unique_ids"])
			# print("Output labels :", pred_labels)
		
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

		return logs
		
	def _postprocess_outputs(self, outputs: Dict[str, Any], image_shapes):
		""" 
		Implements postprocessing using the output binary masks and labels to produce
		1. Output Category Mask
		2. Output Instance Mask

		"""
		pred_binary_masks = outputs["mask_prob_predictions"]
		pred_labels = outputs["class_prob_predictions"]
		ouput_instance_mask, output_category_mask = self.panoptic_inference(pred_labels,pred_binary_masks, image_shapes)
		return ouput_instance_mask, output_category_mask

	def validation_step(self, inputs, model, metrics=None):
		features, labels = inputs
		outputs = model(features, training=False)
		if os.environ.get('PRINT_OUTPUTS') == 'True':
			probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
			pred_labels = tf.argmax(probs, axis=-1)
			print("Probs :", probs)
			print("Target labels :", labels["unique_ids"])
			print("Output labels :", pred_labels)
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
		
		# if self.panoptic_quality_metric is not None:
		# 	pq_metric_labels = {
		# 	'category_mask': labels['category_mask'], # ignore label is 0 
		# 	'instance_mask': labels['instance_mask'],
		# 	'image_info': labels['image_info'],
		# 	}
		# 	# Output from postprocessing will convert the binary masks to category and instance masks with non-contigious ids
		# 	output_category_mask, output_instance_mask = self._postprocess_outputs(outputs, [1280, 1280])
		# 	pq_metric_outputs = {
		# 	'category_mask': output_category_mask,
		# 	'instance_mask': output_instance_mask,
		# 	}
			
		# 	self.panoptic_quality_metric.update_state(
		#   	pq_metric_labels, pq_metric_outputs
	  	# 	)
		if metrics:
			for m in metrics:
				m.update_state(all_losses[m.name])

		return logs
	

	# def aggregate_logs(self, state=None, step_outputs=None):
	# 	is_first_step = not state
	# 	if state is None:
	# 		state = self.panoptic_quality_metric
	# 	state.update_state(
	# 		step_outputs['ground_truths'],
	# 		step_outputs['predictions'])
		
	# 	return state
	
	# def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
	# 	if self.panoptic_quality_metric is not None:
	# 		self._reduce_panoptic_metrics(aggregated_logs)
	# 		self.panoptic_quality_metric.reset_state()	
	# 	return aggregated_logs
	
	# def _reduce_panoptic_metrics(self, logs: Dict[str, Any]):
	# 	"""
	# 	Updates the per class and mean panoptic metrics in the logs.
		
	# 	"""
	# 	result = self.panoptic_quality_metric.result()
	# 	valid_thing_classes = result['valid_thing_classes']
	# 	valid_stuff_classes = result['valid_stuff_classes']
	# 	valid_classes = valid_stuff_classes | valid_thing_classes
	# 	num_categories = tf.math.count_nonzero(valid_classes, dtype=tf.float32)
	# 	num_thing_categories = tf.math.count_nonzero(
	# 		valid_thing_classes, dtype=tf.float32
	# 	)
	# 	num_stuff_categories = tf.math.count_nonzero(
	# 		valid_stuff_classes, dtype=tf.float32
	# 	)
	# 	valid_thing_classes = tf.cast(valid_thing_classes, dtype=tf.float32)
	# 	valid_stuff_classes = tf.cast(valid_stuff_classes, dtype=tf.float32)

	# 	logs['panoptic_quality/All_num_categories'] = num_categories
	# 	logs['panoptic_quality/Things_num_categories'] = num_thing_categories
	# 	logs['panoptic_quality/Stuff_num_categories'] = num_stuff_categories
	# 	for metric in ['pq', 'sq', 'rq']:
	# 		metric_per_class = result[f'{metric}_per_class']
	# 		logs[f'panoptic_quality/All_{metric}'] = tf.math.divide_no_nan(
	# 			tf.reduce_sum(metric_per_class), num_categories
	# 		)
	# 		logs[f'panoptic_quality/Things_{metric}'] = tf.math.divide_no_nan(
	# 			tf.reduce_sum(metric_per_class * valid_thing_classes),
	# 			num_thing_categories,
	# 		)
	# 		logs[f'panoptic_quality/Stuff_{metric}'] = tf.math.divide_no_nan(
	# 			tf.reduce_sum(metric_per_class * valid_stuff_classes),
	# 			num_stuff_categories,
	# 		)
	# 		if self.task_config.panoptic_quality_evaluator.report_per_class_metrics:
	# 			for i, is_valid in enumerate(valid_classes.numpy()):
	# 				if is_valid:
	# 					logs[f'panoptic_quality/{metric}/class_{i}'] = metric_per_class[i]
		
	