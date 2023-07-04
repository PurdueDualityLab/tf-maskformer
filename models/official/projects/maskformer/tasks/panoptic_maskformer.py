import tensorflow as tf

from official.core import base_task
from official.core import task_factory
from typing import Any, Dict, List, Mapping, Optional, Tuple

from official.projects.maskformer.dataloaders import input_reader
from official.vision.dataloaders import input_reader_factory
from official.common import dataset_fn

from official.projects.maskformer.configs import maskformer as exp_cfg
from official.projects.maskformer.modeling.maskformer import MaskFormer
from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.maskformer.dataloaders import panoptic_input

from official.projects.detr.ops.matchers import hungarian_matching
from official.projects.maskformer.losses.maskformer_losses import Loss

import numpy as np
from loguru import logger

@task_factory.register_task_cls(exp_cfg.MaskFormerTask)
class PanopticTask(base_task.Task):
	def build_model(self)-> tf.keras.Model:
		"""Builds MaskFormer Model."""
		# TODO : Remove hardcoded values, Verify the number of classes 
		input_specs = tf.keras.layers.InputSpec(shape=[None] +
									self._task_config.model.input_size)
		
		model = MaskFormer(input_specs= input_specs,
					num_queries=self._task_config.model.num_queries,
					hidden_size=self._task_config.model.hidden_size,
					backbone_endpoint_name=self._task_config.model.backbone_endpoint_name,
					fpn_encoder_layers=self._task_config.model.fpn_encoder_layers,
					detr_encoder_layers=self._task_config.model.detr_encoder_layers,
					num_decoder_layers=self._task_config.model.num_decoder_layers,
					num_classes=self._task_config.model.num_classes,
					)
		
		#ckpt_dir_or_file = self._task_config.init_checkpoint
		#ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
		#ckpt = tf.train.Checkpoint(backbone=model.backbone)
		#status = ckpt.restore(ckpt_dir_or_file)
		#status.expect_partial().assert_existing_objects_matched()
		#print("Loaded checkpoint")
		return model

	def initialize(self, model: tf.keras.Model) -> None:
		"""
		Used to initialize the models with checkpoint
		"""
		"""Loading pretrained checkpoint."""
		if not self._task_config.init_checkpoint:
			return
		ckpt_dir_or_file = self._task_config.init_checkpoint

		# Restoring checkpoint.
		if tf.io.gfile.isdir(ckpt_dir_or_file):
			ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

		if self._task_config.init_checkpoint_modules == 'all':
			ckpt = tf.train.Checkpoint(**model.checkpoint_items)
			status = ckpt.restore(ckpt_dir_or_file)
			status.assert_consumed()
		elif self._task_config.init_checkpoint_modules == 'backbone':
			print("Loading resnet")
			ckpt = tf.train.Checkpoint(backbone=model.backbone)
			status = ckpt.restore(ckpt_dir_or_file)
		print(f"==========FINISHED LOADING {self._task_config.init_checkpoint}===========")

	def build_inputs(self, params, input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
			""" 
			Build panoptic segmentation dataset.

			"""
			
			# tf.profiler.experimental.server.start(6000)
			if params.decoder.type == 'simple_decoder':
				decoder = panoptic_input.TfExampleDecoder(regenerate_source_id = params.regenerate_source_id)
			else:
				raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))
			
			parser = panoptic_input.mask_former_parser(params.parser, is_training = params.is_training, decoder_fn=decoder.decode)
			reader = input_reader.InputFn(params,dataset_fn = dataset_fn.pick_dataset_fn(params.file_type),parser_fn = parser)
			dataset = reader(ctx=input_context)
			tf.profiler.experimental.server.start(6000)

			return dataset


	def build_losses(self, output, labels, aux_outputs=None):
			# TODO : Auxilary outputs
			outputs = {"pred_logits": output["class_prob_predictions"], "pred_masks": output["mask_prob_predictions"]}
			targets = labels

			matcher = hungarian_matching
			no_object_weight = self._task_config.losses.no_object_weight
			# TODO : Remove hardcoded values, number of classes
			loss = Loss(
					num_classes = self._task_config.model.num_classes,
					matcher = matcher,
					eos_coef = no_object_weight,
					cost_class= self._task_config.losses.cost_class,
					cost_dice= self._task_config.losses.cost_dice,
					cost_focal= self._task_config.losses.cost_focal
			)

			calculated_losses = loss(outputs, targets)
			
			# Losses are returned as weighted sum of individual losses
			total_loss = calculated_losses['loss_ce'] + calculated_losses['loss_dice'] + calculated_losses['loss_focal']

			weighted_ce = calculated_losses['loss_ce']
			weighted_focal = calculated_losses['loss_dice']
			weighted_dice = calculated_losses['loss_focal']

			# Not implemented auxilary outputs
			# if aux_outputs is not None:
			#       total_aux_loss = 0.0
			#       # TODO : Remove hardcoding
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
	
			# Preprocess labels to match the format for loss prediction
			# mask_labels = 'unique_ids': unique_ids,
	#     'individual_masks': individual_masks,
			with tf.GradientTape() as tape:
					outputs = model(features, training=True)
					loss = 0.0
					cls_loss = 0.0
					focal_loss = 0.0
					dice_loss = 0.0

					##########################################################
					# TODO : Need to use this for TPU training when we use mirrored startegy

					# print(outputs.shape)
					# exit()
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

					####################################################################
					# Do not use mixed precision for now
					# # Scales back gradient when LossScaleOptimizer is used.
					# if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
					#       grads = optimizer.get_unscaled_gradients(grads)
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
					#####################################################################
					# # Trainer class handles loss metric for you.
					logs = {self.loss: total_loss}

					all_losses = {
							'cls_loss': cls_loss,
							'focal_loss': focal_loss,
						'dice_loss': dice_loss,
					}

					
					# # Metric results will be added to logs for you.
					if metrics:
							for m in metrics:
								m.update_state(all_losses[m.name])
					return logs

	def validation_step(self, inputs, model, optimizer, metrics=None):
			features, labels = inputs
			outputs = model(features, training=False)
					
			loss = 0.0
			cls_loss = 0.0
			focal_loss = 0.0
			dice_loss = 0.0

			total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels)

			num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
			total_loss *= num_replicas_in_sync
			cls_loss *= num_replicas_in_sync
			focal_loss *= num_replicas_in_sync
			dice_loss *= num_replicas_in_sync
			
	#####################################################################
			# # Trainer class handles loss metric for you.
			logs = {self.loss: total_loss}
	
			outputs = {"pred_logits": output["class_prob_predictions"], "pred_masks": output["mask_prob_predictions"]}
			#panoptic_seg, segments_info = PanopticInference(output["pred_logits"], output["pred_masks"], features.shape,  self._task_config.model.num_classes)
	
			#logs.update({'panoptic_seg': panoptic_seg, 'segments_info': segments_info})

			all_losses = {
							'cls_loss': cls_loss,
							'focal_loss': focal_loss,
						'dice_loss': dice_loss,
					}

			# # Metric results will be added to logs for you.
			if metrics:
							for m in metrics:
									m.update_state(all_losses[m.name])
			return logs
