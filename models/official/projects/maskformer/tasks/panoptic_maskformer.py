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
		ckpt_dir_or_file = self._task_config.init_checkpoint

		# Restoring checkpoint.
		if tf.io.gfile.isdir(ckpt_dir_or_file):
			ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

		if self._task_config.init_checkpoint_modules == 'all':
			ckpt = tf.train.Checkpoint(**model.checkpoint_items)
			status = ckpt.restore(ckpt_dir_or_file)
			status.assert_consumed()
		elif self._task_config.init_checkpoint_modules == 'backbone':
			ckpt = tf.train.Checkpoint(backbone=model.backbone)
			status = ckpt.restore(ckpt_dir_or_file)

	def build_inputs(self, params, input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
		""" 
		Build panoptic segmentation dataset.

		"""
		# NOTE : This is not required as we are setting two different flags for data and model
		# if self._task_config.bfloat16:
		# 	params.parser.dtype = "bfloat16"
		# 	params.dtype = "bfloat16"
		
		if params.decoder.type == 'simple_decoder':
			decoder = panoptic_input.TfExampleDecoder(regenerate_source_id = params.regenerate_source_id)
		else:
			raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))
		
		parser = panoptic_input.mask_former_parser(params.parser, is_training = params.is_training, decoder_fn=decoder.decode)
		

		#reader = input_reader.InputFn(params, dataset_fn = dataset_fn.pick_dataset_fn(params.file_type),parser_fn = parser)
		#dataset = reader(ctx=input_context)
		
		# FIXME : Use default Input reader instead of custom input reader (uncomment above lines to use old reader)
		reader = input_reader_factory.input_reader_generator(
          params,
          dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
          decoder_fn=decoder.decode,
          parser_fn=parser.parse_fn(params.is_training))
		
		dataset = reader.read(input_context=input_context)
		for sample in dataset.take(1):
			print(f"unique ids : {sample[1]['unique_ids']}")
			print("individual masks :", sample[1]["individual_masks"].shape)
			print(f"image shape : {sample[0].shape}")
			exit()
		return dataset


	def build_losses(self, output, labels, aux_outputs=None):
		# TODO : Auxilary outputs
		# NOTE : Loss calculation using Bfloat16 hampers the convergence of the model
		outputs = {"pred_logits": tf.cast(output["class_prob_predictions"],tf.float32), "pred_masks": tf.cast(output["mask_prob_predictions"],tf.float32)}
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

			all_losses = {
				'cls_loss': cls_loss,
				'focal_loss': focal_loss,
				'dice_loss': dice_loss,}

					
			# Metric results will be added to logs for you.
			if metrics:
				for m in metrics:
					m.update_state(all_losses[m.name])

			return logs

	def validation_step(self, inputs, model, optimizer, metrics=None):
		pass
