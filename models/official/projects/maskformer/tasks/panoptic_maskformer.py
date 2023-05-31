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
                                            [640, 640, 3])
		
		model = MaskFormer(input_specs= input_specs, hidden_size=256,
                                 backbone_endpoint_name="5",
                                 num_encoder_layers=0,
                                 num_decoder_layers=6,
                                 num_classes=199,
                                 batch_size=1)

		return model
	
	def build_inputs(self, params, input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
		""" 
		Build panoptic segmentation dataset.

		"""
		if params.decoder.type == 'simple_decoder':
			decoder = panoptic_input.TfExampleDecoder(regenerate_source_id = params.regenerate_source_id)
		else:
			raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))
		
		parser = panoptic_input.mask_former_parser(params.parser, is_training = params.is_training, decoder_fn=decoder.decode)
		reader = input_reader.InputFn(params,dataset_fn = dataset_fn.pick_dataset_fn(params.file_type),parser_fn = parser)
		dataset = reader(ctx=input_context)
		# for sample in dataset.take(1):
		# 	print(f"unique ids : {sample[1]['unique_ids']}")
		# 	print(f"unique ids shape : {sample[1]['unique_ids'].shape}")
		# 	print("individual masks :", sample[1]["individual_masks"].shape)
		# 	print(f"image shape : {sample[0].shape}")
		# 	print("individual masks classes :", sample[1]["unique_ids"])
		# 	print("individual masks :", sample[1]["individual_masks"].shape)
		# 	# logger.debug(f"category_mask : {sample[1]['category_mask'].shape}")
		# 	# logger.debug(f"mask_labels :{sample[1]['mask_labels']}")
		# 	# logger.debug(f"instance_mask:{sample[1]['instance_mask'].shape}")
		# 	# print(sample[1]["instance_centers_heatmap"].shape)
		# 	# print(sample[1]["instance_centers_offset"].shape)
		# 	# print(sample[1]["semantic_weights"].shape)
		# 	# print(sample[1]["valid_mask"].shape)
		# 	# print(sample[1]["things_mask"].shape)
			
		# exit()
		return dataset

	def initialize(self, model: tf.keras.Model) -> None:
		"""
		Used to initialize the models with checkpoint
		"""
		#TODO : R50 checkpoint
		pass

	def build_losses(self, output, labels, aux_outputs=None):
		# TODO : Auxilary outputs
		outputs = {"pred_logits": output["class_prob_predictions"], "pred_masks": output["mask_prob_predictions"]}
		targets = labels
		
		print("pred_logits : ", outputs["pred_logits"].shape)
		print("pred_masks : ", outputs["pred_masks"].shape)
		exit()
		matcher = hungarian_matching
		no_object_weight = 0.1
		# TODO : Remove hardcoded values, number of classes
		loss = Loss(
			num_classes = 199,
			matcher = matcher,
			eos_coef = no_object_weight,
			cost_class= 1.0,
			cost_dice= 1.0,
			cost_focal=20.0
		)

		calculated_losses = loss(outputs, targets)
		
		# Losses are returned as weighted sum of individual losses
		total_loss = calculated_losses['loss_ce'] + calculated_losses['loss_dice'] + calculated_losses['loss_focal']

		weighted_ce = calculated_losses['loss_ce']
		weighted_focal = calculated_losses['loss_dice']
		weighted_dice = calculated_losses['loss_focal']

		# Not implemented auxilary outputs
		if aux_outputs is not None:
			total_aux_loss = 0.0
			# TODO : Remove hardcoding
			for i in range(4): #4 number of auxilary outputs
				total_aux_loss += calculated_losses['loss_ce_'+str(i)] + calculated_losses['loss_dice_'+str(i)] + calculated_losses['loss_focal_'+str(i)]
			total_loss = total_loss + total_aux_loss
		

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
			# 	# Computes per-replica loss.
				
			# 	total_loss, cls_loss_, focal_loss_, dice_loss_ = self.build_losses(
			# 		output=output, labels=labels)
			# 	loss += total_loss
			# 	cls_loss += cls_loss_
			# 	focal_loss += focal_loss_
			# 	dice_loss += dice_loss_
			
			# 	scaled_loss = loss
			# 	# For mixed_precision policy, when LossScaleOptimizer is used, loss is
			# 	# scaled for numerical stability.
			# 	if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
			# 		scaled_loss = optimizer.get_scaled_loss(scaled_loss)
		##########################################################################
			
			# TODO : Add auxiallary losses
			total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels)
			tvars = model.trainable_variables
			
			grads = tape.gradient(total_loss, tvars)

			####################################################################
			# Do not use mixed precision for now
			# # Scales back gradient when LossScaleOptimizer is used.
			# if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
			# 	grads = optimizer.get_unscaled_gradients(grads)
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
		pass
