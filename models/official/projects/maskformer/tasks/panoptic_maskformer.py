import tensorflow as tf

from official.core import base_task
from typing import Any, Dict, List, Mapping, Optional, Tuple
from official.projects.maskformer.modeling.maskformer import MaskFormer
from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.maskformer.dataloaders import panoptic_input
from official.vision.dataloaders import input_reader
from official.vision.dataloaders import input_reader_factory
from official.common import dataset_fn

class PanopticTask(base_task.Task):
	
	def build_model(self)-> tf.keras.Model:
		"""Builds MaskFormer Model."""
		# TODO(ibrahim): Connect to params in config.
		model = MaskFormer()

		return model
	
	def build_inputs(self, params, input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
		""" 
		Build panoptic segmentation dataset.

		"""
		pass

	def build_losses(self, class_prob_outputs, mask_prob_outputs, class_targets, mask_targets):
		outputs = {"pred_logits": class_prob_outputs, "pred_masks": mask_prob_outputs}
		targets = {"labels": class_targets, "masks": mask_targets}
		
		# _compute_loss = Loss(init loss here...)
		# return _compute_loss(outputs, targets)
		raise NotImplementedError
	
	def build_metrics(self, training=True):
		raise NotImplementedError

	def train_step(self, inputs: Tuple[Any, Any],model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, metrics: Optional[List[Any]] = None) -> Dict[str, Any]:
		features, labels = inputs
		with tf.GradientTape() as tape:
			outputs = model(features, training=True)

			#TODO Change to maskformer loss
			loss = 0.0
			cls_loss = 0.0
			box_loss = 0.0
			giou_loss = 0.0

			for output in outputs:
				# Computes per-replica loss.
				layer_loss, layer_cls_loss, layer_box_loss, layer_giou_loss = self.build_losses(
					outputs=output, labels=labels, aux_losses=model.losses)
				loss += layer_loss
				cls_loss += layer_cls_loss
				box_loss += layer_box_loss
				giou_loss += layer_giou_loss
			
			scaled_loss = loss
			# For mixed_precision policy, when LossScaleOptimizer is used, loss is
			# scaled for numerical stability.
			if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
				scaled_loss = optimizer.get_scaled_loss(scaled_loss)
		
		tvars = model.trainable_variables
		grads = tape.gradient(scaled_loss, tvars)
		# Scales back gradient when LossScaleOptimizer is used.
		if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
			grads = optimizer.get_unscaled_gradients(grads)
		optimizer.apply_gradients(list(zip(grads, tvars)))

		# Multiply for logging.
		# Since we expect the gradient replica sum to happen in the optimizer,
		# the loss is scaled with global num_boxes and weights.
		# To have it more interpretable/comparable we scale it back when logging.
		num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
		loss *= num_replicas_in_sync
		cls_loss *= num_replicas_in_sync
		box_loss *= num_replicas_in_sync
		giou_loss *= num_replicas_in_sync

		# Trainer class handles loss metric for you.
		logs = {self.loss: loss}

		all_losses = {
			'cls_loss': cls_loss,
			'box_loss': box_loss,
		   'giou_loss': giou_loss,
		}
		# Metric results will be added to logs for you.
		if metrics:
			for m in metrics:
				m.update_state(all_losses[m.name])
		return logs

	def validation_step(self, inputs, model, optimizer, metrics=None):
		raise NotImplementedError
