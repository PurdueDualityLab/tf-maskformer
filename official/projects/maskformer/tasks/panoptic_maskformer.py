# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Panoptic MaskFormer Task definition."""

from absl import logging
import tensorflow as tf

from official.core import base_task
from official.core import task_factory
from typing import Any, Dict, List, Optional, Tuple

from official.vision.dataloaders import input_reader_factory
from official.common import dataset_fn

from official.projects.maskformer.configs import maskformer as maskformer_cfg
from official.projects.maskformer.modeling.maskformer import MaskFormer
from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.maskformer.dataloaders import panoptic_input

from official.projects.detr.ops.matchers import hungarian_matching
from official.projects.maskformer.losses.maskformer_losses import Loss

from official.projects.maskformer.losses.inference import PanopticInference
from official.projects.maskformer.losses.quality import PanopticQualityMetric
from official.vision.modeling import backbones

import time

@task_factory.register_task_cls(maskformer_cfg.MaskFormerTask)
class PanopticTask(base_task.Task):
  # pylint: disable=line-too-long
  """A single-replica view of training procedure.

  PanopticTask task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def build_model(self):
    """Builds MaskFormer Model."""
    logging.info('[INFO] Building MaskFormer model.')

    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self._task_config.model.input_size)

    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=self._task_config.model.backbone,
        norm_activation_config=self._task_config.model.norm_activation)
    logging.info('[INFO] Backbone build successful.')

    model = MaskFormer(
        backbone=backbone,
        input_specs=input_specs,
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
    logging.info('[INFO] Maskformer model build successful.')

    self.ITER_IDX = 1

    return model

  def initialize(self, model: tf.keras.Model) -> None:
    """
    Used to initialize the models with checkpoint
    """

    logging.info(
        'Initializing model from checkpoint: %s',
        self._task_config.init_checkpoint)

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
      logging.info('[INFO] Loaded whole model from %s', ckpt_dir_or_file)

    elif self._task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
      logging.info('[INFO] Finished loading backbone checkpoint from %s',
                   ckpt_dir_or_file)
    else:
      raise ValueError('Not a valid module to initialize from: {}'.format(
          self._task_config.init_checkpoint_modules))

  def build_inputs(
          self, params: Dict[str, Any], input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
    """Build panoptic segmentation dataset.
    Args: 
      params: a dictionary of parameters.
      input_context: a tf.distribute.InputContext instance.
    Returns: 
      A tf.data.Dataset instance.
    """
    logging.info('[INFO] Building panoptic segmentation dataset.')

    if params.decoder.type == 'simple_decoder':
      decoder = panoptic_input.TfExampleDecoder(
          regenerate_source_id=params.regenerate_source_id)
    else:
      raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))

    parser = panoptic_input.mask_former_parser(
        params.parser,
        is_training=params.is_training,
        decoder_fn=decoder.decode)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)
    return dataset

  def build_losses(self, output: Dict[str, Any], labels: Dict[str, Any], aux_outputs=None):
    # pylint: disable=line-too-long
    """ Builds panoptic segmentation losses for batch of images.
    Args: 
      output: a dictionary of output tensors.
      labels: a dictionary of input tensors.
      aux_outputs: a list of auxiliary output tensors.
    Returns:
      A tuple of total loss, weighted cross entropy loss, weighted focal loss, and weighted dice loss.
    """

    outputs = {"pred_logits": output["class_prob_predictions"],
               "pred_masks": output["mask_prob_predictions"]}

    if not aux_outputs:
      logging.info('[INFO] Deep Supervision disabled (FALSE).')
    else:
      logging.info('[INFO] Deep Supervision enabled (TRUE).')
      outputs["pred_masks"] = output["mask_prob_predictions"][-1]
      outputs["pred_logits"] = output["class_prob_predictions"][-1]

      formatted_aux_output = []

      for i in range(len(output["class_prob_predictions"][:-1])):
        formatted_aux_output += [{"pred_logits": output["class_prob_predictions"]
                                  [i], "pred_masks": output["mask_prob_predictions"][i]}]

      outputs.update({"aux_outputs": formatted_aux_output})

    targets = labels

    matcher = hungarian_matching

    no_object_weight = self._task_config.losses.background_cls_weight
    loss = Loss(
        num_classes=self._task_config.model.num_classes,
        matcher=matcher,
        eos_coef=no_object_weight,
        cost_class=self._task_config.losses.cost_class,
        cost_dice=self._task_config.losses.cost_dice,
        cost_focal=self._task_config.losses.cost_focal,
        ignore_label=self._task_config.train_data.parser.ignore_label)

    calculated_losses = loss(outputs, targets)

    # Losses are returned as weighted sum of individual losses
    logging.info('[INFO] Weighting losses.')
    total_loss = calculated_losses['loss_ce'] + \
        calculated_losses['loss_dice'] + calculated_losses['loss_focal']

    weighted_ce = calculated_losses['loss_ce']
    weighted_dice = calculated_losses['loss_dice']
    weighted_focal = calculated_losses['loss_focal']

    aux_outputs = outputs.get('aux_outputs')

    if aux_outputs is not None:
      logging.info('[INFO] Weighting multi-scale losses.')
      total_aux_loss = 0.0
      for i in range(len(aux_outputs)):
        total_aux_loss += calculated_losses['loss_ce_' + str(
            i)] + calculated_losses['loss_dice_' + str(i)] + calculated_losses['loss_focal_' + str(i)]
      total_loss = total_loss + total_aux_loss

    return total_loss, weighted_ce, weighted_focal, weighted_dice

  def build_metrics(self, training=True):
    # pylint: disable=line-too-long
    """Builds panoptic metrics.
    Args: 
      training: a boolean indicating if training is enabled.
    Returns:
      A PanopticQuality or PanopticQualityV2 class instance depending on compute platform  
    """

    metrics = []
    metric_names = ['cls_loss', 'focal_loss', 'dice_loss']
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

    if not training:
      logging.info('[INFO] Building Panoptic Inference and Quality.')

      pq_config = self._task_config.panoptic_quality_evaluator

      self.panoptic_inference = PanopticInference(
          num_classes=self._task_config.model.num_classes,
          background_class_id=pq_config.ignored_label
      )

      self.panoptic_quality_metric = PanopticQualityMetric(
          on_tpu=self._task_config.model.on_tpu,
          pq_config=pq_config
      )

    return metrics

  def train_step(self,
                 inputs: Tuple[Any,
                               Any],
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None) -> Dict[str,
                                                              Any]:
    # pylint: disable=line-too-long
    """Does forward and backward.
    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.
    Returns:
      A dictionary of logs.
    """
    start = time.time() 
    logging.info('[INFO] Starting training step: %s', self.ITER_IDX)

    features, labels = inputs

    with tf.GradientTape() as tape:
      outputs = model(features, training=True)

      logging.info('[INFO] Building Losses: %s', self.ITER_IDX)

      total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(
          output=outputs, labels=labels, aux_outputs=model._deep_supervision)
      scaled_loss = total_loss

      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        total_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)

    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
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

    logging.info('[INFO] Losses: total_loss(%s), cls_loss(%s), focal_loss(%s), dice_loss(%s): %s',
                 float(total_loss), float(cls_loss), float(focal_loss), float(dice_loss), self.ITER_IDX)

    logs = {self.loss: total_loss}

    all_losses = {
        'cls_loss': cls_loss,
        'focal_loss': focal_loss,
        'dice_loss': dice_loss, }

    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])

    logging.info('[INFO] Finished training step: %s', self.ITER_IDX)
    logging.info('[INFO] Time Taken for step %s: %f', self.ITER_IDX, time.time() - start)

    self.ITER_IDX += 1

    return logs

  def _postprocess_outputs(
          self, labels: Dict[str, Any], outputs: Dict[str, Any], image_shapes: List[int], deep_supervision: bool):
    # pylint: disable=line-too-long
    """Postprocesses the predictions and labels for Panoptic Quality Metric. Returns metrics.
    Args:
      outputs: a dictionary of output tensors.
      image_shapes: a list of image shapes.
      deep_supervision: a boolean indicating if deep supervision is enabled.
    Returns:
      Result of Panoptic Quality Metric (processed)
    """

    pred_binary_masks = outputs["mask_prob_predictions"]
    pred_labels = outputs["class_prob_predictions"]
    if deep_supervision:
      pred_binary_masks = pred_binary_masks[-1]
      pred_labels = pred_labels[-1]

    output_instance_mask, output_contiguous_mask = self.panoptic_inference(
        pred_labels, pred_binary_masks, image_shapes)


    # PQ Module expects the following format for labels and inputs. 
    ## We use 'category_mask' = contiguous mask as the PQ module expects the same.
    
    pq_metric_labels = {
        'category_mask': labels['contiguous_mask'],
        'instance_mask': labels['instance_mask']
    }

    pq_metric_inputs = {
        'category_mask': output_contiguous_mask,
        'instance_mask': output_instance_mask
    }

    if output_instance_mask.shape[0] == 0 or output_contiguous_mask.shape[0] == 0:
      raise ValueError('Post Processed Predictions are empty. \
      This can only happen if the model has not been trained. \
      Please train the model before running eval.')

    results = self.panoptic_quality_metric(
        pq_metric_labels, pq_metric_inputs)

    return results

  def validation_step(self, inputs: Tuple[Dict[str, Any]], model: tf.keras.Model, metrics=None):
    """Operated in eval mode.
    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      metrics: a nested structure of metrics objects.
    Returns:
      A dictionary of logs.
    """
    start = time.time() 

    logging.info('[INFO] Starting validation step: %s', self.ITER_IDX)

    features, labels = inputs
  
    outputs = model(features, training=False)

    logging.info('[INFO] Building Losses: %s', self.ITER_IDX)

    total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(
        output=outputs, labels=labels, aux_outputs=model._deep_supervision)

    num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
    total_loss *= num_replicas_in_sync
    cls_loss *= num_replicas_in_sync
    focal_loss *= num_replicas_in_sync
    dice_loss *= num_replicas_in_sync
    logs = {self.loss: total_loss}

    logging.info('[INFO] Losses: total_loss(%s), cls_loss(%s), focal_loss(%s), dice_loss(%s): %s',
                 float(total_loss), float(cls_loss), float(focal_loss), float(dice_loss), self.ITER_IDX)

    all_losses = {
        'cls_loss': cls_loss,
        'focal_loss': focal_loss,
        'dice_loss': dice_loss,
    }

    results = self._postprocess_outputs(
        labels, outputs, [640, 640], model._deep_supervision)

    for key, value in results.items():
      logs[key] = value

    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])

    logging.info('[INFO] Finished validation step: %s', self.ITER_IDX)
    logging.info('[INFO] Time Taken for step %s: %f', self.ITER_IDX, time.time() - start)

    self.ITER_IDX += 1

    return logs