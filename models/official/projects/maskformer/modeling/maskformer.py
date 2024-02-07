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

"""MaskFormer Model Definition."""

import tensorflow as tf

from official.projects.maskformer.modeling.decoder.transformer_decoder import MaskFormerTransformer
from official.projects.maskformer.modeling.layers.nn_block import MLPHead
from official.projects.maskformer.modeling.decoder.transformer_pixel_decoder import TransformerFPN

import os


class MaskFormer(tf.keras.Model):
  """Maskformer"""

  def __init__(self,
               backbone,
               input_specs,
               fpn_feat_dims=256,
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation='relu',
               use_bias=False,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               num_queries=100,
               hidden_size=256,
               fpn_encoder_layers=6,
               detr_encoder_layers=0,
               num_decoder_layers=6,
               dropout_rate=0.1,
               backbone_endpoint_name='5',
               num_classes=199,
               batch_size=1,
               bfloat16=False,
               which_pixel_decoder='fpn',
               deep_supervision=False,
               **kwargs):
    """MaskFormer initialization function.
    Args:
      backbone: Backbone model for feature extraction.
      input_specs: Input specifications.
      fpn_feat_dims: `int`, FPN feature dimensions.
      data_format: `str`, Data format ('channels_first' or 'channels_last').
      dilation_rate: `tuple`, Dilation rate for convolution.
      groups: `int`, Number of groups for convolution.
      activation: `str`, Activation function.
      use_bias: `bool`, Whether to use bias.
      kernel_initializer: `str`, Kernel weights initializer.
      bias_initializer: `str`, Bias initializer.
      kernel_regularizer: Regularizer for kernel weights.
      bias_regularizer: Regularizer for bias.
      activity_regularizer: Regularizer for layer activity.
      kernel_constraint: Constraint for kernel weights.
      bias_constraint: Constraint for bias.
      num_queries: `int`, Number of query positions.
      hidden_size: `int`, Size of hidden layers.
      fpn_encoder_layers: `int`, Number of FPN encoder layers.
      detr_encoder_layers: `int`, Number of DETR encoder layers.
      num_decoder_layers: `int`, Number of decoder layers.
      dropout_rate: `float`, Dropout rate.
      backbone_endpoint_name: `str`, Endpoint name in backbone.
      num_classes: `int`, Number of classes.
      batch_size: `int`, Batch size.
      bfloat16: `bool`, Whether to use bfloat16.
      which_pixel_decoder: `str`, Type of pixel decoder.
      deep_supervision: `bool`, Use deep supervision.
    """

    super(MaskFormer, self).__init__(**kwargs)
    self._backbone = backbone
    self._input_specs = input_specs
    self._batch_size = batch_size
    self._num_classes = num_classes

    # Pixel Deocder paramters.
    self._fpn_feat_dims = fpn_feat_dims
    self._data_format = data_format
    self._dilation_rate = dilation_rate
    self._groups = groups
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._activity_regularizer = activity_regularizer
    self._kernel_constraint = kernel_constraint
    self._bias_constraint = bias_constraint

    # DETRTransformer parameters.
    self._fpn_encoder_layers = fpn_encoder_layers
    self._detr_encoder_layers = detr_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._num_queries = num_queries
    self._hidden_size = hidden_size
    self._dropout_rate = dropout_rate
    if hidden_size % 2 != 0:
      raise ValueError("hidden_size must be a multiple of 2.")
    self._bfloat16 = bfloat16
    self._pixel_decoder = which_pixel_decoder
    self._deep_supervision = deep_supervision

    # Backbone feature extractor.
    self._backbone_endpoint = backbone_endpoint_name

  def build(self, image_shape=None):
    """Builds the MaskFormer model."""
    self.pixel_decoder = TransformerFPN(
        batch_size=self._batch_size,
        fpn_feat_dims=self._fpn_feat_dims,
        data_format=self._data_format,
        dilation_rate=self._dilation_rate,
        groups=self._groups,
        activation=self._activation,
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        num_encoder_layers=self._fpn_encoder_layers,
        bfloat16=self._bfloat16)

    self.transformer = MaskFormerTransformer(
        num_queries=self._num_queries,
        hidden_size=self._hidden_size,
        num_encoder_layers=self._detr_encoder_layers,
        num_decoder_layers=self._num_decoder_layers,
        dropout_rate=self._dropout_rate,
        deep_supervision=self._deep_supervision)

    self.head = MLPHead(num_classes=self._num_classes,
                        hidden_dim=self._hidden_size,
                        mask_dim=self._fpn_feat_dims,
                        deep_supervision=self._deep_supervision)

    super(MaskFormer, self).build(image_shape)

  @property
  def backbone(self) -> tf.keras.Model:
    return self._backbone

  def get_config(self):
    return {
        "backbone": self._backbone,
        "backbone_endpoint_name": self._backbone_endpoint_name,
        "num_queries": self._num_queries,
        "hidden_size": self._hidden_size,
        "num_classes": self._num_classes,
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "dropout_rate": self._dropout_rate,
        "deep_supervision": self._deep_supervision,
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def process_feature_maps(self, maps):
    new_dict = {}
    for k in maps.keys():
      new_dict[k[0]] = maps[k]
    return new_dict

  def call(self, image: tf.Tensor, training=False):
    """ 
    Args:
      image: `tf.Tensor`, Input image.
      training: `bool`, Training or not.

    Returns:
      seg_pred: `tf.Tensor`, Segmentation prediction.
    """
    backbone_feature_maps = self._backbone(image)
    backbone_feature_maps_procesed = self.process_feature_maps(
        backbone_feature_maps)

    mask_features, transformer_enc_feat = self.pixel_decoder(
        backbone_feature_maps_procesed, image)

    transformer_features = self.transformer({"features": transformer_enc_feat})

    if self._deep_supervision:
      transformer_features = tf.convert_to_tensor(transformer_features)

    seg_pred = self.head({"per_pixel_embeddings": mask_features,
                          "per_segment_embeddings": transformer_features})
    
    return seg_pred
