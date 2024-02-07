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

"""Feature Pyramid Networks used in MaskFormer."""

import tensorflow as tf
from official.vision.ops.spatial_transform_ops import nearest_upsampling


class CNNFPN(tf.keras.layers.Layer):
  """MaskFormer Feature Pyramid Networks."""

  def __init__(self,
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
               **kwargs):
    """FPN initialization function.
    
    Args:
      fpn_feat_dims: `int`, Feature dimension of the FPN.
      data_format: `str`, Format of the data ('channels_first' or 'channels_last').
      dilation_rate: `tuple`, Dilation rate for convolution operations.
      groups: `int`, Number of groups for grouped convolution.
      activation: `str`, Activation function to use.
      use_bias: `bool`, Whether to use bias in the convolution layers.
      kernel_initializer: `str`, Initializer for the kernel weights.
      bias_initializer: `str`, Initializer for the bias vectors.
      kernel_regularizer: `regularizer`, Regularizer for the kernel weights.
      bias_regularizer: `regularizer`, Regularizer for the bias vectors.
      activity_regularizer: `regularizer`, Regularizer for the output of the layer.
      kernel_constraint: `constraint`, Constraint for the kernel weights.
      bias_constraint: `constraint`, Constraint for the bias vectors.
      **kwargs: Additional keyword arguments.
    """

    super(CNNFPN, self).__init__(**kwargs)

    # conv2d params
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

    if tf.keras.backend.image_data_format() == 'channels_last':
      # format: (batch_size, height, width, channels)
      self._channels_last = True
    else:
      # format: (batch_size, channels, width, height)
      self._channels_last = False

  def build(self, multilevel_features):
    conv_args = {
        "data_format": self._data_format,
        "dilation_rate": self._dilation_rate,
        "groups": self._groups,
        "activation": self._activation,
        "use_bias": self._use_bias,
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "kernel_regularizer": self._kernel_regularizer,
        "bias_regularizer": self._bias_regularizer,
        "activity_regularizer": self._activity_regularizer,
        "kernel_constraint": self._kernel_constraint,
        "bias_constraint": self._bias_constraint
    }

    input_levels = list(multilevel_features.keys())
    levels = input_levels[:-1]

    self._conv2d_op_lateral = []
    self._lateral_groupnorm = []
    for level in levels[::-1]:
      lateral = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                       kernel_size=(1, 1),
                                       padding='same',
                                       name=f"lateral_{level}",
                                       **conv_args)
      lateral_norm = tf.keras.layers.GroupNormalization(
          name=f"lateral_norm_{level}")
      self._conv2d_op_lateral.append(lateral)
      self._lateral_groupnorm.append(lateral_norm)

    self._conv2d_op_down = []
    self._down_groupnorm = []
    down = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                  strides=(1, 1),
                                  kernel_size=(3, 3),
                                  padding='same',
                                  name="down_initial_conv",
                                  **conv_args)
    down_norm = tf.keras.layers.GroupNormalization(name="down_initial_norm")
    self._down_groupnorm.append(down_norm)
    self._conv2d_op_down.append(down)

    for level in levels[::-1]:
      down = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                    strides=(1, 1),
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name=f"down_{level}",
                                    **conv_args)
      down_norm = tf.keras.layers.GroupNormalization(name=f"down_norm_{level}")
      self._conv2d_op_down.append(down)
      self._down_groupnorm.append(down_norm)

    self._conv2d_op_mask = tf.keras.layers.Conv2D(
        filters=self._fpn_feat_dims,
        kernel_size=(3, 3),
        padding='same',
        name="mask_proj",
        **conv_args)

    self._relu1 = tf.keras.layers.ReLU()
    self._relu2 = tf.keras.layers.ReLU()

    if not self._channels_last:
      self._permute1 = tf.keras.layers.Permute((2, 3, 1))
      self._permute2 = tf.keras.layers.Permute((2, 3, 1))

    super(CNNFPN, self).build(multilevel_features)

  def call(self, multilevel_features):
    """Returns the FPN features for a given multilevel features.

    Args:
      multilevel_features: a `dict` containing `int` keys for continuous feature
        levels, e.g., [2, 3, 4, 5]. The values are corresponding features with
        shape [batch_size, height_l, width_l, num_filters].

    Returns:
      Mask projection
    """
    input_levels = list(multilevel_features.keys())

    feat = multilevel_features[input_levels[-1]]

    if not self._channels_last:
      feat = self._permute_1(feat)

    down = self._conv2d_op_down[0](feat)
    down = self._down_groupnorm[0](down)
    down = self._relu1(down)

    levels = input_levels[:-1]
    for i, level in enumerate(levels[::-1]):
      feat = multilevel_features[level]

      if not self._channels_last:
        feat = self._permute_2(multilevel_features[level])

      lateral = self._conv2d_op_lateral[i](feat)
      lateral = self._lateral_groupnorm[i](lateral)
      down = nearest_upsampling(down, 2) + lateral
      down = self._conv2d_op_down[i + 1](down)
      down = self._down_groupnorm[i + 1](down)
      down = self._relu2(down)

    mask = self._conv2d_op_mask(down)

    return mask
