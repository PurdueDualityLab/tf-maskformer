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

"""Pixel Decoder Network for MaskFormer.""" 

import tensorflow as tf
import math
from official.modeling import tf_utils

from official.projects.maskformer.modeling.transformer.transformer import TransformerEncoder

def position_embedding_sine(attention_mask,
                            num_pos_features=256,
                            temperature=10000.,
                            normalize=True,
                            scale=2 * math.pi):
  # pylint: disable=line-too-long
  """Sine-based positional embeddings for 2D images.

  Args:
          attention_mask: a `bool` Tensor specifying the size of the input image to
          the Transformer and which elements are padded, of size [batch_size,
          height, width]
          num_pos_features: a `int` specifying the number of positional features,
          should be equal to the hidden size of the Transformer network
          temperature: a `float` specifying the temperature of the positional
          embedding. Any type that is converted to a `float` can also be accepted.
          normalize: a `bool` determining whether the positional embeddings should be
          normalized between [0, scale] before application of the sine and cos
          functions.
          scale: a `float` if normalize is True specifying the scale embeddings before
          application of the embedding function.

  Returns:
          embeddings: a `float` tensor of the same shape as input_tensor specifying
          the positional embeddings based on sine features.
  """
  if num_pos_features % 2 != 0:
    raise ValueError(
        "Number of embedding features (num_pos_features) must be even when "
        "column and row embeddings are concatenated.")
  num_pos_features = num_pos_features // 2

  # Produce row and column embeddings based on total size of the image
  # <tf.float>[batch_size, height, width]
  attention_mask = tf.cast(attention_mask, tf.float32)
  row_embedding = tf.cumsum(attention_mask, 1)
  col_embedding = tf.cumsum(attention_mask, 2)

  if normalize:
    eps = 1e-6
    row_embedding = row_embedding / (row_embedding[:, -1:, :] + eps) * scale
    col_embedding = col_embedding / (col_embedding[:, :, -1:] + eps) * scale

  dim_t = tf.range(num_pos_features, dtype=row_embedding.dtype)
  dim_t = tf.pow(temperature, 2 * (dim_t // 2) / num_pos_features)

  # Creates positional embeddings for each row and column position
  # <tf.float>[batch_size, height, width, num_pos_features]
  pos_row = tf.expand_dims(row_embedding, -1) / dim_t
  pos_col = tf.expand_dims(col_embedding, -1) / dim_t
  pos_row = tf.stack(
      [tf.sin(pos_row[:, :, :, 0::2]),
       tf.cos(pos_row[:, :, :, 1::2])], axis=4)
  pos_col = tf.stack(
      [tf.sin(pos_col[:, :, :, 0::2]),
       tf.cos(pos_col[:, :, :, 1::2])], axis=4)

  # final_shape = pos_row.shape.as_list()[:3] + [-1]
  final_shape = tf_utils.get_shape_list(pos_row)[:3] + [-1]
  pos_row = tf.reshape(pos_row, final_shape)
  pos_col = tf.reshape(pos_col, final_shape)
  output = tf.concat([pos_row, pos_col], -1)

  embeddings = tf.cast(output, tf.float32)
  return embeddings

class TransformerFPN(tf.keras.layers.Layer):
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
               num_encoder_layers=0,
               bfloat16=True,
               **kwargs):
    """FPN initialization function.

      Args:
        fpn_feat_dims (int): The number of feature dimensions in the FPN outputs.
        data_format (str): The data format ('channels_first' or 'channels_last').
        dilation_rate (tuple): The dilation rate for dilated convolution.
        groups (int): The number of groups for grouped convolution.
        activation (str): The activation function to use.
        use_bias (bool): Whether to use bias in the convolution layers.
        kernel_initializer (str): Initializer for the kernel weights.
        bias_initializer (str): Initializer for the bias vectors.
        kernel_regularizer (regularizer): Regularizer function for the kernel weights.
        bias_regularizer (regularizer): Regularizer function for the bias vectors.
        activity_regularizer (regularizer): Regularizer function applied to the output.
        kernel_constraint (constraint): Constraint function applied to the kernel weights.
        bias_constraint (constraint): Constraint function applied to the bias vectors.
        num_encoder_layers (int): The number of encoder layers in the transformer.
        bfloat16 (bool): Whether to use bfloat16 precision.
        **kwargs: Additional keyword arguments for layer configuration.
    """
    super(TransformerFPN, self).__init__(**kwargs)

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
    self._num_encoder_layers = num_encoder_layers
    self._bfloat16 = bfloat16

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
        "activation": None,
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

    self._input_proj = tf.keras.layers.Conv2D(
        filters=self._fpn_feat_dims,
        kernel_size=(
            1,
            1),
        padding='same',
        name=f"input_proj",
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='glorot_uniform',
    )

    self._transformer_encoder = TransformerEncoder(
        norm_first=False,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        intermediate_dropout=0.1,
        num_layers=self._num_encoder_layers,
    )
    self._interpolations = []
    self._conv2d_op_lateral = []
    self._lateral_groupnorm = []
    for level in levels[::-1]:
      lateral = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                       kernel_size=(1, 1),
                                       padding='same',
                                       name=f"lateral_{level}",
                                       kernel_initializer='glorot_uniform',
                                       use_bias=False,
                                       # **conv_args
                                       )
      lateral_norm = tf.keras.layers.GroupNormalization(
          name=f"lateral_norm_{level}")
      interpolate = tf.keras.layers.Resizing(
          multilevel_features[level][1],
          multilevel_features[level][2],
          interpolation="nearest")

      self._conv2d_op_lateral.append(lateral)
      self._lateral_groupnorm.append(lateral_norm)
      self._interpolations.append(interpolate)

    self._conv2d_op_down = []
    self._down_groupnorm = []

    down = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                  strides=(1, 1),
                                  kernel_size=(3, 3),
                                  padding='same',
                                  name="down_initial_conv",
                                  kernel_initializer='glorot_uniform',
                                  use_bias=False,
                                  # **conv_args
                                  )
    down_norm = tf.keras.layers.GroupNormalization(name="down_initial_norm")
    self._down_groupnorm.append(down_norm)
    self._conv2d_op_down.append(down)

    for level in levels[::-1]:
      # pylint: disable=line-too-long
      down = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims, 
                                    strides=(1, 1),
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name=f"down_{level}",
                                    kernel_initializer='glorot_uniform', use_bias=False,
                                    # **conv_args
                                    )
      down_norm = tf.keras.layers.GroupNormalization(name=f"down_norm_{level}")
      self._conv2d_op_down.append(down)
      self._down_groupnorm.append(down_norm)

    self._conv2d_op_mask = tf.keras.layers.Conv2D(
        filters=self._fpn_feat_dims,
        kernel_size=(3, 3),
        padding='same',
        name="mask_proj",
        kernel_initializer='glorot_uniform',
        use_bias=True, bias_initializer='glorot_uniform',
        # **conv_args
    )

    self._relu1 = tf.keras.layers.ReLU()
    self._relu2 = tf.keras.layers.ReLU()

    if not self._channels_last:
      self._permute1 = tf.keras.layers.Permute((2, 3, 1))
      self._permute2 = tf.keras.layers.Permute((2, 3, 1))

    super(TransformerFPN, self).build(multilevel_features)

  def _generate_image_mask(
          self,
          inputs: tf.Tensor,
          target_shape: tf.Tensor) -> tf.Tensor:
    mask = tf.expand_dims(
        tf.cast(
            tf.not_equal(
                tf.reduce_sum(
                    inputs,
                    axis=-1),
                0),
            inputs.dtype),
        axis=-1)
    mask = tf.image.resize(
        mask,
        target_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return mask

  def call(self, multilevel_features, image):
    # pylint: disable=line-too-long
    """
    Returns the FPN features for a given multilevel features.

    Args:
      multilevel_features: a `dict` containing `int` keys for continuous feature
              levels, e.g., [2, 3, 4, 5]. The values are corresponding features with
              shape [batch_size, height_l, width_l, num_filters].

    Returns:
      Mask projection
    """
    input_levels = list(multilevel_features.keys())

    # use the low resolution features first
    feat = multilevel_features[input_levels[-1]]

    if not self._channels_last:
      feat = self._permute_1(feat)
    shape = tf.shape(feat)
    mask = self._generate_image_mask(image, shape[1: 3])
    features = self._input_proj(feat)
    pos_embed = position_embedding_sine(
        mask[:, :, :, 0], num_pos_features=self._fpn_feat_dims)

    # with options({"layout_optimizer": False}):
    features = tf.reshape(features, [tf.shape(
        features)[0], -1, tf.shape(features)[-1]])  # (2, 400, 2048)
    pos_embed = tf.reshape(pos_embed, [tf.shape(pos_embed)[
                           0], -1, tf.shape(pos_embed)[-1]])  # (2, 400, 256)
    transformer = self._transformer_encoder(
        features, attention_mask=None, pos_embed=pos_embed)
    transformer = tf.reshape(transformer, [tf.shape(transformer)[0], tf.shape(feat)[
                             1], tf.shape(feat)[2], tf.shape(transformer)[-1]])

    down = self._conv2d_op_down[0](transformer)
    down = self._down_groupnorm[0](down)
    down = self._relu1(down)
    transformer_encoder_features = down

    levels = input_levels[:-1]
    for i, level in enumerate(levels[::-1]):
      feat = multilevel_features[level]

      if not self._channels_last:
        feat = self._permute_2(multilevel_features[level])

      lateral = self._conv2d_op_lateral[i](feat)
      lateral = self._lateral_groupnorm[i](lateral)

      down = self._interpolations[i](down) + lateral

      down = self._conv2d_op_down[i + 1](down)
      down = self._down_groupnorm[i + 1](down)
      down = self._relu2(down)

    mask_features = self._conv2d_op_mask(down)

    return mask_features, transformer_encoder_features
