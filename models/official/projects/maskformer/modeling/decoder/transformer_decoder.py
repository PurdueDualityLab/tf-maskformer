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

"""Transformer Decoder Network in MaskFormer, uses DETRTransformer.""" 

import math
import tensorflow as tf
from official.modeling import tf_utils
from official.projects.maskformer.modeling.decoder.detr_transformer import DETRTransformer
from typing import Any, Dict

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


class MaskFormerTransformer(tf.keras.layers.Layer):
  def __init__(self,
               num_queries,
               hidden_size,
               num_encoder_layers=0,
               num_decoder_layers=6,
               deep_supervision=False,
               dropout_rate=0.1,
               **kwargs):
    super().__init__(**kwargs)

    # Embeddings parameters.
    self._num_queries = num_queries
    self._hidden_size = hidden_size
    if hidden_size % 2 != 0:
      raise ValueError("hidden_size must be a multiple of 2.")

    # DETRTransformer parameters.
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._dropout_rate = dropout_rate
    self._deep_supervision = deep_supervision

  def build(self, input_shape):

    self._transformer = DETRTransformer(
        num_encoder_layers=self._num_encoder_layers,
        num_decoder_layers=self._num_decoder_layers,
        dropout_rate=self._dropout_rate,
        deep_supervision=self._deep_supervision)

    self._query_embeddings = self.add_weight(
        "detr/query_embeddings",
        shape=[self._num_queries, self._hidden_size],
        initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
        dtype=tf.float32)

    super(MaskFormerTransformer, self).build(input_shape)

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

  def call(self, inputs: Dict[str, Any]):
    """ 
    Passes the input image features through the customized DETR Transformer

    Args:
      inputs: A dictionary of inputs.

    Returns: 
      A dictionary of decoded list of features.
    """
    features = inputs['features']

    batch_size = tf.shape(features)[0]
    shape = tf.shape(features)
    mask = self._generate_image_mask(features, shape[1: 3])
    pos_embed = position_embedding_sine(
        mask[:, :, :, 0], num_pos_features=self._hidden_size)

    pos_embed = tf.reshape(pos_embed, [batch_size, -1, self._hidden_size])
    features = tf.reshape(features, [batch_size, -1, self._hidden_size])
    mask = None
    decoded_list = self._transformer({
        "inputs":
        features,
            "targets":
        tf.tile(
            tf.expand_dims(self._query_embeddings, axis=0),
            (batch_size, 1, 1)),
            "pos_embed": pos_embed,
            "mask": mask,
    })
    return decoded_list

  def get_config(self):
    return {
        "num_queries": self._num_queries,
        "hidden_size": self._hidden_size,
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "dropout_rate": self._dropout_rate,
    }
