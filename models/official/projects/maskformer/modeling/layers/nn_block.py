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

"""MLP Head for Instance and Semantic Probability Masks."""

import tensorflow as tf
import math
from typing import Any, Dict


class MLPHead(tf.keras.layers.Layer):
  def __init__(self,
               num_classes: int,
               hidden_dim: int,
               deep_supervision: bool,
               mask_dim: int):
    """MLPHead initialization function.
    Args:
      num_classes: `int`, Number of classes.
      hidden_dim: `int`, Dimension of hidden layer.
      deep_supervision: `bool`, Use deep supervision.
      mask_dim: `int`, Dimension for mask.
    """

    super().__init__()

    self._num_classes = num_classes
    self._hidden_dim = hidden_dim
    self._mask_dim = mask_dim
    self._deep_supervision = deep_supervision

  def build(self, input_shape):
    self._mlp = MLP(self._hidden_dim, self._hidden_dim, self._mask_dim, 3)
    sqrt_k = math.sqrt(1.0 / self._hidden_dim)
    self._linear_classifier = tf.keras.layers.Dense(
        self._num_classes + 1,
        name="class_embed",
        kernel_initializer=tf.keras.initializers.RandomUniform(
            -sqrt_k,
            sqrt_k),
        bias_initializer=tf.keras.initializers.RandomUniform(
            -sqrt_k,
            sqrt_k))

    super(MLPHead, self).build(input_shape)

  def call(self, inputs: Dict[str, Any]):
    """Passes the per_pixel_embeddings and per_segment_embeddings through the MLPHead.
    Args: 
      inputs: A dictionary of inputs.
    Returns:
      A dictionary of class and mask probability tensors.
    """

    per_pixel_embeddings = inputs['per_pixel_embeddings']  # mask feat
    # transformer feat
    per_segment_embeddings = inputs['per_segment_embeddings']

    class_prob_prediction = self._linear_classifier(per_segment_embeddings)
    mask_embedding = self._mlp(per_segment_embeddings)

    if self._deep_supervision:
      # mask embedding: [l, batch_size, num_queries, hidden_dim]
      mask_prob_prediction = tf.einsum(
          "lbqc,bhwc->lbhwq",
          mask_embedding,
          per_pixel_embeddings)
    else:
      # mask embedding: [batch_size, num_queries, hidden_dim]
      mask_prob_prediction = tf.einsum(
          "bqc,bhwc->bhwq", mask_embedding, per_pixel_embeddings)

    return {'class_prob_predictions': class_prob_prediction,
            'mask_prob_predictions': mask_prob_prediction}


class MLP(tf.keras.layers.Layer):
  def __init__(self,
               input_dim: int,
               hidden_dim: int,
               output_dim: int,
               num_layers: int):
    super().__init__()

    self._input_dim = input_dim
    self._hidden_dim = hidden_dim
    self._output_dim = output_dim
    self._num_layers = num_layers

  def build(self, input_shape):
    layer_dims = [(self._input_dim, self._hidden_dim)]
    for _ in range(self._num_layers - 2):
      layer_dims.append((self._hidden_dim, self._hidden_dim))
    layer_dims.append((self._hidden_dim, self._output_dim))
    sqrt_k = math.sqrt(1.0 / self._hidden_dim)

    self._layers = []
    for i, dim in enumerate(layer_dims):
      if (i < self._num_layers - 1):
        self._layers.append(
            tf.keras.layers.Dense(
                dim[1],
                activation=tf.nn.relu,
                bias_initializer=tf.keras.initializers.RandomUniform(
                    -sqrt_k,
                    sqrt_k)))
      else:
        # Final Layer
        self._layers.append(
            tf.keras.layers.Dense(
                dim[1],
                activation=None,
                bias_initializer=tf.keras.initializers.RandomUniform(
                    -sqrt_k,
                    sqrt_k)))

  def call(self, x):
    for layer in self._layers:
      x = layer(x)
    return x
