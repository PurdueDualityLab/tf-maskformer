# -*- coding: utf-8 -*-
"""maskformer_pixel_decoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OBXY3edFnqeY4OMRihQLzEPrxuVChtsR
"""

import tensorflow as tf
import tensorflow_addons as tfa
from official.vision.ops.spatial_transform_op import nearest_upsampling

class Fpn(tf.keras.layers.Layer):
  """Feature pyramid networks."""

  def __init__(self,
               fpn_feat_dims=256,
               channels_last = True,
               **kwargs):
    super(Fpn, self).__init__(**kwargs)
    """FPN initialization function.
    Args:
      fpn_feat_dims: Feature dimension of the fpn
      channels_last: Determines if shape is (bs, H, W, C)
    """
    self._fpn_feat_dims = fpn_feat_dims
    if tf.keras.backend.image_data_format() == 'channels_last':
        self._channels_last = True
    else:
        self._channels_last = False
    
  def build(self, multilevel_features):
    self._conv2d_op_lateral = tf.keras.layers.Conv2D(
          filters=self._fpn_feat_dims,
          kernel_size=(1, 1),
          padding='same')
    
    self._conv2d_op_down = tf.keras.layers.Conv2D(
          filters=self._fpn_feat_dims,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same')
    
    self._conv2d_op_mask = tf.keras.layers.Conv2D(
          filters=self._fpn_feat_dims,
          kernel_size=(3, 3),
          padding='same')
    
    super(Fpn, self).build(multilevel_features)

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
    with tf.name_scope('fpn'):
      # Adds top-down path.
      feats = multilevel_features
      outputs = []
      
      if not self._channels_last:
        feat = tf.keras.layers.Permute((2,3,1))(feats[input_levels[-1]])
      else:
        feat = feats[input_levels[-1]]

      down = tf.keras.layers.Conv2D(
          filters=self._fpn_feat_dims,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same')(feat)
      down = tfa.layers.GroupNormalization()(down)
      down = tf.keras.layers.ReLU()(down)

      levels = input_levels[:-1]
      for level in levels[::-1]:
        if not self._channels_last:
          feat = tf.keras.layers.Permute((2,3,1))(feats[level])
        else:
          feat = feats[level]
        
        lateral = tf.keras.layers.Conv2D(
          filters=self._fpn_feat_dims,
          kernel_size=(1, 1),
          padding='same')(feat)

        down = nearest_upsampling(down,2) + lateral
        
        down = tf.keras.layers.Conv2D(
          filters=self._fpn_feat_dims,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same')(down)
        down = tfa.layers.GroupNormalization()(down)
        down = tf.keras.layers.ReLU()(down)
   
      mask = self._conv2d_op_mask(down)

    return mask
