from typing import Optional

import tensorflow as tf
from official.vision.modeling.backbones import resnet

def build_maskformer_backbone(
      model_id: int = 50,
      depth_multiplier: float = 1.0,
      stem_type: str = 'v0',
      resnetd_shortcut: bool = False,
      replace_stem_max_pool: bool = False,
      se_ratio: Optional[float] = None,
      init_stochastic_depth_rate: float = 0.0,
      scale_stem: bool = True,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bn_trainable: bool = True,
      **kwargs):
    return resnet.ResNet(# PASS VARIABLES)