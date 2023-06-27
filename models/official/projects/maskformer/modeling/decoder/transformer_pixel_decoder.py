import tensorflow as tf
import tensorflow_addons as tfa
from official.vision.ops.spatial_transform_ops import nearest_upsampling
from official.projects.detr.modeling.detr import position_embedding_sine
from official.projects.detr.modeling.transformer import TransformerEncoder
from official.projects.maskformer.modeling.decoder.detr_transformer import DETRTransformer

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
                 num_encoder_layers = 0,
                 **kwargs):
        """FPN initialization function.
        Args:
          fpn_feat_dims: `int`, Feature dimension of the fpn.
          
          TODO: fill in new args
          
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

        self._input_proj = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                                  kernel_size=(1, 1),
                                                  padding='same',
                                                  name = f"input_proj",
                                                  use_bias = True)
        self._transformer_encoder = TransformerEncoder(norm_first=False,
                                                       dropout_rate = .1,
                                                       num_layers=self._num_encoder_layers)
        self._interpolations = []                                               
        self._conv2d_op_lateral = []
        self._lateral_groupnorm = []
        for level in levels[::-1]:
            lateral = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                             kernel_size=(1, 1),
                                             padding='same',
                                             name = f"lateral_{level}",
                                             **conv_args)
            lateral_norm = tf.keras.layers.GroupNormalization(name = f"lateral_norm_{level}")
            interpolate = tf.keras.layers.Resizing(
              multilevel_features[level][1], multilevel_features[level][2], interpolation = "nearest")

            self._conv2d_op_lateral.append(lateral)
            self._lateral_groupnorm.append(lateral_norm)
            self._interpolations.append(interpolate)

        self._conv2d_op_down = []
        self._down_groupnorm = []
        down = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                      strides=(1, 1),
                                      kernel_size=(3, 3),
                                      padding='same',
                                      name = "down_initial_conv",
                                      **conv_args)
        down_norm = tf.keras.layers.GroupNormalization(name = "down_initial_norm")
        self._down_groupnorm.append(down_norm)
        self._conv2d_op_down.append(down)
        
        for level in levels[::-1]:
            down = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims,
                                          strides=(1, 1),
                                          kernel_size=(3, 3),
                                          padding='same',
                                          name = f"down_{level}",
                                          **conv_args)
            down_norm = tf.keras.layers.GroupNormalization(name = f"down_norm_{level}")
            self._conv2d_op_down.append(down)
            self._down_groupnorm.append(down_norm)

        self._conv2d_op_mask = tf.keras.layers.Conv2D(
            filters=self._fpn_feat_dims,
            kernel_size=(3, 3),
            padding='same',
            name = "mask_proj",
            **conv_args)
        
        self._relu1 = tf.keras.layers.ReLU()
        self._relu2 = tf.keras.layers.ReLU()

        if not self._channels_last:
            self._permute1 = tf.keras.layers.Permute((2, 3, 1))
            self._permute2 = tf.keras.layers.Permute((2, 3, 1))

        super(TransformerFPN, self).build(multilevel_features)
    
    def _generate_image_mask(self, features: tf.Tensor) -> tf.Tensor:
        """Generates image mask from input image."""
        mask = tf.zeros([features.shape[0],features.shape[1],features.shape[2]])
        mask = tf.cast(mask, dtype = bool)
        return mask

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
        
        mask = self._generate_image_mask(feat)
        pos_embed = position_embedding_sine(
            mask, num_pos_features=self._fpn_feat_dims)

        features = self._input_proj(feat)
        
        transformer = self._transformer_encoder(features, None, pos_embed)
        

        down = self._conv2d_op_down[0](transformer)
        down = self._down_groupnorm[0](down)
        down = self._relu1(down)

        levels = input_levels[:-1]
        for i, level in enumerate(levels[::-1]):
            feat = multilevel_features[level]

            if not self._channels_last:
                feat = self._permute_2(multilevel_features[level])

            lateral = self._conv2d_op_lateral[i](feat)
            lateral = self._lateral_groupnorm[i](lateral)

            down = self._interpolations[i](down) + lateral

            down = self._conv2d_op_down[i + 1](down)
            down = self._down_groupnorm[i+1](down)
            down = self._relu2(down)

        mask = self._conv2d_op_mask(down)

        return mask, transformer
