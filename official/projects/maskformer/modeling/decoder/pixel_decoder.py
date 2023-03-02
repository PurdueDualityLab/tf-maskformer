import tensorflow as tf
import tensorflow_addons as tfa
from official.vision.ops.spatial_transform_ops import nearest_upsampling

class Fpn(tf.keras.layers.Layer):
    """Feature pyramid networks."""

    def __init__(self,
                 fpn_feat_dims=256,
                 **kwargs):
        """FPN initialization function.

        Args:
        fpn_feat_dims: Feature dimension of the fpn
        """
        super(Fpn, self).__init__(**kwargs)

        self._fpn_feat_dims = fpn_feat_dims
        # TODO(Isaac): Add Conv2D parameteres to constructor.
        # TODO(Isaac): Add GroupNormalization parameters to constructor.

        if tf.keras.backend.image_data_format() == 'channels_last':
            self._channels_last = True
        else:
            self._channels_last = False

    def build(self, multilevel_features):
        # TODO(Isaac): Add Conv2D parameters to layers.
        # TODO(Isaac): Add GroupNormalization parameters to layers.

        input_levels = list(multilevel_features.keys())
        levels = input_levels[:-1]

        self._conv2d_op_lateral = []
        for _ in levels[::-1]:
            lateral = tf.keras.layers.Conv2D(
                filters=self._fpn_feat_dims,
                kernel_size=(1, 1),
                padding='same')
            self._conv2d_op_lateral.append(lateral)

        self._conv2d_op_down = []
        down = tf.keras.layers.Conv2D(
            filters=self._fpn_feat_dims,
            strides=(1, 1),
            kernel_size=(3, 3),
            padding='same')
        self._conv2d_op_down.append(down)
        for _ in levels[::-1]:
            down = tf.keras.layers.Conv2D(
                filters=self._fpn_feat_dims,
                strides=(1, 1),
                kernel_size=(3, 3),
                padding='same')
            self._conv2d_op_down.append(down)

        self._conv2d_op_mask = tf.keras.layers.Conv2D(
            filters=self._fpn_feat_dims,
            kernel_size=(3, 3),
            padding='same')

        self._group_norm1 = tfa.layers.GroupNormalization()
        self._group_norm2 = tfa.layers.GroupNormalization()
        
        self._relu1 = tf.keras.layers.ReLU()
        self._relu2 = tf.keras.layers.ReLU()

        if not self._channels_last:
            self._permute1 = tf.keras.layers.Permute((2, 3, 1))
            self._permute2 = tf.keras.layers.Permute((2, 3, 1))

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

        feat = multilevel_features[input_levels[-1]]

        if not self._channels_last:
            feat = self._permute_1(feat)

        down = self._conv2d_op_down[0](feat)
        down = self._group_norm1(down)
        down = self._relu1(down)

        levels = input_levels[:-1]
        for i, level in enumerate(levels[::-1]):
            feat = multilevel_features[level]

            if not self._channels_last:
                feat = self._permute_2(multilevel_features[level])

            lateral = self._conv2d_op_lateral[i](feat)

            down = nearest_upsampling(down, 2) + lateral

            down = self._conv2d_op_down[i + 1](down)
            down = self._group_norm2(down)
            down = self._relu2(down)

        mask = self._conv2d_op_mask(down)

        return mask
