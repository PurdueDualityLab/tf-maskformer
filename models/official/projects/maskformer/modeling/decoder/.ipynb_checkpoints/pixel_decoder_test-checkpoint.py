# -*- coding: utf-8 -*-
"""pixel_decoder_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lGoBLb-lNoR9fJ0VVm6g00Cbpk40hdwy
"""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations

from official.projects.maskformer.modeling.decoder.pixel_decoder import Fpn


class FpnTest(parameterized.TestCase, tf.test.TestCase):

    @parameterized.named_parameters(('test1', 256),)
    def test_pass_through(self, dim):

        multilevel_features = {
            "2": tf.ones([1, 160, 160, 256]),
            "3": tf.ones([1, 80, 80, 512]),
            "4": tf.ones([1, 40, 40, 1024]),
            "5": tf.ones([1, 20, 20, 2048])
        }

        # TODO(Isaac): Add the additional parameters.
        decoder = Fpn(fpn_feat_dims=dim)
        output_mask = decoder(multilevel_features)

        expected_output_mask = multilevel_features["2"].shape.as_list()

        self.assertAllEqual(output_mask.shape.as_list(), expected_output_mask)

    @combinations.generate(
        combinations.combine(
            strategy=[
                strategy_combinations.cloud_tpu_strategy,
                strategy_combinations.one_device_strategy_gpu,
            ],
            use_sync_bn=[False, True],
        ))
    def test_sync_bn_multiple_devices(self, strategy, use_sync_bn):
        """Test for sync bn on TPU and GPU devices."""

        tf.keras.backend.set_image_data_format('channels_last')

        with strategy.scope():

            multilevel_features = {
                2: tf.ones([1, 160, 160, 256]),
                3: tf.ones([1, 80, 80, 512]),
                4: tf.ones([1, 40, 40, 1024]),
                5: tf.ones([1, 20, 20, 2048])}

            decoder = Fpn()
            _ = decoder(multilevel_features)


if __name__ == '__main__':
    tf.test.main()
