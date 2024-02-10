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

"""Tests for MaskFormer FPN."""

from absl.testing import parameterized
import tensorflow as tf
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
