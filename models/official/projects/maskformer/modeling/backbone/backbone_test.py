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

"""Tests for tensorflow_models.official.vision.modeling.backbones.resnet."""

from absl.testing import parameterized
import tensorflow as tf

from official.vision.modeling.backbones import resnet


class ResNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (640, 50),
  )
  def test_network_creation(self, input_size, model_id):
    tf.keras.backend.set_image_data_format('channels_last')

    network = resnet.ResNet(model_id=model_id)
    self.assertEqual(network.count_params(), 23561152)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    self.assertAllEqual(
        [1, 80, 80, 512], endpoints['3'].shape.as_list(), "failure on 3")
    self.assertAllEqual(
        [1,  40, 40, 1024], endpoints['4'].shape.as_list(), "failure on 4")
    self.assertAllEqual(
        [1, 20, 20, 2048], endpoints['5'].shape.as_list(), "failure on 5")


if __name__ == '__main__':
  tf.test.main()
