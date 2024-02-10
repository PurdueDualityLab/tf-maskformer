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

"""Tests for MaskFormer Transformer."""

from absl.testing import parameterized
import tensorflow as tf

from official.projects.maskformer.modeling.decoder.transformer_decoder import MaskFormerTransformer


class MaskFormerTransformerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('test1', '5', 8, 100, 256, 10,))
  def test_pass_through(self,
                        backbone_endpoint_name,
                        batch_size,
                        num_queries,
                        hidden_size,
                        num_classes):

    multilevel_features = {
        "2": tf.ones([1, 160, 160, 256]),
        "3": tf.ones([1, 80, 80, 512]),
        "4": tf.ones([1, 40, 40, 1024]),
        "5": tf.ones([1, 20, 20, 2048])
    }

    transformer = MaskFormerTransformer(backbone_endpoint_name=backbone_endpoint_name, # pylint: disable=line-too-long
                                        batch_size=batch_size,
                                        num_queries=num_queries,
                                        hidden_size=hidden_size,
                                        num_classes=num_classes,
                                        num_encoder_layers=0,
                                        num_decoder_layers=6,
                                        dropout_rate=0.1)

    input_image = tf.ones((1, 640, 640, 3))
    expected_output_shape = [6, 8, 100, 256]

    output = transformer(
        {"image": input_image, "features": multilevel_features})
    output_shape = [len(output)] + output[0].shape.as_list()

    self.assertAllEqual(output_shape, expected_output_shape)


if __name__ == '__main__':
  tf.test.main()
