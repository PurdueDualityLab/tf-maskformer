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

"""Tests for MaskFormer model."""

from official.projects.maskformer.modeling.maskformer import MaskFormer
from official.projects.maskformer.modeling.backbone.backbone import build_maskformer_backbone
from absl.testing import parameterized
import tensorflow as tf


class MaskFormerTest(tf.test.TestCase, parameterized.TestCase):
  # pylint: disable=line-too-long
  """
  # This module tests whether the MaskFormer class works (output shapes, sem_seg_head outputs) as expected. 
  """
  @parameterized.named_parameters(
      ('test1', 256, 100, 256, '5', 6, 0, 6, 199, 1, False, 'fpn', False))
  # pylint: disable=line-too-long
  def test_mask_former_pass_through(self, fpn_feat_dims, num_queries, hidden_size,
                                    backbone_endpoint_name, fpn_encoder_layers, detr_encoder_layers,
                                    num_decoder_layers, num_classes, batch_size,
                                    bfloat16, which_pixel_decoder, deep_supervision):
    input_shape = (640, 640, 3)
    input_specs = tf.keras.layers.InputSpec(shape=[None] + list(input_shape))
    backbone = build_maskformer_backbone(model_id=50)

    # pylint: disable=line-too-long
    maskformer = MaskFormer(backbone=backbone, input_specs=input_specs,
                            fpn_feat_dims=fpn_feat_dims, num_queries=num_queries,
                            hidden_size=hidden_size, fpn_encoder_layers=fpn_encoder_layers,
                            detr_encoder_layers=detr_encoder_layers, num_decoder_layers=num_decoder_layers,
                            num_classes=num_classes, batch_size=batch_size, bfloat16=bfloat16,
                            which_pixel_decoder=which_pixel_decoder, deep_supervision=deep_supervision)

    input_image = tf.ones((batch_size, *input_shape))

    expected_output_shapes = {
        "class_prob_predictions": (batch_size, num_queries, num_classes + 1),
        "mask_prob_predictions": (batch_size, 160, 160, num_queries),
    }

    output = maskformer(input_image)

    for key, expected_output_shape in expected_output_shapes.items():
      self.assertAllEqual(output[key].shape, expected_output_shape)


if __name__ == '__main__':
  tf.test.main()
