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

"""Tests for PanopticQualityMetric class."""

import tensorflow as tf
from absl.testing import parameterized
from official.projects.maskformer.losses.quality import PanopticQualityMetric
from official.projects.maskformer.configs.maskformer import PanopticQuality


class PanopticQualityMetricTest(tf.test.TestCase, parameterized.TestCase):
  # pylint: disable=line-too-long
  """
  This module tests the PanopticQuality class and checks if it properly generates panoptic metrics.
  """
  @parameterized.named_parameters(('test1',))
  def testMetricComputation(self):
    pq_config = PanopticQuality()

    batch_size, height, width = 2, 640, 640
    num_categories = pq_config.num_categories

    # (IDs are converted from contigious (0-133) to original (0-199) within PanopticInference)
    # (We pass them in as is into PanopticQualityMetric, which will handle the conversion)
    # Normally, for category_masks, max_val should be 199.0, but for testing purposes, we set it to 1.0
    # And normally, for instance_masks, max_val should be 100, but for testing purposes, we set it to 10

    # pylint: disable=line-too-long
    pq_metric_inputs = {
        'category_mask': tf.random.uniform(shape=[batch_size, height, width], minval=0.0, maxval=1.0, dtype=tf.float32),
        'instance_mask': tf.random.uniform(shape=[batch_size, height, width], minval=0, maxval=10, dtype=tf.int32)
    }
    # pylint: disable=line-too-long
    pq_metric_labels = {
        'category_mask': tf.random.uniform(shape=[batch_size, height, width, 1], minval=0.0, maxval=1.0, dtype=tf.float32),
        'instance_mask': tf.random.uniform(shape=[batch_size, height, width, 1], minval=0, maxval=10, dtype=tf.int32)
    }

    on_tpu = False
    pq_metric = PanopticQualityMetric(on_tpu, pq_config)

    for key in pq_metric_inputs:
      self.assertIsInstance(pq_metric_inputs[key], tf.Tensor)

    metrics = pq_metric(pq_metric_labels, pq_metric_inputs)

    # Since PanopticQualityMetric instantiates PanopticQualityV2 or PanopticQuality,
    # we only need to check if the expected keys are present in the metrics dictionary

    # pylint: disable=line-too-long
    expected_keys = [
        'panoptic_quality/All_num_categories', 'panoptic_quality/Things_num_categories', 'panoptic_quality/Stuff_num_categories',
        'panoptic_quality/All_pq', 'panoptic_quality/All_rq', 'panoptic_quality/All_sq',
        'panoptic_quality/Things_pq', 'panoptic_quality/Things_rq', 'panoptic_quality/Things_sq',
        'panoptic_quality/Stuff_pq', 'panoptic_quality/Stuff_rq', 'panoptic_quality/Stuff_sq',
    ]
    for key in expected_keys:
      self.assertIn(key, metrics, msg=f"Key {key} not found in metrics.")

    # pylint: disable=line-too-long
    metric_formats = [
        'panoptic_quality/pq/class_{}', 'panoptic_quality/rq/class_{}', 'panoptic_quality/sq/class_{}'
    ]
    for category_id in range(1, num_categories + 1):
      for metric_format in metric_formats:
        key = metric_format.format(category_id)
        self.assertIn(
            key, metrics, msg=f"Class-specific metric key {key} not found in metrics.")


if __name__ == '__main__':
  tf.test.main()
