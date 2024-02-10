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
import pickle


class PanopticQualityMetricTest(tf.test.TestCase, parameterized.TestCase):
  # pylint: disable=line-too-long
  """
  # This module tests the PanopticQuality class and checks if it properly generates panoptic metrics.
  Procuedure for testing:
  1. From within the _postprocess_outputs() func in tasks/panoptic_maskformer.py, store the pq_metric_labels, pq_metric_inputs in .pickle files. This will be easier to load.
  2. You can also store the pq_config in a .pickle file in this format: {'a': pq_config} from the build_metrics() function in tasks/panoptic_maskformer.py.
  2. Load these .pickle files below.
  3. This test ensures all the correct shapes and types are returned.
  4. Originally, the input tensors will have original category IDs, but will be converted to contigious category IDs for metrics.
  """

  @parameterized.named_parameters(('test1',))
  def testMetricComputation(self):
    main_pth = '/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models/official/projects/maskformer/losses/test'

    # Load pq_config, saved using {'a': pq_config}
    with open(main_pth+'/pq_config.pickle', 'rb') as f:
      pq_config = pickle.load(f)['a']

    on_tpu = False

    # Load PQ metric labels and outputs
    with open(main_pth+'/pq_metric_inputs.pickle', 'rb') as f:
      pq_metric_inputs = pickle.load(f)
    with open(main_pth+'/pq_metric_labels.pickle', 'rb') as f:
      pq_metric_labels = pickle.load(f)

    pq_metric = PanopticQualityMetric(on_tpu, pq_config)

    # Assert that all the values within these dictionaries are of type tf.Tensor
    for key in pq_metric_inputs:
      self.assertIsInstance(pq_metric_inputs[key], tf.Tensor)

    # Simulate invoking the metric with predefined labels and outputs
    metrics = pq_metric(pq_metric_labels, pq_metric_inputs)

    expected_keys = [
        'panoptic_quality/All_num_categories', 'panoptic_quality/Things_num_categories', 'panoptic_quality/Stuff_num_categories',
        'panoptic_quality/All_pq', 'panoptic_quality/All_rq', 'panoptic_quality/All_sq',
        'panoptic_quality/Things_pq', 'panoptic_quality/Things_rq', 'panoptic_quality/Things_sq',
        'panoptic_quality/Stuff_pq', 'panoptic_quality/Stuff_rq', 'panoptic_quality/Stuff_sq',
    ]
    for key in expected_keys:
      self.assertIn(key, metrics, msg=f"Key {key} not found in metrics.")

    # Assert that all of the keys are present from 1 to pq_config.num_categories in the specified formats
    metric_formats = [
        'panoptic_quality/pq/class_{}', 'panoptic_quality/rq/class_{}', 'panoptic_quality/sq/class_{}'
    ]
    for category_id in range(1, pq_config.num_categories + 1):
      for metric_format in metric_formats:
        key = metric_format.format(category_id)
        self.assertIn(
            key, metrics, msg=f"Class-specific metric key {key} not found in metrics.")


if __name__ == '__main__':
  tf.test.main()
