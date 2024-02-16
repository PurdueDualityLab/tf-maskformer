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

"""Panoptic Quality Module."""

import tensorflow as tf
from typing import Any, Dict
from official.vision.evaluation import panoptic_quality
from official.projects.maskformer.losses.mapper import _get_contiguous_to_original

class PanopticQualityMetric:
  """Panoptic Quality metric class."""

  def __init__(self, on_tpu: bool, pq_config):
    # pylint: disable=line-too-long
    """Initialize.
    Args:
      on_tpu: `bool`, Whether the model is running on TPU.
      pq_config: `maskformer.configs.maskformer.PanopticQuality`, The configuration for the PQ metric.
    Returns: 
      None
    """

    self.cat_id_map, _, _ = _get_contiguous_to_original()
    self.is_thing_dict_bool = pq_config.is_thing
    self.on_tpu = on_tpu
    self.num_categories = pq_config.num_categories

    if self.on_tpu:
      self.metric = panoptic_quality.PanopticQualityV2(
          num_categories=pq_config.num_categories + 1,
          is_thing=self.is_thing_dict_bool,
          ignored_label=pq_config.ignored_label,
          max_num_instances=pq_config.max_num_instances,
          rescale_predictions=pq_config.rescale_predictions,
      )
    else:
      self.metric = panoptic_quality.PanopticQuality(
          num_categories=pq_config.num_categories + 1,
          ignored_label=pq_config.ignored_label,
          max_instances_per_category=pq_config.max_num_instances,
          offset=pq_config.max_num_instances**3,
      )

  def __call__(self, pq_metric_labels: Dict[str, Any], pq_metric_inputs: Dict[str, Any]): # pylint: disable=line-too-long
     # pylint: disable=line-too-long
    """Generates panoptic metrics.
    PanopticQuality and PanopticQualityV2 require category IDs to be contiguous. 
    Args:
      pq_metric_labels: a dictionary of panoptic segmentation labels.
      pq_metric_inputs: a dictionary of panoptic inference inputs.
    Returns:
      A dictionary of panoptic metrics.
    """

    # [bsize, h, w, 1] -> [bsize, h, w]
    pq_metric_labels = {key: tf.squeeze(value, axis=-1)
                        for key, value in pq_metric_labels.items()}

    # There are different PQ implementations for TPU and GPU/CPU
    # TPU implementation requires a lot of memory bandwidth
    if self.on_tpu:
      self.metric.update_state(
          pq_metric_labels, pq_metric_inputs
      )
      results = self.metric.result()
    else:
      self.metric.compare_and_accumulate(
        {
          key: value.numpy() for key,
          value in pq_metric_labels.items()
        }, 
        {
          key: value.numpy() for key,
          value in pq_metric_inputs.items()
        }
      )
      results = self.metric.result(self.is_thing_dict_bool) # pylint: disable=too-many-function-args

    return self._reduce_aggregated_results(results)

  def _reduce_aggregated_results(self, aggregated_results: Dict[str, Any]):
    reduced_metrics = self._reduce_metrics(aggregated_results)

    # Reset the metric state, avoids direct aggregation of results from different steps
    if self.on_tpu:
      self.metric.reset_state()
    else:
      self.metric.reset()

    return reduced_metrics

  def _reduce_metrics(self, results: Dict[str, Any]):
    # pylint: disable=line-too-long
    """
    Routes the results to the appropriate reduction function based on the compute platform (TPU or GPU/CPU).
    """

    if self.on_tpu:
      return self._reduce_panoptic_metrics_v2(results)
    else:
      return self._reduce_panoptic_metrics(results)

  def _reduce_panoptic_metrics(self, results: Dict[str, Any]):
    # pylint: disable=line-too-long
    """Updates the per class and mean panoptic metrics in the reduced_metrics. 
       CPU/GPU implementation.
    Args: 
      results: a dictionary of results.
    Returns:
      A dictionary of reduced metrics.
    """

    reduced_metrics = {}

    categories = ['All', 'Things', 'Stuff']
    for category in categories:
      key = f'panoptic_quality/{category}_num_categories'
      reduced_metrics[key] = results[f'{category}_num_categories']

    categories = ['All', 'Things', 'Stuff']
    metrics = ['pq', 'rq', 'sq']
    for category in categories:
      for metric in metrics:
        key = f'panoptic_quality/{category}_{metric}'
        reduced_metrics[key] = results[f'{category}_{metric}']

    metrics = ['pq', 'rq', 'sq']
    for metric in metrics:
      for i in range(
              1,
              self.num_categories + 1):
        key = f'panoptic_quality/{metric}/class_{self.convert_contiguous_to_original(i)}'
        reduced_metrics[key] = results[f'{metric}_per_class'][i - 1]

    return reduced_metrics

  def _reduce_panoptic_metrics_v2(self, results: Dict[str, Any]):
    # pylint: disable=line-too-long
    """Updates the per class and mean panoptic metrics in the reduced_metrics.
       TPU implementation.
    Args: 
      results: a dictionary of results.
    Returns:
      A dictionary of reduced metrics.
    """

    reduced_metrics = {}

    valid_thing_classes = results['valid_thing_classes']
    valid_stuff_classes = results['valid_stuff_classes']
    valid_classes = valid_stuff_classes | valid_thing_classes
    num_categories = tf.math.count_nonzero(valid_classes, dtype=tf.float32)
    num_thing_categories = tf.math.count_nonzero(
        valid_thing_classes, dtype=tf.float32
    )
    num_stuff_categories = tf.math.count_nonzero(
        valid_stuff_classes, dtype=tf.float32
    )
    valid_thing_classes = tf.cast(valid_thing_classes, dtype=tf.float32)
    valid_stuff_classes = tf.cast(valid_stuff_classes, dtype=tf.float32)

    reduced_metrics['panoptic_quality/All_num_categories'] = num_categories
    reduced_metrics['panoptic_quality/Things_num_categories'] = num_thing_categories
    reduced_metrics['panoptic_quality/Stuff_num_categories'] = num_stuff_categories

    for metric in ['pq', 'sq', 'rq']:
      metric_per_class = results[f'{metric}_per_class']
      reduced_metrics[f'panoptic_quality/All_{metric}'] = tf.math.divide_no_nan(
          tf.reduce_sum(metric_per_class), num_categories)
      reduced_metrics[f'panoptic_quality/Things_{metric}'] = tf.math.divide_no_nan(
          tf.reduce_sum(metric_per_class * valid_thing_classes),
          num_thing_categories,
      )
      reduced_metrics[f'panoptic_quality/Stuff_{metric}'] = tf.math.divide_no_nan(
          tf.reduce_sum(metric_per_class * valid_stuff_classes),
          num_stuff_categories,
      )
      if self.task_config.panoptic_quality_evaluator.report_per_class_metrics:
        for i, is_valid in enumerate(valid_classes.numpy()):
          if is_valid:
            reduced_metrics[f'panoptic_quality/{metric}/class_{self.convert_contiguous_to_original(i)}'] = metric_per_class[i]

    return reduced_metrics

  def convert_contiguous_to_original(self, i: int):
    # pylint: disable=line-too-long
    """
    Converts single ID to corresponding original ID.
    """
    return int(self.cat_id_map.lookup(tf.cast(i, tf.int32)))