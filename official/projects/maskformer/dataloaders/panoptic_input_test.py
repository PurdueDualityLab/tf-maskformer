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

"""Tests for MaskFormer data preparation and loading."""

import os
import numpy as np
import tensorflow as tf
from official.projects.maskformer.dataloaders.panoptic_input import TfExampleDecoder


class DataLoaderTest(tf.test.TestCase):
  # pylint: disable=line-too-long
  """ 
  # This module tests whether the DataLoader class works (output shapes, forward pass) as expected.
  """
  def setUp(self):
    super(DataLoaderTest, self).setUp()
    self.tfrecord_path = "/depot/davisjam/data/akshath/datasets"

  def test_dataloader(self):
    file_paths = tf.io.gfile.glob(
        os.path.join(self.tfrecord_path, "*.tfrecord"))
    decoder = TfExampleDecoder()
    image_count = 0
    all_class_stats = [0 for _ in range(134)]

    for each_file in file_paths:
      raw_dataset = tf.data.TFRecordDataset(
          os.path.join(self.tfrecord_path, each_file))
      print("Reading file:", each_file)
      for raw_record in raw_dataset.take(-1):
        data = decoder.decode(raw_record)
        image = data['image']
        contiguous_mask = tf.cast(
            data['groundtruth_panoptic_contiguous_mask'][:, :, 0], dtype=tf.float32) # pylint: disable=line-too-long
        instance_mask = tf.cast(
            data['groundtruth_panoptic_instance_mask'][:, :, 0], dtype=tf.float32) # pylint: disable=line-too-long
        category_mask = tf.cast(
            data['groundtruth_panoptic_category_mask'][:, :, 0], dtype=tf.float32) # pylint: disable=line-too-long
        class_ids = tf.cast(tf.sparse.to_dense(
            data['groundtruth_panoptic_class_ids'], default_value=255), dtype=tf.int32) # pylint: disable=line-too-long

        h, w, c = image.shape

        # Assertions
        self.assertAllEqual(image.shape, (h, w, c))
        self.assertAllEqual(contiguous_mask.shape, (h, w))
        self.assertAllEqual(instance_mask.shape, (h, w))
        self.assertAllEqual(category_mask.shape, (h, w))
        self.assertLessEqual(tf.reduce_max(image), 255)
        self.assertGreaterEqual(tf.reduce_min(image), 0)
        self.assertLessEqual(tf.reduce_max(contiguous_mask), 133)
        self.assertGreaterEqual(tf.reduce_min(contiguous_mask), 0)

        # Compare with background class
        unique_class_ids = np.unique(contiguous_mask.numpy())
        unique_class_ids = unique_class_ids[unique_class_ids != 0]

        for each_class_id in unique_class_ids:
          all_class_stats[int(each_class_id)] += 1

        self.assertEqual(len(unique_class_ids),
                         len(np.unique(class_ids.numpy())))

        image_count += 1

    print("Total images:", image_count)
    print("All class stats:", all_class_stats)


if __name__ == '__main__':
  tf.test.main()
