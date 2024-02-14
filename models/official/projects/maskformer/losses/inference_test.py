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

"""Tests for PanopticInference class."""

import tensorflow as tf
from absl.testing import parameterized
import numpy as np
from official.projects.maskformer.losses.inference import PanopticInference

class PanopticInferenceTest(tf.test.TestCase, parameterized.TestCase):
    """
    This module tests the PanopticInference class and checks if it properly returns the instance and category masks.
    """
    @parameterized.named_parameters(('test1',))
    def testInferenceShapes(self):
        image_shape = [2, 640, 640]  
        num_instances = 100  
        num_classes = 134  
        batch_size = image_shape[0]
        height, width = image_shape[1:]

        pred_labels_load = tf.random.uniform(
            [batch_size, num_instances, num_classes], minval=-20, maxval=20, dtype=tf.float32
        )
        pred_masks_load = tf.random.uniform(
            [batch_size, 160, 160, num_instances], minval=-20, maxval=20, dtype=tf.float32
        )

        panoptic_inference = PanopticInference(
            num_classes=num_classes - 1,  # Assuming num_classes includes background
            background_class_id=0
        )

        instance_masks, category_masks = panoptic_inference(
            pred_labels_load, pred_masks_load, image_shape[1:]
        )

        # Assertions to ensure outputs have correct shapes
        self.assertEqual(instance_masks.shape[1:], (height, width),
                         msg="Instance mask shape does not match expected shape.")
        self.assertEqual(category_masks.shape[1:], (height, width),
                         msg="Category mask shape does not match expected shape.")

        # Ensure the batch size of the output matches the input logits/masks batch size
        self.assertEqual(instance_masks.shape[0], pred_labels_load.shape[0],
                         msg="Instance mask batch size does not match input batch size.")
        self.assertEqual(category_masks.shape[0], pred_masks_load.shape[0],
                         msg="Category mask batch size does not match input batch size.")

        # Note: The shapes will not match if the tensors do not cross a threshold within the PanopticInference call.
        # An error will be raised during time of 'eval' in tasks/panoptic_maskformer.py if this happens. 
        # This cannot happen unless 'eval' is called with a model that has not been trained.

if __name__ == '__main__':
    tf.test.main()