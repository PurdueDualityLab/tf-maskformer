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
	# pylint: disable=line-too-long
  """
  # This module tests the PanopticInference class and checks if it properly returns the instance and category masks.
  Procuedure for testing: 
  1. From within the _postprocess_outputs() func in tasks/panoptic_maskformer.py, store the pred_logits, pred_masks, image_shape in .npy files.
  2. Load these .npy files below. 
  3. This test ensures all the correct shapes and types are returned.
  4. Originally, the input tensors will have continuous category IDs, but the output masks will have original category IDs. This will be changed within ./quality.py
  """
  @parameterized.named_parameters(('test1',))
  def testInferenceShapes(self):
    main_pth = '/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models/official/projects/maskformer/losses/test'
    image_shape = [3, 640, 640]

    # Loading test data
    pred_logits_load = tf.convert_to_tensor(
        np.load(main_pth + "/pred_labels.npy"))
    pred_masks_load = tf.convert_to_tensor(
        np.load(main_pth + "/pred_binary_masks.npy"))

    # Initialize PanopticInference with appropriate configurations
    panoptic_inference = PanopticInference(
        num_classes=133,
        background_class_id=0
    )

    # Invoke the PanopticInference call with loaded data
    instance_masks, category_masks = panoptic_inference(
        pred_logits_load, pred_masks_load, image_shape[1:])

    # Assertions to ensure outputs have correct shapes
    self.assertEqual(instance_masks.shape[1:], (image_shape[1], image_shape[2]),
                     msg="Instance mask shape does not match expected shape.")
    self.assertEqual(category_masks.shape[1:], (image_shape[1], image_shape[2]),
                     msg="Category mask shape does not match expected shape.")

    # Ensure the batch size of the output matches the input logits/masks batch size
    self.assertEqual(instance_masks.shape[0], pred_logits_load.shape[0],
                     msg="Instance mask batch size does not match input batch size.")  # pylint: disable=line-too-long
    self.assertEqual(category_masks.shape[0], pred_masks_load.shape[0],
                     msg="Category mask batch size does not match input batch size.")  # pylint: disable=line-too-long

    # Ensure masks are integer types
    self.assertEqual(instance_masks.dtype, tf.int32,
                     msg="Instance masks are not of type tf.int32.")
    self.assertEqual(category_masks.dtype, tf.int32,
                     msg="Category masks are not of type tf.int32.")


if __name__ == '__main__':
  tf.test.main()
