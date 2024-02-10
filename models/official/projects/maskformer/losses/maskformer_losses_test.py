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

"""Tests for tensorflow_models.official.projects.maskformer.losses.maskformer_losses."""

from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.detr.ops.matchers import hungarian_matching
from absl.testing import parameterized
import tensorflow as tf
from official.vision.ops import preprocess_ops
import numpy as np
import pickle


class LossTest(tf.test.TestCase, parameterized.TestCase):
  # pylint: disable=line-too-long
  """
  # This module tests the Loss class and checks if it properly computes the losses.
  Procuedure for testing:
  1. From within the _postprocess_outputs() func in tasks/panoptic_maskformer.py, store the outputs and targets in .npy files.
  2. Load these .npy files below.
  3. This test will ensure that all the computed losses match the loaded losses.
  4. You can test this using outputs from the PyTorch MaskFormer model as well. But, ensure to comment out line 325 in __call__ in /maskformer_losses.py when doing so. 
  """
  @parameterized.named_parameters(('test1',))
  def test_pass_through(self):
    matcher = hungarian_matching
    main_pth = "/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models/official/projects/maskformer/losses/test"

    # Initialize Loss
    loss = Loss(
        num_classes=133,
        matcher=matcher,
        eos_coef=0.1,
        cost_class=1.0,
        cost_dice=1.0,
        cost_focal=20.0,
        ignore_label=133
    )

    # Load test data
    loaded_losses = pickle.load(open(main_pth+"/losses.pkl", "rb"))
    aux_outputs = [
        {"pred_logits": tf.convert_to_tensor(np.load(main_pth+f"/tensors/aux_outputs_pred_logits{i}.npy")),
         "pred_masks": tf.convert_to_tensor(np.load(main_pth+f"/tensors/aux_outputs_pred_masks{i}.npy"))}
        for i in range(5)
    ]
    pred_logits_load = tf.convert_to_tensor(
        np.load(main_pth+"/tensors/output_pred_logits.npy"))
    pred_masks_load = tf.transpose(tf.convert_to_tensor(
        np.load(main_pth+"/tensors/output_pred_masks.npy")), [0, 2, 3, 1])

    outputs = {
        "pred_logits": pred_logits_load,
        "pred_masks": pred_masks_load,
        "aux_outputs": aux_outputs
    }

    # Prepare targets
    targets = {}
    target_labels_1 = tf.convert_to_tensor(
        np.load(main_pth+'/tensors/targets_labels_0.npy'))
    target_labels_2 = tf.convert_to_tensor(
        np.load(main_pth+'/tensors/targets_labels_1.npy'))
    target_labels_1_padded = preprocess_ops.clip_or_pad_to_fixed_size(
        target_labels_1, 100, 133)
    target_labels_2_padded = preprocess_ops.clip_or_pad_to_fixed_size(
        target_labels_2, 100, 133)
    target_labels_stacked = tf.stack(
        [target_labels_1_padded, target_labels_2_padded], axis=0)

    target_masks_1 = tf.transpose(tf.convert_to_tensor(
        np.load(main_pth+'/tensors/targets_masks_0.npy')), [1, 2, 0])
    target_masks_2 = tf.transpose(tf.convert_to_tensor(
        np.load(main_pth+'/tensors/targets_masks_1.npy')), [1, 2, 0])
    target_masks_1 = tf.image.resize(tf.cast(target_masks_1, float), (640, 640))
    target_masks_2 = tf.image.resize(tf.cast(target_masks_2, float), (640, 640))
    target_masks_1_padded = preprocess_ops.clip_or_pad_to_fixed_size(
        tf.transpose(target_masks_1, [2, 0, 1]), 100, 133)
    target_masks_2_padded = preprocess_ops.clip_or_pad_to_fixed_size(
        tf.transpose(target_masks_2, [2, 0, 1]), 100, 133)
    targets['unique_ids'] = target_labels_stacked
    targets["individual_masks"] = tf.expand_dims(
        tf.stack([target_masks_1_padded, target_masks_2_padded], axis=0), -1)

    # Compute losses
    losses = loss(outputs, targets)

    # Output losses for inspection
    print(
        f"Classification loss : {losses['loss_ce'].numpy()} Dice Loss : {losses['loss_dice'].numpy()} Focal Loss : {losses['loss_focal'].numpy()}")

    for i in range(5):
      print(f'loss_ce_{str(i)}: ', losses[f'loss_ce_{str(i)}'])
      print(f'loss_focal_{str(i)}: ', losses[f'loss_focal_{str(i)}'])
      print(f'loss_dice_{str(i)}: ', losses[f'loss_dice_{str(i)}'])

    print("[INFO] Total Loss:", losses['loss_ce'].numpy() +
          losses['loss_dice'].numpy() + losses['loss_focal'].numpy())

    # Assert computed losses match loaded losses
    self.assertAllEqual(losses, loaded_losses)


if __name__ == '__main__':
  tf.test.main()
