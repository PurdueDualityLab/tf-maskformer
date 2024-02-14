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
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        matcher = hungarian_matching
        loss = Loss(
            num_classes=133,
            matcher=matcher,
            eos_coef=0.1,
            cost_class=1.0,
            cost_dice=1.0,
            cost_focal=20.0,
            ignore_label=133
        )

        batch_size = 2
        num_queries = 100 
        height, width = 640, 640
        num_classes = 133 
        outputs = {
            "pred_logits": tf.random.uniform((batch_size, num_queries, num_classes), dtype=tf.float32),
            "pred_masks": tf.random.uniform((batch_size, height, width, num_queries), dtype=tf.float32),
            "aux_outputs": [
                {
                    "pred_logits": tf.random.uniform((batch_size, num_queries, num_classes), dtype=tf.float32),
                    "pred_masks": tf.random.uniform((batch_size, height, width, num_queries), dtype=tf.float32)
                } for _ in range(5)
            ]
        }

        targets = {
            'unique_ids': tf.random.uniform((batch_size, num_queries), minval=0, maxval=num_classes, dtype=tf.int32),
            "individual_masks": tf.random.uniform((batch_size, num_queries, height, width, 1), dtype=tf.float32)
        }

        # Compute losses
        losses = loss(outputs, targets)

        print(f"[INFO] Classification loss: {losses['loss_ce'].numpy()}, Dice Loss: {losses['loss_dice'].numpy()}, Focal Loss: {losses['loss_focal'].numpy()}")

        for i in range(5):
            print(f"loss_ce_{str(i)}: ", losses[f"loss_ce_{str(i)}"])
            print(f"loss_focal_{str(i)}: ", losses[f"loss_focal_{str(i)}"])
            print(f"loss_dice_{str(i)}: ", losses[f"loss_dice_{str(i)}"])

        total_loss = losses['loss_ce'].numpy() + losses['loss_dice'].numpy() + losses['loss_focal'].numpy()
        print("[INFO] Total Loss (w/o deep_supervision):", total_loss)
        
        total_loss_deep_supervision = total_loss
        for i in range(5):
            total_loss_deep_supervision += losses[f"loss_ce_{str(i)}"].numpy() + losses[f"loss_focal_{str(i)}"].numpy() + losses[f"loss_dice_{str(i)}"].numpy()
        print("[INFO] Total Loss (w/ deep_supervision):", total_loss_deep_supervision)

        self.assertGreater(total_loss_deep_supervision, total_loss) 

if __name__ == '__main__':
    tf.test.main()