from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.detr.ops.matchers import hungarian_matching
from absl.testing import parameterized
import tensorflow as tf

import numpy as np
from loguru import logger
import pickle

class LossTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        matcher = hungarian_matching
        mask_weight = 20.0
        dice_weight = 1.0
        no_object_weight = 0.1
        weight_dict = {"loss_ce":1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        losses = ["labels", "masks"]

        loss = Loss(
            num_classes = 133,
            matcher = matcher,
            # weight_dict = weight_dict,
            eos_coef = no_object_weight,
            losses = losses
        )
        
        # outputs = {"pred_logits":tf.convert_to_tensor(np.load("output_pred_logits.npy")), "pred_masks":tf.convert_to_tensor(np.load("output_pred_masks.npy"))}
        # print(f"outputs['pred_logits'] shape is {outputs['pred_logits'].shape}")
        # print(f"outputs['pred_masks'] shape is {outputs['pred_masks'].shape}")
        loaded_outputs = np.load("outputs.npy", allow_pickle=True).item()

        outputs = {
            "pred_logits": tf.convert_to_tensor(loaded_outputs["pred_logits"]),
            "pred_masks": tf.convert_to_tensor(loaded_outputs["pred_masks"]),
            "aux_outputs": [
                {key: tf.convert_to_tensor(value) for key, value in aux_output.items()}
                for aux_output in loaded_outputs["aux_outputs"]
            ]
        }

        # Load the new_targets_dict NumPy array
        loaded_targets_dict = np.load("new_targets_dict.npy", allow_pickle=True).item()

        # Convert the NumPy arrays to TensorFlow tensors and recreate the new_targets list
        targets = [
            {
                "labels": tf.convert_to_tensor(loaded_targets_dict[idx]["labels"]),
                "masks": tf.convert_to_tensor(loaded_targets_dict[idx]["masks"]),
            }
            for idx in loaded_targets_dict
        ]

        # logger.debug(f"LOADED TARGET LABELS: {targets[0]['labels'].shape}")
        # logger.debug(f"LOADED TARGET MASKS: {targets[0]['masks'].shape}")
        # logger.debug(f"outputs is {outputs}")
        losses = loss(outputs, targets)
        logger.critical(losses)
        
        for k in list(losses.keys()):
            if k in weight_dict:
                print(f"Loss shapes {k} - {losses[k].shape}, Loss value {k}- {losses[k]}")
                losses[k] *= weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        # TODO: Check if this is correct
        # self.assertAllEqual(losses, )

if __name__ == '__main__':
    tf.test.main()