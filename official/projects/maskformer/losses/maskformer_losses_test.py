from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.detr.ops.matchers import hungarian_matching
from absl.testing import parameterized
import tensorflow as tf

import numpy as np

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
            weight_dict = weight_dict,
            eos_coef = no_object_weight,
            losses = losses,
            cost_class=1,
            cost_focal=mask_weight,
            cost_dice=dice_weight
        )
        
        outputs = {"pred_logits":tf.convert_to_tensor(np.load("output_pred_logits.npy")), "pred_masks":tf.convert_to_tensor(np.load("output_pred_masks.npy"))}
        targets_labels = tf.convert_to_tensor(np.load("targets_labels.npy"))
        targets_masks = tf.convert_to_tensor(np.load("targets_masks.npy"))
        targets = list()
        for i in range(tf.shape(targets_labels)[0]):
            targets.append({"labels":tf.expand_dims(targets_labels[i], axis=0), "masks":tf.expand_dims(targets_masks[i], axis=0)})
        print(loss.call(outputs, targets))

if __name__ == '__main__':
    tf.test.main()