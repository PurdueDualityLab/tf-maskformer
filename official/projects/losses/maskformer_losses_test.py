from official.projects.losses.maskformer_losses import Loss
from research.object_detection.matchers.hungarian_matcher import HungarianBipartiteMatcher
from absl.testing import parameterized
import tensorflow as tf

import pickle

class FocalLossTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        matcher = HungarianBipartiteMatcher()
        mask_weight = 20.0
        dice_weight = 1.0
        no_object_weight = 0.1
        weight_dict = {"loss_ce":1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        losses = ["labels", "masks"]

        loss = Loss(
            num_classes = 171,
            matcher = matcher,
            weight_dict = weight_dict,
            eos_coef = no_object_weight,
            losses = losses,
        )
        
        params = pickle.load('params.pickle')
        print(params)

        # self.assertAllEqual(
        #     output["class_prob_predictions"].shape.as_list(), expected_class_probs_shape)
        # self.assertAllEqual(
        #     output["mask_prob_predictions"].shape.as_list(), expected_mask_probs_shape)


if __name__ == '__main__':
    tf.test.main()