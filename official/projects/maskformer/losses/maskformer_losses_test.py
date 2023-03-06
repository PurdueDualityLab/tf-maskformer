from official.projects.maskformer.losses.maskformer_losses import Loss
from research.object_detection.matchers.hungarian_matcher import HungarianBipartiteMatcher
from absl.testing import parameterized
import tensorflow as tf
import torch

import pickle

class LossTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        matcher = HungarianBipartiteMatcher()
        mask_weight = 20.0
        dice_weight = 1.0
        no_object_weight = 0.1
        weight_dict = {"loss_ce":1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        losses = ["labels", "masks"]

        da_loss = Loss(
            num_classes = 171,
            matcher = matcher,
            weight_dict = weight_dict,
            eos_coef = no_object_weight,
            losses = losses
        )
        
        with open("params.pickle", "rb") as f:
            params = pickle.load(f)
        
        print(loss.call(params["outputs"], params["targets"]))
        # li = ["outputs", "targets"]
        # # for l in li:
        # #     for key in params[l]:
        # #         print(key, torch.tensor(params[l][key]).shape)
        
        # for i, dic in enumerate(params["targets"]):
        #     print(i)
        #     print("labels", dic["labels"].shape)
        #     print("targets", dic["masks"].shape)
        # self.assertAllEqual(
        #     output["class_prob_predictions"].shape.as_list(), expected_class_probs_shape)
        # self.assertAllEqual(
        #     output["mask_prob_predictions"].shape.as_list(), expected_mask_probs_shape)


if __name__ == '__main__':
    tf.test.main()