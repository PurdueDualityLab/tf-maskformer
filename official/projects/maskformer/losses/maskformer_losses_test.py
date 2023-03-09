from official.projects.maskformer.losses.maskformer_losses import Loss
from research.object_detection.matchers.hungarian_matcher import HungarianBipartiteMatcher
from research.object_detection.core.region_similarity_calculator import DETRSimilarity
from absl.testing import parameterized
import tensorflow as tf
import torch

import pickle

class LossTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        similarity_calc = DETRSimilarity()
        matcher = HungarianBipartiteMatcher()
        mask_weight = 20.0
        dice_weight = 1.0
        no_object_weight = 0.1
        weight_dict = {"loss_ce":1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        losses = ["labels", "masks"]

        loss = Loss(
            num_classes = 171,
            similarity_calc = similarity_calc,
            matcher = matcher,
            weight_dict = weight_dict,
            eos_coef = no_object_weight,
            losses = losses
        )
        
        with open("losses_test.pkl", "rb") as f:
            params = pickle.load(f)
        
        print(loss.call(params["outputs"], params["targets"]))

if __name__ == '__main__':
    tf.test.main()