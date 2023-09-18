from absl.testing import parameterized
import tensorflow as tf
import sys
sys.path.append("/home/isaacjaeminin/tf-maskformer/models")
from official.projects.maskformer.losses.inference import PanopticInference
from official.projects.maskformer.modeling.maskformer import MaskFormer
import numpy as np
from official.vision.ops import preprocess_ops

class PanopticInferenceTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        """
        # This test is supposed to give PQ stuff and PQ things metrics for fixed tensor inputs
        # Load pytorch output and targets for testing the PQ stuff and PQ things metrics
        Procduere for testing:
        1. Save the input image (without normalization), GT instance (will have non-contigious ids) and panoptic masks (will have non-contigious ids) from TF code
        2. With saved TF image as input obtain output individual masks from the PyTorch Model with final weights
        """
        background_class_id = 0
        
        main_pth = "/depot/qqiu/data/vishal/tf-maskformer/tensors_for_PQ_metric"

        # Load pytorch predictions
        image_shape = [3, 640, 640]
        pred_logits_load = tf.convert_to_tensor(np.load(main_pth+"/output_pred_logits.npy")) 
        pred_masks_load = tf.convert_to_tensor(np.load(main_pth+"/output_pred_masks.npy")) 
        pred_masks_load = tf.transpose(pred_masks_load, [0,2,3,1]) # (1,100, h, w) -> (1, h, w, 100) (reshaping according to TF model outputs)
        
        # Pytorch code uses 133 as backgorund class id and TF code uses 0 as background class id so we need to swap them
        
        # shift all classes by 1 and replace 133 with 0 (background class id)
        # Load the instance and category masks from TF code
        instance_mask_gt = tf.convert_to_tensor(np.load(main_pth+"/instance_mask.npy"))
        category_mask_gt = tf.convert_to_tensor(np.load(main_pth+"/category_mask.npy"))

        outputs = {
            "class_prob_predictions": pred_logits_load,
            "mask_prob_predictions": pred_masks_load,
        }
        
        inference = PanopticInference(num_classes=134, background_class_id=background_class_id, object_mask_threshold=0.25, class_score_threshold=0.25)
        instance_mask_predicted, category_mask_predicted = inference(outputs["class_prob_predictions"], 
                                                                        outputs["mask_prob_predictions"],
                                                                       image_shape)
        # Save the instance and category masks from TF code
        np.save(main_pth+"/instance_mask_predicted.npy", instance_mask_predicted.numpy())
        np.save(main_pth+"/category_mask_predicted.npy", category_mask_predicted.numpy())
        print("instance_mask_predicted", instance_mask_predicted.shape)
        print("category_mask_predicted", category_mask_predicted.shape)
        exit()
        assert instance_mask_predicted == instance_mask_gt
        assert category_mask_predicted == category_mask_gt

if __name__ == '__main__':
    tf.test.main()
