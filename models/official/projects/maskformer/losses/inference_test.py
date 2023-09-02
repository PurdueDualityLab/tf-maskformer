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
        background_class_id = 0
        # This test is supposed to give PQ stuff and PQ things metrics for fixed tensor inputs
        # Load pytorch output and targets for testing the PQ stuff and PQ things metrics
        main_pth = "/home/vishalpurohit55595/tf-maskformer/models/official/projects/maskformer/losses"
        
        # Load input image normalized
        input_image =  tf.convert_to_tensor(np.load(main_pth+"/tensors/images.npy"))
        # B, C, H, W -> B, H, W, C
        input_image = tf.transpose(input_image, [0,2,3,1])
        # Load pytorch predictions
        pred_logits_load = tf.convert_to_tensor(np.load(main_pth+"/tensors/output_pred_logits.npy")) 
        pred_masks_load = tf.convert_to_tensor(np.load(main_pth+"/tensors/output_pred_masks.npy"))
        pred_masks_load = tf.transpose(pred_masks_load, [0,2,3,1]) # (1,100, h, w) -> (1, h, w, 100) (actual outptu of our model)
        
        outputs = {
            "class_prob_predictions": pred_logits_load,
            "mask_prob_predictions": pred_masks_load,
        }
        
        # Load targets 
        # targets = {}
        # target_labels_1 = tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_labels_0.npy'))
        # target_labels_2 = tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_labels_1.npy'))

        # target_labels_1_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_labels_1, 100,background_class_id)
        # target_labels_2_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_labels_2, 100,background_class_id)
        
        # target_labels_stacked = tf.stack([target_labels_1_padded, target_labels_2_padded], axis=0) # Stacking the two tensors along the batch dimension
       
        # target_masks_1 = tf.transpose(tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_0.npy')), [1,2,0])
        # target_masks_2 = tf.transpose(tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_1.npy')), [1,2,0])

        inference = PanopticInference(num_classes=134, background_class_id=background_class_id, object_mask_threshold=0.85)
        instance_mask_predicted, category_mask_predicted = inference(outputs["class_prob_predictions"], 
                                                                        outputs["mask_prob_predictions"],
                                                                        input_image.shape)
        print("instance_mask_predicted.shape", instance_mask_predicted.shape)
        print("category_mask_predicted.shape", category_mask_predicted.shape)
        
        




if __name__ == '__main__':
    tf.test.main()
