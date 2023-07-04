from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.detr.ops.matchers import hungarian_matching
from absl.testing import parameterized
import tensorflow as tf
from official.vision.ops import preprocess_ops
import numpy as np
import shutil
class LossTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        matcher = hungarian_matching
        no_object_weight = 0.1
       
        losses = ["labels", "masks"]
        self.weight_dict = {
            "ce_loss" : 1.0,
            "focal_loss" : 20.0,
            "dice_loss" : 1.0,
        }
        # FIXME : The number of class is dependednt on if we use contigious ids or not.
        loss = Loss(
            num_classes = 133,
            matcher = matcher,
            eos_coef = no_object_weight,
            cost_class= 1.0,
            cost_dice= 1.0,
            cost_focal=20.0
        )
        
        main_pth = "/depot/qqiu/data/vishal/projects/tf-maskformer/models/official/projects/maskformer/losses"
        
        # Load prediction from PyTorch Model
        aux_out_0 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits0.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks0.npy"))}
        aux_out_1 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits1.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks1.npy"))}
        aux_out_2 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits2.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks2.npy"))}
        aux_out_3 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits3.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks3.npy"))}
        aux_out_4 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits4.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks4.npy"))}
        aux_outputs = [aux_out_0, aux_out_1, aux_out_2, aux_out_3, aux_out_4]
        pred_logits_load = tf.convert_to_tensor(np.load(main_pth+"/tensors/output_pred_logits.npy")) 
        pred_masks_load = tf.convert_to_tensor(np.load(main_pth+"/tensors/output_pred_masks.npy"))
        
        # Load targets used by PyTorch Model
        target_labels_1 = tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_labels_0.npy'))
        target_labels_2 = tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_labels_1.npy'))
        target_masks_1 = tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_0.npy'))
        target_masks_2 = tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_1.npy'))
        print("Shape of target_labels_1 :", target_labels_1.shape)
        # Print shapes of raw tensors from PyTorch Model
        # print("Shape of pred_logits_load: ", pred_logits_load.shape)
        # print("Shape of pred_masks_load: ", pred_masks_load.shape)
        # print("Shape of target_labels_1: ", target_labels_1.shape)
        # print("Shape of target_labels_2: ", target_labels_2.shape)
        # print("Shape of target_masks_1: ", target_masks_1.shape)
        # print("Shape of target_masks_2: ", target_masks_2.shape)
        
        # Preprocess the PyTorch tensors to match the dimension output from our model
        pred_masks_load = tf.transpose(pred_masks_load, [0,2,3,1]) # (2,100, h, w) -> (2, h, w, 100) (actual outptu of our model)
        
        outputs = {
            "pred_logits": pred_logits_load,
            "pred_masks": pred_masks_load,
            "aux_outputs": aux_outputs 
        }

        # Preprocess the PyTorch tensors by padding them with zeros to make the shape constant to 100 and batch the target tensors
        targets = {}
        
        # pad target labels tensor with zeros to make it of shape (100, )
        target_labels_1_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_labels_1, 100)
        target_labels_2_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_labels_2, 100)
        # Batching the target labels
        target_labels_stacked = tf.stack([target_labels_1_padded, target_labels_2_padded], axis=0) 
        
        # Rsize the target masks to (640, 640) and then pad it with zeros to make it of shape (100, 640, 640)
        target_masks_1 =  tf.transpose(target_masks_1, [1,2,0])
        target_masks_2 =  tf.transpose(target_masks_2, [1,2,0])
        target_masks_1 =  tf.image.resize(tf.cast(target_masks_1, float), (640, 640))
        target_masks_2 =  tf.image.resize(tf.cast(target_masks_2,float), (640, 640))

        # Reoder the masks again back to [100, h,w]
        target_masks_1 =  tf.transpose(target_masks_1, [2,0,1])
        target_masks_2 =  tf.transpose(target_masks_2, [2,0,1])

        # Transpose so that preprocess_ops can work and pad it with zeros to make it of shape (100, H, W)
        target_masks_1_padded = tf.concat([target_masks_1, tf.zeros([100-target_masks_1.shape[0], 640, 640])], axis=0)
        target_masks_2_padded = tf.concat([target_masks_2, tf.zeros([100-target_masks_2.shape[0], 640, 640])], axis =0)
        
        # batch the target masks
        target_masks_stacked = tf.stack([target_masks_1_padded, target_masks_2_padded], axis=0) # Stacking the two tensors along the batch dimension
        targets["unique_ids"] = target_labels_stacked
        targets["individual_masks"] = tf.expand_dims(tf.cast(target_masks_stacked, tf.bool), -1)

        # print("Shape of pred_logits_load: ", outputs["pred_logits"].shape)
        # print("Shape of pred_masks_load: ", outputs["pred_masks"].shape)
        # print("Shape of target_labels_stacked: ", targets["unique_ids"].shape)
        # print("Shape of target_masks_stacked: ", targets["individual_masks"].shape)
        # exit()
       
        # Save the outputs and targets to a file to be used by pytorch code 
        main_path_pytorch = "/depot/qqiu/data/vishal/MaskFormer_pytorch/tf_tensors"
        np.save(main_path_pytorch+"/tf_output_pred_logits.npy", outputs["pred_logits"].numpy())
        np.save(main_path_pytorch+"/tf_output_pred_masks.npy", outputs["pred_masks"].numpy())
        np.save(main_path_pytorch+"/tf_targets_labels.npy", targets["unique_ids"].numpy())
        np.save(main_path_pytorch+"/tf_targets_masks.npy", targets["individual_masks"].numpy())
        # src_pth = "/depot/qqiu/data/vishal/projects/tf-maskformer/models/official/projects/maskformer/losses/tensors"
        # dst_pth = "/depot/qqiu/data/vishal/MaskFormer_pytorch/tf_tensors"
        # shutil.rmtree(dst_pth)
        # shutil.copytree(src_pth, dst_pth)
        print("[INFO] TF Tensors Copy Successful!!!...")
        # Vectorized loss function accepts batched inputs
        losses = loss(outputs, targets)
       
        print("Losses are : ", losses)
        print("Total Loss is :", losses['loss_ce'] + losses['loss_dice'] + losses['loss_focal'])
        
        # for i in range(4):
        #     print(f"Total aux Loss {i} : losses['loss_ce_'+{str(i)}] + losses['loss_dice_'+{str(i)}] + losses['loss_focal_'+{str(i)}]")
        # TODO: Check if this is correct
        # self.assertAllEqual(losses, )

if __name__ == '__main__':
    tf.test.main()