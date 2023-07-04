from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.detr.ops.matchers import hungarian_matching
from absl.testing import parameterized
import tensorflow as tf
from official.vision.ops import preprocess_ops
import numpy as np
import shutil
import pickle

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
        loss = Loss(
            num_classes = 133,
            matcher = matcher,
            eos_coef = no_object_weight,
            cost_class= 1.0,
            cost_dice= 1.0,
            cost_focal=20.0
        )
        
        # outputs = {"pred_logits":tf.convert_to_tensor(np.load("output_pred_logits.npy")), "pred_masks":tf.convert_to_tensor(np.load("output_pred_masks.npy"))}
        # print(f"outputs['pred_logits'] shape is {outputs['pred_logits'].shape}")
        # print(f"outputs['pred_masks'] shape is {outputs['pred_masks'].shape}")

        main_pth = "/depot/qqiu/data/vishal/projects/tf-maskformer/models/official/projects/maskformer/losses"
        aux_out_0 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits0.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks0.npy"))}
        aux_out_1 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits1.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks1.npy"))}
        aux_out_2 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits2.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks2.npy"))}
        aux_out_3 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits3.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks3.npy"))}
        aux_out_4 = {"pred_logits" : tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_logits4.npy")), "pred_masks": tf.convert_to_tensor(np.load(main_pth+"/tensors/aux_outputs_pred_masks4.npy"))}
        aux_outputs = [aux_out_0, aux_out_1, aux_out_2, aux_out_3, aux_out_4]
        pred_logits_load = tf.convert_to_tensor(np.load(main_pth+"/tensors/output_pred_logits.npy")) 
        pred_masks_load = tf.convert_to_tensor(np.load(main_pth+"/tensors/output_pred_masks.npy"))
        # print("pred_logits_load shape is ", pred_logits_load.shape)
        # print("pred_masks_load shape is ", pred_masks_load.shape)
        # exit()
        # TODO
        pred_masks_load = tf.transpose(pred_masks_load, [0,2,3,1]) # (1,100, h, w) -> (1, h, w, 100) (actual outptu of our model)
        
        outputs = {
            "pred_logits": pred_logits_load,
            "pred_masks": pred_masks_load,
            "aux_outputs": aux_outputs 
        }

        # Load the new_targets_dict NumPy array
        targets = {}
        
        # pad tensor with zeros to make it of shape (100, )
        target_labels_1 = tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_labels_0.npy'))
        target_labels_2 = tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_labels_1.npy'))
        print("Target labels 1  shape is ", target_labels_1.shape)
        target_labels_1_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_labels_1, 100)
        target_labels_2_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_labels_2, 100)
        target_labels_stacked = tf.stack([target_labels_1_padded, target_labels_2_padded], axis=0) # Stacking the two tensors along the batch dimension
       
        target_masks_1 = tf.transpose(tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_0.npy')), [1,2,0])
        target_masks_2 = tf.transpose(tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_1.npy')), [1,2,0])
        
     
        target_masks_1 =  tf.image.resize(tf.cast(target_masks_1, float), (640, 640))
        target_masks_2 =  tf.image.resize(tf.cast(target_masks_2,float), (640, 640))

        # Resize the tensor to fixed H and W and then pad it with zeros to make it of shape (100, H, W)
        target_masks_1_reshaped = tf.transpose(target_masks_1, [2,0,1])
        target_masks_2_reshaped = tf.transpose(target_masks_2, [2,0,1])

        target_masks_1_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_masks_1_reshaped, 100)
        target_masks_2_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_masks_2_reshaped, 100)
        
        target_masks_stacked = tf.stack([target_masks_1_padded, target_masks_2_padded], axis=0) # Stacking the two tensors along the batch dimension
        targets["unique_ids"] = target_labels_stacked
        targets["individual_masks"] = tf.expand_dims(target_masks_stacked, -1)

        # print("targets['unique_ids'] shape is : ", targets["unique_ids"].shape)
        # print("targets['individual_masks'] shape is : ", targets["individual_masks"].shape)
        # print("outputs['pred_masks'] shape is : ", outputs["pred_masks"].shape)
        # exit()
        main_pth = "/depot/qqiu/data/vishal/projects/tf-maskformer/models/official/projects/maskformer/losses/tensors"
        # np.save(main_pth+'/individual_masks.npy', targets["individual_masks"].numpy())
        # np.save(main_pth+'/unique_ids.npy', targets["unique_ids"].numpy())
        # np.save(main_pth+'/pred_masks.npy', outputs["pred_masks"].numpy())
        # np.save(main_pth+'/pred_logits.npy', outputs["pred_logits"].numpy())
       
        # src_pth = "/depot/qqiu/data/vishal/projects/tf-maskformer/models/official/projects/maskformer/losses/tensors"
        # dst_pth = "/depot/qqiu/data/vishal/MaskFormer_pytorch/tensors"
        # shutil.rmtree(dst_pth)
        # shutil.copytree(src_pth, dst_pth)
        # print("[INFO] Tensors Copy Successful!!!...")
        
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