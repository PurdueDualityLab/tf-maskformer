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
            cost_focal=20.0,
            ignore_label = 133
        )
        USE_PAD_FIXED_SIZE = True
        # outputs = {"pred_logits":tf.convert_to_tensor(np.load("output_pred_logits.npy")), "pred_masks":tf.convert_to_tensor(np.load("output_pred_masks.npy"))}
        # print(f"outputs['pred_logits'] shape is {outputs['pred_logits'].shape}")
        # print(f"outputs['pred_masks'] shape is {outputs['pred_masks'].shape}")

        main_pth = "/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models/official/projects/maskformer/losses"
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
        
        if USE_PAD_FIXED_SIZE:
            target_labels_1_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_labels_1, 100,133)
            target_labels_2_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_labels_2, 100,133)
        else:
            classes_list_1_padded = tf.TensorArray(tf.float32, size=100)
            classes_list_2_padded = tf.TensorArray(tf.float32, size=100)
            counter = 0
            counter_1 = 0
            for i in range(0, target_labels_1.shape[0]):
                classes_list_1_padded = classes_list_1_padded.write(counter, tf.cast(target_labels_1[i], tf.float32))
                counter += 1
            for j in range(100-target_labels_1.shape[0]):
                classes_list_1_padded = classes_list_1_padded.write(counter, 133.0)
                counter += 1

            for i in range(0, target_labels_2.shape[0]):
                classes_list_2_padded = classes_list_2_padded.write(counter_1, tf.cast(target_labels_2[i], tf.float32))
                counter_1 += 1
            for j in range(100-target_labels_2.shape[0]):
                classes_list_2_padded = classes_list_2_padded.write(counter_1, 133.0)
                counter_1 += 1
            target_labels_1_padded = classes_list_1_padded.stack()
            target_labels_2_padded = classes_list_2_padded.stack()

        target_labels_stacked = tf.stack([target_labels_1_padded, target_labels_2_padded], axis=0) # Stacking the two tensors along the batch dimension
       
        target_masks_1 = tf.transpose(tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_0.npy')), [1,2,0])
        target_masks_2 = tf.transpose(tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_1.npy')), [1,2,0])
        
     
        target_masks_1 =  tf.image.resize(tf.cast(target_masks_1, float), (640, 640))
        target_masks_2 =  tf.image.resize(tf.cast(target_masks_2,float), (640, 640))

        # Resize the tensor to fixed H and W and then pad it with zeros to make it of shape (100, H, W)
        target_masks_1_reshaped = tf.transpose(target_masks_1, [2,0,1])
        target_masks_2_reshaped = tf.transpose(target_masks_2, [2,0,1])

        if USE_PAD_FIXED_SIZE:
            target_masks_1_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_masks_1_reshaped, 100, 133)
            target_masks_2_padded = preprocess_ops.clip_or_pad_to_fixed_size(target_masks_2_reshaped, 100, 133)
        else:
            
            mask_list_1_padded = tf.TensorArray(tf.float32, size=100)
            mask_list_2_padded = tf.TensorArray(tf.float32, size=100)
            counter_2 = 0
            counter_3 = 0
            for i in range(0, target_masks_1_reshaped.shape[0]):
                mask_list_1_padded = mask_list_1_padded.write(counter_2, target_masks_1_reshaped[i])
                counter_2 += 1
            for j in range(100-target_masks_1_reshaped.shape[0]):
                mask_list_1_padded = mask_list_1_padded.write(counter_2, tf.zeros(target_masks_1_reshaped[0].shape))
                counter_2 += 1

            for i in range(0, target_masks_2_reshaped.shape[0]):
                mask_list_2_padded = mask_list_2_padded.write(counter_3, target_masks_2_reshaped[i])
                counter_3 += 1
            for j in range(100-target_masks_2_reshaped.shape[0]):
                mask_list_2_padded = mask_list_2_padded.write(counter_3,  tf.zeros(target_masks_2_reshaped[0].shape))
                counter_3 += 1

            target_masks_1_padded = mask_list_1_padded.stack()
            target_masks_2_padded = mask_list_2_padded.stack()

        target_masks_stacked = tf.stack([target_masks_1_padded, target_masks_2_padded], axis=0) # Stacking the two tensors along the batch dimension
        
        targets['unique_ids'] = target_labels_stacked
        targets["individual_masks"] = tf.expand_dims(target_masks_stacked, -1)

        # print("targets['unique_ids'] shape is : ", targets["unique_ids"].shape)
        # print("targets['individual_masks'] shape is : ", targets["individual_masks"].shape)
        # print("outputs['pred_masks'] shape is : ", outputs["pred_masks"].shape)
        # main_pth = "/depot/qqiu/data/vishal/projects/tf-maskformer/models/official/projects/maskformer/losses/tf_tensors"
        # np.save(main_pth+'/tf_targets_masks.npy', targets["individual_masks"].numpy())
        # np.save(main_pth+'/tf_targets_labels.npy', targets["unique_ids"].numpy())
        # np.save(main_pth+'/tf_output_pred_masks.npy', outputs["pred_masks"].numpy())
        # np.save(main_pth+'/tf_output_pred_logits.npy', outputs["pred_logits"].numpy())
       
        # src_pth = "/depot/qqiu/data/vishal/projects/tf-maskformer/models/official/projects/maskformer/losses/tf_tensors"
        # dst_pth = "/depot/qqiu/data/vishal/MaskFormer_pytorch/tf_tensors"
        # shutil.rmtree(dst_pth)
        # shutil.copytree(src_pth, dst_pth)
        # print("[INFO] Tensors Copy Successful!!!...")
        # exit()
        # Vectorized loss function accepts batched inputs
        print("[INFO] Calling loss function....")
        losses = loss(outputs, targets)


        print(f"Classification loss : {losses['loss_ce'].numpy()} Dice Loss : {losses['loss_dice'].numpy()} Focal Loss : {losses['loss_focal'].numpy()}")

        for i in range(5):
            print(f'loss_ce_{str(i)}: ', losses[f'loss_ce_{str(i)}'])
            print(f'loss_focal_{str(i)}: ', losses[f'loss_focal_{str(i)}'])
            print(f'loss_dice_{str(i)}: ', losses[f'loss_dice_{str(i)}'])

        print("Total Loss is :", losses['loss_ce'].numpy() + losses['loss_dice'].numpy() + losses['loss_focal'].numpy())
        
        print()
        exit() 


        # TODO: Check if this is correct
        # self.assertAllEqual(losses, )

if __name__ == '__main__':
    tf.test.main()
