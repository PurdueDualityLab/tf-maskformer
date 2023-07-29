from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.detr.ops.matchers import hungarian_matching
from absl.testing import parameterized
import tensorflow as tf

import numpy as np

import pickle

class LossTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        matcher = hungarian_matching
        no_object_weight = 0.1
    
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
        outputs = {
            "pred_logits": pred_logits_load,
            "pred_masks": pred_masks_load,
            "aux_outputs": aux_outputs 
        }

        # Load the new_targets_dict NumPy array
        targets = []
        # TODO :  Caution the below loop is for each image in the batch
        for i in range(2): # Here 2 is for batch size 
            targets.append(
                {
                    "labels": tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_labels_'+str(i)+'.npy')),
                    "masks": tf.convert_to_tensor(np.load(main_pth+'/tensors/targets_masks_'+str(i)+'.npy')),
                }
            )


        losses = loss(outputs, targets)
       

        print("Losses are : ", losses)
        print("Total Loss is :", losses['loss_ce'] + losses['loss_dice'] + losses['loss_focal'])
        # for i in range(4):
        #     print(f"Total aux Loss {i} : losses['loss_ce_'+{str(i)}] + losses['loss_dice_'+{str(i)}] + losses['loss_focal_'+{str(i)}]")
        # TODO: Check if this is correct
        # self.assertAllEqual(losses, )

if __name__ == '__main__':
    tf.test.main()