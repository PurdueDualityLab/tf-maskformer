import tensorflow as tf
import numpy as np
from official.projects.maskformer.losses.mapper import _get_contigious_to_original, _get_original_to_contigious

class PanopticInference:
    """Panoptic Inference"""
    def __init__(self, num_classes=134, background_class_id=0, object_mask_threshold=0.1, class_score_threshold=0.25, overlap_threshold=0.2):

        self.num_classes = num_classes
        self.background_class_id = background_class_id
        self.object_mask_threshold = object_mask_threshold
        self.class_score_threshold = class_score_threshold
        self.cat_id_map, self.is_thing_dict, _ = _get_contigious_to_original()
        self.overlap_threshold = overlap_threshold


    def __call__(self, pred_logits, mask_pred, image_shape):
        """
        mask_pred: (batch, height, width, num_predictions)
        pred_logits: (batch, num_predictions, num_classes)
        """
        
         # maps from contiguous category id to original category id
        
        instance_masks = []
        category_masks = []

        for each_batch in range(tf.shape(pred_logits)[0]):
            mask_pred_b = mask_pred[each_batch]
            pred_logits_b = pred_logits[each_batch]
            mask_pred_b_resized =  tf.image.resize(mask_pred_b, (image_shape[0], image_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
            mask_pred_b_sigmoid = tf.keras.activations.sigmoid(mask_pred_b_resized)
            probs = tf.keras.activations.softmax(pred_logits_b, axis=-1) # (num_predictions, num_classes) (2,100,134)
            scores = tf.reduce_max(probs, axis=-1) 
            labels = tf.argmax(probs, axis=-1)
            tf.print("Scores :", scores)
            ################################################## Only for testing instance and category mask ##################################################
            # Replace '133' with '0' (background class id) and increment all other classes by 1
            # print("Labels  before :", labels)
            # labels = tf.where(tf.math.equal(labels, 133), 0, labels+1)
            # print("Labels  after :", labels)
            # exit()
            ################################################## Only for testing instance and category mask ##################################################
            keep = tf.math.logical_and(tf.math.not_equal(labels, self.background_class_id), scores > self.object_mask_threshold)
            
            mask_pred_b_sigmoid = tf.transpose(mask_pred_b_sigmoid, (2, 0, 1)) # ( num_predictions, height, width)
            
            # Give batch of predictions to the function
            curr_masks = tf.boolean_mask(mask_pred_b_sigmoid, keep) # (num_predictions, height, width)
            curr_classes = tf.boolean_mask(labels, keep) # (num_predictions)
            curr_scores = tf.boolean_mask(scores, keep) # (num_predictions)
            cur_prob_masks = tf.reshape(curr_scores,(-1, 1, 1)) * curr_masks
            

            height, width = tf.shape(curr_masks)[1], tf.shape(curr_masks)[2]

            # Create  category mask and instance mask
            with tf.device(curr_masks.device):
                category_mask = tf.zeros((height, width), dtype=tf.int32)
                instance_mask = tf.zeros((height, width), dtype=tf.int32)

            
            if tf.shape(curr_masks)[0] == 0:
                continue
            else:
                
                _VOID_INSTANCE_ID = 0
                instance_id = 0
                cur_mask_ids = tf.argmax(cur_prob_masks, 0)
                
                for k in range(tf.shape(curr_classes)[0]):
                    pred_class = curr_classes[k]
                    
                    # isthing = self.is_thing_dict[self.cat_id_map[int(pred_class)]]

                    binary_mask = tf.math.equal(cur_mask_ids, tf.cast(k, tf.int64))
                    binary_mask = tf.cast(binary_mask, tf.int32)
                    
                    mask_area = tf.math.reduce_sum(binary_mask)
                    original_area = tf.math.reduce_sum(tf.cast(curr_masks[k] >= 0.5, tf.int32))
                    pred_class = curr_classes[k]
                    class_score = curr_scores[k]

                    if mask_area > 0 and original_area > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue
                        category_id = self.cat_id_map.lookup(tf.cast(pred_class, tf.int32))
                        binary_mask = tf.cast(binary_mask, tf.bool)
                        category_mask = tf.where(binary_mask, category_id, category_mask)
                        if self.is_thing_dict.lookup(category_id):
                            
                            instance_mask = tf.where(binary_mask, instance_id, instance_mask)
                            instance_id += 1
                        else:
                            instance_mask = tf.where(binary_mask, _VOID_INSTANCE_ID, instance_mask)
                            
            instance_masks.append(instance_mask)
            category_masks.append(category_mask)

        instance_masks_stacked = tf.stack(instance_masks, axis=0)
        category_masks_stacked = tf.stack(category_masks, axis=0)
        return instance_masks_stacked, category_masks_stacked

