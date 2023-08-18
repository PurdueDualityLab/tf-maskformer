import tensorflow as tf
import numpy as np
from official.projects.maskformer.losses.mapper import _get_contigious_to_original, _get_original_to_contigious

class PanopticInference:
    """Panoptic Inference"""
    def __init__(self, num_classes=134, background_class_id=133, object_mask_threshold=0.25, class_score_threshold=0.25):

        self.num_classes = num_classes
        self.background_class_id = background_class_id
        self.object_mask_threshold = object_mask_threshold
        self.class_score_threshold = class_score_threshold

    def __call__(self, pred_logits, mask_pred, image_shape):
        
        cat_id_map, is_thing_dict = _get_contigious_to_original() # maps from contiguous category id to original category id
        # Apply softmax and sigmoid on the predictions and predicted masks
        instance_masks = []
        category_masks = []

        for each_batch in range(pred_logits.shape[0]):
            mask_pred_b = mask_pred[each_batch]
            pred_logits_b = pred_logits[each_batch]
            mask_pred_b_resized =  tf.image.resize(mask_pred_b, (image_shape[1], image_shape[2]), method=tf.image.ResizeMethod.BILINEAR)
            mask_pred_b_sigmoid = tf.keras.activations.sigmoid(mask_pred_b_resized)
            probs = tf.keras.activations.softmax(pred_logits_b, axis=-1) # (batch, num_predictions, num_classes) (2,100,134)
            scores = tf.reduce_max(probs, axis=-1) 
            labels = tf.argmax(probs, axis=-1)
    
            keep = tf.math.logical_and(tf.math.not_equal(labels, self.background_class_id), scores > self.object_mask_threshold)
            
            permute = tf.keras.layers.Permute((2,0,1)) 
            mask_pred_b_sigmoid = permute(mask_pred_b_sigmoid) # (batch, num_predictions, height, width)
            
            # Give batch of predictions to the function
            curr_masks = tf.boolean_mask(mask_pred_b_sigmoid, keep) # (num_predictions, height, width)
            curr_classes = tf.boolean_mask(labels, keep) # (num_predictions)
            curr_scores = tf.boolean_mask(scores, keep) # (num_predictions)
            print("Curr masks : ", curr_masks.shape)
            print("Curr classes : ", curr_classes.shape)
            print("Curr scores : ", curr_scores.shape)
            exit()
            
            height, width = tf.shape(curr_masks)[-2:]

            # Create  category mask and instance mask
            with tf.device(curr_masks.device):
                category_mask = tf.zeros((height, width), dtype=tf.int32)
                instance_mask = tf.zeros((height, width), dtype=tf.int32)

            if tf.shape(curr_masks)[0] == 0:
                raise  ValueError("No masks to process")
            else:
                _VOID_INSTANCE_ID = 0
                instance_id = 0
                for k in range(curr_classes.shape[0]):
                    binary_mask = curr_masks[k] >= self.object_mask_threshold
                    pred_class = curr_classes[k].numpy()
                    class_score = curr_scores[k].numpy()
                    if class_score >= self.class_score_threshold:
                        category_id = cat_id_map[pred_class]
                        if is_thing_dict[category_id]:
                            category_mask = tf.where(binary_mask, category_id, category_mask)
                            instance_id += 1
                        else:
                            instance_mask = tf.where(binary_mask, instance_id, instance_mask)
                            instance_id = _VOID_INSTANCE_ID
        
            return instance_mask, category_mask

