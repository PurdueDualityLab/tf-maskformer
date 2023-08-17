import tensorflow as tf
import numpy as np
from mapper import _get_contigious_to_original, _get_original_to_contigious

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
        mask_pred =  tf.image.resize(mask_pred, (image_shape[0], image_shape[1]), method=tf.image.ResizeMethod.BILINEAR)
        mask_pred = tf.keras.activations.sigmoid(mask_pred)
        probs = tf.keras.activations.softmax(pred_logits, axis=-1) # (batch, num_predictions, num_classes) (2,100,134)
        scores = tf.reduce_max(probs, axis=-1) 
        labels = tf.argmax(probs, axis=-1)
        
        # Keep only those masks that have high confidence and are not background
        # TODO : Some operations may be incompatible on TPU
        print("Background class id", self.background_class_id)
        keep = tf.math.logical_and(tf.math.not_equal(labels, self.background_class_id), scores > self.object_mask_threshold)
        print("keep", keep)
        exit()
        curr_scores = scores[keep]
        curr_classes = labels[keep]

        permute = tf.keras.layers.Permute((3,1,2)) 
        mask_pred = permute(mask_pred) # (batch, num_predictions, height, width)
        curr_masks = mask_pred[keep]
    
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

