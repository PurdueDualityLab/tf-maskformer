import tensorflow as tf
import numpy as np

class PanopticInference:
    """Panoptic Inference"""
    def __init__(self, num_classes=133, background_class_id=0):

        self.num_classes = num_classes
        self.background_class_id = background_class_id
        

    def call(self, pred_logits, mask_pred, image_shape):
        
        cat_id_map = {}
        is_this_thing = {}
        # Apply softmax and sigmoid on the predictions and predicted masks
        mask_pred =  tf.image.resize(mask_pred, (image_shape[1], image_shape[2]), method=tf.image.ResizeMethod.BILINEAR)
        mask_pred = tf.keras.activations.sigmoid(mask_pred)
        probs = tf.keras.activations.softmax(pred_logits, axis=-1)
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        
        # Keep only those masks that have high confidence and are not background
        # TODO : Some operations may be incompatible on TPU
       
        object_mask_threshold = 0.25
        keep = tf.math.logical_and(tf.math.not_equal(labels, self.background_class_id), scores > object_mask_threshold)
        curr_scores = scores[keep]
        curr_classes = labels[keep]

       
        permute = tf.keras.layers.Permute((3,1,2)) 
        mask_pred = permute(mask_pred) # (batch, num_predictions, height, width)
        curr_masks = mask_pred[keep]
        curr_mask_cls = pred_logits[keep]
        # Save the masks here and visualize the output
        np.save("output_masks.npy", curr_masks.numpy())
        np.save("output_cls.npy", curr_mask_cls.numpy())
        exit()
        # curr_prob_masks = tf.reshape(curr_scores, [-1, 1, 1]) * curr_masks
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
                binary_mask = curr_masks[k] >= 0.5
                pred_class = curr_classes[k].numpy()
                class_score = curr_scores[k].numpy()
                if class_score >= 0.5:
                    category_id = cat_id_map[pred_class]
                    if is_this_thing[category_id]:
                        category_mask = tf.where(binary_mask, category_id, category_mask)
                        instance_id += 1
                    else:
                        instance_mask = tf.where(binary_mask, instance_id, instance_mask)
                        instance_id = _VOID_INSTANCE_ID
        
            return instance_mask, category_mask

