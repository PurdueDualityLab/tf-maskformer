import tensorflow as tf

class PanopticInference():
    
    def call(self, pred_logits, mask_pred, image_shape, num_classes):
        #interpolate = tf.keras.layers.Resizing(
        #            image_shape[1], image_shape[2], interpolation = "bilinear")
        #mask_pred = interpolate(mask_pred)
        mask_pred =  tf.image.resize(mask_pred, (image_shape[1], image_shape[2]), method=tf.image.ResizeMethod.BILINEAR)
        probs = tf.keras.activations.softmax(pred_logits, axis=-1)
        print("Probs_shape: ", probs.shape)
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        mask_pred = tf.keras.activations.sigmoid(mask_pred)

        exit()
        config_num_classes = num_classes
        object_mask_threshold = 0.0
        keep = tf.math.logical_and(tf.math.not_equal(labels, config_num_classes), scores > object_mask_threshold)
        curr_scores = scores[keep]
        curr_classes = labels[keep]

        #print(mask_pred.shape)
        #print(keep.shape)
        permute = tf.keras.layers.Permute((3,1,2))
        mask_pred = permute(mask_pred)
        curr_masks = mask_pred[keep]
        print(f"curr_masks[keep]: {curr_masks.shape}")
        #curr_masks = tf.boolean_mask(mask_pred, keep)
        curr_mask_cls = pred_logits[keep]
        print(f"mask_true[keep]: {curr_mask_cls.shape}")
        curr_mask_cls = tf.slice(curr_mask_cls, [0, 0], [-1, curr_mask_cls.shape[1] - 1])

        curr_prob_masks = tf.reshape(curr_scores, [-1, 1, 1]) * curr_masks

        height, width = tf.shape(curr_masks)[-2:]

        with tf.device(curr_masks.device):
            panoptic_seg = tf.zeros((height, width), dtype=tf.int32)
        segments_info = []

        current_segment_id = 0

        if tf.shape(curr_masks)[0] == 0:
            return panoptic_seg, segments_info
        else:
            curr_masks_ids = tf.argmax(curr_prob_masks, axis=0)
            stuff_memory = {}

            for k in range(curr_classes.shape[0]):
                pred_class = curr_classes[k].numpy()
                # is_thing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                is_thing = True # TODO(ibrahim): FIX when get configs.

                mask = curr_masks_ids == k
               # mask_area = tf.reduce_sum(mask).numpy()
                mask_area = tf.reduce_sum(tf.cast(mask, tf.float32)).numpy()
                original_area = tf.reduce_sum(tf.cast(curr_masks[k] >= 0.5, tf.float32)).numpy()

                if mask_area > 0 and original_area > 0:
                    config_overlap_threshold = 0.8
                    if mask_area / original_area < config_overlap_threshold:
                        continue
                    
                    if not is_thing:
                        if int(pred_class) in stuff_memory:
                            panoptic_seg[mask] = stuff_memory[int(pred_class)]
                            continue
                        else:
                            stuff_memory[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                   # panoptic_seg[mask] = current_segment_id
                    panoptic_seg = tf.where(mask, current_segment_id, panoptic_seg)
                    segments_info.append({
                        "id": current_segment_id,
                        "is_thing": bool(is_thing),
                        "category_id": int(pred_class),
                    })
            
            return panoptic_seg, segments_info

