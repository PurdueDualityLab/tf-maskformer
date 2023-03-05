import tensorflow as tf

class PanopticInference():
    def call(self, mask_true, mask_pred):
        probs = tf.keras.activations.softmax(mask_true, axis=-1)
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        mask_pred = tf.keras.activations.sigmoid(mask_pred)

        keep = tf.math.logical_and(tf.math.not_equal(labels, self.sem_seg_head.num_classes), scores > self.object_mask_threshold)
        curr_scores = scores[keep]
        curr_classes = labels[keep]
        curr_masks = mask_pred[keep]
        curr_mask_true = mask_true[keep]
        cur_mask_cls = tf.slice(cur_mask_cls, [0, 0], [-1, cur_mask_cls.shape[1] - 1])

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
                is_thing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask = curr_masks_ids == k
                mask_area = tf.reduce_sum(mask).numpy()
                original_area = tf.reduce_sum(curr_masks[k] >= 0.5).numpy()

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    
                    if not is_thing:
                        if int(pred_class) in stuff_memory:
                            panoptic_seg[mask] = stuff_memory[int(pred_class)]
                            continue
                        else:
                            stuff_memory[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append({
                        "id": current_segment_id,
                        "is_thing": bool(is_thing),
                        "category_id": int(pred_class),
                    })
            
            return panoptic_seg, segments_info

