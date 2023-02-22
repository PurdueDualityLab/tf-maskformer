import tensorflow as tf

from official.vision.losses import focal_loss

class FocalLoss(focal_loss.FocalLoss):
    """Implements a Focal loss for segmentation problems.
    Reference:
        [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278).
    """

    def __init__(self, alpha=0.25, gamma=2):
        """Initializes `FocalLoss`.
        Args:
        alpha: The `alpha` weight factor for binary class imbalance.
        gamma: The `gamma` focusing parameter to re-weight loss.
        reduction and name?
        """
        super().__init__(alpha, gamma)

    def call(self, y_true, y_pred, num_masks):
        """Invokes the `FocalLoss`.
        Args:
        y_true: A tensor of size [batch, num_anchors, num_classes]. 
        Stores the binary classification lavel for each element in y_pred.
        y_pred: A tensor of size [batch, num_anchors, num_classes]. 
        The predictions of each example.
        num_masks: The number of masks.

        Returns:
        Loss float `Tensor`.
        """
        weighted_loss = super().call(y_true, y_pred)
        loss = tf.math.reduce_sum(tf.math.reduce_mean(weighted_loss,axis=1)) / num_masks
        return loss

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self):
        pass

class Loss():
    def __init__(num_classes, empty_weight):
        self.num_classes = num_classes
        self.empty_weight = empty_weight

    def _get_pred_permutation_idx(self, indices):
        batch_idx = tf.concat([tf.fill(pred,i) for i, (pred,_) in enumerate(indices)], axis=0)
        pred_idx = tf.concat([pred for (pred,) in indices], axis=0)
        return batch_idx, pred_idx
    
    def _get_true_permutation_idx(self, indices):
        batch_idx = tf.concat([tf.fill(true,i) for i, (_,true) in enumerate(indices)], axis=0)
        true_idx = tf.concat([true for (_,true) in indices], axis=0)
        return batch_idx, true_idx

class ClassificationLoss(Loss):
    def call(self, outputs, y_true, indices, num_masks):
        assert "pred_logits" in outputs

        pred_logits = outputs["pred_logits"]

        idx = super()._get_pred_permutation_idx(indices)
        true_classes_o = tf.concat([t["labels"][J] for t, (_, J) in zip(y_true, indices)], axis=0)
        true_classes = tf.cast(tf.fill(pred_logits.shape[:2], super().num_classes), dtype=tf.int64) # device?
        # target_classes = tf.tensor_scatter_nd_update(target_classes, tf.expand_dims(idx, axis=1), target_classes_o)
        true_classes[idx] = true_classes_o

        # loss_ce = tf.nn.softmax_cross_entropy_with_logits(y_true, tf.transpose(pred_logits,(1,2)))
        loss_ce = tf.nn.weighted_cross_entropy_with_logits(y_true, tf.transpose(pred_logits,(1,2)), super().empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

class MaskLoss(Loss):
    def call(self, outputs, y_true, indices, num_masks):
        assert "pred_masks" in outputs

        pred_idx = super()._get_pred_permutation_idx(indices)
        true_idx = super()._get_true_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks[pred_idx]
        masks = [t["masks"] for t in y_true]

        pred_masks = tf.reshape(pred_masks[:, 0], -1)
        true_masks = tf.reshape(true_masks, -1)
        true_masks = tf.reshape(true_masks, pred_masks.shape)
        losses = {
            "loss_mask": FocalLoss().call(pred_masks, true_masks, num_masks),
            "loss_dice": DiceLoss().call(pred_masks, true_masks, num_masks)
        }
        return losses