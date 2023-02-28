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
    def __init__(num_classes, matcher, eos_coef, losses):
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = tf.ones(self.num_classes + 1)
        empty_weight = tf.tensor_scatter_nd_update(empty_weight_tf, [[self.num_classes]], [self.eos_coef])
        self.empty_weight = tf.Variable(empty_weight_tf, trainable=False, name='empty_weight')        

    def _get_pred_permutation_idx(self, indices):
        batch_idx = tf.concat([tf.fill(pred,i) for i, (pred,_) in enumerate(indices)], axis=0)
        pred_idx = tf.concat([pred for (pred,) in indices], axis=0)
        return batch_idx, pred_idx
    
    def _get_true_permutation_idx(self, indices):
        batch_idx = tf.concat([tf.fill(true,i) for i, (_,true) in enumerate(indices)], axis=0)
        true_idx = tf.concat([true for (_,true) in indices], axis=0)
        return batch_idx, true_idx
    
    def get_loss(self, loss, outputs, y_true, indices, num_masks):
        loss_map = {"labels": ClassificationLoss().call, "masks": MaskLoss().call}
        assert loss in loss_map
        return loss_map[loss](outputs, y_true, indices, num_masks)

    def call(self, outputs, y_true):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             y_true: list of dicts, such that len(targets) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # TODO: check matcher doc
        indices = self.matcher(outputs_without_aux, y_true)

        num_masks = sum(len(t["labels"]) for t in y_true)
        num_masks = tf.convert_to_tensor([num_masks], dtype=tf.float64) # device?
        
        # TODO: figure out how to implement lines 168-171 from PT

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, y_true, indices, num_masks))
        
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, y_true)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, y_true, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses


class ClassificationLoss(Loss):
    def call(self, outputs, y_true, indices, num_masks):
        assert "pred_logits" in outputs

        pred_logits = outputs["pred_logits"]

        idx = super()._get_pred_permutation_idx(indices)
        true_classes_o = tf.concat([t["labels"][J] for t, (_, J) in zip(y_true, indices)], axis=0)
        # TODO: check if i need to change the device
        true_classes = tf.cast(tf.fill(pred_logits.shape[:2], super().num_classes), dtype=tf.int64) # device?
        true_classes = tf.tensor_scatter_nd_update(true_classes, tf.expand_dims(idx, axis=1), true_classes_o)

        # loss_ce = tf.nn.softmax_cross_entropy_with_logits(y_true, tf.transpose(pred_logits,(1,2)))
        # loss_ce = tf.nn.weighted_cross_entropy_with_logits(y_true, tf.transpose(pred_logits,(1,2)), super().empty_weight)
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes, logits=tf.transpose(pred_logits, [0, 2, 1]))
        weighted_loss_ce = tf.reduce_mean(tf.multiply(loss_ce, super().empty_weight))
        losses = {"loss_ce": weighted_loss_ce}
        return losses

class MaskLoss(Loss):
    def call(self, outputs, y_true, indices, num_masks):
        assert "pred_masks" in outputs

        pred_idx = super()._get_pred_permutation_idx(indices)
        true_idx = super()._get_true_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks[pred_idx]
        masks = [t["masks"] for t in y_true]

        # TODO: implement these next 2 lines + helpers
        true_masks, valid = nested_tensor_from_tensor_list(masks)
        true_masks = tf.cast(true_masks, pred_masks.dtype) # device?
        true_masks = true_masks[true_idx]

        pred_masks = tf.image.resize(pred_masks[..., tf.newaxis], true_masks.shape[-2:], method='bilinear', align_corners=False)[..., 0]
        pred_masks = tf.reshape(pred_masks[:, 0], -1)

        true_masks = tf.reshape(true_masks, -1)
        true_masks = tf.reshape(true_masks, pred_masks.shape)
        losses = {
            "loss_mask": FocalLoss().call(pred_masks, true_masks, num_masks),
            "loss_dice": DiceLoss().call(pred_masks, true_masks, num_masks)
        }
        return losses