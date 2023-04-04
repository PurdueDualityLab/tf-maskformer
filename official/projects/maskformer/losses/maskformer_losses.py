import tensorflow as tf

from official.vision.losses import focal_loss
from scipy.optimize import linear_sum_assignment

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

    def batch(self, y_true, y_pred):
        hw = tf.cast(tf.shape(y_pred)[1], dtype=tf.float32)

        positive_label_mask = tf.equal(y_true, 1.0)
        prob = tf.keras.activations.sigmoid(y_pred)

        focal_pos = tf.pow(1 - prob, self._gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred), logits=y_pred)
        focal_neg = tf.pow(prob, self._gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_pred), logits=y_pred)
        if self._alpha >= 0:
            focal_pos = focal_pos * self._alpha
            focal_neg = focal_neg * (1 - self._alpha)
        loss = tf.linalg.matmul(focal_pos, y_true, transpose_b=True) + tf.linalg.matmul(focal_neg, (1-y_true), transpose_b=True)
        return loss / hw


class DiceLoss(tf.keras.losses.Loss):
    # TODO: figure out dice loss stuff
    def call(self, y_true, y_pred, num_masks):
        y_pred = tf.reshape(tf.keras.activations.sigmoid(y_pred), -1)
        y_true = tf.reshape(y_true, -1)
        numerator = 2 * tf.reduce_sum(y_pred * y_true, axis=-1)
        denominator = tf.reduce_sum(y_pred, axis=-1) + tf.reduce_sum(y_true, axis=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return tf.reduce_sum(loss) / num_masks
    
    def batch(self, y_true, y_pred):
        y_pred = tf.keras.activations.sigmoid(y_pred)
        numerator = 2 * tf.linalg.matmul(y_pred, y_true, transpose_b=True)
        denominator = tf.reduce_sum(y_pred, axis=-1)[:, tf.newaxis] + tf.reduce_sum(y_true, axis=-1)[tf.newaxis, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

class Loss(tf.keras.losses.Loss):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, cost_class = 1, cost_focal = 1, cost_dice = 1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = tf.ones(self.num_classes + 1)
        empty_weight = tf.tensor_scatter_nd_update(empty_weight, [[self.num_classes]], [self.eos_coef])
        self.empty_weight = tf.Variable(empty_weight, trainable=False, name='empty_weight')   
        self.cost_class = cost_class
        self.cost_focal = cost_focal
        self.cost_dice = cost_dice

    def _get_pred_permutation_idx(self, indices):
        batch_idx = tf.concat([tf.fill(tf.shape(pred),i) for i, (pred,_) in enumerate(indices)], axis=0)
        pred_idx = tf.concat([pred for (pred,_) in indices], axis=0)
        return batch_idx, pred_idx
    
    def _get_true_permutation_idx(self, indices):
        batch_idx = tf.concat([tf.fill(true,i) for i, (_,true) in enumerate(indices)], axis=0)
        true_idx = tf.concat([true for (_,true) in indices], axis=0)
        return batch_idx, true_idx
    
    def get_classification_loss(self, outputs, y_true, indices, num_masks):
        assert "pred_logits" in outputs

        pred_logits = outputs["pred_logits"]
        idx = self._get_pred_permutation_idx(indices)
        true_classes_o = tf.concat([t["labels"][int(J)] for t, (_, J) in zip(y_true, indices)], axis=0)

        with tf.device(pred_logits.device):
            true_classes = tf.cast(tf.fill(pred_logits.shape[:2], self.num_classes), dtype=tf.int64) # device?
        
        true_classes = tf.tensor_scatter_nd_update(true_classes, idx, [true_classes_o])

        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes, logits=pred_logits)
        weighted_loss_ce = tf.reduce_mean(tf.multiply(loss_ce, self.empty_weight))
        losses = {"loss_ce": weighted_loss_ce}
        return losses

    def get_mask_loss(self, outputs, y_true, indices, num_masks):
        assert "pred_masks" in outputs

        pred_idx = self._get_pred_permutation_idx(indices)
        true_idx = self._get_true_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]
        pred_masks = tf.gather(pred_masks, pred_idx)
        masks = [t["masks"] for t in y_true]

        true_masks, valid = Utils.nested_tensor_from_tensor_list(masks).decompose()
        true_masks = true_masks.to(pred_masks)
        true_masks = true_masks[true_idx]

        pred_masks = tf.image.resize(pred_masks[:, tf.newaxis], true_masks.shape[-2:], method='bilinear', align_corners=False)[:, 0]
        pred_masks = tf.reshape(pred_masks[:, 0], -1)

        true_masks = tf.reshape(true_masks, -1)
        true_masks = tf.reshape(true_masks, pred_masks.shape)
        losses = {
            "loss_mask": FocalLoss()(pred_masks, true_masks, num_masks),
            "loss_dice": DiceLoss()(pred_masks, true_masks, num_masks)
        }
        return losses

    def get_loss(self, loss, outputs, y_true, indices, num_masks):
        loss_map = {"labels": self.get_classification_loss, "masks": self.get_mask_loss}
        assert loss in loss_map
        return loss_map[loss](outputs, y_true, indices, num_masks)
    
    def compute_indices(self, outputs, y_true):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        # print(batch_size, num_queries)

        masks = [v["masks"] for v in y_true]
        h_max = max([m.shape[1] for m in masks])
        w_max = max([m.shape[2] for m in masks])

        # print(masks, h_max, w_max)

        indices = list()
        # print("bs", batch_size)
        for b in range(batch_size):
            out_prob = tf.nn.softmax(outputs["pred_logits"][b], axis=-1)
            out_mask = outputs["pred_masks"][b]
            # print(tf.shape(out_prob), tf.shape(out_mask))
            tgt_ids = y_true[b]["labels"]

            with tf.device(out_mask.device):
                tgt_mask = y_true[b]["masks"]

            cost_class = -tf.gather(out_prob, tgt_ids, axis=1)
            # print(cost_class)

            tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
            # print(out_mask.shape)
            tgt_mask = tf.image.resize(tgt_mask[..., tf.newaxis], out_mask.shape[-2:], method='nearest')[..., 0]
            # print(tf.shape(tgt_mask))
            out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], -1])
            tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0], -1])
            # print("outmask after flatten", tf.shape(out_mask))
            # print("tgtmask after flatten", tf.shape(tgt_mask))
            cost_focal = FocalLoss().batch(tgt_mask, out_mask)
            cost_dice = DiceLoss().batch(tgt_mask, out_mask)
            # print(cost_focal)
            cost = (
                self.cost_focal * cost_focal
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )

            # print(cost)

            # cost = tf.reshape(cost, [num_queries, -1])
            # print(tf.reshape(cost, [1, num_queries, -1]))

            _, idx = self.matcher(tf.reshape(cost, [1, num_queries, -1]))
            idx = tf.stop_gradient(idx)
            # row = tf.
            idx = tf.math.argmax(idx, axis=1)
            # print("row\n", row)
            # print("col\n", col)
            # print("lsa\n", linear_sum_assignment(cost))
            # idx = linear_sum_assignment(cost)
            # print("idx shape", tf.shape(idx))
            # indices.append(linear_sum_assignment(cost))
            # print(idx)
            # indices.append((list(range(len(idx))), idx))
            indices.append(linear_sum_assignment(cost))
        # print(indices)
        return [
            (tf.convert_to_tensor(i, dtype=tf.int32), tf.convert_to_tensor(j, dtype=tf.int32))
            for i, j in indices
        ]

    def call(self, outputs, y_true):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             y_true: list of dicts, such that len(targets) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.compute_indices(outputs_without_aux, y_true)
        num_masks = sum(len(t["labels"]) for t in y_true)
        num_masks = tf.convert_to_tensor([num_masks], dtype=tf.float64) # device?
        
        if Utils.is_dist_avail_and_initialized():
            num_masks = tf.distribute.get_strategy().reduce(tf.distribute.ReduceOp.SUM, num_masks, axis=None)
        num_masks = tf.maximum(num_masks / tf.distribute.get_strategy().num_replicas_in_sync, 1.0)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, y_true, indices, num_masks))
        
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.compute_indices(aux_outputs, y_true)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, y_true, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses


class ClassificationLoss():
    def call(self, outputs, y_true, indices, num_masks):
        assert "pred_logits" in outputs

        pred_logits = outputs["pred_logits"]

        idx = Loss._get_pred_permutation_idx(indices)
        true_classes_o = tf.concat([t["labels"][J] for t, (_, J) in zip(y_true, indices)], axis=0)

        with tf.device(pred_logits.device):
            true_classes = tf.cast(tf.fill(pred_logits.shape[:2], super().num_classes), dtype=tf.int64) # device?
        true_classes = tf.tensor_scatter_nd_update(true_classes, tf.expand_dims(idx, axis=1), true_classes_o)

        # loss_ce = tf.nn.softmax_cross_entropy_with_logits(y_true, tf.transpose(pred_logits,(1,2)))
        # loss_ce = tf.nn.weighted_cross_entropy_with_logits(y_true, tf.transpose(pred_logits,(1,2)), super().empty_weight)
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes, logits=tf.transpose(pred_logits, [0, 2, 1]))
        weighted_loss_ce = tf.reduce_mean(tf.multiply(loss_ce, super().empty_weight))
        losses = {"loss_ce": weighted_loss_ce}
        return losses

class MaskLoss():
    def call(self, outputs, y_true, indices, num_masks):
        assert "pred_masks" in outputs

        pred_idx = Loss._get_pred_permutation_idx(indices)
        true_idx = Loss._get_true_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]
        pred_masks = pred_masks[pred_idx]
        masks = [t["masks"] for t in y_true]

        true_masks, valid = Utils.nested_tensor_from_tensor_list(masks).decompose()
        # true_masks = tf.cast(true_masks, pred_masks.dtype) # device?
        true_masks = true_masks.to(pred_masks)
        true_masks = true_masks[true_idx]

        pred_masks = tf.image.resize(pred_masks[:, tf.newaxis], true_masks.shape[-2:], method='bilinear', align_corners=False)[:, 0]
        pred_masks = tf.reshape(pred_masks[:, 0], -1)

        true_masks = tf.reshape(true_masks, -1)
        true_masks = tf.reshape(true_masks, pred_masks.shape)
        losses = {
            "loss_mask": FocalLoss()(pred_masks, true_masks, num_masks),
            "loss_dice": DiceLoss()(pred_masks, true_masks, num_masks)
        }
        return losses

class Utils():
    def _max_by_axis(the_list):
        all_max = the_list[0]
        for sublist in the_list[1:]:
            for idx, item in enumerate(sublist):
                all_max[idx] = max(all_max[idx], item)
        return all_max

    class NestedTensor(object):
        def __init__(self, tensors, mask=None):
            self.tensors = tf.convert_to_tensor(tensors)
            self.mask = tf.convert_to_tensor(mask) if mask is not None else None

        def to(self, device):
            # type: (Device) -> NestedTensor # noqa
            with tf.device(device):
                cast_tensor = tf.identity(self.tensors)
                cast_mask = tf.identity(self.mask) if self.mask is not None else None
            return NestedTensor(cast_tensor, cast_mask)

        def decompose(self):
            return self.tensors, self.mask

        def __repr__(self):
            return str(self.tensors)
    
    def nested_tensor_from_tensor_list(tensor_list):
        if tf.rank(tensor_list[0]).numpy() == 3:
            # TODO: figure out ONNX stuff
            # if tf.executing_eagerly():
            #     return _onnx_nested_tensor_from_tensor_list(tensor_list)
            
            max_size = tf.reduce_max([tf.shape(img) for img in tensor_list], axis=0)
            batch_shape = tf.concat([[len(tensor_list)], max_size], axis=0)
            batch_size, num_channels, height, width = batch_shape
            with tf.device(tensor_list[0].device):
                tensor = tf.zeros(batch_shape, dtype=tensor_list[0].dtype)
                mask = tf.ones((batch_size, height, width), dtype=tf.bool)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].assign(img)
                m[:img.shape[1], :img.shape[2]].assign(False)
        else:
            raise ValueError("not supported")
        return NestedTensor(tensor, mask)
    
    def is_dist_avail_and_initialized():
        if not tf.distribute.has_strategy():
            return False
        if not tf.distribute.in_cross_replica_context():
            return False
        return True