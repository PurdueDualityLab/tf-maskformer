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
    def call(self, y_true, y_pred, num_masks):
        y_pred = tf.reshape(tf.keras.activations.sigmoid(y_pred), [tf.shape(y_pred)[0], -1])
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
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
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, cost_class=1, cost_focal=1, cost_dice=1):
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
        idx = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
        for i, (pred,_) in enumerate(indices):
            idx = idx.write(i, [i, pred.numpy()[0]])
        return idx.stack()
    
    def _get_true_permutation_idx(self, indices):
        idx = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
        for i, (_,true) in enumerate(indices):
            idx = idx.write(i, [i, true.numpy()[0]])
        return idx.stack()
    
    def get_classification_loss(self, outputs, y_true, indices, num_masks):
        assert "pred_logits" in outputs

        pred_logits = outputs["pred_logits"]
        idx = self._get_pred_permutation_idx(indices)
        true_classes_o = tf.concat([t["labels"][int(J)] for t, (_, J) in zip(y_true, indices)], axis=0)
        with tf.device(pred_logits.device):
            true_classes = tf.cast(tf.fill(pred_logits.shape[:2], self.num_classes), dtype=tf.int64) # device?
        true_classes = tf.tensor_scatter_nd_update(true_classes, idx, [true_classes_o])

        #TODO: unsure which TensorFlow function matches the PyTorch counterpart
        loss_ce = tf.nn.softmax_cross_entropy_with_logits(labels=true_classes, logits=tf.transpose(pred_logits, perm=[0,2,1]))
        weighted_loss_ce = tf.reduce_mean(tf.multiply(loss_ce, self.empty_weight))
        losses = {"loss_ce": weighted_loss_ce}
        return losses

    def get_mask_loss(self, outputs, y_true, indices, num_masks):
        assert "pred_masks" in outputs

        pred_idx = self._get_pred_permutation_idx(indices)
        true_idx = self._get_true_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]

        temp = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        for i, (batch, idx) in enumerate(pred_idx):
            temp = temp.write(i, [tf.gather(tf.gather(pred_masks, batch), idx)])
        pred_masks = tf.squeeze(temp.stack(), [1])
        masks = [t["masks"] for t in y_true]

        true_masks, valid = Utils.nested_tensor_from_tensor_list(masks).decompose()
        with tf.device(pred_masks.device):
            true_masks = tf.cast(true_masks, pred_masks.dtype)

        temp = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        for i, (batch, idx) in enumerate(true_idx):
            temp = temp.write(i, [tf.gather(tf.gather(true_masks, batch), idx)])
        true_masks = tf.squeeze(temp.stack(), [1])
        
        #TODO: the following function doesn't match the PyTorch counterpart
        pred_masks = tf.image.resize(pred_masks[:, None], tf.shape(true_masks)[-2:], method='bilinear')
        
        # uncomment to run the rest of the computations without resize
        # pred_masks = tf.random.uniform([1, 1, 750, 1333])

        pred_masks = tf.reshape(pred_masks[:, 0], [tf.shape(pred_masks)[0],-1])

        true_masks = tf.reshape(true_masks, [tf.shape(true_masks)[0],-1])
        true_masks = tf.reshape(true_masks, pred_masks.shape)
        losses = {
            "loss_mask": FocalLoss().call(pred_masks, true_masks, num_masks),
            "loss_dice": DiceLoss().call(pred_masks, true_masks, num_masks)
        }
        return losses

    def get_loss(self, loss, outputs, y_true, indices, num_masks):
        loss_map = {"labels": self.get_classification_loss, "masks": self.get_mask_loss}
        assert loss in loss_map
        return loss_map[loss](outputs, y_true, indices, num_masks)
    
    def compute_indices(self, outputs, y_true):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        masks = [v["masks"] for v in y_true]
        h_max = max([m.shape[1] for m in masks])
        w_max = max([m.shape[2] for m in masks])

        indices = list()
        for b in range(batch_size):
            out_prob = tf.nn.softmax(outputs["pred_logits"][b], axis=-1)
            out_mask = outputs["pred_masks"][b]
            tgt_ids = y_true[b]["labels"]

            with tf.device(out_mask.device):
                tgt_mask = y_true[b]["masks"]

            cost_class = -tf.gather(out_prob, tgt_ids, axis=1)

            tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
            tgt_mask = tf.image.resize(tgt_mask[..., tf.newaxis], out_mask.shape[-2:], method='nearest')[..., 0]
            out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], -1])
            tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0], -1])

            cost_focal = FocalLoss().batch(tgt_mask, out_mask)
            cost_dice = DiceLoss().batch(tgt_mask, out_mask)
            cost = (
                self.cost_focal * cost_focal
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )

            _, idx = self.matcher(tf.reshape(cost, [1, num_queries, -1]))
            idx = tf.stop_gradient(idx)
            idx = tf.math.argmax(idx, axis=1)
            indices.append(linear_sum_assignment(cost))
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
        num_masks = tf.maximum(num_masks / tf.distribute.get_strategy().num_replicas_in_sync, 1.0).numpy()[0]

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
            max_size = tf.reduce_max([tf.shape(img) for img in tensor_list], axis=0)
            padded_imgs = []
            padded_masks = []

            for img in tensor_list:
                padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
                padded_img = tf.pad(img, [[0, padding[2]], [0, padding[1]], [0, padding[0]]], mode="CONSTANT")
                padded_imgs.append(padded_img)
                
                with tf.device(img.device): 
                    m = tf.zeros_like(img[0], dtype=tf.int32)
                padded_mask = tf.pad(m, [[0, padding[2]], [0, padding[1]]], mode="CONSTANT", constant_values=1)
                padded_masks.append(tf.cast(padded_mask, tf.bool))
            
            tensor = tf.stack(padded_imgs)
            mask = tf.stack(padded_masks)
        else:
            raise ValueError("not supported")
        return Utils.NestedTensor(tensor, mask)
    
    def is_dist_avail_and_initialized():
        if not tf.distribute.has_strategy():
            return False
        if not tf.distribute.in_cross_replica_context():
            return False
        return True