import tensorflow as tf
from official.vision.losses import focal_loss
import numpy as np
from official.projects.detr.ops import matchers
from scipy.optimize import linear_sum_assignment


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
        with tf.device(device):
            cast_tensor = tf.identity(self.tensors)
            cast_mask = tf.identity(self.mask) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
    # def is_dist_avail_and_initialized():
    #     if not tf.distribute.has_strategy():
    #         return False
    #     if not tf.distribute.in_cross_replica_context():
    #         return False
    #     return True
    
def nested_tensor_from_tensor_list(tensor_list):
    if tf.rank(tensor_list[0]).numpy() == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])

        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        with tf.device(device):
            tensor = tf.zeros(batch_shape, dtype=dtype)
            mask = tf.ones((b, h, w), dtype=tf.bool)

        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img = tf.Variable(pad_img)
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].assign(img)
            pad_img = tf.convert_to_tensor(pad_img)

            m = tf.Variable(m)
            false_tensor = tf.zeros((img.shape[1], img.shape[2]), dtype=tf.bool)
            m[:img.shape[1], :img.shape[2]].assign(false_tensor)
            m = tf.convert_to_tensor(m)
        return NestedTensor(tensor, mask)
    else:
        raise ValueError("not supported")


class FocalLossMod(focal_loss.FocalLoss):
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
        super().__init__(alpha, gamma, reduction='none')

    def call(self, y_true, y_pred):
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
        
        loss = tf.math.reduce_sum(weighted_loss,axis=-1)
        return loss

    def batch(self, y_true, y_pred):
        hw = tf.cast(tf.shape(y_pred)[1], dtype=tf.float32)
        prob = tf.keras.activations.sigmoid(y_pred)
        focal_pos = tf.pow(1 - prob, self._gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred), logits=y_pred)
        focal_neg = tf.pow(prob, self._gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_pred), logits=y_pred)
        
        if self._alpha >= 0:
            focal_pos = focal_pos * self._alpha
            focal_neg = focal_neg * (1 - self._alpha)
        loss = tf.einsum("nc,mc->nm", focal_pos, y_true) + tf.einsum(
        "nc,mc->nm", focal_neg, (1 - y_true)
        )
        
        return loss / hw


class DiceLoss(tf.keras.losses.Loss):
   
    def __init__(self):
        super().__init__(reduction='none')

    def call(self, y_true, y_pred):
        y_pred = tf.reshape(tf.keras.activations.sigmoid(y_pred), -1)
        y_true = tf.reshape(y_true, -1)
        numerator = 2 * tf.reduce_sum(y_pred * y_true, axis=-1)
        denominator = tf.reduce_sum(y_pred, axis=-1) + tf.reduce_sum(y_true, axis=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss
    
    def batch(self, y_true, y_pred):
        # y_pred = tf.keras.activations.sigmoid(y_pred)
        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[1]])
        numerator = 2 * tf.einsum("nc,mc->nm", y_pred, y_true)
        denominator = tf.reduce_sum(y_pred, axis=-1)[:, tf.newaxis] + tf.reduce_sum(y_true, axis=-1)[tf.newaxis, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        
        return loss

class Loss(tf.keras.losses.Loss):
    def __init__(self, num_classes, matcher, eos_coef, losses, cost_class = 1, cost_focal = 1, cost_dice = 1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        # self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = tf.ones(self.num_classes + 1)
        empty_weight = tf.tensor_scatter_nd_update(empty_weight, [[self.num_classes]], [self.eos_coef])
        self.empty_weight = tf.Variable(empty_weight, trainable=False, name='empty_weight')   
        self.empty_weight = tf.reshape(self.empty_weight, (self.num_classes + 1, 1))

        self.cost_class = cost_class
        self.cost_focal = cost_focal
        
        self.cost_dice = cost_dice

    def memory_efficient_matcher(self, outputs, y_true):
        batch_size, num_queries = outputs["pred_logits"].shape[:2] # Bsize, num_queries, num_preds
        
        masks = [v["masks"] for v in y_true]
        h_max = max([m.shape[1] for m in masks])
        w_max = max([m.shape[2] for m in masks])

        indices = list()
        for b in range(batch_size):
            out_prob = tf.nn.softmax(outputs["pred_logits"][b], axis=-1)
            out_mask = outputs["pred_masks"][b]
            tgt_ids = y_true[b]["labels"]
            num_extra_masks = out_mask.shape[0] - y_true[b]["masks"].shape[0]
            
            with tf.device(out_mask.device):
                tgt_mask = y_true[b]["masks"]
            cost_class = -tf.gather(out_prob, tgt_ids, axis=1)

            print("[INFO] target Masks :" , tgt_mask.shape)
            print("[INFO] Output Masks :" , out_mask.shape)
            tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
            tgt_mask = tf.image.resize(tgt_mask[..., tf.newaxis], out_mask.shape[-2:], method='nearest')[..., 0]
            out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], -1])
            tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0], -1])
            
            cost_focal = FocalLossMod().batch(tgt_mask, out_mask)
            cost_dice = DiceLoss().batch(tgt_mask, out_mask)
           
            C = (
                self.cost_focal * cost_focal
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )

            C = tf.reshape(C, (1, num_queries, -1)) # Shape of C should be [batchsize, num_queries, num_queries]
            print("[INFO] Cost matrix before padding :", C.shape)
            # Pad C to ensure compatibility with TF matcher
            C_padded = tf.concat([C, tf.zeros([1, num_queries, num_extra_masks], dtype=C.dtype)],-1)
            print("[INFO] C_padded :", C_padded.shape)
            _, inds = matchers.hungarian_matching(C_padded) # ouptut is binary tensor
            indices.append(tf.stop_gradient(inds))
        return tf.concat([each_batch for each_batch in indices], 0)
    
    def get_mask_loss(self, outputs, y_true, num_masks):
        
        pred_masks = outputs

        true_masks = y_true
        print("[INFO GET MASK] predmasks :", pred_masks.shape)
        print("[INFO GET MASK] true_masks :", true_masks.shape)
        # true_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        
        # true_masks = true_masks.to(pred_masks)
        # true_masks = true_masks[true_idx]

        pred_masks = tf.image.resize(pred_masks, true_masks.shape[-2:], method='bilinear')[:, 0]
        print("[INFO GET MASK] predmasks resize :", pred_masks.shape)
        pred_masks = tf.reshape(pred_masks[:, 0], -1)
        print("[INFO GET MASK] predmasks reshape :", pred_masks.shape)
        true_masks = tf.reshape(true_masks, -1)
        print("[INFO GET MASK] true_masks reshape :", true_masks.shape)
        true_masks = tf.reshape(true_masks, pred_masks.shape)
        print("[INFO GET MASK] true_masks reshape_1:", true_masks.shape)
        
        losses = {
            "loss_mask": FocalLossMod()(pred_masks, true_masks, num_masks),
            "loss_dice": DiceLoss()(pred_masks, true_masks, num_masks)
        }
        return losses
    
    def call(self, outputs, y_true):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             y_true: list of dicts, such that len(y_true) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        indices = self.memory_efficient_matcher(outputs_without_aux, y_true) # (batchsize, num_queries, num_queries)
        # print("[INFO] Predicted indices shape :", indices)
        print("[INFO] Predicted logits shape :", outputs["pred_logits"].shape)
        print("[INFO] Predicted masks shape :", outputs["pred_masks"].shape)
        print("[INFO] Target labels shape before padding:", y_true[0]['labels'].shape, y_true[1]['labels'].shape)
        print("[INFO] Target labels shape before padding:", y_true[0]['masks'].shape, y_true[1]['masks'].shape)
        
        target_index = tf.math.argmax(indices, axis=1) #[batchsize, 100]
        print("[INFO] Target Index :", target_index)
        cls_outputs = outputs["pred_logits"]
        cls_masks = outputs["pred_masks"]
        # create  batched tensors for loss calculation with padded zeros
        batched_target_labels = []
        batched_target_masks = []

        for each_batch in y_true:
            num_zeros = cls_outputs.shape[1] - each_batch['labels'].shape[0]
            tgt_ids = tf.expand_dims(tf.concat([each_batch['labels'], tf.ones([num_zeros], dtype=tf.int64)*self.num_classes],0),0)
            batched_target_labels.append(tgt_ids)

            zeros_masks = tf.zeros([num_zeros, each_batch["masks"].shape[1], each_batch["masks"].shape[2]], dtype=tf.bool)
            tgt_mask = each_batch["masks"]
            tgt_mask = tf.expand_dims(tf.concat([tgt_mask, zeros_masks], 0),0)
            batched_target_masks.append(tgt_mask)

        target_classes = tf.concat(batched_target_labels, 0)
        target_masks = tf.concat(batched_target_masks, 0)
        
        print("[INFO] Target labels shape after padding:", target_classes.shape)
        print("[INFO] Target masks shape after padding:", target_masks.shape)

        cls_assigned = tf.gather(cls_outputs, target_index, batch_dims=1, axis=1)
        mask_assigned = tf.gather(cls_masks, target_index, batch_dims=1, axis=1)

        background = tf.equal(target_classes, 0)
        
        num_masks = tf.reduce_sum(tf.cast(tf.logical_not(background), tf.float32), axis=-1)
        
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_classes, logits=cls_assigned)
        cls_loss = self.cost_class * tf.where(background, 0.1 * xentropy, xentropy)
        cls_weights = tf.where(background, 0.1 * tf.ones_like(cls_loss), tf.ones_like(cls_loss))
    
        
        num_masks_per_replica = tf.reduce_sum(num_masks)
        cls_weights_per_replica = tf.reduce_sum(cls_weights)
        replica_context = tf.distribute.get_replica_context()
        num_masks_sum, cls_weights_sum = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,[num_masks_per_replica, cls_weights_per_replica])

        # Final losses
        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss), cls_weights_sum)
        
        losses = {'focal_loss' : [], 'dice_loss': []}

        for b in range(batch_size):
            out_mask = mask_assigned[b]
            with tf.device(out_mask.device):
                tgt_mask = target_masks[b]
            tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
            tgt_mask = tf.image.resize(tgt_mask[..., tf.newaxis], out_mask.shape[-2:], method='nearest')[..., 0]
            out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], -1])
            tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0], -1])
           
            focal_loss_calc =  FocalLossMod()
            focal_loss = focal_loss_calc.batch(out_mask, tgt_mask)
            dice_loss =  DiceLoss().batch(out_mask, tgt_mask)
            
            losses['focal_loss'].append(tf.expand_dims(tf.reduce_mean(focal_loss,-1),0))
            losses['dice_loss'].append(tf.expand_dims(tf.reduce_mean(dice_loss, -1),0))
        
        batched_focal_loss = tf.concat(losses['focal_loss'], 0)
        print("[INFO] Batched focal loss shape :", batched_focal_loss.shape)
        batched_dice_loss = tf.concat(losses['dice_loss'], 0)
        
        focal_loss_weighted = self.cost_focal * tf.where(background, tf.zeros_like(batched_focal_loss), batched_focal_loss)
        dice_loss_weighted = self.cost_dice * tf.where(background, tf.zeros_like(batched_dice_loss), batched_dice_loss)
        
        focal_loss_final = tf.math.divide_no_nan(tf.reduce_sum(focal_loss_weighted), num_masks_sum)
        dice_loss_final = tf.math.divide_no_nan(tf.reduce_sum(dice_loss_weighted), num_masks_sum)

        print(f"[INFO] CE loss : {cls_loss}")
        print(f"[INFO] Focal loss : {focal_loss_final}")
        print(f"[INFO] Dice loss : {dice_loss_final}")
        exit()
        # print(self.get_mask_loss()

        # num_masks = sum(len(t["labels"]) for t in y_true)
        # num_masks = tf.convert_to_tensor([num_masks], dtype=tf.float64)
        
        # if Utils.is_dist_avail_and_initialized():
        #     num_masks = tf.distribute.get_strategy().reduce(tf.distribute.ReduceOp.SUM, num_masks, axis=None)

        # num_masks = tf.maximum(num_masks / tf.distribute.get_strategy().num_replicas_in_sync, 1.0)
        
        # losses = {}

        # for loss in self.losses:
        #     losses.update(self.get_loss(loss, outputs, y_true, indices, num_masks))
        # # losses.update({loss: self.get_loss(loss, outputs, y_true, indices, num_masks) for loss in self.losses})

        # if "aux_outputs" in outputs:
        #     for i, aux_outputs in enumerate(outputs["aux_outputs"]):
        #         indices = self.memory_efficient_matcher(aux_outputs, y_true)
        #         for loss in self.losses:
        #             l_dict = self.get_loss(loss, aux_outputs, y_true, indices, num_masks)
        #             l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
        #             losses.update(l_dict)
        
        # return losses
    

    

    

    # def get_loss(self, loss, outputs, y_true, indices, num_masks):
    #     loss_map = {"labels": self.get_classification_loss, "masks": self.get_mask_loss}
    #     assert loss in loss_map
    #     return loss_map[loss](outputs, y_true, indices, num_masks)
    


# class MaskLoss():
#     def call(self, outputs, y_true, indices, num_masks):
#         assert "pred_masks" in outputs

#         pred_idx = Loss._get_pred_permutation_idx(indices)
#         true_idx = Loss._get_true_permutation_idx(indices)
#         pred_masks = outputs["pred_masks"]
#         pred_masks = pred_masks[pred_idx]
#         masks = [t["masks"] for t in y_true]

#         true_masks, valid = Utils.nested_tensor_from_tensor_list(masks).decompose()
#         # true_masks = tf.cast(true_masks, pred_masks.dtype) # device?
#         true_masks = true_masks.to(pred_masks)
#         true_masks = true_masks[true_idx]

#         pred_masks = tf.image.resize(pred_masks[:, tf.newaxis], true_masks.shape[-2:], method='bilinear', align_corners=False)[:, 0]
#         pred_masks = tf.reshape(pred_masks[:, 0], -1)

#         true_masks = tf.reshape(true_masks, -1)
#         true_masks = tf.reshape(true_masks, pred_masks.shape)
#         losses = {
#             "loss_mask": FocalLoss()(pred_masks, true_masks, num_masks),
#             "loss_dice": DiceLoss()(pred_masks, true_masks, num_masks)
#         }
#         return losses

