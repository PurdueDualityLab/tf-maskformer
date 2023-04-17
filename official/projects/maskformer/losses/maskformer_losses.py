import tensorflow as tf
from official.vision.losses import focal_loss
import numpy as np
from official.projects.detr.ops import matchers
from scipy.optimize import linear_sum_assignment


from loguru import logger

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
        # logger.critical(f"y_pred is {y_pred}")
        # positive_label_mask = tf.equal(y_true, 1.0)
        prob = tf.keras.activations.sigmoid(y_pred)
        # logger.critical(f"gama is {self._gamma}")
        # logger.critical(f"alpha is {self._alpha}")
        # logger.critical(f"prob is {prob}")
        # logger.critical(f"BCE is {tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred), logits=y_pred)}")

        focal_pos = tf.pow(1 - prob, self._gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred), logits=y_pred)
        focal_neg = tf.pow(prob, self._gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_pred), logits=y_pred)
        # logger.critical(f"focal_pos is {focal_pos}")
        # logger.critical(f"focal_neg is {focal_neg}")
        if self._alpha >= 0:
            focal_pos = focal_pos * self._alpha
            focal_neg = focal_neg * (1 - self._alpha)
        loss = tf.einsum("nc,mc->nm", focal_pos, y_true) + tf.einsum(
        "nc,mc->nm", focal_neg, (1 - y_true)
        )
        
        # loss = tf.linalg.matmul(focal_pos, y_true, transpose_b=True) + tf.linalg.matmul(focal_neg, (1-y_true), transpose_b=True)
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
        # y_pred = tf.keras.activations.sigmoid(y_pred)
        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[1]])

        # logger.debug(f"y_pred is {y_pred}")
        # logger.debug(f"y_pred shape is {y_pred.shape}")
        numerator = 2 * tf.einsum("nc,mc->nm", y_pred, y_true)
        # logger.debug(f"numerator type is {type(numerator)}")

        # numerator = 2 * tf.linalg.matmul(y_pred, y_true, transpose_b=True)
        denominator = tf.reduce_sum(y_pred, axis=-1)[:, tf.newaxis] + tf.reduce_sum(y_true, axis=-1)[tf.newaxis, :]
        # logger.critical(f"numerator is {numerator}")
        # logger.critical(f"denominator is {denominator}") #TODO: ALMOST THE SAME
        loss = 1 - (numerator + 1) / (denominator + 1)
        # logger.critical(f"loss is {loss}")
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

    def call(self, outputs, y_true):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             y_true: list of dicts, such that len(y_true) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        # print("Targets masks :", y_true[0]["masks"][0,0,0:10])
        # print("Targets labels :", y_true[0]["labels"])
        # print("Output logits :", outputs["pred_logits"][0])
        # print("Output masks :", outputs["pred_masks"][0,0,0:10])
        indices = self.memory_efficient_matcher(outputs_without_aux, y_true) # (batchsize, num_queries, num_queries)
        logger.debug("[INFO] Indices shape :", indices.shape)
        
        target_index = tf.math.argmax(indices, axis=1)
        logger.error(f"target_index is {target_index}")
        cls_outputs = outputs["pred_logits"]
        cls_masks = outputs["pred_masks"]
        # create  batched tensors for loss calculation with padded zeros
        batched_target_labels = []
        batched_target_masks = []
        for each_batch in y_true:
            num_zeros = cls_outputs.shape[1] - each_batch['labels'].shape[0]
            tgt_ids = tf.expand_dims(tf.concat([each_batch['labels'], tf.ones([num_zeros], dtype=tf.int64) * (self.num_classes)] ,0),0)
            batched_target_labels.append(tgt_ids)

            zeros_masks = tf.zeros([num_zeros, each_batch["masks"].shape[1], each_batch["masks"].shape[2]], dtype=tf.bool)
            tgt_mask = each_batch["masks"]
            tgt_mask = tf.expand_dims(tf.concat([tgt_mask, zeros_masks], 0),0)
            batched_target_masks.append(tgt_mask)

        target_classes = tf.concat(batched_target_labels, 0)
        target_masks = tf.concat(batched_target_masks, 0)
        # print(cls_outputs.shape, cls_masks.shape, target_classes.shape, target_masks.shape)
        
        cls_assigned = tf.gather(cls_outputs, target_index, batch_dims=1, axis=1)
        mask_assigned = tf.gather(cls_masks, target_index, batch_dims=1, axis=1)

        background = tf.equal(target_classes, 0)
        num_masks = tf.reduce_sum(tf.cast(tf.logical_not(background), tf.float32), axis=-1)
        logger.debug(f"[INFO] Classes assigned shape : {cls_assigned.shape}")
        logger.debug(f"[INFO] Classes Target shape : {target_classes.shape}")

        true_classes_torch = tf.convert_to_tensor(np.load("target_classes.npy")) 
        pred_logits_torch = tf.convert_to_tensor(np.load("src_logits.npy"))
        empty_weight_torch = tf.convert_to_tensor(np.load("empty_weight.npy"))
        # loss_ce = F.cross_entropy(pred_logits_torch.transpose(1, 2), true_classes_torch, empty_weight_torch)
        
        logger.debug(f"target_classes value is {target_classes}")
        logger.debug(f"true_classes_torch value is {true_classes_torch}")

        logger.debug(f"cls_assigned value is {cls_assigned}")
        logger.debug(f"pred_logits_torch value is {pred_logits_torch}")

        # print("[INFO] Classes assigned transposed shape :", tf.transpose(cls_assigned, perm=[0,2,1]).shape)
        # TODO : add background_cls_weight 
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes_torch, logits=pred_logits_torch)
        # logger.critical(f"[INFO] Xentropy : {xentropy}")
        # weighted_loss_ce = tf.math.divide_no_nan(tf.reduce_sum(xentropy), self.empty_weight)
        # weighted_loss_ce = tf.reduce_mean(xentropy * self.empty_weight)
        logger.critical(f"[INFO] Weighted loss ce : {weighted_loss_ce}")

        cls_loss = self.cost_class * tf.where(background, 0.1 * xentropy, xentropy)
        cls_weights = tf.where(background, 0.1 * tf.ones_like(cls_loss), tf.ones_like(cls_loss))
        # ###############################################################################################
        # pred_masks = outputs["pred_masks"]
        

        # pred_masks = tf.gather(pred_masks, pred_idx[1], axis=1)[0]
        
        # masks = [t["masks"] for t in y_true]

        # true_masks, valid = Utils.nested_tensor_from_tensor_list(masks).decompose()
        # true_masks = true_masks.to(pred_masks)
        # true_masks = true_masks[true_idx]

        # pred_masks = tf.image.resize(pred_masks[:, tf.newaxis], true_masks.shape[-2:], method='bilinear', align_corners=False)[:, 0]
        # pred_masks = tf.reshape(pred_masks[:, 0], -1)

        # true_masks = tf.reshape(true_masks, -1)
        # true_masks = tf.reshape(true_masks, pred_masks.shape)

        # dice_loss = DiceLoss()(pred_masks, true_masks, num_masks)

        # focal_loss = FocalLoss()(pred_masks, true_masks, num_masks)
        ###############################################################################################
        num_masks_per_replica = tf.reduce_sum(num_masks)
        logger.debug(f"num_masks_per_replica is {num_masks_per_replica}")
        cls_weights_per_replica = tf.reduce_sum(cls_weights)
        replica_context = tf.distribute.get_replica_context()
        num_masks_sum, cls_weights_sum = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
        [num_masks_per_replica, cls_weights_per_replica])

        # Final losses
        weighted_loss_ce = tf.math.divide_no_nan(tf.reduce_sum(cls_loss), cls_weights_sum)
        logger.critical(f"[INFO] Weighted loss ce : {weighted_loss_ce}")
        logger.critical(f"[INFO] Pytorch Loss ce : 0.28803113102912903")

        # exit()
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
    
    # def _get_pred_permutation_idx(self, indices):
    #     batch_idx = tf.concat([tf.fill(tf.shape(pred),i) for i, (pred,_) in enumerate(indices)], axis=0)
    #     batch_idx = tf.cast(batch_idx, tf.int64)
    #     pred_idx = tf.concat([pred for (pred,_) in indices], axis=0)
    #     return batch_idx, pred_idx

    # def _get_true_permutation_idx(self, indices):
    #     batch_idx = tf.concat([tf.fill(true,i) for i, (_,true) in enumerate(indices)], axis=0)
    #     true_idx = tf.concat([true for (_,true) in indices], axis=0)
    #     return batch_idx, true_idx
    
    # def get_classification_loss(self, outputs, y_true, indices, num_masks):
    #     assert "pred_logits" in outputs
    #     pred_logits = outputs["pred_logits"]
    #     idx = self._get_pred_permutation_idx(indices)
    #     true_classes_o = tf.concat([tf.gather(t["labels"], J) for t, (_, J) in zip(y_true, indices)], axis=0)
    #     with tf.device(pred_logits.device):
    #         true_classes = tf.cast(tf.fill(pred_logits.shape[:2], self.num_classes), dtype=tf.int64) # device?
        
    #     true_classes = tf.tensor_scatter_nd_update(true_classes, idx, true_classes_o)
    #     loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes, logits=pred_logits)
    #     weighted_loss_ce = tf.reduce_mean(loss_ce * self.empty_weight)
    #     losses = {"loss_ce": weighted_loss_ce}
    #     print("[INFO] Losses CE :", weighted_loss_ce)
    #     return losses

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
            num_extra_masks = self.num_classes+1 - y_true[b]["masks"].shape[0] # TODO
            logger.debug(f"[INFO] Num extra masks : {num_extra_masks}")
            logger.debug(f"[INFO] out_mask shape : {out_mask.shape}")
            with tf.device(out_mask.device):
                zeros_masks = tf.zeros([num_extra_masks, y_true[b]["masks"].shape[1], y_true[b]["masks"].shape[2]], dtype=tf.bool)
                tgt_mask = y_true[b]["masks"]
                tgt_mask = tf.concat([tgt_mask, zeros_masks], 0)


            tgt_ids = tf.concat([tgt_ids, tf.zeros([num_extra_masks], dtype=tf.int64)],0) 
            
            cost_class = -tf.gather(out_prob, tgt_ids, axis=1)
            logger.debug(f"cost_class shape : {cost_class.shape}")
            tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
            
            tgt_mask = tf.image.resize(tgt_mask[..., tf.newaxis], out_mask.shape[-2:], method='nearest')[..., 0]
            
            out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], -1])
            tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0], -1])
            
            cost_focal = FocalLoss().batch(tgt_mask, out_mask)
            
            cost_dice = DiceLoss().batch(tgt_mask, out_mask)
           
            
            C = (
                self.cost_focal * cost_focal
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )

            
            C = tf.reshape(C, (1, num_queries, -1)) # Shape of C should be [batchsize, num_queries, num_queries]
            # print("Total Cost :", C)
            valid = tf.expand_dims(tf.cast(tf.not_equal(tgt_ids, 0), dtype=C.dtype), axis=1)
            max_cost = (self.cost_class * 0.0 + self.cost_dice * 4. + self.cost_focal * 0.0)
            logger.debug(f"C shape : {C.shape}")
            logger.debug(f"valid shape : {valid.shape}")
            total_cost = (1 - valid) * max_cost + valid * C

            total_cost = tf.where(tf.logical_or(tf.math.is_nan(total_cost), tf.math.is_inf(total_cost)),max_cost * tf.ones_like(total_cost, dtype=total_cost.dtype),
            total_cost)
            
            _, inds = matchers.hungarian_matching(total_cost) # ouptut is binary tensor
            # _, scipy_assignment = linear_sum_assignment(total_cost[0].numpy())
            # print("[INFO] Scipy Indices :", scipy_assignment)
            # print("[INFO] TF Assignements :", tf.math.argmax(inds, axis=1))
            # print("[INFO] TF Assignements :", np.where(inds.numpy()[0])[1])
            indices.append(tf.stop_gradient(inds))
        # for b in range(batch_size):
        #     out_prob = tf.nn.softmax(outputs["pred_logits"][b], axis=-1)
        #     out_mask = outputs["pred_masks"][b]
        #     tgt_ids = y_true[b]["labels"]
        #     # print("[INFO] target masks :",y_true[b]["masks"].shape)
        #     # print("[INFO] out_mask masks :",out_mask.shape)
        #     # print("[INFO] out_prob :",out_prob.shape) # (100, 134) (total_num_classes, number of predicted masks)
        #     # TODO : Pad masks with zeros to match the number of detected objects and the number of objects in the target 
        #     original_num_obj =  y_true[b]["masks"].shape[0]
        #     num_extra_masks = out_mask.shape[0] - y_true[b]["masks"].shape[0]
        #     # print("[INFO] Adding extra masks with zeros:", num_extra_masks)
            
        #     with tf.device(out_mask.device):
        #         zeros_masks = tf.zeros([num_extra_masks, y_true[b]["masks"].shape[1], y_true[b]["masks"].shape[2]], dtype=tf.bool)
        #         tgt_mask = y_true[b]["masks"]
        #         tgt_mask = tf.concat([tgt_mask, zeros_masks], 0)

        #     # print("[INFO] Padded target masks :",tgt_mask.shape)
            
        #     tgt_ids = tf.concat([tgt_ids, tf.zeros([num_extra_masks], dtype=tf.int64)],0) 
        #     # print("[INFO] New target ids :",tgt_ids.shape)
        #     cost_class = -tf.gather(out_prob, tgt_ids, axis=1)
            
        #     tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
            
        #     tgt_mask = tf.image.resize(tgt_mask[..., tf.newaxis], out_mask.shape[-2:], method='nearest')[..., 0]
            
        #     out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], -1])
        #     tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0], -1])
            
        #     cost_focal = FocalLoss().batch(tgt_mask, out_mask)
            
        #     cost_dice = DiceLoss().batch(tgt_mask, out_mask)
        #     # print("[INFO] Cost Class :", cost_class[:, 0:original_num_obj])
        #     # print("[INFO] Cost Focal :", cost_focal[:, 0:original_num_obj])
        #     # print("[INFO] Cost Dice :", cost_dice[:, 0:original_num_obj])

        #     C = (
        #         self.cost_focal * cost_focal
        #         + self.cost_class * cost_class
        #         + self.cost_dice * cost_dice
        #     )
        #     # [2, 100, 100]
        #     # [2, 100, 7]
        #     # print("Cost C : ", C)
        #     # exit()
        #     C = tf.reshape(C, (1, num_queries, -1)) # Shape of C should be [batchsize, num_queries, num_queries]
           
        
        #     # This code is taken from  - Reuse matcher from DETR
        #     # weights, assignment = matchers.hungarian_matching(C) 
        #     # indices.append(assignment)
        #     # using linear sum assignment from  scipy.optimize
        #     # indices.append(linear_sum_assignment(C))
        #     # print("[INFO] Indices from scipy :",linear_sum_assignment(C[0])[1])
        #     _, inds = matchers.hungarian_matching(C)
            
        #     print("[INFO] Indices from TF impelemtation:", inds)
        #     indices.append(tf.stop_gradient(inds))
        #     exit()
        
        return tf.concat([each_batch for each_batch in indices], 0)

    # def get_mask_loss(self, outputs, y_true, indices, num_masks):
    #     assert "pred_masks" in outputs

    #     pred_idx = self._get_pred_permutation_idx(indices)
    #     true_idx = self._get_true_permutation_idx(indices)
    #     pred_masks = outputs["pred_masks"]
        

    #     pred_masks = tf.gather(pred_masks, pred_idx[1], axis=1)[0]
        

    #     masks = [t["masks"] for t in y_true]

    #     true_masks, valid = Utils.nested_tensor_from_tensor_list(masks).decompose()
    #     true_masks = true_masks.to(pred_masks)
    #     true_masks = true_masks[true_idx]

    #     pred_masks = tf.image.resize(pred_masks[:, tf.newaxis], true_masks.shape[-2:], method='bilinear', align_corners=False)[:, 0]
    #     pred_masks = tf.reshape(pred_masks[:, 0], -1)

    #     true_masks = tf.reshape(true_masks, -1)
    #     true_masks = tf.reshape(true_masks, pred_masks.shape)
    #     losses = {
    #         "loss_mask": FocalLoss()(pred_masks, true_masks, num_masks),
    #         "loss_dice": DiceLoss()(pred_masks, true_masks, num_masks)
    #     }
    #     # logger.critical(f"losses is {losses}")
    #     return losses

    # def get_loss(self, loss, outputs, y_true, indices, num_masks):
    #     loss_map = {"labels": self.get_classification_loss, "masks": self.get_mask_loss}
    #     assert loss in loss_map
    #     return loss_map[loss](outputs, y_true, indices, num_masks)
    

# class ClassificationLoss():
#     def call(self, outputs, y_true, indices, num_masks):
#         assert "pred_logits" in outputs

#         pred_logits = outputs["pred_logits"]

#         idx = Loss._get_pred_permutation_idx(indices)
#         true_classes_o = tf.concat([t["labels"][J] for t, (_, J) in zip(y_true, indices)], axis=0)

#         with tf.device(pred_logits.device):
#             true_classes = tf.cast(tf.fill(pred_logits.shape[:2], super().num_classes), dtype=tf.int64) # device?
#         true_classes = tf.tensor_scatter_nd_update(true_classes, tf.expand_dims(idx, axis=1), true_classes_o)

#         # loss_ce = tf.nn.softmax_cross_entropy_with_logits(y_true, tf.transpose(pred_logits,(1,2)))
#         # loss_ce = tf.nn.weighted_cross_entropy_with_logits(y_true, tf.transpose(pred_logits,(1,2)), super().empty_weight)
#         loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes, logits=tf.transpose(pred_logits, [0, 2, 1]))
#         weighted_loss_ce = tf.reduce_mean(tf.multiply(loss_ce, super().empty_weight))
#         # logger.critical(f"weighted_loss_ce is {weighted_loss_ce}")
#         losses = {"loss_ce": weighted_loss_ce}
#         return losses

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
            max_size = Utils._max_by_axis([list(img.shape) for img in tensor_list])

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
        else:
            raise ValueError("not supported")
        
        return NestedTensor(tensor, mask)
    
    def is_dist_avail_and_initialized():
        if not tf.distribute.has_strategy():
            return False
        if not tf.distribute.in_cross_replica_context():
            return False
        return True