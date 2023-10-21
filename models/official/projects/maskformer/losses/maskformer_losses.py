import tensorflow as tf
from official.vision.losses import focal_loss
from official.projects.detr.ops import matchers
import numpy as np
import os


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
        weighted_loss = super().call(y_true, y_pred) # [b, 100, h*w]
        
        loss = tf.math.reduce_mean(weighted_loss, axis=-1) # [b, 100]
        return loss

    def batch(self, y_true, y_pred):
        """
        y_true: (b_size, 100 (num objects), h*w)
        y_pred: (b_size, 100 (num objects), h*w)
        """
        hw = tf.cast(tf.shape(y_pred)[-1], dtype=tf.float32) #
        prob = tf.keras.activations.sigmoid(y_pred)
        focal_pos = tf.pow(1 - prob, self._gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred), logits=y_pred)
        focal_neg = tf.pow(prob, self._gamma) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_pred), logits=y_pred)
        
        if self._alpha >= 0:
            focal_pos = tf.cast(focal_pos * self._alpha, tf.float32)
            focal_neg = tf.cast(focal_neg * (1 - self._alpha), tf.float32)
        loss = tf.einsum("bnc,bmc->bnm",focal_pos,y_true) + tf.einsum(
        "bnc,bmc->bnm", focal_neg,(1 - y_true)
        )
        return loss/hw
    


class DiceLoss(tf.keras.losses.Loss):
   
    def __init__(self):
        super().__init__(reduction='none')

    def call(self, y_true, y_pred):
        """
        y_true: (b size, 100, h, w)
        """
        y_pred = tf.reshape(tf.keras.activations.sigmoid(y_pred), (tf.shape(y_pred)[0],tf.shape(y_pred)[1],-1))
        y_true = tf.reshape(y_true, (tf.shape(y_true)[0],tf.shape(y_true)[1],-1))
        numerator = 2 * tf.reduce_sum(y_pred * y_true, axis=-1)
        denominator = tf.reduce_sum(y_pred, axis=-1) + tf.reduce_sum(y_true, axis=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        
        return loss
    
    def batch(self, y_true, y_pred):
        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0],tf.shape(y_pred)[1],-1])
        y_pred = tf.cast(y_pred, tf.float32)
        numerator = 2 * tf.einsum("bnc,bmc->bnm", y_pred, y_true)
        op1 = tf.transpose(tf.reduce_sum(y_pred, axis=-1)[:, tf.newaxis], [0, 2, 1])	
        op2 = tf.transpose(tf.expand_dims(tf.reduce_sum(y_true, axis=-1), axis=-1), [0, 2, 1])
        denominator = op1 + op2 
        loss = 1 - (numerator + 1) / (denominator + 1)
        
        return loss

class Loss:
    def __init__(self, num_classes, matcher, eos_coef, cost_class = 1.0, cost_focal = 20.0, cost_dice = 1.0, ignore_label =0):
       
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.cost_class = cost_class
        self.cost_focal = cost_focal
        self.cost_dice = cost_dice
        self.ignore_label = ignore_label

    
    def memory_efficient_matcher(self, outputs, y_true):
        out_mask = outputs["pred_masks"]
        tgt_ids = tf.cast(y_true["unique_ids"], dtype=tf.int64)
        
        with tf.device(out_mask.device):
            tgt_mask = y_true["individual_masks"]
        
        cost_class = tf.gather(-tf.nn.softmax(outputs["pred_logits"]), tgt_ids, batch_dims=1, axis=-1)
        
        tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
        # reorder the tgt mask so that tf.image.resize can be applied
        tgt_mask = tf.transpose(tgt_mask, perm=[0,2,3,1]) # [b, h, w, 100]
        tgt_mask = tf.image.resize(tgt_mask, tf.shape(out_mask)[-2:], method='bilinear') # [b, h, w, 100]
        
        # undo the reordering done for tf.image.resize
        tgt_mask_permuted = tf.transpose(tgt_mask, perm=[0,3,1,2]) # [b, 100, h, w]
        out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], tf.shape(out_mask)[1], -1]) # [b, 100, h*w]
     
        tgt_mask_permuted = tf.reshape(tgt_mask_permuted, [tf.shape(tgt_mask_permuted)[0],tf.shape(tgt_mask_permuted)[1], -1]) # [b, 100, h*w]
        
        cost_focal = FocalLossMod().batch(tgt_mask_permuted, out_mask)
        cost_dice = DiceLoss().batch(tgt_mask_permuted, out_mask)

        # for type bfloat16, we need to cast the cost tensors to float32
        self.cost_class = tf.cast(self.cost_class, dtype=cost_class.dtype)
        self.cost_focal = tf.cast(self.cost_focal, dtype=cost_focal.dtype)
        self.cost_dice = tf.cast(self.cost_dice, dtype=cost_dice.dtype)

        total_cost = (
                self.cost_focal * cost_focal
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
        
        max_cost = (
                    self.cost_class * 0.0 +
                    self.cost_focal * 4.0 +
                    self.cost_dice * 0.0
                    )

        
        # Append highest cost where there are no objects : No object class == 0 (self.ignore_label)
        valid = tf.expand_dims(tf.cast(tf.not_equal(tgt_ids, self.ignore_label), dtype=total_cost.dtype), axis=1)
        total_cost = (1 - valid) * max_cost + valid * total_cost
        total_cost = tf.where(
        tf.logical_or(tf.math.is_nan(total_cost), tf.math.is_inf(total_cost)),
        max_cost * tf.ones_like(total_cost, dtype=total_cost.dtype),
        total_cost)
       
        _, inds = matchers.hungarian_matching(total_cost)
        indices = tf.stop_gradient(inds)
       
        return indices

    def get_loss(self, outputs, y_true, indices):
      
        target_index = tf.math.argmax(indices, axis=1) #[batchsize, 100]
        target_labels = y_true["unique_ids"] #[batchsize, num_gt_objects]
        cls_outputs = outputs["pred_logits"] # [batchsize, num_queries, num_classes] [1,100,134]
        cls_masks = outputs["pred_masks"]# [batchsize,num_queries, h, w]
        individual_masks = y_true["individual_masks"] # [batchsize, num_gt_objects, h, w,]
        
        
        
        cls_assigned = tf.gather(cls_outputs, target_index, batch_dims=1, axis=1)
        mask_assigned = tf.gather(cls_masks, target_index, batch_dims=1, axis=1)

        target_classes = tf.cast(target_labels, dtype=tf.int32)
        background = tf.equal(target_classes, self.ignore_label) 
        
        num_masks = tf.reduce_sum(tf.cast(tf.logical_not(background), tf.float32), axis=-1)
        
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_classes, logits=cls_assigned)
        
        cls_loss =  tf.where(background, self.eos_coef * xentropy, xentropy)
        
        cls_weights = tf.where(background, self.eos_coef * tf.ones_like(cls_loss), tf.ones_like(cls_loss))
        
        num_masks_per_replica = tf.reduce_sum(num_masks)
        
        cls_weights_per_replica = tf.reduce_sum(cls_weights)
        
        replica_context = tf.distribute.get_replica_context()
        num_masks_sum, cls_weights_sum = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,[num_masks_per_replica, cls_weights_per_replica])
        
        # Final losses
        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss), cls_weights_sum)
        
        out_mask = mask_assigned
        tgt_mask = individual_masks

        tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)

        # transpose to make it compatible with tf.image.resize
        # Resize out mask instead of resizing gt mask
        out_mask =  tf.transpose(out_mask, perm=[0,2,3,1]) # [b, h, w, 100]
        out_mask = tf.image.resize(out_mask, [tf.shape(tgt_mask)[2], tf.shape(tgt_mask)[3]], method='bilinear')

        # #undo the transpose 
        out_mask = tf.transpose(out_mask, perm=[0,3,1,2])
        # Calculate the focal loss and dice loss only where valid mask is true
        # FIXME : May be we don't need valid masks
        # out_mask =  tf.where(tf.expand_dims(tf.squeeze(valid_masks, -1), axis=1), out_mask, tf.zeros_like(out_mask))
        # tgt_mask =  tf.where(tf.expand_dims(tf.squeeze(valid_masks, -1), axis=1), tgt_mask, tf.zeros_like(tgt_mask))
        out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], tf.shape(out_mask)[1], -1]) # [b, 100, h*w]
        tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0],tf.shape(tgt_mask)[1], -1])
        
       
        focal_loss = FocalLossMod(alpha=0.25, gamma=2)(tgt_mask, out_mask)
        
        focal_loss_weighted = tf.where(background, tf.zeros_like(focal_loss), focal_loss)
        focal_loss_final = tf.math.divide_no_nan(tf.math.reduce_sum(tf.math.reduce_sum(focal_loss_weighted, axis=-1)), num_masks_sum)
            
        dice_loss = DiceLoss()(tgt_mask, out_mask)
        dice_loss_weighted = tf.where(background, tf.zeros_like(dice_loss), dice_loss)
        dice_loss_final = tf.math.divide_no_nan(tf.math.reduce_sum(tf.math.reduce_sum(dice_loss_weighted, axis=-1)), num_masks_sum)
        
        return cls_loss, focal_loss_final, dice_loss_final
    
    def __call__(self, outputs, y_true):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             y_true: list of dicts, such that len(y_true) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        # NOTE : Change shape of pred and target masks to [batch_size, num_queries, h, w]
        outputs_without_aux["pred_masks"] = tf.transpose(outputs["pred_masks"], perm=[0,3,1,2])
        y_true["individual_masks"] = tf.squeeze(y_true["individual_masks"], axis=-1)
        indices = self.memory_efficient_matcher(outputs_without_aux, y_true) # (batchsize, num_queries, num_queries)
            
        losses = {}

        outputs["pred_masks"] = tf.transpose(outputs["pred_masks"], perm=[0,3,1,2])
        cls_loss_final, focal_loss_final, dice_loss_final = self.get_loss(outputs, y_true, indices)
        
        losses.update({"loss_ce": self.cost_class*cls_loss_final,
                    "loss_focal": self.cost_focal*focal_loss_final,
                    "loss_dice": self.cost_dice*dice_loss_final})
        
        # FIXME : check if we need to add aux_outputs
        # if "aux_outputs" in outputs and outputs["aux_outputs"] is not None:
        #     for i, aux_outputs in enumerate(outputs["aux_outputs"]):
        #         indices = self.memory_efficient_matcher(aux_outputs, y_true)
        #         # for loss in self.losses:
        #         cls_loss_, focal_loss_, dice_loss_ = self.get_loss(batch_size, aux_outputs, y_true, indices)
                
        #         l_dict = {"loss_ce" + f"_{i}": self.cost_class * cls_loss_,
        #                    "loss_focal" + f"_{i}": self.cost_focal *focal_loss_,
        #                    "loss_dice" + f"_{i}": self.cost_dice * dice_loss_}
        #         losses.update(l_dict)
        
        return losses
    
