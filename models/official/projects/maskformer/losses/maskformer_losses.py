import tensorflow as tf
from official.vision.losses import focal_loss
from official.projects.detr.ops import matchers
from loguru import logger
# tf.compat.v1.enable_eager_execution()

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
        # self.background_indices = background_indices

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
        # background_indices = tf.expand_dims(self.background_indices, axis=0)
        weighted_loss = super().call(y_true, y_pred)
        # print("weighted loss :", weighted_loss.shape) #(1, 100, 442368)
        # mean over all pixels
        loss = tf.math.reduce_mean(weighted_loss, axis=-1)
        # logger.debug("loss shape: {}".format(loss.shape))
        # logger.debug("loss: {}".format(loss))
        return loss

    def batch(self, y_true, y_pred):
        """
        y_true: (b_size, 100 (num objects), h*w)
        y_pred: (b_size, 100 (num objects), h*w)
        """
        hw = tf.cast(tf.shape(y_pred)[-1], dtype=tf.float32) #[100, h, w]
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
        y_true: (b size, 100, h*w)
        """
        # FIXME : cast y_true to bfloat16 for loss calculation
        y_true = tf.cast(y_true, tf.bfloat16)
        y_pred = tf.reshape(tf.keras.activations.sigmoid(y_pred), (y_pred.shape[0],y_pred.shape[1],-1))
        y_true = tf.reshape(y_true, (y_true.shape[0],tf.shape(y_true)[1],-1))
        numerator = 2 * tf.reduce_sum(y_pred * y_true, axis=-1)
        denominator = tf.reduce_sum(y_pred, axis=-1) + tf.reduce_sum(y_true, axis=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        
        return loss
    
    def batch(self, y_true, y_pred):
        # y_pred = tf.keras.activations.sigmoid(y_pred)
        y_pred = tf.sigmoid(y_pred)
        y_pred = tf.reshape(y_pred, [y_pred.shape[0], -1, y_pred.shape[1]])
        y_pred = tf.transpose(y_pred, [0, 2, 1])
        y_pred = tf.cast(y_pred, tf.float32)
        numerator = 2 * tf.einsum("bnc,bmc->bnm", y_pred, y_true)
        denominator = tf.reduce_sum(y_pred, axis=-1)[:, tf.newaxis] + tf.expand_dims(tf.reduce_sum(y_true, axis=-1), axis=-1)
       

        loss = 1 - (numerator + 1) / (denominator + 1)
        
        return loss

class Loss:
    def __init__(self, num_classes, matcher, eos_coef, cost_class = 1, cost_focal = 1, cost_dice = 1):
       
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.cost_class = cost_class
        self.cost_focal = cost_focal
        self.cost_dice = cost_dice

    
    def memory_efficient_matcher(self, outputs, y_true):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        out_mask = outputs["pred_masks"]
        out_mask = tf.transpose(out_mask, perm=[0,3,1,2])
        
        tgt_ids = tf.cast(y_true["unique_ids"], dtype=tf.int64)
        
        with tf.device(out_mask.device):
            tgt_mask = y_true["individual_masks"]
        tgt_mask = tf.transpose(tgt_mask, perm=[0,2,3,1,4]) # [b, 100, h, w, 1]
        cost_class = tf.gather(-tf.nn.softmax(outputs["pred_logits"]), tgt_ids, batch_dims=1, axis=-1)
        tgt_mask = tf.squeeze(tf.cast(tgt_mask, dtype=tf.float32),axis=-1)
        
        tgt_mask = tf.image.resize(tgt_mask, out_mask.shape[-2:], method='bilinear') # [b, h, w, 100]
        
        out_mask = tf.transpose(out_mask, perm=[0,2,3,1]) # [b, h, w, 100]
        
        out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], tf.shape(out_mask)[-1], -1]) # [b, 100, h*w]
        tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0],tf.shape(tgt_mask)[-1], -1]) # [b, 100, h*w]
        
        cost_focal = FocalLossMod().batch(tgt_mask, out_mask)
        cost_dice = DiceLoss().batch(tgt_mask, out_mask)

        # FIXME : Cast the loss tensors to tf.bfloat16 
        cost_focal = tf.cast(cost_focal, dtype=tf.bfloat16)
        cost_dice = tf.cast(cost_dice, dtype=tf.bfloat16)

        total_cost = (
                self.cost_focal * cost_focal
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
        
        _, inds = matchers.hungarian_matching(total_cost)
        return inds

    
    
    def get_loss(self, batch_size, outputs, y_true, indices):
        
        target_index = tf.math.argmax(indices, axis=1) #[batchsize, 100]
        target_labels = y_true["unique_ids"] #[batchsize, num_gt_objects]
        cls_outputs = outputs["pred_logits"] # [batchsize, num_queries, num_classes] [1,100,134]
        cls_masks = outputs["pred_masks"]# [batchsize, h, w, num_queries]
        individual_masks = y_true["individual_masks"] # [batchsize, num_gt_objects, h, w, 1]

       

        cls_assigned = tf.gather(cls_outputs, target_index, batch_dims=1, axis=1)
        mask_assigned = tf.gather(cls_masks, target_index, batch_dims=1, axis=1)

        target_classes = tf.cast(target_labels, dtype=tf.int32)
        background = tf.equal(target_classes, 0) # Pytorch padds 133 class number where classes are background
        
        num_masks = tf.reduce_sum(tf.cast(tf.logical_not(background), tf.float32), axis=-1)
        ########################################################################################################  
        # TODO: check if we need this!
        # if Utils.is_dist_avail_and_initialized():
        #     num_masks = tf.distribute.get_strategy().reduce(tf.distribute.ReduceOp.SUM, num_masks, axis=None)
        # num_masks = tf.maximum(num_masks / tf.distribute.get_strategy().num_replicas_in_sync, 1.0)
        #########################################################################################################
        
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_classes, logits=cls_assigned)
        cls_loss = self.cost_class * tf.where(background, 0.1 * xentropy, xentropy)
        cls_weights = tf.where(background, 0.1 * tf.ones_like(cls_loss), tf.ones_like(cls_loss))
    
        num_masks_per_replica = tf.reduce_sum(num_masks)
        cls_weights_per_replica = tf.reduce_sum(cls_weights)
        replica_context = tf.distribute.get_replica_context()
        num_masks_sum, cls_weights_sum = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                                                    [num_masks_per_replica, cls_weights_per_replica])
       
        # Final losses
        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss), cls_weights_sum)
        losses = {'focal_loss' : 0.0, 'dice_loss': 0.0}
        
        
        out_mask = tf.transpose(cls_masks, perm=[0,3,1,2])
        with tf.device(out_mask.device):
            tgt_mask = individual_masks

        tgt_mask = tf.transpose(tgt_mask, perm=[0,2,3,1,4])
        tgt_mask = tf.squeeze(tf.cast(tgt_mask, dtype=tf.float32),axis=-1)
        tgt_mask = tf.image.resize(tgt_mask, out_mask.shape[-2:], method='bilinear') # [b, h, w, 100]
        out_mask = tf.transpose(out_mask, perm=[0,2,3,1]) # [b, h, w, 100]
        
        out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], tf.shape(out_mask)[-1], -1]) # [b, 100, h*w]
        tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0],tf.shape(tgt_mask)[-1], -1])
        focal_loss = FocalLossMod()(tgt_mask, out_mask)
        dice_loss = DiceLoss()(tgt_mask, out_mask)
        
    
        losses['focal_loss'] = focal_loss
        losses['dice_loss'] = dice_loss
        background_new = background

        focal_loss_weighted = tf.where(background_new, tf.zeros_like(focal_loss), focal_loss)
        dice_loss_weighted = tf.where(background_new, tf.zeros_like(dice_loss), dice_loss)
        
        focal_loss_final = tf.math.divide_no_nan(tf.math.reduce_sum(focal_loss_weighted), num_masks_sum)
        # FIXME: check if we need to cast dice_loss_weighted to tf.bfloat16
        dice_loss_final = tf.math.divide_no_nan(tf.math.reduce_sum(dice_loss_weighted), tf.cast(num_masks_sum, tf.bfloat16))

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
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        indices = self.memory_efficient_matcher(outputs_without_aux, y_true) # (batchsize, num_queries, num_queries)
              
        losses = {}
      
        cls_loss_final, focal_loss_final, dice_loss_final = self.get_loss(batch_size, outputs, y_true, indices)
        
        losses.update({"loss_ce": self.cost_class*cls_loss_final,
                    "loss_focal": self.cost_focal*focal_loss_final,
                    "loss_dice": self.cost_dice*dice_loss_final})
        
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
    
