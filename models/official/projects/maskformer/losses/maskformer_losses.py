import tensorflow as tf

from official.vision.losses import focal_loss
from scipy.optimize import linear_sum_assignment
from loguru import logger
import numpy as np

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
        logger.error(f"Focal loss: {loss}")
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
        self.empty_weight = tf.reshape(self.empty_weight, (self.num_classes + 1, 1))

        self.cost_class = cost_class
        self.cost_focal = cost_focal
        self.cost_dice = cost_dice

    def _get_pred_permutation_idx(self, indices):
        # TODO: high priority, fix this!!!
        logger.debug(f"indices is {indices}")
        batch_idx = tf.concat([tf.fill(tf.shape(pred),i) for i, (pred,_) in enumerate(indices)], axis=0)
        batch_idx = tf.cast(batch_idx, tf.int64)
        pred_idx = tf.concat([pred for (pred,_) in indices], axis=0)
        logger.debug(f"batch_idx is {batch_idx} dtype is {batch_idx.dtype}")
        logger.debug(f"pred_idx is {pred_idx} dtype is {pred_idx.dtype}")
        return batch_idx, pred_idx



    def _get_true_permutation_idx(self, indices):
        batch_idx = tf.concat([tf.fill(true,i) for i, (_,true) in enumerate(indices)], axis=0)
        true_idx = tf.concat([true for (_,true) in indices], axis=0)
        return batch_idx, true_idx
    
    def get_classification_loss(self, outputs, y_true, indices, num_masks):
        assert "pred_logits" in outputs
        pred_logits = outputs["pred_logits"]
        idx = self._get_pred_permutation_idx(indices)

        # for t, (_, J) in zip(y_true, indices):
        #     J = tf.reduce_sum(J)
        #     logger.critical(J)
            # logger.critical(t["labels"][J])
        # true_classes_o = tf.concat([t["labels"][J] for t, (_, J) in zip(y_true, indices)], axis=0)
        true_classes_o = tf.concat([tf.gather(t["labels"], J) for t, (_, J) in zip(y_true, indices)], axis=0)

        # logger.critical(true_classes_o)
        with tf.device(pred_logits.device):
            true_classes = tf.cast(tf.fill(pred_logits.shape[:2], self.num_classes), dtype=tf.int64) # device?
        
        # logger.critical(true_classes)
        # logger.critical(idx)
        # logger.critical(true_classes_o)
        # logger.critical(idx[1][:, tf.newaxis])
        # logger.critical(tf.squeeze(true_classes_o))
        logger.critical(idx[1][:, tf.newaxis])
        true_classes = tf.tensor_scatter_nd_update(true_classes[0], idx[1][:, tf.newaxis], tf.squeeze(true_classes_o))
        true_classes = tf.reshape(true_classes, (1, -1))

            # logger.debug(f"pred_logits.shape is {pred_logits.shape}")
        # pred_logits_t = tf.transpose(pred_logits, perm=[0, 2, 1])
        
        # logger.critical(true_classes.shape)
        # logger.critical(pred_logits.shape)
        # true_classes_tiled = tf.tile(true_classes, [1, pred_logits_t.shape[1], 1])
        # true_classes_tiled = tf.tile(true_classes, tf.reshape([pred_logits_t.shape[1], 1], [2]))
        
        # logger.critical(true_classes_tiled.shape)
        # true_classes_repeat = tf.repeat(true_classes_tiled, repeats=pred_logits_t.shape[0], axis=1)
        # logger.critical(true_classes_repeat.shape)
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes, logits=pred_logits)
        logger.critical(loss_ce.shape) # (1, 100)
        # logger.critical(f"tf loss_ce: {loss_ce}") # (1, 100)
        logger.critical(self.empty_weight.shape) # (134, )
        # logger.critical(self.empty_weight) # (134, )

        true_classes_torch = tf.convert_to_tensor(np.load("target_classes.npy")) 
        pred_logits_torch = tf.convert_to_tensor(np.load("src_logits.npy"))
        empty_weight_torch = tf.convert_to_tensor(np.load("empty_weight.npy"))
        
        
        logger.critical(true_classes)
        logger.critical(true_classes_torch)

        logger.critical(pred_logits)
        logger.critical(pred_logits_torch)
        
        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_classes_torch, logits=pred_logits_torch)
        # logger.critical(f"torch loss_ce: {loss_ce}")

        weighted_loss_ce = tf.reduce_mean(loss_ce * self.empty_weight)
        losses = {"loss_ce": weighted_loss_ce}
        logger.critical(weighted_loss_ce)
        return losses

    def get_mask_loss(self, outputs, y_true, indices, num_masks):
        assert "pred_masks" in outputs

        pred_idx = self._get_pred_permutation_idx(indices)
        true_idx = self._get_true_permutation_idx(indices)
        pred_masks = outputs["pred_masks"]
        logger.critical(f"pred_masks.shape is {pred_masks.shape}")
        logger.critical(f"pred_idx is {pred_idx}") #TODO: Doesn't match!!

        pred_masks = tf.gather(pred_masks, pred_idx[1], axis=1)[0]
        # logger.critical(f"pred_masks is {pred_masks}")

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
        logger.critical(f"losses is {losses}")
        return losses

    def get_loss(self, loss, outputs, y_true, indices, num_masks):
        loss_map = {"labels": self.get_classification_loss, "masks": self.get_mask_loss}
        assert loss in loss_map
        logger.debug(f"loss is {loss}")
        return loss_map[loss](outputs, y_true, indices, num_masks)
    
    def memory_efficient_matcher(self, outputs, y_true):
        # TODO: High priority!!! Debug facal loss and DiceLoss first!!!

        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        # print(batch_size, num_queries)
        logger.debug(f"targets masks shape is {y_true[0]['masks'].shape}")
        masks = [v["masks"] for v in y_true]
        h_max = max([m.shape[1] for m in masks])
        w_max = max([m.shape[2] for m in masks])
        logger.debug(f"mask is masks: {masks[0].shape}")
        logger.debug(f"mask padding size: {h_max}, {w_max}")
        # print(masks, h_max, w_max)

        indices = list()
        # print("bs", batch_size)
        for b in range(batch_size):
            # logger.debug(f"b is {b}")
            # logger.debug(f"batch_size is {batch_size}")
            # logger.error(f"out_prob is {outputs['pred_logits'][b]}")
            out_prob = tf.nn.softmax(outputs["pred_logits"][b], axis=-1)
            # logger.error(f"out_prob is {out_prob}")
            out_mask = outputs["pred_masks"][b]
            # print(tf.shape(out_prob), tf.shape(out_mask))
            tgt_ids = y_true[b]["labels"]
            
            
            with tf.device(out_mask.device):
                tgt_mask = y_true[b]["masks"]
            # logger.error(f"tgt_ids is {tgt_ids}")
            # logger.error(f"tgt_mask is {tgt_mask}")
            cost_class = -tf.gather(out_prob, tgt_ids, axis=1)
            # print(cost_class)
            # logger.error(f"cost class is {cost_class}") # TODO: DIFF!!!
            tgt_mask = tf.cast(tgt_mask, dtype=tf.float32)
            # print(out_mask.shape)
            tgt_mask = tf.image.resize(tgt_mask[..., tf.newaxis], out_mask.shape[-2:], method='nearest')[..., 0]
            # print(tf.shape(tgt_mask))
            out_mask = tf.reshape(out_mask, [tf.shape(out_mask)[0], -1])
            tgt_mask = tf.reshape(tgt_mask, [tf.shape(tgt_mask)[0], -1])
            # print("outmask after flatten", tf.shape(out_mask))
            # print("tgtmask after flatten", tf.shape(tgt_mask))
            cost_focal = FocalLoss().batch(tgt_mask, out_mask)
            # focalloss = FocalLoss(0.25, 2)
            # cost_focal_nonbatch = focalloss(tgt_mask, out_mask)
            cost_dice = DiceLoss().batch(tgt_mask, out_mask)
            # print(cost_focal)
            # logger.debug(f"cost focal  is {cost_focal}") # TESTED!!
            # logger.debug(f"cost dice  is {cost_dice}") # TODO: A LITTLE DIFF! CONTINUE WORKING
            # exit(-1)
            # logger.debug(f"cost_focal_nonbatch is {cost_focal_nonbatch}")
            # logger.debug(f"cost dice  is {cost_dice}")
            # logger.debug(f"self.cost_focal is {self.cost_focal}")
            # logger.debug(f"self.cost_class is {self.cost_class}")
            C = (
                self.cost_focal * cost_focal
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            # logger.debug(f"cost shape before reshape is {C.shape}")
            # print(cost)
            C = tf.reshape(C, (num_queries, -1))
            # logger.debug(f"cost shape after reshape is {C.shape}")
            



            # _, idx = self.matcher(tf.reshape(cost, [1, num_queries, -1]))
            # idx = tf.stop_gradient(idx)
            # # row = tf.
            # idx = tf.math.argmax(idx, axis=1)
            # # print("row\n", row)
            # # print("col\n", col)
            # # print("lsa\n", linear_sum_assignment(cost))
            # # idx = linear_sum_assignment(cost)
            # # print("idx shape", tf.shape(idx))
            # # indices.append(linear_sum_assignment(cost))
            # # print(idx)
            # # indices.append((list(range(len(idx))), idx))
            indices.append(linear_sum_assignment(C))
            # logger.debug indices lenth and indices[0].shape
            logger.debug(f"indices length: {len(indices)}")
            logger.debug(f"indices[0] shape: {indices[0]}")
        return [
            (tf.convert_to_tensor(i, dtype=tf.int64), tf.convert_to_tensor(j, dtype=tf.int64))
            for i, j in indices
        ]

    def call(self, outputs, y_true):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             y_true: list of dicts, such that len(targets) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # logger.debug("outputs_without_aux[pred_logits].shape: {}".format(outputs_without_aux["pred_logits"].shape))
        # logger.debug("outputs_without_aux[pred_masks].shape: {}".format(outputs_without_aux["pred_masks"].shape))
        indices = self.memory_efficient_matcher(outputs_without_aux, y_true)
        
        # logger.critical(f"matcher output indices: {indices}")
        num_masks = sum(len(t["labels"]) for t in y_true)
        num_masks = tf.convert_to_tensor([num_masks], dtype=tf.float64) # device?
        
        if Utils.is_dist_avail_and_initialized():
            num_masks = tf.distribute.get_strategy().reduce(tf.distribute.ReduceOp.SUM, num_masks, axis=None)
        num_masks = tf.maximum(num_masks / tf.distribute.get_strategy().num_replicas_in_sync, 1.0)
        # logger.debug(f"num_masks is {num_masks}")
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, y_true, indices, num_masks))
        # losses.update({loss: self.get_loss(loss, outputs, y_true, indices, num_masks) for loss in self.losses})

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.memory_efficient_matcher(aux_outputs, y_true)
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
        logger.critical(f"weighted_loss_ce is {weighted_loss_ce}")
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
        # TODO: I AM HERE
        if tf.rank(tensor_list[0]).numpy() == 3:
            # TODO: figure out ONNX stuff
            # if tf.executing_eagerly():
            #     return _onnx_nested_tensor_from_tensor_list(tensor_list)
            
            max_size = tf.reduce_max([tf.shape(img) for img in tensor_list], axis=0)
            batch_shape = tf.concat([[len(tensor_list)], max_size], axis=0)
            batch_size, num_channels, height, width = batch_shape
            logger.debug(f"batch_shape is {batch_shape}")
            logger.debug(f"batch_size is {batch_size}")
            logger.debug(f"num_channels is {num_channels}")
            logger.debug(f"height is {height}")
            logger.debug(f"width is {width}")
            with tf.device(tensor_list[0].device):
                tensor = tf.zeros(batch_shape, dtype=tensor_list[0].dtype)
                mask = tf.ones((batch_size, height, width), dtype=tf.bool)
            # for img, pad_img, m in zip(tensor_list, tensor, mask):
            #     pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].assign(img)
            #     m[:img.shape[1], :img.shape[2]].assign(False)
           # Iterate through the input tensors and pad them with zeros
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                logger.critical(f"img.shape is {img.shape}")
                logger.critical(f"pad_img.shape is {pad_img.shape}")
                logger.critical(f"m.shape is {m.shape}")
                logger.critical(f"max_size is {max_size}")
                # pad_shape = tf.pad(max_size - tf.shape(img), [[0, 0], [0, 1]], constant_values=0)
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].assign(img)
                exit(-1)
            # for idx, img in enumerate(tensor_list):
            #     pad_shape = tf.pad(max_size - tf.shape(img), [[0, 0], [0, 1]], constant_values=0)
            #     pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].assign(img)
            #     mask[idx][:img.shape[0], :img.shape[1]].assign(tf.zeros((img.shape[0], img.shape[1]), dtype=tf.bool))

        else:
            raise ValueError("not supported")
        return NestedTensor(tensor, mask)
    
    def is_dist_avail_and_initialized():
        if not tf.distribute.has_strategy():
            return False
        if not tf.distribute.in_cross_replica_context():
            return False
        return True