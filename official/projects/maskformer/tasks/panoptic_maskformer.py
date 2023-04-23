import tensorflow as tf

from official.core import base_task

from official.projects.maskformer.maskformer import MaskFormer
from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.detr.ops.matchers import hungarian_matching

class PanopticTask(base_task.Task):
    def build_inputs(self, params):
        raise NotImplementedError
    
    def build_model(self):
        # TODO(ibrahim): Connect to params in config.
        model = MaskFormer()

        return model

    def build_losses(self, class_prob_outputs, mask_prob_outputs, class_targets, mask_targets):
        outputs = {"pred_logits": class_prob_outputs, "pred_masks": mask_prob_outputs}
        targets = {"labels": class_targets, "masks": mask_targets}

        #TODO: Connect to params in config.
        mask_weight = 20.0
        dice_weight = 1.0
        no_object_weight = 0.1
        losses = ["labels", "masks"]
        num_classes = 133
        matcher = hungarian_matching
        weight_dict = {"loss_ce":1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        eos_coef = 0.1
        losses = ["labels", "masks"]
        cost_class = 1
        cost_focal = 1
        cost_dice = 1

        _compute_loss = Loss(num_classes, matcher, weight_dict, eos_coef, losses, cost_class, cost_focal, cost_dice)
        return _compute_loss(outputs, targets)
    
    def build_metrics(self, training=True):
        raise NotImplementedError

    def train_step(self, inputs, model, optimizer, metrics=None):
        raise NotImplementedError

    def validation_step(self, inputs, model, optimizer, metrics=None):
        raise NotImplementedError
