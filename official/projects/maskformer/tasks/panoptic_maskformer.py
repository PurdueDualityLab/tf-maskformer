import tensorflow as tf

from official.core import base_task

from official.projects.maskformer.maskformer import MaskFormer
from official.projects.maskformer.losses.maskformer_losses import Loss
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

        # _compute_loss = Loss(init loss here...)
        # return _compute_loss(outputs, targets)
        raise NotImplementedError
    
    def build_metrics(self, training=True):
        raise NotImplementedError

    def train_step(self, inputs, model, optimizer, metrics=None):
        raise NotImplementedError

    def validation_step(self, inputs, model, optimizer, metrics=None):
        raise NotImplementedError
