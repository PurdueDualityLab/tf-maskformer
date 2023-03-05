import tensorflow as tf

from official.core import base_task

from official.projects.maskformer.maskformer import MaskFormer
class PanopticTask(base_task.Task):
    def build_inputs(self, params):
        raise NotImplementedError
    
    def build_model(self):
        # TODO(ibrahim): Connect to params in config.
        model = MaskFormer()

        return model

    def build_losses(self, labels, model_outputs, aux_losses=None):
        raise NotImplementedError
    
    def build_metrics(self, training=True):
        raise NotImplementedError

    def train_step(self, inputs, model, optimizer, metrics=None):
        raise NotImplementedError

    def validation_step(self, inputs, model, optimizer, metrics=None):
        raise NotImplementedError
