import numpy as np
import tensorflow as tf


from official.projects.maskformer.configs import maskformer
from official.projects.maskformer.dataloaders import coco
from official.vision.configs import backbones

from official.projects.maskformer.dataloaders import input_reader
from official.vision.dataloaders import input_reader_factory

from official.projects.maskformer.configs import maskformer as exp_cfg
from official.projects.maskformer.modeling.maskformer import MaskFormer
from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.maskformer.dataloaders import panoptic_input
from official.projects.maskformer.tasks import panoptic_maskformer

from official.projects.detr.ops.matchers import hungarian_matching
from official.projects.maskformer.losses.maskformer_losses import Loss

# TODO: remove unnecessary imports

class PanopticTest(tf.test.TestCase):

    def test_train_step(self):
        pass

    
    def test_eval_step(self):
        config = exp_cfg.MaskFormerTask(
            model=exp_cfg.MaskFormer(
                input_size=[640, 640, 3],
                num_encoder_layers=1,
                num_decoder_layers=1,
                num_classes=133,
                backbone=backbones.Backbone(
                    type='resnet',
                    resnet=backbones.ResNet(model_id=10, bn_trainable=False))
            ),
            validation_data=coco.COCODataConfig(
                # TODO
            )
        )
        task = panoptic_maskformer.PanopticTask(config)
        model = task.build_model()
        metrics = task.build_metrics(training=False)
        dataset = task.build_inputs(config.validation_data)
        iterator = iter(dataset)
        logs = task.validation_step(next(iterator), model, metrics)
        state = task.aggregate_logs(step_outputs=logs)
        task.reduce_aggregated_logs(state)
            
if __name__ == '__main__':
    tf.test.main()