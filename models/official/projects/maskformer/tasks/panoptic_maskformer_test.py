import os
import numpy as np
import tensorflow as tf

from official.vision.configs import common

from official.projects.maskformer import optimization
from official.projects.maskformer.configs import maskformer
from official.projects.maskformer.dataloaders import coco
from official.projects.maskformer.configs import maskformer as exp_cfg
from official.projects.maskformer.modeling.maskformer import MaskFormer
from official.projects.maskformer.tasks import panoptic_maskformer
from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.maskformer.losses.inference import PanopticInference


# TODO: remove unnecessary imports
COCO_INPUT_PATH_BASE = 'gs://cam2-datasets/coco_panoptic/'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000

class PanopticTest(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        self.train_batch_size = 2
        self.eval_batch_size = 2
        self.steps_per_epoch = 100

    def test_train_step(self):
        train_steps = 300 * self.steps_per_epoch  # 300 epochs
        decay_at = train_steps - 100 * self.steps_per_epoch
        config = exp_cfg.MaskFormerTask(
            model=MaskFormer(
                input_size=[640, 640, 3],
                norm_activation=common.NormActivation()),
            train_data=coco.DataConfig(
                input_path=os.path.join(COCO_INPUT_PATH_BASE, 'tfrecords/train*'),
                is_training=True,
                global_batch_size=2,
                ),
            )
        task = panoptic_maskformer.PanopticTask(config)
        model = task.build_model()
        dataset = task.build_inputs(config.train_data)
        iterator = iter(dataset)
        opt_cfg = optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'detr_adamw',
                  'detr_adamw': {
                      'weight_decay_rate': 1e-4,
                      'global_clipnorm': 0.1,
                      # Avoid AdamW legacy behavior.
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [decay_at],
                      'values': [0.0001, 1.0e-05]
                  }
              },
          }),
        optimizer = panoptic_maskformer.PanopticTask(opt_cfg)
        task.train_step(next(iterator), model, optimizer)

    
    def test_eval_step(self):
        config = exp_cfg.MaskFormerTask(
            model=MaskFormer(
              input_size=[640, 640, 3],
              norm_activation=common.NormActivation()),
            validation_data=coco.DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'tfrecords/val*'),
              is_training=False,
              global_batch_size=self.eval_batch_size,
              drop_remainder=False,
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