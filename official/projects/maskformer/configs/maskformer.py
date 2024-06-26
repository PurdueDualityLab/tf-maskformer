# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaskFormer configurations."""

import dataclasses
import os
from typing import List, Optional, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.configs import backbones
from official.vision.configs import common
from official.projects.maskformer.utils import optimization
from official.projects.maskformer.losses.mapper import _is_thing


@dataclasses.dataclass
class Parser(hyperparams.Config):
  """Config definitions for parser"""
  output_size: List[int] = None
  min_scale: float = 0.3
  aspect_ratio_range: List[float] = (0.5, 2.0)
  min_overlap_params: List[float] = (0.0, 1.4, 0.2, 0.1)
  max_retry: int = 50
  pad_output: bool = False
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = (1280, 1280)
  ignore_label: int = 0
  aug_rand_hflip: bool = True
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  dtype: str = 'float32'
  seed: int = None


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  tfds_name: str = ''
  tfds_split: str = 'train'
  global_batch_size: int = 0
  is_training: bool = False
  regenerate_source_id: bool = False
  dtype: str = 'float32'
  decoder: common.DataDecoder = common.DataDecoder()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True
  parser: Parser = Parser()


@dataclasses.dataclass
class Losses(hyperparams.Config):
  class_offset: int = 0
  background_cls_weight: float = 0.1
  l2_weight_decay: float = 1e-4
  cost_class = 1.0
  cost_dice = 1.0
  cost_focal = 20.0


@dataclasses.dataclass
class MaskFormer(hyperparams.Config):
  """MaskFormer model definations."""
  num_queries: int = 100
  hidden_size: int = 256
  # There are 134 classes (stuff + things + no object/background) for
  # panoptic segmentation.
  num_classes: int = 133
  fpn_encoder_layers: int = 6
  detr_encoder_layers: int = 0
  num_decoder_layers: int = 6
  which_pixel_decoder: str = 'transformer_fpn'
  deep_supervision: bool = False
  on_tpu: bool = False
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet(model_id=50, bn_trainable=False))
  norm_activation: common.NormActivation = common.NormActivation()
  backbone_endpoint_name: str = '5'


@dataclasses.dataclass
class PanopticQuality(hyperparams.Config):
  """MaskFormer model pq evaluator config."""
  num_categories: int = 133  
  is_thing: List[bool] = dataclasses.field(default_factory=_is_thing) # pylint: disable=line-too-long
  ignored_label: int = 0
  rescale_predictions: bool = False
  max_num_instances: int = 100


@dataclasses.dataclass
class MaskFormerTask(cfg.TaskConfig):
  model: MaskFormer = MaskFormer()
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = ""
  init_checkpoint_modules: Union[str, List[str]] = 'backbone'  # all, backbone
  per_category_metrics: bool = False
  bfloat16: bool = False
  panoptic_quality_evaluator: PanopticQuality = PanopticQuality()

COCO_INPUT_PATH_BASE = os.environ.get('TFRECORDS_DIR')
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000
SET_MODEL_BFLOAT16 = False
SET_DATA_BFLOAT16 = True


@exp_factory.register_config_factory('maskformer_coco_panoptic')
def maskformer_coco_panoptic() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""

  train_batch_size = int(os.environ.get('TRAIN_BATCH_SIZE'))
  eval_batch_size = int(os.environ.get('EVAL_BATCH_SIZE'))
  no_obj_cls_weight = float(os.environ.get('NO_OBJ_CLS_WEIGHT'))
  deep_supervision = True if str(os.environ.get(
      'DEEP_SUPERVISION')) == 'True' else False
  on_tpu = True if str(os.environ.get('ON_TPU')) == 'True' else False
  image_size = int(os.environ.get('IMG_SIZE'))

  # Don't write ckpts frequently. Slows down the training
  ckpt_interval = (COCO_TRAIN_EXAMPLES // train_batch_size) * 10

  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  train_steps = 300 * steps_per_epoch  # 300 epochs
  decay_at = train_steps - 100 * steps_per_epoch  # 200 epochs
  config = cfg.ExperimentConfig(
      task=MaskFormerTask(
          init_checkpoint="",
          init_checkpoint_modules='backbone',
          bfloat16=SET_MODEL_BFLOAT16,
          model=MaskFormer(
              input_size=[image_size, image_size, 3],
              norm_activation=common.NormActivation(),
              which_pixel_decoder='transformer_fpn',
              deep_supervision=deep_supervision,
              on_tpu=on_tpu,
              num_classes=133,),  # Extra class will be added automatically for background
          losses=Losses(
              background_cls_weight=no_obj_cls_weight,
          ),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=1000,
              dtype='bfloat16' if SET_DATA_BFLOAT16 else 'float32',
              parser=Parser(
                  output_size=[image_size, image_size],
                  min_scale=0.3,
                  aspect_ratio_range=(0.5, 2.0),
                  min_overlap_params=(0.0, 1.4, 0.2, 0.1),
                  max_retry=50,
                  pad_output=True,
                  resize_eval_groundtruth=True,
                  groundtruth_padded_size=[image_size, image_size],
                  ignore_label=0,
                  aug_rand_hflip=True,
                  aug_scale_min=1.0,
                  aug_scale_max=1.0,
                  dtype='bfloat16' if SET_DATA_BFLOAT16 else 'float32',
                  seed=2045,
              )
          ),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
              parser=Parser(
                  output_size=[image_size, image_size],
                  pad_output=True,
                  seed=2045,
                  ignore_label=0,
                  dtype='bfloat16' if SET_DATA_BFLOAT16 else 'float32',
              )

          )),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          steps_per_loop=steps_per_epoch,
          summary_interval=ckpt_interval,
          checkpoint_interval=ckpt_interval,
          validation_interval=5 * steps_per_epoch,
          # run validation after every epoch (not efficient, but we want to see
          # the results)sss
          best_checkpoint_export_subdir='best_ckpt',
          max_to_keep=1,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'maskformer_adamw',
                  'maskformer_adamw': {
                      'weight_decay_rate': 0.0001,
                      'global_clipnorm': 0.1,
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [decay_at],
                      'values': [float(os.environ.get('BASE_LR')), float(os.environ.get('BASE_LR')) / 10]  # pylint: disable=line-too-long
                  }
              },

          })),
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config
