# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""DETR configurations."""

import dataclasses
import os
from typing import List, Optional, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
# from official.projects.detr import optimization
# from official.projects.detr.dataloaders import coco
from official.vision.configs import backbones
from official.vision.configs import common


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
    groundtruth_padded_size: Optional[List[int]] = None
    ignore_label: int = 0
    aug_rand_hflip: bool = True
    aug_scale_min: float = 1.0
    aug_scale_max: float = 1.0
    color_aug_ssd: bool = False
    brightness: float = 0.2
    saturation: float = 0.3
    contrast: float = 0.5
    aug_type: Optional[common.Augmentation] = None
    sigma: float = 8.0
    small_instance_area_threshold: int = 4096
    small_instance_weight: float = 3.0
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
  dtype: str = 'bfloat16'
  decoder: common.DataDecoder = common.DataDecoder()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True
  parser: Parser = Parser()


@dataclasses.dataclass
class Losses(hyperparams.Config):
  # TODO update these for maskformer
  class_offset: int = 0
  lambda_cls: float = 1.0
  lambda_box: float = 5.0
  lambda_giou: float = 2.0
  background_cls_weight: float = 0.1
  l2_weight_decay: float = 1e-4


@dataclasses.dataclass
class MaskFormer(hyperparams.Config):
  # TODO update these for maskformer
  """MaskFormer model definations."""
  num_queries: int = 100
  hidden_size: int = 256
  num_classes: int = 91  # 0: background
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet(model_id=50, bn_trainable=False))
  norm_activation: common.NormActivation = common.NormActivation()
  backbone_endpoint_name: str = '5'


@dataclasses.dataclass
class MaskFormerTask(cfg.TaskConfig):
  model: MaskFormer = MaskFormer()
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[str, List[str]] = 'all'  # all, backbone
  annotation_file: Optional[str] = None
  per_category_metrics: bool = False

COCO_INPUT_PATH_BASE = '/depot/davisjam/data/vishal/datasets/coco'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('maskformer_coco_panoptic')
def maskformer_coco_panoptic() -> cfg.ExperimentConfig:
  """Config to get results that matches the paper."""
  train_batch_size = 64
  eval_batch_size = 64
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  train_steps = 300 * steps_per_epoch  # 300 epochs
  decay_at = train_steps - 100 * steps_per_epoch  # 200 epochs
  config = cfg.ExperimentConfig(
      task=MaskFormerTask(
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                       'instances_val2017.json'),
          model=MaskFormer(
              input_size=[1333, 1333, 3],
              norm_activation=common.NormActivation()),
          losses=Losses(),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              shuffle_buffer_size=1000,
          ),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
          )),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          validation_interval=5 * steps_per_epoch,
          max_to_keep=1,
          best_checkpoint_export_subdir='best_ckpt',
          best_checkpoint_eval_metric='AP',
          optimizer_config=optimization.OptimizationConfig({
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
          })),
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config
