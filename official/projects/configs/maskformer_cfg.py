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
from official.vision.configs import backbones
from official.vision.configs import common

COCO_INPUT_PATH_BASE = '/scratch/gilbreth/abuynits/coco_ds/'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000




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
    class_offset: int = 0
    lambda_cls: float = 1.0
    lambda_box: float = 5.0
    lambda_giou: float = 2.0
    background_cls_weight: float = 0.1
    l2_weight_decay: float = 1e-4


@dataclasses.dataclass
class MaskFormer(hyperparams.Config):
    """Detr model definations."""
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
class DetrTask(cfg.TaskConfig):
    # model: Detr = Detr()
    train_data: cfg.DataConfig = cfg.DataConfig()
    validation_data: cfg.DataConfig = cfg.DataConfig()
    losses: Losses = Losses()
    init_checkpoint: Optional[str] = None
    init_checkpoint_modules: Union[str, List[str]] = 'all'  # all, backbone
    annotation_file: Optional[str] = None
    per_category_metrics: bool = False


@exp_factory.register_config_factory('detr_coco_tfrecord')
def detr_coco_tfrecord() -> cfg.ExperimentConfig:
    """Config to get results that matches the paper."""
    train_batch_size = 64
    eval_batch_size = 64
    steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
    train_steps = 300 * steps_per_epoch  # 300 epochs
    decay_at = train_steps - 100 * steps_per_epoch  # 200 epochs
    config = cfg.ExperimentConfig(
        task=DetrTask(
            init_checkpoint='',
            init_checkpoint_modules='backbone',
            annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
                                         'instances_val2017.json'),
            # model=Detr(
                # input_size=[1333, 1333, 3],
                # norm_activation=common.NormActivation()),
            losses=Losses(),
            train_data=DataConfig(
                input_path=os.path.join(COCO_INPUT_PATH_BASE, 'tfrecords/val*'),
                is_training=True,
                global_batch_size=train_batch_size,
                shuffle_buffer_size=1000,
                parser = Parser(
                    output_size = [400,400],
                    min_scale = 0.3,
                    aspect_ratio_range = (0.5, 2.0),
                    min_overlap_params = (0.0, 1.4, 0.2, 0.1),
                    max_retry = 50,
                    pad_output = True,
                    resize_eval_groundtruth = True,
                    groundtruth_padded_size = None,
                    ignore_label = 0,
                    aug_rand_hflip = True,
                    aug_scale_min = 1.0,
                    aug_scale_max = 1.0,
                    color_aug_ssd = False,
                    brightness = 0.2,
                    saturation = 0.3,
                    contrast = 0.5,
                    aug_type = None,
                    sigma = 8.0,
                    small_instance_area_threshold = 4096,
                    small_instance_weight = 3.0,
                    dtype = 'float32',
                    seed = 4096,
                )
            ),
            validation_data=DataConfig(
                input_path=os.path.join(COCO_INPUT_PATH_BASE, 'tfrecords/val*'),
                is_training=False,
                global_batch_size=eval_batch_size,
                drop_remainder=False,
                parser = Parser(
                    output_size = [400,400],
                    min_scale = 0.3,
                    aspect_ratio_range = (0.5, 2.0),
                    min_overlap_params = (0.0, 1.4, 0.2, 0.1),
                    max_retry = 50,
                    pad_output = True,
                    resize_eval_groundtruth = True,
                    groundtruth_padded_size = None,
                    ignore_label = 0,
                    aug_rand_hflip = True,
                    aug_scale_min = 1.0,
                    aug_scale_max = 1.0,
                    color_aug_ssd = False,
                    brightness = 0.2,
                    saturation = 0.3,
                    contrast = 0.5,
                    aug_type = None,
                    sigma = 8.0,
                    small_instance_area_threshold = 4096,
                    small_instance_weight = 3.0,
                    dtype = 'float32',
                    seed = 4096,
                )
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
            #optimizer_config=optimization.OptimizationConfig({
            #    'optimizer': {
            #        'type': 'detr_adamw',
            #        'detr_adamw': {
            #            'weight_decay_rate': 1e-4,
            #            'global_clipnorm': 0.1,
            #            # Avoid AdamW legacy behavior.
            #            'gradient_clip_norm': 0.0
            #        }
            #    },
            #    'learning_rate': {
            #        'type': 'stepwise',
            #        'stepwise': {
            #            'boundaries': [decay_at],
            #            'values': [0.0001, 1.0e-05]
            #        }
            #    },
            #})
            ),
        restrictions=[
            'task.train_data.is_training != None',
        ])
    return config
# 
# ExperimentConfig(task=DetrTask(init_checkpoint='', model=None,
                            #    train_data=DataConfig(input_path='coco/train*', tfds_name='', tfds_split='train',
                                                    #  global_batch_size=64, is_training=True, drop_remainder=True,
                                                    #  shuffle_buffer_size=1000, cache=False, cycle_length=None,
                                                    #  block_length=1, deterministic=None, sharding=True,
                                                    #  enable_tf_data_service=False, tf_data_service_address=None,
                                                    #  tf_data_service_job_name=None, tfds_data_dir='',
                                                    #  tfds_as_supervised=False, tfds_skip_decoding_feature='', seed=None,
                                                    #  prefetch_buffer_size=None, dtype='bfloat16',
                                                    #  decoder=DataDecoder(type='simple_decoder',
                                                                        #  simple_decoder=TfExampleDecoder(
                                                                            #  regenerate_source_id=False,
                                                                            #  mask_binarize_threshold=None),
                                                                        #  label_map_decoder=TfExampleDecoderLabelMap(
                                                                            #  regenerate_source_id=False,
                                                                            #  mask_binarize_threshold=None,
                                                                            #  label_map='')), file_type='tfrecord'),
                            #    validation_data=DataConfig(input_path='coco/val*', tfds_name='', tfds_split='train',
                                                        #   global_batch_size=64, is_training=False, drop_remainder=False,
                                                        #   shuffle_buffer_size=10000, cache=False, cycle_length=None,
                                                        #   block_length=1, deterministic=None, sharding=True,
                                                        #   enable_tf_data_service=False, tf_data_service_address=None,
                                                        #   tf_data_service_job_name=None, tfds_data_dir='',
                                                        #   tfds_as_supervised=False, tfds_skip_decoding_feature='',
                                                        #   seed=None, prefetch_buffer_size=None, dtype='bfloat16',
                                                        #   decoder=DataDecoder(type='simple_decoder',
                                                                            #   simple_decoder=TfExampleDecoder(
                                                                                #   regenerate_source_id=False,
                                                                                #   mask_binarize_threshold=None),
                                                                            #   label_map_decoder=TfExampleDecoderLabelMap(
                                                                                #   regenerate_source_id=False,
                                                                                #   mask_binarize_threshold=None,
                                                                                #   label_map='')), file_type='tfrecord'),
                            #    name=None, differential_privacy_config=None,
                            #    losses=Losses(class_offset=0, lambda_cls=1.0, lambda_box=5.0, lambda_giou=2.0,
                                            #  background_cls_weight=0.1, l2_weight_decay=0.0001),
                            #    init_checkpoint_modules='backbone', annotation_file='coco/instances_val2017.json',
                            #    per_category_metrics=False), trainer=TrainerConfig(optimizer_config=OptimizationConfig(
    # optimizer=OptimizerConfig(type=None,
                            #   sgd=SGDConfig(clipnorm=None, clipvalue=None, global_clipnorm=None, name='SGD', decay=0.0,
                                            # nesterov=False, momentum=0.0),
                            #   sgd_experimental=SGDExperimentalConfig(clipnorm=None, clipvalue=None,
                                                                    #  global_clipnorm=None, name='SGD', nesterov=False,
                                                                    #  momentum=0.0, jit_compile=False),
                            #   adam=AdamConfig(clipnorm=None, clipvalue=None, global_clipnorm=None, name='Adam',
                                            #   beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
                            #   adam_experimental=AdamExperimentalConfig(clipnorm=None, clipvalue=None,
                                                                    #    global_clipnorm=None, name='Adam', beta_1=0.9,
                                                                    #    beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                                                    #    jit_compile=False),
                            #   adamw=AdamWeightDecayConfig(clipnorm=None, clipvalue=None, global_clipnorm=None,
                                                        #   name='AdamWeightDecay', beta_1=0.9, beta_2=0.999,
                                                        #   epsilon=1e-07, amsgrad=False, weight_decay_rate=0.0,
                                                        #   include_in_weight_decay=None, exclude_from_weight_decay=None,
                                                        #   gradient_clip_norm=1.0),
                            #   lamb=LAMBConfig(clipnorm=None, clipvalue=None, global_clipnorm=None, name='LAMB',
                                            #   beta_1=0.9, beta_2=0.999, epsilon=1e-06, weight_decay_rate=0.0,
                                            #   exclude_from_weight_decay=None, exclude_from_layer_adaptation=None),
                            #   rmsprop=RMSPropConfig(clipnorm=None, clipvalue=None, global_clipnorm=None, name='RMSprop',
                                                    # rho=0.9, momentum=0.0, epsilon=1e-07, centered=False),
                            #   lars=LARSConfig(clipnorm=None, clipvalue=None, global_clipnorm=None, name='LARS',
                                            #   momentum=0.9, eeta=0.001, weight_decay_rate=0.0, nesterov=False,
                                            #   classic_momentum=True, exclude_from_weight_decay=None,
                                            #   exclude_from_layer_adaptation=None),
                            #   adagrad=AdagradConfig(clipnorm=None, clipvalue=None, global_clipnorm=None, name='Adagrad',
                                                    # initial_accumulator_value=0.1, epsilon=1e-07),
                            #   slide=SLIDEConfig(clipnorm=None, clipvalue=None, global_clipnorm=None, name='SLIDE',
                                                # beta_1=0.9, beta_2=0.999, epsilon=1e-06, weight_decay_rate=0.0,
                                                # weight_decay_type='inner', exclude_from_weight_decay=None,
                                                # exclude_from_layer_adaptation=None,
                                                # include_in_sparse_layer_adaptation=None, sparse_layer_learning_rate=0.1,
                                                # do_gradient_rescaling=True, norm_type='layer',
                                                # ratio_clip_norm=100000.0),
                            #   adafactor=AdafactorConfig(clipnorm=None, clipvalue=None, global_clipnorm=None,
                                                        # name='Adafactor', factored=True,
                                                        # multiply_by_parameter_scale=True, beta1=None, decay_rate=0.8,
                                                        # step_offset=0, clipping_threshold=1.0,
                                                        # min_dim_size_to_factor=128, epsilon1=1e-30, epsilon2=0.001)),
    # ema=None, learning_rate=LrConfig(type=None, constant=ConstantLrConfig(name='Constant', learning_rate=0.1),
                                    #  stepwise=StepwiseLrConfig(name='PiecewiseConstantDecay', boundaries=None,
                                                            #    values=None, offset=0),
                                    #  exponential=ExponentialLrConfig(name='ExponentialDecay',
                                                                    #  initial_learning_rate=None, decay_steps=None,
                                                                    #  decay_rate=None, staircase=None, offset=0),
                                    #  polynomial=PolynomialLrConfig(name='PolynomialDecay', initial_learning_rate=None,
                                                                #    decay_steps=None, end_learning_rate=0.0001,
                                                                #    power=1.0, cycle=False, offset=0),
                                    #  cosine=CosineLrConfig(name='CosineDecay', initial_learning_rate=None,
                                                        #    decay_steps=None, alpha=0.0, offset=0),
                                    #  power=DirectPowerLrConfig(name='DirectPowerDecay', initial_learning_rate=None,
                                                            #    power=-0.5),
                                    #  power_linear=PowerAndLinearDecayLrConfig(name='PowerAndLinearDecay',
                                                                            #   initial_learning_rate=None,
                                                                            #   total_decay_steps=None, power=-0.5,
                                                                            #   linear_decay_fraction=0.1, offset=0),
                                    #  power_with_offset=PowerDecayWithOffsetLrConfig(name='PowerDecayWithOffset',
                                                                                    # initial_learning_rate=None,
                                                                                    # power=-0.5, offset=0,
                                                                                    # pre_offset_learning_rate=1000000.0),
                                    #  step_cosine_with_offset=StepCosineLrConfig(name='StepCosineDecayWithOffset',
                                                                                # boundaries=None, values=None,
                                                                                # offset=0)),
    # warmup=WarmupConfig(type=None, linear=LinearWarmupConfig(name='linear', warmup_learning_rate=0, warmup_steps=None),
                        # polynomial=PolynomialWarmupConfig(name='polynomial', power=1, warmup_steps=None))),
                                                                                #   train_tf_while_loop=True,
                                                                                #   train_tf_function=True,
                                                                                #   eval_tf_function=True,
                                                                                #   eval_tf_while_loop=False,
                                                                                #   allow_tpu_summary=False,
                                                                                #   steps_per_loop=1848,
                                                                                #   summary_interval=1848,
                                                                                #   checkpoint_interval=1848,
                                                                                #   max_to_keep=1,
                                                                                #   continuous_eval_timeout=3600,
                                                                                #   train_steps=554400,
                                                                                #   validation_steps=78,
                                                                                #   validation_interval=9240,
                                                                                #   best_checkpoint_export_subdir='best_ckpt',
                                                                                #   best_checkpoint_eval_metric='AP',
                                                                                #   best_checkpoint_metric_comp='higher',
                                                                                #   loss_upper_bound=1000000.0,
                                                                                #   recovery_begin_steps=0,
                                                                                #   recovery_max_trials=0,
                                                                                #   validation_summary_subdir='validation'),
                #  runtime=RuntimeConfig(distribution_strategy='mirrored', enable_xla=False, gpu_thread_mode=None,
                                    #    dataset_num_private_threads=None, per_gpu_thread_count=0, tpu=None, num_gpus=0,
                                    #    worker_hosts=None, task_index=-1, all_reduce_alg=None, num_packs=1,
                                    #    mixed_precision_dtype=None, loss_scale=None, run_eagerly=False,
                                    #    batchnorm_spatial_persistent=False, tpu_enable_xla_dynamic_padder=None,
                                    #    num_cores_per_replica=1, default_shard_dim=-1))
# 