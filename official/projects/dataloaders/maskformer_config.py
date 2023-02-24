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

"""Config template to train Mask R-CNN."""

import base_config
from official.modeling.hyperparams import params_dict

# pylint: disable=line-too-long
MASK_FORMER_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
MASK_FORMER_CFG.override({
    'type': 'mask_former',
    'train': {
        'iterations_per_loop': 100,
        'batch_size': 64,
        'total_steps': 22500,
        'num_cores_per_replica': None,
        'input_partition_dims': None,
        'optimizer': {
            'type': 'momentum',
            'momentum': 0.9,
            'nesterov': True,  # `False` is better for TPU v3-128.
        },
        'learning_rate': {
            'type': 'step',
            'warmup_learning_rate': 0.0067,
            'warmup_steps': 500,
            'init_learning_rate': 0.08,
            'learning_rate_levels': [0.008, 0.0008],
            'learning_rate_steps': [15000, 20000],
        },
        'checkpoint': {
            'path': '',
            'prefix': '',
        },
        # One can use 'RESNET_FROZEN_VAR_PREFIX' to speed up ResNet training
        # when loading from the checkpoint.
        'frozen_variable_prefix': '',
        'train_file_pattern': '/content/drive/MyDrive/coco_tfrecords',  # TODO: fix this sketchiness
        'train_dataset_type': 'tfrecord',
        # TODO(b/142174042): Support transpose_input option.
        'transpose_input': False,
        'l2_weight_decay': 0.0001,
        'gradient_clip_norm': 0.0,
        'input_sharding': False,
    },
    'eval': {
        'eval_file_pattern': '/content/drive/MyDrive/coco_tfrecords2',  # TODO: fix this sketchiness
        'type': 'box_and_mask',
        'num_images_to_visualize': 0,
    },
    'architecture': {
        'parser': 'mask_former_parser',
        'min_level': 2,
        'max_level': 6,
        'include_mask': True,
        'mask_target_size': 28,
    },
    'mask_former': {
        'output_size': [1024, 1024],
        'num_channels': 3,
        'rpn_match_threshold': 0.7,
        'rpn_unmatched_threshold': 0.3,
        'rpn_batch_size_per_im': 256,
        'rpn_fg_fraction': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 1.0,
        'aug_scale_max': 1.0,
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
        'mask_crop_size': 112,
    },
    'anchor': {
        'num_scales': 1,
        'anchor_size': 8,
    },
    'rpn_head': {
        'num_convs': 2,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': False,
    },
    'frcnn_head': {
        'num_convs': 0,
        'num_filters': 256,
        'use_separable_conv': False,
        'num_fcs': 2,
        'fc_dims': 1024,
        'use_batch_norm': False,
    },
    'mrcnn_head': {
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': False,
    },
    'rpn_score_loss': {
        'rpn_batch_size_per_im': 256,
    },
    'rpn_box_loss': {
        'huber_loss_delta': 1.0 / 9.0,
    },
    'frcnn_box_loss': {
        'huber_loss_delta': 1.0,
    },
    'roi_proposal': {
        'rpn_pre_nms_top_k': 2000,
        'rpn_post_nms_top_k': 1000,
        'rpn_nms_threshold': 0.7,
        'rpn_score_threshold': 0.0,
        'rpn_min_size_threshold': 0.0,
        'test_rpn_pre_nms_top_k': 1000,
        'test_rpn_post_nms_top_k': 1000,
        'test_rpn_nms_threshold': 0.7,
        'test_rpn_score_threshold': 0.0,
        'test_rpn_min_size_threshold': 0.0,
        'use_batched_nms': False,
    },
    'roi_sampling': {
        'num_samples_per_image': 512,
        'fg_fraction': 0.25,
        'fg_iou_thresh': 0.5,
        'bg_iou_thresh_hi': 0.5,
        'bg_iou_thresh_lo': 0.0,
        'mix_gt_boxes': True,
    },
    'mask_sampling': {
        'num_mask_samples_per_image': 128,  # Typically = `num_samples_per_image` * `fg_fraction`.
    },
    'postprocess': {
        'pre_nms_num_boxes': 1000,
    },
}, is_strict=False)

MASK_FORMER_RESTRICTIONS = []
# pylint: enable=line-too-long
