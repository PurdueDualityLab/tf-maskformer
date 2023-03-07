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

"""Config template to train Maskformer"""

from official.projects.configs import base_config
from official.modeling.hyperparams import params_dict

# pylint: disable=line-too-long
MASK_FORMER_CFG = params_dict.ParamsDict(base_config.BASE_CFG)

MASK_FORMER_CFG.override({
  "input": {
    "dataset_mapper_name": "mask_former_semantic",
    "color_aug_ssd": False,
    "crop": {
      "single_category_max_area": 1.0,
      "min_size_train": (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
      "type": "absolute_range",
      "size": (384, 600),
    },
    "size_divisibility": -1,
    "image_size": [400,400],
    "min_scale": 0.5,
    "max_scale": 2.0,
    "resize_eval_groundtruth": True,
    "groundtruth_padded_size": None,
    "ignore_label": 0,
    "aug_rand_hflip": True,
    "aug_type": None,
    "sigma": 8.0,
    "small_instance_area_threshold": 4096,
    "small_instance_weight": 3.0,
    "dtype": 'float32',
  },
  "train": {
    "iterations_per_loop": 100,
    "batch_size": 64,
    "total_steps": 22500,
    "num_cores_per_replica": None,
    "input_partition_dims": None,
    "optimizer": {
      "type": [
        "momentum"
      ],
      "momentum": [
        0.9
      ],
      "nesterov": [
        True
      ]
    },
    "learning_rate": {
      "type": "step",
      "warmup_learning_rate": 0.0067,
      "warmup_steps": 500,
      "init_learning_rate": 0.08,
      "learning_rate_levels": [
        0.008,
        0.0008
      ],
      "learning_rate_steps": [
        15000,
        20000
      ]
    },
    "frozen_variable_prefix": "",
    "train_file_pattern": [
      "/scratch/gilbreth/abuynits/coco_ds/tfrecords/train*"
      ],
    "train_dataset_type": "tfrecord",
    "transpose_input": False,
    "l2_weight_decay": 0.0001,
    "gradient_clip_norm": 0.0,
    "input_sharding": False
  },
  "eval": {
    "eval_file_pattern": [
      "/scratch/gilbreth/abuynits/coco_ds/tfrecords/val*"
    ],
    "type": "box_and_mask",
    "num_images_to_visualize": 0
  },
  "solver": {
    "weight_decay_embed": 0.0,
    "optimizer": "ADAMW",
    "backbone_multiplier": 0.1
  },
  "model": {
    "mask_former": {
      "deep_supervision": True,
      "no_object_weight": 0.1,
      "class_weight": 1.0,
      "dice_weight": 1.0,
      "mask_weight": 20.0,
      "nheads": 8,
      "dropout": 0.1,
      "dim_feedforward": 2048,
      "enc_layers": 0,
      "dec_layers": 6,
      "pre_norm": False,
      "hidden_dim": 256,
      "num_object_queries": 100,
      "transformer_in_feature": "res5",
      "enforce_input_proj": False,
      "test": {
        "semantic_on": True,
        "instance_on": False,
        "panoptic_on": False,
        "object_mask_threshold": 0.0,
        "overlap_threshold": 0.0,
        "sem_seg_postprocessing_before_inference": False
      },
      "size_divisibility": 32,
      "transformer_decoder_name": "MultiScaleMaskedTransformerDecoder",
      "train_num_points": 12544,
      "oversample_ratio": 3.0,
      "importance_sample_ratio": 0.75
    },
    "sem_seg_head": {
      "mask_dim": 256,
      "transformer_enc_layers": 0,
      "pixel_decoder_name": "BasePixelDecoder",
      "deformable_transformer_encoder_in_features": [
        "res3",
        "res4",
        "res5"
      ],
      "deformable_transformer_encoder_n_points": 4,
      "deformable_transformer_encoder_n_heads": 8
    },
    "swin": {
      "pretrain_img_size": 224,
      "patch_size": 4,
      "embed_dim": 96,
      "depths": [
        2,
        2,
        6,
        2
      ],
      "num_heads": [
        3,
        6,
        12,
        24
      ],
      "window_size": 7,
      "mlp_ratio": 4.0,
      "qkv_bias": True,
      "qk_scale": None,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "drop_path_rate": 0.3,
      "APE": False,
      "patch_norm": True,
      "out_features": [
        "res2",
        "res3",
        "res4",
        "res5"
      ],
      "use_checkpoint": False
    }
  }
}, is_strict=False)

MASK_FORMER_RESTRICTIONS = []
# pylint: enable=line-too-long
