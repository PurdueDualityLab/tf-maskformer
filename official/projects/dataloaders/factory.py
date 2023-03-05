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

"""Model architecture factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from panoptic_input import mask_former_parser


def parser_generator(params,mode):
    #TODO: need to specify the param dict
    """Generator function for various dataset parser."""
    if params.architecture.parser == 'mask_former_parser':
        anchor_params = params.anchor
        parser_params = params.input
        parser_fn = mask_former_parser(
            output_size=parser_params.image_size,
            resize_eval_groundtruth=parser_params.resize_eval_groundtruth,
            groundtruth_padded_size=parser_params.groundtruth_padded_size,
            ignore_label=parser_params.ignore_label,
            aug_rand_hflip=parser_params.aug_rand_hflip,
            aug_scale_min=parser_params.min_scale,
            aug_scale_max=parser_params.max_scale,
            aug_type=parser_params.aug_type,
            sigma=parser_params.sigma,
            small_instance_area_threshold=parser_params.small_instance_area_threshold,
            small_instance_weight=parser_params.small_instance_weight,
            dtype=parser_params.dtype,
            mode=mode,
        )
    else:
        raise ValueError('Parser %s is not supported.' % params.architecture.parser)

    return parser_fn
