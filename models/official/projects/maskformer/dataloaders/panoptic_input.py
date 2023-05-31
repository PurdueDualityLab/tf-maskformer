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

"""Data parser and processing for Panoptic Deeplab."""

from typing import List, Optional

import numpy as np
import tensorflow as tf
from loguru import logger
from official.vision.dataloaders import parser
from official.vision.dataloaders import tf_example_decoder
from official.vision.ops import augment
from official.vision.ops import preprocess_ops
from official.core import config_definitions as cfg
tf.compat.v1.enable_eager_execution()
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 10], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

def _compute_gaussian_from_std(sigma):
    """Computes the Gaussian and its size from a given standard deviation."""
    size = int(6 * sigma + 3)
    x = np.arange(size, dtype=float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    gaussian = tf.constant(
        np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)),
        dtype=tf.float32)
    return gaussian, size


class TfExampleDecoder(tf_example_decoder.TfExampleDecoder):
    """Tensorflow Example proto decoder."""

    def __init__(
            self,
            regenerate_source_id: bool = True,
            panoptic_category_mask_key: str = 'image/panoptic/category_mask',
            panoptic_instance_mask_key: str = 'image/panoptic/instance_mask'):
        super(TfExampleDecoder,
              self).__init__(
            include_mask=True,
            regenerate_source_id=regenerate_source_id)
        self._panoptic_category_mask_key = panoptic_category_mask_key
        self._panoptic_instance_mask_key = panoptic_instance_mask_key
        self._image_height_key = 'image/height'
        self._image_width_key = 'image/width'
        self._image_key = ""
        self._panoptic_keys_to_features = {
            self._panoptic_category_mask_key:
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            self._panoptic_instance_mask_key:
                tf.io.FixedLenFeature((), tf.string, default_value=''),

        }


    def decode(self, serialized_example):
        decoded_tensors = super(TfExampleDecoder,
                                self).decode(serialized_example)
        
    
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, self._panoptic_keys_to_features)
        
        category_mask = tf.io.decode_png(
            parsed_tensors[self._panoptic_category_mask_key], channels=1)
        instance_mask = tf.io.decode_png(
            parsed_tensors[self._panoptic_instance_mask_key], channels=1)
        

        category_mask.set_shape([None, None, 1])
        instance_mask.set_shape([None, None, 1])

        decoded_tensors.update({
            'groundtruth_panoptic_category_mask': category_mask,
            'groundtruth_panoptic_instance_mask': instance_mask
        })
        
        return decoded_tensors


class mask_former_parser(parser.Parser):
    """Parser to parse an image and its annotations into a dictionary of tensors."""

    def __init__(
            self,
            params: cfg.DataConfig,
            decoder_fn = None,
            is_training = False,
            ):
        """Initializes parameters for parsing annotations in the dataset.
    
        Args:
          output_size: `Tensor` or `list` for [height, width] of output image. The
            output_size should be divided by the largest feature stride 2^max_level.
          resize_eval_groundtruth: `bool`, if True, eval groundtruth masks are
            resized to output_size.
          groundtruth_padded_size: `Tensor` or `list` for [height, width]. When
            resize_eval_groundtruth is set to False, the groundtruth masks are
            padded to this size.
          ignore_label: `int` the pixel with ignore label will not used for training
            and evaluation.
          aug_rand_hflip: `bool`, if True, augment training with random
            horizontal flip.
          aug_scale_min: `float`, the minimum scale applied to `output_size` for
            data augmentation during training.
          aug_scale_max: `float`, the maximum scale applied to `output_size` for
            data augmentation during training.
          aug_type: An optional Augmentation object with params for AutoAugment.
          sigma: `float`, standard deviation for generating 2D Gaussian to encode
            centers.
          small_instance_area_threshold: `int`, small instance area threshold.
          small_instance_weight: `float`, small instance weight.
          dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
        """
        
        # general settings
        self._output_size = params.output_size
        self._mask_null = 0
        self._dtype = params.dtype
        self._pad_output = params.pad_output
        self._seed = params.seed
        # TODO
        self._max_instances = 100
        self._decoder = decoder_fn
        
        if self._pad_output == True and self._output_size is None:
            raise Exception("Error: no output pad provided")
        if self._decoder == None:
            print("assuming default decoder")
            self._decoder = TfExampleDecoder()
        
        self._is_training = is_training
        if is_training == None:
            print("assuming training mode")
            self._is_training = True
        
       
        self._resize_eval_groundtruth = params.resize_eval_groundtruth
        if (not params.resize_eval_groundtruth) and (params.groundtruth_padded_size is None):
            raise ValueError(
                'groundtruth_padded_size ([height, width]) needs to be'
                'specified when resize_eval_groundtruth is False.')
        self._groundtruth_padded_size = params.groundtruth_padded_size
        self._ignore_label = params.ignore_label

        # Data augmentation
        self._aug_rand_hflip = params.aug_rand_hflip
        self._aug_scale_min = params.aug_scale_min
        self._aug_scale_max = params.aug_scale_max
        
        # Auto Augment
        if params.aug_type and aug_type.type:
            if aug_type.type == 'autoaug':
                self._augmenter = augment.AutoAugment(
                    augmentation_name=aug_type.autoaug.augmentation_name,
                    cutout_const=aug_type.autoaug.cutout_const,
                    translate_const=aug_type.autoaug.translate_const)
            else:
                raise ValueError('Augmentation policy {} not supported.'.format(
                    aug_type.type))
        else:
            self._augmenter = None
        
        #Cropping:
        self._min_scale = params.min_scale
        self._aspect_ratio_range = params.aspect_ratio_range
        self._min_overlap_params = params.min_overlap_params
        self._max_retry = params.max_retry


        
        # color augmentation
        self._color_aug_ssd = params.color_aug_ssd
        self._brightness = params.brightness
        self._saturation = params.saturation
        self._contrast = params.contrast
        
        self._sigma = params.sigma
        self._gaussian, self._gaussian_size = _compute_gaussian_from_std(
            self._sigma)
        self._gaussian = tf.reshape(self._gaussian, shape=[-1])
        self._small_instance_area_threshold = params.small_instance_area_threshold
        self._small_instance_weight = params.small_instance_weight

        self._thing_dataset_contiguous_id = []
        self._thing_dataset_id = []

        self._stuff_dataset_contiguous_id = []
        self._stuff_dataset_id = []

        for i, cat in enumerate(COCO_CATEGORIES):
            if cat["isthing"]:
                self._thing_dataset_contiguous_id.append(i)
                self._thing_dataset_id.append(cat["id"])
            else:
                self._stuff_dataset_contiguous_id.append(i)
                self._stuff_dataset_id.append(cat["id"])
        
        self._stuff_table = tf.lookup.StaticHashTable(
                        initializer=tf.lookup.KeyValueTensorInitializer(
                            keys=tf.constant(self._stuff_dataset_id),
                            values=tf.constant(self._stuff_dataset_contiguous_id),
                        ),
                        default_value=tf.constant(-1),
                        
                    )
        self._thing_table = tf.lookup.StaticHashTable(
                        initializer=tf.lookup.KeyValueTensorInitializer(
                            keys=tf.constant(self._thing_dataset_id),
                            values=tf.constant(self._thing_dataset_contiguous_id),
                        ),
                        default_value=tf.constant(-5),
                       
                    )
        

    def _resize_and_crop_mask(self, mask, image_info, crop_dims, is_training):
        """Resizes and crops mask using `image_info` dict."""
        
        image_scale = image_info[2, :]
        offset = image_info[3, : ]
        im_height = int(image_info[0][0])
        im_width = int(image_info[0][1])

        # print(mask.shape)
        
        mask = tf.reshape(mask, shape=[1, im_height, im_width, 1])
        # print(mask.shape)
        mask += 1

        if is_training or self._resize_eval_groundtruth:
            # print("using image offset:",offset)
            mask = preprocess_ops.resize_and_crop_masks(
                mask,
                image_scale,
                crop_dims,
                offset)
        else:
            mask = tf.image.pad_to_bounding_box(
                mask, 0, 0,
                self._groundtruth_padded_size[0],
                self._groundtruth_padded_size[1])
        mask -= 1

        # Assign ignore label to the padded region.
        mask = tf.where(
            tf.equal(mask, -1),
            self._ignore_label * tf.ones_like(mask),
            mask)
        mask = tf.squeeze(mask, axis=0)
        return mask
    
    # @tf.function
    # def compare_masks(self, each_id):
    #     # with tf.name_scope("compare_masks"):
    #     return tf.equal(instance_mask, each_id)
        
    def _parse_data(self, data, is_training):
        image = data['image']
        
        # Auto-augment (if configured)
        if self._augmenter is not None and is_training:
            image = self._augmenter.distort(image)
        
        # Normalize and prepare image and masks
        image = preprocess_ops.normalize_image(image)
        category_mask = tf.cast(
            data['groundtruth_panoptic_category_mask'][:, :, 0],
            dtype=tf.float32)
        instance_mask = tf.cast(
            data['groundtruth_panoptic_instance_mask'][:, :, 0],
            dtype=tf.float32)
        
        # applies by pixel augmentation (saturation, brightness, contrast)
        if self._color_aug_ssd:
            image = preprocess_ops.color_jitter(
                image = image,
                brightness = self._brightness,
                contrast = self._contrast,
                saturation = self._saturation,
                seed = self._seed,
            )
        # Flips image randomly during training.
        if self._aug_rand_hflip and is_training:
            # print("doing random flip")
            masks = tf.stack([category_mask, instance_mask], axis=0)
            image, _, masks = preprocess_ops.random_horizontal_flip(
                image=image, 
                masks=masks,
                seed = self._seed)

            category_mask = masks[0]
            instance_mask = masks[1]
            
        # Resize and crops image.
        
        masks = tf.stack([category_mask, instance_mask], axis=0)
        masks = tf.expand_dims(masks, -1)
       
        # Resizes and crops image.
        cropped_image, masks = preprocess_ops.random_crop_image_masks(
            img = image,
            masks = masks,
            min_scale = self._min_scale,
            aspect_ratio_range = self._aspect_ratio_range,
            min_overlap_params = self._min_overlap_params,
            max_retry = self._max_retry,
            seed = self._seed,
        )
                                                                      
                                                                      
        category_mask = tf.squeeze(masks[0])
        instance_mask = tf.squeeze(masks[1])
        
        
        
        crop_im_size = tf.cast(tf.shape(cropped_image)[0:2], tf.int32)
        
        # Resize image
        image, image_info = preprocess_ops.resize_and_crop_image(
            cropped_image,
            self._output_size if self._pad_output else crop_im_size,
            self._output_size if self._pad_output else crop_im_size,
            aug_scale_min=self._aug_scale_min if self._pad_output or not self._is_training else 1.0,
            aug_scale_max=self._aug_scale_max  if self._pad_output or not self._is_training else 1.0)
     
        category_mask = self._resize_and_crop_mask(
            category_mask,
            image_info,
            self._output_size if self._pad_output else crop_im_size,
            is_training=is_training)
        instance_mask = self._resize_and_crop_mask(
            instance_mask,
            image_info,
            self._output_size if self._pad_output else crop_im_size,
            is_training=is_training)
        
        (unique_ids, individual_masks) = self._get_individual_masks(
                instance_mask=instance_mask[:, :, 0])

        
        

        # Resize image and masks to output size.
        image = tf.image.resize(image, self._output_size, method='nearest')
        category_mask = tf.image.resize(category_mask, self._output_size, method='nearest')
        instance_mask = tf.image.resize(instance_mask, self._output_size, method='nearest')
        individual_masks = tf.image.resize(individual_masks, self._output_size, method='nearest')

        # pad the individual masks to the max number of instances and unique ids
#         individual_masks = tf.pad(individual_masks, [[0, self._max_instances - tf.shape(individual_masks)[0]], [0, 0], [0, 0], [0,0]], constant_values=self._ignore_label)
        unique_ids = tf.pad(unique_ids, [[0, self._max_instances - tf.shape(unique_ids)[0]]], constant_values=self._ignore_label)
        # Cast image to float and set shapes of output.
        image = tf.cast(image, dtype=self._dtype)
        category_mask = tf.cast(category_mask, dtype=self._dtype)
        instance_mask = tf.cast(instance_mask, dtype=self._dtype)
        individual_masks = tf.cast(individual_masks, dtype=self._dtype)
        unique_ids =  tf.cast(unique_ids, dtype=tf.float32)

        valid_mask = tf.not_equal(
            category_mask, self._ignore_label)
        things_mask = tf.not_equal(
            instance_mask, self._ignore_label)

        
        labels = {
            'category_mask': category_mask,
            'instance_mask': instance_mask,
            'valid_mask': valid_mask,
            'things_mask': things_mask,
            'image_info': image_info,
            'unique_ids': unique_ids,
            'individual_masks': individual_masks,
        }
        return image, labels

    def _parse_train_data(self, data):
        """Parses data for training."""
        return self._parse_data(data=data, is_training=True)

    def _parse_eval_data(self, data):
        """Parses data for evaluation."""
        return self._parse_data(data=data, is_training=False)

    
    def _get_individual_masks(self, instance_mask):
        
        unique_instance_ids, _ = tf.unique(tf.reshape(instance_mask, [-1]))
        individual_mask_list = tf.TensorArray(tf.float32, size=100) 
#                                               dynamic_size=True)
        counter = 0
        for instance_id in unique_instance_ids:
            # if instance_id == self._ignore_label:
            #     continue

            mask = tf.equal(instance_mask, instance_id)
            individual_mask_list = individual_mask_list.write(counter, tf.expand_dims(tf.cast(mask, tf.float32), axis=2))
            counter += 1

        for idx in tf.range(100-tf.size(unique_instance_ids)):
            new_mask = tf.zeros(tf.shape(instance_mask))
            individual_mask_list = individual_mask_list.write(idx, tf.expand_dims(tf.cast(new_mask, tf.float32), axis=2))
        
        return (unique_instance_ids, individual_mask_list.stack())

    def __call__(self, value):
        """Parses data to an image and associated training labels.
        Args:
          value: a string tensor holding a serialized tf.Example proto.
        Returns:
          image, labels: if is_training, see _parse_train_data.
          {'images': image, 'labels': labels}: if is_training
        """
        with tf.name_scope('parser'):
            data = self._decoder(value)

            if self._is_training:
                return self._parse_train_data(data)
            else:
                return self._parse_eval_data(data)
