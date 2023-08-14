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

RESIZE_SCALES = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)

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
        self._panoptic_contigious_mask_key = 'image/panoptic/contiguous_mask'
        self._class_ids_key = 'image/panoptic/class_ids'
        self._class_instance_ids_key = 'image/panoptic/instance_ids'
        self._image_height_key = 'image/height'
        self._image_width_key = 'image/width'
        self._image_key = ""
        self._panoptic_keys_to_features = {
            self._panoptic_category_mask_key:
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            self._panoptic_instance_mask_key:
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            self._panoptic_contigious_mask_key:
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            self._class_ids_key:
                tf.io.VarLenFeature(tf.int64),    
            self._class_instance_ids_key:
                tf.io.VarLenFeature(tf.int64),    
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
        contigious_mask = tf.io.decode_png(
            parsed_tensors[self._panoptic_contigious_mask_key], channels=1)
        class_ids = parsed_tensors[self._class_ids_key]
        instance_ids = parsed_tensors[self._class_instance_ids_key]
        category_mask.set_shape([None, None, 1])
        instance_mask.set_shape([None, None, 1])
        contigious_mask.set_shape([None, None, 1])
        decoded_tensors.update({
            'groundtruth_panoptic_category_mask': category_mask,
            'groundtruth_panoptic_instance_mask': instance_mask,
            'groundtruth_panoptic_contigious_mask': contigious_mask,
            'groundtruth_panoptic_class_ids': class_ids,
            'groundtruth_panoptic_instance_ids': instance_ids,
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
        print("params:", params)
        self._output_size = params.output_size
        self._mask_null = 0
        self._dtype = params.dtype
        self._pad_output = params.pad_output
        self._seed = params.seed
        # TODO
        self._max_instances = 100
        self._decoder = decoder_fn
        self._resize_scales = RESIZE_SCALES
        if self._pad_output == True and self._output_size is None:
            raise Exception("Error: no output pad provided")
        
        self._groundtruth_padded_size = params.groundtruth_padded_size
        
        if self._decoder == None:
            self._decoder = TfExampleDecoder()
        
        self._is_training = is_training

        if is_training == None:
            self._is_training = True
        

        self._ignore_label = params.ignore_label
        # Data augmentation
        self._aug_rand_hflip = params.aug_rand_hflip
        self._aug_scale_min = params.aug_scale_min
        self._aug_scale_max = params.aug_scale_max
        #Cropping:
        self._min_scale = params.min_scale
        self._aspect_ratio_range = params.aspect_ratio_range
        self._min_overlap_params = params.min_overlap_params
        self._max_retry = params.max_retry


    # def _resize_and_crop_mask(self, mask, image_info, is_training):
    #     """Resizes and crops mask using `image_info` dict."""
    #     offset = image_info[3, : ]
    #     im_height = int(image_info[0][0])
    #     im_width = int(image_info[0][1])
        
    #     mask = tf.reshape(mask, shape=[1, im_height, im_width, 1])
    #     mask += 1

    #     if is_training:
    #         image_scale = image_info[2, :]
    #         offset = image_info[3, :]
    #         mask = preprocess_ops.resize_and_crop_masks(
    #             mask,
    #             image_scale,
    #             self._output_size,
    #             offset)
    #     else:
    #         mask = tf.image.pad_to_bounding_box(
    #             mask, 0, 0,
    #             self._groundtruth_padded_size[0],
    #             self._groundtruth_padded_size[1])
    #     mask -= 1

    #     # Assign ignore label to the padded region.
    #     mask = tf.where(
    #         tf.equal(mask, -1),
    #         self._ignore_label * tf.ones_like(mask),
    #         mask)
    #     mask = tf.squeeze(mask, axis=0)
        
    #     return mask
    
    def _resize_and_crop_mask(self, mask, image_info, crop_dims, is_training):
        """Resizes and crops mask using `image_info` dict."""
       
        image_scale = image_info[2, :]
        offset = image_info[3, : ]
        im_height = int(image_info[0][0])
        im_width = int(image_info[0][1])
        
        mask = tf.reshape(mask, shape=[1, im_height, im_width, 1])
        mask += 1

        if is_training:
            assert crop_dims != None
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
    # def _parse_data(self, data, is_training):
    #     image = data['image']
    
    #     # Normalize and prepare image and masks
    #     image = preprocess_ops.normalize_image(image)
    #     category_mask = tf.cast(
    #         data['groundtruth_panoptic_category_mask'][:, :, 0],
    #         dtype=tf.float32)
    #     instance_mask = tf.cast(
    #         data['groundtruth_panoptic_instance_mask'][:, :, 0],
    #         dtype=tf.float32)
    #     contigious_mask = tf.cast(data['groundtruth_panoptic_contigious_mask'][:, :, 0],
    #         dtype=tf.float32)
    #     class_ids = tf.sparse.to_dense(data['groundtruth_panoptic_class_ids'], default_value=0)
    #     instance_ids = tf.sparse.to_dense(data['groundtruth_panoptic_instance_ids'], default_value=0)
    #     class_ids = tf.cast(class_ids, dtype=tf.float32)
    #     instance_ids = tf.cast(instance_ids, dtype=tf.float32)
        
    #     # Flips image randomly during training.
    #     if self._aug_rand_hflip and is_training:
           
    #         masks = tf.stack([category_mask, instance_mask, contigious_mask], axis=0)
    #         image, _, masks = preprocess_ops.random_horizontal_flip(
    #             image=image, 
    #             masks=masks,
    #             seed = self._seed,
    #             prob=0.5)

    #         category_mask = masks[0]
    #         instance_mask = masks[1]
    #         contigious_mask = masks[2]

    #     do_crop = tf.greater(tf.random.uniform([]), 0.5)
    #     # masks = tf.stack([category_mask, instance_mask, contigious_mask], axis=0)
    #     # masks = tf.expand_dims(masks, -1)
        
    #     if do_crop:
    #         print("///////////////////////// Inside Cropping /////////////////////////")
    #         # category_mask = masks[0]
    #         # instance_mask = masks[1]
    #         # contigious_mask = masks[2]
    #         index = tf.random.categorical(tf.zeros([1, 3]), 1)[0]
    #         scales = tf.gather([400.0, 500.0, 600.0], index, axis=0)
    #         short_side = scales[0]
    #         image, image_info = preprocess_ops.resize_image(image, short_side)
    #         # image_info[0] --> original image size
    #         # image_info[1] --> scaled image size
    #         # image_info[2] --> y_scale, x_scale
    #         # image_info[3] --> offset

    #         category_mask = self._resize_and_crop_mask(category_mask, image_info, is_training)
    #         instance_mask = self._resize_and_crop_mask(instance_mask, image_info, is_training)
    #         contigious_mask = self._resize_and_crop_mask(contigious_mask, image_info, is_training)

    #         # Do cropping
    #         shape = tf.cast(image_info[1], dtype=tf.int32) # resized image h,w
    #         h = tf.random.uniform([],
    #                         384,
    #                         tf.math.minimum(shape[0], 600),
    #                         dtype=tf.int32)
    #         w = tf.random.uniform([],
    #                         384,
    #                         tf.math.minimum(shape[1], 600),
    #                         dtype=tf.int32)
    #         i = tf.random.uniform([], 0, shape[0] - h + 1, dtype=tf.int32)
    #         j = tf.random.uniform([], 0, shape[1] - w + 1, dtype=tf.int32)
    #         image = tf.image.crop_to_bounding_box(image, i, j, h, w)
    #         category_mask = tf.image.crop_to_bounding_box(category_mask, i, j, h, w)
    #         instance_mask = tf.image.crop_to_bounding_box(instance_mask, i, j, h, w)
    #         contigious_mask = tf.image.crop_to_bounding_box(contigious_mask, i, j, h, w)
    #         # masks = tf.stack([category_mask, instance_mask, contigious_mask], axis=0)
            
    #     scales = tf.constant(self._resize_scales, dtype=tf.float32)
    #     index = tf.random.categorical(tf.zeros([1, 11]), 1)[0]
    #     scales = tf.gather(scales, index, axis=0)

    #     short_side = scales[0]
    #     image, image_info = preprocess_ops.resize_image(image, short_side,
    #                                                 max(self._output_size))
    #     # Resize and crop masks.
    #     # masks = preprocess_ops.resize_and_crop_masks(
    #     #         masks,
    #     #         image_info[2, :],
    #     #         image_info[1, :],
    #     #         image_info[3, :],  ignore_label=self._ignore_label)
    #     category_mask = self._resize_and_crop_mask(category_mask, image_info, is_training)
    #     instance_mask = self._resize_and_crop_mask(instance_mask, image_info, is_training)
    #     contigious_mask = self._resize_and_crop_mask(contigious_mask, image_info, is_training)

    #     # category_mask = masks[0]
    #     # instance_mask = masks[1]
    #     # contigious_mask = masks[2]

    #     image = tf.image.pad_to_bounding_box(image, 0, 0, self._output_size[0],
    #                                      self._output_size[1])
    #     category_mask = tf.image.pad_to_bounding_box(category_mask, 0, 0, self._output_size[0], self._output_size[1])
    #     instance_mask = tf.image.pad_to_bounding_box(instance_mask, 0, 0, self._output_size[0], self._output_size[1])
    #     contigious_mask = tf.image.pad_to_bounding_box(contigious_mask, 0, 0, self._output_size[0], self._output_size[1])

    #     individual_masks, classes = self._get_individual_masks(
    #             class_ids=class_ids,contig_instance_mask=contigious_mask, instance_id = instance_ids, instance_mask=instance_mask)

    #     # image = tf.image.resize(image, self._output_size, method='nearest')
    #     # category_mask = tf.image.resize(category_mask, self._output_size, method='nearest')
    #     # instance_mask = tf.image.resize(instance_mask, self._output_size, method='nearest')
    #     # individual_masks = tf.image.resize(individual_masks, self._output_size, method='nearest')

    #     # Cast image to float and set shapes of output.
    #     image = tf.cast(image, dtype=self._dtype)
    #     category_mask = tf.cast(category_mask, dtype=self._dtype)
    #     instance_mask = tf.cast(instance_mask, dtype=self._dtype)
    #     individual_masks = tf.cast(individual_masks, dtype=self._dtype)
        

    #     valid_mask = tf.not_equal(
    #         category_mask, self._ignore_label)
    #     things_mask = tf.not_equal(
    #         instance_mask, self._ignore_label)

       
    #     labels = {
    #         'category_mask': category_mask,
    #         'instance_mask': instance_mask,
    #         'contigious_mask': contigious_mask, 
    #         'valid_mask': valid_mask,
    #         'things_mask': things_mask,
    #         'image_info': image_info,
    #         'unique_ids': classes,
    #         'individual_masks': individual_masks,
    #     }
    #     return image, labels
    def _parse_data(self, data, is_training):
        image = data['image']
        # Normalize and prepare image and masks
        image = preprocess_ops.normalize_image(image)
        category_mask = tf.cast(
            data['groundtruth_panoptic_category_mask'][:, :, 0],
            dtype=tf.float32)
        instance_mask = tf.cast(
            data['groundtruth_panoptic_instance_mask'][:, :, 0],
            dtype=tf.float32)
        contigious_mask = tf.cast(data['groundtruth_panoptic_contigious_mask'][:, :, 0],
            dtype=tf.float32)
        class_ids = tf.sparse.to_dense(data['groundtruth_panoptic_class_ids'], default_value=0)
        instance_ids = tf.sparse.to_dense(data['groundtruth_panoptic_instance_ids'], default_value=0)
        class_ids = tf.cast(class_ids, dtype=tf.float32)
        instance_ids = tf.cast(instance_ids, dtype=tf.float32)
        image_info = None
        # Flips image randomly during training.
        if self._aug_rand_hflip and is_training:
            masks = tf.stack([category_mask, instance_mask, contigious_mask], axis=0)
            image, _, masks = preprocess_ops.random_horizontal_flip(
                image=image, 
                masks=masks,
                seed = self._seed,
                prob=tf.where(self._aug_rand_hflip, 0.5, 0.0))

            category_mask = masks[0]
            instance_mask = masks[1]
            contigious_mask = masks[2]

            # Resize and crops image.
            masks = tf.stack([category_mask, instance_mask, contigious_mask], axis=0)
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
            contigious_mask = tf.squeeze(masks[2])
            
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
            contigious_mask = self._resize_and_crop_mask(
                contigious_mask,
                image_info,
                self._output_size if self._pad_output else crop_im_size,
                is_training=is_training)
        
        if not is_training:
            # Resize the image
            image = tf.image.resize(image, self._output_size, method='nearest')
            # expand the dims for the each mask
            category_mask = tf.expand_dims(category_mask, -1)
            instance_mask = tf.expand_dims(instance_mask, -1)
            contigious_mask = tf.expand_dims(contigious_mask, -1)

            category_mask = tf.image.resize(category_mask, self._output_size, method='nearest')
            instance_mask = tf.image.resize(instance_mask, self._output_size, method='nearest')
            contigious_mask = tf.image.resize(contigious_mask, self._output_size, method='nearest')
        

        individual_masks, classes = self._get_individual_masks(
                class_ids=class_ids,contig_instance_mask=contigious_mask, instance_id = instance_ids, instance_mask=instance_mask)

        # Resize image and masks to output size.
        image = tf.image.resize(image, self._output_size, method='nearest')
        category_mask = tf.image.resize(category_mask, self._output_size, method='nearest')
        instance_mask = tf.image.resize(instance_mask, self._output_size, method='nearest')
        individual_masks = tf.image.resize(individual_masks, self._output_size, method='nearest')
        
        # Cast image to float and set shapes of output.
        image = tf.cast(image, dtype=self._dtype)
        category_mask = tf.cast(category_mask, dtype=self._dtype)
        instance_mask = tf.cast(instance_mask, dtype=self._dtype)
        individual_masks = tf.cast(individual_masks, dtype=self._dtype)
        

        valid_mask = tf.not_equal(
            category_mask, self._ignore_label)
        things_mask = tf.not_equal(
            instance_mask, self._ignore_label)

       
        labels = {
            'category_mask': category_mask,
            'instance_mask': instance_mask,
            'contigious_mask': contigious_mask, 
            'valid_mask': valid_mask,
            'things_mask': things_mask,
            'image_info': image_info,
            'unique_ids': classes,
            'individual_masks': individual_masks,
        }
        return image, labels
    
    def _parse_train_data(self, data):
        """Parses data for training."""
        return self._parse_data(data=data, is_training=True)

    def _parse_eval_data(self, data):
        """Parses data for evaluation."""
        return self._parse_data(data=data, is_training=False)

    
    def _get_individual_masks(self, class_ids, contig_instance_mask, instance_id, instance_mask):
        
        individual_mask_list = tf.TensorArray(tf.float32, size=self._max_instances) 
        classes_list = tf.TensorArray(tf.float32, size=self._max_instances)
        counter = 0
        counter_1 = 0

        for class_id in class_ids:
            mask = tf.equal(contig_instance_mask, class_id)
            mask = tf.logical_and(mask, tf.equal(instance_mask, instance_id[counter]))
            if tf.greater(tf.reduce_sum(tf.cast(mask,tf.float32), [0,1,2]),0):
                classes_list = classes_list.write(counter_1, tf.cast(class_id, tf.float32))
                individual_mask_list = individual_mask_list.write(counter, tf.cast(mask, tf.float32))
                counter_1 += 1
                counter += 1
            else:
                classes_list = classes_list.write(counter_1, tf.cast(133, tf.float32))
                new_mask = tf.zeros(tf.shape(contig_instance_mask))
                individual_mask_list = individual_mask_list.write(counter, tf.cast(new_mask, tf.float32))
                counter_1 += 1
                counter += 1
            
        for idx in tf.range(100-tf.size(class_ids)):
            new_mask = tf.zeros(tf.shape(contig_instance_mask))
            individual_mask_list = individual_mask_list.write(counter, tf.cast(new_mask, tf.float32))
            classes_list = classes_list.write(counter_1, tf.cast(133, tf.float32))
            counter_1 += 1
            counter += 1
        
        return individual_mask_list.stack(), classes_list.stack()

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
