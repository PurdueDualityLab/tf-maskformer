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

"""Convert raw COCO dataset to TFRecord format.

This scripts follows the label map decoder format and supports detection
boxes, instance masks and captions.

Example usage:
		python create_coco_tf_record.py --logtostderr \
			--image_dir="${TRAIN_IMAGE_DIR}" \
			--image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
			--object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
			--caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
			--output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
			--num_shards=100
"""

import collections
import json
import logging
import os
from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np

from pycocotools import mask
import tensorflow as tf
from official.vision.data import tfrecord_lib


flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
    '(PNG encoded) in the result. default: False.')
flags.DEFINE_multi_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string(
    'image_info_file', '', 'File containing image information. '
    'Tf Examples in the output files correspond to the image '
    'info entries in this file. If this file is not provided '
    'object_annotations_file is used if present. Otherwise, '
    'caption_annotations_file is used to get image info.')
flags.DEFINE_string(
    'object_annotations_file', '', 'File containing object '
    'annotations - boxes and instance masks.')
flags.DEFINE_string('caption_annotations_file', '', 'File containing image '
                    'captions.')
flags.DEFINE_string(
    'panoptic_annotations_file',
    '',
    'File containing panoptic '
    'annotations.')
flags.DEFINE_string('panoptic_masks_dir', '',
                    'Directory containing panoptic masks annotations.')
flags.DEFINE_boolean(
    'include_panoptic_masks', False, 'Whether to include category and '
    'instance masks in the result. These are required to run the PQ evaluator '
    'default: False.')
flags.DEFINE_boolean(
    'panoptic_skip_crowd', False, 'Whether to skip crowd or not for panoptic '
    'annotations. default: False.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', None,
    ('Number of parallel processes to use. '
     'If set to 0, disables multi-processing.'))


FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

_VOID_LABEL = 0
_VOID_INSTANCE_ID = 0
_THING_CLASS_ID = 1
_STUFF_CLASSES_OFFSET = 90

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1,
     "id": 14, "name": "parking meter"},
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
    {"color": [255, 208, 186], "isthing": 1,
     "id": 43, "name": "tennis racket"},
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
    {"color": [106, 154, 176], "isthing": 0,
     "id": 145, "name": "playingfield"},
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
    {"color": [183, 121, 142], "isthing": 0,
     "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0,
     "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0,
     "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0,
     "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0,
     "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0,
     "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0,
     "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0,
     "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0,
     "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0,
     "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0,
     "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0,
     "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0,
     "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0,
     "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]


def coco_segmentation_to_mask_png(segmentation, height, width, is_crowd):
  """Encode a COCO mask segmentation as PNG string."""
  run_len_encoding = mask.frPyObjects(segmentation, height, width)
  binary_mask = mask.decode(run_len_encoding)
  if not is_crowd:
    binary_mask = np.amax(binary_mask, axis=2)

  return tfrecord_lib.encode_mask_as_png(binary_mask)


def generate_coco_panoptics_masks(segments_info, mask_path,
                                  include_panoptic_masks,
                                  is_category_thing):
  # pylint: disable=line-too-long
  """Creates masks for panoptic segmentation task.
  Args:
    segments_info: a list of dicts, where each dict has keys: [u'id',
            u'category_id', u'area', u'bbox', u'iscrowd'], detailing information for
            each segment in the panoptic mask.
    mask_path: path to the panoptic mask.
    include_panoptic_masks: bool, when set to True, category and instance
            masks are included in the outputs. Set this to True, when using
            the Panoptic Quality evaluator.
    is_category_thing: a dict with category ids as keys and, 0/1 as values to
            represent "stuff" and "things" classes respectively.
  Returns:
    A dict with keys: [u'semantic_segmentation_mask', u'category_mask',
            u'instance_mask']. The dict contains 'category_mask' and 'instance_mask'
            only if `include_panoptic_eval_masks` is set to True.
  """
  rgb_mask = tfrecord_lib.read_image(mask_path)
  r, g, b = np.split(rgb_mask, 3, axis=-1)

  # decode rgb encoded panoptic mask to get segments ids
  # refer https://cocodataset.org/#format-data
  segments_encoded_mask = (r + g * 256 + b * (256**2)).squeeze()
  _meta = {}

  thing_dataset_id_to_contiguous_id = {}
  stuff_dataset_id_to_contiguous_id = {}
  # Reserve class :0 for background hence we add +1
  for i, cat in enumerate(COCO_CATEGORIES):
    if cat["isthing"]:
      thing_dataset_id_to_contiguous_id[cat["id"]] = i + 1
    else:
      stuff_dataset_id_to_contiguous_id[cat["id"]] = i + 1

  _meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
  _meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
  # All required masks
  semantic_segmentation_mask = np.ones_like(
      segments_encoded_mask, dtype=np.uint8) * _VOID_LABEL
  if include_panoptic_masks:
    category_mask = np.ones_like(
        segments_encoded_mask, dtype=np.uint8) * _VOID_LABEL
    instance_mask = np.ones_like(
        segments_encoded_mask, dtype=np.uint8) * _VOID_INSTANCE_ID
    contiguous_id_mask = np.ones_like(
        segments_encoded_mask, dtype=np.uint8) * _VOID_INSTANCE_ID

  class_ids = []
  instance_ids = []
  for idx, segment in enumerate(segments_info):
    segment_id = segment['id']
    category_id = segment['category_id']
    is_crowd = segment['iscrowd']

    if FLAGS.panoptic_skip_crowd and is_crowd:
      continue

    if category_id in _meta["thing_dataset_id_to_contiguous_id"]:
      contiguous_id = _meta["thing_dataset_id_to_contiguous_id"][category_id]
    else:
      contiguous_id = _meta["stuff_dataset_id_to_contiguous_id"][category_id]

    if is_category_thing[category_id]:
      # This for thing
      encoded_category_id = _THING_CLASS_ID
      instance_id = idx + 1
    else:
      # This is for stuff (for stuff no instance id)
      encoded_category_id = category_id - _STUFF_CLASSES_OFFSET
      instance_id = _VOID_INSTANCE_ID

    segment_mask = (segments_encoded_mask == segment_id)
    semantic_segmentation_mask[segment_mask] = encoded_category_id

    if include_panoptic_masks:
      category_mask[segment_mask] = category_id
      instance_mask[segment_mask] = instance_id
      contiguous_id_mask[segment_mask] = contiguous_id
      class_ids.append(contiguous_id)
      instance_ids.append(instance_id)

  outputs = {
      'semantic_segmentation_mask': tfrecord_lib.encode_mask_as_png(
          semantic_segmentation_mask)
  }
  if include_panoptic_masks:
    outputs.update({
        'category_mask': tfrecord_lib.encode_mask_as_png(category_mask),
        'instance_mask': tfrecord_lib.encode_mask_as_png(instance_mask),
        'class_ids': class_ids,
        'contiguous_id_mask': tfrecord_lib.encode_mask_as_png(contiguous_id_mask),
        'instance_ids': instance_ids
    })
  return outputs


def coco_annotations_to_lists(bbox_annotations, id_to_name_map,
                              image_height, image_width, include_masks):
  """Converts COCO annotations to feature lists."""

  data = dict((k, list()) for k in
              ['xmin', 'xmax', 'ymin', 'ymax', 'is_crowd',
               'category_id', 'category_names', 'area'])
  if include_masks:
    data['encoded_mask_png'] = []

  num_annotations_skipped = 0

  for object_annotations in bbox_annotations:
    (x, y, width, height) = tuple(object_annotations['bbox'])

    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    data['xmin'].append(float(x) / image_width)
    data['xmax'].append(float(x + width) / image_width)
    data['ymin'].append(float(y) / image_height)
    data['ymax'].append(float(y + height) / image_height)
    data['is_crowd'].append(object_annotations['iscrowd'])
    category_id = int(object_annotations['category_id'])
    data['category_id'].append(category_id)
    data['category_names'].append(id_to_name_map[category_id].encode('utf8'))
    data['area'].append(object_annotations['area'])

    if include_masks:
      data['encoded_mask_png'].append(
          coco_segmentation_to_mask_png(object_annotations['segmentation'],
                                        image_height, image_width,
                                        object_annotations['iscrowd'])
      )

  return data, num_annotations_skipped


def bbox_annotations_to_feature_dict(
        bbox_annotations,
        image_height,
        image_width,
        id_to_name_map,
        include_masks):
  """Convert COCO annotations to an encoded feature dict."""

  data, num_skipped = coco_annotations_to_lists(
      bbox_annotations, id_to_name_map, image_height, image_width,
      include_masks)
  feature_dict = {}
  if len(bbox_annotations) != num_skipped:
    feature_dict = {
        'image/object/bbox/xmin': tfrecord_lib.convert_to_feature(data['xmin']),
        'image/object/bbox/xmax': tfrecord_lib.convert_to_feature(data['xmax']),
        'image/object/bbox/ymin': tfrecord_lib.convert_to_feature(data['ymin']),
        'image/object/bbox/ymax': tfrecord_lib.convert_to_feature(data['ymax']),
        'image/object/class/text': tfrecord_lib.convert_to_feature(
            data['category_names']
        ),
        'image/object/class/label': tfrecord_lib.convert_to_feature(
            data['category_id']
        ),
        'image/object/is_crowd': tfrecord_lib.convert_to_feature(
            data['is_crowd']
        ),
        'image/object/area': tfrecord_lib.convert_to_feature(
            data['area'], 'float_list'
        ),
    }
    if include_masks:
      feature_dict['image/object/mask'] = tfrecord_lib.convert_to_feature(
          data['encoded_mask_png']
      )

  return feature_dict, num_skipped


def encode_caption_annotations(caption_annotations):
  captions = []
  for caption_annotation in caption_annotations:
    captions.append(caption_annotation['caption'].encode('utf8'))

  return captions


def create_tf_example(image,
                      image_dirs,
                      panoptic_masks_dir=None,
                      bbox_annotations=None,
                      id_to_name_map=None,
                      caption_annotations=None,
                      panoptic_annotation=None,
                      is_category_thing=None,
                      include_panoptic_masks=False,
                      include_masks=False):
  # pylint: disable=line-too-long
  """Converts image and annotations to a tf.Example proto.
  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
            u'width', u'date_captured', u'flickr_url', u'id']
    image_dirs: list of directories containing the image files.
    panoptic_masks_dir: `str` of the panoptic masks directory.
    bbox_annotations:
            list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
                    u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
                    coordinates in the official COCO dataset are given as [x, y, width,
                    height] tuples using absolute coordinates where x, y represent the
                    top-left (0-indexed) corner.  This function converts to the format
                    expected by the Tensorflow Object Detection API (which is which is
                    [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
                    size).
    id_to_name_map: a dict mapping category IDs to string names.
    caption_annotations:
            list of dict with keys: [u'id', u'image_id', u'str'].
    panoptic_annotation: dict with keys: [u'image_id', u'file_name',
            u'segments_info']. Where the value for segments_info is a list of dicts,
            with each dict containing information for a single segment in the mask.
    is_category_thing: `bool`, whether it is a category thing.
    include_panoptic_masks: `bool`, whether to include panoptic masks.
    include_masks: Whether to include instance segmentations masks
            (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG,
            does not exist, or is not unique across image directories.
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  if len(image_dirs) > 1:
    full_paths = [os.path.join(image_dir, filename)
                  for image_dir in image_dirs]
    full_existing_paths = [p for p in full_paths if tf.io.gfile.exists(p)]
    if not full_existing_paths:
      raise ValueError(
          '{} does not exist across image directories.'.format(filename))
    if len(full_existing_paths) > 1:
      raise ValueError(
          '{} is not unique across image directories'.format(filename))
    full_path, = full_existing_paths
  # If there is only one image directory, it's not worth checking for existence,
  # since trying to open the file will raise an informative error message if it
  # does not exist.
  else:
    image_dir, = image_dirs
    full_path = os.path.join(image_dir, filename)

  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()

  feature_dict = tfrecord_lib.image_info_to_feature_dict(
      image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

  num_annotations_skipped = 0
  if bbox_annotations:
    box_feature_dict, num_skipped = bbox_annotations_to_feature_dict(
        bbox_annotations, image_height, image_width, id_to_name_map,
        include_masks)
    num_annotations_skipped += num_skipped
    feature_dict.update(box_feature_dict)

  if caption_annotations:
    encoded_captions = encode_caption_annotations(caption_annotations)
    feature_dict.update(
        {'image/caption': tfrecord_lib.convert_to_feature(encoded_captions)})

  if panoptic_annotation:
    segments_info = panoptic_annotation['segments_info']

    panoptic_mask_filename = os.path.join(
        panoptic_masks_dir,
        panoptic_annotation['file_name'])
    encoded_panoptic_masks = generate_coco_panoptics_masks(
        segments_info, panoptic_mask_filename, include_panoptic_masks,
        is_category_thing)
    feature_dict.update(
        {'image/segmentation/class/encoded': tfrecord_lib.convert_to_feature(
            encoded_panoptic_masks['semantic_segmentation_mask'])})

    if include_panoptic_masks:
      feature_dict.update({
          'image/panoptic/category_mask': tfrecord_lib.convert_to_feature(
                          encoded_panoptic_masks['category_mask']),
          'image/panoptic/instance_mask': tfrecord_lib.convert_to_feature(
              encoded_panoptic_masks['instance_mask']),
          'image/panoptic/class_ids': tfrecord_lib.convert_to_feature(
              encoded_panoptic_masks['class_ids'], value_type="int64_list"),
          'image/panoptic/contiguous_mask': tfrecord_lib.convert_to_feature(
              encoded_panoptic_masks['contiguous_id_mask']),
          'image/panoptic/instance_ids': tfrecord_lib.convert_to_feature(
              encoded_panoptic_masks['instance_ids'], value_type="int64_list")
      })

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example, num_annotations_skipped


def _load_object_annotations(object_annotations_file):
  """Loads object annotation JSON file."""
  with tf.io.gfile.GFile(object_annotations_file, 'r') as fid:
    obj_annotations = json.load(fid)

  images = obj_annotations['images']
  id_to_name_map = dict((element['id'], element['name']) for element in
                        obj_annotations['categories'])

  img_to_obj_annotation = collections.defaultdict(list)
  logging.info('Building bounding box index.')
  for annotation in obj_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_obj_annotation[image_id].append(annotation)

  missing_annotation_count = 0
  for image in images:
    image_id = image['id']
    if image_id not in img_to_obj_annotation:
      missing_annotation_count += 1

  logging.info('%d images are missing bboxes.', missing_annotation_count)

  return img_to_obj_annotation, id_to_name_map


def _load_caption_annotations(caption_annotations_file):
  """Loads caption annotation JSON file."""
  with tf.io.gfile.GFile(caption_annotations_file, 'r') as fid:
    caption_annotations = json.load(fid)

  img_to_caption_annotation = collections.defaultdict(list)
  logging.info('Building caption index.')
  for annotation in caption_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_caption_annotation[image_id].append(annotation)

  missing_annotation_count = 0
  images = caption_annotations['images']
  for image in images:
    image_id = image['id']
    if image_id not in img_to_caption_annotation:
      missing_annotation_count += 1

  logging.info('%d images are missing captions.', missing_annotation_count)

  return img_to_caption_annotation


def _load_panoptic_annotations(panoptic_annotations_file):
  """Loads panoptic annotation from file."""
  with tf.io.gfile.GFile(panoptic_annotations_file, 'r') as fid:
    panoptic_annotations = json.load(fid)

  img_to_panoptic_annotation = dict()
  logging.info('Building panoptic index.')
  for annotation in panoptic_annotations['annotations']:
    image_id = annotation['image_id']
    img_to_panoptic_annotation[image_id] = annotation

  is_category_thing = dict()
  for category_info in panoptic_annotations['categories']:
    is_category_thing[category_info['id']] = category_info['isthing'] == 1

  missing_annotation_count = 0
  images = panoptic_annotations['images']
  for image in images:
    image_id = image['id']
    if image_id not in img_to_panoptic_annotation:
      missing_annotation_count += 1
  logging.info(
      '%d images are missing panoptic annotations.', missing_annotation_count)

  return img_to_panoptic_annotation, is_category_thing


def _load_images_info(images_info_file):
  with tf.io.gfile.GFile(images_info_file, 'r') as fid:
    info_dict = json.load(fid)
  return info_dict['images']


def generate_annotations(images, image_dirs,
                         panoptic_masks_dir=None,
                         img_to_obj_annotation=None,
                         img_to_caption_annotation=None,
                         img_to_panoptic_annotation=None,
                         is_category_thing=None,
                         id_to_name_map=None,
                         include_panoptic_masks=False,
                         include_masks=False):
  """Generator for COCO annotations."""

  for image in images:
    object_annotation = (img_to_obj_annotation.get(image['id'], None) if
                         img_to_obj_annotation else None)

    caption_annotaion = (img_to_caption_annotation.get(image['id'], None) if
                         img_to_caption_annotation else None)

    panoptic_annotation = (img_to_panoptic_annotation.get(image['id'], None) if
                           img_to_panoptic_annotation else None)
    yield (image, image_dirs, panoptic_masks_dir, object_annotation,
           id_to_name_map, caption_annotaion, panoptic_annotation,
           is_category_thing, include_panoptic_masks, include_masks)


def _create_tf_record_from_coco_annotations(images_info_file,
                                            image_dirs,
                                            output_path,
                                            num_shards,
                                            object_annotations_file=None,
                                            caption_annotations_file=None,
                                            panoptic_masks_dir=None,
                                            panoptic_annotations_file=None,
                                            include_panoptic_masks=False,
                                            include_masks=False):
  # pylint: disable=line-too-long
  """Loads COCO annotation json files and converts to tf.Record format.
  Args:
    images_info_file: JSON file containing image info. The number of tf.Examples
            in the output tf Record files is exactly equal to the number of image info
            entries in this file. This can be any of train/val/test annotation json
            files Eg. 'image_info_test-dev2017.json',
            'instance_annotations_train2017.json',
            'caption_annotations_train2017.json', etc.
    image_dirs: List of directories containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: Number of output files to create.
    object_annotations_file: JSON file containing bounding box annotations.
    caption_annotations_file: JSON file containing caption annotations.
    panoptic_masks_dir: Directory containing panoptic masks.
    panoptic_annotations_file: JSON file containing panoptic annotations.
    include_panoptic_masks: Whether to include 'category_mask'
            and 'instance_mask', which is required by the panoptic quality evaluator.
    include_masks: Whether to include instance segmentations masks
            (PNG encoded) in the result. default: False.
  """

  logging.info('writing to output path: %s', output_path)

  images = _load_images_info(images_info_file)

  img_to_obj_annotation = None
  img_to_caption_annotation = None
  id_to_name_map = None
  img_to_panoptic_annotation = None
  is_category_thing = None
  if object_annotations_file:
    img_to_obj_annotation, id_to_name_map = (
        _load_object_annotations(object_annotations_file))
  if caption_annotations_file:
    img_to_caption_annotation = (
        _load_caption_annotations(caption_annotations_file))
  if panoptic_annotations_file:
    img_to_panoptic_annotation, is_category_thing = (
        _load_panoptic_annotations(panoptic_annotations_file))

  coco_annotations_iter = generate_annotations(
      images=images,
      image_dirs=image_dirs,
      panoptic_masks_dir=panoptic_masks_dir,
      img_to_obj_annotation=img_to_obj_annotation,
      img_to_caption_annotation=img_to_caption_annotation,
      img_to_panoptic_annotation=img_to_panoptic_annotation,
      is_category_thing=is_category_thing,
      id_to_name_map=id_to_name_map,
      include_panoptic_masks=include_panoptic_masks,
      include_masks=include_masks)

  num_skipped = tfrecord_lib.write_tf_record_dataset(
      output_path, coco_annotations_iter, create_tf_example, num_shards,
      multiple_processes=_NUM_PROCESSES.value)

  logging.info('Finished writing, skipped %d annotations.', num_skipped)


def main(_):
  assert FLAGS.image_dir, '`image_dir` missing.'
  assert (FLAGS.image_info_file or FLAGS.object_annotations_file or
          FLAGS.caption_annotations_file), ('All annotation files are '
                                            'missing.')
  if FLAGS.image_info_file:
    images_info_file = FLAGS.image_info_file
  elif FLAGS.object_annotations_file:
    images_info_file = FLAGS.object_annotations_file
  else:
    images_info_file = FLAGS.caption_annotations_file

  directory = os.path.dirname(FLAGS.output_file_prefix)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  _create_tf_record_from_coco_annotations(images_info_file, FLAGS.image_dir,
                                          FLAGS.output_file_prefix,
                                          FLAGS.num_shards,
                                          FLAGS.object_annotations_file,
                                          FLAGS.caption_annotations_file,
                                          FLAGS.panoptic_masks_dir,
                                          FLAGS.panoptic_annotations_file,
                                          FLAGS.include_panoptic_masks,
                                          FLAGS.include_masks)


if __name__ == '__main__':
  app.run(main)
