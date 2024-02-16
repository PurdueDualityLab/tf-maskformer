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


"""Non-contigious class ids to contigious class ids mapping."""

import tensorflow as tf

# pylint: disable=line-too-long
COCO_CATEGORIES = [{"color": [220, 20, 60], "isthing":1, "id":1, "name":"person", "contiguous_id":1}, {"color": [119, 11, 32], "isthing":1, "id":2, "name":"bicycle", "contiguous_id":2}, {"color": [0, 0, 142], "isthing":1, "id":3, "name":"car", "contiguous_id":3}, {"color": [0, 0, 230], "isthing":1, "id":4, "name":"motorcycle", "contiguous_id":4}, {"color": [106, 0, 228], "isthing":1, "id":5, "name":"airplane", "contiguous_id":5}, {"color": [0, 60, 100], "isthing":1, "id":6, "name":"bus", "contiguous_id":6}, {"color": [0, 80, 100], "isthing":1, "id":7, "name":"train", "contiguous_id":7}, {"color": [0, 0, 70], "isthing":1, "id":8, "name":"truck", "contiguous_id":8}, {"color": [0, 0, 192], "isthing":1, "id":9, "name":"boat", "contiguous_id":9}, {"color": [250, 170, 30], "isthing":1, "id":10, "name":"traffic light", "contiguous_id":10}, {"color": [100, 170, 30], "isthing":1, "id":11, "name":"fire hydrant", "contiguous_id":11}, {"color": [220, 220, 0], "isthing":1, "id":13, "name":"stop sign", "contiguous_id":12}, {"color": [175, 116, 175], "isthing":1, "id":14, "name":"parking meter", "contiguous_id":13}, {"color": [250, 0, 30], "isthing":1, "id":15, "name":"bench", "contiguous_id":14}, {"color": [165, 42, 42], "isthing":1, "id":16, "name":"bird", "contiguous_id":15}, {"color": [255, 77, 255], "isthing":1, "id":17, "name":"cat", "contiguous_id":16}, {"color": [0, 226, 252], "isthing":1, "id":18, "name":"dog", "contiguous_id":17}, {"color": [182, 182, 255], "isthing":1, "id":19, "name":"horse", "contiguous_id":18}, {"color": [0, 82, 0], "isthing":1, "id":20, "name":"sheep", "contiguous_id":19}, {"color": [120, 166, 157], "isthing":1, "id":21, "name":"cow", "contiguous_id":20}, {"color": [110, 76, 0], "isthing":1, "id":22, "name":"elephant", "contiguous_id":21}, {"color": [174, 57, 255], "isthing":1, "id":23, "name":"bear", "contiguous_id":22}, {"color": [199, 100, 0], "isthing":1, "id":24, "name":"zebra", "contiguous_id":23}, {"color": [72, 0, 118], "isthing":1, "id":25, "name":"giraffe", "contiguous_id":24}, {"color": [255, 179, 240], "isthing":1, "id":27, "name":"backpack", "contiguous_id":25}, {"color": [0, 125, 92], "isthing":1, "id":28, "name":"umbrella", "contiguous_id":26}, {"color": [209, 0, 151], "isthing":1, "id":31, "name":"handbag", "contiguous_id":27}, {"color": [188, 208, 182], "isthing":1, "id":32, "name":"tie", "contiguous_id":28}, {"color": [0, 220, 176], "isthing":1, "id":33, "name":"suitcase", "contiguous_id":29}, {"color": [255, 99, 164], "isthing":1, "id":34, "name":"frisbee", "contiguous_id":30}, {"color": [92, 0, 73], "isthing":1, "id":35, "name":"skis", "contiguous_id":31}, {"color": [133, 129, 255], "isthing":1, "id":36, "name":"snowboard", "contiguous_id":32}, {"color": [78, 180, 255], "isthing":1, "id":37, "name":"sports ball", "contiguous_id":33}, {"color": [0, 228, 0], "isthing":1, "id":38, "name":"kite", "contiguous_id":34}, {"color": [174, 255, 243], "isthing":1, "id":39, "name":"baseball bat", "contiguous_id":35}, {"color": [45, 89, 255], "isthing":1, "id":40, "name":"baseball glove", "contiguous_id":36}, {"color": [134, 134, 103], "isthing":1, "id":41, "name":"skateboard", "contiguous_id":37}, {"color": [145, 148, 174], "isthing":1, "id":42, "name":"surfboard", "contiguous_id":38}, {"color": [255, 208, 186], "isthing":1, "id":43, "name":"tennis racket", "contiguous_id":39}, {"color": [197, 226, 255], "isthing":1, "id":44, "name":"bottle", "contiguous_id":40}, {"color": [171, 134, 1], "isthing":1, "id":46, "name":"wine glass", "contiguous_id":41}, {"color": [109, 63, 54], "isthing":1, "id":47, "name":"cup", "contiguous_id":42}, {"color": [207, 138, 255], "isthing":1, "id":48, "name":"fork", "contiguous_id":43}, {"color": [151, 0, 95], "isthing":1, "id":49, "name":"knife", "contiguous_id":44}, {"color": [9, 80, 61], "isthing":1, "id":50, "name":"spoon", "contiguous_id":45}, {"color": [84, 105, 51], "isthing":1, "id":51, "name":"bowl", "contiguous_id":46}, {"color": [74, 65, 105], "isthing":1, "id":52, "name":"banana", "contiguous_id":47}, {"color": [166, 196, 102], "isthing":1, "id":53, "name":"apple", "contiguous_id":48}, {"color": [208, 195, 210], "isthing":1, "id":54, "name":"sandwich", "contiguous_id":49}, {"color": [255, 109, 65], "isthing":1, "id":55, "name":"orange", "contiguous_id":50}, {"color": [0, 143, 149], "isthing":1, "id":56, "name":"broccoli", "contiguous_id":51}, {"color": [179, 0, 194], "isthing":1, "id":57, "name":"carrot", "contiguous_id":52}, {"color": [209, 99, 106], "isthing":1, "id":58, "name":"hot dog", "contiguous_id":53}, {"color": [5, 121, 0], "isthing":1, "id":59, "name":"pizza", "contiguous_id":54}, {"color": [227, 255, 205], "isthing":1, "id":60, "name":"donut", "contiguous_id":55}, {"color": [147, 186, 208], "isthing":1, "id":61, "name":"cake", "contiguous_id":56}, {"color": [153, 69, 1], "isthing":1, "id":62, "name":"chair", "contiguous_id":57}, {"color": [3, 95, 161], "isthing":1, "id":63, "name":"couch", "contiguous_id":58}, {"color": [163, 255, 0], "isthing":1, "id":64, "name":"potted plant", "contiguous_id":59}, {"color": [119, 0, 170], "isthing":1, "id":65, "name":"bed", "contiguous_id":60}, {"color": [0, 182, 199], "isthing":1, "id":67, "name":"dining table", "contiguous_id":61}, {"color": [0, 165, 120], "isthing":1, "id":70, "name":"toilet", "contiguous_id":62}, {"color": [183, 130, 88], "isthing":1, "id":72, "name":"tv", "contiguous_id":63}, {"color": [95, 32, 0], "isthing":1, "id":73, "name":"laptop", "contiguous_id":64}, {"color": [130, 114, 135], "isthing":1, "id":74, "name":"mouse", "contiguous_id":65}, {"color": [110, 129, 133], "isthing":1, "id":75, "name":"remote", "contiguous_id":66}, {"color": [166, 74, 118], "isthing":1, "id":76, "name":"keyboard", "contiguous_id":67}, {"color": [219, 142, 185], "isthing":1, "id":77, "name":"cell phone", "contiguous_id":68}, {"color": [
    79, 210, 114], "isthing":1, "id":78, "name":"microwave", "contiguous_id":69}, {"color": [178, 90, 62], "isthing":1, "id":79, "name":"oven", "contiguous_id":70}, {"color": [65, 70, 15], "isthing":1, "id":80, "name":"toaster", "contiguous_id":71}, {"color": [127, 167, 115], "isthing":1, "id":81, "name":"sink", "contiguous_id":72}, {"color": [59, 105, 106], "isthing":1, "id":82, "name":"refrigerator", "contiguous_id":73}, {"color": [142, 108, 45], "isthing":1, "id":84, "name":"book", "contiguous_id":74}, {"color": [196, 172, 0], "isthing":1, "id":85, "name":"clock", "contiguous_id":75}, {"color": [95, 54, 80], "isthing":1, "id":86, "name":"vase", "contiguous_id":76}, {"color": [128, 76, 255], "isthing":1, "id":87, "name":"scissors", "contiguous_id":77}, {"color": [201, 57, 1], "isthing":1, "id":88, "name":"teddy bear", "contiguous_id":78}, {"color": [246, 0, 122], "isthing":1, "id":89, "name":"hair drier", "contiguous_id":79}, {"color": [191, 162, 208], "isthing":1, "id":90, "name":"toothbrush", "contiguous_id":80}, {"color": [255, 255, 128], "isthing":0, "id":92, "name":"banner", "contiguous_id":81}, {"color": [147, 211, 203], "isthing":0, "id":93, "name":"blanket", "contiguous_id":82}, {"color": [150, 100, 100], "isthing":0, "id":95, "name":"bridge", "contiguous_id":83}, {"color": [168, 171, 172], "isthing":0, "id":100, "name":"cardboard", "contiguous_id":84}, {"color": [146, 112, 198], "isthing":0, "id":107, "name":"counter", "contiguous_id":85}, {"color": [210, 170, 100], "isthing":0, "id":109, "name":"curtain", "contiguous_id":86}, {"color": [92, 136, 89], "isthing":0, "id":112, "name":"door-stuff", "contiguous_id":87}, {"color": [218, 88, 184], "isthing":0, "id":118, "name":"floor-wood", "contiguous_id":88}, {"color": [241, 129, 0], "isthing":0, "id":119, "name":"flower", "contiguous_id":89}, {"color": [217, 17, 255], "isthing":0, "id":122, "name":"fruit", "contiguous_id":90}, {"color": [124, 74, 181], "isthing":0, "id":125, "name":"gravel", "contiguous_id":91}, {"color": [70, 70, 70], "isthing":0, "id":128, "name":"house", "contiguous_id":92}, {"color": [255, 228, 255], "isthing":0, "id":130, "name":"light", "contiguous_id":93}, {"color": [154, 208, 0], "isthing":0, "id":133, "name":"mirror-stuff", "contiguous_id":94}, {"color": [193, 0, 92], "isthing":0, "id":138, "name":"net", "contiguous_id":95}, {"color": [76, 91, 113], "isthing":0, "id":141, "name":"pillow", "contiguous_id":96}, {"color": [255, 180, 195], "isthing":0, "id":144, "name":"platform", "contiguous_id":97}, {"color": [106, 154, 176], "isthing":0, "id":145, "name":"playingfield", "contiguous_id":98}, {"color": [230, 150, 140], "isthing":0, "id":147, "name":"railroad", "contiguous_id":99}, {"color": [60, 143, 255], "isthing":0, "id":148, "name":"river", "contiguous_id":100}, {"color": [128, 64, 128], "isthing":0, "id":149, "name":"road", "contiguous_id":101}, {"color": [92, 82, 55], "isthing":0, "id":151, "name":"roof", "contiguous_id":102}, {"color": [254, 212, 124], "isthing":0, "id":154, "name":"sand", "contiguous_id":103}, {"color": [73, 77, 174], "isthing":0, "id":155, "name":"sea", "contiguous_id":104}, {"color": [255, 160, 98], "isthing":0, "id":156, "name":"shelf", "contiguous_id":105}, {"color": [255, 255, 255], "isthing":0, "id":159, "name":"snow", "contiguous_id":106}, {"color": [104, 84, 109], "isthing":0, "id":161, "name":"stairs", "contiguous_id":107}, {"color": [169, 164, 131], "isthing":0, "id":166, "name":"tent", "contiguous_id":108}, {"color": [225, 199, 255], "isthing":0, "id":168, "name":"towel", "contiguous_id":109}, {"color": [137, 54, 74], "isthing":0, "id":171, "name":"wall-brick", "contiguous_id":110}, {"color": [135, 158, 223], "isthing":0, "id":175, "name":"wall-stone", "contiguous_id":111}, {"color": [7, 246, 231], "isthing":0, "id":176, "name":"wall-tile", "contiguous_id":112}, {"color": [107, 255, 200], "isthing":0, "id":177, "name":"wall-wood", "contiguous_id":113}, {"color": [58, 41, 149], "isthing":0, "id":178, "name":"water-other", "contiguous_id":114}, {"color": [183, 121, 142], "isthing":0, "id":180, "name":"window-blind", "contiguous_id":115}, {"color": [255, 73, 97], "isthing":0, "id":181, "name":"window-other", "contiguous_id":116}, {"color": [107, 142, 35], "isthing":0, "id":184, "name":"tree-merged", "contiguous_id":117}, {"color": [190, 153, 153], "isthing":0, "id":185, "name":"fence-merged", "contiguous_id":118}, {"color": [146, 139, 141], "isthing":0, "id":186, "name":"ceiling-merged", "contiguous_id":119}, {"color": [70, 130, 180], "isthing":0, "id":187, "name":"sky-other-merged", "contiguous_id":120}, {"color": [134, 199, 156], "isthing":0, "id":188, "name":"cabinet-merged", "contiguous_id":121}, {"color": [209, 226, 140], "isthing":0, "id":189, "name":"table-merged", "contiguous_id":122}, {"color": [96, 36, 108], "isthing":0, "id":190, "name":"floor-other-merged", "contiguous_id":123}, {"color": [96, 96, 96], "isthing":0, "id":191, "name":"pavement-merged", "contiguous_id":124}, {"color": [64, 170, 64], "isthing":0, "id":192, "name":"mountain-merged", "contiguous_id":125}, {"color": [152, 251, 152], "isthing":0, "id":193, "name":"grass-merged", "contiguous_id":126}, {"color": [208, 229, 228], "isthing":0, "id":194, "name":"dirt-merged", "contiguous_id":127}, {"color": [206, 186, 171], "isthing":0, "id":195, "name":"paper-merged", "contiguous_id":128}, {"color": [152, 161, 64], "isthing":0, "id":196, "name":"food-other-merged", "contiguous_id":129}, {"color": [116, 112, 0], "isthing":0, "id":197, "name":"building-other-merged", "contiguous_id":130}, {"color": [0, 114, 143], "isthing":0, "id":198, "name":"rock-merged", "contiguous_id":131}, {"color": [102, 102, 156], "isthing":0, "id":199, "name":"wall-other-merged", "contiguous_id":132}, {"color": [250, 141, 255], "isthing":0, "id":200, "name":"rug-merged", "contiguous_id":133}]


def _get_contiguous_to_original():
  # pylint: disable=line-too-long
  """
  Maps contiguous IDs to original COCO category IDs ending in 200, with 0 reserved for the background class.
  Also returns a mapping for whether each category is a "thing" (1) or "stuff" (0), and a boolean tensor indicating the same.
  """
  contigious_id = 1  # since 0 is reserved for background class
  keys_tensor_1, vals_tensor_1 = [], []
  keys_tensor_2, vals_tensor_2 = [], []
  thing_tensor_bool = []
  contigious_to_original = {}
  contigious_to_original_thing = {}
  for each_category in COCO_CATEGORIES:
    keys_tensor_1.append(contigious_id)
    vals_tensor_1.append(each_category["id"])
    keys_tensor_2.append(each_category["id"])
    vals_tensor_2.append(each_category["isthing"])
    thing_tensor_bool.append(tf.cast(each_category["isthing"], tf.bool))
    contigious_id += 1

  contigious_to_original = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          tf.constant(keys_tensor_1),
          tf.constant(vals_tensor_1)),
      default_value=-1)
  contigious_to_original_thing = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          tf.constant(keys_tensor_2),
          tf.constant(vals_tensor_2)),
      default_value=-1)
  return contigious_to_original, contigious_to_original_thing, thing_tensor_bool


def _get_original_to_contiguous():
  # pylint: disable=line-too-long
  """
  Maps original COCO category IDs to contiguous IDs starting from 1, with 0 reserved for the background class.
  Also returns a mapping for whether each category is a "thing" (1) or "stuff" (0), and a boolean tensor indicating the same.
  """
  contiguous_id = 1  # since 0 is reserved for background class
  keys_tensor_original_to_contiguous, vals_tensor_original_to_contiguous = [], []
  keys_tensor_id_to_isthing, vals_tensor_id_to_isthing = [], []
  thing_tensor_bool = []

  for each_category in COCO_CATEGORIES:
    original_id = each_category["id"]
    is_thing = each_category["isthing"]

    keys_tensor_original_to_contiguous.append(original_id)
    vals_tensor_original_to_contiguous.append(contiguous_id)

    keys_tensor_id_to_isthing.append(original_id)
    vals_tensor_id_to_isthing.append(is_thing)
    thing_tensor_bool.append(tf.cast(is_thing, tf.bool))

    contiguous_id += 1

  original_to_contiguous = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(keys_tensor_original_to_contiguous),
          values=tf.constant(vals_tensor_original_to_contiguous)),
      default_value=-1)

  original_id_to_isthing = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          keys=tf.constant(keys_tensor_id_to_isthing),
          values=tf.constant(vals_tensor_id_to_isthing)),
      default_value=-1)

  return original_to_contiguous, original_id_to_isthing, thing_tensor_bool


def _is_thing():
  # pylint: disable=line-too-long
  """
  Returns a bool tensor of size 133, with True for thing classes.
  """
  return [0] + [bool(x['isthing']) for x in COCO_CATEGORIES]
