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

"""Tests for tensorflow_models.official.projects.maskformer.losses.inference."""

from absl.testing import parameterized
import tensorflow as tf
from official.projects.maskformer.losses.inference import PanopticInference
from official.projects.maskformer.modeling.maskformer import MaskFormer
import numpy as np
from official.vision.evaluation import panoptic_quality
from official.projects.maskformer.losses.mapper import _get_contigious_to_original, _get_original_to_contigious


class PanopticInferenceTest(tf.test.TestCase, parameterized.TestCase):
	@parameterized.named_parameters(('test1',))
	def test_pass_through(self):
		"""
		# This test is supposed to give PQ stuff and PQ things metrics for fixed tensor inputs
		# Load pytorch output and targets for testing the PQ stuff and PQ things metrics
		Procduere for testing:
		1. Save the input image (without normalization), GT instance (will have non-contigious ids) and panoptic masks (will have non-contigious ids) from TF code
		2. With saved TF image as input obtain output individual masks from the PyTorch Model with final weights
		"""
		background_class_id = 0
		cat_id_map, is_thing_dict = _get_contigious_to_original()
		
		is_thing_dict_bool = tf.cast(list(is_thing_dict.values()), dtype=tf.bool)
		main_pth = "/depot/qqiu/data/vishal/tf-maskformer/tensors_for_PQ_metric"

		# Load pytorch predictions
		image_shape = [3, 640, 640]
		pred_logits_load = tf.convert_to_tensor(np.load(main_pth+"/output_pred_logits.npy")) 
		pred_masks_load = tf.convert_to_tensor(np.load(main_pth+"/output_pred_masks.npy")) 
		pred_masks_load = tf.transpose(pred_masks_load, [0,2,3,1]) # (1,100, h, w) -> (1, h, w, 100) (reshaping according to TF model outputs)
		
		# Pytorch code uses 133 as backgorund class id and TF code uses 0 as background class id so we need to swap them
		
		# shift all classes by 1 and replace 133 with 0 (background class id)
		# Load the instance and category masks from TF code
		instance_mask_gt = tf.convert_to_tensor(np.load(main_pth+"/instance_mask.npy"))
		category_mask_gt = tf.convert_to_tensor(np.load(main_pth+"/category_mask.npy"))

		outputs = {
			"class_prob_predictions": pred_logits_load,
			"mask_prob_predictions": pred_masks_load,
		}
		
		inference = PanopticInference(num_classes=134, background_class_id=background_class_id, object_mask_threshold=0.25, class_score_threshold=0.25, overlap_threshold=0.25)
		instance_mask_predicted, category_mask_predicted = inference(outputs["class_prob_predictions"], 
																		outputs["mask_prob_predictions"],
																	   image_shape)
		# Save the instance and category masks from TF code
		np.save(main_pth+"/instance_mask_predicted.npy", instance_mask_predicted.numpy())
		np.save(main_pth+"/category_mask_predicted.npy", category_mask_predicted.numpy())
		
		# test PQ metrics
		self.panoptic_quality_metric = panoptic_quality.PanopticQualityV2(
				num_categories=133,
				is_thing=is_thing_dict_bool,# Not used in PQ Evaluator
				ignored_label=0,
				max_num_instances=100,
				rescale_predictions=False,
			)
		
		pq_metric_labels = {
			'category_mask': tf.squeeze(category_mask_gt, -1), # ignore label is 0 
			'instance_mask': tf.squeeze(instance_mask_gt, -1),
			}
		
	   
		pq_metric_outputs = {
			'category_mask': category_mask_predicted,
			'instance_mask': instance_mask_predicted,
			}
		self.panoptic_quality_metric.update_state(
		  	pq_metric_labels, pq_metric_outputs
	  		)
		result = self.panoptic_quality_metric.result()
		print("Processing the result.....")
		valid_thing_classes = result['valid_thing_classes']
		valid_stuff_classes = result['valid_stuff_classes']
		valid_classes = valid_stuff_classes | valid_thing_classes
		num_categories = tf.math.count_nonzero(valid_classes, dtype=tf.float32)
		num_thing_categories = tf.math.count_nonzero(
			valid_thing_classes, dtype=tf.float32
		)
		num_stuff_categories = tf.math.count_nonzero(
			valid_stuff_classes, dtype=tf.float32
		)
		
		valid_thing_classes = tf.cast(valid_thing_classes, dtype=tf.float32)
		valid_stuff_classes = tf.cast(valid_stuff_classes, dtype=tf.float32)
		logs = {}
		logs['panoptic_quality/All_num_categories'] = num_categories
		logs['panoptic_quality/Things_num_categories'] = num_thing_categories
		logs['panoptic_quality/Stuff_num_categories'] = num_stuff_categories
		for metric in ['pq']:
			metric_per_class = result[f'{metric}_per_class']
			logs[f'panoptic_quality/All_{metric}'] = tf.math.divide_no_nan(
				tf.reduce_sum(metric_per_class), num_categories
			)
			logs[f'panoptic_quality/Things_{metric}'] = tf.math.divide_no_nan(
				tf.reduce_sum(metric_per_class * valid_thing_classes),
				num_thing_categories,
			)
			logs[f'panoptic_quality/Stuff_{metric}'] = tf.math.divide_no_nan(
				tf.reduce_sum(metric_per_class * valid_stuff_classes),
				num_stuff_categories,
			)
			# if self.task_config.panoptic_quality_evaluator.report_per_class_metrics:
			# 	for i, is_valid in enumerate(valid_classes.numpy()):
			# 		if is_valid:
			# 			logs[f'panoptic_quality/{metric}/class_{i}'] = metric_per_class[i]
		print(logs)
if __name__ == '__main__':
	tf.test.main()
