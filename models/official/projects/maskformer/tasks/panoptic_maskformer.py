import os
from absl import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from official.core import base_task
from official.core import task_factory
from official.core import train_utils
from typing import Any, Dict, List, Mapping, Optional, Tuple

from official.projects.maskformer.dataloaders import input_reader
from official.vision.dataloaders import input_reader_factory
from official.common import dataset_fn

from official.projects.maskformer.configs import maskformer as maskformer_cfg
from official.projects.maskformer.modeling.maskformer import MaskFormer
from official.projects.maskformer.losses.maskformer_losses import Loss
from official.projects.maskformer.dataloaders import panoptic_input

from official.projects.detr.ops.matchers import hungarian_matching
from official.projects.maskformer.losses.maskformer_losses import Loss

from official.vision.evaluation import panoptic_quality
from official.projects.maskformer.losses.inference import PanopticInference
from official.vision.modeling import backbones

from official.projects.maskformer.losses.mapper import _get_contigious_to_original, _get_original_to_contigious

import numpy as np

@task_factory.register_task_cls(maskformer_cfg.MaskFormerTask)
class PanopticTask(base_task.Task):
	"""A single-replica view of training procedure.

	PanopticTask task provides artifacts for training/evalution procedures, including
	loading/iterating over Datasets, initializing the model, calculating the loss,
	post-processing, and customized metrics with reduction.
	"""
	def build_model(self):
		"""Builds MaskFormer Model."""

		self.class_ids = {}
		self.plot_collection = {} 
		self.plot_collection_labels = {0:[]}
		self.temp = 0	
		self.background_empty_mask = {}
		self.labelled_empty_mask = {}
		self.background_non_empty_mask = {}
		self.class_id_counts = {}
		self.log_dir = os.environ.get('LOG_DIR')
		self.run_number = os.environ.get('RUN_NUMBER')

		if self.log_dir:
			try: 
				os.mkdir(self.log_dir)
			except: 
				pass 
			os.mkdir(os.path.join(self.log_dir, self.run_number)) # If there is existing, then throw error
			self.log_dir = os.path.join(self.log_dir, self.run_number)

			with open(os.path.join(self.log_dir, 'checking_labels.txt'), 'w') as file:
				pass
			
			with open(os.path.join(self.log_dir, 'settings.txt'), 'w') as file:
				file.write("RUN: " + str(os.environ.get('RUN_NUMBER')) + '\n')
				file.write("BSIZE: " + str(os.environ.get('TRAIN_BATCH_SIZE'))+ '\n')
				file.write("BASE_LR: " + str(os.environ.get('BASE_LR'))+ '\n')
				file.write("NO_OBJ_CLS_WEIGHT: " + str(os.environ.get('NO_OBJ_CLS_WEIGHT'))+ '\n')
			
		logging.info('Building MaskFormer model.')
		input_specs = tf.keras.layers.InputSpec(shape=[None] + self._task_config.model.input_size)
		
		backbone = backbones.factory.build_backbone(input_specs=input_specs,
					backbone_config=self._task_config.model.backbone,
					norm_activation_config=self._task_config.model.norm_activation)
		logging.info('Backbone build successful.')
		model = MaskFormer(backbone=backbone,input_specs= input_specs,
							num_queries=self._task_config.model.num_queries,
							hidden_size=self._task_config.model.hidden_size,
							backbone_endpoint_name=self._task_config.model.backbone_endpoint_name,
							fpn_encoder_layers=self._task_config.model.fpn_encoder_layers,
							detr_encoder_layers=self._task_config.model.detr_encoder_layers,
							num_decoder_layers=self._task_config.model.num_decoder_layers,
							num_classes=self._task_config.model.num_classes,
							bfloat16=self._task_config.bfloat16, 
							which_pixel_decoder=self._task_config.model.which_pixel_decoder,
							)
		logging.info('Maskformer model build successful.')
		inputs = tf.keras.Input(shape=input_specs.shape[1:])
		model(inputs)
		model.summary() 
		return model

	def initialize(self, model: tf.keras.Model) -> None:
		"""
		Used to initialize the models with checkpoint
		"""

		logging.info('Initializing model from checkpoint: %s', self._task_config.init_checkpoint)
		if not self._task_config.init_checkpoint:
			return
		ckpt_dir_or_file = self._task_config.init_checkpoint

		# Restoring ckpt
		if tf.io.gfile.isdir(ckpt_dir_or_file):
			ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

		if self._task_config.init_checkpoint_modules == 'all':
			# ckpt = tf.train.Checkpoint(**model.checkpoint_items)
			ckpt = tf.train.Checkpoint(model=model)
			status = ckpt.restore(ckpt_dir_or_file)
			status.expect_partial().assert_existing_objects_matched()
			logging.info('Loaded whole model from %s',ckpt_dir_or_file)
			
		elif self._task_config.init_checkpoint_modules == 'backbone':
			ckpt = tf.train.Checkpoint(backbone=model.backbone)
			status = ckpt.restore(ckpt_dir_or_file)
			status.expect_partial().assert_existing_objects_matched()
			logging.info('Finished loading backbone checkpoint from %s',
					ckpt_dir_or_file)
		else:
			raise ValueError('Not a valid module to initialize from: {}'.format(
				self._task_config.init_checkpoint_modules))

	def build_inputs(self, params, input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
		""" 
		Build panoptic segmentation dataset.

		"""
		logging.info('Building panoptic segmentation dataset.')
		if params.decoder.type == 'simple_decoder':
			decoder = panoptic_input.TfExampleDecoder(regenerate_source_id = params.regenerate_source_id)
		else:
			raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))
		
		parser = panoptic_input.mask_former_parser(params.parser, is_training = params.is_training, decoder_fn=decoder.decode)
	
		reader = input_reader_factory.input_reader_generator(
		  params,
		  dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
		  decoder_fn=decoder.decode,
		  parser_fn=parser.parse_fn(params.is_training))
		
		dataset = reader.read(input_context=input_context)
		# print("Saving dataset")
		# for sample in dataset.take(1):
		# 	# print(f"unique idsin dataset take : {sample[1]['unique_ids']}")
		# 	# print("individual masks :", sample[1]["individual_masks"].shape)
		# 	np.save("contigious_mask.npy", sample[1]["contigious_mask"].numpy())
		# 	# print(f"image shape : {sample[0].shape}")
		# 	np.save("individual_masks.npy", sample[1]["individual_masks"].numpy())
		# 	np.save("unique_ids.npy", sample[1]["unique_ids"].numpy())
		# 	np.save("images.npy", sample[0].numpy())
		# 	np.save("category_mask.npy", sample[1]["category_mask"].numpy())	
		# 	np.save("instance_mask.npy", sample[1]["instance_mask"].numpy())
		# 	exit()
		return dataset


	def build_losses(self, output, labels, aux_outputs=None):
		# TODO : Auxilary outputs
		outputs = {"pred_logits": output["class_prob_predictions"], "pred_masks": output["mask_prob_predictions"]}
		targets = labels

		matcher = hungarian_matching
		no_object_weight = self._task_config.losses.background_cls_weight
		loss = Loss(num_classes = self._task_config.model.num_classes,
					matcher = matcher,
					eos_coef = no_object_weight,
					cost_class= self._task_config.losses.cost_class,
					cost_dice= self._task_config.losses.cost_dice,
					cost_focal= self._task_config.losses.cost_focal, ignore_label=self._task_config.train_data.parser.ignore_label)

		calculated_losses = loss(outputs, targets)
		
		# Losses are returned as weighted sum of individual losses
		total_loss = calculated_losses['loss_ce'] + calculated_losses['loss_dice'] + calculated_losses['loss_focal']

		weighted_ce = calculated_losses['loss_ce']
		weighted_dice = calculated_losses['loss_dice']
		weighted_focal = calculated_losses['loss_focal']

		# Not implemented auxilary outputs
		# if aux_outputs is not None:
		#       total_aux_loss = 0.0
		#       for i in range(4): #4 number of auxilary outputs
		#               total_aux_loss += calculated_losses['loss_ce_'+str(i)] + calculated_losses['loss_dice_'+str(i)] + calculated_losses['loss_focal_'+str(i)]
		#       total_loss = total_loss + total_aux_loss
		

		return total_loss, weighted_ce, weighted_focal, weighted_dice
		
	def build_metrics(self, training=True):
		"""Builds panoptic metrics."""
		metrics = []
		metric_names = ['cls_loss', 'focal_loss', 'dice_loss']
		for name in metric_names:
			metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
		# TODO : Need panoptic quality metric for evaluation
		if not training:
			print("[INFO] Building panoptic quality metric ")
			_, _, thing_tensor_bool = _get_contigious_to_original()
			self.is_thing_dict_bool = thing_tensor_bool
			pq_config = self._task_config.panoptic_quality_evaluator
			self.panoptic_quality_metric = panoptic_quality.PanopticQualityV2(
				num_categories=pq_config.num_categories,
				is_thing=self.is_thing_dict_bool,
				ignored_label=pq_config.ignored_label,
				max_num_instances=pq_config.max_num_instances,
				rescale_predictions=pq_config.rescale_predictions,
			)
			self.panoptic_inference = PanopticInference(
				num_classes=self._task_config.model.num_classes+1, 
				background_class_id=pq_config.ignored_label
			)
		return metrics
		
	def _log_classes(self, labels: Dict[str, Any]) -> List[Dict[int, int]]:
		""" 
		Logs all the class IDs viewed during training and evaluation.
		
		Returns: 
		A dictionary of class ids and their counts across all images in batch
		"""

		all_unique_ids = labels["unique_ids"]._numpy()
		classes_in_batch = []
		for size in range(all_unique_ids.shape[0]):
			unique_ids = all_unique_ids[size, :]
			classes_in_image = {}
			for class_id in unique_ids: 
				if class_id in classes_in_image: 
					classes_in_image[class_id] += 1
				else: 
					classes_in_image[class_id] = 1
			classes_in_batch.append(classes_in_image)

			for class_id in unique_ids: 
				if class_id in self.class_ids: 
					self.class_ids[class_id] += 1
				else: 
					self.class_ids[class_id] = 1

		return classes_in_batch

	def _check_contigious_mask(self, labels: Dict[str, Any]):
		"""	
		Checks if all the contigious masks are mapped properly from the category masks 

		Returns:
		EagerTensor with correctly mapped contigious masks
		"""
		mapping_dict  = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, \
		19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, \
		39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, \
		58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, \
		80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80, 92: 81, 93: 82, 95: 83, 100: 84, 107: 85, 109: 86, 112: 87, \
		118: 88, 119: 89, 122: 90, 125: 91, 128: 92, 130: 93, 133: 94, 138: 95, 141: 96, 144: 97, 145: 98, 147: 99, 148: 100, 149: 101, 151: 102, \
		154: 103, 155: 104, 156: 105, 159: 106, 161: 107, 166: 108, 168: 109, 171: 110, 175: 111, 176: 112, 177: 113, 178: 114, 180: 115, 181: 116, \
		184: 117, 185: 118, 186: 119, 187: 120, 188: 121, 189: 122, 190: 123, 191: 124, 192: 125, 193: 126, 194: 127, 195: 128, 196: 129, 197: 130, \
		198: 131, 199: 132, 200: 133}

		category_mask = labels["category_mask"]._numpy()
		contigious_mask = labels["contigious_mask"]._numpy()

		for size in range(category_mask.shape[0]):
			cat = category_mask[size]
			cont = contigious_mask[size, :, :, :]
			mapped_cat = np.expand_dims(np.array([[mapping_dict.get(int(x), int(x)) for x in row] for row in cat]), axis=-1)
			if not np.array_equal(mapped_cat, cont): 
				contigious_mask[size, :, :, :] = mapped_cat
			
		return tf.convert_to_tensor(contigious_mask)

	def _check_induvidual_masks(self, labels: Dict[str, Any], class_id_counts: List[Dict[int, int]]):
		"""
		Checks if all the induvidual masks are given the correct instance id

		Returns:
		EagerTensor with correctly mapped induvidual masks
		"""

		# mapping_dict  = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, \
		# 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, \
		# 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, \
		# 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, \
		# 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80, 92: 81, 93: 82, 95: 83, 100: 84, 107: 85, 109: 86, 112: 87, \
		# 118: 88, 119: 89, 122: 90, 125: 91, 128: 92, 130: 93, 133: 94, 138: 95, 141: 96, 144: 97, 145: 98, 147: 99, 148: 100, 149: 101, 151: 102, \
		# 154: 103, 155: 104, 156: 105, 159: 106, 161: 107, 166: 108, 168: 109, 171: 110, 175: 111, 176: 112, 177: 113, 178: 114, 180: 115, 181: 116, \
		# 184: 117, 185: 118, 186: 119, 187: 120, 188: 121, 189: 122, 190: 123, 191: 124, 192: 125, 193: 126, 194: 127, 195: 128, 196: 129, 197: 130, \
		# 198: 131, 199: 132, 200: 133}

		induvidual_masks = labels["individual_masks"]._numpy()
		# contig_mask = labels["contigious_mask"]._numpy().copy()
		# instance_mask = labels["instance_mask"]._numpy().copy()
		# zero_mask = np.zeros((induvidual_masks.shape[2], induvidual_masks.shape[3]), dtype=induvidual_masks.dtype)
		class_ids = labels["unique_ids"]._numpy().copy()

			# induvidual_masks_in_image = induvidual_masks[size, :, :, :, :]
			# instance_mask_in_image = instance_mask[size, :, :, :]
			# contig_mask_in_image = contig_mask[size, :, :, :]
			# combined_mask = np.array([[tuple((contig_mask_in_image[i, j], instance_mask_in_image[i, j])) for j in range(contig_mask_in_image.shape[1])] for i in range(contig_mask_in_image.shape[0])])
			
			# with open('/depot/davisjam/data/akshath/exps/tf/indu_masks/indu_masks.txt', 'w') as file: 
			# 	file.write(str(combined_mask) + '\n')
			# 	file.write(str(np.unique(combined_mask, axis=0)) + '\n')

			# for a in np.unique(instance_mask_in_image): 
			# 	plt.imshow(instance_mask_in_image == a)
			# 	plt.savefig(f'/depot/davisjam/data/akshath/exps/tf/indu_masks/my_image__{size}_{a}.png')

			# unique_ids = class_ids[size, :]
			# # np.save('/depot/davisjam/data/akshath/exps/tf/indu_masks/instance.npy', instance_mask_in_image)
			# return 
			# for i, class_id in enumerate(unique_ids):
			# 	if class_id != 0:
			# 		print(class_id)		
					# instance_mask_in_image[instance_mask_in_image == i]	
					# if induvidual_masks_in_image[i,:,:,:]
				# if not np.all((induvidual_masks_in_image[i,:,:,:] == 0) | (induvidual_masks_in_image[i,:,:,:] == mapped_id)):
					# induvidual_masks_in_image[i, :, :, :] = np.array([[mapped_id for x in row] for row in induvidual_masks_in_image[i, :, :, :]])

		for size in range(len(class_ids)): 

			# background_non_empty_mask = 0 
			labelled_empty_mask = 0 
			# background_empty_mask = 0

			for i, mask in enumerate(induvidual_masks[size, :, :, :, :]):
				if class_ids[size][i] != 0:
					if np.all(mask == 0): 
						labelled_empty_mask += 1
						class_ids[size][i] = 0						

			self.labelled_empty_mask[self.temp] = labelled_empty_mask

			with open(os.path.join(self.log_dir, 'background_empty_mask.txt'), 'w') as file: 
				file.write(str(self.background_empty_mask) + '\n')
			with open(os.path.join(self.log_dir, 'labelled_empty_mask.txt'), 'w') as file: 
				file.write(str(self.labelled_empty_mask) + '\n')
			with open(os.path.join(self.log_dir, 'background_non_empty_mask.txt'), 'w') as file: 
				file.write(str(self.background_non_empty_mask) + '\n')
			with open(os.path.join(self.log_dir, 'class_id_counts.txt'), 'w') as file: 
				file.write(str(self.class_id_counts) + '\n')
			with open(os.path.join(self.log_dir, 'class_ids.txt'), 'w') as file: 
				file.write(str(self.class_ids) + '\n')
		
		return tf.convert_to_tensor(induvidual_masks)


	def train_step(self, inputs: Tuple[Any, Any],model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, metrics: Optional[List[Any]] = None) -> Dict[str, Any]:
		"""
		Does forward and backward.

		Args:
		inputs: a dictionary of input tensors.
		model: the model, forward pass definition.
		optimizer: the optimizer for this training step.
		metrics: a nested structure of metrics objects.

		Returns:
		A dictionary of logs.
		"""
						
		features, labels = inputs

		# features = tf.convert_to_tensor(np.load('/depot/davisjam/data/akshath/exps/resnet/raw/features.npy')) 
		# for val in labels: 
		# 	labels[val] = tf.convert_to_tensor(np.load(f'/depot/davisjam/data/akshath/exps/resnet/raw/{val}.npy'))

		# np.save('/depot/davisjam/data/akshath/exps/tf/resnet/raw/features.npy', tf.cast(features, np.float32)._numpy())
		# for lab in labels: 
			# np.save(f'/depot/davisjam/data/akshath/exps/tf/resnet/raw/{lab}.npy', tf.cast(labels[lab], np.float32)._numpy())


		# self.temp += 2
		# all_unique_ids = labels["unique_ids"]._numpy()
		# for size in range(all_unique_ids.shape[0]):
		# 	unique_ids = all_unique_ids[size, :]
		# 	for class_id in unique_ids: 
		# 		if class_id in self.class_ids: 
		# 			self.class_ids[class_id] += 1
		# 		else: 
		# 			self.class_ids[class_id] = 1

		# print(self.temp)
		# with open(os.path.join(self.log_dir, 'class_ids.txt'), 'w') as file: 
			# file.write(str(self.class_ids) + '\n')

		# self._log_classes(labels)
		# labels["individual_masks"] = self._check_induvidual_masks(labels, self._log_classes(labels))

		# # for param in model.trainable_variables:
		# # 	name = param.name.replace('/', '-')
		# # 	np.save(f"/depot/davisjam/data/akshath/exps/tf/weights_biases/{name}.npy", param.numpy())  

		# # with open('/depot/davisjam/data/akshath/exps/tf/indu_masks/indu_masks.txt', 'w') as file: 
		# # 	file.write(str(labels) + '\n')


		# # raise ValueError('Init') 
		
		# # labels["individual_masks"] = self._check_induvidual_masks(labels, self._log_classes(labels))
		# # labels["contigious_mask"] = self._check_contigious_mask(labels)

		with tf.GradientTape() as tape:
			outputs = model(features, training=True)
			# print(backbone_feature_maps_procesed.keys())
			
			# for val in backbone_feature_maps_procesed: 
			# 	print(backbone_feature_maps_procesed[val])
			# 	print(backbone_feature_maps_procesed[val].numpy())
			# 	np.save(os.path.join('/depot/davisjam/data/akshath/exps/resnet/tf', 'backbone_feature_maps_procesed_' + str(val) + '.npy'), backbone_feature_maps_procesed[val].numpy())

			##########################################################
			# FIXME : This loop must be used for auxilary outputs
			loss = 0.0
			cls_loss = 0.0
			focal_loss = 0.0
			dice_loss = 0.0
			
			# for output in outputs:
			#       # Computes per-replica loss.	
			#       total_loss, cls_loss_, focal_loss_, dice_loss_ = self.build_losses(
			#               output=output, labels=labels)
			#       loss += total_loss
			#       cls_loss += cls_loss_
			#       focal_loss += focal_loss_
			#       dice_loss += dice_loss_
			
			#       scaled_loss = loss
			#       # For mixed_precision policy, when LossScaleOptimizer is used, loss is
			#       # scaled for numerical stability.
			
			##########################################################################
			
			total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels)
			scaled_loss = total_loss

			if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
				total_loss = optimizer.get_scaled_loss(scaled_loss)
		
		print('Total loss : ', total_loss)
		print('Cls loss : ', cls_loss)
		print('Focal loss : ', focal_loss)
		print('Dice loss : ', dice_loss)

		tvars = model.trainable_variables	
		grads = tape.gradient(scaled_loss,tvars)

		if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
			grads = optimizer.get_unscaled_gradients(grads)
		optimizer.apply_gradients(list(zip(grads, tvars)))
		
		if os.environ.get('PRINT_OUTPUTS') == 'True':
			probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
			pred_labels = tf.argmax(probs, axis=-1)
			print("Target labels :", labels["unique_ids"])
			print("Output labels :", pred_labels)

		# temp = {} 
		# for grad, param in zip(grads, tvars): 
		# 	temp[param.name] = tf.norm(grad).numpy()

		# for param in temp: 
		# 	if param not in self.plot_collection: 
		# 		self.plot_collection[param] = []
		# 	else: 
		# 		self.plot_collection[param]	+= [temp[param]]
		# self.plot_collection_labels[0] += [len(np.unique(pred_labels).tolist())]

		self.temp += int(os.environ.get('TRAIN_BATCH_SIZE'))
		with open(os.path.join(self.log_dir, 'checking_labels.txt'), 'a') as file:
			file.write(str(self.temp) + '\n')
			file.write(str(labels["unique_ids"].numpy()) + '\n')
			file.write(str(pred_labels.numpy())+ '\n')
			file.write(f"{total_loss}, {cls_loss}, {focal_loss}, {dice_loss}" + '\n')
			file.write('-----------------------------------' + '\n')

		# if (sum(temp.values()) == 0) or (len(np.unique(pred_labels).tolist()) == 1 and np.unique(pred_labels).tolist()[0] == 0): 
		# 	with open('/depot/davisjam/data/akshath/exps/tf/editing_layers/numIters.txt', 'a') as file: 
		# 		file.write(str('numIters : ' + str(self.temp)) + '\n')
		# 	with open('/depot/davisjam/data/akshath/exps/tf/vishal_plot/dict.txt', 'w') as file: 
		# 		file.write(str(self.plot_collection))
		# 	with open('/depot/davisjam/data/akshath/exps/tf/vishal_plot/dict_labels.txt', 'w') as file: 
		# 			file.write(str(self.plot_collection_labels))

		# 	raise ValueError('Stop2')
		
		# # Multiply for logging.
		# # Since we expect the gradient replica sum to happen in the optimizer,
		# # the loss is scaled with global num_boxes and weights.
		# # # To have it more interpretable/comparable we scale it back when logging.
		num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
		total_loss *= num_replicas_in_sync
		cls_loss *= num_replicas_in_sync
		focal_loss *= num_replicas_in_sync
		dice_loss *= num_replicas_in_sync
		
		logs = {self.loss: total_loss}

		all_losses = {
			'cls_loss': cls_loss,
			'focal_loss': focal_loss,
			'dice_loss': dice_loss,}
		
		if metrics:
			for m in metrics:
				m.update_state(all_losses[m.name])
		return logs
		
	def _postprocess_outputs(self, outputs: Dict[str, Any], image_shapes):
		""" 
		Implements postprocessing using the output binary masks and labels to produce
		1. Output Category Mask
		2. Output Instance Mask

		"""
		pred_binary_masks = outputs["mask_prob_predictions"]
		pred_labels = outputs["class_prob_predictions"]
		ouput_instance_mask, output_category_mask = self.panoptic_inference(pred_labels,pred_binary_masks, image_shapes)
		return ouput_instance_mask, output_category_mask

	def validation_step(self, inputs, model, metrics=None):
		features, labels = inputs
		outputs = model(features, training=False)
		if os.environ.get('PRINT_OUTPUTS') == 'True':
			probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
			pred_labels = tf.argmax(probs, axis=-1)
			print("Probs :", probs)
			print("Target labels :", labels["unique_ids"])
			print("Output labels :", pred_labels)
		total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels)
		
		
		num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync
		total_loss *= num_replicas_in_sync
		cls_loss *= num_replicas_in_sync
		focal_loss *= num_replicas_in_sync
		dice_loss *= num_replicas_in_sync
		logs = {self.loss: total_loss}
		all_losses = {
				'cls_loss': cls_loss,
				'focal_loss': focal_loss,
				'dice_loss': dice_loss,
			}
		
		if self.panoptic_quality_metric is not None:
			pq_metric_labels = {
			'category_mask': labels['category_mask'], # ignore label is 0 
			'instance_mask': labels['instance_mask'],
			}
			# Output from postprocessing will convert the binary masks to category and instance masks with non-contigious ids
			output_category_mask, output_instance_mask = self._postprocess_outputs(outputs, [640, 640])
			pq_metric_outputs = {
			'category_mask': output_category_mask,
			'instance_mask': output_instance_mask,
			}
			
			self.panoptic_quality_metric.update_state(
		  	pq_metric_labels, pq_metric_outputs
	  		)
		if metrics:
			for m in metrics:
				m.update_state(all_losses[m.name])

		return logs
	

	def aggregate_logs(self, state=None, step_outputs=None):
		is_first_step = not state
		if state is None:
			state = self.panoptic_quality_metric
		state.update_state(
			step_outputs['ground_truths'],
			step_outputs['predictions'])
		
		return state
	
	def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
		if self.panoptic_quality_metric is not None:
			self._reduce_panoptic_metrics(aggregated_logs)
			self.panoptic_quality_metric.reset_state()	
		return aggregated_logs
	
	def _reduce_panoptic_metrics(self, logs: Dict[str, Any]):
		"""
		Updates the per class and mean panoptic metrics in the logs.
		
		"""
		result = self.panoptic_quality_metric.result()
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

		logs['panoptic_quality/All_num_categories'] = num_categories
		logs['panoptic_quality/Things_num_categories'] = num_thing_categories
		logs['panoptic_quality/Stuff_num_categories'] = num_stuff_categories
		for metric in ['pq', 'sq', 'rq']:
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
		
	