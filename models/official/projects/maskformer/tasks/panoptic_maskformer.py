import os
from absl import logging
import tensorflow as tf
import json
import matplotlib.pyplot as plt 
from official.core import base_task
from official.core import task_factory
from official.core import train_utils
from typing import Any, Dict, List, Mapping, Optional, Tuple
import zipfile
import shutil

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
from official.core.train_utils import try_count_params
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
        logging.info('Building MaskFormer model.')
        self.total_zips = [] 
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
        # intialize the model
        dummy_input = tf.zeros((1, self._task_config.model.input_size[0], self._task_config.model.input_size[1], 3))
        model(dummy_input, training=False)
        logging.info('Number of trainable parameters: %s', try_count_params(model, trainable_only=True))
        logging.info('Number of parameters: %s', try_count_params(model, trainable_only=False))


        self.DATA_IDX = 0
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
                    background_class_id=pq_config.ignored_label, 
                    )
            return metrics



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


        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
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

            # TODO : Add auxiallary losses
            total_loss, cls_loss, focal_loss, dice_loss = self.build_losses(output=outputs, labels=labels)
            scaled_loss = total_loss

            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                total_loss = optimizer.get_scaled_loss(scaled_loss)
        tvars = model.trainable_variables    
        grads = tape.gradient(scaled_loss,tvars)

        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            grads = optimizer.get_unscaled_gradients(grads)        
        optimizer.apply_gradients(list(zip(grads, tvars)))

        if os.environ.get('PRINT_OUTPUTS') == 'True':
            probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
            pred_labels = tf.argmax(probs, axis=-1)


        # # Multiply for logging.
        # # Since we expect the gradient replica sum to happen in the optimizer,
        # # the loss is scaled with global num_boxes and weights.
        # # To have it more interpretable/comparable we scale it back when logging.
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
        # if os.environ.get('PRINT_OUTPUTS') == 'True':
        #     probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
        #     pred_labels = tf.argmax(probs, axis=-1)
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


        pq_metric_labels = {
        'category_mask': labels['category_mask'], # ignore label is 0 
        'instance_mask': labels['instance_mask'],
        }
        output_instance_mask, output_category_mask = self._postprocess_outputs(outputs, [640, 640])
        pq_metric_outputs = {
        'category_mask': output_category_mask,
        'instance_mask': output_instance_mask,
        }

        self.panoptic_quality_metric.update_state(
            pq_metric_labels, pq_metric_outputs
            )

        if os.environ.get('PRINT_OUTPUTS') == 'True':
            probs = tf.keras.activations.softmax(outputs["class_prob_predictions"], axis=-1)
            pred_labels = tf.argmax(probs, axis=-1)
            # print("Target labels :", labels["unique_ids"])
            # print("Output labels :", pred_labels)
            print("Saving outputs...........", self.DATA_IDX)
            # Save model inputs and outputs for visualization.
            # convert from bfloat16 to unit8 for image

            try: 
                os.mkdir(os.environ.get('FART'))
            except: 
                pass
            
            try: 
                name_list = [] 
                name_list += [f"{os.environ.get('FART')}/input_img_"+str(self.DATA_IDX)+".npy"]
                np.save(f"{os.environ.get('FART')}/input_img_"+str(self.DATA_IDX)+".npy", tf.cast(features, dtype=tf.float32).numpy())
                name_list += [f"{os.environ.get('FART')}/output_labels_"+str(self.DATA_IDX)+".npy"]
                np.save(f"{os.environ.get('FART')}/output_labels_"+str(self.DATA_IDX)+".npy", outputs["class_prob_predictions"].numpy())
                name_list += [f"{os.environ.get('FART')}/target_labels_"+str(self.DATA_IDX)+".npy"]
                np.save(f"{os.environ.get('FART')}/target_labels_"+str(self.DATA_IDX)+".npy", labels["unique_ids"].numpy())
                name_list += [f"{os.environ.get('FART')}/output_masks_"+str(self.DATA_IDX)+".npy"]
                np.save(f"{os.environ.get('FART')}/output_masks_"+str(self.DATA_IDX)+".npy", tf.cast(outputs["mask_prob_predictions"], dtype=tf.float32).numpy())
                name_list += [f"{os.environ.get('FART')}/target_masks_"+str(self.DATA_IDX)+".npy"]
                np.save(f"{os.environ.get('FART')}/target_masks_"+str(self.DATA_IDX)+".npy", tf.cast(labels["individual_masks"], dtype=tf.float32).numpy())
                name_list += [f"{os.environ.get('FART')}/output_instance_mask_"+str(self.DATA_IDX)+".npy"]
                np.save(f"{os.environ.get('FART')}/output_instance_mask_"+str(self.DATA_IDX)+".npy", tf.cast(output_instance_mask, dtype=tf.float32).numpy())
                name_list += [f"{os.environ.get('FART')}/output_category_mask_"+str(self.DATA_IDX)+".npy"]
                np.save(f"{os.environ.get('FART')}/output_category_mask_"+str(self.DATA_IDX)+".npy", tf.cast(output_category_mask, dtype=tf.float32).numpy())
                with zipfile.ZipFile(f'{os.environ.get("FART")}/output_{str(self.DATA_IDX)}.zip', 'w',
                                compression=zipfile.ZIP_DEFLATED,
                                compresslevel=9) as zf:
                    for name in name_list: 
                        zf.write(name, arcname=os.path.basename(name))
                        os.remove(name)
                self.total_zips += [f'{os.environ.get("FART")}/output_{str(self.DATA_IDX)}.zip']
                del name_list

                self.DATA_IDX += 1
                
                if self.DATA_IDX > 10: 
                    print('\n'.join(self.total_zips))
                    exit()
            except Exception as e: 
                shutil.rmtree(os.environ.get('FART'))
                exit()  

        if metrics:
            for m in metrics:
                m.update_state(all_losses[m.name])

        return logs


    # def aggregate_logs(self, state=None, step_outputs=None):
    #     is_first_step = not state
    #     if state is None:
    #         state = self.panoptic_quality_metric
    #     state.update_state(
    #         step_outputs['ground_truths'],
    #         step_outputs['predictions'])

    #     return state

    # def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    #     if self.panoptic_quality_metric is not None:
    #         self._reduce_panoptic_metrics(aggregated_logs)
    #         self.panoptic_quality_metric.reset_state()    
    #     return aggregated_logs

    # def _reduce_panoptic_metrics(self, logs: Dict[str, Any]):
    #     """
    #     Updates the per class and mean panoptic metrics in the logs.
    #     """

    #     result = self.panoptic_quality_metric.result()
    #     valid_thing_classes = result['valid_thing_classes']
    #     valid_stuff_classes = result['valid_stuff_classes']
    #     valid_classes = valid_stuff_classes | valid_thing_classes
    #     num_categories = tf.math.count_nonzero(valid_classes, dtype=tf.float32)
    #     num_thing_categories = tf.math.count_nonzero(
    #         valid_thing_classes, dtype=tf.float32
    #     )
    #     num_stuff_categories = tf.math.count_nonzero(
    #         valid_stuff_classes, dtype=tf.float32
    #     )
    #     valid_thing_classes = tf.cast(valid_thing_classes, dtype=tf.float32)
    #     valid_stuff_classes = tf.cast(valid_stuff_classes, dtype=tf.float32)

    #     logs['panoptic_quality/All_num_categories'] = num_categories
    #     logs['panoptic_quality/Things_num_categories'] = num_thing_categories
    #     logs['panoptic_quality/Stuff_num_categories'] = num_stuff_categories
    #     for metric in ['pq', 'sq', 'rq']:
    #         metric_per_class = result[f'{metric}_per_class']
    #         logs[f'panoptic_quality/All_{metric}'] = tf.math.divide_no_nan(
    #             tf.reduce_sum(metric_per_class), num_categories
    #         )
    #         logs[f'panoptic_quality/Things_{metric}'] = tf.math.divide_no_nan(
    #             tf.reduce_sum(metric_per_class * valid_thing_classes),
    #             num_thing_categories,
    #         )
    #         logs[f'panoptic_quality/Stuff_{metric}'] = tf.math.divide_no_nan(
    #             tf.reduce_sum(metric_per_class * valid_stuff_classes),
    #             num_stuff_categories,
    #         )
    #         # if self.task_config.panoptic_quality_evaluator.report_per_class_metrics:
    #         #     for i, is_valid in enumerate(valid_classes.numpy()):
    #         #         if is_valid:
    #         #             logs[f'panoptic_quality/{metric}/class_{i}'] = metric_per_class[i]


