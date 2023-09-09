#!/bin/bash
# v128 default
fusermount -u ~/datasets
fusermount -u ~/models
gcsfuse --implicit-dirs cam2-datasets ~/datasets
gcsfuse cam2-models ~/models
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/detr_exp1"
export TPU_NAME="tf-train-1"
export TPU_SOFTWARE="2.12.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="europe-west4-a"
export CONFIG_FILE="configs/detr_tpu_v128.yaml"
# export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
# task.validation_data.global_batch_size=2,task.model.which_pixel_decoder=transformer_fpn,\
# task.init_checkpoint_modules=all,\
# task.init_checkpoint=$MODEL_DIR"
nohup python3 train.py \
	--experiment detr_coco_tfrecord\
	--mode train \
	--model_dir $MODEL_DIR \
	--config_file $CONFIG_FILE \
    --tpu $TPU_NAME \
	> logs_detr_exp1.txt &

