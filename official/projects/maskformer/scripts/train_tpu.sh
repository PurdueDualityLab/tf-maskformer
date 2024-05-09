#!/bin/bash
train_bsize=128
eval_bsize=8

export PYTHONPATH=$PYTHONPATH:/path/to/models
export RESNET_CKPT="/path/to/resnet50_pretrained/checkpoint"
export MODEL_DIR="/path/to/output/models"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize

export TPU_NAME="your-tpu-name"
export TPU_SOFTWARE="version"
export TPU_PROJECT="your-gcp-project-id"
export TPU_ZONE="your-tpu-zone"
export TFRECORDS_DIR="/path/to/datasets"

export ON_TPU=True 

export BASE_LR=0.0001
export IMG_SIZE=640
export NO_OBJ_CLS_WEIGHT=0.001

export DEEP_SUPERVISION=True

export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$TRAIN_BATCH_SIZE,\
task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint=$RESNET_CKPT"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES
