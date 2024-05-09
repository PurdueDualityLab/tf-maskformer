#!/bin/bash
train_bsize=16
eval_bsize=16

export PYTHONPATH=$PYTHONPATH:/path/to/models
export RESNET_CKPT="/path/to/pretrained/checkpoints/resnet50/ckpt"
export MODEL_DIR="./model_dir/"
export MASKFORMER_CKPT="/path/to/maskformer/pretrained/ckpt"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize

export TPU_NAME="your-tpu-name"
export TPU_SOFTWARE="tpu-version"
export TPU_PROJECT="gcp-project-id"
export TPU_ZONE="tpu-zone"
export TFRECORDS_DIR="/path/to/tfrecords"

export ON_TPU=True 

export BASE_LR=0.0001
export IMG_SIZE=640
export NO_OBJ_CLS_WEIGHT=0.001

export DEEP_SUPERVISION=True

export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
task.validation_data.global_batch_size=$EVAL_BATCH_SIZE,task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint_modules=all,\
task.init_checkpoint=$MASKFORMER_CKPT"
python3 train.py \
  --experiment maskformer_coco_panoptic \
  --mode eval \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES
