#!/bin/bash
train_bsize=2
eval_bsize=2

# Set up environment variables with generic paths
export PYTHONPATH=$PYTHONPATH:/path/to/tf-maskformer/
export RESNET_CKPT="/path/to/pretrained_ckpts/resnet50/ckpt"
export TFRECORDS_DIR="/path/to/datasets/coco/tfrecords"
export MODEL_DIR="./model_dir_2/"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize

export ON_TPU=False

export BASE_LR=0.0001
export IMG_SIZE=640
export NO_OBJ_CLS_WEIGHT=0.001

export DEEP_SUPERVISION=False

export LOG="/path/to/logs/losses.txt"

export OVERRIDES="runtime.distribution_strategy=one_device,runtime.num_gpus=1,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$train_bsize,\
task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint=$RESNET_CKPT"
python3 train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES
