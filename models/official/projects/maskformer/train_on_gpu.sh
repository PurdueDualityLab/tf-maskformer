#!/bin/bash

train_bsize=1
eval_bsize=1
export PYTHONPATH=/depot/davisjam/data/akshath/MaskFormer_tf/tf-maskformer/models
export MODEL_DIR="./"
export TFRECORDS_DIR="/depot/davisjam/data/vishal/datasets/coco/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.00005
export NO_OBJ_CLS_WEIGHT=0.1
export IMG_SIZE=640
export PRINT_OUTPUTS=True
# Akshath
export ON_GPU=True
export OVERRIDES="runtime.distribution_strategy=one_device,runtime.num_gpus=1,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$train_bsize,\
task.model.which_pixel_decoder=transformer_fpn"
python3 train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES > /depot/davisjam/data/akshath/exps/tf/outputs/printing_panoptic_maskformer.txt

