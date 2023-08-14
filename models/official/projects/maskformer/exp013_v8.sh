#!/bin/bash
train_bsize=32
eval_bsize=16
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export RESNET_CKPT="gs://cam2-models/maskformer_vishal_exps/resnet50_pretrained/tfmg/ckpt-62400"
export MASKFORMER_CKPT="gs://cam2-models/maskformer_vishal_exps/EXP12_v8/ckpt-77616"
export MODEL_DIR="gs://cam2-models/maskformer_vishal_exps/EXP13_v128"
export TPU_NAME="tf-debug-10"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="europe-west4-a"
export TFRECORDS_DIR="gs://cam2-datasets/coco_panoptic/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.0001
export IMG_SIZE=640
export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$TRAIN_BATCH_SIZE,task.model.which_pixel_decoder=transformer_fpn"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES