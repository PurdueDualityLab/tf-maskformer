#!/bin/bash
train_bsize=16
eval_bsize=16

export PYTHONPATH=$PYTHONPATH:/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models
export RESNET_CKPT="/depot/davisjam/data/vishal/pretrained_ckpts/tfmg_resnet50/ckpt-62400"
export MODEL_DIR="./model_dir/"
export MASKFORMER_CKPT="/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models/official/projects/maskformer/pretrained_ckpts/newest/ckpt-482328"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize

export TPU_NAME="tf-debug-3"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"
export TFRECORDS_DIR="gs://cam2-datasets/coco_panoptic/tfrecords"

export BASE_LR=0.0001
export IMG_SIZE=640
export NO_OBJ_CLS_WEIGHT=0.1

export DEEP_SUPERVISION=True

export ON_TPU=True 
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