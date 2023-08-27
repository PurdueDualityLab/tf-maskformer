#!/bin/bash
train_bsize=128
eval_bsize=128
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export RESNET_CKPT="gs://cam2-models/maskformer_vishal_exps/resnet50_pretrained/tfmg/ckpt-62400"
export MODEL_DIR="gs://cam2-models/maskformer_vishal_exps/EXP14_v128"
export TPU_NAME="tf-train-1"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="europe-west4-a"
export TFRECORDS_DIR="gs://cam2-datasets/coco_panoptic/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.0001
export IMG_SIZE=1280
export PRINT_OUTPUTS=False
export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$TRAIN_BATCH_SIZE,\
task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint=$RESNET_CKPT"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode train_and_eval \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES
