#!/bin/bash
train_bsize=16
eval_bsize=16
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/maskformer_vishal_exps/EXP13_v8_eval"
export MASKFORMER_CKPT="gs://cam2-models/maskformer_vishal_exps/EXP20_v8/ckpt-18480"
export RESNET_CKPT="gs://cam2-models/maskformer_vishal_exps/resnet50_pretrained/tfmg/ckpt-62400"
export TPU_NAME="tf-debug-3"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"
export TFRECORDS_DIR="gs://cam2-datasets/coco_panoptic/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.0001
export IMG_SIZE=640
export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
task.validation_data.global_batch_size=$EVAL_BATCH_SIZE,task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint_modules=all,\
task.init_checkpoint=$MASKFORMER_CKPT"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode eval \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES