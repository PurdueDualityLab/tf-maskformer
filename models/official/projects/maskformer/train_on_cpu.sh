#!/bin/bash
train_bsize=8
eval_bsize=8
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export RESNET_CKPT="gs://cam2-models/maskformer_vishal_exps/resnet50_pretrained/tfmg/ckpt-62400"
export MASKFORMER_CKPT="gs://cam2-models/maskformer_vishal_exps/EXP12_v8/ckpt-77616"
export MODEL_DIR="gs://cam2-models/maskformer_vishal_exps/EXP01_CPU"
export TFRECORDS_DIR="gs://cam2-datasets/coco_panoptic/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.0001
export IMG_SIZE=640
export OVERRIDES="runtime.distribution_strategy=one_device,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$TRAIN_BATCH_SIZE,task.model.which_pixel_decoder=transformer_fpn"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode train_and_eval \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES
