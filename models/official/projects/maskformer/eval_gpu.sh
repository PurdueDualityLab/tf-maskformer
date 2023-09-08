#!/bin/bash
train_bsize=2
eval_bsize=2
export PYTHONPATH=$PYTHONPATH:/depot/qqiu/data/vishal/tf-maskformer/models
export MODEL_DIR="./"
export MASKFORMER_CKPT="/depot/qqiu/data/vishal/tf-maskformer/EXP18_v128/ckpt-28644"
export TFRECORDS_DIR="/depot/davisjam/data/vishal/datasets/coco/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.0001
export NO_OBJ_CLS_WEIGHT=0.1
export IMG_SIZE=1280
export PRINT_OUTPUTS=True
export OVERRIDES="runtime.distribution_strategy=one_device,runtime.num_gpus=1,runtime.mixed_precision_dtype=float32,\
task.validation_data.global_batch_size=$EVAL_BATCH_SIZE,\
task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint_modules=all,\
task.init_checkpoint=$MASKFORMER_CKPT"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode eval \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES