#!/bin/bash
# module load anaconda/2020.11-py38
# module load cuda/11.7.0 cudnn/cuda-11.7_8.6 gcc/6.3.0
# conda activate tfmaskformer
train_bsize=1
eval_bsize=1

export PYTHONPATH=$PYTHONPATH:/depot/davisjam/data/akshath/MaskFormer_tf/tf-maskformer/models
export RESNET_CKPT="/depot/davisjam/data/vishal/pretrained_ckpts/tfmg_resnet50/ckpt-62400"
export TFRECORDS_DIR="/depot/davisjam/data/vishal/datasets/coco/tfrecords"

export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize

export BASE_LR=0.0001
export IMG_SIZE=640
export NO_OBJ_CLS_WEIGHT=0.01

export DEEP_SUPERVISION=True
export ON_CPU=True
export PRINT_OUTPUTS=False

export OVERRIDES="runtime.distribution_strategy=one_device,runtime.mixed_precision_dtype=float32,\
task.validation_data.global_batch_size=$EVAL_BATCH_SIZE,task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint_modules=all,\
task.init_checkpoint=$MASKFORMER_CKPT"
python3 train.py \
  --experiment maskformer_coco_panoptic \
  --mode eval \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES