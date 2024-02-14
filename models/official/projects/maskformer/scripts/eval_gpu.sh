#!/bin/bash
module load gcc/9.3.0 
module load anaconda/2020.11-py38
module load cuda/11.7.0 cudnn/cuda-11.7_8.6 gcc/6.3.0
conda activate tfmaskformer
train_bsize=2
eval_bsize=2

export PYTHONPATH=$PYTHONPATH:/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models
export MODEL_DIR="./model_dir/"
export MASKFORMER_CKPT="/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models/official/projects/maskformer/pretrained_ckpts/newest/ckpt-482328"
export RESNET_CKPT="/depot/davisjam/data/vishal/pretrained_ckpts/tfmg_resnet50/ckpt-62400"
export TFRECORDS_DIR="/depot/davisjam/data/vishal/datasets/coco/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize

export ON_TPU=False

export BASE_LR=0.0001
export IMG_SIZE=640
export NO_OBJ_CLS_WEIGHT=0.001

export DEEP_SUPERVISION=True

export OVERRIDES="runtime.distribution_strategy=one_device,runtime.mixed_precision_dtype=float32,\
task.validation_data.global_batch_size=$EVAL_BATCH_SIZE,task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint_modules=all,\
task.init_checkpoint=$MASKFORMER_CKPT"
python3 train.py \
  --experiment maskformer_coco_panoptic \
  --mode eval \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES