#!/bin/bash
# module load gcc/9.3.0 
# conda activate maskformer  
# cd ~/tf-maskformer/models/official/projects/maskformer
# module load anaconda/2020.11-py38
train_bsize=1
eval_bsize=1
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="./"
export RESNET_CKPT="/depot/davisjam/data/vishal/pretrained_ckpts/tfmg_resnet50/ckpt-62400"
export TFRECORDS_DIR="/depot/davisjam/data/vishal/datasets/coco/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.0001
export IMG_SIZE=640

export NO_OBJ_CLS_WEIGHT=0.01
export MASK_THRESH=0.4
export CLASS_THRESH=0.4
export OVERLAP_THRESH=0.3
export FART=~/tf-maskformer/predicted_npy/exp26_0.4_0.3_0.01
export LOG=~/tf-maskformer/predicted_npy/exp26_0.4_0.3_0.01


export DEEP_SUPERVISION=True
export ON_CPU=True
export PRINT_OUTPUTS=True

export OVERRIDES="runtime.distribution_strategy=one_device,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$TRAIN_BATCH_SIZE,\
task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint=$RESNET_CKPT"
python3 train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES

