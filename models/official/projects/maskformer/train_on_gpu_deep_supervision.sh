#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-00:10
#SBATCH --mem=80G
#SBATCH --constraint=a100
#SBATCH -A standby
#SBATCH --output=/home/isaeed/tf-maskformer/models/official/projects/maskformer/ib_exps/exp01_deep_supervision.out
#SBATCH --error=/home/isaeed/tf-maskformer/models/official/projects/maskformer/ib_exps/exp01_deep_supervision.error

# module load gcc/9.3.0 
cd /home/isaeed/tf-maskformer/
module load anaconda/2020.11-py38
conda activate /home/isaeed/.conda/envs/cent7/2020.11-py38/tfvenv/
module load cuda/11.7.0 cudnn/cuda-11.7_8.6 gcc/6.3.0

train_bsize=1
eval_bsize=1
export PYTHONPATH=$PYTHONPATH:/home/isaeed/tf-maskformer/models
export MODEL_DIR="./"
export RESNET_CKPT="/depot/davisjam/data/vishal/pretrained_ckpts/tfmg_resnet50/ckpt-62400"
export TFRECORDS_DIR="/depot/davisjam/data/vishal/datasets/coco/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.00005
export NO_OBJ_CLS_WEIGHT=0.1
export DEEP_SUPERVISION=True
export IMG_SIZE=640
export PRINT_OUTPUTS=True
export OVERRIDES="runtime.distribution_strategy=one_device,runtime.num_gpus=1,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$train_bsize,\
task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint=$RESNET_CKPT"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES

