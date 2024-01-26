# !/bin/bash
# module load gcc/9.3.0 
# module load anaconda/2020.11-py38
# module load cuda/11.7.0 cudnn/cuda-11.7_8.6 gcc/6.3.0
# conda activate tfmaskformer
train_bsize=4
eval_bsize=4

export PYTHONPATH=$PYTHONPATH:/depot/davisjam/data/akshath/MaskFormer_vishal/tf-maskformer/models
export RESNET_CKPT="/depot/davisjam/data/vishal/pretrained_ckpts/tfmg_resnet50/ckpt-62400"
export TFRECORDS_DIR="/depot/davisjam/data/vishal/datasets/coco/tfrecords"
export MODEL_DIR="./model_dir/"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize

export BASE_LR=0.0001
export IMG_SIZE=640
export NO_OBJ_CLS_WEIGHT=0.01

export DEEP_SUPERVISION=0
export ON_TPU=0
export PRINT_OUTPUTS=True

export OVERRIDES="runtime.distribution_strategy=one_device,runtime.num_gpus=1,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$train_bsize,\
task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint=$RESNET_CKPT"
python3 train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES
