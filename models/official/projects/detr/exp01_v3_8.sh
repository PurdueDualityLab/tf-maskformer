#!/bin/bash
train_bsize=64
eval_bsize=8
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/detr_vishal_exps/EXP02_v38"
export TPU_NAME="tf-debug-2"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
# export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
# task.train_data.global_batch_size=$TRAIN_BATCH_SIZE,\
# task.model.which_pixel_decoder=transformer_fpn,\
# task.init_checkpoint=$RESNET_CKPT"
export CONFIG_FILE="models/official/projects/detr/configs/detr_tpu_v3_640_train.yaml"
python3 models/official/projects/detr/train.py \
  --experiment detr_coco_tfrecord \
  --mode train_and_eval \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --config_file $CONFIG_FILE 