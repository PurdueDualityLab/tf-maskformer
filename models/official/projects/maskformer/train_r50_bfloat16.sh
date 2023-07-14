#!/bin/bash
fusermount -u ~/datasets
fusermount -u ~/models
gcsfuse --implicit-dirs cam2-datasets ~/datasets
gcsfuse cam2-models ~/models
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/maskformer_vishal_exps/EXP_02_BFLOAT"
export BACKBONE_DIR="gs://cam2-models/maskformer_dummy/resnet50_v1"
export TPU_NAME="tf-debug-3"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"
export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=mixed_bfloat16,\
task.train_data.global_batch_size=128,"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES

