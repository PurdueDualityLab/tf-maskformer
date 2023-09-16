#!/bin/bash
train_bsize=2
eval_bsize=2
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/detr_vishal_exps/EXP01_cpu"
export RESNET_CKPT="gs://cam2-models/maskformer_vishal_exps/resnet50_pretrained/tfmg/ckpt-62400"
export TPU_NAME="tf-debug-1"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export OVERRIDES="runtime.distribution_strategy=one_device,runtime.mixed_precision_dtype=float32,\
task.train_data.global_batch_size=$TRAIN_BATCH_SIZE,\
task.init_checkpoint=$RESNET_CKPT"
export CONFIG_FILE="models/official/projects/detr/configs/detr_tpu_v3_640_train_cpu.yaml"
python3 models/official/projects/detr/train.py \
  --experiment detr_coco_tfrecord \
  --mode train_and_eval \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES