#!/bin/bash
fusermount -u ~/datasets
fusermount -u ~/models
gcsfuse --implicit-dirs cam2-datasets ~/datasets
gcsfuse cam2-models ~/models
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/maskformer_tpu_profiling"
export BACKBONE_DIR="gs://cam2-models/maskformer_dummy/resnet50_v1"
export DATA_PTH="coco_panoptic"
export TPU_NAME="tf-debug-1"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"
# export OVERRIDES="task.validation_data.input_path=${DATA_PTH},\
# task.train_data.input_path=${DATA_PTH},\
# runtime.distribution_strategy=tpu"
export OVERRIDES="runtime.distribution_strategy=tpu,\
runtime.enable_xla=True,\
trainer.optimizer_config.learning_rate.stepwise.boundaries=[369600],"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES

