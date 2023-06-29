#!/bin/bash
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export BACKBONE_DIR="gs://cam2-models/maskformer_dummy/resnet50_v1"
export DATA_PTH="coco_panoptic"
export MODEL_DIR="gs://cam2-models/maskformer_resnet/v2"
export TPU_NAME="tf-debug-eu-1"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-f"
export OVERRIDES="runtime.distribution_strategy=tpu,\
runtime.enable_xla=True,\
trainer.train_steps=554400,\
trainer.optimizer_config.learning_rate.stepwise.boundaries=[369600],
"
python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode eval \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES