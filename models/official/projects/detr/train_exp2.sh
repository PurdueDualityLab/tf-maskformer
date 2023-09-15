#!/bin/bash
# v3-8 default
fusermount -u ~/datasets
fusermount -u ~/models
gcsfuse --implicit-dirs cam2-datasets ~/datasets
gcsfuse cam2-models ~/models
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/detr_vishal_exps"
export TPU_NAME="tf-debug-3"
export TPU_SOFTWARE="2.12.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"
export CONFIG_FILE="configs/detr_tpu_v3_640.yaml"
# export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
# task.validation_data.global_batch_size=2,task.model.which_pixel_decoder=transformer_fpn,\
# task.init_checkpoint_modules=all,\
# task.init_checkpoint=$MODEL_DIR"
python3 models/projects/official/detr/train.py \
	--experiment detr_coco_tfrecord\
	--mode train \
	--model_dir $MODEL_DIR \
	--config_file $CONFIG_FILE \
    --tpu $TPU_NAME 
