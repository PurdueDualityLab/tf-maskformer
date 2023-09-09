#!/bin/bash
fusermount -u ~/datasets
fusermount -u ~/models
gcsfuse --implicit-dirs cam2-datasets ~/datasets
gcsfuse cam2-models ~/models
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/detr_test_eval"
export TPU_NAME="tf-debug-2"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"
# export OVERRIDES="runtime.distribution_strategy=tpu,runtime.mixed_precision_dtype=float32,\
# task.validation_data.global_batch_size=2,task.model.which_pixel_decoder=transformer_fpn,\
# task.init_checkpoint_modules=all,\
# task.init_checkpoint=$MODEL_DIR"
nohup python3 train.py \
	--experiment detr_coco_tfrecord\
	--mode eval \
	--model_dir $MODEL_DIR \
    --tpu $TPU_NAME \
	> logs_detr.txt &

