#!/bin/bash
train_bsize=1
eval_bsize=1
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/maskformer_vishal_exps/EXP26_v8_eval"
export MASKFORMER_CKPT="gs://cam2-models/maskformer_vishal_exps/EXP26_v8/ckpt-482328"
export RESNET_CKPT="gs://cam2-models/maskformer_vishal_exps/resnet50_pretrained/tfmg/ckpt-62400"
export TFRECORDS_DIR="gs://cam2-datasets/coco_panoptic/tfrecords"
export TRAIN_BATCH_SIZE=$train_bsize
export EVAL_BATCH_SIZE=$eval_bsize
export BASE_LR=0.0001
export IMG_SIZE=640

export NO_OBJ_CLS_WEIGHT=0.01
export MASK_THRESH=0.4
export CLASS_THRESH=0.4
export OVERLAP_THRESH=0.3
export FART=~/tf-maskformer/predicted_npy/exp26_0.4_0.3_0.01

export DEEP_SUPERVISION=True
export ON_CPU=True
export PRINT_OUTPUTS=True
export OVERRIDES="runtime.distribution_strategy=one_device,runtime.mixed_precision_dtype=float32,\
task.validation_data.global_batch_size=$EVAL_BATCH_SIZE,task.model.which_pixel_decoder=transformer_fpn,\
task.init_checkpoint_modules=all,\
task.init_checkpoint=$MASKFORMER_CKPT"
python3 train.py \
  --experiment maskformer_coco_panoptic \
  --mode eval \
  --model_dir $MODEL_DIR \
  --params_override=$OVERRIDES
