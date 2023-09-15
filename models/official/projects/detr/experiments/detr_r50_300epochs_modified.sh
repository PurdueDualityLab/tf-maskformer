#!/bin/bash
python3 official/projects/detr/train.py \
  --experiment=detr_coco_tfrecord \
  --mode=train_and_eval \
  --model_dir="./" \
  --params_override=task.init_checkpoint='/depot/davisjam/data/vishal/pretrained_ckpts/tfmg_resnet50/ckpt-62400',trainer.train_steps=554400,trainer.optimizer_config.learning_rate.stepwise.boundaries="[369600]"
  
