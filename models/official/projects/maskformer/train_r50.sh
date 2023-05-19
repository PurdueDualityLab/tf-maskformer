$ export MODEL_DIR="gs://cam2-models/maskformer"
$ export DATA_PTH="gs://cam2-datasets/coco_panoptic"
$ export TPU_NAME="tf-train-2"
$ export TPU_PROJECT="red-atlas-305317"
$ export TPU_ZONE=europe-west4-a
$ export OVERRIDES="task.validation_data.input_path=${DATA_PTH},\
task.train_data.input_path=${DATA_PTH},\
runtime.distribution_strategy=tpu"

$ python3 models/official/projects/maskformer/train.py \
  --experiment maskformer_coco_panoptic \
  --mode train \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES

