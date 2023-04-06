$ export MODEL_DIR="gs://<path-to-model-directory>"
$ export TPU_NAME="<tpu-name>"
$ export ANNOTATION_FILE="gs://<path-to-coco-annotation-json>"
$ export TRAIN_DATA="gs://<path-to-train-data>"
$ export EVAL_DATA="gs://<path-to-eval-data>"
$ export OVERRIDES="task.validation_data.input_path=${EVAL_DATA},\
task.train_data.input_path=${TRAIN_DATA},\
task.annotation_file=${ANNOTATION_FILE},\
runtime.distribution_strategy=tpu"


$ python3 train.py \
  --experiment panoptic_r50_coco \
  --config_file configs/experiments/panoptic_coco_r50.yaml \
  --mode train \
  --model_dir $MODEL_DIR \
  --tpu $TPU_NAME \
  --params_override=$OVERRIDES

