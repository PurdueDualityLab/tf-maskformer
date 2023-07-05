fusermount -u ~/datasets
fusermount -u ~/models
gcsfuse --implicit-dirs cam2-datasets ~/datasets
gcsfuse cam2-models ~/models
export PYTHONPATH=$PYTHONPATH:~/tf-maskformer/models
export MODEL_DIR="gs://cam2-models/detr_test"
export TPU_NAME="tf-train-1"
export TPU_SOFTWARE="2.11.0"
export TPU_PROJECT="red-atlas-305317"
export TPU_ZONE="us-central1-a"

nohup python3 train.py \
	--experiment detr_coco_tfrecord\
	--mode train \
	--model_dir $MODEL_DIR \
	--tpu $TPU_NAME \
	> logs_tfrecords_256.txt &

#	--params_override=task.init_checkpoint='gs://cam2-models/maskformer_resnet/ckpt-118280'\
#	> logs_tfrecords.txt &
