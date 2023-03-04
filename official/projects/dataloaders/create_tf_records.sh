DATA_DIR=$1
GET_ANNOT='wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
GET_VAL='wget http://images.cocodataset.org/zips/val2017.zip'
GET_PAN='wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip'
GET_TRAIN='wget http://images.cocodataset.org/zips/train2017.zip'

downloaded_panoptic=true
run_dir=$(pwd)
if ! cd "$DATA_DIR"; then
  echo "ERROR: can't access working directory ($DATA_DIR)" >&2
  exit 1
fi

if [ -f "$DATA_DIR/zips/annotations_trainval2017.zip" ]; then
    echo "annotations_trainval2017.zip exists!"
else 
    `$GET_ANNOT`
fi

if [ -f "$DATA_DIR/zips/panoptic_annotations_trainval2017.zip" ]; then
    echo "panoptic_annotations_trainval2017.zip exists!"
    downloaded_panoptic=false
else 
    `$GET_PAN`
fi


if [ -f "$DATA_DIR/zips/val2017.zip" ]; then
    echo "val2017.zip exists!"
else 
    `$GET_VAL`
fi

if [ -f "$DATA_DIR/zips/train2017.zip" ]; then
    echo "train2017.zip exists!"
else 
    `$GET_TRAIN`
fi
cd $run_dir

unzip $DATA_DIR/"*".zip -d $DATA_DIR;
mkdir $DATA_DIR/zips;
mv $DATA_DIR/*.zip $DATA_DIR/zips;

if $downloaded_panoptic ; then
    unzip $DATA_DIR/annotations/panoptic_train2017.zip -d $DATA_DIR
    unzip $DATA_DIR/annotations/panoptic_val2017.zip -d $DATA_DIR
fi

python3 official/vision/data/create_coco_tf_record.py \
  --logtostderr  \
  --image_dir="$DATA_DIR/val2017" \
  --object_annotations_file="$DATA_DIR/annotations/instances_val2017.json"  \
  --output_file_prefix="$DATA_DIR/tfrecords/val"  \
  --panoptic_annotations_file="$DATA_DIR/annotations/panoptic_val2017.json" \
  --panoptic_masks_dir="$DATA_DIR/panoptic_val2017" \
  --num_shards=8 \
  --include_masks \
  --include_panoptic_masks

python3 official/vision/data/create_coco_tf_record.py \
  --logtostderr  \
  --image_dir="$DATA_DIR/train2017" \
  --object_annotations_file="$DATA_DIR/annotations/instances_train2017.json"  \
  --output_file_prefix="$DATA_DIR/tfrecords/train"  \
  --panoptic_annotations_file="$DATA_DIR/annotations/panoptic_train2017.json" \
  --panoptic_masks_dir="$DATA_DIR/panoptic_train2017" \
  --num_shards=150 \
  --include_masks \
  --include_panoptic_masks
