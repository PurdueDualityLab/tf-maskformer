use strict;
#use warnings;
use 5.010;
use warnings;
use constant INPUT_PARAM => 3;
if ($#ARGV != INPUT_PARAM) {
 print "ERROR: wrong number of inputs!\n";
 exit 1;
}
coco_file_path = argv[0];
tfrecord_file_path = argv[1];
build_coco_path = argv[2];
unless (-d $coco_file_path) {
  print "$coco_file_path does not exist!\n";
  exit 1;
}
unless (-d $tfrecord_file_path) {
  print "$tfrecord_file_path does not exist!\n";
  exit 1;
}
unless (-e $build_coco_path) {
  print "$build_coco_path does not exist!\n";
  exit 1;
}
# download urls for train, validation and annotation
train_coco_url = "http://images.cocodataset.org/zips/train2017.zip";
val_coco_url = "http://images.cocodataset.org/zips/val2017.zip";
annotations_coco_url = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip";
# download the file paths
system("wget $val_coco_path -p $coco_file_path");
system("wget $train_coco_path -p $coco_file_path");
system("wget $annotations_coco_path -p $coco_file_path");

# unzip the file paths
system("unzip $coco_file_path/train2017.zip &> /dev/null");
system("unzip $coco_file_path/val2017.zip &> /dev/null");
system("unzip $coco_file_path/panoptic_annotations_trainval2017.zip &> /dev/null");

system("python $build_coco_path --coco_root=$coco_file_path --output_dir=$tfrecord_file_path");