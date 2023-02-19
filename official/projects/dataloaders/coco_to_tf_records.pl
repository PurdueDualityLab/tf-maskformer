use strict;
#use warnings;
use 5.010;
use warnings;
use constant INPUT_PARAM => 3;
if ($#ARGV != INPUT_PARAM) {
 print "ERROR: wrong number of inputs!\n expected 3 got $#ARGV\n";
 exit 1;
}
my $coco_file_path = $ARGV[0];
my $tfrecord_file_path = $ARGV[1];
my $build_coco_path = $ARGV[2];
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
my $train_coco_url = "http://images.cocodataset.org/zips/train2017.zip";
my $val_coco_url = "http://images.cocodataset.org/zips/val2017.zip";
my $annotations_coco_url = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip";
# download the file paths
unless (-d $tfrecord_file_path) {
  print "$tfrecord_file_path does not exist!\n";
  exit 1;
}

unless (-d "$coco_file_path/train2017.zip") {
  print "downloading $coco_file_path/train2017.zip\n";
  system("wget $train_coco_url -P $coco_file_path");
}else{
  print "$coco_file_path/train2017.zip already exists!\n";
}
unless (-d "$coco_file_path/val2017.zip") {
  print "downloading $coco_file_path/val2017.zip\n";
  system("wget $val_coco_url -P $coco_file_path");
}else{
  print "$coco_file_path/val2017.zip already exists!\n";
}
unless (-d "$coco_file_path/panoptic_annotations_trainval2017.zip") {
  print "downloading $coco_file_path/panoptic_annotations_trainval2017.zip\n";
  system("wget $annotations_coco_url -P $coco_file_path");
}else{
print "$coco_file_path/panoptic_annotations_trainval2017.zip already exists!\n";
}

system("unzip $coco_file_path/train2017.zip -d $coco_file_path");
system("unzip $coco_file_path/val2017.zip -d $coco_file_path");
system("unzip $coco_file_path/panoptic_annotations_trainval2017.zip -d $coco_file_path");

system("python $build_coco_path --coco_root=$coco_file_path --output_dir=$tfrecord_file_path");