import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from official.projects.maskformer.dataloaders import panoptic_input
import os
import tensorflow as tf
from panoptic_input import mask_former_parser

tfrecord_path = "/home/vishalpurohit55595/datasets/coco_panoptic/tfrecords"  # specify the filepath to tfrecord
# get list of tfrecord files
file_paths = tf.io.gfile.glob(tfrecord_path + "/*.tfrecord")
decoder = panoptic_input.TfExampleDecoder()
image_count = 0

for each_file in file_paths:
    raw_dataset = tf.data.TFRecordDataset(os.path.join(tfrecord_path, each_file))
    print("Reading file :", os.path.join(tfrecord_path, each_file))
    for raw_record in raw_dataset.take(-1):
        data = decoder.decode(raw_record)
        # print("Decoded file :", os.path.join(tfrecord_path, each_file))
        image = data['image']
        contigious_mask = tf.cast(data['groundtruth_panoptic_contigious_mask'][:, :, 0],
            dtype=tf.float32)
        instance_mask = tf.cast(
            data['groundtruth_panoptic_instance_mask'][:, :, 0],
            dtype=tf.float32)
        category_mask = tf.cast(
            data['groundtruth_panoptic_category_mask'][:, :, 0],
            dtype=tf.float32)
        
        h,w,c = image.shape
        
        assert len(image.shape) == 3 
        assert image.shape[-1] == 3
        assert image.numpy().all() <= 255
        assert image.numpy().all() >= 0
        assert image.numpy().shape[0] >= 0
        assert image.numpy().shape[1] >= 0

        assert len(contigious_mask.shape) == 2
        assert contigious_mask.numpy().all() <= 132
        assert contigious_mask.numpy().all() >= 0
        assert contigious_mask.numpy().shape[0] >= 0
        assert contigious_mask.numpy().shape[1] >= 0
        assert contigious_mask.numpy().shape[0] == h
        assert contigious_mask.numpy().shape[1] == w

        assert len(category_mask.shape) == 2
        assert category_mask.numpy().all() <= 199
        assert category_mask.numpy().all() > 0
        assert category_mask.numpy().shape[0] >= 0
        assert category_mask.numpy().shape[1] >= 0
        assert category_mask.numpy().shape[0] == h
        assert category_mask.numpy().shape[1] == w

        assert len(instance_mask.shape) == 2
        assert instance_mask.numpy().all() <= 132
        assert instance_mask.numpy().all() >= 0
        assert instance_mask.numpy().shape[0] >= 0
        assert instance_mask.numpy().shape[1] >= 0
        assert instance_mask.numpy().shape[0] == h
        assert instance_mask.numpy().shape[1] == w

        image_count += 1
    
print("Total images :", image_count)
