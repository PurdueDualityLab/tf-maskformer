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
        assert len(image.shape) == 3
        assert image.shape[-1] == 3
        assert image <= 255
        image_count += 1
    
print("Total images :", image_count)
