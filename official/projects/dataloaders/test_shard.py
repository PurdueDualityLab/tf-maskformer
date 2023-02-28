import tensorflow as tf
import factory
from official.projects.configs import factory_config
from official.projects.configs import mode_keys as ModeKeys
from official.projects.dataloaders.distributed_executor import DistributedExecutor
from panoptic_input import mask_former_parser
from PIL import Image
import numpy as np
import cv2
parser_fn = mask_former_parser([1024,1024])


def display_im(feat):
    for key in feat.keys():
        if key != "image":
            print(f"{key}: {feat[key]}")

    print(f"Image shape: {feat['image'].shape}")
    
    plt.figure(figsize=(7, 7))
    plt.imshow(feat["image"].numpy())
    plt.show()
def display_pil_im(np_data):
    norm_image = cv2.normalize(np_data, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    im = Image.fromarray(norm_image, 'RGB')
    im.save("/home/abuynits/projects/tf-maskformer/official/projects/dataloaders/img.png")
file_path = "/scratch/gilbreth/abuynits/coco_ds/tfrecords/val-00000-of-00008.tfrecord"
raw_dataset = tf.data.TFRecordDataset(file_path)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    #print(raw_record)
    parsed_record = parser_fn(raw_record)
    print("==================\n\n\n")
    print(parsed_record[1]) # prints image dictionary holding all mask info
    print(type(parsed_record[0])) # print type of image input to model
    print(type(parsed_record[1]))
    np_data = parsed_record[0].numpy()
    print(np_data)
    print(np_data.shape)
    print(parsed_record[1].keys())
    display_pil_im(np_data)




exit(1)
files = tf.io.matching_files("/scratch/gilbreth/abuynits/coco_ds/tfrecords/val-00000-of-00008.tfrecord")
print("CARDINALITY:+++++++++++++++",files.cardinality().numpy())



print("\n\n\n")
raw_dataset = tf.data.TFRecordDataset(file)
print(raw_dataset)
print("\n\n\n")
print(tf.data.experimental.cardinality(raw_dataset))
for raw_record in raw_dataset.take(0):
    print("==================")
    example = tf.train.Example()
    example = example.ParseFromString(raw_record.numpy())
    parsed_ex = train_input_fn(example)
    print(example)
    display_im(example)
    print("==================")
print("=========DONE===========")