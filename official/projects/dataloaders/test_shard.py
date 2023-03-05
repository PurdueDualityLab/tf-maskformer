import sys
sys.path.append("/home/abuynits/projects/tf-maskformer")
import tensorflow as tf
import factory
from official.projects.configs import factory_config
from official.projects.configs import mode_keys as ModeKeys
from panoptic_input import mask_former_parser
from PIL import Image
import numpy as np
import cv2
from skimage import segmentation
from skimage import color 


parser_fn = mask_former_parser([1024,1024])
file_path = "/scratch/gilbreth/abuynits/coco_ds/tfrecords/val-00002-of-00008.tfrecord"# specify the filepath to tfrecord
save_im_path = "/home/abuynits/projects/img.png" # image save path for displaying image
im_mask_path = "/home/abuynits/projects/mask.png" # image save path for displaying image
combined_im_path = "/home/abuynits/projects/combined_im2.png" # image save path for displaying image
# returns de-normalized tensor
def get_un_normalized_np(im_tensor):
    np_data = im_tensor.numpy()
    print(np_data.shape)

    norm_image = cv2.normalize(np_data, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image
    
# displays an image
def display_pil_im(np_data,file_path,greyscale=False):
    print(np_data)
    print(np_data.shape)
    if greyscale == False: 
        im = Image.fromarray(np_data, 'RGB')
    else:
        im = Image.fromarray(np_data,'L')
    im.save(file_path)
def get_overlayed_im(im,mask):
    im = im.astype(np.uint8)
    print(mask)
    print(mask.shape)
    mask = (color.label2rgb(mask)*255).astype(np.uint8)
    print(mask.shape)
    print("mask:",mask)
    print(im.shape)
    out = cv2.addWeighted(im, 0.5, mask, 0.5,0)
    return im,mask,out

raw_dataset = tf.data.TFRecordDataset(file_path)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    parsed_record = parser_fn(raw_record)
    print("==================\n\n\n")
    print(parsed_record[1]) # prints image dictionary holding all mask info
    print(parsed_record[1]['category_mask'])
    print(parsed_record[1].keys()) # prints available dictionary info keys
    im,mask,combined_im = get_overlayed_im(
        get_un_normalized_np(parsed_record[0]),
        tf.squeeze(parsed_record[1]['category_mask']).numpy()
        )
    display_pil_im(im,save_im_path) # displays the pil image
    display_pil_im(mask,im_mask_path)
    display_pil_im(combined_im,combined_im_path)