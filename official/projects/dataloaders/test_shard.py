import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from skimage import segmentation
from skimage import color

from official.projects.dataloaders import input_reader
from official.projects.dataloaders import panoptic_input
from official.common import dataset_fn
import sys
import tensorflow as tf
from official.core import exp_factory as factory
from official.projects.configs import maskformer_cfg

params = factory.get_exp_config("detr_coco_tfrecord").task.train_data
decoder_cfg = params.decoder
if decoder_cfg.type == 'simple_decoder':
    decoder = panoptic_input.TfExampleDecoder(
        regenerate_source_id = params.regenerate_source_id)
else:
    raise ValueError('Unknown decoder type: {}!'.format(
        params.decoder.type))

parser_fn = panoptic_input.mask_former_parser(params.parser,
                            decoder.decode)
reader = input_reader.InputFn(
    params,
    dataset_fn = dataset_fn.pick_dataset_fn(params.file_type),
    parser_fn = parser_fn)

cfg_test = factory.get_exp_config("detr_coco_tfrecord")
# parser_fn = mask_former_parser([400,400])


file_path = "/scratch/gilbreth/abuynits/coco_ds/tfrecords/val-00002-of-00008.tfrecord"  # specify the filepath to tfrecord
save_im_path = "/home/abuynits/projects/img.png"  # image save path for displaying image
im_mask_path = "/home/abuynits/projects/mask.png"  # image save path for displaying image
combined_im_path = "/home/abuynits/projects/combined_im2.png"  # image save path for displaying image


# returns de-normalized tensor
def get_un_normalized_np(im_tensor):
    np_data = im_tensor.numpy()
    print(np_data.shape)

    norm_image = cv2.normalize(np_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image


# displays an image
def display_pil_im(np_data, file_path, greyscale=False):
    # print(np_data)
    # print(np_data.shape)
    if greyscale == False:
        im = Image.fromarray(np_data, 'RGB')
    else:
        im = Image.fromarray(np_data, 'L')
    im.save(file_path)


def get_overlayed_im(im, mask):
    im = im.astype(np.uint8)
    mask = (color.label2rgb(mask) * 255).astype(np.uint8)
    print("image shape", im.shape)
    print("mask shape", mask.shape)
    out = cv2.addWeighted(im, 0.5, mask, 0.5, 0)
    return im, mask, out


raw_dataset = tf.data.TFRecordDataset(file_path)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    parsed_record = parser_fn(raw_record)
    print("==================\n\n\n")
    print(parsed_record[1])  # prints image dictionary holding all mask info
    print(parsed_record[1]['category_mask'])
    print(parsed_record[1].keys())  # prints available dictionary info keys
    im, mask, combined_im = get_overlayed_im(
        get_un_normalized_np(parsed_record[0]),
        tf.squeeze(parsed_record[1]['category_mask']).numpy()
    )
    display_pil_im(im, save_im_path)  # displays the pil image
    display_pil_im(mask, im_mask_path)
    display_pil_im(combined_im, combined_im_path)
