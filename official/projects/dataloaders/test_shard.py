import tensorflow as tf
import factory
from official.projects.configs import factory_config
from official.projects.configs import mode_keys as ModeKeys
from official.projects.dataloaders.distributed_executor import DistributedExecutor
from panoptic_input import mask_former_parser
from PIL import Image
parser_fn = mask_former_parser([1024,1024])


def display_im(feat):
    for key in feat.keys():
        if key != "image":
            print(f"{key}: {feat[key]}")

    print(f"Image shape: {feat['image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(feat["image"].numpy())
    plt.show()
def display_pil_im(tensor):
    im = Image.fromarray(tensor)
    im.show()
raw_dataset = tf.data.TFRecordDataset("/scratch/gilbreth/abuynits/coco_ds/tfrecords/val-00000-of-00008.tfrecord")
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    #print(raw_record)
    parsed_record = parser_fn(raw_record)
    print("==================\n\n\n")
    print(parsed_record[1])
    print(type(parsed_record[0]))
    print(type(parsed_record[1]))
    print(parsed_record[0])
    print(parsed_record[1].keys())
    display_pil_im(parsed_record[0].numpy())




exit(1)
files = tf.io.matching_files("/scratch/gilbreth/abuynits/coco_ds/tfrecords/val-00000-of-00008.tfrecord")
print("CARDINALITY:+++++++++++++++",files.cardinality().numpy())
#file = "/scratch/gilbreth/abuynits/coco_ds/tfrecords/train-00001-of-00032.tfrecord"
# raw_ds = tf.data.TFRecordDataset(file)
# def input_fn(filename):
#     dataset = tf.data.TFRecordDataset(filename)
#     dataset = dataset.shuffle(10).repeat()
#     dataset = dataset.map(parser_fn)
#     dataset = dataset.batch(10)
#     return dataset

# inp_f = input_fn(file)
# print(inp_f)
# print("start")
# for x,y in inp_f:
#     print(x)
# print("done")


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