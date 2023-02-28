import tensorflow as tf
import factory
from official.projects.configs import factory_config
from official.projects.configs import mode_keys as ModeKeys
from official.projects.dataloaders.distributed_executor import DistributedExecutor
from panoptic_input import mask_former_parser

parser_fn = mask_former_parser([1024,1024])




raw_dataset = tf.data.TFRecordDataset("/scratch/gilbreth/abuynits/coco_ds/tfrecords/val-00000-of-00008.tfrecord")
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    #print(raw_record)
    parsed_record = parser_fn(raw_record)
    print("==================\n\n\n")
    print(parsed_record)




exit(1)
files = tf.io.matching_files("/scratch/gilbreth/abuynits/coco_ds/tfrecords/val-00000-of-00008.tfrecord")
print("CARDINALITY:+++++++++++++++",files.cardinality().numpy())
#file = "/scratch/gilbreth/abuynits/coco_ds/tfrecords/train-00001-of-00032.tfrecord"
# raw_ds = tf.data.TFRecordDataset(file)
def input_fn(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.shuffle(10).repeat()
    dataset = dataset.map(parser_fn)
    dataset = dataset.batch(10)
    return dataset

inp_f = input_fn(file)
print(inp_f)
print("start")
for x,y in inp_f:
    print(x)
print("done")



exit(1)
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
    print("==================")
print("=========DONE===========")


# parsed_ds = raw_dataset.map(parser_fn)
# #one epoch
# parsed_ds = parsed_ds.repeat(1)
# dataset = dataset.batch(10) 

# #Use prefetch() to overlap the producer and consumer.
# print(tf.data.experimental.cardinality(parsed_ds).numpy())
# print(dataset)
# for d in dataset:
#         print("\n\n",d)
# print("done")




# # example = tf.train.Example()
# # example.ParseFromString(elem.numpy())
# for x in elem:
#         display_im(x)
# print("HEREREWWRE")
# print(elem)
#raw_dataset = raw_dataset.shuffle(20).repeat()
#raw_dataset = raw_dataset.map(parser_fn)
# for raw_record in raw_dataset.take(1):
#     print("==================")
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     parsed_ex = train_input_fn(example)
#     print(example)
#     print("==================")
# print("done")
# shards = tf.data.Dataset.from_tensor_slices(files)
# dataset = shards.interleave(
#         map_func=tf.data.TFRecordDataset,
#         cycle_length=32,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)

# dataset = dataset.map(
#         parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# dataset = dataset.batch(batch_size)
# # dataset = dataset.cache() # have to cache after map

# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# print(dataset)



# params = factory_config.config_generator('mask_former')
# batch_size = 32
# mode=ModeKeys.TRAIN



#distrib_exec = DistributedExecutor(strategy, params)

#iterable_ds = distrib_exec.get_input_iterator(train_input_fn, strategy)
# ex = next(iter(dataset))
# for ab in iterable_ds:
#  display_im(ab)

"""

"""