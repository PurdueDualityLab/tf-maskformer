import tensorflow as tf
import factory
from official.projects.configs import factory_config
from official.projects.configs import mode_keys as ModeKeys
from official.projects.dataloaders.distributed_executor import DistributedExecutor
from panoptic_input import mask_former_parser

parser_fn = mask_former_parser()

files = tf.io.matching_files("/scratch/gilbreth/abuynits/coco_ds/coco_tfrecords/train-00027-of-01000.tfrecord")

raw_dataset = tf.data.TFRecordDataset("/scratch/gilbreth/abuynits/coco_ds/coco_tfrecords/train-00003-of-01000.tfrecord")
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    #print(raw_record)
    parsed_record = parser_fn(raw_record)
    print("==================\n\n\n")
    print(parsed_record)
print("==================")
shards = tf.data.Dataset.from_tensor_slices(files)
dataset = shards.interleave(
        map_func=tf.data.TFRecordDataset,
        cycle_length=32,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.map(
        parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.batch(batch_size)
# dataset = dataset.cache() # have to cache after map

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print(dataset)



params = factory_config.config_generator('mask_former')
batch_size = 32
mode=ModeKeys.TRAIN



#distrib_exec = DistributedExecutor(strategy, params)

#iterable_ds = distrib_exec.get_input_iterator(train_input_fn, strategy)
ex = next(iter(dataset))
for ab in iterable_ds:
 display_im(ab)