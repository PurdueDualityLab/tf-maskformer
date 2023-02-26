import input_reader
from official.projects.configs import factory_config
from official.common import distribute_utils
from official.modeling.hyperparams import params_dict
import distributed_executor as executor
from absl import flags
from official.utils import hyperparams_flags
from official.utils.flags import core as flags_core
import sys
from panoptic_input import TfExampleDecoder

FLAGS = flags.FLAGS
argv = FLAGS(sys.argv)
hyperparams_flags.initialize_common_flags()
flags_core.define_log_steps()
train_input_fn = None
eval_input_fn = None

params = factory_config.config_generator('mask_former')

params.override(
    {
        'strategy_type': FLAGS.strategy_type,
        'model_dir': FLAGS.model_dir,
        'strategy_config': executor.strategy_flags_dict(),
    },
    is_strict=False)
params = params_dict.override_params_dict(
    params, FLAGS.config_file, is_strict=True)
params = params_dict.override_params_dict(
    params, FLAGS.params_override, is_strict=True)

training_file_pattern = params.train.train_file_pattern
eval_file_pattern = params.eval.eval_file_pattern
print(f"training file pattern:{training_file_pattern}")
print(f"eval file pattern:{eval_file_pattern}")
if not training_file_pattern and not eval_file_pattern:
    raise ValueError('Must provide at least one of training_file_pattern and '
                     'eval_file_pattern.')

if training_file_pattern:
    # Use global batch size for single host.
    train_input_fn = input_reader.InputFn(
        file_pattern=training_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.TRAIN,
        batch_size=params.train.batch_size)

if eval_file_pattern:
    eval_input_fn = input_reader.InputFn(
        file_pattern=eval_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.PREDICT_WITH_GT,
        batch_size=params.eval.batch_size,
        num_examples=params.eval.eval_samples)
# call it this way to get dataset object
strategy_config = params.strategy_config
distribute_utils.configure_cluster(strategy_config.worker_hosts,
                                   strategy_config.task_index)
strategy = distribute_utils.get_distribution_strategy(
    distribution_strategy=params.strategy_type,
    num_gpus=strategy_config.num_gpus,
    all_reduce_alg=strategy_config.all_reduce_alg,
    num_packs=strategy_config.num_packs,
    tpu_address=strategy_config.tpu)
import matplotlib as plt


def display_im(feat):
    for key in feat.keys():
        if key != "image":
            print(f"{key}: {feat[key]}")

    print(f"Image shape: {feat['image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(feat["image"].numpy())
    plt.show()


a = train_input_fn()
# pain_and_suffering = DistributedExecutor(strategy, params)
# iterator = a.make_one_shot_iterator()
# ex = next(iterator)
# print(a)
# display_im(ex)


# iterable_ds = pain_and_suffering.get_input_iterator(train_input_fn, strategy)

# for a in iterable_ds.map(TfExampleDecoder.decode):


# ex = next(iterable_ds)
# display_im(ex)
decoder = TfExampleDecoder()


def decode_fn(serializable_example):
    return decoder.decode(serializable_example)


decoded_ds = a.map(decode_fn)

for features in decoded_ds:
    print(features)
    # display_im(features)

#
# for x in a:
#     print(x)
# ex = next(ds)
# print(ex)
