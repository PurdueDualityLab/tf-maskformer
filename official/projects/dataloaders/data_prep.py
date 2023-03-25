from official.projects.dataloaders import input_reader
from official.projects.configs import mode_keys as ModeKeys
from official.projects.configs import factory_config
from official.projects.dataloaders import panoptic_input
from official.common import distribute_utils
from official.modeling.hyperparams import params_dict
from absl import flags
from official.utils import hyperparams_flags
from official.utils.flags import core as flags_core
import sys
import tensorflow as tf

from official.projects.configs import maskformer_cfg
from official.core import exp_factory as factory

# FLAGS = flags.FLAGS
# argv = FLAGS(sys.argv)
# hyperparams_flags.initialize_common_flags()
# flags_core.define_log_steps()
# train_input_fn = None
# eval_input_fn = None

# params = factory_config.config_generator('mask_former')

# params = params_dict.override_params_dict(
#     params, FLAGS.config_file, is_strict=True)
# params = params_dict.override_params_dict(
#     params, FLAGS.params_override, is_strict=True)

# training_file_pattern = params.train.train_file_pattern
# eval_file_pattern = params.eval.eval_file_pattern
# print(f"training file pattern:{training_file_pattern}")
# print(f"eval file pattern:{eval_file_pattern}")
# if not training_file_pattern and not eval_file_pattern:
#     raise ValueError('Must provide at least one of training_file_pattern and '
#                      'eval_file_pattern.')

# if training_file_pattern:
#     # Use global batch size for single host.
#     train_input_fn = input_reader.InputFn(
#         file_pattern=training_file_pattern,
#         params=params,
#         mode=ModeKeys.TRAIN,
#         batch_size=params.train.batch_size)

# if eval_file_pattern:
#     eval_input_fn = input_reader.InputFn(
#         file_pattern=eval_file_pattern,
#         params=params,
#         mode=ModeKeys.PREDICT_WITH_GT,
#         batch_size=params.eval.batch_size,
#         num_examples=params.eval.eval_samples)

# train_ds = train_input_fn()
# print(train_ds)

def build_inputs(params):
    decoder_cfg = params.task.train_data.decoder
    if decoder_cfg.type == 'simple_decoder':
        decoder = panoptic_input.TfExampleDecoder(
            regenerate_source_id = False)
    else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type))
    
    parser = panoptic_input.mask_former_parser(params.task.train_data,
                                decoder.decode)
    return input_reader(
        params.DataConfig,
        dataset_fn = dataset_fn.pick_dataset_fn(params.file_type),
        parser_fn = parser)
cfg_test = factory.get_exp_config("detr_coco_tfrecord")
print(cfg_test.task.train_data.get('decoder'))
print(build_inputs(factory.get_exp_config("detr_coco_tfrecord")))