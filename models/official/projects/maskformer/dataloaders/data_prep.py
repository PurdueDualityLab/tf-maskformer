# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data preparation for MaskFormer."""

from official.projects.dataloaders import input_reader
from official.projects.configs import mode_keys as ModeKeys
from official.projects.configs import factory_config
from official.modeling.hyperparams import params_dict
from absl import flags
from official.utils import hyperparams_flags
from official.utils.flags import core as flags_core
import sys

FLAGS = flags.FLAGS
argv = FLAGS(sys.argv)
hyperparams_flags.initialize_common_flags()
flags_core.define_log_steps()
train_input_fn = None
eval_input_fn = None

params = factory_config.config_generator('mask_former')

params = params_dict.override_params_dict(
    params, FLAGS.config_file, is_strict=True)
params = params_dict.override_params_dict(
    params, FLAGS.params_override, is_strict=True)

training_file_pattern = params.train.train_file_pattern
eval_file_pattern = params.eval.eval_file_pattern
print(f"Training file pattern:{training_file_pattern}")
print(f"Eval file pattern:{eval_file_pattern}")
if not training_file_pattern and not eval_file_pattern:
  raise ValueError('Must provide at least one of training_file_pattern and '
                   'eval_file_pattern.')

if training_file_pattern:
  # Use global batch size for single host.
  train_input_fn = input_reader.InputFn(
      file_pattern=training_file_pattern,
      params=params,
      mode=ModeKeys.TRAIN,
      batch_size=params.train.batch_size)

if eval_file_pattern:
  eval_input_fn = input_reader.InputFn(
      file_pattern=eval_file_pattern,
      params=params,
      mode=ModeKeys.PREDICT_WITH_GT,
      batch_size=params.eval.batch_size,
      num_examples=params.eval.eval_samples)

train_ds = train_input_fn()
