# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Data loader and input processing."""

from official.core import config_definitions as cfg
from typing import Optional, Text
import tensorflow as tf


class InputFn(object):
    """Input function that creates dataset from files."""

    def __init__(self,
                 params: cfg.DataConfig,
                 dataset_fn,
                 parser_fn,
                 num_examples: Optional[int] = -1):
        """Initialize.
        Args:
          file_pattern: the file pattern for the data example (TFRecords).
          params: the parameter object for constructing example parser and model.
          mode: ModeKeys.TRAIN or ModeKeys.Eval
          batch_size: the data batch size.
          num_examples: If positive, only takes this number of examples and raise
            tf.errors.OutOfRangeError after that. If non-positive, it will be
            ignored.
        """
        self._is_training = params.is_training
        self._file_pattern = params.input_path
        
        self._batch_size = params.global_batch_size
        self._shuffle_buffer_size = params.shuffle_buffer_size
        self._num_examples = num_examples
        self._parser_fn = parser_fn
        
        self._dataset_fn = dataset_fn
        if dataset_fn is None:
            
            self._dataset_fn = tf.data.TFRecordDataset

        self._input_sharding = (not self._is_training)
        try:
            if self._is_training:
                self._input_sharding = params.train.input_sharding
            else:
                self._input_sharding = params.eval.input_sharding
        except AttributeError:
            pass

    def __call__(self, ctx=None, batch_size: int = None):
        """Provides tf.data.Dataset object.
        Args:
          ctx: context object.
          batch_size: expected batch size input data.
        Returns:
          tf.data.Dataset object.
        """
        if not batch_size:
            batch_size = self._batch_size
        assert batch_size is not None
        dataset = tf.data.Dataset.list_files(self._file_pattern,
           shuffle=self._is_training)
        
        
        if self._input_sharding and ctx and ctx.num_input_pipelines > 1:
            dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
        dataset = dataset.cache()
        if self._is_training:
            dataset = dataset.repeat()

        dataset = dataset.interleave(
            map_func=self._dataset_fn,
            cycle_length=32,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #dataset = dataset.cache()
        #dataset = dataset.map(self._parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
 
        if self._is_training:
           #dataset = dataset.repeat()
            dataset = dataset.shuffle(self._shuffle_buffer_size)
        if self._num_examples > 0:
            dataset = dataset.take(self._num_examples)
        
        # Parses the fetched records to input tensors for model function.
        
        dataset = dataset.map(
            self._parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
