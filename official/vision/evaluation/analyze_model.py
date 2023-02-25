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

"""Metrics involving the Model"""

import numpy as np
import tensorflow as tf

class ModelAnalysis:
    """Computes metrics including the number of parameters for a model, the FLOPS (Floating Point Operations per Second)
    and activations"""

    def __init__(self, model):
        """
        Args:
            model: A reference to a tf.keras model
        """
        self.model = model

    def get_parameters(self):
        """

        Returns: a number corresponding to the parameter count of trainable and un-trainable variables within a model

        """
        return self.model.count_params()

