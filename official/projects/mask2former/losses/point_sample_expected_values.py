import numpy as np
import tensorflow as tf

basic_exp = tf.constant([[[0.4979663, 0.4045799],
                          [0.5969796, 0.7357068],
                          [0.2395323, 0.2803430],
                          [0.4051819, 0.5656694],
                          [0.6149617, 0.4268470]]])

batch_exp = np.array([[
                       [0.5737163],
                       [0.7603718],
                       [0.08194923],
                       [0.44565836],
                       [0.48250335]
                      ],
                      [
                       [0.6739312],
                       [0.5845002],
                       [0.11694234],
                       [0.2802606],
                       [0.5458771]
                      ],
                      [
                       [0.20102058],
                       [0.58093923],
                       [0.8336261],
                       [0.23765936],
                       [0.43750793]
                      ],
                      [
                       [0.4061655],
                       [0.36624318],
                       [0.47478595],
                       [0.39743522],
                       [0.7312156]
                      ],
                      [
                       [0.6187626],
                       [0.87563187],
                       [0.40689203],
                       [0.47812977],
                       [0.58478826]
                      ]])