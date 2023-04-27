import numpy as np
import tensorflow as tf
from point_sample import point_sample
import point_sample_expected_values as test_vals

class PointSampleTest(tf.test.TestCase):

    def testBasic(self):

        tf.random.set_seed(0)
        basic_in = tf.random.uniform([1, 14, 10, 2])

        basic_coords = tf.random.uniform([1, 5, 2])


        basic_res = point_sample(basic_in, 
                            tf.tile(basic_coords, [basic_in.shape[0], 1, 1]), 
                            align_corners=False)

        self.assertAllClose(test_vals.basic_exp, basic_res)

    def testMultipleBatches(self):

        tf.random.set_seed(0)
        batch_in = tf.random.uniform([5, 10, 10, 1])

        batch_coords = tf.random.uniform([1, 5, 2])

        batch_res = point_sample(batch_in, 
                            tf.tile(batch_coords, [batch_in.shape[0], 1, 1]), 
                            align_corners=False)

        self.assertAllClose(test_vals.batch_exp, batch_res)


if __name__ == '__main__':
    tf.test.main()