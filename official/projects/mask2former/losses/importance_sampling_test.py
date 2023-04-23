import tensorflow as tf
from official.projects.mask2former.losses.importance_sampling import (
    get_uncertain_point_coords_with_randomness,
    calculate_uncertainty,
)
from official.projects.mask2former.losses.importance_sampling_expected_values import (
    oneBatch,
    largeBatch,
    squareSrcMask,
    largeNumPoints,
)


class ImportanceSamplingTest(tf.test.TestCase):
    OVERSAMPLE_RATIO = 3.0
    IMPORTANCE_SAMPLE_RATIO = 0.75
    NUM_POINTS = 15

    def testOneBatch(self):
        tf.random.set_seed(0)
        src_mask = tf.random.uniform(shape=[1, 10, 14, 1])

        expected_res = oneBatch

        point_coords = get_uncertain_point_coords_with_randomness(
            coarse_logits=src_mask,
            uncertainty_func=(lambda logits: calculate_uncertainty(logits)),
            num_points=self.NUM_POINTS,
            oversample_ratio=self.OVERSAMPLE_RATIO,
            importance_sample_ratio=self.IMPORTANCE_SAMPLE_RATIO,
        )

        self.assertAllClose(point_coords, expected_res)
        self.assertEqual(point_coords.shape[0], src_mask.shape[0])
        self.assertEqual(point_coords.shape[1], self.NUM_POINTS)
        self.assertEqual(point_coords.shape[2], 2)

    def testLargeBatchSize(self):
        tf.random.set_seed(0)
        src_mask = tf.random.uniform(shape=[10, 10, 14, 1])

        expected_res = largeBatch

        point_coords = get_uncertain_point_coords_with_randomness(
            coarse_logits=src_mask,
            uncertainty_func=(lambda logits: calculate_uncertainty(logits)),
            num_points=self.NUM_POINTS,
            oversample_ratio=self.OVERSAMPLE_RATIO,
            importance_sample_ratio=self.IMPORTANCE_SAMPLE_RATIO,
        )

        self.assertAllClose(point_coords, expected_res)
        self.assertEqual(point_coords.shape[0], src_mask.shape[0])
        self.assertEqual(point_coords.shape[1], self.NUM_POINTS)
        self.assertEqual(point_coords.shape[2], 2)

    def testSquareSrcMask(self):
        tf.random.set_seed(0)
        src_mask = tf.random.uniform(shape=[3, 15, 15, 1])

        expected_res = squareSrcMask

        point_coords = get_uncertain_point_coords_with_randomness(
            coarse_logits=src_mask,
            uncertainty_func=(lambda logits: calculate_uncertainty(logits)),
            num_points=self.NUM_POINTS,
            oversample_ratio=self.OVERSAMPLE_RATIO,
            importance_sample_ratio=self.IMPORTANCE_SAMPLE_RATIO,
        )

        self.assertAllClose(point_coords, expected_res)
        self.assertEqual(point_coords.shape[0], src_mask.shape[0])
        self.assertEqual(point_coords.shape[1], self.NUM_POINTS)
        self.assertEqual(point_coords.shape[2], 2)

    def testLargeNumPoints(self):
        tf.random.set_seed(0)
        src_mask = tf.random.uniform(shape=[1, 10, 14, 1])

        expected_res = largeNumPoints

        point_coords = get_uncertain_point_coords_with_randomness(
            coarse_logits=src_mask,
            uncertainty_func=(lambda logits: calculate_uncertainty(logits)),
            num_points=300,
            oversample_ratio=self.OVERSAMPLE_RATIO,
            importance_sample_ratio=self.IMPORTANCE_SAMPLE_RATIO,
        )

        self.assertAllClose(point_coords, expected_res)
        self.assertEqual(point_coords.shape[0], src_mask.shape[0])
        self.assertEqual(point_coords.shape[1], 300)
        self.assertEqual(point_coords.shape[2], 2)


if __name__ == "__main__":
    tf.test.main()
