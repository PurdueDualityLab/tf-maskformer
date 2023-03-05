from official.projects.losses.maskformer_losses import FocalLoss
from absl.testing import parameterized
import tensorflow as tf

class FocalLossTest(tf.test.TestCase, parameterized.TestCase):
    # TODO(ibrahim): Add more testcases.
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):

        model = MaskFormer()

        input_image = tf.ones((1, 640, 640, 3))

        expected_class_probs_shape = [1, 100, 172]
        expected_mask_probs_shape = [1, 160, 160, 100]

        output = model(input_image)

        self.assertAllEqual(
            output["class_prob_predictions"].shape.as_list(), expected_class_probs_shape)
        self.assertAllEqual(
            output["mask_prob_predictions"].shape.as_list(), expected_mask_probs_shape)


if __name__ == '__main__':
    tf.test.main()