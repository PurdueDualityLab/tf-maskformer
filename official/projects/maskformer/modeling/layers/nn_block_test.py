from absl.testing import parameterized
import tensorflow as tf

from official.projects.maskformer.modeling.layers.nn_block import MLPHead

class MaskFormerTransformerTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(("test1", 256, 256, 171))
    def test_pass_through(self,
                          mask_dim,
                          hidden_size,
                          num_classes):

        MLP_head = MLPHead(
            num_classes=num_classes, hidden_dim=hidden_size, mask_dim=mask_dim)

        inputs = {"per_segment_embeddings": tf.ones((6, 8, 100, 256)),
                 "per_pixel_embeddings": tf.ones((8, 160, 160, 256))}

        expected_class_probs_shape = [8, 100, 172]
        expected_mask_probs_shape = [8, 100, 160, 160]

        output = MLP_head(inputs)

        self.assertAllEqual(output["class_prob_predictions"].shape.as_list(), expected_class_probs_shape)
        self.assertAllEqual(output["mask_prob_predictions"].shape.as_list(), expected_mask_probs_shape)

if __name__ == '__main__':
    tf.test.main()
