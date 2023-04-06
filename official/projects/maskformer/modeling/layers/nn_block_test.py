from absl.testing import parameterized
import tensorflow as tf

from official.projects.maskformer.modeling.layers.nn_block import MLPHead

class MaskFormerTransformerTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(("test1", "coco_stuff", 256, 256, 171, 100, 8), ("test2", "coco_panoptic", 256, 256, 133, 100, 1))
    def test_pass_through(self,
                          testcase_input_name,
                          mask_dim,
                          hidden_size,
                          num_classes,
                          num_queries,
                          batch_size):

        mlp_head = MLPHead(
            num_classes=num_classes, hidden_dim=hidden_size, mask_dim=mask_dim)

        testcase_inputs = {
            "coco_stuff": {
                "per_segment_embeddings": tf.ones((6, 8, 100, 256)),
                "per_pixel_embeddings": tf.ones((8, 160, 160, 256))
            },
            "coco_panoptic": {
                "per_segment_embeddings": tf.ones((6, 1, 100, 256)),
                "per_pixel_embeddings": tf.ones((1, 152, 228, 256))
            }
        }
        # expected_class_probs_shape = [8, 100, 172]
        expected_class_probs_shape = [batch_size, num_queries, num_classes + 1]

        # expected_mask_probs_shape = [8, 100, 160, 160]
        expected_mask_probs_shape = [batch_size,
                                    testcase_inputs[testcase_input_name]["per_pixel_embeddings"].shape[1],
                                    testcase_inputs[testcase_input_name]["per_pixel_embeddings"].shape[2],
                                    num_queries]

        output = mlp_head(testcase_inputs[testcase_input_name])

        self.assertAllEqual(output["class_prob_predictions"].shape.as_list(), expected_class_probs_shape)
        self.assertAllEqual(output["mask_prob_predictions"].shape.as_list(), expected_mask_probs_shape)

if __name__ == '__main__':
    tf.test.main()
