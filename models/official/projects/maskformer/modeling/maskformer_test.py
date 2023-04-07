from official.projects.maskformer.modeling.maskformer import MaskFormer
from absl.testing import parameterized
import tensorflow as tf


class MaskFormerTest(tf.test.TestCase, parameterized.TestCase):
    # TODO(ibrahim): Add more testcases.
    @parameterized.named_parameters(('test1', 'coco_stuff', 100, 171), ('test2', 'coco_panoptic', 100, 133))
    def test_pass_through(self, testcase_input_name, num_queries, num_classes):

        model = MaskFormer(num_queries=num_queries, num_classes=num_classes)

        # input_image = tf.ones((1, 640, 640, 3))
        testcase_input = {
            "coco_stuff": tf.ones((1, 640, 640, 3)),
            "coco_panoptic": tf.ones((1, 608, 911, 3))
        }
        
        # TODO(ibrahim): Add num_queries and make expected output shape dynamic after adding parameters.
        # expected_class_probs_shape = [1, 100, 172]
        # expected_mask_probs_shape = [1, 160, 160, 100]

        testcases_expected_output = {
            "coco_stuff": {
                "class_prob_predictions": [1, 100, 172],
                "mask_prob_predictions": [1, 160, 160, 100]
            },
            "coco_panoptic": {
                "class_prob_predictions": [1, num_queries, 134], 
                "mask_prob_predictions": [1, 152, 228, num_queries]
            }
        }

        output = model(testcase_input[testcase_input_name])

        self.assertAllEqual(
            output["class_prob_predictions"].shape.as_list(), testcases_expected_output[testcase_input_name]["class_prob_predictions"])
        self.assertAllEqual(
            output["mask_prob_predictions"].shape.as_list(), testcases_expected_output[testcase_input_name]["mask_prob_predictions"])


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.test.main()
