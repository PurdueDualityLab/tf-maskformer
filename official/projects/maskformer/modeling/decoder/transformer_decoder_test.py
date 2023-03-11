from absl.testing import parameterized
import tensorflow as tf

# from transformer import MaskFormerTransformer
from official.projects.maskformer.modeling.decoder.transformer_decoder import MaskFormerTransformer


class MaskFormerTransformerTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(("test1", "coco_stuff", "5", 8, 100, 256, 171,),
                                    ("test2", "coco_panoptic", "5", 1, 100, 256, 133,))
    def test_pass_through(self,
                          testcase_input_name,
                          backbone_endpoint_name,
                          batch_size,
                          num_queries,
                          hidden_size,
                          num_classes):

        transformer = MaskFormerTransformer(backbone_endpoint_name=backbone_endpoint_name,
                                            batch_size=batch_size,
                                            num_queries=num_queries,
                                            hidden_size=hidden_size,
                                            num_classes=num_classes,
                                            num_encoder_layers=0,
                                            num_decoder_layers=6,
                                            dropout_rate=0.1)

        testcase_input_image = {
            "coco_stuff": tf.ones((1, 640, 640, 3)),
            "coco_panoptic": tf.ones((1, 736, 975, 3)),
        }
        
        testcase_backbone_inputs = {
            "coco_stuff": {
                "2": tf.ones([1, 160, 160, 256]),
                "3": tf.ones([1, 80, 80, 512]),
                "4": tf.ones([1, 40, 40, 1024]),
                "5": tf.ones([1, 20, 20, 2048])
            },
            "coco_panoptic": {
                "2": tf.ones([1, 184, 244, 256]),
                "3": tf.ones([1, 92, 122, 512]),
                "4": tf.ones([1, 38, 57, 1024]),
                "5": tf.ones([1, 19, 29, 2048])
            }
        }

        expected_output_shape = [6, batch_size, num_queries, 256]

        output = transformer(
            {"image": testcase_input_image[testcase_input_name], "features": testcase_backbone_inputs[testcase_input_name]})
        output_shape = [len(output)] + output[0].shape.as_list()

        self.assertAllEqual(output_shape, expected_output_shape)


if __name__ == "__main__":
    tf.test.main()
