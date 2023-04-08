from absl.testing import parameterized
import tensorflow as tf

# from transformer import MaskFormerTransformer
from official.projects.maskformer.modeling.decoder.transformer_decoder import MaskFormerTransformer

class MaskFormerTransformerTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('test1', '5', 8, 100, 256, 10,))
    def test_pass_through(self,
                        backbone_endpoint_name,
                        batch_size,
                        num_queries,
                        hidden_size,
                        num_classes):    

        multilevel_features = {
            "2": tf.ones([1, 160, 160, 256]),
            "3": tf.ones([1, 80, 80, 512]),
            "4": tf.ones([1, 40, 40, 1024]),
            "5": tf.ones([1, 20, 20, 2048])
        }

        transformer = MaskFormerTransformer(backbone_endpoint_name=backbone_endpoint_name,
                                            batch_size=batch_size,
                                            num_queries=num_queries,
                                            hidden_size=hidden_size,
                                            num_classes=num_classes,
                                            num_encoder_layers=0,
                                            num_decoder_layers=6,
                                            dropout_rate=0.1)

        input_image = tf.ones((1, 640, 640, 3))
        expected_output_shape = [6, 8, 100, 256]

        output = transformer({"image": input_image, "features": multilevel_features })
        output_shape = [len(output)] + output[0].shape.as_list()

        self.assertAllEqual(output_shape, expected_output_shape)

if __name__ == '__main__':
    tf.test.main()
