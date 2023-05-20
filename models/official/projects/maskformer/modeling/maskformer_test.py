from official.projects.maskformer.modeling.maskformer import MaskFormer
from absl.testing import parameterized
import tensorflow as tf

class MaskFormerTest(tf.test.TestCase, parameterized.TestCase):
    # TODO(ibrahim): Add more testcases.
    @parameterized.named_parameters(('test1', 256, 100, 256, "5", 0, 6, 199, 1))
    def test_pass_through(self,
                        fpn_feat_dims,
                        num_queries,
                        hidden_size,
                        backbone_endpoint_name,
                        num_encoder_layers,
                        num_decoder_layers,
                        num_classes,
                        batch_size):    
        input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            [640, 640, 3])    
        maskformer = MaskFormer(input_specs= input_specs, hidden_size=hidden_size,
                                 backbone_endpoint_name=backbone_endpoint_name,
                                 num_encoder_layers=num_encoder_layers,
                                 num_decoder_layers=num_decoder_layers,
                                 num_classes=num_classes,
                                 batch_size=batch_size)

        input_image = tf.ones((1, 640, 640, 3))
        
        # expected_class_probs_shape = [1, 100, 172]
        # expected_mask_probs_shape = [1, 160, 160, 100]
        expected_class_probs_shape = [1, 100, 101] # B, dim of logits, number of classes
        expected_mask_probs_shape = [1, 184, 275, 100] # B,H,W,C

        output = maskformer(input_image)
        print(output.keys())
        exit()
        self.assertAllEqual(
            output["class_prob_predictions"].shape.as_list(), expected_class_probs_shape)
        self.assertAllEqual(
            output["mask_prob_predictions"].shape.as_list(), expected_mask_probs_shape)


if __name__ == '__main__':
    tf.test.main()

