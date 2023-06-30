from absl.testing import parameterized
import tensorflow as tf
import sys
sys.path.append("/home/isaacjaeminin/tf-maskformer/models")
from official.projects.maskformer.losses.inference import PanopticInference
from official.projects.maskformer.modeling.maskformer import MaskFormer

class PanopticInferenceTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):
        input_image = tf.cast(tf.ones((1, 640, 640, 3)), tf.bfloat16)
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        input_specs = tf.keras.layers.InputSpec(shape=
                                                [1,640,640,3])
        model = MaskFormer(input_specs = input_specs)

        expected_class_probs_shape = [1, 100, 134]
        expected_mask_probs_shape = [1, 160, 160, 100]

        output = model(input_image)
        self.assertAllEqual(
            output["class_prob_predictions"].shape.as_list(), expected_class_probs_shape)
        self.assertAllEqual(
            output["mask_prob_predictions"].shape.as_list(), expected_mask_probs_shape)
        
        out = PanopticInference().call(pred_logits=output["class_prob_predictions"], mask_pred=output["mask_prob_predictions"],
                image_shape = input_image.shape, num_classes = 133)
        print(out)




if __name__ == '__main__':
    tf.test.main()
