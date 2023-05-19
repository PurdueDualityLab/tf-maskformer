from absl.testing import parameterized
import tensorflow as tf

from official.projects.maskformer.losses.inference import PanopticInference
from official.projects.maskformer.maskformer import MaskFormer

class PanopticInferenceTest(tf.test.TestCase, parameterized.TestCase):
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
        
        out = PanopticInference().call(mask_true=output["class_prob_predictions"], mask_pred=output["mask_prob_predictions"])
        print(out)




if __name__ == '__main__':
    tf.test.main()