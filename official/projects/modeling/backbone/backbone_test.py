from absl.testing import parameterized
import tensorflow as tf

from official.vision.modeling.backbones import resnet

class ResNetTest(parameterized.TestCase, tf.test.TestCase):
    
    @parameterized.parameters(
        (640, 50),
    )
    def test_network_creation(self, input_size, model_id):
        tf.keras.backend.set_image_data_format('channels_last')

        network = resnet.ResNet(model_id=model_id)
        self.assertEqual(network.count_params(), 23561152)

        inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
        endpoints = network(inputs)

        self.assertAllEqual(
            [1, 80, 80, 512]
        , endpoints['3'].shape.as_list(), "failure on 3")
        self.assertAllEqual(
            [1,  40, 40, 1024]
        , endpoints['4'].shape.as_list(), "failure on 4")
        self.assertAllEqual(
            [1, 20, 20, 2048]
        , endpoints['5'].shape.as_list(), "failure on 5")

if __name__ == '__main__':
  tf.test.main()