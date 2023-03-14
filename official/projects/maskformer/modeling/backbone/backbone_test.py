from absl.testing import parameterized
import math
import tensorflow as tf

from official.vision.modeling.backbones import resnet

class ResNetTest(parameterized.TestCase, tf.test.TestCase):
    
    @parameterized.parameters(
        (1, 640, 640, 50), (1, 608, 911, 50)
    )
    def test_network_creation(self, batch_size, width, height, model_id):
        tf.keras.backend.set_image_data_format('channels_last')

        network = resnet.ResNet(model_id=model_id)
        self.assertEqual(network.count_params(), 23561152)

        inputs = tf.keras.Input(shape=(width, height, 3), batch_size=1)
        endpoints = network(inputs)

        for x in endpoints.values():
           print(x.shape)

        self.assertAllEqual(
           [batch_size, int(math.ceil(width / 2**2)), int(math.ceil(height / 2**2)), 256]
           , endpoints['2'].shape.as_list(), "failure on 2")
        
        self.assertAllEqual(
           [batch_size, int(math.ceil(width / 2**3)), int(math.ceil(height / 2**3)), 512]
           , endpoints['3'].shape.as_list(), "failure on 3")
        
        self.assertAllEqual(
           [batch_size, int(math.ceil(width / 2**4)), int(math.ceil(height / 2**4)), 1024]
           , endpoints['4'].shape.as_list(), "failure on 4")
        
        self.assertAllEqual(
           [batch_size, int(math.ceil(width / 2**5)), int(math.ceil(height / 2**5)), 2048]
           , endpoints['5'].shape.as_list(), "failure on 5")

if __name__ == '__main__':
  tf.test.main()