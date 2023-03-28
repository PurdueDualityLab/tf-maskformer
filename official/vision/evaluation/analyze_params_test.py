import tensorflow as tf
import tensorflow.keras as ks
from official.vision.evaluation import analyze_model

class paramaterTesting(tf.test.TestCase):
  def __init__(self):
    self = self
  def test_model1(self):
    #Tensorflow Model 1
    input_shape = (28, 28, 1)
    tfModel = ks.Sequential()
    tfModel.add(ks.layers.Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    tfModel.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
    tfModel.add(ks.layers.Flatten()) # Flattening the 2D arrays for fully connected layers
    tfModel.add(ks.layers.Dense(128, activation=tf.nn.relu))
    tfModel.add(ks.layers.Dropout(0.2))
    tfModel.add(ks.layers.Dense(10,activation=tf.nn.softmax))
    ma = analyze_model.ModelAnalysis(tfModel)
    result = ma.get_parameters()
    expected_result = 600000
    self.assertAllClose(expected_result, result, atol=1e4)
    #Pytorch Model that the Tensorflow Model is tested against
    # class NeuralNet(nn.Module):
    #     def __init__(self):
    #         super(NeuralNet, self).__init__()
    #         self.conv = nn.Conv2d(1, 28, kernel_size=3)
    #         self.pool = nn.MaxPool2d(2)
    #         self.hidden= nn.Linear(28*13*13, 128)
    #         self.drop = nn.Dropout(0.2)
    #         self.out = nn.Linear(128, 10)
    #         self.act = nn.ReLU()
    #     def forward(self, x):
    #         x = self.act(self.conv(x)) # [batch_size, 28, 26, 26]
    #         x = self.pool(x) # [batch_size, 28, 13, 13]
    #         x = x.view(x.size(0), -1) # [batch_size, 28*13*13=4732]
    #         x = self.act(self.hidden(x)) # [batch_size, 128]
    #         x = self.drop(x)
    #         x = self.out(x) # [batch_size, 10]
    #         return x
    # pytorchModel = NeuralNet()
pt = paramaterTesting()
pt.test_model1()