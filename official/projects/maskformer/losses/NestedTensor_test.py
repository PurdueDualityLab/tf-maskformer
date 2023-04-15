#Tensorflow unit test code for function NestedTensor()
import tensorflow as tf
import numpy as np


def _max_by_axis(the_list):
        all_max = the_list[0]
        for sublist in the_list[1:]:
            for idx, item in enumerate(sublist):
                all_max[idx] = max(all_max[idx], item)
        return all_max
 
class TestPad(tf.test.TestCase):
    def test_pad(self):
        tensor_list = np.load('tensor_list_batch2.npy', allow_pickle=True)
        tensor_list = [tf.convert_to_tensor(tensor) for tensor in tensor_list]

        
        loaded_img = np.load("img_batch2.npy")
        loaded_pad_img = np.load("pad_img_batch2.npy")
        loaded_pad_img_after_torch = np.load("pad_img_after_batch2.npy")
        loaded_img = tf.convert_to_tensor(loaded_img)
        loaded_pad_img = tf.convert_to_tensor(loaded_pad_img)
        loaded_pad_img_after_torch = tf.convert_to_tensor(loaded_pad_img_after_torch)
        
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])

        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        with tf.device(device):
            tensor = tf.zeros(batch_shape, dtype=dtype)
            mask = tf.ones((b, h, w), dtype=tf.bool)


        for img, pad_img, m in zip(tensor_list, tensor, mask):
            self.assertAllEqual(img, loaded_img)
            self.assertAllEqual(pad_img, loaded_pad_img)
            
            
            pad_img = tf.Variable(pad_img)
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].assign(img)

            self.assertAllEqual(loaded_pad_img_after_torch, pad_img)
            break
