#Tensorflow unit test code
import tensorflow as tf
import numpy as np
from tensorflow import Tensor

from loguru import logger

# load img.npy, pad_img.npy, and pad_img_after.npy and test the following:
# using .copy_ function to copy the values from pad_img to img
# write the test using tensorflow functions
# write the test using numpy functions

def _max_by_axis(the_list):
        all_max = the_list[0]
        for sublist in the_list[1:]:
            for idx, item in enumerate(sublist):
                all_max[idx] = max(all_max[idx], item)
        return all_max

class NestedTensor(object):
        def __init__(self, tensors, mask=None):
            self.tensors = tf.convert_to_tensor(tensors)
            self.mask = tf.convert_to_tensor(mask) if mask is not None else None

        def to(self, device):
            # type: (Device) -> NestedTensor # noqa
            with tf.device(device):
                cast_tensor = tf.identity(self.tensors)
                cast_mask = tf.identity(self.mask) if self.mask is not None else None
            return NestedTensor(cast_tensor, cast_mask)

        def decompose(self):
            return self.tensors, self.mask

        def __repr__(self):
            return str(self.tensors)
        

def _onnx_nested_tensor_from_tensor_list(tensor_list) -> NestedTensor:
    max_size = tf.reduce_max([tf.shape(img) for img in tensor_list], axis=0)
    padded_imgs = []
    padded_masks = []

    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = tf.pad(img, [[0, padding[2]], [0, padding[1]], [0, padding[0]]], mode="CONSTANT")
        padded_imgs.append(padded_img)
        
        with tf.device(img.device): 
            m = tf.zeros_like(img[0], dtype=tf.int32)
        padded_mask = tf.pad(m, [[0, padding[2]], [0, padding[1]]], mode="CONSTANT", constant_values=1)
        padded_masks.append(tf.cast(padded_mask, tf.bool))
    
    tensor = tf.stack(padded_imgs)
    mask = tf.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)
 
class TestPad(tf.test.TestCase):
    def test_pad(self):
        tensor_list = np.load('tensor_list_batch2.npy', allow_pickle=True)
        tensor_list = [tf.convert_to_tensor(tensor) for tensor in tensor_list]
        logger.debug(f"tensor_list[0].shape is {tensor_list[0].shape}")
        logger.debug(f"tensor_list[1].shape is {tensor_list[1].shape}")
        logger.debug(f"len(tensor_list) is {len(tensor_list)}")
        
        loaded_img = np.load("img_batch2.npy")
        loaded_pad_img = np.load("pad_img_batch2.npy")
        loaded_pad_img_after_torch = np.load("pad_img_after_batch2.npy")
        loaded_img = tf.convert_to_tensor(loaded_img)
        loaded_pad_img = tf.convert_to_tensor(loaded_pad_img)
        loaded_pad_img_after_torch = tf.convert_to_tensor(loaded_pad_img_after_torch)
        
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        logger.debug(f"max_size is {max_size}")

        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape

        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        with tf.device(device):
            tensor = tf.zeros(batch_shape, dtype=dtype)
            mask = tf.ones((b, h, w), dtype=tf.bool)

        # logger.debug(f"tensor.shape is {tensor.shape}")
        # logger.debug(f"mask.shape is {mask.shape}")
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            logger.debug(f"img.shape is {img.shape}")
            logger.debug(f"loaded_img.shape is {loaded_img.shape}")
            logger.debug(f"pad_img.shape is {pad_img.shape}")
            logger.debug(f"loaded_pad_img.shape is {loaded_pad_img.shape}")

            self.assertAllEqual(img, loaded_img)
            self.assertAllEqual(pad_img, loaded_pad_img)
            
            
            # Implementation 2
            pad_img_2 = tf.Variable(pad_img)
            pad_img_2[:img.shape[0], :img.shape[1], :img.shape[2]].assign(img)

            self.assertAllEqual(loaded_pad_img_after_torch, pad_img_2)
            break



        


     
        # self.assertAllEqual(pad_img_after, pad_img_after_torch)

if __name__ == "__main__":
    tf.test.main()


'''
max_size = _max_by_axis([list(img.shape) for img in tensor_list])
# min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
batch_shape = [len(tensor_list)] + max_size
b, c, h, w = batch_shape
logger.debug(f"batch_shape is {batch_shape}")
logger.debug(f"batch_size is {b}")
logger.debug(f"num_channels is {c}")
logger.debug(f"height is {h}")
logger.debug(f"width is {w}")
dtype = tensor_list[0].dtype
device = tensor_list[0].device
tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
for img, pad_img, m in zip(tensor_list, tensor, mask):
logger.critical(f"img.shape is {img.shape}")
logger.critical(f"pad_img.shape is {pad_img.shape}")
logger.critical(f"m.shape is {m.shape}")
logger.critical(f"max_size is {max_size}")
logger.error(f"img is {img}")
logger.error(f"pad_img is {pad_img}")

np.save("img_batch2.npy", img.detach().cpu().numpy())
np.save("pad_img_batch2.npy", pad_img.detach().cpu().numpy())
pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
logger.error(f"pad_img_after is {pad_img}")
np.save("pad_img_after_batch2.npy", pad_img.detach().cpu().numpy())
m[: img.shape[1], : img.shape[2]] = False
'''