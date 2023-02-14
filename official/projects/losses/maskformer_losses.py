import tensorflow as tf

from official.vision.losses import focal_loss

class FocalLoss(focal_loss.FocalLoss):
    """Implements a Focal loss for segmentation problems.
    Reference:
        [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278).
    """

    def __init__(self, alpha=0.25, gamma=2):
        """Initializes `FocalLoss`.
        Args:
        alpha: The `alpha` weight factor for binary class imbalance.
        gamma: The `gamma` focusing parameter to re-weight loss.
        reduction and name?
        """
        super().__init__(alpha, gamma)

    def call(self, y_true, y_pred, num_masks):
        """Invokes the `FocalLoss`.
        Args:
        y_true: A tensor of size [batch, num_anchors, num_classes]. 
        Stores the binary classification lavel for each element in y_pred.
        y_pred: A tensor of size [batch, num_anchors, num_classes]. 
        The predictions of each example.
        num_masks: The number of masks.

        Returns:
        Loss float `Tensor`.
        """
        weighted_loss = super().call(y_true, y_pred)
        loss = tf.math.reduce_sum(tf.math.reduce_mean(weighted_loss,axis=1)) / num_masks
        return loss

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self):
        pass

class ClassificationLoss():
    def call(self, outputs, y_true, indices, num_masks):
        pass

class MaskLoss():
    def call(self, outputs, y_true, indices, num_masks):
        pass

# focal = FocalLoss()
# output_shape = [8, 100, 160, 160]
# tf_pred = tf.random.uniform(output_shape)
# tf_true = tf.random.uniform(output_shape,0,1)
# print(focal.call(tf_true, tf_pred,8))