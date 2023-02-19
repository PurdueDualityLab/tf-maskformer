from absl.testing import parameterized
import tensorflow as tf

import sys
sys.path.insert(0, 'C:\\programmingStuff\\tensorflowStuff\\tf-maskformer')

from official.projects.maskformer.maskformer import MaskFormer


class MaskFormerTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('test1',))
    def test_pass_through(self):    

        model = MaskFormer()

        input_image = tf.ones((1, 640, 640, 3))
        class_prob_prediction, mask_prob_prediction = model(input_image)

        print("class_prob_prediction:", class_prob_prediction.shape)
        print("mask_prob_prediction:", mask_prob_prediction.shape)

    # backbone_endpoint_name = '5'
    # model = detr.DETR(backbone, backbone_endpoint_name, num_queries,
    #                   hidden_size, num_classes)
    # outs = model(tf.ones((batch_size, image_size, image_size, 3)))
    # self.assertLen(outs, 6)  # intermediate decoded outputs.
    # for out in outs:
    #   self.assertAllEqual(
    #       tf.shape(out['cls_outputs']), (batch_size, num_queries, num_classes))
    #   self.assertAllEqual(
    #       tf.shape(out['box_outputs']), (batch_size, num_queries, 4))

    # def test_get_from_config_detr_transformer(self):
    #   config = {
    #       'num_encoder_layers': 1,
    #       'num_decoder_layers': 2,
    #       'dropout_rate': 0.5,
    #   }
    #   detr_model = detr.DETRTransformer.from_config(config)
    #   retrieved_config = detr_model.get_config()

    #   self.assertEqual(config, retrieved_config)

    # def test_get_from_config_detr(self):
    #   config = {
    #       'backbone': resnet.ResNet(50, bn_trainable=False),
    #       'backbone_endpoint_name': '5',
    #       'num_queries': 2,
    #       'hidden_size': 4,
    #       'num_classes': 10,
    #       'num_encoder_layers': 4,
    #       'num_decoder_layers': 5,
    #       'dropout_rate': 0.5,
    #   }
    #   detr_model = detr.DETR.from_config(config)
    #   retrieved_config = detr_model.get_config()

    #   self.assertEqual(config, retrieved_config)


if __name__ == '__main__':
    tf.test.main()
