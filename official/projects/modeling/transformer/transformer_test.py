from absl.testing import parameterized
import tensorflow as tf

# from official.projects.modeling.transformer.transformer import MaskFormerTransformer # import error ;-;)/
from transformer import MaskFormerTransformer
from official.vision.modeling.backbones import resnet

class MaskFormerTransformerTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('test1', '5', 8, (8, 640, 640, 3), 10, 128, 10,))
    def test_pass_through(self,
                        backbone_endpoint_name,
                        batch_size,
                        input_image_shape,
                        num_queries,
                        hidden_size,
                        num_classes):    

        backbone = resnet.ResNet(50, bn_trainable=False)
        transformer = MaskFormerTransformer(backbone_endpoint_name=backbone_endpoint_name,
                                            batch_size=batch_size,
                                            num_queries=num_queries,
                                            hidden_size=hidden_size,
                                            num_classes=num_classes,
                                            num_encoder_layers=6,
                                            num_decoder_layers=6,
                                            dropout_rate=0.1)

        # input_image = tf.keras.Input(shape=input_image_shape)
        input_image = tf.ones((1, 640, 640, 3))
        backbone_feature_maps = backbone(input_image)

        # for i, x in backbone_feature_maps.items():
        #     print(i, ":", x.shape, "\n")
        output = transformer({"image": input_image, "features": backbone_feature_maps })

        print("output len: ", len(output))
        for i, x in enumerate(output):
            print("output[", i, "].shape: ", x.shape)

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
