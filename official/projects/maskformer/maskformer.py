import tensorflow as tf

from official.vision.modeling.backbones import resnet
from official.projects.maskformer.modeling.transformer.transformer import MaskFormerTransformer
from official.projects.maskformer.modeling.decoder.pixel_decoder import Fpn
from official.projects.maskformer.modeling.layers.nn_block import MaskFormerPredictor


class MaskFormer(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._backbone = resnet.ResNet(50)
        self._transformer_decoder = MaskFormerTransformer(backbone_endpoint_name='5',
                                                          batch_size=8,
                                                          num_queries=100,
                                                          hidden_size=256,
                                                          num_classes=10,
                                                          num_encoder_layers=0,
                                                          num_decoder_layers=6,
                                                          dropout_rate=0.1)

        self._pixel_decoder = Fpn(fpn_feat_dims=256)
        # self._transformer_predictor = MaskFormerPredictor(
        #     num_classes=171, hidden_dim=256, mask_dim=256)

    def call(self, inputs):
        feature_maps = self._backbone(inputs)

        
        print("\nbackbone output: ", type(feature_maps))
        for k, v in feature_maps.items():
            print(f"{k} - {v.shape}")

        per_segment_embeddings = self._transformer_decoder({"image": inputs, "features": feature_maps })
        print("\ntransformer decoder output: ")
        for k, v in per_segment_embeddings.items():
            print(f"{k} - {v.shape}")

        per_pixel_embeddings = self._pixel_decoder(feature_maps)
        print("\npixel decoder output: ")
        for k, v in per_pixel_embeddings.items():
            print(f"{k} - {v.shape}")

        # class_prob_prediction, mask_prob_prediction = self._transformer_predictor(
        #     {'per_pixel_embeddings': per_pixel_embeddings, 'per_segment_embeddings': per_segment_embeddings})

        # return class_prob_prediction, mask_prob_prediction

        return 1, 2
