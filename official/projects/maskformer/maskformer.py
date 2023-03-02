import tensorflow as tf

from official.vision.modeling.backbones import resnet
from official.projects.maskformer.modeling.decoder.transformer_decoder import MaskFormerTransformer
from official.projects.maskformer.modeling.decoder.pixel_decoder import Fpn
from official.projects.maskformer.modeling.layers.nn_block import MLPHead

# TODO(ibrahim): Add all parameters model parameters and remove hardcoding.
class MaskFormer(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._backbone = resnet.ResNet(50)
        self._transformer_decoder = MaskFormerTransformer(backbone_endpoint_name='5',
                                                          batch_size=1,
                                                          num_queries=100,
                                                          hidden_size=256,
                                                          num_classes=171,
                                                          num_encoder_layers=0,
                                                          num_decoder_layers=6,
                                                          dropout_rate=0.1)

        self._pixel_decoder = Fpn(fpn_feat_dims=256)
        self._MLP_head = MLPHead(
            num_classes=171, hidden_dim=256, mask_dim=256)

    def call(self, inputs):
        feature_maps = self._backbone(inputs)

        per_segment_embeddings = self._transformer_decoder(
            {"image": inputs, "features": feature_maps})
        print("\n\nper_segment_embeddings:", tf.shape(per_segment_embeddings))

        per_pixel_embeddings = self._pixel_decoder(feature_maps)
        print("\n\nper_pixel_embeddings:", tf.shape(per_pixel_embeddings))

        class_and_mask_probs = self._MLP_head(
            {'per_pixel_embeddings': per_pixel_embeddings, 'per_segment_embeddings': tf.stack(per_segment_embeddings)})

        return class_and_mask_probs
