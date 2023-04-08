import tensorflow as tf

from official.vision.modeling.backbones import resnet
from official.projects.maskformer.modeling.decoder.transformer_decoder import MaskFormerTransformer
from official.projects.maskformer.modeling.layers.nn_block import MLPHead
from official.projects.maskformer.modeling.decoder.transformer_pixel_decoder import TransformerFPN

# TODO(ibrahim): Add all parameters model parameters and remove hardcoding.
class MaskFormer(tf.keras.Model):
  """Maskformer"""
  def __init__(self,
               fpn_feat_dims=256,
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation='relu',
               use_bias=False,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               num_queries=100,
               hidden_size=256,
               num_encoder_layers=0,
               num_decoder_layers=6,
               dropout_rate=0.1,
               backbone_endpoint_name='5',
               num_classes=171,
               batch_size=1,
               **kwargs):
    self._batch_size = batch_size
    self._num_classes = num_classes

    # Pixel Deocder paramters.
    self._fpn_feat_dims = fpn_feat_dims
    self._data_format = data_format
    self._dilation_rate = dilation_rate
    self._groups = groups
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._activity_regularizer = activity_regularizer
    self._kernel_constraint = kernel_constraint
    self._bias_constraint = bias_constraint

    # DETRTransformer parameters.
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._num_queries = num_queries
    self._hidden_size = hidden_size
    self._dropout_rate = dropout_rate
    self._backbone_endpoint = backbone_endpoint_name
    

    super(MaskFormer, self).__init__(**kwargs)

  def build(self, image):
    #backbone
    self.backbone = resnet.ResNet(50, bn_trainable=False)
    #decoders
    self.pixel_decoder = TransformerFPN(batch_size = self._batch_size,
                            fpn_feat_dims=self._fpn_feat_dims,
                            data_format=self._data_format,
                            dilation_rate=self._dilation_rate,
                            groups=self._groups,
                            activation=self._activation,
                            use_bias=self._use_bias,
                            kernel_initializer=self._kernel_initializer,
                            bias_initializer=self._bias_initializer,
                            kernel_regularizer=self._kernel_regularizer,
                            bias_regularizer=self._bias_regularizer,
                            activity_regularizer=self._activity_regularizer,
                            kernel_constraint=self._kernel_constraint,
                            bias_constraint=self._bias_constraint)
    self.transformer = MaskFormerTransformer(backbone_endpoint_name=self._backbone_endpoint,
                                            batch_size=self._batch_size,
                                            num_queries=self._num_queries,
                                            hidden_size=self._hidden_size,
                                            num_encoder_layers=self._num_encoder_layers,
                                            num_decoder_layers=self._num_decoder_layers,
                                            dropout_rate=self._dropout_rate)
    #Heads
    # self.pixel_predictor = tf.keras.layers.Conv2D(filters=self._num_classes,
    #                                               strides=(1, 1),
    #                                               kernel_size=(1, 1),
    #                                               padding='valid')
    self.head = MLPHead(num_classes=self._num_classes, 
                        hidden_dim=self._hidden_size, 
                        mask_dim=self._fpn_feat_dims)
   
    super(MaskFormer, self).build(image)
 
  def process_feature_maps(self, maps):
    new_dict = {}
    for k in maps.keys():
      new_dict[k[0]] = maps[k]
    return new_dict

  def call(self, image):
    # image = tf.reshape(image, [1, 800, 1135, 3])
    # image = tf.ones((1, 640, 640, 3))
    backbone_feature_maps = self.backbone(image)
    print("[INFO] Backbone Features [res2]:", backbone_feature_maps["2"].shape)
    print("[INFO] Backbone Features [res3]:", backbone_feature_maps["3"].shape)
    print("[INFO] Backbone Features [res4]:", backbone_feature_maps["4"].shape)
    print("[INFO] Backbone Features [res5]:", backbone_feature_maps["5"].shape)
    mask_features, transformer_enc_feat = self.pixel_decoder(self.process_feature_maps(backbone_feature_maps))
    print("[INFO] Mask Features :", mask_features.shape)
    print("[INFO] Transformer ENC Features :", transformer_enc_feat.shape)

    transformer_features = self.transformer({"features": transformer_enc_feat})
    print("[INFO] Transformer Features :", transformer_features.shape)
    seg_pred = self.head({"per_pixel_embeddings" : mask_features,
                          "per_segment_embeddings": transformer_features})
    print("[INFO] Seg Pred Class Prob:", seg_pred["class_prob_predictions"].shape)
    print("[INFO] Seg Pred Mask Prob:", seg_pred['mask_prob_predictions'].shape)
    return seg_pred
