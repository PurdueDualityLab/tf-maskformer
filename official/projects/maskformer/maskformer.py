import tensorflow as tf

from official.vision.modeling.backbones import resnet
from official.projects.maskformer.modeling.decoder.transformer_decoder import MaskFormerTransformer
from official.projects.maskformer.modeling.decoder.pixel_decoder import Fpn
from official.projects.maskformer.modeling.layers.nn_block import MLPHead

# TODO(ibrahim): Add all parameters model parameters and remove hardcoding.
class Maskformer(tf.keras.Model):
  """Maskformer"""
  def __init__(self,
              num_classes = 171,
              batch_size = 1,
               **kwargs):
    self.num_classes = num_classes
    self.batch_size = batch_size
    super(Maskformer, self).__init__(**kwargs)

  def build(self, image):
    self.batch_size = self.batch_size
    #backbone
    self.backbone = resnet.ResNet(50, bn_trainable=False)
    #decoders
    self.pixel_decoder = Fpn()
    self.transformer = MaskFormerTransformer(backbone_endpoint_name='5',
                                            batch_size=self.batch_size,
                                            num_queries=100,
                                            hidden_size=256,
                                            num_classes=self.num_classes,
                                            num_encoder_layers=0,
                                            num_decoder_layers=6,
                                            dropout_rate=0.1)
    #Heads
    self.pixel_predictor = tf.keras.layers.Conv2D(filters=self.num_classes,
                                                  strides=(1, 1),
                                                  kernel_size=(1, 1),
                                                  padding='valid')
    self.head = MLPHead(num_classes = self.num_classes, 
                                    hidden_dim = 256, 
                                    mask_dim = 256)
   
    super(Maskformer, self).build(image)
 
  def process_feature_maps(self, maps):
    new_dict = {}
    for k in maps.keys():
      new_dict[k[0]] = maps[k]
    return new_dict

  def call(self, image):
    backbone_feature_maps = self.backbone(image)
    
    mask_features = self.pixel_decoder(self.process_feature_maps(backbone_feature_maps))
    per_pixel_pred = self.pixel_predictor(mask_features)
    
    transformer_features = self.transformer({"features": backbone_feature_maps})
    
    seg_pred = self.head({"per_pixel_embeddings" : mask_features,
                          "per_segment_embeddings": transformer_features})
    return seg_pred
