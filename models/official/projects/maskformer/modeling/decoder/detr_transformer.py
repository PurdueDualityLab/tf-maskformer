import math
import tensorflow as tf
from official.projects.maskformer.modeling.transformer import transformer
from official.modeling import tf_utils

class DETRTransformer(tf.keras.layers.Layer):
  """Encoder and Decoder of DETR."""

  def __init__(self, num_encoder_layers=6, num_decoder_layers=6,
               dropout_rate=0.1, deep_supervision=False, **kwargs):
    super().__init__(**kwargs)
    self._dropout_rate = dropout_rate
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._deep_supervision = deep_supervision 

  def build(self, input_shape=None):
    if self._num_encoder_layers > 0:
      self._encoder = transformer.TransformerEncoder(
          attention_dropout_rate=self._dropout_rate,
          dropout_rate=self._dropout_rate,
          intermediate_dropout=self._dropout_rate,
          norm_first=False,
          num_layers=self._num_encoder_layers)
    else:
      self._encoder = None

    self._decoder = transformer.TransformerDecoder(
        attention_dropout_rate=self._dropout_rate,
        dropout_rate=self._dropout_rate,
        intermediate_dropout=self._dropout_rate,
        norm_first=False,
        num_layers=self._num_decoder_layers)
    super(DETRTransformer, self).build(input_shape)

  def get_config(self):
    return {
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "dropout_rate": self._dropout_rate,
    }

  def call(self, inputs):
    sources = inputs["inputs"]
    targets = inputs["targets"]
    pos_embed = inputs["pos_embed"]
    
    memory = sources
    target_shape = tf_utils.get_shape_list(targets)
    cross_attention_mask = None
    target_shape = tf.shape(targets)
    self_attention_mask = tf.ones([target_shape[0], 1, target_shape[1], target_shape[1]],dtype=tf.float32)
    
    decoded = self._decoder(
        tf.zeros_like(targets),
        memory,
        # TODO(b/199545430): self_attention_mask could be set to None when this
        # bug is resolved. Passing ones for now.
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
        return_all_decoder_outputs=self._deep_supervision,
        input_pos_embed=targets,
        memory_pos_embed=pos_embed,)
    
    return decoded
