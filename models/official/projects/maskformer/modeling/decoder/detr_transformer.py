import math
import tensorflow as tf

from official.projects.detr.modeling.detr import position_embedding_sine
from official.projects.detr.modeling import transformer
from official.modeling import tf_utils

class DETRTransformer(tf.keras.layers.Layer):
  """Encoder and Decoder of DETR."""

  def __init__(self, num_encoder_layers=6, num_decoder_layers=6,
               dropout_rate=0.1, **kwargs):
    super().__init__(**kwargs)
    self._dropout_rate = dropout_rate
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers

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
    super().build(input_shape)

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
    mask = inputs["mask"]
    input_shape = tf_utils.get_shape_list(sources)
    if mask is not None:
      source_attention_mask = tf.tile(
          tf.expand_dims(mask, axis=1), [1, input_shape[1], 1])
    else:
      source_attention_mask = None
    if self._encoder is not None:
      memory = self._encoder(
          sources, attention_mask=source_attention_mask, pos_embed=pos_embed)
    else:
      memory = sources

    target_shape = tf_utils.get_shape_list(targets)
    target_shape = tf.shape(targets)
    
    if mask is not None:
      cross_attention_mask = tf.tile(
          tf.expand_dims(mask, axis=1), [1, target_shape[1], 1])
      self_attention_mask=tf.ones(
            (target_shape[0], target_shape[1], target_shape[1]))
    else:
      cross_attention_mask = None
      self_attention_mask = None
    
    # FIXME : The decoder uses float32 so cast all inputs to float32
    # memory = tf.cast(memory, tf.float32)
    # pos_embed = tf.cast(pos_embed, tf.float32)
    # targets = tf.cast(targets, tf.float32)
    decoded = self._decoder(
        tf.zeros_like(targets),
       memory,
        # TODO(b/199545430): self_attention_mask could be set to None when this
        # bug is resolved. Passing ones for now.
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
        return_all_decoder_outputs=False,
        input_pos_embed=targets,
        memory_pos_embed=pos_embed,)
    
    # FIXME : Return decode as bfloat16
    return decoded
