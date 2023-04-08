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
      
    decoded = self._decoder(
        tf.zeros_like(targets),
        memory,
        # TODO(b/199545430): self_attention_mask could be set to None when this
        # bug is resolved. Passing ones for now.
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
        return_all_decoder_outputs=False,
        input_pos_embed=targets,
        memory_pos_embed=pos_embed)
    return decoded
  
class MaskFormerTransformer(tf.keras.layers.Layer):
    def __init__(self,
               backbone_endpoint_name,
               batch_size,
               num_queries,
               hidden_size,
               num_encoder_layers=6,
               num_decoder_layers=6,
               dropout_rate=0.1,
               **kwargs):
        super().__init__(**kwargs)
        
        self._backbone_endpoint_name = backbone_endpoint_name
        
        # Embeddings parameters.
        self._batch_size = batch_size
        self._num_queries = num_queries
        self._hidden_size = hidden_size
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be a multiple of 2.")


        # DETRTransformer parameters.
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dropout_rate = dropout_rate
  


    def build(self, input_shape):
        self._transformer = DETRTransformer(num_encoder_layers=self._num_encoder_layers,
                                            num_decoder_layers=self._num_decoder_layers,
                                            dropout_rate=self._dropout_rate)

        self._query_embeddings = self.add_weight(
            "detr/query_embeddings",
            shape=[self._num_queries, self._hidden_size],
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
            dtype=tf.float32)
        
        sqrt_k = math.sqrt(1.0 / self._hidden_size)
        
        self._input_proj = tf.keras.layers.Conv2D(
            self._hidden_size, 1, name="detr/conv2d")

    def _generate_image_mask(self, features: tf.Tensor) -> tf.Tensor:
        """Generates image mask from input image."""
        mask = tf.zeros([features.shape[0],features.shape[1],features.shape[2]])
        mask = tf.cast(mask, dtype = bool)
        return mask
    
    def call(self, inputs):
        features = inputs['features'][self._backbone_endpoint_name]

        mask = self._generate_image_mask(features)

        pos_embed = position_embedding_sine(
            mask, num_pos_features=self._hidden_size)
        pos_embed = tf.reshape(pos_embed, [self._batch_size, -1, self._hidden_size])

        features = tf.reshape(
            self._input_proj(features), [self._batch_size, -1, self._hidden_size])

        decoded_list = self._transformer({
            "inputs":
                features,
            "targets":
                tf.tile(
                    tf.expand_dims(self._query_embeddings, axis=0),
                    (self._batch_size, 1, 1)),
            "pos_embed": pos_embed,
            "mask": None,
        })

        return decoded_list
    
    def get_config(self):
        return {
            "backbone_endpoint_name": self._backbone_endpoint_name,
            "num_queries": self._num_queries,
            "hidden_size": self._hidden_size,
            "num_encoder_layers": self._num_encoder_layers,
            "num_decoder_layers": self._num_decoder_layers,
            "dropout_rate": self._dropout_rate,
        }
