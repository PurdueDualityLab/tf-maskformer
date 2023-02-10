import math
import tensorflow as tf

from official.projects.detr.modeling.detr import DETRTransformer, position_embedding_sine

class MaskFormerTransformer(tf.keras.layers.Layer):
    def __init__(self,
               backbone_endpoint_name,
               batch_size,
               num_queries,
               hidden_size,
               num_classes,
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

        self._num_classes = num_classes

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

        self._class_embed = tf.keras.layers.Dense(
            self._num_classes,
            kernel_initializer=tf.keras.initializers.RandomUniform(-sqrt_k, sqrt_k),
            name="detr/cls_dense")
        
        self._input_proj = tf.keras.layers.Conv2D(
            self._hidden_size, 1, name="detr/conv2d")

    def _generate_image_mask(self, inputs: tf.Tensor,
                            target_shape: tf.Tensor) -> tf.Tensor:
        """Generates image mask from input image."""
        mask = tf.expand_dims(
            tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=-1), 0), inputs.dtype),
            axis=-1)
        mask = tf.image.resize(
            mask, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return mask
    
    def call(self, inputs):
        input_image = inputs['image']
        features = inputs['features'][self._backbone_endpoint_name]

        mask = self._generate_image_mask(input_image, tf.shape(features)[1: 3])

        pos_embed = position_embedding_sine(
            mask[:, :, :, 0], num_pos_features=self._hidden_size)
        pos_embed = tf.reshape(pos_embed, [self._batch_size, -1, self._hidden_size])

        features = tf.reshape(
            self._input_proj(features), [self._batch_size, -1, self._hidden_size])
        mask = tf.reshape(mask, [self._batch_size, -1])

        decoded_list = self._transformer({
            "inputs":
                features,
            "targets":
                tf.tile(
                    tf.expand_dims(self._query_embeddings, axis=0),
                    (self._batch_size, 1, 1)),
            "pos_embed": pos_embed,
            "mask": mask,
        })

        return decoded_list
    
    def get_config(self):
        return {
            "backbone_endpoint_name": self._backbone_endpoint_name,
            "num_queries": self._num_queries,
            "hidden_size": self._hidden_size,
            "num_classes": self._num_classes,
            "num_encoder_layers": self._num_encoder_layers,
            "num_decoder_layers": self._num_decoder_layers,
            "dropout_rate": self._dropout_rate,
        }