import tensorflow as tf

'''
Transformer Parameters:

enc_layers: int,
dec_layers: int,
nheads: int,
dropout: float,
dim_feedforward: int,
pre_norm: bool,
enforce_input_project: bool
'''


class MLPHead(tf.keras.layers.Layer):
    def __init__(self,
                 num_classes: int,
                 hidden_dim: int,
                 deep_supervision: bool,
                 mask_dim: int):
        super().__init__()

        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._mask_dim = mask_dim
        self._deep_supervision = deep_supervision

    def build(self, input_shape):
        self._mlp = MLP(self._hidden_dim, self._hidden_dim, self._mask_dim, 3)
        self._linear_classifier = tf.keras.layers.Dense(self._num_classes + 1)
        # No Softmax used in their code? Need to figure out!!
        # self.linear_classifier = tf.keras.layers.Dense(input_shape=hidden_dim, out_dim=num_classes + 1, activation=None)

        # self.dec_supervision = dec_supervision

    def call(self, inputs):
        per_pixel_embeddings = inputs['per_pixel_embeddings'] # mask feat
        per_segment_embeddings = inputs['per_segment_embeddings'] #transformer feat
       
        class_prob_prediction = self._linear_classifier(per_segment_embeddings)
        mask_embedding = self._mlp(per_segment_embeddings)

        if self._deep_supervision:
            # mask embedding: [l, batch_size, num_queries, hidden_dim]
            mask_prob_prediction = tf.einsum("lbqc,bhwc->lbhwq", mask_embedding, per_pixel_embeddings)
        else:
            # mask embedding: [batch_size, num_queries, hidden_dim]
            mask_prob_prediction = tf.einsum("bqc,bhwc->bhwq", mask_embedding, per_pixel_embeddings)
            
        return {'class_prob_predictions': class_prob_prediction,'mask_prob_predictions': mask_prob_prediction}


class MLP(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._num_layers = num_layers

    def build(self, input_shape):
        layer_dims = [(self._input_dim, self._hidden_dim)]
        for _ in range(self._num_layers - 2):
            layer_dims.append((self._hidden_dim, self._hidden_dim))
        layer_dims.append((self._hidden_dim, self._output_dim))

        self._layers = []
        for i, dim in enumerate(layer_dims):
            if(i < self._num_layers - 1):
                self._layers.append(tf.keras.layers.Dense(
                    dim[1], activation=tf.nn.relu))
            else:
                # Final Layer
                self._layers.append(
                    tf.keras.layers.Dense(dim[1], activation=None))

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
