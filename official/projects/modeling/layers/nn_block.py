import tensorflow as tf

class MaskFormerPredictor(tf.keras.layers.Layer):
    def __init__(self,
                 num_classes,
                 hidden_dim,
                 nheads,
                 dopout,
                 dim_feedforward,
                 enc_layers,
                 dec_layers,
                 pre_norm,
                 deep_supervision,
                 mask_dim,
                 enforce_input_project):
        super().__init__()

        def build(self, input_shape):
            self._mlp = None
            self._linear_classifier = None
        
        # dot product thing.
        def _get_mask_predictions(self):
            return None

        def call(self, inputs):
            per_pixel_embeddings = inputs['per_pixel_embeddings']
            per_segment_embeddings = inputs['per_segment_embeddings']

            class_prob_prediction = self._linear_classifier(per_segment_embeddings)
            mask_embedding = self._mlp(per_segment_embeddings)

            mask_prob_prediction = self._get_mask_predictions(per_segment_embeddings, mask_embedding)

            return {'class_prob_predictions': class_prob_prediction,'mask_prob_predictions': mask_prob_prediction}
        

class MLP(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        layer_dims = [(input_dim, hidden_dim)]
        for cnt in range(num_layers - 2):
            layer_dims.append((hidden_dim, hidden_dim))
        layer_dims.append((hidden_dim, output_dim))
        #layer_dims contains the input and output dimension of each layer in the form (input dimension, output dimension)
        layers = []
        for i, dim in enumerate(layer_dims):
            if(i < num_layers - 1):
                layers.append(tf.keras.layers.Dense(out_dim=dim[1], activation=tf.nn.relu))
            else:
                #Final Layer
                layers.append(tf.keras.layers.Dense(out_dim=dim[1]))

    def __call__(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x