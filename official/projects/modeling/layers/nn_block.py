import tensorflow as tf

'''
enc_layers: int,
dec_layers: int,
nheads: int,
dropout: float,
dim_feedforward: int,
pre_norm: bool,
'''

class MaskFormerPredictor(tf.keras.layers.Layer):
    def __init__(self,
                 num_classes: int,
                 hidden_dim: int,
                 deep_supervision: bool,
                 mask_dim: int,
                 enforce_input_project: bool):
        super().__init__()
        self.mlp = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.linear_classifier = tf.keras.layers.Dense(input_shape=hidden_dim, out_dim=num_classes + 1, activation=None)
        # No Softmax used in their code? Need to figure out!!
            

    # def build(self, input_shape):
    #     self._linear_classifier = None
        
    # dot product
    def get_mask_predictions(self):
        # Have to figure out where they use per_pixel_embeddings
        return None

    def call(self, inputs):
        per_pixel_embeddings = inputs['per_pixel_embeddings']
        per_segment_embeddings = inputs['per_segment_embeddings']

        class_prob_prediction = self.linear_classifier(per_segment_embeddings)

        mask_embedding = self.mlp(per_segment_embeddings)
        mask_prob_prediction = self.get_mask_predictions(per_pixel_embeddings, mask_embedding)

        return {'class_prob_predictions': class_prob_prediction,'mask_prob_predictions': mask_prob_prediction}
        

class MLP(tf.keras.layers.Layer):
    def __init__(self,
                input_dim: int, 
                hidden_dim: int, 
                output_dim: int, 
                num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        layer_dims = [(input_dim, hidden_dim)]
        for _ in range(num_layers - 2):
            layer_dims.append((hidden_dim, hidden_dim))
        layer_dims.append((hidden_dim, output_dim))
        #layer_dims contains the input and output dimension of each layer in the form (input dimension, output dimension)
        layers = []
        for i, dim in enumerate(layer_dims):
            if(i < num_layers - 1):
                layers.append(tf.keras.layers.Dense(input_shape=(dim[0],), out_dim=dim[1], activation=tf.nn.relu))
            else:
                #Final Layer
                layers.append(tf.keras.layers.Dense(input_shape=(dim[0],), out_dim=dim[1], activation=None))

    def call(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x