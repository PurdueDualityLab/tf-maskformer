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

class MaskFormerPredictor(tf.keras.layers.Layer):
    def __init__(self,
                 num_classes: int,
                 hidden_dim: int,
                #  dec_supervision: bool,
                 mask_dim: int):
        super().__init__()
        self.mlp = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.linear_classifier = tf.keras.layers.Dense(input_shape=hidden_dim, out_dim=num_classes + 1, activation=tf.keras.activations.softmax)
        # No Softmax used in their code? Need to figure out!!
        # self.linear_classifier = tf.keras.layers.Dense(input_shape=hidden_dim, out_dim=num_classes + 1, activation=None)
        
        # self.dec_supervision = dec_supervision
            
def call(self, inputs):
        per_pixel_embeddings = inputs['per_pixel_embeddings']
        per_segment_embeddings = inputs['per_segment_embeddings']

        class_prob_prediction = self.linear_classifier(per_segment_embeddings)
        mask_embedding = self.mlp(per_segment_embeddings[-1])
        mask_prob_prediction = tf.einsum("bqc,bchw->bqhw", mask_embedding, per_pixel_embeddings)

        return class_prob_prediction, mask_prob_prediction
        

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
        self.layers = []
        for i, dim in enumerate(layer_dims):
            if(i < num_layers - 1):
                self.layers.append(tf.keras.layers.Dense(dim[1], activation=tf.nn.relu))
            else:
                #Final Layer
                self.layers.append(tf.keras.layers.Dense(dim[1], activation=None))

    def call(self, x): 
        for layer in self.layers:
            x = layer(x)
        return x