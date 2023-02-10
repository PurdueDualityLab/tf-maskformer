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
