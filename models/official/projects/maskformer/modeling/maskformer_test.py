from official.projects.maskformer.modeling.maskformer import MaskFormer
from absl.testing import parameterized
import tensorflow as tf
from official.vision.modeling.backbones.resnet import ResNet

class MaskFormerTest(tf.test.TestCase, parameterized.TestCase):
    # TODO(ibrahim): Add more testcases.
    @parameterized.named_parameters(('test1', 256, 100, 256, "5", 6, 0, 6, 133, 1))
    def test_pass_through(self,
                        fpn_feat_dims,
                        num_queries,
                        hidden_size,
                        backbone_endpoint_name,
                        fpn_encoder_layers,
                        detr_encoder_layers,
                        num_decoder_layers,
                        num_classes,
                        batch_size):    
        input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            [640, 640, 3])    
        # maskformer = MaskFormer(input_specs= input_specs, hidden_size=hidden_size,
        #                          backbone_endpoint_name=backbone_endpoint_name,
        #                          fpn_encoder_layers=fpn_encoder_layers,
        #                          detr_encoder_layers=detr_encoder_layers,
        #                          num_decoder_layers=num_decoder_layers,
        #                          num_classes=num_classes,
        #                          batch_size=batch_size)
        backbone = ResNet(input_specs=input_specs, model_id=50)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        maskformer = MaskFormer(backbone=backbone, input_specs= input_specs,
							num_queries=100,
							hidden_size=hidden_size,
							backbone_endpoint_name=backbone_endpoint_name,
							fpn_encoder_layers=fpn_encoder_layers,
							detr_encoder_layers=detr_encoder_layers,
							num_decoder_layers=num_decoder_layers,
							num_classes=num_classes,
							bfloat16=False, 
							which_pixel_decoder='transformer_fpn',)
        
        with tf.summary.create_file_writer("./logs").as_default():
            # Visualize the computation graph
            tf.summary.trace_on(graph=True, profiler=True)
            
            # Define a dummy input for visualization (shape should match your model's input shape)
            dummy_input = tf.ones((2, 640, 640, 3))
            
            # Run the model to trace the computation graph
            maskformer(dummy_input)
            
            # Save the traced computation graph to the log directory
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir="./logs")        
        
        # input_image = tf.ones((2, 640, 640, 3))
        
        # expected_class_probs_shape = [1, 100, 172]
        # expected_mask_probs_shape = [1, 160, 160, 100]
        # expected_class_probs_shape = [1, 100, 134] # B, dim of logits, number of classes
        # expected_mask_probs_shape = [1, 160, 160, 100] # B,H,W,C

        # output = maskformer(input_image)
        # print(output.keys())
        # print(maskformer.summary())
        
        # self.assertAllEqual(
        #     output["class_prob_predictions"].shape.as_list(), expected_class_probs_shape)
        # self.assertAllEqual(
        #     output["mask_prob_predictions"].shape.as_list(), expected_mask_probs_shape)


if __name__ == '__main__':
    tf.test.main()

