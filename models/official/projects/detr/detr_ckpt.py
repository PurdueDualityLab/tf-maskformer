import tensorflow as tf

# Load the checkpoint
checkpoint_dir = 'gs://cam2-models/detr_exp2/'
ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint_reader = tf.train.load_checkpoint(ckpt_path)

# Print out all the layers (variables)
all_variables = checkpoint_reader.get_variable_to_shape_map()
for var_name in all_variables:
    print(var_name, all_variables[var_name])
