from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
latest_ckp = tf.train.latest_checkpoint('gs://cam2-models/maskformer_vishal_exps/EXP19_v8')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')