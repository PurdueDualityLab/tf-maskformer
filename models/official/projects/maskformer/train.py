"""TensorFlow Model Garden Vision training driver."""

from absl import app
from absl import flags
import gin

from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance
from official.projects.maskformer.configs import maskformer
from official.projects.maskformer.tasks import panoptic_maskformer
import tensorflow as tf
from cloud_tpu_client import Client
import os

FLAGS = flags.FLAGS
def main(_):
	# This works only for TPU v3 version
	c = Client(os.environ['TPU_NAME'], zone=os.environ['TPU_ZONE'], project=os.environ['TPU_PROJECT'])
	c.configure_tpu_version(os.environ["TPU_SOFTWARE"], restart_type='ifNeeded')
	c.wait_for_healthy()
	gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
	params = train_utils.parse_configuration(FLAGS)
	model_dir = FLAGS.model_dir
	
	if 'train' in FLAGS.mode:
		# Pure eval modes do not output yaml files. Otherwise continuous eval job
		# may race against the train job for writing the same file.
		train_utils.serialize_config(params, model_dir)

	# Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
	# can have significant impact on model speeds by utilizing float16 in case of
	# GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
	# dtype is float16

	# resolver = tf.distribute.cluster_resolver.TPUClusterResolver()

	# Uncomment to test on TPU
	if params.runtime.mixed_precision_dtype:
		performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
	distribution_strategy = distribute_utils.get_distribution_strategy(
			distribution_strategy="tpu",
        		all_reduce_alg=params.runtime.all_reduce_alg,
			num_gpus=params.runtime.num_gpus,
			tpu_address=params.runtime.tpu)
        
	# tf.profiler.experimental.server.start(6000)
        #cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = os.environ["TPU_NAME"], zone = os.environ['TPU_ZONE'], project = os.environ['TPU_PROJECT'])
	#tf.config.experimental_connect_to_cluster(cluster_resolver)
	#tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
      #  distribution_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

	# 		num_gpus=1)
	
	# Below code is independent of compute platform
	with distribution_strategy.scope():
		task = task_factory.get_task(params.task, logging_dir=model_dir)
		
	tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
	train_lib.run_experiment(
			distribution_strategy=distribution_strategy,
			task=task,
			mode=FLAGS.mode,
			params=params,
			model_dir=model_dir)

	train_utils.save_gin_config(FLAGS.mode, model_dir)

if __name__ == '__main__':
	tfm_flags.define_flags()
	flags.mark_flags_as_required(['experiment', 'mode', 'model_dir'])
	app.run(main)
