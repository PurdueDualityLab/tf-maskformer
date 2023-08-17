import tensorflow as tf
from cloud_tpu_client import Client

import os
@tf.function
def matmul_fn(x, y):
  z = tf.matmul(x, y)
  return z

if __name__ == "__main__":
	c = Client("tf-train-1", zone="europe-west4-a", project="red-atlas-305317")
	c.configure_tpu_version("2.11.0", restart_type='ifNeeded')
	c.wait_for_healthy()
	cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu="tf-train-1", project="red-atlas-305317", zone="europe-west4-a")
	tf.config.experimental_connect_to_cluster(cluster_resolver)
	tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
	strategy = tf.distribute.TPUStrategy(cluster_resolver)
	print("All devices: ", tf.config.list_logical_devices('TPU'))
	a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

	z = strategy.run(matmul_fn, args=(a, b))
	print(z)