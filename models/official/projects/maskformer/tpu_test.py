import tensorflow as tf

if __name__ == "__main__":
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu="tf-debug-2")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)