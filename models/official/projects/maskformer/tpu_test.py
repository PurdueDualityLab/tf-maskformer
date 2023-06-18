import tensorflow as tf

if __name__ == "__main__":
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu="tf-train-1", project="red-atlas-305317", zone="us-central1-a")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)