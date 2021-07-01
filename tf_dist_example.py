import tensorflow as tf
import tensorflow_datasets as tfds
import json
import os

os.environ["TF_CONFIG"] = json.dumps({"cluster":
                                          {"worker": ["172.16.16.5:12345", "172.16.16.6:12345"]},
                                      "task":
                                          {"type": "worker", "index": 1}
                                      })

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.AUTO)
#strategy = tf.distribute.MirroredStrategy()

tfds.disable_progress_bar()
BUFFER_SIZE = 10000
NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS

def make_datasets_unbatched():
  # 将 MNIST 数据从 (0, 255] 缩放到 (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(with_info=True,
                             name='mnist',
                             as_supervised=True)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
#dist_dataset = strategy.experimental_distribute_dataset(train_datasets)
dist_dataset = train_datasets.with_options(options)

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model


with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(x=dist_dataset, epochs=10, steps_per_epoch=20)

