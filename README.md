# Tensorflow分布式训练

## 架构介绍

与[ Pytorch 分布式训练](https://github.com/Jackxiini/Pytorch-distributed-learning)相同，Tensorflow 默认使用 RingAllReduce 的分布式方法来进行分布式训练。Tensorflow 也支持与[ MXNet 分布式训练](http://agroup.baidu.com/zhongce_saas/md/article/4091526)相同的 Parameter Server 方法的分布式训练，但是不做推荐。相对而言 RingAllReduce 的性能要优于 Parameter Server，Parameter Server 存在带宽瓶颈的问题。

TensorFlow 中 Parameter Server 架构目前仅支持异步训练模式， 而 RingAllReuce 架构则采用同步训练模式。

本文主要介绍如何使用 Tensorflow 的 `MultiWorkerMirroredStrategy` API 来进行多机多卡的分布式训练。需要注意，本文仅支持 Tensorflow 2.X 版本。

## 训练策略

本文只对 RingAllReduce方法的策略做介绍，所以只会涉及到 `MirroredStrategy` 和 `MultiWorkerMirroredStrategy`。其他的如 Parameter Server 的策略 `ParameterServerStrategy` 则不做阐述。

`MirroredStrategy` 是一种单机的同步的分布式训练策略。它支持在一台机器的多个 GPU 之间进行分布式训练，它会在每个 GPU 上创建一个模型副本，模型中的每个变量 (Variables) 都会进行镜像复制并放置到相应的 GPU 上，这些变量被称作镜像变量 (MirroredVariable)。

`MirroredStrategy` 策略通过 AllReduce 算法使得所有镜像变量在每个 GPU 之间保持同步更新， AllReduce 算法默认使用英伟达的 NcclAllReduce。

`MirroredStrategy` 策略会自动使用所有能被 TensorFlow 发现的 GPU 来做分布式训练，如果只想使用部分的 GPU 则可以通过 devices 参数来指定。

`MultiWorkerMirroredStrategy` 策略与 `MirroredStrategy` 策略很相似，可以理解为是 `MirroredStrategy` 策略的多机的同步的分布式训练版本，它也会在每一台机器上创建所有变量的副本。`MultiWorkerMirroredStrategy` 策略中运行的每一个节点称为一个 worker ，该 worker 节点上可以包含零或多个 GPU 。多个 worker 节点之间使用 AllReduce 算法来保持模型变量的同步更新， TensorFlow 里将这一操作称为 CollectiveOps。 CollectiveOps 会在 TensorFlow 模型运行时自动根据硬件，网络拓扑以及张量的大小来自动选择合适的 AllReduce 算法来进行网络通信以完成变量更新。

`MultiWorkerMirroredStrategy` 策略目前有两种可供选择的 CollectiveOps 。 一种为 `CollectiveCommunication.RING` ，它使用 gRPC 作为通信层实现了基于环的 AllReduce 操作。 另一种为 `CollectiveCommunication.NCCL`， 它使用了英伟达的 NCCL 库来实现 AllReduce 操作。在实际使用中，可以基于自己的运行环境选择合适的 CollectiveOps，或者使用 `CollectiveCommunication.AUTO` 交由 TensorFlow 运行时自行选择。

以下是创建代码的一个例子：
```
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
						tf.distribute.experimental.CollectiveCommunication.AUTO)
```

## 定义集群
TensorFlow 中定义集群配置信息的标准方式是使用 `TF_CONFIG` 环境变量来实现的，该环境变量定义了集群中所有节点的配置信息，包括所有 worker 节点的网络地址，当前 worker 节点的索引 (index) 以及当前 worker 节点的角色 (type)。

如果所有 worker 节点都不包含 GPU ，则该策略会退化为使用 CPU 在多个 worker 节点间进行分布式训练。如果集群中的 worker 节点数量只有一个则该策略会退化为 `MirroredStrategy` 策略。

以下是一个环境变量的设置例子：
```python
{
  "cluster": {
    "chief": ["host1:port"],
    "worker": ["host2:port", "host3:port"],
    "ps": ["host4:port"],
    "evaluator": ["host5:port"]
  },
  "task": {
    "type": "worker",
    "index": 0
  }
}
```
其中，`chief`和`worker`节点大致相同，区别在于，`chief`会做一些额外的工作，比如保存检查点模型，生成 Tensorboard 等。若不指定 `chief`节点，那么第一个`worker`将被默认为`chief`节点。

`worker`节点用于执行训练任务。

`ps`节点用于储存变量，类似于 MXNet 中的 Server 节点。此节点只能在使用 `ParameterServerStrategy` 的训练策略时使用。

`evaluator` 节点用来专门做交叉检验，一般也是在使用 `ParameterServerStrategy` 策略时才会使用。

所有节点上的`TF_CONFIG` 环境变量中的`cluster`需要一致，只需要调整 `task`的参数即可。`type`为节点扮演的角色，`index`需要从`0`起算。节点的`task`需要与节点在`cluster`中的信息一致。例如，节点 ( host2:port ) 的`type`必须是`worker`，`index`为`0`。

为了增加节点使用的灵活性，我们在 python 程序中常通过 `os.environ["TF_CONFIG"]` 来指定集群的信息以实现按需创建，以便在单个物理节点中启动多个集群节点。

## 训练流程
0. 确认各节点之间已开通互信，可以免密 ssh 通信
1. 不同节点使用 `TF_CONFIG`中信息启动该节点的训练任务。Tensorflow 按照环境变量的值使用相应的 `ip:port`来启动当前节点的 gRPC 服务，并监听其他节点的 gRPC 服务。
2. 所有节点 gRPC 服务准备就绪后，各个 worker 节点开始使用自己的数据集训练。
3. 每个 Batch 过后，Tensorflow 会根据分布式策略去更新 worker 的变量， 完成后进行下一个 Batch。
4. 完成所有训练后，训练结束，所有节点的 gRPC 服务关闭。

## 案例演示

注意：首先先要确认各节点间可以免密通信，建立免密方法可参考[此文档](https://github.com/Jackxiini/Trust-relationship-configuration-between-Linux-servers/blob/main/%E6%93%8D%E4%BD%9C%E6%AD%A5%E9%AA%A4.md)。

1.载入需要使用的包
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import os
```

2.紧接着我们需要定义`TF_CONFIG`和策略，这两部分需要在脚本最前面写，且需要把`TF_CONFIG`写在策略前边

```python
os.environ["TF_CONFIG"] = json.dumps({"cluster":
                                          {"worker": ["172.16.16.5:12345", "172.16.16.6:12345"]},
                                      "task":
                                          {"type": "worker", "index": 1}
                                      })

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.AUTO)
```
3.载入数据集，对数据集做预处理，此处我们使用 `tensorflow_datasets`包下载数据集，数据结构为`tf.data.Dataset`
```python
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
```
4.读取到数据集后，我们必须做以下操作来分发数据。数据必须是`tf.data.Dataset`格式才能做此操作

```python
train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
dist_dataset = train_datasets.with_options(options)
```
若格式不是`Dataset`，我们可以用以下方式转换：
```python
train_x, train_y = np.array(...), np.array(...)
val_x, val_y = np.array(...), np.array(...)

# Wrap data in Dataset objects.
train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))
```

5.创建神经网络结构，优化器，Loss等
```python
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
```
6.使用`strategy.scope`来分发模型并用`fit`函数训练
```python
with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(x=dist_dataset, epochs=10, steps_per_epoch=20)
```
7.最后在各节点控制台运行python脚本即可启动分布式

若我们没有在脚本中定义`TF_CONFIG`，则我们也可以通过以下方式启动分布式
```
TF_CONFIG='{"cluster": {"worker": ["10.157.106.90:9000", "10.157.106.151:9000"]}, "task": {"index": 0, "type": "worker"}}' python tf_dist_example.py
TF_CONFIG='{"cluster": {"worker": ["10.157.106.90:9000", "10.157.106.151:9000"]}, "task": {"index": 1, "type": "worker"}}' python tf_dist_example.py
```
