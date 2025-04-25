---
title: TensorFlow 2.0 学习笔记（2）：神经网络模型构建
date: 2018-09-06 10:03:00
categories:
- 人工智能/tensorflow
tags: 人工智能
---
本文介绍TensorFlow 2.0中构建神经网络模型的方法，主要围绕Keras API展开。

## Keras：TensorFlow 2.0的核心高层API

TensorFlow 2.0将Keras作为其官方高级API，使模型构建和训练变得简单高效。Keras提供了三种构建模型的方式：

1. **Sequential API**：最简单的模型构建方式，适用于层按顺序堆叠的简单模型
2. **Functional API**：更灵活的模型构建方式，支持多输入多输出和复杂的层连接
3. **Model子类化**：完全自定义的模型构建方式，适合复杂的研究场景

## 模型参数初始化

在神经网络中，权重初始化是很重要的一步。TensorFlow 2.0提供了多种初始化器：

```python
# 常用的初始化器
from tensorflow.keras import initializers

# 正态分布随机初始化
initializer = initializers.RandomNormal(mean=0.0, stddev=0.05)

# 截断正态分布随机初始化（限制在平均值周围的标准差范围内）
initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05)

# 均匀分布随机初始化
initializer = initializers.RandomUniform(minval=-0.05, maxval=0.05)

# 常量初始化
initializer = initializers.Constant(value=0.2)

# 全零初始化
initializer = initializers.Zeros()

# 全一初始化
initializer = initializers.Ones()

# He初始化（适用于ReLU激活函数）
initializer = initializers.HeNormal()

# Xavier/Glorot初始化（适用于tanh激活函数）
initializer = initializers.GlorotNormal()
```

在创建层时使用初始化器：

```python
import tensorflow as tf

# 在Dense层中使用初始化器
layer = tf.keras.layers.Dense(
    units=10,
    kernel_initializer=tf.keras.initializers.HeNormal(),
    bias_initializer=tf.keras.initializers.Zeros()
)
```

## Sequential API：构建简单模型

使用Sequential API构建一个简单的多层感知机：

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 构建模型
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 查看模型结构
model.summary()
```

## Functional API：构建复杂模型

使用Functional API构建多输入模型：

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 定义输入
image_input = tf.keras.Input(shape=(28, 28, 1), name='image_input')
metadata_input = tf.keras.Input(shape=(10,), name='metadata')

# 图像处理分支
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

# 元数据处理分支
y = layers.Dense(32, activation='relu')(metadata_input)

# 合并两个分支
combined = layers.concatenate([x, y])

# 输出层
output = layers.Dense(10, activation='softmax')(combined)

# 创建包含两个输入的模型
model = models.Model(
    inputs=[image_input, metadata_input],
    outputs=output
)

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 查看模型结构
model.summary()
```

## 模型子类化：完全自定义模型

通过继承`tf.keras.Model`类创建自定义模型：

```python
import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # 定义层
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs, training=False):
        # 定义前向传播
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
    
# 创建模型实例
model = CustomModel()

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 需要先构建模型结构才能查看summary
model.build(input_shape=(None, 28, 28, 1))
model.summary()
```

## 保存和加载模型

TensorFlow 2.0提供了简便的模型保存和加载机制：

```python
# 保存整个模型（包括权重、配置和优化器状态）
model.save('my_model.h5')

# 仅保存权重
model.save_weights('model_weights.h5')

# 加载整个模型
loaded_model = tf.keras.models.load_model('my_model.h5')

# 加载权重（需要先构建相同结构的模型）
model.load_weights('model_weights.h5')
```

## SavedModel 格式

对于生产环境，建议使用SavedModel格式保存模型：

```python
# 保存为SavedModel格式
model.save('saved_model_dir', save_format='tf')

# 加载SavedModel格式的模型
loaded_model = tf.keras.models.load_model('saved_model_dir')
```

SavedModel格式的优势在于可以跨平台部署，支持TensorFlow Serving、TensorFlow Lite等生产环境。

在下一篇笔记中，我们将介绍TensorFlow 2.0中的训练与优化技术。

> 本文是TensorFlow 2.0学习笔记系列的第二篇，专注于神经网络模型构建方法。
