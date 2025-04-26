---
title: TensorFlow 2.0 学习笔记（1）：基础概念
date: 2025-04-01 8:55:00
categories:
- 人工智能/tensorflow
tags: 人工智能,框架
---
本文整理了TensorFlow 2.0的基础知识，帮助大家快速入门这一强大的深度学习框架。

## TensorFlow 2.0 简介

TensorFlow 2.0是谷歌开发的开源机器学习框架，相比TensorFlow 1.x，其使用更加简洁直观，主要特点包括：

1. 以Keras为核心的高层API
2. 即时执行模式（Eager Execution）
3. 更简洁的模型构建方式
4. 改进的分布式训练支持

## 基本概念

### 张量（Tensor）

张量是TensorFlow中的核心数据结构，表示多维数组。

- **0阶张量**：标量，表示单个数值
  ```python
  scalar = tf.constant(123)
  ```

- **1阶张量**：向量，表示一维数组
  ```python
  vector = tf.constant([1, 2, 3])
  ```

- **2阶张量**：矩阵，表示二维数组
  ```python
  matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  ```

- **高阶张量**：表示三维或更高维度的数组
  ```python
  tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
  ```

判断张量的阶数，就是看其维度的数量。TensorFlow 2.0中，可以通过`tf.rank()`函数或`.shape`属性查看张量的维度信息。

### 数据类型

TensorFlow支持多种数据类型，常见的有：
- `tf.float32`：32位浮点数
- `tf.float64`：64位浮点数
- `tf.int32`：32位整数
- `tf.int64`：64位整数
- `tf.bool`：布尔类型
- `tf.string`：字符串类型

### 即时执行模式（Eager Execution）

TensorFlow 2.0默认启用即时执行模式，可以立即计算操作结果，无需构建计算图和创建会话。这使得调试更加容易，代码更加直观。

```python
import tensorflow as tf

# 直接创建和操作张量
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
result = a + b

# 直接打印结果，无需会话
print(result)  # tf.Tensor([4. 6.], shape=(2,), dtype=float32)
```

### 自动微分

TensorFlow 2.0内置自动微分系统，通过`tf.GradientTape`记录操作，用于自动计算梯度。

```python
import tensorflow as tf

# 创建变量
x = tf.Variable(3.0)

# 使用GradientTape记录操作以计算梯度
with tf.GradientTape() as tape:
    y = x * x

# 计算dy/dx
dy_dx = tape.gradient(y, x)
print(dy_dx)  # tf.Tensor(6.0, shape=(), dtype=float32)
```

### 变量（Variable）

变量用于存储模型的参数，可以在训练过程中被更新。

```python
# 创建变量
weights = tf.Variable(tf.random.normal([3, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')
```

## 简单示例：线性回归

下面是一个使用TensorFlow 2.0实现简单线性回归的例子：

```python
import tensorflow as tf
import numpy as np

# 准备数据
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32)

# 创建模型参数
W = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

# 定义模型
def linear_model(x):
    return W * x + b

# 定义损失函数
def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# 定义优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 训练模型
for epoch in range(100):
    # 使用GradientTape跟踪梯度
    with tf.GradientTape() as tape:
        predictions = linear_model(X)
        loss = loss_fn(predictions, y)
    
    # 计算梯度
    gradients = tape.gradient(loss, [W, b])
    
    # 更新参数
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}")

# 测试模型
print(f"Final model: y = {W.numpy()}x + {b.numpy()}")
print(f"Prediction for x=5.0: {linear_model(5.0).numpy()}")
```

在TensorFlow 2.0中，我们不再需要创建计算图和会话，代码更加简洁直观，更符合Python的编程习惯。这是TensorFlow 2.0相比1.x版本的一个重大改进。

> 本文是TensorFlow 2.0学习笔记系列的第一篇，后续将介绍更多高级特性和实用技巧。
