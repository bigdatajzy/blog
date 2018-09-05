---
title: tensorflow 学习资料整理
date: 2018-9-05 8:55:00
categories:
- 人工智能/tensorflow
tags: 人工智能,框架
---

学习tensorflow之前我们需要先统一一下认知
===========

一、基本概念
===========

基于Tensorflow的NN：用张量表示数据，用计算图搭建神经网络，用会话执行计算图，优化线上的权重（参数），得到模型。
===========

张量：张量就是作为数组（列表），用“阶”表示张量的维度。
===========

0阶张量称作标量，表示一个单独的数；

举例 S=123

1阶张量称作向量，表示一个一维数组；

举例 V=[1，2，3]

2阶张量称作矩阵，表示一个二维数组，它可以有i 行j列个元素，每个元素可以用行号和列号共同索引到；

举例 m=[[1,2,3],[4,5,6],[7,8,9]]

判断张量是几阶的，就通过张量右边的方括号数，0个是0阶，n个是n阶，张量可以表示0阶到n阶数组（列表）；

举例 t=[[[...]]]为3阶。

数据类型：Tenserflow的数据类型有tf.float32、tf.int32等。
============

举例

我们实现tensorflow的加法：

import tensorflow as tf    //引入模块

a = tf.constant([1.0,2.0]) //定义一个张量等于[1.0,2.0]

b = tf.constant([3.0,4.0]) //定义一个张量等于[3.0,4.0]

result = a + b             //实现a加b的加法

print result               //打印出结果

可以打印出这样一句话：Tensor("add:0",shape=(2, ),dtype=float32),意思为result是一名称为add：0的张量，shape=（2，） 表示一维数组长度为2，dtype=float32表示数据类型为浮点型。

计算图（Graph）：搭建神经网络的计算过程，只承载一个或多个计算节点的一张图，只搭建网络不运算。
=============
举例

神经网络的基本模型是神经元，神经元的基本模型就是数学中的乘法、加法运算。我们搭建的层数为2两个子节点的计算图，x1、x2表示输入，w1、w2分别是x1到y和x2到y的权重，y=x1*w1+x2*w2.

实现我们上述的计算图：

import tensorflow as tf      #引入模块

x = tf.constant([[1.0,2.0]]) #定义一个2阶张量等于[[1.0,2.0]]

w = tf.constant([3.0],[4.0]) #定义一个2阶张量等于[[3.0],[4.0]]

y = tf.matmul(x,w)           #实现xw矩阵乘法

print y                      #打印出结果

可以打印出这样一句话：Tensor（“matmul：0”，shape（1，1），dtype=float32），从这里我们
可以看出，print的结果显示y是一个张量，只搭建承载计算过程的计算图，并没有运算，如果我们想得到运算结果就要用到“会话Session（）”了。

会话（Session）：执行计算图中的结点运算。
=============

我们用with结构实现，语法如下：

with tf.Session() as sess:

  print sess.run(y)

举例

对于刚刚所述的计算图，我们执行Session（）会话可得到矩阵相乘结果：

import tensorflow as tf        #引入模块

x = tf.constant([[1.0,2.0]])   #定义一个2阶张量等于[[1.0,2.0]]

w = tf.constant([[3.0],[4.0]]) #定义一个2阶张量等于[[3.0],[4.0]]

y = tf.matmul(x,w)             #实现xw矩阵乘法

print y                        #打印出结果

with tf.Session() as sess:

  print sess.run(y)            #执行会话并打印出执行后的结果

可以打印出这样的结果：

Tensor（“matmul:0”,shape(1,1),dtype=float32）

[[11.]]

我们可以看到，运行Session（）会话前只打印出y是个张量的提示，运行Session（）会话后打印出
了y的结果1.0*3.0+2.0*4.0 = 11.0.




>文章来源于tensorflow笔记
