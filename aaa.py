#!/usr/bin/python
#-*- coding: utf-8
# import cPickle as pickle
# #读取pkl文件
# fr = open('./datasets/ptb_char/data_char.pkl')    #open的参数是pkl文件的路径
# inf = pickle.load(fr)#读取pkl文件的内容shape=[5017483,393043.442424]
# print(type(inf[0]))
# print(len(inf[2]))

import numpy as np
import tensorflow as tf
import numpy
indices = tf.reshape(tf.range(0, 10 ,1), [10, 1])
labels=tf.expand_dims(tf.constant([0,2,3,6,7,9,1,3,5,4]),1)
print(indices)
print(labels)
onehot = tf.sparse_to_dense(
      tf.concat(values=[indices, labels], axis=1),
      [10, 10], 1.0, 0.0)
with tf.Session()  as sess:
      a = sess.run(onehot)
      print(a)
