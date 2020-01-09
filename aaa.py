#!/usr/bin/python
#-*- coding: utf-8
import cPickle as pickle

fr = open('./datasets/ptb_char/data_char.pkl')    #open的参数是pkl文件的路径
inf = pickle.load(fr)#读取pkl文件的内容shape=[5017483,393043.442424]
print(type(inf[0]))
print(len(inf[2]))
print("aaaaaaaaaaaaaaaa")

