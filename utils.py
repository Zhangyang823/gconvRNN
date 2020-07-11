#!/usr/bin/python
#-*- coding: utf-8
import tensorflow as tf
import numpy as np
import math
import pickle
import os
import json
from datetime import datetime
from IPython import embed
import tensorflow.contrib.slim as slim
from scipy.sparse import coo_matrix
from tqdm import trange

def save_config(model_dir, config):
    '''
    save config params in a form of param.json in model directory
    '''
    param_path = os.path.join(model_dir, "params.json")

    print("[*] PARAM path: %s" %param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def prepare_dirs(config):
    if config.load_path: #'--load_path', type=str, default=""
        config.model_name = "{}_{}".format(config.task, config.load_path)  #'--task', type=str, default='ptb_char',
                                                                              #choices=['ptbchar', 'swissmt'], help=''
    else:
        config.model_name = "{}_{}".format(config.task, get_time())

    config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory '%s' created" %path)


def pklLoad(fname):
    with open(fname, 'rb') as f:
        print("aaa")
        # print(pickle.load(f)) #show file
        return pickle.load(f)



def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def convert_to_one_hot(a, max_val=None):
    N = a.size
    data = np.ones(N,dtype=int)
    sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
    return np.array(sparse_out.todense())


class BatchLoader(object):
    def __init__(self, data_dir, dataset_name, batch_size, seq_length):
        train_fname = os.path.join(data_dir, dataset_name, 'ptb.char.train.txt')
        # valid_fname = os.path.join(data_dir, dataset_name, 'ptb.char.valid.txt')
        # test_fname = os.path.join(data_dir, dataset_name, 'ptb.char.test.txt')
        # input_fnames = [train_fname, valid_fname, test_fname]
        input_fnames = [train_fname]

        # vocab_fname = os.path.join(data_dir, dataset_name, 'vocab_char.pkl')
        tensor_fname = os.path.join(data_dir, dataset_name, 'bigfeature.pkl')
        Adj_fname = os.path.join(data_dir, dataset_name, 'adj.pkl')

        # if not os.path.exists(vocab_fname) or not os.path.exists(tensor_fname) or not os.path.exists(Adj_fname):
        #     print("Creating vocab...")
        #     #创建pkl文件
        #     self.text_to_tensor(input_fnames, vocab_fname, tensor_fname, Adj_fname)

        # print("Loading vocab...")

        adj = pklLoad(Adj_fname)
        all_data = pklLoad(tensor_fname) #读取文本数字信息
        # self.idx2char, self.char2idx = pklLoad(vocab_fname)
        # vocab_size = len(self.idx2char)

        # print("Char vocab size: %d" % (len(self.idx2char)))
        self.sizes = []
        self.all_batches = []
        self.all_data = all_data
        self.adj = adj

        print("Reshaping tensors...")
        for split, data in enumerate(all_data):  # split = 0:train, 1:valid, 2:test
            #Cutting training sample for check profile fast..(Temporal)
            #if split==0:
            #    #Only for training set
            #    length = data.shape[0]
            #    data = data[:int(length/4)]
            print(split)
            print('sadeyhfuysssseeeeeeeeeeeeeeeeee')
            print (data)
            length = data.shape[0]
            # 代码的意思是想把 data 分成 batch_size 个 batch
            # 每个batch里面的一条数据的长度是 seq_length
            # 而不够一个 batch 的数据就丢掉不要了

            data = data[: batch_size * seq_length * int(math.floor(length / (batch_size * seq_length)))]
            #将data弄成1000的整数，例如原本5017483变成5017000
            ydata = np.zeros_like(data)
            #移位
            ydata[:-1] = data[1:].copy()
            ydata[-1] = data[0].copy()

            # 不明白 没看出来 if else 有什么区别
            tt = [-1, batch_size, seq_length,2,110]
            if split < 2:
                x_batches = list(data.reshape(tt))
                #变成20*50=1000,方便后面处理
                y_batches = list(ydata.reshape(tt))
                self.sizes.append(len(x_batches))
            else:
                x_batches = list(data.reshape(tt))
                y_batches = list(ydata.reshape(tt))
                self.sizes.append(len(x_batches))

            # 将数据作为x，移位的数据作为y，存入all_batches
            self.all_batches.append([x_batches, y_batches])

        self.batch_idx = [0, 0, 0]
        print("data load done. Number of batches in train: %d, val: %d, test: %d"  \
              % (self.sizes[0], self.sizes[1], self.sizes[2]))

    def next_batch(self, split_idx):
        # cycle around to beginning
        if self.batch_idx[split_idx] >= self.sizes[split_idx]:
            self.batch_idx[split_idx] = 0
        idx = self.batch_idx[split_idx]

        self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
        #all_batches[0][0][0],第一个0表示train,第二个0表示x_batch,第三个0表示x_batch第一行
        return self.all_batches[split_idx][0][idx], \
               self.all_batches[split_idx][1][idx]

    def reset_batch_pointer(self, split_idx, batch_idx=None):
        if batch_idx == None:
            batch_idx = 0
        self.batch_idx[split_idx] = batch_idx

    def text_to_tensor(self, input_files, vocab_fname, tensor_fname, Adj_fname):
        counts = []
        char2idx = {}
        idx2char = []

        output = []
        # 这部分的for循环的作用是：
        # 对每个数据文件(train\test\valid)作预处理
        for input_file in input_files:
            count = 0
            output_chars = []
            with open(input_file) as f:
                # for：
                # 使 char2idx 变成26个英文字母+其他文中出现的字符的字典
                # 使 idx2char 变成存储字符的list，里面的字符都是唯一的
                for line in f:
                    line = ''.join(line.split())
                    chars_in_line = list(line)
                    chars_in_line.append('|')
                    for char in chars_in_line:
                        # 他的意图是将line的每个字符变成字典里面对应的数字
                        # 然后将转换出来的数字放入output_chars里面，形成序列
                        if char not in char2idx:
                            idx2char.append(char)
                            # print("idx: %d, char: %s" %(len(idx2char), char))
                            char2idx[char] = len(idx2char) - 1
                        output_chars.append(char2idx[char])
                        count += 1
            counts.append(count)
            output.append(np.array(output_chars))
        keys=list(char2idx.keys())
        print (keys,len(keys))
        train_data = output[0]
        train_data_shift = np.zeros_like(train_data)
        # train_data的第一个字符移到最后面，就变成了train_data_shift
        train_data_shift[:-1] = train_data[1:].copy()
        train_data_shift[-1] = train_data[0].copy()

        # Co-occurance
        Adj = np.zeros([len(idx2char), len(idx2char)])
        for x, y in zip(train_data, train_data_shift):
            Adj[x, y] += 1
        #将adj是50*50的矩阵，序列数据xy存在的值加1，为权重。

        # Make Adj symmetric & visualize it

        # pickle 的作用是将对象转化为文件存在磁盘中
        # 下次直接读取文件就能还原对象
        print("Number of chars : train %d, val %d, test %d" % (counts[0], counts[1], counts[2]))
        pklSave(vocab_fname, [idx2char, char2idx])
        pklSave(tensor_fname, output)
        pklSave(Adj_fname, Adj)

# data_loader = BatchLoader('datasets','ptb_char',20, 50)
# data_loader.reset_batch_pointer(0)
# for k in trange(data_loader.sizes[0], desc="[per_batch]"):
#     # Fetch training data
#     batch_x, batch_y = data_loader.next_batch(0)
#     batch_x_onehot = convert_to_one_hot(batch_x, 50)
