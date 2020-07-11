#!/usr/bin/python
#-*- coding: utf-8
import sys
import numpy as np
import tensorflow as tf
from config import get_config
from trainer import Trainer
from utils import prepare_dirs, save_config
from deepexplain.tensorflow import DeepExplain

#config = None

def main(_):

    #Directory generating.. for saving
    prepare_dirs(config)

    #Random seed settings
    rng = np.random.RandomState(config.random_seed) #123
    tf.set_random_seed(config.random_seed)
    # sess = tf.Session()
    #Model training

    trainer = Trainer(config, rng)
    save_config(config.model_dir, config)

    if config.is_train:
        trainer.train()
        print("done")
        # trainer = Trainer(config, rng)
        # logits = trainer.model.rnn_output
        # xi, yi = trainer.data_loader.next_batch(0)
        # reshaped = xi.reshape(
        #     [trainer.config.batch_size, trainer.config.num_time_steps, trainer.config.feat_in,
        #      trainer.config.num_node])
        # # [20,50,1,50]->[20,50,1,50] batchsize,num_node,1,numtime_steps
        # xi = np.transpose(reshaped, (0, 3, 2, 1))
        # # x = np.reshape(xi[0, :, :, 0], [110, 2])
        # yreshaped = yi.reshape(
        #     [trainer.config.batch_size, trainer.config.num_time_steps, trainer.config.feat_in,
        #      trainer.config.num_node])
        # yi = np.transpose(yreshaped, (0, 3, 2, 1))
        # # y = np.reshape(yi[0, :, :, 0], [110, 2])
        # applyy = tf.multiply(logits, yi)
        # # print("[*] Checking if previous run exists in {}"
        # #       "".format(trainer.config.model_dir))
        # # latest_checkpoint = tf.train.latest_checkpoint(trainer.model_dir)
        # # trainer.saver.restore(sess, latest_checkpoint)  # 加载到当前环境中
        # print("[*] Saved result exists! loading...")
        #
        # print("asdfghjklweeeeeerty")
        # print(logits)
        # # data_loader=BatchLoader(data_dir, dataset_name,
        # #             batch_size, num_time_steps)
        #
        # print ("ok")
        # attributions = de.explain('grad*input', applyy, trainer.model.rnn_input_seq, xi)
        # np.savetxt('0_features.csv', attributions[0], delimiter=', ')
        # print ('Done')

    else:
        if not config.load_path:
            raise Exception(
                "[!] You should specify `load_path` to "
                "load a pretrained model")
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
