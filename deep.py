import sys
import numpy as np
import tensorflow as tf
from config import get_config
from trainer import Trainer
from utils import prepare_dirs, save_config
from deepexplain.tensorflow import DeepExplain
def main():
    prepare_dirs(config)
    rng = np.random.RandomState(config.random_seed)  # 123
    tf.set_random_seed(config.random_seed)

    model=Trainer(config, rng)
    latest_checkpoint = tf.train.latest_checkpoint(config.model_dir)
    trainer.saver.restore(sess, latest_checkpoint)
    with DeepExplain(session=sess) as de:
        logits = trainer.model
        xi, yi = trainer.data_loader.next_batch(0)
        reshaped = xi.reshape(
            [trainer.config.batch_size, trainer.config.num_time_steps, trainer.config.feat_in,
             trainer.config.num_node])
        # [20,50,1,50]->[20,50,1,50] batchsize,num_node,1,numtime_steps
        xi = np.transpose(reshaped, (0, 3, 2, 1))
        # x = np.reshape(xi[0, :, :, 0], [110, 2])
        yreshaped = yi.reshape(
            [trainer.config.batch_size, trainer.config.num_time_steps, trainer.config.feat_in,
             trainer.config.num_node])
        yi = np.transpose(yreshaped, (0, 3, 2, 1))
        # y = np.reshape(yi[0, :, :, 0], [110, 2])
        applyy = tf.multiply(logits, yi)
        print("[*] Checking if previous run exists in {}"
              "".format(trainer.config.model_dir))
        latest_checkpoint = tf.train.latest_checkpoint(trainer.model_dir)
        trainer.saver.restore(sess, latest_checkpoint)  # 加载到当前环境中
        print("[*] Saved result exists! loading...")

        print("asdfghjklweeeeeerty")
        print(logits)
        # data_loader=BatchLoader(data_dir, dataset_name,
        #             batch_size, num_time_steps)

        print ("ok")
        attributions = de.explain('grad*input', applyy, trainer.model.rnn_input_seq, xi)
        np.savetxt('0_features.csv', attributions[0], delimiter=', ')
        print ('Done')


if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
