import numpy as np
from main import run_app
import tensorflow as tf
import os
from utils import pp

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

if __name__ == '__main__':
    start_index = 2 
    runs = 1
    for i in xrange(start_index, start_index+runs):
        flags = {
            "learning_rate": 0.0002,  # "Learning rate of for adam [0.0002]"
            "beta1": 0.5,  # "Momentum term of adam [0.5]"
            "train_size": np.inf,  # "The size of train images [np.inf]"
            "input_height": None, # Set automatically, if the dataset is [mnist_stacked, cifar10 or celebA]
            "input_width": None,
            "output_height": None,
            "output_width": None,
            "crop": True,  # "True for training, False for testing [False]"

            "main_output_dir": 0,  # "The root directory to output to."
            "checkpoint_dir": None,  # "checkpoint", "Directory name to save the checkpoints [checkpoint]"
            "sample_dir": None,  # "Directory name to save the image samples [samples]"
            "input_fname_pattern": "*.jpg",

            "batch_size": 3,  # "The size of batch images [64]"
            "sample_num": 64,
            "epoch": 60,  # "Epoch to train [100]"
            "dataset": "celebA",  # "The name of dataset [celebA, mnist, lsun, mnist_stacked, cifar10]"
            "train": True,  # "True for training, False for testing [False]"
            "unrolled_gan": True, #Whether or not to use unrolled_gan
            "unrolling_steps": 5,
            "z_uniform": False,
            "z_dim": 256,
            "ali": False,

            "visualize": True,  # "True for visualizing, False for nothing [False]"
            "eval_infvo_lbfgsb_maxiter": -1,    # "UnrolledGAN's Inferene via Optimization evaluation.
                                                # maxiter for the l-bfgs-b scipy implementation."
            "eval_mnist_stacked_examples": -1  # "number of examples to generate for UGAN's MNIST stacked evalutation"
        }
        FLAGS = AttributeDict(flags)

        FLAGS.main_output_dir = str("Unrolled/" + flags["dataset"]) + "/ali_dcgan_run" + str(i)
        FLAGS.sample_dir = FLAGS.main_output_dir + "/samples"
        FLAGS.checkpoint_dir = FLAGS.main_output_dir + "/checkpoint"

        # now that settings are configured, run the GAN
        output_str = run_app(FLAGS)

        text_file = open(FLAGS.main_output_dir + "/output.txt", "w")
        text_file.writelines(pp.pformat(FLAGS))
        text_file.write("\n")
        text_file.write("Result: %s" % output_str)
        text_file.close()

        tf.reset_default_graph()
