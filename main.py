import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, show_all_variables
from eval import *

import tensorflow as tf


def run_app(FLAGS):
  pp.pprint(FLAGS)
  if FLAGS.input_width is None:
      if FLAGS.dataset == "mnist_stacked":
          FLAGS.input_width = 28
          FLAGS.input_height = 28
          FLAGS.output_width = 28
          FLAGS.output_height = 28
      elif FLAGS.dataset == "cifar10":
          FLAGS.input_width = 32
          FLAGS.input_height = 32
          FLAGS.output_width = 32
          FLAGS.output_height = 32
      elif FLAGS.dataset == "celebA":
          FLAGS.input_width = 108
          FLAGS.input_height = 108
          FLAGS.output_width = 64
          FLAGS.output_height = 64
          FLAGS.crop = True

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  #FLAGS.checkpoint_dir = os.path.join(FLAGS.main_output_dir, FLAGS.checkpoint_dir)
  #FLAGS.sample_dir = os.path.join(FLAGS.main_output_dir, FLAGS.sample_dir)
  print "CREATING DIRECTORY"
  print FLAGS.main_output_dir
  print FLAGS.sample_dir

  if not os.path.exists(FLAGS.main_output_dir):
    os.makedirs(FLAGS.main_output_dir)
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.sample_num,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        config=FLAGS,
        z_dim=FLAGS.z_dim
    )

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      
    if FLAGS.visualize:
        get_random_samples(sess, dcgan, FLAGS)
    if FLAGS.eval_infvo_lbfgsb_maxiter > 0:
        ivo_result =  eval_inference_via_optimization(sess, dcgan, FLAGS)
        print("inference_via_optimisation's score:" + str(ivo_result))
        return ivo_result
    if FLAGS.eval_mnist_stacked_examples > 0:
        assert(FLAGS.dataset == "mnist_stacked")
        result = eval_mnist_stacked(sess, dcgan, FLAGS)
        return result

    return "finished."
