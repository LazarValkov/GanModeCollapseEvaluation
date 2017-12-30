from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from data_providers import *
from collections import OrderedDict
from keras.optimizers import Adam
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

_graph_replace = tf.contrib.graph_editor.graph_replace
def remove_original_op_attributes(graph):
    """Remove _original_op attribute from all operations in a graph."""
    for op in graph.get_operations():
        op._original_op = None


def graph_replace(*args, **kwargs):
    """Monkey patch graph_replace so that it works with TF 1.0"""
    remove_original_op_attributes(tf.get_default_graph())
    return _graph_replace(*args, **kwargs)


def extract_update_dict(update_ops):
    """Extract variables and their new values from Assign and AssignAdd ops.

    Args:
        update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()

    Returns:
        dict mapping from variable values to their updated value
    """
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = OrderedDict()
    for update in update_ops:
        var_name = update.op.inputs[0].name
        var = name_to_var[var_name]
        value = update.op.inputs[1]
        if update.op.type == 'Assign':
            updates[var.value()] = value
        elif update.op.type == 'AssignAdd':
            updates[var.value()] = var + value
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd" % update_ops.op.type)
    return updates

def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    try:
        return st.StochasticTensor(
            ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs))
    except:
        return st.StochasticTensor(
            ds.MultivariateNormalDiag(loc=tf.zeros(shape), scale_diag=tf.ones(shape), **kwargs))


class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, config=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop
    self.batch_size = batch_size
    self.sample_num = sample_num
    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width
    self.z_dim = z_dim
    self.gf_dim = gf_dim
    self.df_dim = df_dim
    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim
    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    if dataset_name == 'cifar10':
      self.data = CIFAR10DataProvider(batch_size=self.batch_size)
      self.c_dim = 3
    elif dataset_name == 'mnist_stacked':
      self.data = MNISTStackedDataProvider(batch_size=self.batch_size)
      self.c_dim = 3
    else:
      self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
      self.c_dim = imread(self.data[0]).shape[-1]

    self.grayscale = (self.c_dim == 1)
    self.config = config

    self.build_model()

  def build_model(self):
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    def sigmoid_cross_entropy_with_logits(x, y):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    self.inputs = tf.placeholder(tf.float32, [None] + image_dims, name='real_images')

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G = self.generator(self.z)
    self.sampler = self.sampler(self.z)

    if self.config.ali:
        self.R, self.reconstructed_means = self.reconstructor(self.inputs)
        self.R_, self.reconstructed_means_ = self.reconstructor(self.G, reuse=True)
        self.D, self.D_logits = self.discriminator_joint(self.inputs, self.R)
        self.D_, self.D_logits_ = self.discriminator_joint(self.G, self.z, reuse=True)
    else:
        self.D, self.D_logits = self.discriminator(self.inputs)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    reconstruction = self.config.ali

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.r_vars = [var for var in t_vars if 'r_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name or (reconstruction and 'r_' in var.name)]

    if self.config.ali:
        print "Using ALI losses"
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.zeros_like(self.D))) + \
                      tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    elif self.config.unrolled_gan:
        print "Using Unrolled GAN losses"
        # Vanilla discriminator update
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        d_opt = Adam(lr=2e-4, beta_1=self.config.beta1, epsilon=1e-8)
        updates = d_opt.get_updates(self.d_vars, [], self.d_loss)
        self.d_optim = tf.group(*updates, name="d_train_op")
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        # Unroll optimization of the discrimiantor
        if self.config.unrolling_steps > 0:
            # Get dictionary mapping from variables to their update value after one optimization step
            update_dict = extract_update_dict(updates)
            cur_update_dict = update_dict
            for i in xrange(self.config.unrolling_steps - 1):
                # Compute variable updates given the previous iteration's updated variable
                cur_update_dict = graph_replace(update_dict, cur_update_dict)
            # Final unrolled loss uses the parameters at the last time step
            unrolled_loss = graph_replace(self.g_loss, cur_update_dict)
        else:
            unrolled_loss = self.g_loss

        # Optimize the generator on the unrolled loss
        g_train_opt = tf.train.AdamOptimizer(1e-4, self.config.beta1, epsilon=1e-8)
        self.g_optim = g_train_opt.minimize(unrolled_loss, var_list=self.g_vars)
    else:
        print "Using VANILLA GAN losses"
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))


    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    self.saver = tf.train.Saver()
    self.sampler_w_var = self.sampler_w_var()

  def train(self, config):
    d_optim = None
    g_optim = None
    if self.config.unrolled_gan:
        #the optimizers for unrolled gan are already created
        d_optim = self.d_optim
        g_optim = self.g_optim
    else:
        print "not unrolled_gan, creating new AdamOptimizers"
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./" + config.main_output_dir + "/logs", self.sess.graph)

    if self.config.z_uniform:
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
    else:
        sample_z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))


    if config.dataset == 'mnist_stacked' or config.dataset == 'cifar10':
      sample_inputs, sample_labels = self.data.next()
      self.data.new_epoch()
    else:
      sample_files = self.data[0:self.batch_size]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'cifar10' or config.dataset == 'mnist_stacked':
        batch_idxs = self.data.num_batches
      else:
        self.data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      # TODO: for debugging purposes only, remove.
      #batch_idxs = 2
      for idx in xrange(0, batch_idxs):
        if config.dataset == 'cifar10' or config.dataset == 'mnist_stacked':
          batch_images, batch_labels = self.data.next()
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        if self.config.z_uniform:
            batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        else:
            batch_z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.inputs: batch_images, self.z: batch_z})
        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.inputs: batch_images, self.z: batch_z})

        print "counter = " + str(counter)
        if np.mod(counter, 20000) == 1:
            self.writer.add_summary(summary_str, counter)
            self.writer.add_summary(summary_str, counter)
            self.save(config.checkpoint_dir, counter)

        if np.mod(counter, 2000) == 1:
            errD_fake = self.d_loss_fake.eval({ self.z: batch_z, self.inputs: batch_images })
            errD_real = self.d_loss_real.eval({ self.z: batch_z, self.inputs: batch_images })
            errG = self.g_loss.eval({self.z: batch_z, self.inputs: batch_images})

            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                  % (epoch, idx, batch_idxs,
                     time.time() - start_time, errD_fake + errD_real, errG))
        else:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f"
                  % (epoch, idx, batch_idxs, time.time() - start_time))

        if np.mod(counter, 5000) == 1:
          try:
            samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
            )
            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            save_images(samples, [manifold_h, manifold_w],
                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
          except:
            print("one pic error!...")

        counter += 1

      if config.dataset == 'cifar10' or config.dataset == 'mnist_stacked':
        self.data.new_epoch()

    self.save(config.checkpoint_dir, counter+1)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h3_shape = h3.get_shape().as_list()
      h3_reshaped = tf.reshape(h3, [-1, h3_shape[1]*h3_shape[2]*h3_shape[3]])
      h4 = linear(h3_reshaped, 1, 'd_h3_lin')

      return tf.nn.sigmoid(h4), h4

  def discriminator_joint(self, image, z, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h3_shape = h3.get_shape().as_list()
      h3_reshaped = tf.reshape(h3, [-1, h3_shape[1] * h3_shape[2] * h3_shape[3]])

      z_shaped = tf.reshape(z, [self.batch_size, self.z_dim], name="rehaping_z")
      h3_flat_z_concat = tf.concat([h3_reshaped, z_shaped], axis=1)
      c_h0 = lrelu(linear(h3_flat_z_concat, 200, 'd_z_h3_h0'))
      h4 = linear(c_h0, 1, 'd_h3_lin')
      return tf.nn.sigmoid(h4), h4

  def reconstructor(self, image, reuse=False):
    with tf.variable_scope("reconstructor") as scope:
      if reuse:
          scope.reuse_variables()
      print("RECONSTRUCTOR:")
      print image.get_shape()

      flattenedImg = tf.reshape(image, [-1, self.output_height * self.output_width * self.c_dim])

      print flattenedImg.get_shape()

      h = slim.fully_connected(flattenedImg, 500, activation_fn=tf.nn.relu, trainable=True, scope="r_layer1")
      h = slim.fully_connected(h, 500, activation_fn=tf.nn.relu, trainable=True, scope="r_layer2")
      h = slim.fully_connected(h, self.z_dim*2, activation_fn=tf.nn.tanh, trainable=True, scope="r_output_layer")
      a = h[:, :self.z_dim]
      b = h[:, self.z_dim:]
      return st.StochasticTensor(ds.Normal(a, tf.exp(b), name="r_z")), a

  def generator(self, z):
    with tf.variable_scope("generator") as scope:

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
      self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))
      self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))
      h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))
      h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))
      h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
      return tf.nn.tanh(h4)

  def sampler(self, z):
    """
    sampler is constructed in addition to the generator, because it sets train=False for the batch_norm
    :return:
    """
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
      # project `z` and reshape
      h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                      [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))
      h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))
      h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))
      h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))
      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
      return tf.nn.tanh(h4)

  def sampler_w_var(self):
    """
    represents the random latent space as a tensorflow variable.
    this allows said variable to be optimised and is used in Evaluation/Inference_via_optimization
    :return:
    """
    z_shape = [self.sample_num, self.z_dim]

    self.z_var = tf.get_variable("z_var", shape=z_shape, dtype=tf.float32, trainable=False)
    self.z_var_pl = tf.placeholder(dtype=tf.float32, shape=z_shape, name="z_var_placeholder")
    self.z_var_assign = tf.assign(self.z_var, self.z_var_pl, name="z_var_assign")

    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
      # project `z` and reshape
      h0 = tf.reshape(linear(self.z_var, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                      [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))
      h1 = deconv2d(h0, [self.sample_num, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))
      h2 = deconv2d(h1, [self.sample_num, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))
      h3 = deconv2d(h2, [self.sample_num, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))
      h4 = deconv2d(h3, [self.sample_num, s_h, s_w, self.c_dim], name='g_h4')
      return tf.nn.tanh(h4)

  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.dataset_name, self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
