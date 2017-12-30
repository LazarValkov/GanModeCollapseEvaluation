import tensorflow as tf
import numpy as np
import os
import math
from time import gmtime, strftime
from data_providers import CIFAR10DataProvider
from utils import save_images, inverse_transform
import tensorflow.contrib.slim as slim


def get_random_samples(sess, dcgan, config):

    dir_str = config.main_output_dir + '/random_samples/'
    filename = "samples"
    if not os.path.exists(dir_str):
        os.makedirs(dir_str)
    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    timestr = strftime("%Y%m%d%H%M%S", gmtime())

    number_of_batches = config.sample_num // config.batch_size
    samples = np.zeros((0, config.output_height, config.output_width, dcgan.c_dim), dtype=np.float32)
    for i in xrange(number_of_batches):
        print str(i) + " / " + str(number_of_batches)
        if dcgan.config.z_uniform:
            z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
        else:
            z_sample = np.random.normal(0, 1, size=(config.batch_size, dcgan.z_dim))

        generated_samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        print "generated_samples.shape"
        print generated_samples.shape
        samples = np.vstack((samples, generated_samples))

        save_images(generated_samples, [image_frame_dim, image_frame_dim],
                    (dir_str + '%s_%s_%s.png') % (filename, timestr, str(i)))

    np.save((dir_str + "%s_%s.npy") % (filename, timestr), samples)


def get_inference_via_optimization(sess, dcgan, config, data):
    # decide on batch_size images (picked randomly) to try to reconstruct
    targets = data.inputs[np.random.randint(data.inputs.shape[0], size=config.sample_num)]
    print "TARGETS SHAPE"

    # setup objective
    g_loss = tf.nn.l2_loss(dcgan.sampler_w_var - targets)

    a = (dcgan.sampler_w_var + 1.) / 2.
    b = (targets + 1.) / 2.
    mse = ((a - b) ** 2)
    mse_2d = tf.reshape(mse, [config.sample_num, 32 * 32 * 3])
    mse = tf.reduce_mean(mse_2d, axis=1, keep_dims=True)
    '''
    mse = ((dcgan.sampler_w_var - targets) ** 2)
    mse_2d = tf.reshape(mse, [config.batch_size, 32 * 32 * 3])
    mse = tf.reduce_mean(mse_2d, axis=1, keep_dims=True)
    '''

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        loss=g_loss,
        var_list=[dcgan.z_var],
        method='L-BFGS-B',
        options={'maxiter': config.eval_infvo_lbfgsb_maxiter,
                 'disp': True})

    # run the optimization from 3 different initializations
    results_images = []
    results_errors = []
    num_of_random_restarts = 3

    sample_z_dim = dcgan.z_dim

    for i in xrange(num_of_random_restarts):
        # randomly initialize z
        if dcgan.config.z_uniform:
            z_sample = np.random.uniform(-1, 1, size=(config.sample_num, sample_z_dim))
        else:
            z_sample = np.random.normal(0, 1, size=(config.sample_num, sample_z_dim))

        # assign it to the variable
        sess.run(dcgan.z_var_assign, {dcgan.z_var_pl: z_sample})

        optimizer.minimize(sess)

        generated_samples = sess.run(dcgan.sampler_w_var)
        generated_samples_mse = sess.run(mse)
        results_images.append(generated_samples)
        results_errors.append(generated_samples_mse)

    # select the best out of all random restarts
    best_images = np.zeros_like(results_images[0])
    best_images_errors = np.zeros_like(results_errors[0])
    for image_index in xrange(config.sample_num):
        best_img = results_images[0][image_index]
        best_img_error = results_errors[0][image_index][0]
        for indep_run_index in xrange(1, num_of_random_restarts):
            if best_img_error > results_errors[indep_run_index][image_index][0]:
                best_img_error = results_errors[indep_run_index][image_index][0]
                best_img = results_images[indep_run_index][image_index]
        best_images[image_index] = best_img
        best_images_errors[image_index][0] = best_img_error

    filename = "test"

    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    dir_str = config.main_output_dir + '/eval_ivo/'
    timestr = strftime("%Y%m%d%H%M%S", gmtime())
    if not os.path.exists(dir_str):
        os.makedirs(dir_str)

    save_images(best_images[:config.batch_size], [image_frame_dim, image_frame_dim],
                (dir_str + '%s_%s.png') % (filename, timestr))
    save_images(targets[:config.batch_size], [image_frame_dim, image_frame_dim],
                (dir_str + '%s_%s_true.png') % (filename, timestr))

    # np.save(dir_str + timestr + "_best_images" + ".npy", best_images)
    # np.save(dir_str + timestr + "_targets" + ".npy", targets)
    # np.save(dir_str + timestr + "_best_images_errors" + ".npy", best_images_errors)

    return best_images_errors


def eval_inference_via_optimization(sess, dcgan, config):
    # assert config.dataset == 'cifar10'
    # set numpy state, so that you get the same images every time
    np.random.seed(1337)
    if config.dataset == 'cifar10':
        data = CIFAR10DataProvider(batch_size=config.sample_num)
    else:
        raise NotImplementedError("Inference via optimisation is only implemented for CIFAR10 atm." +
                                  "You can easily implement it for other datasets.")

    best_images_errors = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(1):
        best_images_errors_i = get_inference_via_optimization(sess, dcgan, config, data)
        best_images_errors = np.vstack((best_images_errors, best_images_errors_i))

    mean = np.mean(best_images_errors)
    return mean

# ******************************************** Code for evaluating the performance on MNIST_Stacked **************

class MNISTClassifier:
    def __init__(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.cnn(self.x)
        self.predictions = tf.argmax(self.output_logits, 1)
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output_logits))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

        # optimizer = tf.train.AdamOptimizer()
        # train_op = slim.learning.create_train_op(cross_entropy, optimizer, summarize_gradients=True)


        correct_prediction = tf.equal(tf.argmax(self.output_logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.saver = tf.train.Saver()

    def cnn(self, images):
        x_image = tf.reshape(images, [-1, 28, 28, 1])
        net = slim.layers.conv2d(x_image, 50, [5, 5], scope='conv1')
        net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.layers.conv2d(x_image, 50, [5, 5], scope='conv2')
        net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.layers.flatten(net, scope='flatten3')
        net = slim.layers.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fully_connected1')
        net = tf.nn.dropout(net, self.keep_prob)
        self.output_logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='fully_connected2')

    def train(self, sess, mnist, epochs=100):
        i = 1
        loz = 0
        while mnist.train.epochs_completed < epochs:
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            loz, _ = sess.run([self.cross_entropy, self.train_step]
                              , feed_dict={self.x: batch_xs, self.labels: batch_ys, self.keep_prob: 0.5})
            i += 1
            if i % 100 == 0:
                print(loz)

    def test(self, sess, mnist):
        print(sess.run(self.accuracy,
                       feed_dict={self.x: mnist.validation.images, self.labels: mnist.validation.labels, self.keep_prob: 1.0}))

    def save(self, sess, uri="mnist_classifier/mnist_classifier.ckpt"):
        if not os.path.exists("mnist_classifier"):
            os.makedirs("mnist_classifier")
        self.saver.save(sess, uri)

    def restore(self, sess, uri="mnist_classifier/mnist_classifier.ckpt"):
        self.saver.restore(sess, uri)


def eval_mnist_stacked_generate_images(sess, dcgan, config):
    # generate N images
    n = config.eval_mnist_stacked_examples

    generated = np.zeros((n, 3, 28, 28), dtype=np.float32)

    n_generated = 0
    while True:
        to_be_added_count = config.batch_size
        if n_generated + config.batch_size > n:
            to_be_added_count = n - n_generated

        if config.z_uniform:
            sample_z = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
        else:
            sample_z = np.random.normal(0, 1, size=(config.batch_size, dcgan.z_dim))

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z})
        samples = inverse_transform(samples)
        samples = np.transpose(samples, (0, 3, 1, 2))

        generated[n_generated:n_generated + to_be_added_count] = samples[0:to_be_added_count]

        n_generated += to_be_added_count

        print n_generated

        if n_generated >= n:
            break

    """
    # save to an nparray of shape [N, 3, 28, 28]
    dir_str = './' + config.main_output_dir + '/eval_stacked_mnist/'
    if not os.path.exists(dir_str):
        os.makedirs(dir_str)
    np.save(dir_str + "eval_mnist_stacked_gen_dataset.npy", generated)"""
    return generated


def eval_mnist_stacked(sess_main, dcgan, config):
    # 1. Load Data
    # dataset = np.load("data/eval_mnist_stacked_gen_dataset.npy")
    dataset = eval_mnist_stacked_generate_images(sess_main, dcgan, config)

    g_1 = tf.Graph()
    with g_1.as_default():
        model = MNISTClassifier()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        model.restore(sess)

        total_modes = 1000

        # 2. Select N samples
        np.random.shuffle(dataset)
        selected = dataset  # [0:10000]
        # 3. Split them in order to get 3*N examples
        ds = np.reshape(selected, (selected.shape[0] * 3, 28, 28))
        # 4. Predict labels on these examples
        ds = np.reshape(ds, (-1, 28 * 28))
        batch_size = 1000
        num_points = ds.shape[0]
        print num_points
        print batch_size
        assert num_points % batch_size == 0

        predictions = np.zeros((ds.shape[0], 1))

        for i in xrange(num_points // batch_size):
            start_index = batch_size * i
            end_index = batch_size * (i + 1)
            predictions[start_index:end_index] = np.reshape(sess.run(model.predictions,
                                                                     {model.x: ds[start_index:end_index],
                                                                      model.keep_prob: 1.0}), (-1, 1))
        # predictions = sess.run(model.predictions, {model.x: ds, model.keep_prob: 1.0})
        # 5. Reshape to [N, 3]
        predictions = np.reshape(predictions, (-1, 3))
        # 6. Multiply by a constant to get the mode number
        const = np.array([[100.], [10.], [1.]])
        modes = np.matmul(predictions, const)
        # 7. Iterate and record the number of modes
        modes_count = np.zeros((total_modes, 1))
        for i in xrange(modes.shape[0]):
            modeNum = int(modes[i])
            modes_count[modeNum] += 1

        num_covered_modes = np.count_nonzero(modes_count)

        # calculate KL
        modes_count_normalized = modes_count / np.sum(modes_count)
        kl = 0
        Pdata = 1. / total_modes
        for i in xrange(total_modes):
            if int(modes_count[i]) == 0:
                continue
            kl += modes_count_normalized[i] * (math.log(modes_count_normalized[i]) - math.log(Pdata))

        print("num_covered_modes= " + str(num_covered_modes))
        print("kl= " + str(kl))
        return kl
