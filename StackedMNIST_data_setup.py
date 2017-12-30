from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import os

images = None

MNIST_DIM = 28


def prepare_data():
    if not os.path.exists("data"):
        os.makedirs("data")

    mnist = input_data.read_data_sets("data/MNIST/", one_hot=False)

    #test_lables = np.expand_dims(mnist.test.labels, axis=1)
    #train_labels = np.expand_dims(mnist.train.labels, axis=1)
    #valid_labels = np.expand_dims(mnist.validation.labels, axis=1)

    all_images = np.vstack((mnist.test.images, mnist.train.images, mnist.validation.images))
    global images
    images = all_images
    print images.shape


def get_random_digit_image():
    indx = randint(0, images.shape[0]-1)
    img = images[indx]
    resized_img = np.resize(img, [MNIST_DIM, MNIST_DIM])
    return resized_img


def create_dataset(samples):
    dataset = np.zeros((samples, 3, MNIST_DIM, MNIST_DIM), dtype=np.float32)

    for i in xrange(samples):
        img_ch0 = get_random_digit_image()
        img_ch1 = get_random_digit_image()
        img_ch2 = get_random_digit_image()

        dataset[i][0] = img_ch0
        dataset[i][1] = img_ch1
        dataset[i][2] = img_ch2

    #at this point, the dataset is in CHW format, with float32 in the range [0, 1]
    return dataset


def saveDataset(dataset, dir="data/mnist_stacked"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = dir + "/dataset.npy"
    np.save(filename, dataset)

'''
Create a dataset
'''
prepare_data()
dataset = create_dataset(50000)
saveDataset(dataset)

'''
Uncomment the following lines to visualise 1 data point.

dataset = np.load("data/mnist_stacked/dataset.npy")
print dataset.shape
i = dataset[7]
i = np.transpose(i, (1, 2, 0))
imgplot = plt.imshow(i)
plt.show()
'''
