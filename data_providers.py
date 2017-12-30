# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import os
import numpy as np
DEFAULT_SEED = 1337


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch

    # Python 3.x compatibility
    def __next__(self):
        return self.next()


class OneOfKDataProvider(DataProvider):
    """1-of-K classification target data provider.

    Transforms integer target labels to binary 1-of-K encoded targets.

    Derived classes must set self.num_classes appropriately.
    """

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(OneOfKDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1-of-K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1-of-K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

class CIFAR10DataProvider(OneOfKDataProvider):
    """Data provider for CIFAR-10 object images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new CIFAR-10 data provider object.

        Args:
            which_set: One of 'train' or 'valid'. Determines which
                portion of the CIFAR-10 data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.num_classes = 10
        dataDir = "./data/cifar10/"
        inputs = np.zeros((0, 3072), np.uint8)
        targets_list = []
        for filename in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
            data_path = os.path.join(dataDir, '{0}'.format(filename))
            assert os.path.isfile(data_path), (
                '!!!!!Data file does not exist at expected path: ' + data_path
            )
            dict_tmp = unpickle(data_path)
            inputs = np.vstack((inputs, dict_tmp['data']))
            targets_list = targets_list + dict_tmp['labels']
            dbg = 3
        targets = np.array(targets_list)
        targets_onehot = np.zeros((targets.shape[0], self.num_classes))
        targets_onehot[np.arange(targets.shape[0]), targets] = 1

        inputs = inputs.astype(np.float32)
        inputs = (inputs/255.0)*2 - 1
        inputs = np.transpose(np.reshape(inputs, (-1, 3, 32, 32)), (0, 2, 3, 1))

        # pass the loaded data to the parent class __init__
        super(CIFAR10DataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

class MNISTStackedDataProvider(OneOfKDataProvider):
    """Data provider for MNIST Staked images."""

    def __init__(self, batch_size=100, max_num_batches=-1, shuffle_order=True, rng=None):
        self.num_classes = 10
        dataDir = "./data/mnist_stacked/"
        data_path = os.path.join(dataDir, 'dataset.npy')
        inputs = np.load(data_path)
        inputs = (inputs) * 2 - 1
        inputs = np.transpose(inputs, (0, 2, 3, 1))

        targets = np.zeros((inputs.shape[0], 1), dtype=np.int16)
        # pass the loaded data to the parent class __init__
        super(MNISTStackedDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)
