import os
import sys
import tensorflow as tf
import numpy as np

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

#imagenet mean
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """ A wrapper class around the new Tensorflows dataset pipeline

    """
    def __init__ (self, path_file, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):
        """Create a new ImageDataGenerator.

        Receives a path file to a text file, which consists of many lines where each line has two
        parts. The first part is the image patch path and the second part is the class name.
        Using this datasets, this can be used to train a neural network

        Arguments:
        ---------
        path_file: path to the text file contain all the patch path and class number
        mode: either, training or testing dataset.
        batch_size: number of images per batch.
        num_classes: number of classes in the dataset.
        shuffle: whether or not to shuffle the data in the dataset and the initial file list
        buffer_size: Number of images used as buffer for tensorflows shuffling of the dataset.

        """
        self.path_file = path_file
        self.num_classes = num_classes

        #retrieve the path from the text file
        self._read_path_file()

        #number of samples in the dataset
        self.data_size = len(self.labels)

        #initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        #convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        #create a dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        #distringuing between train and test
        if mode =='training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)
            data = data.prefetch(buffer_size=100*batch_size)

        elif mode == 'testing':
            data = data.map(self._parse_function_test, num_parallel_calls=8)
            data = data.prefetch(buffer_size=100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." %(mode))

        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        data = data.batch(batch_size)
        self.data = data


    def _read_path_file(self):
        """ Read the contents of the path file and store it in the list."""
        self.img_paths = []
        self.labels = []

        with open(self.path_file, 'r') as fr:
            lines = fr.readlines()

            for line in lines:
                items = line.rstrip().split("\t")
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))


    def _shuffle_lists(self):
        """ Co-joined shuffling the list of paths and labels. """
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)

        self.img_paths = []
        self.labels = []

        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])


    def _parse_function_train(self, filename, label):
        """ Input parser for the samples of the training set. """
        #convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        #load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224,224])

        #image centering
        #img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
        #img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)

        #RGB -> BGR
        #img_bgr = img_centered[:, :, ::-1]
        img_bgr = img_resized[:, :, ::-1]

        return img_bgr, one_hot


    def _parse_function_test(self, filename, label):
        """ Input parser for the samples of the tesing set. """
        #convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        #load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224,224])

        #image centering
        #img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
        #img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)

        #RGB -> BGR
        #img_bgr = img_centered[:, :, ::-1]
        img_bgr = img_resized[:, :, ::-1]

        return img_bgr, one_hot

