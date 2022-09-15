'''
	Python 3.6
'''

#some basic imports and setups
import os
import cv2
import time
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from network.googlenet import googlenet
from network.vggnet import vgg_16
from network.vggnet import vgg_19
from network.alexnet_v2 import Alexnet_v2
from network.resnet_v1 import resnet_v1_50
from network.resnet_v1_s import resnet_v1_101

from tensorflow.data import Iterator
from data.image_generation import ImageDataGenerator
from config.config import config_class_level
from config.config import train_parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prediction_model(network_architecture):
    '''
        to-do: data IO imgs
    '''

    test_patch_gc_ngc_path = config_class_level['test_patch_gc_ngc_bg_path']
    num_classes = train_parameters['num_classes']
    batch_size = 1

    with tf.device('/cpu:0'):
        te_data = ImageDataGenerator(test_patch_gc_ngc_path,
                                     mode='testing',
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=False)
        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(te_data.data.output_types,
                                           te_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the iterators
    testing_init_op = iterator.make_initializer(te_data.data)
    test_batches_per_epoch = int(np.floor(te_data.data_size / batch_size))
    print("Number of testing batches: {0}".format(test_batches_per_epoch))

    '''
        Model Architecture Defintion
    '''

    #placeholder for input and dropout rate
    x = tf.placeholder(tf.float32, [1, 224, 224, 3])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    if network_architecture == "alexnet":
        logits, endpoints = Alexnet_v2(x, num_classes=num_classes, is_training=False, dropout_keep_prob=keep_prob,\
                                   spatial_squeeze=True, scope='alexnet_v2')

    elif network_architecture == "vgg16":
        logits, model = vgg_16(x, num_classes=num_classes, is_training=False, dropout_keep_prob=keep_prob, spatial_squeeze=True,\
                           scope='vgg_16', fc_conv_padding='valid', global_pool=False)

    elif network_architecture == "vgg19":
        logits, model = vgg_19(x, num_classes=num_classes, is_training=False, dropout_keep_prob=keep_prob, spatial_squeeze=True,\
                           scope='vgg_19', fc_conv_padding='valid', global_pool=False)

    elif network_architecture == "googlenet":
        logits, model = googlenet(x, keep_prob, num_classes, is_training=False, restore_logits = None, scope='')

    elif network_architecture == "resnet50":
        logits, endpoints = resnet_v1_50(x, num_classes=num_classes, is_training=False, global_pool=True, output_stride=None,\
                                     reuse=None, scope='resnet_v1_50')

    elif network_architecture == "resnet101":
        logits, endpoints = resnet_v1_101(x, num_classes=num_classes, is_training=False, global_pool=True, output_stride=None,\
                                      spatial_squeeze=True, store_non_strided_activations=False, min_base_depth=8,\
                                      depth_multiplier=1, reuse=None, scope='resnet_v1_101')

    #define activation of last layer as score
    score = logits

    #create op to calculate softmax 
    softmax = tf.nn.softmax(score)

    with tf.name_scope("confusion"):
        confusion = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(score, 1), num_classes)

    '''
        Predict
    '''

    # to-do
    network_names = ["alexnet", "vgg16", "vgg19", "googlenet", "resnet50", "resnet101"]
    checkpoint_files = ['alexnet-Train-Test-Model_Epoch_20-2019-04-05-20:45:04.ckpt',
                        'vgg16-Train-Test-Model_Epoch_20-2019-04-07-12:27:06.ckpt',
                        'vgg19-Train-Test-Model_Epoch_20-2019-04-10-22:42:17.ckpt',
                        'googlenet-Train-Test-Model_Epoch_20-2019-04-05-18:01:14.ckpt',
                        'resnet50-Train-Test-Model_Epoch_20-2019-04-06-00:20:06.ckpt',
                        'resnet101-Train-Test-Model_Epoch_20-2019-04-09-16:23:41.ckpt'
                        ]

    output_path = 'output/checkpoint'
    network_idx = network_names.index(network_architecture)
    checkpointfile = output_path + '/' + network_architecture + '/' + checkpoint_files[network_idx]

    prediction_file = 'output/prediction' + '/' + network_architecture + '.pred'

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        saver.restore(sess, checkpointfile)
        sess.run(testing_init_op)

        with open(prediction_file, 'w') as fw:
            header = 'Serial'
            for j in range(num_classes):
                    header = header + '\t' + 'Class-' + str(j)
            header = header + '\t' + 'Actual'
            fw.write(header + '\n')

            for batch_no in range(test_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch)
                probs = sess.run(softmax, feed_dict={x: img_batch, keep_prob: 1})

                for i in range(batch_size):
                    serial = batch_no * batch_size + i

                    line = str(serial)
                    for j in range(num_classes):
                        pred = probs[i, j]
                        line = line + '\t' + str(pred)
                    for k in range(len(label_batch[i, :])):
                        y = label_batch[i, k]
                        line = line + '\t' + str(y)
                    fw.write(line + '\n')

