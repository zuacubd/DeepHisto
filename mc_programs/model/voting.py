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
from network.importgraph import ImportGraph

from tensorflow.data import Iterator
from data.image_generation import ImageDataGenerator
from config.config import config_class_level
from config.config import train_parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def voting_models():
    '''
        to-do: data IO imgs
    '''

    test_patch_gc_ngc_path = config_class_level['test_patch_gc_ngc_path']
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
    network_names = ["alexnet", "vgg16", "vgg19", "googlenet", "resnet50", "resnet101"]
    checkpoint_files = ['alexnet-train-test-main-model_epoch20-2019-03-27-03:47:06.ckpt',
                        'vgg16-Train_Test_model_step_epoch11-2019-03-28-12:26:19.ckpt',
                        'vgg19-Train_Test_model_step_epoch10-2019-03-28-12:19:57.ckpt',
                        'googlenet-train-test-main-model_epoch20-2019-03-27-00:17:01.ckpt',
                        'resnet50-Train_Test_model_step_epoch20-2019-03-27-03:20:42.ckpt',
                        'resnet101-train-test-main-model_epoch20-2019-03-27-17:18:08.ckpt'
                       ]
    output_path = 'output/checkpoint'
    models = []
    for idx in range(network_names):
        network_name = network_names[idx]
        model = ImportGraph(output_path + '/' + checkpoint_files[idx])
        models.append(model)

    #saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        #saver.restore(sess, checkpointfile)
        sess.run(testing_init_op)

        # Initialize all variables
        # sess.run(tf.global_variables_initializer())
        #cm_running_total = None
        for _ in range(test_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            #cm_numpy_array = sess.run(confusion, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.} )
            #if cm_running_total is None:
            #    cm_running_total = cm_numpy_array
            #else:
            #    cm_running_total += cm_numpy_array

            probs = sess.run(softmax, feed_dict={x: img_batch, keep_prob: 1})
            print (probs.shape)
            print (probs)
            print (probs[0,0])
        #Testing done for the current epoch
        #print (cm_running_total)
        #test_acc = np.diag(cm_running_total).sum()/cm_running_total.sum()
        #print ("Epoch: {}".format(epoch))
        #print("{} Testing Accuracy={:.4f}".format(datetime.now(), test_acc))

        #recall = np.diag(cm_running_total)/cm_running_total.sum(axis=0)
        #prec = np.diag(cm_running_total)/cm_running_total.sum(axis=1)

        #for i in range(len(recall)):
        #    print("Class: {} , Recall: {:.4f}, Precision: {:.4f}".format(i, recall[i], prec[i]))
        #    print("{} Saving checkpoint of model...".format(datetime.now()))
        #
        #for i, image in enumerate(imgs):
        #    img = cv2.imread(image)
        #    # Convert image to float32 and resize to (227x227)
        #    img = cv2.resize(img.astype(np.float32), (227,227))
        #    # Reshape as needed to feed into model
        #    img = img.reshape((1,227,227,3))
        #
        #    # Run the session and calculate the class probability
        #    probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
        #    print(i, probs, np.argmax(probs), labels[i])

        #sess.close()
