import os
import sys
import numpy as np
import time

import tensorflow as tf
from datetime import datetime

from network.googlenet import googlenet
from network.vgg16_trainable import Vgg16
from network.vgg19_trainable import Vgg19
from network.alexnet_v2 import Alexnet_v2
from network.resnet_v1 import resnet_v1_50
from network.resnet_v1_s import resnet_v1_101
from tensorflow.data import Iterator
from data.image_generation import ImageDataGenerator
from config.config import config_class_level
from config.config import train_parameters
from config.config import imagenet_pretrain_weights

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def testing_model(network_architecture):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #call to configuration files

    train_patch_gc_ngc_path = config_class_level['train_patch_gc_ngc_path']
    test_patch_gc_ngc_path = config_class_level['test_patch_gc_ngc_path']
    learning_rate = train_parameters['learning_rate']
    num_epochs = train_parameters['num_epochs']
    batch_size = train_parameters['batch_size']
    dropout_rate = train_parameters['dropout_rate']
    num_classes = train_parameters['num_classes']
    display_step = train_parameters['display_step']
    save_step = train_parameters['save_step']
    alexnet_pretrain = imagenet_pretrain_weights['alexnet']

    print ("Preparing dataset ...")
    '''
        Data IO: tf.data
    '''
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(train_patch_gc_ngc_path,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)

        te_data = ImageDataGenerator(test_patch_gc_ngc_path,
                                 mode='testing',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=False)


        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    testing_init_op = iterator.make_initializer(te_data.data)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
    print("Number of training batches: {0}".format(train_batches_per_epoch))

    test_batches_per_epoch = int(np.floor(te_data.data_size / batch_size))
    print("Number of testing batches: {0}".format(test_batches_per_epoch))


    '''
	Graph Definition
    '''
    #print ("Constructing graph ...")
    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    if network_architecture == "alexnet":
        #model = AlexNet(x, keep_prob, num_classes, skip=None)
        #logits = model.fc8
        logits, endpoints = Alexnet_v2(x, num_classes=num_classes, is_training=False, dropout_keep_prob=keep_prob,\
                           spatial_squeeze=True, scope='alexnet_v2')
        #logits = model['fc8']

    elif network_architecture == "vgg16":
        model = Vgg16(x, keep_prob, num_classes, skip=None)
        logits = model.fc8

    elif network_architecture == "vgg19":
        model = Vgg19(x, keep_prob, num_classes, skip=None)
        logits = model.fc8

    elif network_architecture == "googlenet":
        logits, model = googlenet(x, keep_prob, num_classes, is_training = False, restore_logits = None, scope='')

    elif network_architecture == "resnet50":
        logits, endpoints = resnet_v1_50(x, num_classes, is_training=False, global_pool=True, \
                                         output_stride=None, reuse=None, scope='resnet_v1_50')

    elif network_architecture == "resnet101":
        logits, endpoints = resnet_v1_101(x, num_classes=num_classes, is_training=False, global_pool=True, \
                                          output_stride=None,spatial_squeeze=True, store_non_strided_activations=False, \
                                          min_base_depth=8, depth_multiplier=1, reuse=None, scope='resnet_v1_101')
    # Link variable to model output
    #score = model.fc8 #change this according to the network
    score = logits

    # trainable variables
    var_list = [v for v in tf.trainable_variables()]

    # Op for calculating the loss
    # cross entropy loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=y))

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary
    for gradient, var in gradients:
        if gradient is None:
            gradient = 0
        tf.summary.histogram(var.name.replace(":", "_") + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name.replace(":", "_"), var)

    # Add the loss to summary
    tf.summary.scalar('Loss', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('Accuracy', accuracy)

    with tf.name_scope("confusion"):
        #correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        confusion = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(score, 1), num_classes)
    # Add the accuracy to the summary
    tf.summary.scalar('Confusion', confusion)


    # Evaluation op: Recall of the model per-class
    #with tf.name_scope("recall"):
    #    recall = [0.0] * num_classes
    #   recall_op = [[]] * num_classes
    #
    #    for k in range(num_classes):
    #        recall[k], recall_op[k] = tf.metrics.recall(labels=tf.cast(tf.equal(tf.argmax(y, 1), k), tf.float32), \
    #                                        predictions=tf.cast(tf.equal(tf.argmax(score, 1), k), tf.float32))

    # Add the accuracy to the summary
    #tf.summary.scalar('Recall_0', recall[0])
    #tf.summary.scalar('Recall_1', recall[1])

    # Evaluation op: Precision of the model per-class
    #with tf.name_scope("precision"):
    #    precision = [0.0] * 2
    #    precision_op = [[]] * 2
    #
    #    for k in range(2):
    #        precision[k], precision_op[k] = tf.metrics.precision(labels=tf.cast(tf.equal(tf.argmax(y, 1), k), tf.float32), \
    #                                                        predictions=tf.cast(tf.equal(tf.argmax(score, 1), k), tf.float32))
    # Add the accuracy to the summary
    #tf.summary.scalar('Precision_0', precision[0])
    #tf.summary.scalar('Precision_1', precision[1])

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()
    #print ("Done.")

    '''
	Train
    '''
    # to-do:
    newmodel=False
    #pre_model=''
    filewriter_path = 'summary_log'
    dirname = network_architecture

    if not os.path.exists(filewriter_path + '/' + dirname):
        os.makedirs(filewriter_path + '/' + dirname)

    if network_architecture == "alexnet":
        checkpoint_path = "output/checkpoint/alexnet-train-test-main-model_epoch20-2019-03-26-02:45:43.ckpt"

    elif network_architecture == "googlenet":
        checkpoint_path = "output/checkpoint/googlenet-TMPmodel_epoch50-2019-03-13-20:16:46.ckpt"

    elif network_architecture == "vgg16":
        checkpoint_path = "output/checkpoint/vgg16-train-test-main-model_epoch20-2019-03-27-06:28:24.ckpt"

    elif network_architecture == "vgg19":
        checkpoint_path = "output/checkpoint/vgg19-train-test-main-model_epoch20-2019-03-27-12:00:40.ckpt"

    elif network_architecture == "resnet50":
        checkpoint_path = "output/checkpoint/resnet50-train-test-main-model_epoch20-2019-03-26-02:45:43.ckpt"


    # Initialize the summary FileWriter
    writer = tf.summary.FileWriter(filewriter_path + '/' + dirname + '/')

    # Initialize a saver for store model checkpoints
    saver = tf.train.Saver()

    # Start Tensorflow session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the Imagenet pretrained weights into the non-trainable layer
        # Finetune the whole network
        if newmodel:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            #model.load_initial_weights(sess, trainablev=True, layersList=['logits'])
        else:
            saver.restore(sess, checkpoint_path)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

        # Loop over number of epochs
        sess.run(testing_init_op)
        test_acc = 0
        #test_recall_0 = 0
        #test_recall_1 = 0
        #test_prec_0 = 0
        #test_prec_1 = 0
        test_count = 0
        cm_running_total = None
        for _ in range(test_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1

            cm_numpy_array = sess.run(confusion, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.} )
            if cm_running_total is None:
                cm_running_total = cm_numpy_array
            else:
                cm_running_total += cm_numpy_array

            #recall = sess.run(recall_op, feed_dict={x: img_batch,
            #                                    y: label_batch,
            #                                    keep_prob: 1.})
            #test_recall_0 +=recall[0]
            #test_recall_1 +=recall[1]
            #
            #prec = sess.run(precision_op, feed_dict={x: img_batch,
            #                                    y: label_batch,
            #                                    keep_prob: 1.})
            #test_prec_0 +=prec[0]
            #test_prec_1 +=prec[1]

        test_acc /= test_count
        #test_recall_0 /= test_count
        #test_recall_1 /= test_count
        #test_prec_0 /= test_count
        #test_prec_1 /= test_count

        #Testing done
        print("{} Testing Accuracy={:.4f}".format(datetime.now(), test_acc))
#print("{} Testing Accuracy={:.4f}, Recall_0={:.4f}, and Recall_1={:.4f}".format(datetime.now(), test_acc, test_recall_0, test_recall_1))
        #print("{} Testing Accuracy={:.4f}, Precision_0={:.4f}, and Precision_1={:.4f}".format(datetime.now(), test_acc, test_prec_0, test_prec_1))
        print (cm_running_total)
        recall = np.diag(cm_running_total)/cm_running_total.sum(axis=0)
        prec = np.diag(cm_running_total)/cm_running_total.sum(axis=1)
        for i in range(len(recall)):
            print("Class: {} , Recall: {:.4f}, Precision: {:.4f}".format(i, recall[i], prec[i]))
        #print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        #checkpoint_name = os.path.join(checkpoint_path,
        #                                dirname + '-train-test-main-model_epoch'+str(epoch+1) + '-' + \
        #                                str(datetime.now()).replace(' ', '-').split('.')[0] + \
        #                                '.ckpt')
        # to-do:
        #checkpoint_name=''
        #save_path = saver.save(sess, checkpoint_name)
        #print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
