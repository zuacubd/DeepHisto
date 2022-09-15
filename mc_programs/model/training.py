import os
import sys
import numpy as np
import time

import tensorflow as tf
from network.googlenet import googlenet
from network.vggnet import vgg_16
from network.vggnet import vgg_19
from network.alexnet_v2 import Alexnet_v2
from network.resnet_v1 import resnet_v1_50
from network.resnet_v1_s import resnet_v1_101
from datetime import datetime

from tensorflow.data import Iterator
from data.image_generation import ImageDataGenerator
from config.config import config_class_level
from config.config import train_parameters
from config.config import imagenet_pretrain_weights

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def training_model(network_architecture):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #call to configuration files

    train_patch_gc_ngc_path = config_class_level['train_patch_gc_ngc_bg_path']
    test_patch_gc_ngc_path = config_class_level['test_patch_gc_ngc_bg_path']
    learning_rate = train_parameters['learning_rate']
    num_epochs = train_parameters['num_epochs']
    batch_size = train_parameters['batch_size']
    dropout_rate = train_parameters['dropout_rate']
    num_classes = train_parameters['num_classes']
    display_step = train_parameters['display_step']
    save_step = train_parameters['save_step']

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
        logits, endpoints = Alexnet_v2(x, num_classes=num_classes, is_training=True, dropout_keep_prob=keep_prob,\
                           spatial_squeeze=True, scope='alexnet_v2')

    elif network_architecture == "vgg16":
        logits, model = vgg_16(x, num_classes=num_classes, is_training=True, dropout_keep_prob=keep_prob, spatial_squeeze=True, \
               scope='vgg_16', fc_conv_padding='valid', global_pool=False)

    elif network_architecture == "vgg19":
        logits, model = vgg_19(x, num_classes=num_classes, is_training=True, dropout_keep_prob=keep_prob, spatial_squeeze=True, \
               scope='vgg_19', fc_conv_padding='valid', global_pool=False)

    elif network_architecture == "googlenet":
        logits, model = googlenet(x, keep_prob, num_classes, is_training =True, restore_logits = None, scope='')

    elif network_architecture == "resnet50":
        logits, endpoints = resnet_v1_50(x, num_classes, is_training=True, global_pool=True, output_stride=None, \
                                         reuse=None, scope='resnet_v1_50')

    elif network_architecture == "resnet101":
        logits, endpoints = resnet_v1_101(x, num_classes=num_classes, is_training=True, global_pool=True, output_stride=None, \
                                          spatial_squeeze=True, store_non_strided_activations=False, min_base_depth=8, \
                                          depth_multiplier=1, reuse=None, scope='resnet_v1_101')

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
        #recall = np.diag(confusion)/confusion.sum(axis=0)
        #precision = np.diag(confusion)/confusion.sum(axis=1)

    #Add the accuracy to the summary
    #tf.summary.scalar('Confusion', confusion)
    #tf.summary.scalar('Recall_0', recall[0])
    #tf.summary.scalar('Recall_1', recall[1])
    #tf.summary.scalar('Precision_0', precision[0])
    #tf.summary.scalar('Precision_0', precision[1])

   # Merge all summaries together
    merged_summary = tf.summary.merge_all()
    #print ("Done.")

    '''
	Train
    '''
    # to-do:
    newmodel=True
    filewriter_path = 'summary_log'
    dirname = network_architecture

    if not os.path.exists(filewriter_path + '/' + dirname):
        os.makedirs(filewriter_path + '/' + dirname)

    checkpoint_path = 'output/checkpoint/'
    if not os.path.exists(checkpoint_path + '/' + dirname):
        os.makedirs(checkpoint_path + '/' + dirname)
    checkpoint_path = checkpoint_path + '/' + dirname

    # Initialize the summary FileWriter
    writer = tf.summary.FileWriter(filewriter_path + '/' + dirname + '/')

    # Initialize a saver for store model checkpoints
    saver = tf.train.Saver()

    # Start Tensorflow session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

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
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            # Initialize iterator with the training dataset
            print(sess.run(training_init_op))

            for step in range(train_batches_per_epoch):

                # get next batch of data
                # ---------- Fetch Data ------------
                img_batch, label_batch = sess.run(next_batch)
                # ---------- Fetch Data End ------------

                # And run the training op
                # `---------- Train ------------
                sess.run(train_op, feed_dict={x: img_batch, y: label_batch, keep_prob: dropout_rate})
                # ---------- Train End ------------

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

                if step % save_step == 0:
                    checkpoint_name = os.path.join(checkpoint_path,
                                    dirname + '-train_test_model_step_epoch'+str(epoch+1) + '-' + \
                                    str(datetime.now()).replace(' ', '-').split('.')[0] + \
                                    '.ckpt')
                    # to-do:
                    #checkpoint_name=''
                    save_path = saver.save(sess, checkpoint_name)
                    print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


            # Testing the model on the entire testing set
            print("{} Start testing".format(datetime.now()))

            sess.run(testing_init_op)
            cm_running_total = None

            for _ in range(test_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch)
                cm_numpy_array = sess.run(confusion, feed_dict={x: img_batch,
                                                                y: label_batch,
                                                                keep_prob: 1.} )
                if cm_running_total is None:
                    cm_running_total = cm_numpy_array
                else:
                    cm_running_total += cm_numpy_array

            #Testing done for the current epoch
            print (cm_running_total)

            test_acc = np.diag(cm_running_total).sum()/cm_running_total.sum()
            print ("Epoch: {}".format(epoch))
            print("{} Testing Accuracy={:.4f}".format(datetime.now(), test_acc))

            recall = np.diag(cm_running_total)/cm_running_total.sum(axis=0)
            prec = np.diag(cm_running_total)/cm_running_total.sum(axis=1)

            for i in range(len(recall)):
                print("Class: {} , Recall: {:.4f}, Precision: {:.4f}".format(i, recall[i], prec[i]))

            print("{} Saving checkpoint of model...".format(datetime.now()))
            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                            dirname + '-Train-Test-Model_Epoch_'+str(epoch+1) + '-' + \
                                            str(datetime.now()).replace(' ', '-').split('.')[0] + \
                                            '.ckpt')
            # to-do:
            #checkpoint_name=''
            save_path = saver.save(sess, checkpoint_name)
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
