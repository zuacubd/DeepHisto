# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def alexnet_v2_arg_scope(weight_decay=0.0005):
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      activation_fn=nn_ops.relu,
      biases_initializer=init_ops.constant_initializer(0.1),
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope([layers.conv2d], padding='SAME'):
      with arg_scope([layers_lib.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def Alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet_v2'):
  """AlexNet version 2.

  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=[end_points_collection]):
      net = layers.conv2d(
          inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
      net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = layers.conv2d(net, 192, [5, 5], scope='conv2')
      net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = layers.conv2d(net, 384, [3, 3], scope='conv3')
      net = layers.conv2d(net, 384, [3, 3], scope='conv4')
      net = layers.conv2d(net, 256, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with arg_scope(
          [layers.conv2d],
          weights_initializer=trunc_normal(0.005),
          biases_initializer=init_ops.constant_initializer(0.1)):
        net = layers.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
        net = layers_lib.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
        net = layers_lib.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        net = layers.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            biases_initializer=init_ops.zeros_initializer(),
            scope='fc8')

      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points

Alexnet_v2.default_image_size = 224