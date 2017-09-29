# -*- coding: utf-8 -*-
"""
Create a Siamese CNN for poke action predictions on depth image.
"""
import tensorflow as tf
import numpy as np
from poke import Poke

class PokeDepth(Poke):
    """
    Inheritate from the rgb poke prediction model. Inverse, forward, loss and train layers
    can be directly reused.
    """
    def __init__(self, train_layers=['inverse', 'forward', 'siamese'],
                 include_type=0, is_training=1,
                 learning_rate=0.001, epsilon=1e-08, lamb=0.1):
        self.include_type = include_type
        self.is_training = is_training
        self.i1 = tf.placeholder(tf.float32, [None, 227, 227, 3], 'i1')
        self.i2 = tf.placeholder(tf.float32, [None, 227, 227, 3], 'i2')
        self.u = tf.placeholder(tf.float32, [None, 4], 'u_c')
        self.p_labels = tf.placeholder(tf.int32, [None], 'p_d') # discrete
        self.theta_labels = tf.placeholder(tf.int32, [None], 'theta_d')
        self.l_labels = tf.placeholder(tf.int32, [None], 'l_d')


        tf.summary.histogram('u', self.u)
        tf.summary.histogram('p', self.p_labels)
        tf.summary.histogram('theta', self.theta_labels)
        tf.summary.histogram('l', self.l_labels)

        if self.include_type:
            self.type_labels = tf.placeholder(tf.int32, [None], 'type_d')
            self.labels = (self.p_labels, self.theta_labels, self.l_labels, self.type_labels)
            tf.summary.histogram('type', self.type_labels)
        else:
            self.labels = (self.p_labels, self.theta_labels, self.l_labels)

        with tf.variable_scope("siamese"):
            x1_intmd = self.siamese_depth(self.i1)
            self.x1 = tf.nn.relu(self.fc_layer(x1_intmd, 200, 'x'))
            tf.summary.histogram('states', self.x1)

        with tf.variable_scope("siamese", reuse=True):
            x2_intmd = self.siamese_depth(self.i2)
            self.x2 = tf.nn.relu(self.fc_layer(x2_intmd, 200, 'x'))
            tf.summary.histogram('states', self.x2)

        self.raw_predictions = self.inverse(self.x1, self.x2)
        self.classes = self.prediction(self.raw_predictions)
        self.x_pred = self.forward(self.x1, self.u)

        self.loss_i = self.loss_inverse(self.labels, self.raw_predictions)
        self.loss_f = self.loss_forward(self.x2, self.x_pred)
        self.loss = self.loss_joint(self.loss_i, self.loss_f, lamb)

        self.train_op = self.train(train_layers, learning_rate, epsilon, self.loss)
        self.merged = tf.summary.merge_all()

    def siamese_depth(self, i):
        conv1 = self.conv_bn_layer(i, [5, 5], [1, 16], 2,
                                   is_training=self.is_training, initializer_type=1,
                                   name='conv1')
        conv2 = self.conv_bn_layer(conv1, [5, 5], [16, 16], 2,
                                   is_training=self.is_training, initializer_type=1,
                                   name='conv2')
        conv3 = self.conv_bn_layer(conv2, [5, 5], [16, 32], 2,
                                   is_training=self.is_training, initializer_type=1,
                                   name='conv3')
        conv4 = self.conv_bn_layer(conv3, [5, 5], [32, 64], 2,
                                   is_training=self.is_training, initializer_type=1,
                                   name='conv4')
        conv5 = self.conv_bn_layer(conv4, [5, 5], [32, 64], 2,
                                   is_training=self.is_training, initializer_type=1,
                                   name='conv5')
        shape = conv5.get_shape().as_list()
        size = shape[1]*shape[2]*shap[3]
        activations = tf.reshape(conv5, [-1, size], name='x_flattened')
        return activations


    def conv_layer(self, inputs, field_size, channels_size, stride,
                   initializer_type, name, act_func=tf.nn.relu):
        inputs_shape = inputs.get_shape().as_list()
        assert inputs_shape[-1] == channels_size[0], (
            'Number of input channels does not match filter inputs channels.'
        )
        with tf.variable_scope(name):
            filter_size = field_size + channels_size
            bias_size = [channels_size[-1]]

            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)

            weights = tf.get_variable('W', filter_size, initializer=initializer)
            biases = tf.get_variable(
                'b', bias_size, initializer=tf.constant_initializer(.1))

            conv = tf.nn.conv2d(
                inputs, weights, strides=[1,stride,stride,1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            output = act_func(conv_bias)

        return output

    def conv_bn_layer(self, inputs, field_size, channels_size, stride, is_training,
                      initializer_type, name, act_func=tf.nn.relu):
        inputs_shape = inputs.get_shape().as_list()
        assert inputs_shape[-1] == channels_size[0], (
            'Number of input channels does not match filter inputs channels.'
        )
        with tf.variable_scope(name):
            filter_size = field_size + channels_size
            bias_size = [channels_size[-1]]

            if initializer_type:
                initializer = tf.contrib.layers.xavier_initializer()
            else:
                initializer = tf.truncated_normal_initializer(stddev=.1)

            weights = tf.get_variable('W', filter_size, initializer=initializer)
            biases = tf.get_variable(
                'b', bias_size, initializer=tf.constant_initializer(.1))

            conv = tf.nn.conv2d(
                inputs, weights, strides=[1,stride,stride,1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_normed = tf.contrib.layers.batch_norm(conv_bias, is_training=is_training)
            output = act_func(conv_normed)

        return output
