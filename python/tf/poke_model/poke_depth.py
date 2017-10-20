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
                 learning_rate=0.001, epsilon=1e-08, lamb=0.1,
                 include_type=0, corrupted=0, target_size=224):
        self.include_type = include_type
        self.i1 = tf.placeholder(tf.float32, [None, target_size, target_size, 1], 'i1')
        self.i2 = tf.placeholder(tf.float32, [None, target_size, target_size, 1], 'i2')
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

        if corrupted:
            self.i1 = self.corrupt(self.i1, 0.05)
            self.i2 = self.corrupt(self.i2, 0.05)

        with tf.variable_scope("siamese"):
            #x1_intmd = self.siamese_depth(self.i1)
            x1_intmd = self.siamese(self.i1, 1)
            self.x1 = tf.nn.relu(self.fc_layer(x1_intmd, 200, 'x'))
            tf.summary.histogram('states', self.x1)

        with tf.variable_scope("siamese", reuse=True):
            #x2_intmd = self.siamese_depth(self.i2)
            x2_intmd = self.siamese(self.i2, 1)
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

    def siamese(self,i,channels):
        initializer = tf.contrib.layers.xavier_initializer()
        # Convolution 1
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', shape=[11,11,channels,96], initializer=initializer)
            biases = tf.get_variable('biases', shape=[96], initializer=initializer)
            conv = tf.nn.conv2d(i, kernel, [1, 4, 4, 1], padding="SAME")
            conv_bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(conv_bias)
        # Local response normalization
        lrn1 = tf.nn.local_response_normalization(
            conv1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
        # Maxpool 1
        pool1 = tf.nn.max_pool(lrn1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool1')

        # Convolution 2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('weights', shape=[5,5,48,256], initializer=initializer)
            biases = tf.get_variable('biases', shape=[256], initializer=initializer)

            input_a, input_b = tf.split(value=pool1,
                                        num_or_size_splits=2,
                                        axis=3)
            kernel_a, kernel_b = tf.split(value=kernel,
                                          num_or_size_splits=2,
                                          axis=3)
            with tf.variable_scope('A'):
                conv_a = tf.nn.conv2d(
                    input_a, kernel_a, [1, 1, 1, 1], padding="SAME")
            with tf.variable_scope('B'):
                conv_b = tf.nn.conv2d(
                    input_b, kernel_b, [1, 1, 1, 1], padding="SAME")

            conv = tf.concat([conv_a, conv_b],3)
            conv_bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(conv_bias)
        # Local response normalization 2
        lrn2 = tf.nn.local_response_normalization(
            conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        # Maxpool 2
        pool2 = tf.nn.max_pool(lrn2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool2')

        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights', shape=[3,3,256,384], initializer=initializer)
            biases = tf.get_variable('biases', shape=[384], initializer=initializer)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(conv_bias)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', shape=[3,3,192,384], initializer=initializer)
            biases = tf.get_variable('biases', shape=[384], initializer=initializer)

            input_a, input_b = tf.split(value=conv3,
                                        num_or_size_splits=2,
                                        axis=3)
            kernel_a, kernel_b = tf.split(value=kernel,
                                          num_or_size_splits=2,
                                          axis=3)
            with tf.variable_scope('A'):
                conv_a = tf.nn.conv2d(
                    input_a, kernel_a, [1, 1, 1, 1], padding="SAME")
            with tf.variable_scope('B'):
                conv_b = tf.nn.conv2d(
                    input_b, kernel_b, [1, 1, 1, 1], padding="SAME")
            conv = tf.concat([conv_a, conv_b],3)
            conv_bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(conv_bias)

        with tf.variable_scope('conv5') as scope:
            kernel = tf.get_variable('weights', shape=[3,3,192,256], initializer=initializer)
            biases = tf.get_variable('biases', shape=[256], initializer=initializer)

            input_a, input_b = tf.split(value = conv4,
                                        num_or_size_splits=2,
                                        axis=3)
            kernel_a, kernel_b = tf.split(value = kernel,
                                          num_or_size_splits=2,
                                          axis=3)
            with tf.variable_scope('A'):
                conv_a = tf.nn.conv2d(
                    input_a, kernel_a, [1, 1, 1, 1], padding="SAME")
            with tf.variable_scope('B'):
                conv_b = tf.nn.conv2d(
                    input_b, kernel_b, [1, 1, 1, 1], padding="SAME")
            conv = tf.concat([conv_a, conv_b],3)
            conv_bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(conv_bias)
        # Maxpool 3 (at layer 5)
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool5')
        size = pool5.get_shape()[1].value * pool5.get_shape()[2].value * \
               pool5.get_shape()[3].value
        activations = tf.reshape(pool5, [-1, size], name='x_flattened')
        return activations

    def siamese_depth(self, i):
        conv1 = self.conv_layer(i, [5, 5], [1, 16], 2,
                                initializer_type=1, name='conv1')
        conv2 = self.conv_layer(conv1, [5, 5], [16, 32], 2,
                                initializer_type=1,
                                name='conv2')
        conv3 = self.conv_layer(conv2, [5, 5], [32, 64], 2,
                                initializer_type=1,
                                name='conv3')
        conv4 = self.conv_layer(conv3, [5, 5], [64, 128], 2,
                                initializer_type=1,
                                name='conv4')
        conv5 = self.conv_layer(conv4, [5, 5], [128, 256], 2,
                                initializer_type=1,
                                name='conv5')
        shape = conv5.get_shape().as_list()
        size = shape[1]*shape[2]*shape[3]
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

    def corrupt(self, inputs, prob):
        with tf.name_scope('corruption'):
            corrupted = tf.multiply(inputs, tf.cast(tf.random_uniform(shape=tf.shape(inputs),
                                                                      minval=0,
                                                                      maxval=2,
                                                                      dtype=tf.int32), tf.float32))
            corrupted_input = corrupted*prob + inputs*(1-prob)
        return corrupted_input

class PokeTotal(PokeDepth):
    """
    Inheritate from the rgb poke prediction model. Inverse, forward, loss and train layers
    can be directly reused.
    """
    def __init__(self, train_layers=['inverse', 'forward', 'siamese'],
                 learning_rate=0.001, epsilon=1e-08, lamb=0.1, include_type=0, target_size=224):
        self.include_type = include_type
        self.i1 = tf.placeholder(tf.float32, [None, target_size, target_size, 4], 'i1')
        self.i2 = tf.placeholder(tf.float32, [None, target_size, target_size, 4], 'i2')
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
            #x1_intmd = self.siamese_total(self.i1)
            x1_intmd = self.siamese(self.i1, 4)
            self.x1 = tf.nn.relu(self.fc_layer(x1_intmd, 200, 'x'))
            tf.summary.histogram('states', self.x1)

        with tf.variable_scope("siamese", reuse=True):
            #x2_intmd = self.siamese_total(self.i2)
            x2_intmd = self.siamese(self.i2, 4)
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

    def siamese_total(self, i):
        conv1 = self.conv_layer(i, [5, 5], [4, 16], 2,
                                initializer_type=1, name='conv1')
        conv2 = self.conv_layer(conv1, [5, 5], [16, 32], 2,
                                initializer_type=1,
                                name='conv2')
        conv3 = self.conv_layer(conv2, [5, 5], [32, 64], 2,
                                initializer_type=1,
                                name='conv3')
        conv4 = self.conv_layer(conv3, [5, 5], [64, 128], 2,
                                initializer_type=1,
                                name='conv4')
        conv5 = self.conv_layer(conv4, [5, 5], [128, 256], 2,
                                initializer_type=1,
                                name='conv5')
        shape = conv5.get_shape().as_list()
        size = shape[1]*shape[2]*shape[3]
        activations = tf.reshape(conv5, [-1, size], name='x_flattened')
        return activations
