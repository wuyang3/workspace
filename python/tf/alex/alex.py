# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:56:20 2017

@author: wduan
"""

import tensorflow as tf
import numpy as np

def alex(i, layer, reuse=False):
    # Alexnet weights preparartion
    variable_data = np.load(
        "/home/wuyang/workspace/python/tf/alex/bvlc_alexnet.npy",
        encoding='bytes').item()
    conv1_preW = tf.constant(variable_data["conv1"][0])
    conv1_preb = tf.constant(variable_data["conv1"][1])
    conv2_preW = tf.constant(variable_data["conv2"][0])
    conv2_preb = tf.constant(variable_data["conv2"][1])
    conv3_preW = tf.constant(variable_data["conv3"][0])
    conv3_preb = tf.constant(variable_data["conv3"][1])
    conv4_preW = tf.constant(variable_data["conv4"][0])
    conv4_preb = tf.constant(variable_data["conv4"][1])
    conv5_preW = tf.constant(variable_data["conv5"][0])
    conv5_preb = tf.constant(variable_data["conv5"][1])
    fc6_preW = tf.constant(variable_data["fc6"][0])
    fc6_preb = tf.constant(variable_data["fc6"][1])
    fc7_preW = tf.constant(variable_data["fc7"][0])
    fc7_preb = tf.constant(variable_data["fc7"][1])
    fc8_preW = tf.constant(variable_data["fc8"][0])
    fc8_preb = tf.constant(variable_data["fc8"][1])

    with tf.variable_scope('alex', reuse=reuse):
        shape = i.get_shape().as_list()
        assert len(shape) == 4, (
            'input is not batch rgb image'
        )
        if shape[2]!=227:
            i = tf.image.resize_images(i, [227, 227])
        # Convolution 1
        # Keep in mind that you need to use tf.get_variable. you can use
        # tf.constant on the numpy initial value and then pass the tensor as
        # initializer. Use tf.variable_scope since tf.get_variable method will
        # ignore tf.name_scope
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights', initializer=conv1_preW)
            biases = tf.get_variable('biases', initializer=conv1_preb)
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
            kernel = tf.get_variable('weights', initializer=conv2_preW)
            biases = tf.get_variable('biases', initializer=conv2_preb)

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
            kernel = tf.get_variable('weights', initializer=conv3_preW)
            biases = tf.get_variable('biases', initializer=conv3_preb)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(conv_bias)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.get_variable('weights', initializer=conv4_preW)
            biases = tf.get_variable('biases', initializer=conv4_preb)

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
            kernel = tf.get_variable('weights', initializer=conv5_preW)
            biases = tf.get_variable('biases', initializer=conv5_preb)

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

        shape = pool5.get_shape().as_list()
        size = shape[1]*shape[2]*shape[3]
        pool5_reshape = tf.reshape(pool5, [-1, size], name='flattened')

        assert layer in [5, 6, 7, 8, 9], (
            'layer index out of range.'
        )
        if layer==5:
             activations = pool5_reshape

        elif layer==6:
            # Fully connected 6
            with tf.variable_scope('fc6'):
                weights = tf.get_variable(initializer=fc6_preW, name='weights')
                bias = tf.get_variable(initializer=fc6_preb, name='biases')
                activations = tf.nn.relu_layer(pool5_reshape,
                                               weights, bias, name='relu')
        elif layer==7:
            with tf.variable_scope('fc6'):
                weights = tf.get_variable(initializer=fc6_preW, name='weights')
                bias = tf.get_variable(initializer=fc6_preb, name='biases')
                fc6 = tf.nn.relu_layer(pool5_reshape,
                                       weights, bias, name='relu')
            # Fully connected 7
            with tf.variable_scope('fc7'):
                weights = tf.get_variable(initializer=fc7_preW, name='weights')
                bias = tf.get_variable(initializer=fc7_preb, name='biases')
                activations = tf.nn.relu_layer(fc6, weights, bias, name='relu')

        elif layer==8:
            with tf.variable_scope('fc6'):
                weights = tf.get_variable(initializer=fc6_preW, name='weights')
                bias = tf.get_variable(initializer=fc6_preb, name='biases')
                fc6 = tf.nn.relu_layer(pool5_reshape,
                                       weights, bias, name='relu')
            # Fully connected 7
            with tf.variable_scope('fc7'):
                weights = tf.get_variable(initializer=fc7_preW, name='weights')
                bias = tf.get_variable(initializer=fc7_preb, name='biases')
                fc7 = tf.nn.relu_layer(fc6, weights, bias, name='relu')
            # Fully connected 8
            with tf.variable_scope('fc8'):
                weights = tf.get_variable(initializer=fc8_preW, name='weights')
                bias = tf.get_variable(initializer=fc8_preb, name='biases')
                # fc8 = tf.matmul(fc7, weights) + bias
                activations = tf.nn.xw_plus_b(fc7, weights, bias)

        else:
            with tf.variable_scope('fc6'):
                weights = tf.get_variable(initializer=fc6_preW, name='weights')
                bias = tf.get_variable(initializer=fc6_preb, name='biases')
                fc6 = tf.nn.relu_layer(pool5_reshape,
                                       weights, bias, name='relu')
            # Fully connected 7
            with tf.variable_scope('fc7'):
                weights = tf.get_variable(initializer=fc7_preW, name='weights')
                bias = tf.get_variable(initializer=fc7_preb, name='biases')
                fc7 = tf.nn.relu_layer(fc6, weights, bias, name='relu')
            # Fully connected 8
            with tf.variable_scope('fc8'):
                weights = tf.get_variable(initializer=fc8_preW, name='weights')
                bias = tf.get_variable(initializer=fc8_preb, name='biases')
                # fc8 = tf.matmul(fc7, weights) + bias
                fc8 = tf.nn.xw_plus_b(fc7, weights, bias)
            activations = tf.nn.softmax(fc8)

    return activations


def main():
    pixel_depth = 255.0
    resized_height = 227
    resized_width = 227
    num_channels = 3

    graph = tf.Graph()

    with graph.as_default():
        x = tf.placeholder(tf.uint8, [None, None, None, num_channels],
                           name='input')
        to_float = tf.cast(x, tf.float32)
        resized = tf.image.resize_images(to_float,
                                         tf.constant([resized_height, resized_width]))
        activations = alex(resized, 6, reuse=False)
        init = tf.global_variables_initializer()

    sess = tf.Session(graph=graph)
    sess.run(init)

    writer = tf.summary.FileWriter('../logs/alex/', graph=graph)
    writer.close()

    with graph.as_default():
        saver = tf.train.Saver()
        save_path = saver.save(sess, '../logs/alex/alex_vars')

if __name__ == '__main__':
    main()
