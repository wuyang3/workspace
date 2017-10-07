# -*- coding: utf-8 -*-
"""
Based on the result of denoising autoencoder, a recurrent version is trained to disentangle
the state representation
"""
import sys
sys.path.insert(0, '/home/wuyang/workspace/python/tf/alex/')
from alex import alex
import numpy as np
import tensorflow as tf
from poke_autoencoder import ConvAE

class PokeAEFFRNN(ConvAE):
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 in_channels=3, corrupted=0, is_training=True, lstm=0, vae=0):
        self.in_channels = in_channels
        self.is_training = is_training
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        self.u1 = tf.zeros([batch_size, 4], tf.float32)
        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)
        tf.summary.histogram('u2', self.u2)
        tf.summary.histogram('u3', self.u3)

        if corrupted:
            self.i1_corrupted = self.corrupt(self.i1, 0.05)
            tf.summary.image('image1_corrupted', self.i1_corrupted)
            self.encoding, self.feature_map_shape = self.encode(
                self.i1_corrupted, False)
        else:
            self.encoding, self.feature_map_shape = self.encode(self.i1, False)

        if lstm:
            cell_states = hidden_states = tf.zeros_like(self.encoding)
            init_layer1_state = tf.contrib.rnn.LSTMStateTuple(cell_states, hidden_states)
            init_layer2_state = tf.contrib.rnn.LSTMStateTuple(cell_states, hidden_states)
            init_tuple_states = tuple([init_layer1_state, init_layer2_state])
        else:
            init_tuple_states = tuple([tf.zeros_like(self.encoding), tf.zeros_like(self.encoding)])

        self.transit_1, current_state = self.Reccurent_state_transit(
            [self.encoding], [self.u1], init_tuple_states, lstm, vae, False)
        if vae:
            self.transit_1, mean_1, logss_1 = self.hidden_sample(self.transit_1, 'sampling0')
        self.decoding_1 = self.decode(self.transit_1[0], self.feature_map_shape, False)

        self.encoding_2, _ = self.encode(self.decoding_1, True)
        self.transit_2, current_state_2 = self.Reccurent_state_transit(
            [self.encoding_2], [self.u2], current_state, lstm, vae, True)
        if vae:
            self.transit_2, mean_2, logss_2 = self.hidden_sample(self.transit_2, 'sampling1')
        self.decoding_2 = self.decode(self.transit_2[0], self.feature_map_shape, True)

        self.encoding_3, _ = self.encode(self.decoding_2, True)
        self.transit_3, current_state_3 = self.Reccurent_state_transit(
            [self.encoding_3], [self.u3], current_state_2, lstm, vae, True)
        if vae:
            self.transit_3, mean_3, logss_3 = self.hidden_sample(self.transit_3, 'sampling2')
        self.decoding_3 = self.decode(self.transit_3[0], self.feature_map_shape, True)
        tf.summary.image('image_rec1', self.decoding_1)
        tf.summary.image('image_rec2', self.decoding_2)
        tf.summary.image('image_rec3', self.decoding_3)

        if vae:
            self.loss = self.loss_vae_function(self.i1, self.i2, self.i3,
                                               self.decoding_1, self.decoding_2, self.decoding_3,
                                               mean_1+mean_2+mean_3, logss_1+logss_2+logss_3)
        else:
            self.loss = self.loss_function(self.i1, self.i2, self.i3,
                                           self.decoding_1, self.decoding_2, self.decoding_3)

        self.train_op = self.train(self.loss, learning_rate, epsilon)
        self.merged = tf.summary.merge_all()

    def encode(self, img, reuse):
        with tf.variable_scope('encoder', reuse=reuse):
            conv2 = self.conv_layer(
                img, [5, 5], [self.in_channels, self.in_channels],
                stride=1, initializer_type=1, name='conv2')
            conv3 = self.conv_bn_layer(
                conv2, [5, 5], [self.in_channels, 64], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv3')
            conv4 = self.conv_bn_layer(
                conv3, [5, 5], [64, 128], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv4')
            conv5 = self.conv_bn_layer(
                conv4, [5, 5], [128, 256], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv5')
            shape = conv5.get_shape().as_list()
            feature_map_size = shape[1]*shape[2]*shape[3]
            conv5_flat = tf.reshape(
                conv5, [-1, feature_map_size], 'conv5_flat')
            fc6 = self.fc_bn_layer(conv5_flat, 1024, is_training=self.is_training,
                                initializer_type=1, name='fc6')
        return fc6, shape

    def Reccurent_state_transit(self, encoding_series, input_series,
                                init_tuple_states, lstm, vae, reuse):
        batch_size, hidden_size = encoding_series[0].get_shape().as_list()
        assert len(encoding_series)==len(input_series), (
            'input series and encoding series not matching'
        )
        rnn_input_series = []
        for i in range(len(encoding_series)):
            rnn_input_series.append(tf.concat([encoding_series[i], input_series[i]], 1))
        if lstm:
            with tf.variable_scope('LSTM', reuse=reuse):
                cells = [tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True),
                         tf.contrib.rnn.LSTMCell(hidden_size, state_is_tuple=True)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                state_series, current_states = tf.contrib.rnn.static_rnn(
                    cell, rnn_input_series, init_tuple_states)
                if vae:
                    statistics_series = []
                    for i, item in enumerate(state_series):
                        statistics = self.fc_no_activation(
                            item, hidden_size, initializer_type=1, name='iden%d'%(i))
                        statistics_series.append(statistics)
                    state_series = statistics_series

        else:
            with tf.variable_scope('RNN', reuse=reuse):
                cells = [tf.contrib.rnn.BasicRNNCell(hidden_size),
                         tf.contrib.rnn.BasicRNNCell(hidden_size)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                state_series, current_states = tf.contrib.rnn.static_rnn(
                    cell, rnn_input_series, init_tuple_states)
                if vae:
                    statistics_series = []
                    for i, item in enumerate(state_series):
                        statistics = self.fc_no_activation(
                            item, hidden_size, initializer_type=1, name='iden%d'%(i))
                        statistics_series.append(statistics)
                    state_series = statistics_series

        return state_series, current_states

    def hidden_sample(self, statistics_series, name):
        assert type(statistics_series)==list, (
            'Input is not a list of transitted tensors.'
        )
        sampling = []
        mean = []
        logss = []
        with tf.variable_scope(name):
            for item in statistics_series:
                z_mean, z_log_sigma_sq = tf.split(item, 2, 1)
                z = self.hidden_code_sample(z_mean, z_log_sigma_sq, 'sampling')
                sampling.append(z)
                tf.summary.histogram('z', z)

                mean.append(z_mean)
                logss.append(z_log_sigma_sq)
        return sampling, mean, logss

    def decode(self, code, shape, reuse):
        with tf.variable_scope('decoder', reuse=reuse):
            feature_map_size = shape[1]*shape[2]*shape[3]
            fc9 = self.fc_bn_layer(code, feature_map_size, is_training=self.is_training,
                                initializer_type=1, name='fc9')
            deconv5 = tf.reshape(fc9, [shape[0], shape[1], shape[2], shape[3]])
            upsampling4 = self.upsample(
                deconv5, stride=2, name='upsampling4', mode='ZEROS')
            deconv4 = self.deconv_bn_layer(
                upsampling4, [5, 5], [256, 128], is_training=self.is_training,
                initializer_type=1, name='deconv4')
            upsampling3 = self.upsample(
                deconv4, stride=2, name='upsampling3', mode='ZEROS')
            deconv3 = self.deconv_bn_layer(
                upsampling3, [5, 5], [128, 64], is_training=self.is_training,
                initializer_type=1, name='deconv3')
            upsampling2 = self.upsample(
                deconv3, stride=2, name='upsampling2', mode='ZEROS')
            deconv2 = self.deconv_bn_layer(
                upsampling2, [5, 5], [64, self.in_channels], is_training=self.is_training,
                initializer_type=1, name='deconv2')
            deconv1 = self.deconv_layer(deconv2, [5, 5], [self.in_channels, self.in_channels],
                                        initializer_type=1, name='deconv1')
        return deconv1

    def loss_function(self, i1, i2, i3, i_rec1, i_rec2, i_rec3):
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(
                tf.square(i1 - i_rec1)+tf.square(i2 - i_rec2)+tf.square(i3 - i_rec3))
        tf.summary.scalar('loss_rec', loss)
        return loss

    def loss_vae_function(self, i1, i2, i3, i_rec1, i_rec2, i_rec3, mean, logss):
        assert len(mean)==len(logss), (
            'Steps of mean and variance not match'
        )
        with tf.variable_scope('loss'):
            with tf.variable_scope('rec'):
                loss_rec = tf.reduce_sum(
                    tf.square(i1 - i_rec1)+tf.square(i2 - i_rec2)+tf.square(i3 - i_rec3), [1, 2, 3])
            with tf.variable_scope('kl'):
                loss_kl = 0
                for j in range(len(mean)):
                    loss_kl+= -0.5*tf.reduce_sum(1+logss[j]-tf.square(mean[j])-tf.exp(logss[j]), 1)

            # loss averaged over batches thus different from pixel wise diferrence.
            loss = tf.reduce_mean(loss_rec + loss_kl)
            tf.summary.scalar('loss_vae', loss)
            tf.summary.scalar('loss_rec', tf.reduce_mean(loss_rec))
            tf.summary.scalar('loss_kl', tf.reduce_mean(loss_kl))
        return loss

    def train(self, loss, learning_rate, epsilon):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)
        return train_op

class PokeAERNN(ConvAE):
    """
    Autoencoder with recurrent hidden state. Loss is simply mean squared error without variational
    bound loss. Can toggle between plain rnn and lstm mode, denoising or not denoising mode.
    Mind that BN introduced in ENC and DEC makes a difference.
    """
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0, is_training=True, lstm=0):
        self.in_channels = in_channels
        self.is_training = is_training
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        self.u1 = tf.zeros([batch_size, 4], tf.float32)
        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)
        tf.summary.histogram('u2', self.u2)
        tf.summary.histogram('u3', self.u3)

        if corrupted:
            self.i1_corrupted = self.corrupt(self.i1, 0.05)
            tf.summary.image('image1_corrupted', self.i1_corrupted)
            self.identity, self.pose, self.feature_map_shape = self.encode(
                self.i1_corrupted, split_size)
        else:
            self.identity, self.pose, self.feature_map_shape = self.encode(self.i1, split_size)

        self.transit_1, self.transit_2, self.transit_3 = self.Reccurent_state_transit(
            [self.u1, self.u2, self.u3], self.pose, lstm)

        self.decoding_1 = self.decode(self.identity, self.transit_1, self.feature_map_shape, False)
        self.decoding_2 = self.decode(self.identity, self.transit_2, self.feature_map_shape, True)
        self.decoding_3 = self.decode(self.identity, self.transit_3, self.feature_map_shape, True)
        tf.summary.image('image_rec1', self.decoding_1)
        tf.summary.image('image_rec2', self.decoding_2)
        tf.summary.image('image_rec3', self.decoding_3)

        self.loss = self.loss_function(self.i1, self.i2, self.i3,
                                       self.decoding_1, self.decoding_2, self.decoding_3)
        tf.summary.scalar('loss_rec', self.loss)

        self.train_op = self.train(self.loss, learning_rate, epsilon)

        self.merged = tf.summary.merge_all()

    def encode(self, img, split_size):
        with tf.variable_scope('encoder'):
            conv3 = self.conv_layer(
                img, [5, 5], [self.in_channels, 64], stride=2, initializer_type=1, name='conv3')
            conv4 = self.conv_bn_layer(
            #conv4 = self.conv_layer(
                conv3, [5, 5], [64, 128], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv4')
            conv5 = self.conv_bn_layer(
            #conv5 = self.conv_layer(
                conv4, [5, 5], [128, 256], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv5')
            shape = conv5.get_shape().as_list()
            feature_map_size = shape[1]*shape[2]*shape[3]
            conv5_flat = tf.reshape(
                conv5, [-1, feature_map_size], 'conv5_flat')
            fc6 = self.fc_bn_layer(conv5_flat, 1024, is_training=self.is_training,
            #fc6 = self.fc_layer(conv5_flat, 1024,
                                initializer_type=1, name='fc6')
            identity, pose = tf.split(fc6,
                                      num_or_size_splits=[1024-split_size, split_size],
                                      axis=1)
        return identity, pose, shape

    def Reccurent_state_transit(self, input_series, hidden_states, lstm):
        batch_size, split_size = hidden_states.get_shape().as_list()
        if lstm:
            with tf.variable_scope('LSTM'):
                cell_states = tf.zeros_like(hidden_states)
                # LSTMStateTuple(cell_state, hidden_state)
                init_layer1_state = tf.contrib.rnn.LSTMStateTuple(cell_states, hidden_states)
                init_layer2_state = tf.contrib.rnn.LSTMStateTuple(cell_states, cell_states)
                # list to tuple: tuple of two for each layer.
                init_tupe_states = tuple([init_layer1_state, init_layer2_state])
                # use different cells!
                # First layer: input_size=split_size+input:4, cell_size=hidden_size=split_size
                # Second layer: input_size=hidden_size+input:hidden_size(output from previous layer)
                # So sizes are different. If you use one cell, variables are shared automaticly.
                # which will bring error.
                # The second layer has: cell state(initialized), hidden state(initialized)
                # and input from the previous layer(hidden state).
                cells = [tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True),
                         tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True)]
                # accept a list of cell: [cell, cell]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                state_series, current_states = tf.contrib.rnn.static_rnn(
                    cell, input_series, init_tupe_states)
        else:
            with tf.variable_scope('RNN'):
                """
                LSTMCell: state_is_tuple=True accepts and returns two tuples of cell/hidden states.
                MultiRNNCell: state_is_tuple=True, accepts and returns n tuples of states. n is the
                number of cells(layer of cells).
                RNNCell: does not have cell states nor equivalent method as LSTMStateTuple
                """
                hidden_states_2 = tf.zeros_like(hidden_states)
                init_tuple_states = tuple([hidden_states, hidden_states_2])
                cells = [tf.contrib.rnn.BasicRNNCell(split_size),
                         tf.contrib.rnn.BasicRNNCell(split_size)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                state_series, current_states = tf.contrib.rnn.static_rnn(
                    cell, input_series, init_tuple_states)
        return state_series

    def decode(self, identity, transit, shape, reuse):
        with tf.variable_scope('decoder', reuse=reuse):
            feature_map_size = shape[1]*shape[2]*shape[3]
            code = tf.concat([identity, transit], axis=1)
            fc9 = self.fc_bn_layer(code, feature_map_size, is_training=self.is_training,
            #fc9 = self.fc_layer(code, feature_map_size,
                                initializer_type=1, name='fc9')
            deconv5 = tf.reshape(fc9, [shape[0], shape[1], shape[2], shape[3]])
            upsampling4 = self.upsample(
                deconv5, stride=2, name='upsampling4', mode='ZEROS')
            deconv4 = self.deconv_bn_layer(
            #deconv4 = self.deconv_layer(
                upsampling4, [5, 5], [256, 128], is_training=self.is_training,
                initializer_type=1, name='deconv4')
            upsampling3 = self.upsample(
                deconv4, stride=2, name='upsampling3', mode='ZEROS')
            deconv3 = self.deconv_bn_layer(
            #deconv3 = self.deconv_layer(
                upsampling3, [5, 5], [128, 64], is_training=self.is_training,
                initializer_type=1, name='deconv3')
            upsampling2 = self.upsample(
                deconv3, stride=2, name='upsampling2', mode='ZEROS')
            deconv2 = self.deconv_bn_layer(
            #deconv2 = self.deconv_layer(
                upsampling2, [5, 5], [64, self.in_channels], is_training=self.is_training,
                initializer_type=1, name='deconv2')
            deconv1 = self.deconv_layer(deconv2, [5, 5], [self.in_channels, self.in_channels],
                                        initializer_type=1, name='deconv1')
        return deconv1

    def loss_function(self, i1, i2, i3, i_rec1, i_rec2, i_rec3):
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(
                tf.square(i1 - i_rec1)+tf.square(i2 - i_rec2)+tf.square(i3 - i_rec3))
        return loss

    def train(self, loss, learning_rate, epsilon):
        """
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            train_op = optimizer.minimize(loss)
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)
        return train_op


class PokeVAERNN(ConvAE):
    """
    Auto-encoder model for distangling state representations.
    For 240X240 images, it will be troublesome to upsample.
    240 -> 120 -> 60 -> 30 -> 15 -> 8 -> 16 -> 32 ...
    From 15X15, convolution with stride 2 and 5X5 filter will lead to:
    6X6 (valid padding ceil((15-5+1)/2)=6)
    8X8 (same padding ceil(15/2)=8, SAME padding 2 to the left and right)
    """
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0, is_training=True, lstm=0):
        self.in_channels = in_channels
        self.is_training = is_training
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        self.u1 = tf.zeros([batch_size, 4], tf.float32)
        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)

        if corrupted:
            self.i1_corrupted = self.corrupt(self.i1, 0.05)
            tf.summary.image('image1_corrupted', self.i1_corrupted)
            self.identity, self.pose, self.feature_map_shape = self.encode(
                self.i1_corrupted, split_size)
        else:
            self.identity, self.pose, self.feature_map_shape = self.encode(self.i1, split_size)

        self.transit_series = self.Reccurent_state_transit(
            [self.u1, self.u2, self.u3], self.pose, lstm)

        self.sampling, self.mean, self.logss = self.hidden_sample(self.identity, self.transit_series)

        self.decoding_1 = self.decode(self.sampling[0], self.feature_map_shape, False)
        self.decoding_2 = self.decode(self.sampling[1], self.feature_map_shape, True)
        self.decoding_3 = self.decode(self.sampling[2], self.feature_map_shape, True)
        tf.summary.image('image_rec1', self.decoding_1)
        tf.summary.image('image_rec2', self.decoding_2)
        tf.summary.image('image_rec3', self.decoding_3)

        self.loss = self.loss_function(self.i1, self.i2, self.i3,
                                       self.decoding_1, self.decoding_2, self.decoding_3,
                                       self.mean, self.logss)

        self.train_op = self.train(self.loss, learning_rate, epsilon)

        self.merged = tf.summary.merge_all()

    def encode(self, img, split_size):
        with tf.variable_scope('encoder'):
            conv2 = self.conv_layer(
                img, [5, 5], [self.in_channels, self.in_channels], stride=1,
                initializer_type=1, name='conv2')
            conv3 = self.conv_bn_layer(
                conv2, [5, 5], [self.in_channels, 64], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv3')
            conv4 = self.conv_bn_layer(
                conv3, [5, 5], [64, 128], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv4')
            conv5 = self.conv_bn_layer(
                conv4, [5, 5], [128, 256], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv5')
            shape = conv5.get_shape().as_list()
            feature_map_size = shape[1]*shape[2]*shape[3]
            conv5_flat = tf.reshape(
                conv5, [-1, feature_map_size], 'conv5_flat')
            fc6 = self.fc_bn_layer(conv5_flat, 1024, is_training=self.is_training,
                                   initializer_type=1, name='fc6')
            identity, pose = tf.split(fc6,
                                      num_or_size_splits=[1024-split_size, split_size],
                                      axis=1)
            identity = self.fc_no_activation(identity, 1024-split_size,
                                             initializer_type=1, name='iden')
        return identity, pose, shape

    def Reccurent_state_transit(self, input_series, hidden_states, lstm):
        batch_size, split_size = hidden_states.get_shape().as_list()
        transit_series = []

        if lstm:
            with tf.variable_scope('LSTM'):
                cell_states = tf.zeros_like(hidden_states)
                init_layer1_state = tf.contrib.rnn.LSTMStateTuple(cell_states, hidden_states)
                init_layer2_state = tf.contrib.rnn.LSTMStateTuple(cell_states, cell_states)
                init_tupe_states = tuple([init_layer1_state, init_layer2_state])

                cells = [tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True),
                         tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                state_series, current_states = tf.contrib.rnn.static_rnn(
                    cell, input_series, init_tupe_states)

                for j, item in enumerate(state_series):
                    transit = self.fc_no_activation(item, split_size,
                                                    initializer_type=1, name='transit%d'%j)
                    transit_series.append(transit)

        else:
            with tf.variable_scope('RNN'):
                hidden_states_2 = tf.zeros_like(hidden_states)
                init_tuple_states = tuple([hidden_states, hidden_states_2])
                cells = [tf.contrib.rnn.BasicRNNCell(split_size),
                         tf.contrib.rnn.BasicRNNCell(split_size)]
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                state_series, current_states = tf.contrib.rnn.static_rnn(
                    cell, input_series, init_tuple_states)

                for j, item in enumerate(state_series):
                    transit = self.fc_no_activation(item, split_size,
                                                    initializer_type=1, name='transit%d'%j)
                    transit_series.append(transit)
        return transit_series

    def hidden_sample(self, identity, transit_series):
        assert type(transit_series)==list, (
            'Input is not a list of transitted tensors.'
        )
        sampling = []
        mean = []
        logss = []
        with tf.variable_scope('sampling'):
            z_mean_i, z_log_sigma_sq_i = tf.split(identity, 2, 1)
            mean.append(z_mean_i)
            logss.append(z_log_sigma_sq_i)
            z_i = self.hidden_code_sample(z_mean_i, z_log_sigma_sq_i, 'sampling_i')
            for j, item in enumerate(transit_series):
                z_mean_t, z_log_sigma_sq_t = tf.split(item, 2, 1)
                z_t = self.hidden_code_sample(z_mean_t, z_log_sigma_sq_t, 'sampling_%d'%j)
                z = tf.concat([z_i, z_t], 1)
                sampling.append(z)
                tf.summary.histogram('z%d'%j, z)

                mean.append(z_mean_t)
                logss.append(z_log_sigma_sq_t)
        return sampling, mean, logss

    def decode(self, code, shape, reuse):
        with tf.variable_scope('decoder', reuse=reuse):
            feature_map_size = shape[1]*shape[2]*shape[3]
            fc9 = self.fc_bn_layer(code, feature_map_size, is_training=self.is_training,
                                initializer_type=1, name='fc9')
            deconv5 = tf.reshape(fc9, [shape[0], shape[1], shape[2], shape[3]])
            upsampling4 = self.upsample(
                deconv5, stride=2, name='upsampling4', mode='ZEROS')
            deconv4 = self.deconv_bn_layer(
                upsampling4, [5, 5], [256, 128], is_training=self.is_training,
                initializer_type=1, name='deconv4')
            upsampling3 = self.upsample(
                deconv4, stride=2, name='upsampling3', mode='ZEROS')
            deconv3 = self.deconv_bn_layer(
                upsampling3, [5, 5], [128, 64], is_training=self.is_training,
                initializer_type=1, name='deconv3')
            upsampling2 = self.upsample(
                deconv3, stride=2, name='upsampling2', mode='ZEROS')
            deconv2 = self.deconv_bn_layer(
                upsampling2, [5, 5], [64, self.in_channels], is_training=self.is_training,
                initializer_type=1, name='deconv2')
            deconv1 = self.deconv_layer(deconv2, [5, 5], [self.in_channels, self.in_channels],
                                        initializer_type=1, name='deconv1')
        return deconv1

    def loss_function(self, i1, i2, i3, i_rec1, i_rec2, i_rec3, mean, logss):
        assert len(mean)==len(logss), (
            'Steps of mean and variance not match'
        )
        with tf.variable_scope('loss'):
            with tf.variable_scope('rec'):
                loss_rec = tf.reduce_sum(
                    tf.square(i1 - i_rec1)+tf.square(i2 - i_rec2)+tf.square(i3 - i_rec3), [1, 2, 3])
            with tf.variable_scope('kl'):
                loss_kl = 0
                for j in range(len(mean)):
                    loss_kl+= -0.5*tf.reduce_sum(1+logss[j]-tf.square(mean[j])-tf.exp(logss[j]), 1)

            # loss averaged over batches thus different from pixel wise diferrence.
            loss = tf.reduce_mean(loss_rec + loss_kl)
            tf.summary.scalar('loss_vae', loss)
            tf.summary.scalar('loss_rec', tf.reduce_mean(loss_rec))
            tf.summary.scalar('loss_kl', tf.reduce_mean(loss_kl))
        return loss

    def train(self, loss, learning_rate, epsilon):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)
        return train_op


class PokeVAERNNC(PokeVAERNN):
    """
    A compare function is trained in compare_function.py. We need to use that tiny
    network as a comparison function here. Or we can use alexnet as compare function,
    which is tricky since it requires input size of 227x227 and we need to resize all
    input and decoded images. Moreover, it is not clear whether alexnet is sensitive to
    changes of orientation and edges. On the other hand, compare function is trained to
    to regress on the pose of object. So the representation should be relatively more
    sensitive to the pose.
    """
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0, is_training=True, lstm=0):
        self.in_channels = in_channels
        self.is_training = is_training
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        self.u1 = tf.zeros([batch_size, 4], tf.float32)
        #self.alex = alex
        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.image('image3', self.i3)

        if corrupted:
            self.i1_corrupted = self.corrupt(self.i1, 0.05)
            tf.summary.image('image1_corrupted', self.i1_corrupted)
            self.identity, self.pose, self.feature_map_shape = self.encode(
                self.i1_corrupted, split_size)
        else:
            self.identity, self.pose, self.feature_map_shape = self.encode(self.i1, split_size)

        self.transit_series = self.Reccurent_state_transit(
            [self.u1, self.u2, self.u3], self.pose, lstm)

        self.sampling, mean, logss = self.hidden_sample(self.identity, self.transit_series)

        self.decoding_1 = self.decode(self.sampling[0], self.feature_map_shape, False)
        self.decoding_2 = self.decode(self.sampling[1], self.feature_map_shape, True)
        self.decoding_3 = self.decode(self.sampling[2], self.feature_map_shape, True)
        tf.summary.image('image_rec1', self.decoding_1)
        tf.summary.image('image_rec2', self.decoding_2)
        tf.summary.image('image_rec3', self.decoding_3)

        #i1_act = self.alex(self.i1, 6, False)
        #i1_rec_act = self.alex(self.decoding_1, 6, True)
        #i2_act = self.alex(self.i2, 6, True)
        #i2_rec_act = self.alex(self.decoding_2, 6, True)
        #i3_act = self.alex(self.i3, 6, True)
        #i3_rec_act = self.alex(self.decoding_3, 6, True)

        i1_act = self.regress(self.i1, False)
        i1_rec_act = self.regress(self.decoding_1, True)
        i2_act = self.regress(self.i2, True)
        i2_rec_act = self.regress(self.decoding_2, True)
        i3_act = self.regress(self.i3, True)
        i3_rec_act = self.regress(self.decoding_3, True)


        self.loss = self.loss_compare(self.i1, self.i2, self.i3,
                                      self.decoding_1, self.decoding_2, self.decoding_3,
                                      [i1_act, i2_act, i3_act],
                                      [i1_rec_act, i2_rec_act, i3_rec_act],
                                      mean, logss)

        self.train_op = self.partial_train(self.loss, learning_rate, epsilon)

        self.merged = tf.summary.merge_all()

    def regress(self, i, reuse):
        """
        Not trained. All beta, moving mean/variance will be filled from restoration.
        # BN: moving mean/variance are not trainable but needs to be restored.
        # So don't use tf.trainable_variables() when constructing var_list.
        """
        with tf.variable_scope('regression', reuse=reuse):
            conv1 = self.conv_layer(
                i, [5, 5], [self.in_channels, 32], stride=1, initializer_type=1, name='conv1')
            conv2 = self.conv_bn_layer(
                conv1, [5, 5], [32, 64], stride=2, is_training=False,
                initializer_type=1, name='conv2')
            conv3 = self.conv_bn_layer(
                conv2, [5, 5], [64, 128], stride=2, is_training=False,
                initializer_type=1, name='conv3')
            conv4 = self.conv_bn_layer(
                conv3, [5, 5], [128, 256], stride=2, is_training=False,
                initializer_type=1, name='conv4')
            shape = conv4.get_shape().as_list()
            size = shape[1]*shape[2]*shape[3]
            conv4_flat = tf.reshape(conv4, [-1, size], 'conv4_flat')
            fc5 = self.fc_bn_layer(
                conv4_flat, 512, is_training=False, initializer_type=1, name='fc5')
            #fc7 = self.fc_layer(fc5, 3, initializer_type=1, name='fc7')
            tf.summary.histogram('regression', fc5)
        return fc5

    def loss_compare(self, i1, i2, i3, i1_rec, i2_rec, i3_rec, act, act_rec, mean, logss):
        assert len(mean)==len(logss), (
            'Steps of mean and variance not match'
        )
        assert len(act)==len(act_rec), (
            'length of real and reconstruction not match'
        )
        with tf.variable_scope('loss'):
            with tf.variable_scope('rec'):
                loss_rec = tf.reduce_sum(
                    tf.square(i1 - i1_rec)+tf.square(i2 - i2_rec)+tf.square(i3 - i3_rec), [1, 2, 3])
            with tf.variable_scope('kl'):
                loss_kl = 0
                for j in range(len(mean)):
                    loss_kl+= -0.5*tf.reduce_sum(1+logss[j]-tf.square(mean[j])-tf.exp(logss[j]), 1)
            with tf.variable_scope('compare'):
                loss_comp = 0
                for j in range(len(act)):
                    loss_comp+=tf.reduce_sum(tf.square(act[j]-act_rec[j]), 1)

            #loss = tf.reduce_mean(0.5*loss_rec + loss_kl + loss_comp)
            loss = tf.reduce_mean(0.5*loss_rec + loss_kl)
            tf.summary.scalar('loss_vae', loss)
            tf.summary.scalar('loss_rec', tf.reduce_mean(loss_rec))
            tf.summary.scalar('loss_kl', tf.reduce_mean(loss_kl))
            tf.summary.scalar('loss_comp', tf.reduce_mean(loss_comp))
        return loss

    def partial_train(self, loss, learning_rate, epsilon):
        """
        By printing variables to train, you can actually find BatchNorm: beta NEEDS
        to be trained (they are trainable). But BatchNorm:moving_mean, BatchNorm:moving_variance
        are not trained/not trainable, but they are updated during training.
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        vars = tf.trainable_variables()
        train_vars = [v for v in vars
                      if 'regression' not in v.name]
        #for var in train_vars:
        #    print(var)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, var_list=train_vars)
        return train_op
