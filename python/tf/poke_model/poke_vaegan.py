# -*- coding: utf-8 -*-
"""
Create a customized auto-encoder with action unit at the bottomneck for learning distangled state representations. VAEGAN batch normalization included.
"""
import numpy as np
import tensorflow as tf
from poke_autoencoder import ConvAE


class PokeVAEGAN(ConvAE):
    """
    VAEGAN: reconstruction error replaced by high level discriminator activations.
    First layer of encoder, last layer of decoder(discriminator) and last layer of discriminator
    do not use batch normalization.
    In case of batch normalization, it is usually not applied in the first and last layer.
    You want to reconstruct to 0-255 and you want symmetric structure.
    """
    def __init__(self, learning_rate=1e-4, episilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0, is_training=True):
        self.in_channels = in_channels
        self.is_training = is_training
        self.lambda_gan = 0.001
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.u_labels = tf.placeholder(tf.float32, [batch_size, 4], 'u_d')

        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.histogram('u', self.u_labels)

        if corrupted:
            self.i1_corrupted = self.corrupt(self.i1, 0.1)
            tf.summary.image('image1_corrupted', self.i1_corrupted)
            self.encoding, self.feature_map_shape = self.encode(self.i1_corrupted)
        else:
            self.encoding, self.feature_map_shape = self.encode(self.i1)

        self.enc_identity, self.enc_transit = self.transit(self.encoding, self.u_labels, split_size)
        self.z, self.z_mean, self.z_log_sigma_sq = self.hidden_sample(
            self.enc_identity, self.enc_transit)
        #self.z, self.z_mean, self.z_log_sigma_sq = self.hidden_sample_static(self.encoding)

        with tf.variable_scope('generator'):
            self.decoding = self.generate(self.z, self.feature_map_shape)
            tf.summary.image('image2_rec', self.decoding)
        with tf.variable_scope('generator', reuse=True):
            z_uniform = self.random_sample(batch_size, 512)
            self.decoding_uniform = self.generate(z_uniform, self.feature_map_shape)
            tf.summary.image('image2_fake', self.decoding_uniform)

        with tf.variable_scope('discriminator'):
            i2_corrupted = self.corrupt(self.i2, 0.1)
            d_real, activation_real = self.discriminate(i2_corrupted)
            tf.summary.histogram('activation_real', activation_real)
        with tf.variable_scope('discriminator', reuse=True):
            decoding_corrupted = self.corrupt(self.decoding, 0.1)
            d_rec, activation_rec = self.discriminate(decoding_corrupted)
            tf.summary.histogram('activation_rec', activation_rec)
        with tf.variable_scope('discriminator', reuse=True):
            decoding_uniform_corrupted = self.corrupt(self.decoding_uniform, 0.1)
            d_fake, _ = self.discriminate(decoding_uniform_corrupted)

        self.loss = self.loss_function(activation_real, activation_rec,
                                       self.z_mean, self.z_log_sigma_sq,
                                       d_real, d_rec, d_fake)
        self.train_op_e, self.train_op_g, self.train_op_d = self.train(
            self.loss, learning_rate, episilon)
        self.merged = tf.summary.merge_all()

    def encode(self, img):
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
            #conv5_flat = tf.nn.dropout(conv5_flat, 0.7)
            fc6 = self.fc_bn_layer(conv5_flat, 1024, is_training=self.is_training,
                                   initializer_type=1, name='fc6')
            #fc6 = self.fc_no_activation(conv5_flat, 1024, initializer_type=1, name='fc6')
        return fc6, shape

    def transit(self, code, u, split_size):
        with tf.variable_scope('representation'):
            iden, pose = tf.split(code, num_or_size_splits=[1024-split_size, split_size], axis=1)
            identity = self.fc_no_activation(iden, 1024-split_size,
                                             initializer_type=1, name='iden')

            pose_u = tf.concat([pose, u], 1)
            pose_transit_1 = self.fc_bn_layer(
                pose_u, split_size, is_training=self.is_training,
                initializer_type=1, name='transit_1')
            pose_transit_2 = self.fc_bn_layer(
                pose_transit_1, split_size, is_training=self.is_training,
                initializer_type=1, name='transit_2')
            pose_transit = self.fc_no_activation(pose_transit_2,
                                                 split_size, initializer_type=1,
                                                 name='trainsit')

            tf.summary.histogram('identity', identity)
            tf.summary.histogram('pose', pose)
            tf.summary.histogram('pose_transit', pose_transit)
        return identity, pose_transit

    def hidden_sample(self, identity, pose_transit):
        with tf.variable_scope('sampling'):
            z_mean_1, z_log_sigma_square_1 = tf.split(identity, 2, 1)
            z_mean_2, z_log_sigma_square_2 = tf.split(pose_transit, 2, 1)
            z_mean = tf.concat([z_mean_1, z_mean_2], 1)
            z_log_sigma_sq = tf.concat([z_log_sigma_square_1, z_log_sigma_square_2], 1)
            z = self.hidden_code_sample(z_mean, z_log_sigma_sq, 'Gaussian_sampling')
            tf.summary.histogram('z_mean', z_mean)
            tf.summary.histogram('z_log_sigma_sq', z_log_sigma_sq)
            tf.summary.histogram('z_sampling', z)
        return z, z_mean, z_log_sigma_sq

    def hidden_sample_static(self, code):
        with tf.variable_scope('sampling_static'):
            z_mean, z_log_sigma_sq = tf.split(code, 2, 1)
            z = self.hidden_code_sample(z_mean, z_log_sigma_sq, 'Gaussian_sampling_static')
            tf.summary.histogram('z_mean_static', z_mean)
            tf.summary.histogram('z_log_sigma_sq_static', z_log_sigma_sq)
            tf.summary.histogram('z_sampling_static', z)
        return z, z_mean, z_log_sigma_sq

    def generate(self, code, shape):
        # no lrelu seems to lead to better result.
        #fc8 = self.fc_bn_layer(code, 1024, self.is_training, initializer_type=1, name='fc8')
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
                                    initializer_type=1, name='deconv1', act_func=tf.tanh)
        return deconv1

    def discriminate(self, img):
        # block indent M-m x tab
        # first layer of discriminator skips BN.
        #convm1 = self.conv_layer(img, [5, 5], [self.in_channels, 32],
        #                         stride=1, initializer_type=1, name='convm1',
        #                         act_func=self.lrelu)
        conv0 = self.conv_layer(
            img, [5, 5], [self.in_channels, 64], stride=2,
            initializer_type=1, name='conv0', act_func=self.lrelu)
        conv1 = self.conv_bn_layer(
            conv0, [5, 5], [64, 128], stride=2, is_training=self.is_training,
            initializer_type=1, name='conv1', act_func=self.lrelu)
        conv2 = self.conv_bn_layer(
            conv1, [5, 5], [128, 256], stride=2, is_training=self.is_training,
            initializer_type=1, name='conv2', act_func=self.lrelu)
        conv3 = self.conv_bn_layer(conv2, [5, 5], [256, 256], stride=1, is_training=self.is_training,
                                   initializer_type=1, name='conv3', act_func=self.lrelu)
        shape = conv3.get_shape().as_list()
        feature_map_size = shape[1]*shape[2]*shape[3]
        conv3_flat = tf.reshape(
            conv3, [-1, feature_map_size], 'conv3_flat')

        # activation of l th layer for vae reconstruction? lrelu or elu? batch normed?
        #fc0 = self.fc_layer(conv2_flat, 1024,
        #                    initializer_type=1, name='fc0',
        #                    act_func=self.lrelu)
        # dropout?
        #fc0 = tf.nn.dropout(fc0, 0.5)
        # no sigmoid attached because it is fed as logits to sigmoid cross entropy loss.
        fc1 = self.fc_no_activation(conv3_flat, 1, initializer_type=1, name='fc1')

        return fc1, conv3_flat

    def loss_function(self, act_real, act_rec,
                      mean, log_sigma_sq,
                      D_real, D_rec, D_fake):
        loss = dict()
        loss_f = dict()
        with tf.variable_scope('loss'):
            with tf.name_scope('E'):
                loss['dis'] = tf.reduce_sum(0.5*tf.square(act_real-act_rec), 1, keep_dims=True)
                loss['kl'] = -0.5*tf.reduce_sum(
                    1 + log_sigma_sq - tf.square(mean) - tf.exp(log_sigma_sq), 1, keep_dims=True)
                loss_f['E'] = tf.reduce_mean(loss['kl']+loss['dis'])

            with tf.name_scope('D'):
                loss['D_real'] = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_real,
                    labels=tf.random_uniform(tf.shape(D_real), 0.8, 1.0))
                    #labels=tf.ones_like(D_real))
                loss['D_rec'] = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_rec,
                    labels=tf.random_uniform(tf.shape(D_rec), 0.0, 0.2))
                    #labels=tf.zeros_like(D_rec))
                loss['D_fake'] = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_fake,
                    labels=tf.random_uniform(tf.shape(D_fake), 0.0, 0.2))
                    #labels=tf.zeros_like(D_fake))
                loss_f['D'] = tf.reduce_mean(loss['D_real']+0.5*(loss['D_rec']+loss['D_fake']))
                # not trained very well with this loss.
                #loss_f['D'] = tf.reduce_mean(loss['D_real']+loss['D_rec']+loss['D_fake'])

            with tf.name_scope('G'):
                loss['G_rec'] = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_rec,
                    labels=tf.random_uniform(tf.shape(D_rec), 0.8, 1.0))
                    #labels=tf.ones_like(D_rec))
                loss['G_fake'] = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_fake,
                    labels=tf.random_uniform(tf.shape(D_fake), 0.8, 1.0))
                    #labels=tf.ones_like(D_fake))
                loss_f['G'] = tf.reduce_mean(0.5*(loss['G_rec']+loss['G_fake'])+
                                             self.lambda_gan*loss['dis'])

            tf.summary.scalar('loss_D', loss_f['D'])
            tf.summary.scalar('loss_G', loss_f['G'])
            tf.summary.scalar('loss_E', loss_f['E'])
        return loss_f

    def train(self, loss, learning_rate, epsilon):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            #optimizer2 = tf.train.RMSPropOptimizer(learning_rate=0.00005)
            trainables = tf.trainable_variables()
            e_vars = [v for v in trainables if 'encoder' or 'sampling_static' in v.name]
            g_vars = [v for v in trainables if 'generator' in v.name]
            d_vars = [v for v in trainables if 'discriminator' in v.name]
            with tf.control_dependencies(update_ops):
                train_op_e = optimizer.minimize(loss['E'], var_list=e_vars)
                train_op_g = optimizer.minimize(loss['G'], var_list=g_vars)
                train_op_d = optimizer.minimize(loss['D'], var_list=d_vars)
        return train_op_e, train_op_g, train_op_d
