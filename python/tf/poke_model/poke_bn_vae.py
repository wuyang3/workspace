# -*- coding: utf-8 -*-
"""
Create a customized auto-encoder with action unit at the bottomneck for learning distangled state representations. Variational autoencoder version with batch normalization.
"""
import numpy as np
import tensorflow as tf
from poke_autoencoder import ConvAE


class PokeBnVAE(ConvAE):
    """
    Auto-encoder model for distangling state representations.
    For 240X240 images, it will be troublesome to upsample.
    64 -> 32 -> 16 -> 8 -> 1024 -> sample(512) -> 8 -> 16 -> 32...
    From 15X15, convolution with stride 2 and 5X5 filter will lead to:
    6X6 (valid padding ceil((15-5+1)/2)=6)
    8X8 (same padding ceil(15/2)=8, SAME padding 2 to the left and right)
    """
    def __init__(self, learning_rate=1e-4, episilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0, is_training=True):
        self.in_channels = in_channels
        self.is_training = is_training
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.u_labels = tf.placeholder(tf.float32, [batch_size, 4], 'u_d')
        tf.summary.image('image1', self.i1)
        tf.summary.image('image2', self.i2)
        tf.summary.histogram('u', self.u_labels)

        if corrupted:
            self.i1_corrupted = self.corrupt(self.i1, 0.05)
            tf.summary.image('image1_corrupted', self.i1_corrupted)
            self.encoding, self.feature_map_shape = self.encode(self.i1_corrupted)
        else:
            self.encoding, self.feature_map_shape = self.encode(self.i1)

        self.enc_identity, self.enc_transit = self.transit(self.encoding, self.u_labels, split_size)
        self.z, self.z_mean, self.z_log_sigma_sq = self.hidden_sample(
            self.enc_identity, self.enc_transit)
        self.decoding = self.decode(self.z, self.feature_map_shape)
        tf.summary.image('image2_rec', self.decoding)

        self.loss = self.loss_function(self.i2, self.decoding, self.z_mean, self.z_log_sigma_sq)
        self.train_op = self.train(self.loss, learning_rate, episilon)
        self.merged = tf.summary.merge_all()

    def encode(self, img):
        with tf.variable_scope('encoder'):
            #conv3 = self.conv_bn_layer(
            #    img, [5, 5], [self.in_channels, 64], stride=2, is_training=self.is_training,
            #    initializer_type=1, name='conv3')
            conv3 = self.conv_layer(
                img, [5, 5], [self.in_channels, 64], stride=2,
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
            #fc7 = self.fc_bn_layer(fc6, 1024, is_training=self.is_training,
            #                       initializer_type=1, name='fc7')
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

    def decode(self, code, shape):
        with tf.variable_scope('decoder'):
            #fc8 = self.fc_bn_layer(code, 1024, is_training=self.is_training,
            #                       initializer_type=1, name='fc8')
            feature_map_size = shape[1]*shape[2]*shape[3]
            fc9 = self.fc_bn_layer(code, feature_map_size, is_training=self.is_training,
                                initializer_type=1, name='fc9')
            deconv5 = tf.reshape(fc9, [shape[0], shape[1], shape[2], shape[3]])
            upsampling4 = self.upsample(
                deconv5, stride=2, name='upsampling4', mode='COPY')
            deconv4 = self.deconv_bn_layer(
                upsampling4, [5, 5], [256, 128], is_training=self.is_training,
                initializer_type=1, name='deconv4')
            upsampling3 = self.upsample(
                deconv4, stride=2, name='upsampling3', mode='COPY')
            deconv3 = self.deconv_bn_layer(
                upsampling3, [5, 5], [128, 64], is_training=self.is_training,
                initializer_type=1, name='deconv3')
            upsampling2 = self.upsample(
                deconv3, stride=2, name='upsampling2', mode='COPY')
            deconv2 = self.deconv_bn_layer(
                upsampling2, [5, 5], [64, self.in_channels], is_training=self.is_training,
                initializer_type=1, name='deconv2')
            deconv1 = self.deconv_layer(deconv2, [5, 5], [self.in_channels, self.in_channels],
                                        initializer_type=1, name='deconv1')
        return deconv1

    def loss_function(self, img, img_reconstruction, mean, log_sigma_sq):
        with tf.variable_scope('loss'):
            loss_rec = tf.reduce_sum(tf.square(img-img_reconstruction), [1,2,3])
            loss_kl = -0.5*tf.reduce_sum(
                1 + log_sigma_sq - tf.square(mean) - tf.exp(log_sigma_sq), 1)
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
