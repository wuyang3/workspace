# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from poke_ae_rnn import PokeVAERNN
from batch_operation import read_data_list_ae_rnn, batch_images_actions_ae_rnn

class PokeVAERNNDiff(PokeVAERNN):
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0, is_training=True, lstm=0):
        self.in_channels = in_channels
        self.is_training = is_training
        self.i1 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i1')
        self.i2 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i2')
        self.i3 = tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels], 'i3')
        self.u2 = tf.placeholder(tf.float32, [batch_size, 4], 'u2')
        self.u3 = tf.placeholder(tf.float32, [batch_size, 4], 'u3')
        #self.u1 = tf.zeros([batch_size, 4], tf.float32)
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
            [self.u2, self.u3], self.pose, lstm)
            #[self.u1, self.u2, self.u3], self.pose, lstm)

        self.sampling, self.mean, self.logss = self.hidden_sample(self.identity, self.transit_series)

        self.mask_1 = self.decode(self.sampling[0], self.feature_map_shape, False)
        self.mask_2 = self.decode(self.sampling[1], self.feature_map_shape, True)
        #self.mask_3 = self.decode(self.sampling[2], self.feature_map_shape, True)
        tf.summary.image('image_rec2', self.mask_1+self.i1)
        tf.summary.image('image_rec3', self.mask_2+self.i1)
        #tf.summary.image('image_rec3', self.mask_3+self.i1)

        self.loss = self.loss_diff(self.i1, self.i2, self.i3,
                                   self.mask_1, self.mask_2, #self.mask_3,
                                   self.mean, self.logss)

        #self.train_op = self.train(self.loss, learning_rate, epsilon)
        self.train_op = self.train_diff(self.loss, learning_rate, epsilon)

        self.merged = tf.summary.merge_all()

    def loss_diff(self, i1, i2, i3, i_mask2, i_mask3, #i_mask3,
                  mean, logss):
        assert len(mean)==len(logss), (
            'Steps of mean and variance not match'
        )
        #target1 = tf.zeros_like(i1, dtype=tf.float32)
        target2 = i2 - i1
        target3 = i3 - i1

        with tf.variable_scope('loss'):
            with tf.variable_scope('rec'):
                loss_rec = tf.reduce_sum(
                    #tf.square(target1 - i_mask1)+
                    tf.square(target2 - i_mask2)+
                    tf.square(target3 - i_mask3),
                    [1, 2, 3])
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

    def train_diff(self, loss, learning_rate, epsilon):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate, epsilon=epsilon)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)
        return train_op

class PokeVAERNNGD(PokeVAERNN):
    """
    An extra regularizer of image gradient difference loss.
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

        self.loss = self.loss_diff(self.i1, self.i2, self.i3,
                                   self.decoding_1, self.decoding_2, self.decoding_3,
                                   self.mean, self.logss)

        self.train_op = self.train(self.loss, learning_rate, epsilon)

        self.merged = tf.summary.merge_all()

    def get_img_gradient_diff(self, i, i_rec):
        batch_size, height, width, channels = i.get_shape().as_list()
        assert channels == self.in_channels
        i00 = tf.slice(i, [0, 0, 0, 0], [batch_size, height-1, width-1, channels])
        i10 = tf.slice(i, [0, 1, 0, 0], [batch_size, height-1, width-1, channels])
        i01 = tf.slice(i, [0, 0, 1, 0], [batch_size, height-1, width-1, channels])

        i00_rec = tf.slice(i_rec, [0, 0, 0, 0], [batch_size, height-1, width-1, channels])
        i10_rec = tf.slice(i_rec, [0, 1, 0, 0], [batch_size, height-1, width-1, channels])
        i01_rec = tf.slice(i_rec, [0, 0, 1, 0], [batch_size, height-1, width-1, channels])

        gdl10 = tf.abs(tf.abs(i10 - i00) - tf.abs(i10_rec - i00_rec))
        gdl01 = tf.abs(tf.abs(i01 - i00) - tf.abs(i01_rec - i00_rec))

        gdl = tf.reduce_sum(gdl10 + gdl01, axis=[1,2,3])

        return gdl

    def loss_diff(self, i1, i2, i3, i_rec1, i_rec2, i_rec3, mean, logss):
        assert len(mean)==len(logss), (
            'Steps of mean and variance not match'
        )
        with tf.variable_scope('loss'):
            with tf.variable_scope('rec'):
                loss_rec = tf.reduce_sum(
                    tf.square(i1 - i_rec1)+tf.square(i2 - i_rec2)+tf.square(i3 - i_rec3), [1, 2, 3])
            with tf.variable_scope('gdl'):
                loss_gdl = 0
                for i, i_rec in zip([i1, i2, i3], [i_rec1, i_rec2, i_rec3]):
                    loss_gdl+= self.get_img_gradient_diff(i, i_rec)
            with tf.variable_scope('kl'):
                loss_kl = 0
                for j in range(len(mean)):
                    loss_kl+= -0.5*tf.reduce_sum(1+logss[j]-tf.square(mean[j])-tf.exp(logss[j]), 1)

            # loss averaged over batches thus different from pixel wise diferrence.
            loss = tf.reduce_mean(loss_rec + loss_kl + 0.1*loss_gdl)
            tf.summary.scalar('loss_vae', loss)
            tf.summary.scalar('loss_rec', tf.reduce_mean(loss_rec))
            tf.summary.scalar('loss_kl', tf.reduce_mean(loss_kl))
            tf.summary.scalar('loss_gdl', tf.reduce_mean(loss_gdl))
        return loss

def ae_rnn_diff_train():
    shuffle = True
    num_epochs = 8
    batch_size = 16
    num_threads = 4
    min_after_dequeue = 128

    type_img = 1 #1\0
    split_size = 512
    in_channels = 3 #3\1
    corrupted = 0 #1\0

    input_queue = read_data_list_ae_rnn(
        '../../poke/train_cube_table_rnn.txt', num_epochs, shuffle)

    images_1, images_2, images_3, u1s, u2s = batch_images_actions_ae_rnn(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img,
        normalized=1)

    # Initial training parameters.
    learning_rate = 0.0002
    epsilon = 1e-05
    step = 0

    with tf.Session() as sess:
        #poke_ae = PokeVAERNNDiff(learning_rate, epsilon, batch_size, split_size,
        #                         in_channels=in_channels, corrupted=corrupted,
        #                         is_training=True, lstm=1)
        poke_ae = PokeVAERNNGD(learning_rate, epsilon, batch_size, split_size,
                               in_channels=in_channels, corrupted=corrupted,
                               is_training=True, lstm=1)

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeAERNN_new/', sess.graph)

        #saver.restore(sess, tf.train.latest_checkpoint('../logs/pokeAERNN_new/lstm_vae_8/'))
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                i1, i2, i3, u1, u2 = sess.run([images_1, images_2, images_3, u1s, u2s])

                feed_dict = {poke_ae.i1: i1, poke_ae.i2: i2, poke_ae.i3: i3,
                             poke_ae.u2: u1, poke_ae.u3: u2}

                _, loss, summary = sess.run(
                    [poke_ae.train_op, poke_ae.loss, poke_ae.merged],
                    feed_dict=feed_dict)

                if step%10 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%.4f' %(step, loss))

                if step%1000 == 0:
                    saver.save(sess, '../logs/pokeAERNN_new/', global_step=step)
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

if __name__ == '__main__':
    ae_rnn_diff_train()
