# -*- coding: utf-8 -*-
"""
Train auto-encoder for poking data.
For training denoising autoencoder, actions are continuous and not normalized. I didn't add
batch normalization just to staty close to the structure of the original paper.
For training variational autoencoder, actions are conitnuous and not normalized to get better
training results.
For training variational autoencoder with batch normalization, actions should be normalized
otherwise the scale would be too big for the activations of the bottleneck layer.

Including:
training an DAE, VAE, BNVAE for image pairs;
training AERNN(RNN/LSTM), VAERNN(RNN/LSTM), AERNN(LSTM or not, VAE or not) for three steps.
training VAELSTM with compare function for three steps.

Training variable size of steps is included in poke_ae_rnn_multi.py.
Online testing of ae, aernn, aernnc are in poke_test_ae.py.
Online testing of variable step size is also in poke_test_ae.py.
Offline testing of ae, aernn, aernnc and aernn with variable step size are in poke_test_ae_offline.py
"""

import numpy as np
import tensorflow as tf
from poke_autoencoder import PokeAE
from poke_vae import PokeVAE
from poke_bn_vae import PokeBnVAE
from poke_ae_rnn import PokeAERNN, PokeVAERNN, PokeAEFFRNN, PokeVAERNNC
from batch_operation import read_data_list_ae, batch_images_actions_ae
from batch_operation import read_data_list_ae_rnn, batch_images_actions_ae_rnn

def main():
    shuffle = True
    num_epochs = 8
    batch_size = 16
    num_threads = 4
    min_after_dequeue = 256

    type_img = 1 #1\0
    split_size = 256
    in_channels = 3 #3\1
    corrupted = 1 #1\0
    # Reads pathes of data and create a input queue.
    #input_queue = read_data_list_ae(
    #    '../../poke/train_gazebo_ae.txt', num_epochs, shuffle)
    #input_queue = read_data_list_ae(
    #    '../../poke/train_gazebo_vae.txt', num_epochs, shuffle)
    input_queue = read_data_list_ae(
        '../../poke/train_cube_ae.txt', num_epochs, shuffle)

    images_1, images_2, u_cs = batch_images_actions_ae(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img)

    # Initial training parameters.
    learning_rate = 0.0001
    epsilon = 1e-06
    step = 0

    with tf.Session() as sess:
        poke_ae = PokeAE(learning_rate, epsilon, batch_size, split_size=split_size,
                         in_channels=in_channels, corrupted=corrupted, is_training=True)
        #poke_ae = PokeVAE(learning_rate, epsilon, batch_size, split_size=split_size,
        #                 in_channels=in_channels, corrupted=corrupted)
        #poke_ae = PokeBnVAE(learning_rate, epsilon, batch_size, split_size,
        #                    in_channels, corrupted, is_training=True)
        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeAE/', sess.graph)

        #saver.restore(sess, tf.train.latest_checkpoint('../logs/pokeAE/'))
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                i1, i2, uc = sess.run([images_1, images_2, u_cs])

                feed_dict = {poke_ae.i1: i1, poke_ae.i2: i2, poke_ae.u_labels: uc}

                _, loss, summary = sess.run(
                    [poke_ae.train_op, poke_ae.loss, poke_ae.merged],
                    feed_dict=feed_dict)

                if step%10 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%.4f' %(step, loss))

                if step%1000 == 0:
                    saver.save(sess, '../logs/pokeAE/', global_step=step)
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

def ae_rnn_train():
    shuffle = True
    num_epochs = 8
    batch_size = 16
    num_threads = 4
    min_after_dequeue = 128

    type_img = 1 #1\0
    split_size = 64
    in_channels = 3 #3\1
    corrupted = 0 #1\0

    input_queue = read_data_list_ae_rnn(
        '../../poke/train_cube_ae_rnn_1.txt', num_epochs, shuffle)

    images_1, images_2, images_3, u1s, u2s = batch_images_actions_ae_rnn(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img,
        normalized=0)

    # Initial training parameters.
    learning_rate = 0.0003
    epsilon = 1e-05
    step = 0

    with tf.Session() as sess:
        #poke_ae = PokeAERNN(learning_rate, epsilon, batch_size, split_size,
        #                    in_channels=in_channels, corrupted=corrupted,
        #                    is_training=True, lstm=0)
        poke_ae = PokeVAERNN(learning_rate, epsilon, batch_size, split_size,
                             in_channels=in_channels, corrupted=corrupted,
                             is_training=True, lstm=1)
        #poke_ae = PokeAEFFRNN(learning_rate, epsilon, batch_size, in_channels,
        #                      corrupted, is_training=True, lstm=1, vae=1)
        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeAERNN/', sess.graph)

        #saver.restore(sess, tf.train.latest_checkpoint('../logs/pokeAE/'))
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
                    saver.save(sess, '../logs/pokeAERNN/', global_step=step)
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

def ae_rnnc_train():
    shuffle = True
    num_epochs = 3
    batch_size = 16
    num_threads = 4
    min_after_dequeue = 128

    type_img = 1 #1\0
    split_size = 512
    in_channels = 3 #3\1
    corrupted = 0 #1\0

    input_queue = read_data_list_ae_rnn(
        '../../poke/train_cube_ae_rnn_1.txt', num_epochs, shuffle)

    images_1, images_2, images_3, u1s, u2s = batch_images_actions_ae_rnn(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img,
        normalized=0)

    # Initial training parameters.
    learning_rate = 0.0003
    epsilon = 1e-05
    step = 0

    with tf.Session() as sess:
        poke_ae = PokeVAERNNC(learning_rate, epsilon, batch_size, split_size,
                              in_channels=in_channels, corrupted=corrupted,
                              is_training=True, lstm=1)
        vars = tf.global_variables()
        partial_vars = [v for v in vars if 'regression' in v.name]
        init_vars = [v for v in vars if 'regression' not in v.name]
        #for var in partial_vars:
        #    print("to restore: ", var)
        compare_saver = tf.train.Saver(var_list=partial_vars)
        compare_saver.restore(sess, tf.train.latest_checkpoint(
            '/home/wuyang/workspace/python/tf/logs/pokeCompare/2fc_3epoch/'))

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeAERNN/', sess.graph)

        tf.variables_initializer(var_list=init_vars).run()
        #tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        #for var in vars: # all vars are initialized.
        #    if not sess.run(tf.is_variable_initialized(var)):
        #        print('unitialized', var)

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
                    saver.save(sess, '../logs/pokeAERNN/', global_step=step)
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

if __name__ == '__main__':
    #main()
    #ae_rnn_train()
    ae_rnnc_train()
