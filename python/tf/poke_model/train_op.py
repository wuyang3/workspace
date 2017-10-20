# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:10:27 2017

Generate data batch and train the poke model.

@author: wuyang
"""

import numpy as np
import tensorflow as tf
from poke import Poke
from poke_depth import PokeDepth, PokeTotal
from batch_operation import read_data_list, batch_images_actions, feed
from batch_operation import read_data_list_t, batch_images_actions_t

def train_single():
    # Input queue and parameters
    tf.reset_default_graph()
    num_epochs = 5
    shuffle = True
    batch_size = 16
    num_threads = 4
    min_after_dequeue = 512

    include_type=0

    # switch: rgb or depth?
    data_list = '../../poke/train_cube_table_inv_d.txt'
    target_size = 227
    type_img = 0

    input_queue = read_data_list(data_list, num_epochs, shuffle, include_type=include_type)

    images_1, images_2, u_cs, p_ds, theta_ds, l_ds = batch_images_actions(
        input_queue, batch_size, num_threads, min_after_dequeue,
        include_type=include_type, target_size=target_size, type_img=type_img, normalized=0)

    # Initial training parameters.
    train_layers = ['inverse', 'forward', 'siamese']
    learning_rate = 0.0002
    epsilon = 1e-5
    lamb = 0
    step = 0

    with tf.Session() as sess:
        #poke_model = Poke(train_layers, learning_rate, epsilon, lamb, include_type=include_type)
        poke_model = PokeDepth(train_layers, learning_rate, epsilon, lamb,
                               include_type=include_type, corrupted=1, target_size=target_size)

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/poke_inv/depth_alex/', sess.graph)

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                i1, i2, uc, pd, thetad, ld = sess.run(
                    [images_1, images_2, u_cs, p_ds, theta_ds, l_ds])

                feed_dict = feed(poke_model, (i1, i2, uc, pd, thetad, ld), include_type)

                _, loss, loss_i, loss_f, summary = sess.run(
                    [poke_model.train_op, poke_model.loss, poke_model.loss_i,
                     poke_model.loss_f, poke_model.merged], feed_dict=feed_dict)

                if step % 10 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%.4f inverse loss->%.4f forward loss->%.4f'
                          %(step, loss, loss_i, loss_f))
                if step % 1000 == 0:
                    saver.save(sess, '../logs/poke_inv/depth_alex/', global_step=step)
                step += 1
            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')
        finally:
            coord.request_stop()
        coord.join(threads)

    print('epochs of batch enqueuing is %d' %step)

def train_total():
    tf.reset_default_graph()
    # Input queue and parameters
    num_epochs = 5
    shuffle = True
    batch_size = 16
    num_threads = 4
    min_after_dequeue = 512

    include_type=0

    data_list = '../../poke/train_cube_table_inv_t.txt'
    target_size = 227

    input_queue = read_data_list_t(data_list, num_epochs, shuffle, include_type=include_type)

    images_1, images_2, u_cs, p_ds, theta_ds, l_ds = batch_images_actions_t(
        input_queue, batch_size, num_threads, min_after_dequeue,
        include_type=include_type, target_size=target_size, normalized=0)

    # Initial training parameters.
    train_layers = ['inverse', 'forward', 'siamese']
    learning_rate = 0.0002
    epsilon = 1e-5
    lamb = 0
    step = 0

    with tf.Session() as sess:
        poke_model = PokeTotal(train_layers, learning_rate, epsilon, lamb,
                               include_type=include_type, target_size=target_size)

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/poke_inv/total_alex/', sess.graph)

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                i1, i2, uc, pd, thetad, ld = sess.run(
                    [images_1, images_2, u_cs, p_ds, theta_ds, l_ds])

                feed_dict = feed(poke_model, (i1, i2, uc, pd, thetad, ld), include_type)

                _, loss, loss_i, loss_f, summary = sess.run(
                    [poke_model.train_op, poke_model.loss, poke_model.loss_i,
                     poke_model.loss_f, poke_model.merged], feed_dict=feed_dict)

                if step % 10 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%.4f inverse loss->%.4f forward loss->%.4f'
                          %(step, loss, loss_i, loss_f))
                if step % 1000 == 0:
                    saver.save(sess, '../logs/poke_inv/total_alex/', global_step=step)
                step += 1
            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')
        finally:
            coord.request_stop()
        coord.join(threads)

    print('epochs of batch enqueuing is %d' %step)

if __name__ == '__main__':
    train_single()
    train_total()
