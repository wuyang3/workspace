# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:10:27 2017

Generate data batch and train the poke model.

@author: wuyang
"""

import numpy as np
import tensorflow as tf
from poke import Poke
from batch_operation import read_data_list
from batch_operation import batch_images_actions
from batch_operation import feed

# Input queue and parameters
include_type = 1
num_epochs = 4
shuffle = True
batch_size = 16
num_threads = 4
min_after_dequeue = 512

# Reads pathes of data and create a input queue.
input_queue = read_data_list(
    '../../poke/train_gazebo_extra_class.txt', num_epochs, shuffle, include_type)

# From input_queue and batch parameters, return batch of processed
# images and labels.
images_1, images_2, u_cs, p_ds, theta_ds, l_ds, type_ds = batch_images_actions(
    input_queue, batch_size, num_threads, min_after_dequeue, include_type)

# Initial training parameters.
train_layers = ['inverse', 'forward', 'siamese']
learning_rate = 0.000257013213488
epsilon = 2.22850955304e-06
lamb = 0
step = 0

with tf.Session() as sess:
    poke_model = Poke(train_layers, include_type, learning_rate, epsilon, lamb)

    saver = tf.train.Saver(max_to_keep=2)
    train_writer = tf.summary.FileWriter('../logs/poke/', sess.graph)

    #saver.restore(sess, tf.train.latest_checkpoint('../logs/poke/rs_e64/'))

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            i1, i2, uc, pd, thetad, ld, typed = sess.run(
                [images_1, images_2, u_cs, p_ds, theta_ds, l_ds, type_ds])

            feed_dict = feed(poke_model, (i1, i2, uc, pd, thetad, ld, typed), include_type)

            _, loss, loss_i, loss_f, summary = sess.run(
                [poke_model.train_op, poke_model.loss, poke_model.loss_i,
                 poke_model.loss_f, poke_model.merged], feed_dict=feed_dict)

            if step % 10 == 0:
                train_writer.add_summary(summary, step)
                print('step %d: loss->%.4f inverse loss->%.4f forward loss->%.4f'
                      %(step, loss, loss_i, loss_f))
            if step % 1000 == 0:
                saver.save(sess, '../logs/poke/', global_step=step)
            step += 1
        train_writer.close()
    except tf.errors.OutOfRangeError:
        print('Done queuing: epoch limit reached.')
    finally:
        coord.request_stop()
    coord.join(threads)

print('epochs of batch enqueuing is %d' %step)
