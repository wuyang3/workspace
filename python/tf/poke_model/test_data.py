# -*- coding: utf-8 -*-
"""
Test the input pipeline on the pokebot data.

@author: wuyang
"""

import numpy as np
import tensorflow as tf
from batch_operation import read_data_list
from batch_operation import batch_dir_images_actions
from batch_operation import feed

# Input queue parameters
# When shuffle is false and num_threads > 1, it is possible that the order
# is slightly messed. It won't be an issue since the slice of data is correct.
num_epochs = 1
shuffle = False
batch_size = 5
num_threads = 4
min_after_dequeue = 1000

# Read pathes of test data and create a input queue.
input_queue = read_data_list('../../poke/train.txt', num_epochs, shuffle)

# Return batch of processed images and labels for testing.
dir_1, dir_2, images_1, images_2, u_cs, p_ds, theta_ds, l_ds = batch_dir_images_actions(
    input_queue, batch_size, num_threads, min_after_dequeue)

# Initial parameters.
train_layers = ['inverse', 'forward', 'siamese']
step=0

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            d1, d2, i1, i2, uc, pd, thetad, ld = sess.run(
                [dir_1, dir_2, images_1, images_2, u_cs, p_ds, theta_ds, l_ds])

            print d1
            print d2
            print uc
            print pd, thetad, ld
            print(' \n')
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done queuing: epoch limit reached.')
    finally:
        coord.request_stop()
    coord.join(threads)

print('epochs of batch enqueuing is %d' %step)
