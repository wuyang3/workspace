# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 18:10:27 2017

Test input pipeline for reading consecutive images, continuous and discrete actions.

@author: wuyang
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import time

def read_data_list(data_list):
    """Reads a txt file containing pathes to images and labels. Labels here are 
    action values which are contained in numpy array file.

    Args:
        data_list: a txt file containing path to images and action arraies.
    Returns:
        lists of pathes to all images and corresponding labels.
    """
    f = open(data_list, 'r')
    images_1_path = []
    images_2_path = []
    actions_path = []
    actions_d_path = []
    actions_row = []
    for item in f:
        i_1, i_2, a_p, a_r = item.split(' ')
        images_1_path.append('../../poke/'+i_1)
        images_2_path.append('../../poke/'+i_2)
        actions_path.append('../../poke/'+a_p+'/actions.npy')
        actions_d_path.append('../../poke/'+a_p+'/actions_discrete.npy')
        actions_row.append(int(a_r))
    return images_1_path, images_2_path, actions_path, actions_d_path, actions_row

def read_images_actions(input_queue):
    """Consume a directory name and read images and actions.
    Args:
        A tensor conatains filename, filename, filename and string.
    Returns:
        Four tensors: decoded image 1, decoded image 2, continuous actions,
        discrete actions
    """
    #reader = tf.WholeFileReader()
    image_1_file = tf.read_file(input_queue[0])
    image_1 = tf.image.decode_jpeg(image_1_file)
    image_1_float = tf.cast(image_1, tf.float32)
    image_1_resized = tf.image.resize_images(image_1_float, [227, 227])
    image_1_resized.set_shape((227, 227, 3))

    image_2_file = tf.read_file(input_queue[1])
    image_2 = tf.image.decode_jpeg(image_2_file)
    image_2_float = tf.cast(image_2, tf.float32)
    image_2_resized = tf.image.resize_images(image_2_float, [227, 227])
    image_2_resized.set_shape((227, 227, 3))

    actions = np.load(input_queue[2])
    #actions_discrete = np.load(input_queue[3].eval())
    action_c = actions[input_queue[3]]
    #action_d = actions_discrete[input_queue[4].eval()]

    return image_1_resized, image_2_resized, action_c

# Some initial parameters
num_epochs = 1
batch_size = 500
num_threads = 4
min_after_dequeue = 1000

# Reads pathes of two consecutive images and action. List transformed to tensor.
i_1, i_2, a_c, a_d, a_r = read_data_list('../../poke/train_dir.txt')

images_1 = ops.convert_to_tensor(i_1, dtype=dtypes.string)
images_2 = ops.convert_to_tensor(i_2, dtype=dtypes.string)
actions = ops.convert_to_tensor(a_c, dtype=dtypes.string)
#actions_discrete = ops.convert_to_tensor(a_d, dtype=dtypes.string)
#row = ops.convert_to_tensor(a_r, dtype=dtypes.int32)

# Makes a queue. The queue takes in tensors (lists of inputs to be read.)
input_queue = tf.train.slice_input_producer([images_1, images_2, actions], #actions, actions_discrete, row
                                            num_epochs=num_epochs,
                                            shuffle=True)

# From input_queue, reads consecutive images and corresponding action.
image_1, image_2, action = read_images_actions(input_queue)

images_1, images_2, actions = tf.train.batch(
    [image_1, image_2, action], #, action_c, action_d
    batch_size=batch_size,
    num_threads=num_threads,
    #min_after_dequeue=min_after_dequeue,
    capacity=min_after_dequeue+3*batch_size)
cnt=0
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            start_time = time.time()
            i1, i2 = sess.run([images_1, images_2])
            print(i1.shape)
            print(i2.shape)
            print(time.time()-start_time)
            cnt += 1
            #print(ac)
            #print(ad+' \n')
    except tf.errors.OutOfRangeError:
        print('Done queuing: epoch limit reached.')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    print('epochs of batch enqueuing is %d' %cnt)
