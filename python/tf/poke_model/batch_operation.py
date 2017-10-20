# -*- coding: utf-8 -*-
"""
Helper function for:
1. Reading train data directories and labels, creating input queue.
2. Generating input batch from input queue.
3. Creating feed_dict for the poke model.

@author: wuyang
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

def read_data_list(data_list, num_epochs, shuffle, include_type=0):
    """Reads a txt file containing pathes to images and labels. Labels here are 
    action values which are contained in numpy array files.

    Args:
        data_list: a txt file containing path to images and action arraies.
        Lists are transformed into tensor and used for input queue.
    Returns:
        A input queue for data reading and batching.
    """
    assert type(shuffle)==bool, (
        'Requires a bool indicating shuffling or not'
    )
    with open(data_list, 'r') as f:
        images_1_path = []
        images_2_path = []
        u_c = []
        p_d = []
        theta_d = []
        l_d = []
        type_d = []
        for item in f:
            items = item.split(' ')
            images_1_path.append('../../poke/'+items[0])
            images_2_path.append('../../poke/'+items[1])
            """
            Array put in a tuple. When this list of tuple is converted to
            tensor, additional shape of the tuple will be reserved. float32
            can be applied on tuple elements.
            However, for list of string and number, the converted tensor will
            have only one dimension, so does the batch operation on individual
            tensor scalars, e.g. l_d -> l_ds.
            Must first read float as it is and then transform it into int.
            """
            u_c.append((float(items[2]), float(items[3]),
                        float(items[4]), float(items[5])))
            p_d.append(int(float(items[6])))
            theta_d.append(int(float(items[7])))
            l_d.append(int(float(items[8])))
            if include_type:
                type_d.append(int(items[9]))

    i1_t = ops.convert_to_tensor(images_1_path, dtype=dtypes.string)
    i2_t = ops.convert_to_tensor(images_2_path, dtype=dtypes.string)
    # List of tuple converted to a constant tensor of shape (108915, 4)
    uc_t = ops.convert_to_tensor(u_c, dtype=dtypes.float32)
    pd_t = ops.convert_to_tensor(p_d, dtype=dtypes.int32)
    thetad_t = ops.convert_to_tensor(theta_d, dtype=dtypes.int32)
    ld_t = ops.convert_to_tensor(l_d, dtype=dtypes.int32)
    # shuffle or not is defined here.
    if include_type:
        typed_t = ops.convert_to_tensor(type_d, dtype=dtypes.int32)
        input_queue = tf.train.slice_input_producer(
            [i1_t, i2_t, uc_t, pd_t, thetad_t, ld_t, typed_t],
            num_epochs=num_epochs,
            shuffle=shuffle)
    else:
        input_queue = tf.train.slice_input_producer(
            [i1_t, i2_t, uc_t, pd_t, thetad_t, ld_t],
            num_epochs=num_epochs,
            shuffle=shuffle)

    return input_queue

def batch_images_actions(
        input_queue, batch_size, num_threads, min_after_dequeue, include_type=0, target_size=227,
        type_img=1, normalized=0):
    """Consume a input queue with image directories and labels.
    Args:
        A tensor conatains directories and labels.
    Returns:
        Six tensors: decoded images 1, decoded images 2, continuous actions,
        three discrete actions.
        Note that placeholder also accepts numpy array directly. Just feed in the
        array in feed_dict.
    """
    if type_img:
        image_1_file = tf.read_file(input_queue[0])
        image_1 = tf.image.decode_jpeg(image_1_file)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float = image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [target_size, target_size])
        image_1_resized.set_shape((target_size, target_size, 3))

        image_2_file = tf.read_file(input_queue[1])
        image_2 = tf.image.decode_jpeg(image_2_file)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float = image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [target_size, target_size])
        image_2_resized.set_shape((target_size, target_size, 3))

    else:
        image_1_file = tf.read_file(input_queue[0])
        image_1 = tf.image.decode_png(image_1_file, dtype=tf.uint8)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float = image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [target_size, target_size])
        image_1_resized.set_shape((target_size, target_size, 1))

        image_2_file = tf.read_file(input_queue[1])
        image_2 = tf.image.decode_png(image_2_file, dtype=tf.uint8)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float = image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [target_size, target_size])
        image_2_resized.set_shape((target_size, target_size, 1))

    u_c = input_queue[2]
    p_d = input_queue[3]
    theta_d = input_queue[4]
    l_d = input_queue[5]

    # When evaluated in sess.run(), numpy ndarrays are acquired.
    # For a single l_d, a numpy.float32 is acquired.
    if include_type:
        type_d = input_queue[6]
        images_1, images_2, u_cs, p_ds, theta_ds, l_ds, type_ds = tf.train.batch(
            [image_1_resized, image_2_resized, u_c, p_d, theta_d, l_d, type_d],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_after_dequeue+3*batch_size)
        output = (images_1, images_2, u_cs, p_ds, theta_ds, l_ds, type_ds)

    else:
        images_1, images_2, u_cs, p_ds, theta_ds, l_ds = tf.train.batch(
            [image_1_resized, image_2_resized, u_c, p_d, theta_d, l_d],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_after_dequeue+3*batch_size)
        output = (images_1, images_2, u_cs, p_ds, theta_ds, l_ds)

    return output

def batch_dir_images_actions(
        input_queue, batch_size, num_threads, min_after_dequeue, include_type=0):
    """Consume a input queue with image directories and labels.
    Args:
        A tensor conatains directories and labels.
    Returns:
        Eight tensors: image directory 1, image directory 2, decoded images 1,
        decoded images 2, continuous actions, three discrete actions.
    """
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

    u_c = input_queue[2]
    p_d = input_queue[3]
    theta_d = input_queue[4]
    l_d = input_queue[5]

    if include_type:
        type_d = input_queue[6]
        dir_1, dir_2, images_1, images_2, u_cs, p_ds, theta_ds, l_ds, type_ds = tf.train.batch(
            [input_queue[0], input_queue[1], image_1_resized, image_2_resized,
             u_c, p_d, theta_d, l_d, type_d],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_after_dequeue+3*batch_size)
        output = (dir_1, dir_2, images_1, images_2, u_cs, p_ds, theta_ds, l_ds, type_ds)

    else:
        dir_1, dir_2, images_1, images_2, u_cs, p_ds, theta_ds, l_ds = tf.train.batch(
            [input_queue[0], input_queue[1], image_1_resized, image_2_resized,
             u_c, p_d, theta_d, l_d],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_after_dequeue+3*batch_size)
        output = (dir_1, dir_2, images_1, images_2, u_cs, p_ds, theta_ds, l_ds)

    return output

def feed(model, args, include_type=0):
    if include_type:
        feed_dict = {model.i1: args[0], model.i2: args[1],
                     model.u: args[2], model.p_labels: args[3],
                     model.theta_labels: args[4], model.l_labels: args[5],
                     model.type_labels: args[6]}
    else:
        feed_dict = {model.i1: args[0], model.i2: args[1],
                     model.u: args[2], model.p_labels: args[3],
                     model.theta_labels: args[4], model.l_labels: args[5]}
    return feed_dict

def image_feed(name, target_shape, type_img=1, normalized=0):
    image_file = tf.read_file(name)
    if type_img:
        image = tf.image.decode_jpeg(image_file, channels=3)
        num_channels = 3
    else:
        image = tf.image.decode_png(image_file, dtype=tf.uint8)
        num_channels = 1
    image_float = tf.cast(image, tf.float32)
    if normalized:
        image_float = image_float/255.0
    image_resize = tf.image.resize_images(image_float, [target_shape, target_shape])
    image_resize.set_shape((target_shape, target_shape, num_channels))
    image_feed = tf.reshape(image_resize, [-1, target_shape, target_shape, num_channels])

    return image_feed

def image_feed_total(name_img, name_depth, target_shape, normalized=0):
    rgb = image_feed(name_img, target_shape, type_img=1, normalized=normalized)
    depth = image_feed(name_depth, target_shape, type_img=0, normalized=normalized)

    i = tf.concat([rgb, depth], axis=3)
    return i

def image_feed_ae(name, type_img, normalized=0):
    # default batch size of 1.
    if type_img:
        image_file = tf.read_file(name)
        image = tf.image.decode_jpeg(image_file, channels=3)
        image_float = tf.cast(image, tf.float32)
        if normalized:
            image_float = image_float/255.0*2.0-1.0
        image_resize = tf.image.resize_images(image_float, [64, 64])
        image_resize.set_shape((64,64,3))
        image_feed = tf.reshape(image_resize, [-1, 64, 64, 3])
    else:
        image_file = tf.read_file(name)
        image = tf.image.decode_png(image_file, dtype=tf.uint16)
        image_float = tf.cast(image, tf.float32)
        if normalized:
            image_float = image_float/255.0*2.0-1.0
        image_resize = tf.image.resize_images(image_float, [64, 64])
        image_resize.set_shape((64,64,1))
        image_feed = tf.reshape(image_resize, [-1, 64, 64, 1])
    return image_feed

def read_data_list_ae(data_list, num_epochs, shuffle):
    """Reads a txt file containing pathes to images and labels. Labels here are 
    action values. Specific for auto-encoder training data.

    Args:
        data_list: a txt file containing path to images and action arraies.
        Lists are transformed into tensor and used for input queue.
    Returns:
        A input queue for data reading and batching.
    """
    assert type(shuffle)==bool, (
        'Requires a bool indicating shuffling or not'
    )
    with open(data_list, 'r') as f:
        images_1_path = []
        images_2_path = []
        u_c = []
        for item in f:
            items = item.split(' ')
            images_1_path.append('../../poke/'+items[0])
            images_2_path.append('../../poke/'+items[1])
            u_c.append((float(items[2]), float(items[3]),
                        float(items[4]), float(items[5])))

    i1_t = ops.convert_to_tensor(images_1_path, dtype=dtypes.string)
    i2_t = ops.convert_to_tensor(images_2_path, dtype=dtypes.string)
    uc_t = ops.convert_to_tensor(u_c, dtype=dtypes.float32)
    input_queue = tf.train.slice_input_producer(
        [i1_t, i2_t, uc_t],
        num_epochs=num_epochs,
        shuffle=shuffle)

    return input_queue

def batch_images_actions_ae(input_queue, batch_size, num_threads, min_after_dequeue,
                            type_img=1, normalized=0):
    """
    If type_img 1 by default, rgb images. Otherwise depth image.
    """
    image_1_file = tf.read_file(input_queue[0])
    image_2_file = tf.read_file(input_queue[1])
    u_c = input_queue[2]

    if type_img:
        image_1 = tf.image.decode_jpeg(image_1_file)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [64, 64])
        image_1_resized.set_shape((64, 64, 3))

        image_2 = tf.image.decode_jpeg(image_2_file)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float=image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [64, 64])
        image_2_resized.set_shape((64, 64, 3))
    else:
        image_1 = tf.image.decode_png(image_1_file, dtype=tf.uint8)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [64, 64])
        image_1_resized.set_shape((64, 64, 1))

        image_2 = tf.image.decode_png(image_2_file, dtype=tf.uint8)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float=image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [64, 64])
        image_2_resized.set_shape((64, 64, 1))

    images_1, images_2, u_cs = tf.train.batch(
        [image_1_resized, image_2_resized, u_c],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_after_dequeue+3*batch_size)
    output = (images_1, images_2, u_cs)

    return output

def batch_dir_images_actions_ae(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=1):
    """
    Extra directory string included.
    """
    image_1_file = tf.read_file(input_queue[0])
    image_2_file = tf.read_file(input_queue[1])
    u_c = input_queue[2]
    if type_img:
        image_1 = tf.image.decode_jpeg(image_1_file)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [64, 64])
        image_1_resized.set_shape((64, 64, 3))

        image_2 = tf.image.decode_jpeg(image_2_file)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float=image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [64, 64])
        image_2_resized.set_shape((64, 64, 3))

    else:
        image_1 = tf.image.decode_png(image_1_file, dtype=tf.uint16) #? 16 or 8?
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [64, 64])
        image_1_resized.set_shape((64, 64, 1))

        image_2 = tf.image.decode_png(image_2_file, dtype=tf.uint16)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float=image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [64, 64])
        image_2_resized.set_shape((64, 64, 1))

    dir_1, dir_2, images_1, images_2, u_cs = tf.train.batch(
        [input_queue[0], input_queue[1], image_1_resized, image_2_resized, u_c],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_after_dequeue+3*batch_size)
    output = (dir_1, dir_2, images_1, images_2, u_cs)

    return output

def read_data_list_ae_rnn(data_list, num_epochs, shuffle):
    assert type(shuffle)==bool, (
        'Requires a bool indicating shuffling or not'
    )
    with open(data_list, 'r') as f:
        images_1_path = []
        images_2_path = []
        images_3_path = []
        u1 = []
        u2 = []
        for item in f:
            items = item.split(' ')
            images_1_path.append('../../poke/'+items[0])
            images_2_path.append('../../poke/'+items[1])
            images_3_path.append('../../poke/'+items[2])
            u1.append((float(items[3]), float(items[4]),
                        float(items[5]), float(items[6])))
            u2.append((float(items[7]), float(items[8]),
                        float(items[9]), float(items[10])))

    i1_t = ops.convert_to_tensor(images_1_path, dtype=dtypes.string)
    i2_t = ops.convert_to_tensor(images_2_path, dtype=dtypes.string)
    i3_t = ops.convert_to_tensor(images_3_path, dtype=dtypes.string)
    u1_t = ops.convert_to_tensor(u1, dtype=dtypes.float32)
    u2_t = ops.convert_to_tensor(u2, dtype=dtypes.float32)
    input_queue = tf.train.slice_input_producer(
        [i1_t, i2_t, i3_t, u1_t, u2_t],
        num_epochs=num_epochs,
        shuffle=shuffle)

    return input_queue

def batch_images_actions_ae_rnn(input_queue, batch_size, num_threads, min_after_dequeue,
                                type_img=1, normalized=0):
    """
    If type_img 1 by default, rgb images. Otherwise depth image.
    Depth image data are actually stored with 16 bit integers.
    """
    image_1_file = tf.read_file(input_queue[0])
    image_2_file = tf.read_file(input_queue[1])
    image_3_file = tf.read_file(input_queue[2])
    u1 = input_queue[3]
    u2 = input_queue[4]

    if type_img:
        image_1 = tf.image.decode_jpeg(image_1_file)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [64, 64])
        image_1_resized.set_shape((64, 64, 3))

        image_2 = tf.image.decode_jpeg(image_2_file)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float=image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [64, 64])
        image_2_resized.set_shape((64, 64, 3))

        image_3 = tf.image.decode_jpeg(image_3_file)
        image_3_float = tf.cast(image_3, tf.float32)
        if normalized:
            image_3_float=image_3_float/255.0
        image_3_resized = tf.image.resize_images(image_3_float, [64, 64])
        image_3_resized.set_shape((64, 64, 3))
    else:
        image_1 = tf.image.decode_png(image_1_file, dtype=tf.uint8)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [64, 64])
        image_1_resized.set_shape((64, 64, 1))

        image_2 = tf.image.decode_png(image_2_file, dtype=tf.uint8)
        image_2_float = tf.cast(image_2, tf.float32)
        if normalized:
            image_2_float=image_2_float/255.0
        image_2_resized = tf.image.resize_images(image_2_float, [64, 64])
        image_2_resized.set_shape((64, 64, 1))

        image_3 = tf.image.decode_png(image_3_file, dtype=tf.uint8)
        image_3_float = tf.cast(image_3, tf.float32)
        if normalized:
            image_3_float=image_3_float/255.0
        image_3_resized = tf.image.resize_images(image_3_float, [64, 64])
        image_3_resized.set_shape((64, 64, 1))

    images_1, images_2, images_3, u1s, u2s = tf.train.batch(
        [image_1_resized, image_2_resized, image_3_resized, u1, u2],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_after_dequeue+3*batch_size)
    output = (images_1, images_2, images_3, u1s, u2s)

    return output


def read_data_list_t(data_list, num_epochs, shuffle, include_type=0):
    """
    For both rgb and depth data.
    """
    assert type(shuffle)==bool, (
        'Requires a bool indicating shuffling or not'
    )
    with open(data_list, 'r') as f:
        rgb_1_path = []
        rgb_2_path = []
        dep_1_path = []
        dep_2_path = []
        u_c = []
        p_d = []
        theta_d = []
        l_d = []
        type_d = []
        for item in f:
            items = item.split(' ')
            rgb_1_path.append('../../poke/'+items[0])
            rgb_2_path.append('../../poke/'+items[1])
            dep_1_path.append('../../poke/'+items[2])
            dep_2_path.append('../../poke/'+items[3])
            u_c.append((float(items[4]), float(items[5]),
                        float(items[6]), float(items[7])))
            p_d.append(int(float(items[8])))
            theta_d.append(int(float(items[9])))
            l_d.append(int(float(items[10])))
            if include_type:
                type_d.append(int(items[11]))

    r1_t = ops.convert_to_tensor(rgb_1_path, dtype=dtypes.string)
    r2_t = ops.convert_to_tensor(rgb_2_path, dtype=dtypes.string)
    d1_t = ops.convert_to_tensor(dep_1_path, dtype=dtypes.string)
    d2_t = ops.convert_to_tensor(dep_2_path, dtype=dtypes.string)
    uc_t = ops.convert_to_tensor(u_c, dtype=dtypes.float32)
    pd_t = ops.convert_to_tensor(p_d, dtype=dtypes.int32)
    thetad_t = ops.convert_to_tensor(theta_d, dtype=dtypes.int32)
    ld_t = ops.convert_to_tensor(l_d, dtype=dtypes.int32)
    # shuffle or not is defined here.
    if include_type:
        typed_t = ops.convert_to_tensor(type_d, dtype=dtypes.int32)
        input_queue = tf.train.slice_input_producer(
            [r1_t, r2_t, d1_t, d2_t, uc_t, pd_t, thetad_t, ld_t, typed_t],
            num_epochs=num_epochs,
            shuffle=shuffle)
    else:
        input_queue = tf.train.slice_input_producer(
            [r1_t, r2_t, d1_t, d2_t, uc_t, pd_t, thetad_t, ld_t],
            num_epochs=num_epochs,
            shuffle=shuffle)

    return input_queue

def batch_images_actions_t(
        input_queue, batch_size, num_threads, min_after_dequeue, include_type=0, target_size=227,
         normalized=0):
    r_1_file = tf.read_file(input_queue[0])
    r_1 = tf.image.decode_jpeg(r_1_file)
    r_1_float = tf.cast(r_1, tf.float32)
    if normalized:
        r_1_float = r_1_float/255.0
    r_1_resized = tf.image.resize_images(r_1_float, [target_size, target_size])
    r_1_resized.set_shape((target_size, target_size, 3))

    r_2_file = tf.read_file(input_queue[1])
    r_2 = tf.image.decode_jpeg(r_2_file)
    r_2_float = tf.cast(r_2, tf.float32)
    if normalized:
        r_2_float = r_2_float/255.0
    r_2_resized = tf.image.resize_images(r_2_float, [target_size, target_size])
    r_2_resized.set_shape((target_size, target_size, 3))

    d_1_file = tf.read_file(input_queue[2])
    d_1 = tf.image.decode_png(d_1_file, dtype=tf.uint8)
    d_1_float = tf.cast(d_1, tf.float32)
    if normalized:
        d_1_float = d_1_float/255.0
    d_1_resized = tf.image.resize_images(d_1_float, [target_size, target_size])
    d_1_resized.set_shape((target_size, target_size, 1))

    d_2_file = tf.read_file(input_queue[3])
    d_2 = tf.image.decode_png(d_2_file, dtype=tf.uint8)
    d_2_float = tf.cast(d_2, tf.float32)
    if normalized:
        d_2_float = d_2_float/255.0
    d_2_resized = tf.image.resize_images(d_2_float, [target_size, target_size])
    d_2_resized.set_shape((target_size, target_size, 1))

    i1 = tf.concat([r_1_resized, d_1_resized], axis=2)
    i2 = tf.concat([r_2_resized, d_2_resized], axis=2)

    u_c = input_queue[4]
    p_d = input_queue[5]
    theta_d = input_queue[6]
    l_d = input_queue[7]

    if include_type:
        type_d = input_queue[8]
        i1s, i2s, u_cs, p_ds, theta_ds, l_ds, type_ds = tf.train.batch(
            [i1, i2, u_c, p_d, theta_d, l_d, type_d],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_after_dequeue+3*batch_size)
        output = (i1s, i2s, u_cs, p_ds, theta_ds, l_ds, type_ds)

    else:
        i1s, i2s, u_cs, p_ds, theta_ds, l_ds = tf.train.batch(
            [i1, i2, u_c, p_d, theta_d, l_d],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_after_dequeue+3*batch_size)
        output = (i1s, i2s, u_cs, p_ds, theta_ds, l_ds)

    return output
