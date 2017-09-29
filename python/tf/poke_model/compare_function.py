# -*- coding: utf-8 -*-
"""
Train a compact network that regress on the pose of the block. This file include:
Network class, data processing function, data reading and enqueing function, training function.
"""
import glob
import tf as tff
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
import numpy as np
from poke_autoencoder import ConvAE

class CompareFunction(ConvAE):
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 in_channels=3, is_training=True, reuse=False):
        self.in_channels = in_channels
        self.is_training = is_training

        self.i =tf.placeholder(tf.float32, [batch_size, 64, 64, in_channels])
        self.regression = self.regress(self.i, reuse=reuse)

        self.label = tf.placeholder(tf.float32, [batch_size, 3])
        tf.summary.histogram('label', self.label)

        self.loss = self.loss_function(self.regression, self.label)
        self.train_op = self.train(self.loss, learning_rate, epsilon)
        self.merged = tf.summary.merge_all()

    def regress(self, i, reuse):
        with tf.variable_scope('regression', reuse=reuse):
            conv1 = self.conv_layer(
                i, [5, 5], [self.in_channels, 32], stride=1, initializer_type=1, name='conv1')
            conv2 = self.conv_bn_layer(
                conv1, [5, 5], [32, 64], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv2')
            conv3 = self.conv_bn_layer(
                conv2, [5, 5], [64, 128], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv3')
            conv4 = self.conv_bn_layer(
                conv3, [5, 5], [128, 256], stride=2, is_training=self.is_training,
                initializer_type=1, name='conv4')
            shape = conv4.get_shape().as_list()
            size = shape[1]*shape[2]*shape[3]
            conv4_flat = tf.reshape(conv4, [-1, size], 'conv4_flat')
            fc5 = self.fc_bn_layer(
                conv4_flat, 512, is_training=self.is_training, initializer_type=1, name='fc5')
            #fc6 = self.fc_bn_layer(
            #    fc5, 512, is_training=self.is_training, initializer_type=1, name='fc6')
            fc7 = self.fc_layer(fc5, 3, initializer_type=1, name='fc7')
        tf.summary.histogram('regression', fc7)
        return fc7

    def loss_function(self, regression, label):
        with tf.variable_scope('loss'):
            shape1 = regression.get_shape().as_list()
            shape2 = label.get_shape().as_list()
            assert shape1==shape2, (
                'regression and label have different dimensions.'
            )
            loss = tf.reduce_mean(tf.square(regression - label))
        tf.summary.scalar('loss', loss)
        return loss

    def train(self, loss, learning_rate, epsilon):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)

        return train_op

def write_regression_data():
    """
    Pixels should be normalized by 240. Yaw is within [-pi, pi] and transformed into [0, pi/2]
    """
    def bound_to_float(s):
        value = float(s)
        if value < 0.0:
            value = 0.0
        elif value >= 240.0:
            value = 240.0
        else:
            pass
        return value

    train_dir = sorted(glob.glob('/home/wuyang/workspace/python/poke/train_cube/run*'))
    train_dir_total = []

    min_yaw = 0
    max_yaw = 0

    for i, i_dir in enumerate(train_dir):
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))
        rgb_num = [int(item[-8:-4]) for item in rgb_dir]

        train_dir_aug = []

        with open(i_dir+'/actions.dat') as f:
            actions = [line.strip().split() for line in f.readlines()]

        rgb_0 = rgb_dir[0]
        ind = rgb_0.index('ke')+3
        for j, item in enumerate(rgb_num):
            action = actions[item]
            dx = bound_to_float(action[2])/240.0
            dy = bound_to_float(action[3])/240.0

            qx = float(action[8])
            qy = float(action[9])
            qz = float(action[10])
            qw = float(action[11])

            euler = tff.transformations.euler_from_quaternion((qx, qy, qz, qw))

            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            if yaw < 0:
                yaw = yaw+np.pi
            if yaw > np.pi/2:
                yaw = yaw - np.pi/2
            yaw_normalized = yaw/(np.pi/2)
            #yaw_normalized = (yaw+np.pi)/(2*np.pi)
            # yaw range [-3.141590, 3.141587]
            #if yaw < min_yaw:
            #    min_yaw = yaw
            #if yaw > max_yaw:
            #    max_yaw= yaw

            #'%.4e'%roll, '%.4e'%pitch, 
            train_dir_aug.append(
                ' '.join([rgb_dir[j][ind:], '%.4e'%dx, '%.4e'%dy, '%.4e'%yaw_normalized]))

        print('image until %s included'%rgb_dir[j][ind:])
        train_dir_total.extend(train_dir_aug)

    with open('/home/wuyang/workspace/python/poke/compare.txt', 'wb') as f:
        for item in train_dir_total:
            f.write('%s\n'%item)

def read_data_list_compare(data_list, num_epochs, shuffle):
    assert type(shuffle)==bool
    with open(data_list, 'r') as f:
        image_path = []
        label = []
        for item in f:
            items = item.split(' ')
            image_path.append('../../poke/'+items[0])
            label.append((float(items[1]), float(items[2]), float(items[3])))
    i_t = ops.convert_to_tensor(image_path, dtype=dtypes.string)
    label_t = ops.convert_to_tensor(label, dtype=dtypes.float32)

    input_queue = tf.train.slice_input_producer(
        [i_t, label_t], num_epochs=num_epochs, shuffle=shuffle)

    return input_queue

def batch_images_actions_compare(input_queue, batch_size, num_threads, min_after_dequeue,
                                 type_img=1, normalized=0):
    image_file = tf.read_file(input_queue[0])
    label = input_queue[1]

    if type_img:
        image = tf.image.decode_jpeg(image_file)
        image_float = tf.cast(image, tf.float32)
        if normalized:
            image_float=image_float/255.0*2.0-1.0
        image_resized = tf.image.resize_images(image_float, [64, 64])
        image_resized.set_shape((64, 64, 3))

    else:
        image = tf.image.decode_png(image_file, dtype=tf.uint8)
        image_float = tf.cast(image, tf.float32)
        if normalized:
            image_float=image_float/255.0*2.0-1.0
        image_resized = tf.image.resize_images(image_float, [64, 64])
        image_resized.set_shape((64, 64, 1))

    images, labels = tf.train.batch(
        [image_resized, label], batch_size=batch_size, num_threads=num_threads,
        capacity=min_after_dequeue+3*batch_size)

    output = (images, labels)
    return output

def compare_train():
    shuffle = True
    num_epochs = 6
    batch_size = 16
    num_threads = 4
    min_after_dequeue = 128

    type_img = 1 #1\0
    in_channels = 3 #3\1

    learning_rate = 0.0002
    epsilon = 1e-05
    step = 0

    input_queue = read_data_list_compare(
        '../../poke/compare.txt', num_epochs, shuffle)

    images, labels = batch_images_actions_compare(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img,
        normalized=0)

    with tf.Session() as sess:
        poke_compare = CompareFunction(learning_rate, epsilon, batch_size,
                                       in_channels, is_training=True, reuse=False)
        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeCompare/', sess.graph)

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                i, l = sess.run([images, labels])

                feed_dict = {poke_compare.i: i, poke_compare.label: l}

                _, loss, summary = sess.run(
                    [poke_compare.train_op, poke_compare.loss, poke_compare.merged],
                    feed_dict=feed_dict)

                if step%10 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%.4f' %(step, loss))

                if step%1000 == 0:
                    saver.save(sess, '../logs/pokeCompare/', global_step=step)
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

if __name__ == '__main__':
    #write_regression_data()
    compare_train()
