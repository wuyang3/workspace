# -*- coding: utf-8 -*-
"""
Train an multiple step rnn with autoencoder.
Newer version with self.pose directly used for sampling for i0. No zero action is fed to LSTM.
In order to 
"""
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
import numpy as np
import os
from poke_autoencoder import ConvAE
from poke_ae_rnn import PokeVAERNN

class PokeMultiNew(PokeVAERNN):
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0,
                 is_training=True, bp_steps=6):
        self.in_channels = in_channels
        self.is_training = is_training
        self.bp_steps = bp_steps
        self.i = tf.placeholder(tf.float32, [batch_size, bp_steps, 64, 64, in_channels])
        self.u = tf.placeholder(tf.float32, [batch_size, None, 4])

        i_list = tf.split(self.i, bp_steps, axis=1)
        self.i_list = [tf.squeeze(item, axis=1) for item in i_list]
        self.i0 = self.i_list[0]
        for j, item in enumerate(self.i_list):
            tf.summary.image('image%d'%j, item)

        self.identity, self.pose, self.feature_map_shape = self.encode_new(
            self.i0, split_size)
        self.transit_series = self.Recurrent_dynamic_transit(
            self.u, self.pose, reuse=False, num_split=bp_steps-1)

        self.sampling, mean, logss = self.hidden_sample_new(
            self.identity, self.pose, self.transit_series)

        self.decoding_list = []
        for j, sample in enumerate(self.sampling):
            reuse = True
            if j==0:
                reuse=False
            decoding = self.decode(sample, self.feature_map_shape, reuse)
            self.decoding_list.append(decoding)
            tf.summary.image('image_rec%d'%j, decoding)

        self.loss = self.loss_multi_function(self.i_list, self.decoding_list, mean, logss)
        self.train_op = self.train(self.loss, learning_rate, epsilon)

        # a temporal reusing scope for quicker inference during testing.
        # self.transit_temp = self.Recurrent_dynamic_transit(
        #    self.u, self.pose, reuse=True, num_split=1)
        # self.transit_temp_l = self.Recurrent_dynamic_transit(
        #    self.u, self.pose, reuse=True, num_split=5)

        self.merged = tf.summary.merge_all()

    def encode_new(self, img, split_size):
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
            fc6 = self.fc_bn_layer(conv5_flat, 1024, is_training=self.is_training,
                                   initializer_type=1, name='fc6')
            fc7 = self.fc_no_activation(fc6, 1024, initializer_type=1, name='fc7')
            identity, pose = tf.split(fc7,
                                      num_or_size_splits=[1024-split_size, split_size],
                                      axis=1)
        return identity, pose, shape

    def Recurrent_dynamic_transit(self, inputs, hidden_states, reuse=False, num_split=5):
        batch_size, split_size = hidden_states.get_shape().as_list()
        #assert bp_steps==self.bp_steps, (
        #    'Input tensor has wrong number of steps.'
        #)
        transit_series = []
        with tf.variable_scope('LSTM', reuse=reuse):
            cell_states = tf.zeros_like(hidden_states)
            init_layer1_state = tf.contrib.rnn.LSTMStateTuple(cell_states, hidden_states)
            init_layer2_state = tf.contrib.rnn.LSTMStateTuple(cell_states, hidden_states)
            init_tuple_states = tuple([init_layer1_state, init_layer2_state])

            cells = [tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True),
                     tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True)]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            state_series, current_states = tf.nn.dynamic_rnn(
                cell, inputs, initial_state=init_tuple_states)

            temp_list = tf.split(state_series, num_split, axis=1)
            state_list = [tf.squeeze(item, axis=1) for item in temp_list]
            for j, item in enumerate(state_list):
                transit = self.fc_no_activation(
                    item, split_size, initializer_type=1, name='transit%d'%j)
                transit_series.append(transit)

        return transit_series

    def hidden_sample_new(self, identity, pose, transit_series):
        assert type(transit_series)==list, (
            'Input is not a list of transitted tensors.'
        )
        sampling = []
        mean = []
        logss = []
        with tf.variable_scope('sampling'):
            z_mean_i, z_log_sigma_sq_i = tf.split(identity, 2, 1)
            mean.append(z_mean_i)
            logss.append(z_log_sigma_sq_i)
            z_i = self.hidden_code_sample(z_mean_i, z_log_sigma_sq_i, 'sampling_i')

            z_mean_p, z_log_sigma_sq_p = tf.split(pose, 2, 1)
            mean.append(z_mean_p)
            logss.append(z_log_sigma_sq_p)
            z_p = self.hidden_code_sample(z_mean_p, z_log_sigma_sq_p, 'sampling_p')

            z = tf.concat([z_i, z_p], 1)
            sampling.append(z)
            tf.summary.histogram('zi', z)
            for j, item in enumerate(transit_series):
                z_mean_t, z_log_sigma_sq_t = tf.split(item, 2, 1)
                z_t = self.hidden_code_sample(z_mean_t, z_log_sigma_sq_t, 'sampling_%d'%j)
                z = tf.concat([z_i, z_t], 1)
                sampling.append(z)
                tf.summary.histogram('z%d'%j, z)

                mean.append(z_mean_t)
                logss.append(z_log_sigma_sq_t)
        return sampling, mean, logss

    def loss_multi_function(self, i_list, decoding_list, mean, logss):
        assert len(mean)==len(logss) and len(i_list)==len(decoding_list), (
            'Steps not matching'
        )
        with tf.variable_scope('loss'):
            with tf.variable_scope('rec'):
                loss_rec = 0
                for j in range(len(i_list)):
                    loss_rec+=tf.reduce_sum(tf.square(i_list[j]-decoding_list[j]), [1, 2, 3])
            with tf.variable_scope('kl'):
                assert len(mean)==len(logss)
                loss_kl = 0
                for j in range(len(mean)):
                    loss_kl+= -0.5*tf.reduce_sum(1+logss[j]-tf.square(mean[j])-tf.exp(logss[j]), 1)

            # loss averaged over batches thus different from pixel wise diferrence.
            loss = tf.reduce_mean(loss_rec + loss_kl)
            tf.summary.scalar('loss_vae', loss)
            tf.summary.scalar('loss_rec', tf.reduce_mean(loss_rec))
            tf.summary.scalar('loss_kl', tf.reduce_mean(loss_kl))
        return loss

class PokeMultiNew_s(PokeMultiNew):
    def __init__(self, learning_rate=1e-4, epsilon=1e-7, batch_size=16,
                 split_size=128, in_channels=3, corrupted=0,
                 is_training=True, bp_steps=6):
        self.in_channels = in_channels
        self.is_training = is_training
        self.bp_steps = bp_steps
        self.i = tf.placeholder(tf.float32, [batch_size, bp_steps, 64, 64, in_channels])
        self.u = tf.placeholder(tf.float32, [batch_size, None, 4])

        i_list = tf.split(self.i, bp_steps, axis=1)
        self.i_list = [tf.squeeze(item, axis=1) for item in i_list]
        self.i0 = self.i_list[0]
        for j, item in enumerate(self.i_list):
            tf.summary.image('image%d'%j, item)

        self.identity, self.pose, self.feature_map_shape = self.encode_new(
            self.i0, split_size)
        self.z, z_i, self.pose, mean, logss = self.hidden_sample_new(self.identity, self.pose)

        self.transit_series = self.Recurrent_dynamic_transit(
            self.u, self.pose, reuse=False, num_split=bp_steps-1)

        self.decoding_list = []
        decoding = self.decode(self.z, self.feature_map_shape, False)
        self.decoding_list.append(decoding)
        for j, transit in enumerate(self.transit_series):
            tran_t = tf.concat([z_i, transit], axis=1)
            reuse = True
            decoding = self.decode(tran_t, self.feature_map_shape, reuse)
            self.decoding_list.append(decoding)
            tf.summary.image('image_rec%d'%j, decoding)

        self.loss = self.loss_multi_function(self.i_list, self.decoding_list, mean, logss)
        self.train_op = self.train(self.loss, learning_rate, epsilon)

        # a temporal reusing scope for quicker inference during testing.
        self.transit_temp = self.Recurrent_dynamic_transit(
            self.u, self.pose, reuse=True, num_split=1)
        self.transit_temp_l = self.Recurrent_dynamic_transit(
            self.u, self.pose, reuse=True, num_split=5)

        self.merged = tf.summary.merge_all()

    def Recurrent_dynamic_transit(self, inputs, hidden_states, reuse=False, num_split=5):
        batch_size, split_size = hidden_states.get_shape().as_list()
        print 'split size %d'%split_size
        #assert bp_steps==self.bp_steps, (
        #    'Input tensor has wrong number of steps.'
        #)
        transit_series = []
        with tf.variable_scope('LSTM', reuse=reuse):
            cell_states = tf.zeros_like(hidden_states)
            init_layer1_state = tf.contrib.rnn.LSTMStateTuple(cell_states, hidden_states)
            #init_tuple_states = tuple([init_layer1_state])

            cell = tf.contrib.rnn.LSTMCell(split_size, state_is_tuple=True)
            state_series, current_states = tf.nn.dynamic_rnn(
                cell, inputs, initial_state=init_layer1_state)

            temp_list = tf.split(state_series, num_split, axis=1)
            state_list = [tf.squeeze(item, axis=1) for item in temp_list]
            # for j, item in enumerate(state_list):
            #     #transit = self.fc_no_activation(
            #     #    item, split_size, initializer_type=1, name='transit%d'%j)
            #     total = tf.concat([iden, item], axis=1)
            #     print 'total size ', total.get_shape().as_list()
            #     transit_series.append(total)

        return state_list

    def hidden_sample_new(self, identity, pose):
        mean = []
        logss = []
        with tf.variable_scope('sampling'):
            z_mean_i, z_log_sigma_sq_i = tf.split(identity, 2, 1)
            mean.append(z_mean_i)
            logss.append(z_log_sigma_sq_i)
            z_i = self.hidden_code_sample(z_mean_i, z_log_sigma_sq_i, 'sampling_i')

            z_mean_p, z_log_sigma_sq_p = tf.split(pose, 2, 1)
            mean.append(z_mean_p)
            logss.append(z_log_sigma_sq_p)
            z_p = self.hidden_code_sample(z_mean_p, z_log_sigma_sq_p, 'sampling_p')

            z = tf.concat([z_i, z_p], 1)
            tf.summary.histogram('zi', z)
        return z, z_i, z_p, mean, logss

    def loss_multi_function(self, i_list, decoding_list, mean, logss):
        assert len(mean)==len(logss) and len(i_list)==len(decoding_list), (
            'Steps not matching'
        )
        with tf.variable_scope('loss'):
            with tf.variable_scope('rec'):
                loss_rec = 0
                for j in range(len(i_list)):
                    loss_rec+=tf.reduce_sum(tf.square(i_list[j]-decoding_list[j]), [1, 2, 3])
            with tf.variable_scope('kl'):
                assert len(mean)==len(logss)
                loss_kl = 0
                for j in range(len(mean)):
                    loss_kl+= -0.5*tf.reduce_sum(1+logss[j]-tf.square(mean[j])-tf.exp(logss[j]), 1)

            # loss averaged over batches thus different from pixel wise diferrence.
            loss = tf.reduce_mean(loss_rec + loss_kl)
            tf.summary.scalar('loss_vae', loss)
            tf.summary.scalar('loss_rec', tf.reduce_mean(loss_rec))
            tf.summary.scalar('loss_kl', tf.reduce_mean(loss_kl))
        return loss

def read_data_list_multi(data_list, num_epochs, shuffle, bp_steps):
    """
    image_path: [[i1_path, i2_path, i3_path...], [], []]
    label: [[[x1, y1, theta1, l1], [x2, y2, theta2, l2], [x3, y3, theta3, l3], ...],
    [[], [], [], ...],...]
    """
    assert type(shuffle)==bool
    with open(data_list, 'r') as f:
        image_path = []
        label = []
        for item in f:
            items = item.split(';')
            im_paths = items[0].split(' ')
            image_path.append(['../../poke/'+im_path for im_path in im_paths])

            actions_list = []
            for j, action in enumerate(items[1:]):
                temp_list = action.split(' ')
                float_list = [float(a) for a in temp_list]
                actions_list.append(float_list)

            label.append(actions_list)

    i_t = ops.convert_to_tensor(image_path, dtype=dtypes.string)
    label_t = ops.convert_to_tensor(label, dtype=dtypes.float32)

    input_queue = tf.train.slice_input_producer(
        [i_t, label_t], num_epochs=num_epochs, shuffle=shuffle)

    return input_queue

def batch_images_actions_multi(input_queue, batch_size, num_threads, min_after_dequeue,
                               bp_steps, target_size=64, type_img=1, normalized=0):
    path = input_queue[0]
    action = input_queue[1]

    shape0 = path.get_shape().as_list()
    shape1 = action.get_shape().as_list()
    assert shape0[0]==shape1[0]+1==bp_steps, (
        'Truncated time steps not matching'
    )
    #action_full = tf.concat([tf.zeros([1, 4]), action], axis=0) # [5, 4] -> [6, 4]

    image_list = []

    paths = tf.split(path, bp_steps, axis=0)
    for item in paths:
        img_file = tf.read_file(item[0])
        if type_img:
            img = tf.image.decode_jpeg(img_file)
            img_float = tf.cast(img, tf.float32)
            if normalized:
                img_float = img_float/255.0 # or 0-1?
            img_resized = tf.image.resize_images(img_float, [target_size, target_size])
            img_resized.set_shape((target_size, target_size, 3))
        else:
            img = tf.image.decode_png(img_file, dtype=tf.uint8)
            img_float = tf.cast(img, tf.float32)
            if normalized:
                img_float = img_float/255.0
            img_resized = tf.image_resize_images(img_float, [target_size, target_size])
            img_resized.set_shape((target_size, target_size, 3))
        image_list.append(img_resized)
    image_stack = tf.stack(image_list, axis=0) #-> [bp_steps, target_size, target_size, 3 or 1]

    images, actions = tf.train.batch([image_stack, action],
                                     batch_size,
                                     num_threads,
                                     capacity=min_after_dequeue+3*batch_size)
    output = (images, actions)
    return output

def ae_rnn_multi_train(num_epochs=12, batch_size=16, split_size=128,
                       learning_rate=0.0002, epsilon=1e-05, j=1):
    """
    Default training parameters will be used and overrided when doing random search.
    """
    path = '../logs/pokeAERNN_new'
    #path = '../logs/pokeAERNN'
    if not os.path.exists(path+'/rs_%02d/'%j):
        os.makedirs(path+'/rs_%02d/'%j)

    tf.reset_default_graph()

    shuffle = True
    bp_steps = 6
    num_threads = 4
    min_after_dequeue = 256
    type_img = 1 #1\0

    input_queue = read_data_list_multi(
        #'../../poke/train_rnn_6_steps.txt', num_epochs, shuffle, bp_steps)
        '../../poke/train_cube_table_rnn_6.txt', num_epochs, shuffle, bp_steps)

    images, actions = batch_images_actions_multi(
        input_queue, batch_size, num_threads, min_after_dequeue, bp_steps,
        target_size=64, type_img=type_img, normalized=0)

    # Initial training parameters.
    in_channels = 3 #3\1
    corrupted = 0 #1\0
    step = 0

    with tf.Session() as sess:
        # poke_ae = PokeMultiNew(learning_rate, epsilon, batch_size=batch_size,
        #                        split_size=split_size, in_channels=in_channels,
        #                        corrupted=corrupted, is_training=True, bp_steps=bp_steps)

        poke_ae = PokeMultiNew_s(learning_rate, epsilon, batch_size=batch_size,
                                 split_size=split_size, in_channels=in_channels,
                                 corrupted=corrupted, is_training=True, bp_steps=bp_steps)

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter(path+'/rs_%02d/'%j, sess.graph)

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                i, u = sess.run([images, actions])

                feed_dict = {poke_ae.i: i, poke_ae.u: u}

                _, loss, summary = sess.run(
                    [poke_ae.train_op, poke_ae.loss, poke_ae.merged],
                    feed_dict=feed_dict)

                if step%10 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%.4f' %(step, loss))

                if step%1000 == 0:
                    saver.save(sess, path+'/rs_%02d/'%j, global_step=step)
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

def generate_hyperparas():
    learning_rate = 10 ** np.random.uniform(-5, -3) # First search: (-5, -2)
    epsilon = 10 ** np.random.uniform(-7, -2)
    batch_size = 2 ** np.random.choice(np.array([3, 4, 5, 6]))
    idx = np.random.randint(3)
    split_sizes = [128, 256, 512]
    split_size = split_sizes[idx]
    return batch_size, split_size, learning_rate, epsilon

def random_search():
    for i in range(13, 30):
        batch_size, split_size, learning_rate, epsilon = generate_hyperparas()
        print('\nbatch size=%d, split_size=%s, learning rate=%s, epsilon=%s'
              %(batch_size, split_size, learning_rate, epsilon))
        ae_rnn_multi_train(num_epochs=2,
                           batch_size=batch_size,
                           split_size=split_size,
                           learning_rate=learning_rate,
                           epsilon=epsilon, j=i)
        with open('../logs/pokeAERNN/rs_%02d/info.txt'%i, 'w') as f:
            f.writelines('num_epochs = %d\n'%2)
            f.writelines('batch_size = %d\n'%batch_size)
            f.writelines('split_size = %d\n'%split_size)
            f.writelines('learning rate = %s\n'%learning_rate)
            f.writelines('epsilon = %s\n'%epsilon)

if __name__ == '__main__':
    ae_rnn_multi_train()
    #random_search()
