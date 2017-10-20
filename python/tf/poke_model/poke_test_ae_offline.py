# -*- coding: utf-8 -*-
"""
Test single AE with no RNN, AE with RNN, AE with RNN and AE with dynamic RNN with precollected
data. All testings are offline.
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
mpl.rcParams.update({'font.size': 22})

from poke_autoencoder import PokeAE
from poke_vae import PokeVAE
from poke_bn_vae import PokeBnVAE

from poke_ae_rnn import PokeAERNN, PokeVAERNN, PokeAEFFRNN, PokeVAERNNC
from poke_ae_rnn_multi import PokeMultiRNN # inheritate from PokeVAERNN.
from poke_rnn_multi_new import PokeMultiNew # inheritate same. self.pose used for sampling.

from poke_ae_stp import PokeAESTP_py
from poke_ae_cdna import PokeCDNA_py


win_width = 10
win_height = 10

def rect(ax, poke, c):
    x, y, t, l = poke
    dx = -200.0 * l * math.cos(t)
    dy = -200.0 * l * math.sin(t)
    ax.arrow(64.0-y, x, dy, dx, head_width=5, head_length=5, color=c)

def center_to_corner(position):
    p_x, p_y = position
    p_x = 64.0*p_x/240.0-win_width/2.0
    p_y = 64.0*p_y/240.0+win_height/2.0
    return p_x, p_y

def plot_multisample(imgs, imgs_rec, actions, positions, title, plot_start=False):
    """
    imgs/imgs_rec: list of backprop_step images.
    actions: list of backprop_step-1 actions (each as a list).
    positions: list of backprop_step positions (each as a list of x and y).
    title: a string.

    3 steps big: (15, 10) bottom top (0.1, 0.9)
    3 steps small: (7.5, 5) bottom top (0.1, 0.9)
    6 steps: (15, 5) bottom top (0.08, 1)
    """
    num = len(imgs)
    assert (num == len(actions)+1 and num==len(imgs_rec)), (
        'number of images or actions are not matching.'
    )
    fig, axarr = plt.subplots(2, num, figsize=(7.5, 5))
    label_list = ['$t_0$']+['$t_0+%i$'%i for i in range(1, num)]
    for i, ax in enumerate(axarr[0]):
        im = imgs[i][0]
        im = im/255.0
        p_x, p_y = center_to_corner(positions[i])
        ax.imshow(im)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if i==0:
            ax.add_patch(patches.Rectangle(
                (64.0-p_y, p_x), win_height, win_width, edgecolor='b', linewidth=2.0, fill=False))
        else:
            p_x0, p_y0 = center_to_corner(positions[i-1])
            ax.add_patch(patches.Rectangle(
                (64.0-p_y0, p_x0), win_height, win_width, edgecolor='b', linewidth=2.0, fill=False))
            ax.add_patch(patches.Rectangle(
                (64.0-p_y, p_x), win_height, win_width, edgecolor='r', linewidth=2.0, fill=False))

        if i < num-1:
            rect(ax, actions[i], "green")
    for i, ax in enumerate(axarr[1]):
        im = imgs_rec[i][0]
        im = im/im.max()
        p_x, p_y = center_to_corner(positions[i])
        ax.imshow(im)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xlabel(label_list[i])
        if i==0:
            if plot_start:
                p_xs = p_x
                p_ys = p_y
            ax.add_patch(patches.Rectangle(
                (64.0-p_y, p_x), win_height, win_width, edgecolor='b', linewidth=2.0, fill=False))
        else:
            if plot_start:
                ax.add_patch(patches.Rectangle(
                    (64.0-p_ys, p_xs), win_height, win_width, edgecolor='k', linewidth=2.0, fill=False))
            p_x0, p_y0 = center_to_corner(positions[i-1])
            ax.add_patch(patches.Rectangle(
                (64.0-p_y0, p_x0), win_height, win_width, edgecolor='b', linewidth=2.0, fill=False))
            ax.add_patch(patches.Rectangle(
                (64.0-p_y, p_x), win_height, win_width, edgecolor='r', linewidth=2.0, fill=False))
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0.1, top=0.9)
    #fig.suptitle('%s'%title)

def read_data_list_rnn_test(data_list, num_epochs, shuffle):
    assert type(shuffle)==bool, (
        'Requires a bool indicating shuffling or not'
    )
    with open(data_list, 'r') as f:
        images_1_path = []
        images_2_path = []
        images_3_path = []
        u1 = []
        u2 = []
        position1 = []
        position2 = []
        position3 = []
        for item in f:
            items = item.split(' ')
            images_1_path.append('../../poke/'+items[0])
            images_2_path.append('../../poke/'+items[1])
            images_3_path.append('../../poke/'+items[2])
            u1.append((float(items[3]), float(items[4]),
                        float(items[5]), float(items[6])))
            u2.append((float(items[7]), float(items[8]),
                        float(items[9]), float(items[10])))
            position1.append((float(items[11]), float(items[12])))
            position2.append((float(items[13]), float(items[14])))
            position3.append((float(items[15]), float(items[16])))

    i1_t = ops.convert_to_tensor(images_1_path, dtype=dtypes.string)
    i2_t = ops.convert_to_tensor(images_2_path, dtype=dtypes.string)
    i3_t = ops.convert_to_tensor(images_3_path, dtype=dtypes.string)
    u1_t = ops.convert_to_tensor(u1, dtype=dtypes.float32)
    u2_t = ops.convert_to_tensor(u2, dtype=dtypes.float32)
    p1_t = ops.convert_to_tensor(position1, dtype=dtypes.float32)
    p2_t = ops.convert_to_tensor(position2, dtype=dtypes.float32)
    p3_t = ops.convert_to_tensor(position3, dtype=dtypes.float32)
    input_queue = tf.train.slice_input_producer(
        [i1_t, i2_t, i3_t, u1_t, u2_t, p1_t, p2_t, p3_t],
        num_epochs=num_epochs,
        shuffle=shuffle)

    return input_queue

def batch_images_actions_rnn_test(input_queue, batch_size, num_threads, min_after_dequeue,
                                  type_img=1, normalized=0, ts=64):
    """
    If type_img 1 by default, rgb images. Otherwise depth image.
    Depth image data are actually stored with 16 bit integers.
    """
    image_1_file = tf.read_file(input_queue[0])
    image_2_file = tf.read_file(input_queue[1])
    image_3_file = tf.read_file(input_queue[2])
    u1 = input_queue[3]
    u2 = input_queue[4]
    p1 = input_queue[5]
    p2 = input_queue[6]
    p3 = input_queue[7]

    if type_img:
        image_1 = tf.image.decode_jpeg(image_1_file)
        image_1_float = tf.cast(image_1, tf.float32)
        if normalized:
            image_1_float=image_1_float/255.0
        image_1_resized = tf.image.resize_images(image_1_float, [ts, ts])
        image_1_resized.set_shape((ts, ts, 3))

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
        image_1_resized = tf.image.resize_images(image_1_float, [ts, ts])
        image_1_resized.set_shape((ts, ts, 1))

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

    images_1, images_2, images_3, u1s, u2s, p1s, p2s, p3s = tf.train.batch(
        [image_1_resized, image_2_resized, image_3_resized, u1, u2, p1, p2, p3],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=min_after_dequeue+3*batch_size)
    output = (images_1, images_2, images_3, u1s, u2s, p1s, p2s, p3s)

    return output

def ae_rnn_test():
    shuffle = True
    num_epochs = 1
    batch_size = 1
    num_threads = 1
    min_after_dequeue = 16

    type_img = 1
    in_channels = 3
    normalized=0

    path = '/home/wuyang/workspace/python/poke/test_data/'

    input_queue = read_data_list_rnn_test(
        '../../poke/test_cube_table_3.txt', num_epochs, shuffle)
    images_1, images_2, images_3, u1s, u2s, p1s, p2s, p3s = batch_images_actions_rnn_test(
        input_queue, batch_size, num_threads, min_after_dequeue,
        type_img=type_img, normalized=normalized, ts=240)

    step = 0
    with tf.Session() as sess:
        #poke_ae = PokeAERNN(batch_size=batch_size, split_size=512,
        #                    in_channels=3, corrupted=0,
        #                    is_training=False, lstm=1)
        # poke_ae = PokeVAERNN(batch_size=batch_size, split_size=512, in_channels=3, corrupted=0,
        #                      is_training=False, lstm=1)
        #poke_ae = PokeAEFFRNN(batch_size=batch_size, in_channels=3,
        #                      corrupted=0, is_training=False, lstm=1, vae=1)

        # poke_ae = PokeAESTP_py(batch_size=batch_size, in_channels=in_channels, is_training=False,
        #                        num_masks=6, upsample=1, use_bg=1)
        poke_ae = PokeCDNA_py(batch_size=batch_size, in_channels=in_channels, is_training=False,
                              num_masks=4, use_bg=1, adam=1)

        saver = tf.train.Saver()
        #restore_path = '../logs/pokeAERNN/rnn_dae(bn)/'
        #restore_path = '../logs/pokeAERNN/lstm_dae(bn)/'
        #restore_path = '../logs/pokeAERNN/rnn_vae/'
        #restore_path = '../logs/pokeAERNN/lstm_vae/'
        #restore_path = '../logs/pokeAERNN/lstm_vae_little/'

        #restore_path = '../logs/pokeAERNN/ff_rnn_dae(bn)/'
        #restore_path = '../logs/pokeAERNN/ff_lstm_dae(bn)/'
        #restore_path = '../logs/pokeAERNN/ff_lstm_vae/'

        #restore_path = '../logs/pokeAERNN_new/lstm_vae_14/'

        #restore_path = '../logs/pokeAESTP/stp_2/'
        restore_path = '../logs/pokeAECDNA/pyramid_new_2/'

        saver.restore(sess, tf.train.latest_checkpoint(restore_path))
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                imgs_feed, u_feed, positions = sess.run(
                    [[images_1, images_2, images_3], [u1s, u2s], [p1s, p2s, p3s]])

                positions_plot = []
                for item in positions:
                    p = item[0]
                    positions_plot.append([p[0], p[1]])

                actions_plot = []
                for item in u_feed:
                    u = item[0]
                    temp_list = [u[0]*64.0, u[1]*64.0, u[2]*np.pi*2, u[3]*0.04+0.04]
                    actions_plot.append(temp_list)

                feed_dict = {poke_ae.i1: imgs_feed[0],
                             poke_ae.i2: imgs_feed[1],
                             poke_ae.i3: imgs_feed[2],
                             poke_ae.u2: u_feed[0],
                             poke_ae.u3: u_feed[1]}

                imgs_rec, loss_total, i1r = sess.run(
                    [[poke_ae.decoding_1, poke_ae.decoding_2, poke_ae.decoding_3],
                     poke_ae.loss,
                     poke_ae.i1r],
                    feed_dict=feed_dict)

                errors = []
                imgs_feed[0] = i1r
                for n in range(3):
                    px, py = positions[n][0]
                    mean_error_t, _, _ = poke_ae.patch_average_error(
                        imgs_feed[n], imgs_rec[n], win_height, win_width, px/240.0, 1-py/240.0)
                    mean_error = sess.run(mean_error_t)
                    errors.append(mean_error)
                title_l = ['Error %d: %.2f;'%(n, item) for n, item in enumerate(errors)]
                title = ' '.join(title_l)

                plot_multisample(imgs_feed, imgs_rec, actions_plot, positions_plot, title)
                plt.pause(0.5)
                plt.waitforbuttonpress()
                #plt.savefig(path+'rnn%d.png'%step, format='png', dpi=600)
                plt.close()

                #with open(path+'error_rnn_offline.dat', 'ab') as f:
                #   np.savetxt(f, np.array([errors]))

                step+=1
                print 'step %d'%step
            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

def ae_rnn_sample():
    # sampling from the hidden layer. Change the identity and transition unit respectively
    # to uniform (0, 1). Then sample from the identity and transition and concatenate. Then
    # decodings are examined.
    shuffle = True
    num_epochs = 1
    batch_size = 1
    num_threads = 1
    min_after_dequeue = 16

    type_img = 1
    normalized=0
    split_size = 512

    path = '/home/wuyang/workspace/python/poke/test_data/'

    input_queue = read_data_list_rnn_test(
        '../../poke/test_cube_3.txt', num_epochs, shuffle)
    images_1, images_2, images_3, u1s, u2s, p1s, p2s, p3s = batch_images_actions_rnn_test(
        input_queue, batch_size, num_threads, min_after_dequeue,
        type_img=type_img, normalized=normalized)

    step = 0
    with tf.Session() as sess:
        poke_ae = PokeVAERNN(batch_size=batch_size, split_size=split_size,
                             in_channels=3, corrupted=0, is_training=False, lstm=1)
        saver = tf.train.Saver()
        restore_path = '../logs/pokeAERNN/lstm_vae/'

        saver.restore(sess, tf.train.latest_checkpoint(restore_path))
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step = 0
        try:
            while not coord.should_stop():
                imgs_feed, u_feed, positions = sess.run(
                    [[images_1, images_2, images_3], [u1s, u2s], [p1s, p2s, p3s]])

                positions_plot = []
                for item in positions:
                    p = item[0]
                    positions_plot.append([p[0], p[1]])

                actions_plot = []
                for item in u_feed:
                    u = item[0]
                    temp_list = [u[0]*64.0, u[1]*64.0, u[2]*np.pi*2, u[3]*0.04+0.04]
                    actions_plot.append(temp_list)

                feed_dict = {poke_ae.i1: imgs_feed[0],
                             poke_ae.i2: imgs_feed[1],
                             poke_ae.i3: imgs_feed[2],
                             poke_ae.u2: u_feed[0],
                             poke_ae.u3: u_feed[1]}

                samplings, mean, logss, imgs_rec, loss_total = sess.run(
                    [poke_ae.sampling,
                     poke_ae.mean,
                     poke_ae.logss,
                     [poke_ae.decoding_1, poke_ae.decoding_2, poke_ae.decoding_3],
                     poke_ae.loss],
                    feed_dict=feed_dict)
                iden_string = ''
                trans_string = ''
                half_decoding_list = []

                # option 1: sampling from mean and logss.
                temp_iden = tf.zeros_like(poke_ae.identity)
                fake_sampling, _, _ = poke_ae.hidden_sample(temp_iden, poke_ae.transit_series)
                #temp_trans = [tf.zeros_like(item) for item in poke_ae.transit_series]
                #fake_sampling, _, _ = poke_ae.hidden_sample(poke_ae.identity, temp_trans)
                iden_string += '(%.2f, %.2f)'%(np.mean(mean[0]), np.mean(logss[0]))
                for i, item in enumerate(zip(mean[1:], logss[1:])):
                    temp_mean, temp_logss = item
                    trans_string += '(%.2f, %.2f)'%(np.mean(temp_mean), np.mean(temp_logss))
                    sample = fake_sampling[i]
                    half_decoding_list.append(
                        poke_ae.decode(sample, poke_ae.feature_map_shape, True))
                half_decoding = sess.run(half_decoding_list, feed_dict=feed_dict)

                # option 2: directly take the sampling in the graph.
                # for item in samplings:
                #     iden, trans = np.split(item, [(1024-split_size)/2], axis=1)
                #     iden_string+='(%.2f, %.2f)'%(np.mean(iden), np.std(iden))
                #     trans_string+='(%.2f, %.2f)'%(np.mean(trans), np.std(trans))

                #     temp = tf.zeros([batch_size, (1024-split_size)/2])
                #     sample = tf.concat([temp, trans], axis=1)
                #     #temp = tf.zeros([batch_size, split_size/2])
                #     #sample = tf.concat([iden, temp], axis=1)
                #     decoded = poke_ae.decode(sample, poke_ae.feature_map_shape, True)
                #     half_decoding_list.append(decoded)
                # half_decoding = sess.run(half_decoding_list)

                title = iden_string+'; '+trans_string
                plot_multisample(imgs_feed, half_decoding, #imgs_rec,
                                 actions_plot, positions_plot, title,
                                 plot_start=True)
                #plt.pause(0.5)
                plt.savefig(path+'zero_iden_%d.png'%step, format='png', dpi=600)
                #plt.waitforbuttonpress()
                plt.close()

                step+=1
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')
        finally:
            coord.request_stop()
            coord.join(threads)
            print('number of enqueing %d'%step)

def ae_rnn_sanity_check():
    # Using the ground truth location of current image, cut out the same piece from the next images
    # and compare the squared errors of the patches.
    shuffle = True
    num_epochs = 1
    batch_size = 1
    num_threads = 1
    min_after_dequeue = 16

    type_img = 1
    normalized=0
    split_size = 512

    path = '/home/wuyang/workspace/python/poke/test_data/'

    input_queue = read_data_list_rnn_test(
        '../../poke/test_cube_3.txt', num_epochs, shuffle)
    images_1, images_2, images_3, u1s, u2s, p1s, p2s, p3s = batch_images_actions_rnn_test(
        input_queue, batch_size, num_threads, min_after_dequeue,
        type_img=type_img, normalized=normalized)

    step = 0
    with tf.Session() as sess:
        poke_ae = PokeVAERNN(batch_size=batch_size, split_size=split_size,
                             in_channels=3, corrupted=0, is_training=False, lstm=1)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step = 0
        try:
            while not coord.should_stop():
                imgs_feed, u_feed, positions = sess.run(
                    [[images_1, images_2, images_3], [u1s, u2s], [p1s, p2s, p3s]])

                positions_plot = []
                for item in positions:
                    p = item[0]
                    positions_plot.append([p[0], p[1]])

                actions_plot = []
                for item in u_feed:
                    u = item[0]
                    temp_list = [u[0]*64.0, u[1]*64.0, u[2]*np.pi*2, u[3]*0.04+0.04]
                    actions_plot.append(temp_list)

                imgs_shift = [np.ones([1,64,64,3])]
                imgs_shift.extend(imgs_feed[:-1])

                errors = []
                for n in range(1, 3):
                    px, py = positions[n][0]
                    mean_error_t, _, _ = poke_ae.patch_average_error(
                        imgs_feed[n], imgs_shift[n], win_height, win_width, px/240.0, 1-py/240.0)
                    mean_error = sess.run(mean_error_t)
                    errors.append(mean_error)
                # title_l = ['Error %d: %.2f'%(n+1, item) for n, item in enumerate(errors)]
                # title = ' '.join(title_l)
                # plot_multisample(imgs_feed, imgs_shift,
                #                  actions_plot, positions_plot, title)
                # plt.pause(0.5)
                # #plt.savefig(path+'rnn%d.png'%step, format='png', dpi=600)
                # plt.waitforbuttonpress()
                # plt.close()

                with open(path+'error_rnn_offline.dat', 'ab') as f:
                    np.savetxt(f, np.array([errors]))
                step+=1
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')
        finally:
            coord.request_stop()
            coord.join(threads)
            print('number of enqueing %d'%step)

def read_data_list_multi_test(data_list, num_epochs, shuffle, bp_steps):
    """
    image_path: [[i1_path, i2_path, i3_path...], [], []]
    label: [[[x1, y1, theta1, l1], [x2, y2, theta2, l2], [x3, y3, theta3, l3], ...],
    [[], [], [], ...],...]
    positions: [[[px1, py1], [px2, py2], ...], [[px1, py1], [px2, py2], ...], [], ...]
    """
    assert type(shuffle)==bool
    with open(data_list, 'r') as f:
        image_path = []
        label = []
        positions = []
        for item in f:
            items = item.split(';')
            im_paths = items[0].split(' ')
            image_path.append(['../../poke/'+im_path for im_path in im_paths])

            position_list = []
            for j, position in enumerate(items[bp_steps:]):
                temp_list = position.split(' ')
                float_list = [float(a) for a in temp_list]
                position_list.append(float_list)

            actions_list = []
            for j, action in enumerate(items[1:bp_steps]):
                temp_list = action.split(' ')
                float_list = [float(a) for a in temp_list]
                actions_list.append(float_list)

            positions.append(position_list)
            label.append(actions_list)

    i_t = ops.convert_to_tensor(image_path, dtype=dtypes.string)
    label_t = ops.convert_to_tensor(label, dtype=dtypes.float32)
    positions_t = ops.convert_to_tensor(positions, dtype=dtypes.float32)

    input_queue = tf.train.slice_input_producer(
        [i_t, label_t, positions_t], num_epochs=num_epochs, shuffle=shuffle)

    return input_queue

def batch_images_actions_multi_test(input_queue, batch_size, num_threads, min_after_dequeue,
                                    bp_steps, target_size=64, type_img=1,
                                    normalized=0, zero_action=1):
    path = input_queue[0] # bp_steps
    action = input_queue[1] # bp_steps-1 x 4
    position = input_queue[2] # 2 x 4

    shape0 = path.get_shape().as_list()
    shape1 = action.get_shape().as_list()
    assert shape0[0]==shape1[0]+1==bp_steps, (
        'Truncated time steps not matching'
    )
    if zero_action:
        action = tf.concat([tf.zeros([1, 4]), action], axis=0) # [5, 4] -> [6, 4]

    image_list = []

    paths = tf.split(path, bp_steps, axis=0)
    for item in paths:
        img_file = tf.read_file(item[0])
        if type_img:
            img = tf.image.decode_jpeg(img_file)
            img_float = tf.cast(img, tf.float32)
            if normalized:
                img_float = img_float/255.0#*2.0-1.0
            img_resized = tf.image.resize_images(img_float, [target_size, target_size])
            img_resized.set_shape((target_size, target_size, 3))
        else:
            img = tf.image.decode_png(img_file, dtype=tf.uint8)
            img_float = tf.cast(img, tf.float32)
            if normalized:
                img_float = img_float/255.0#*2.0-1.0
            img_resized = tf.image_resize_images(img_float, [target_size, target_size])
            img_resized.set_shape((target_size, target_size, 3))
        image_list.append(img_resized)
    image_stack = tf.stack(image_list, axis=0) #-> [bp_steps, target_size, target_size, 3 or 1]

    images, actions, positions = tf.train.batch([image_stack, action, position],
                                     batch_size,
                                     num_threads,
                                     capacity=min_after_dequeue+3*batch_size)
    output = (images, actions, positions)
    return output

def ae_rnn_multi_test(bp_steps):
    """
    working trained model of PokeMultiRNN and PokeMultiNew are trained on non normalized data.
    For PokeMultiRNN, old code normalize data into (-1, 1) but it is never used.
    For PokeMultiNew, new code normalize data into (0, 1) used.
    Since we are not normalizing anyway, this difference does not really matter.

    pokeAERNN: old dataset with one background.
    pokeAERNN_new: new dataset
    new in pokeAERNN_new: new model where first encoding is directly used for sampling.
    """
    shuffle = True
    num_epochs = 1
    batch_size = 1
    num_threads = 1
    min_after_dequeue = 16

    type_img = 1
    zero_action = 0
    if zero_action:
        u_start=1
    else:
        u_start=0

    path = '/home/wuyang/workspace/python/poke/test_data/'

    input_queue = read_data_list_multi_test(
        '../../poke/test_multi_cube_table_6.txt', num_epochs, shuffle, bp_steps)
    images, actions, positions = batch_images_actions_multi_test(
        input_queue, batch_size, num_threads, min_after_dequeue, bp_steps, target_size=64,
        type_img=type_img, normalized=0, zero_action=zero_action)

    step = 0
    with tf.Session() as sess:
        #poke_ae = PokeMultiRNN(batch_size=batch_size, split_size=128,
        #                       in_channels=3, corrupted=0,
        #                       is_training=False, bp_steps=bp_steps)

        poke_ae = PokeMultiNew(batch_size=batch_size, split_size=128,
                               in_channels=3, corrupted=0,
                               is_training=False, bp_steps=bp_steps)

        saver = tf.train.Saver(max_to_keep=2)

        #restore_path = '../logs/pokeAERNN/6_lstm_vae_8ep/'
        #restore_path = '../logs/pokeAERNN_new/6_lstm_vae_little/'
        restore_path = '../logs/pokeAERNN_new/new_6_lstm_vae_little/'
        saver.restore(sess, tf.train.latest_checkpoint(restore_path))
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                imgs_feed, u_feed, p_feed = sess.run([images, actions, positions])

                temp_list = np.split(imgs_feed, bp_steps, axis=1)
                imgs_list = [np.squeeze(item, axis=1) for item in temp_list]

                positions_plot = []
                for row in p_feed[0]:
                    positions_plot.append([row[0], row[1]])

                actions_plot = [] # bp_steps-1 actions from 1xbp_stepsx4.
                # four zeros at u_feed[0][0]
                for row in u_feed[0][u_start:]:
                    temp_list = [row[0]*64.0, row[1]*64.0, row[2]*np.pi*2, row[3]*0.04+0.04]
                    actions_plot.append(temp_list)

                feed_dict = {poke_ae.i: imgs_feed, poke_ae.u: u_feed}

                imgs_rec, loss_total = sess.run(
                    [poke_ae.decoding_list, poke_ae.loss], feed_dict=feed_dict)

                errors = []
                for n, row in enumerate(p_feed[0]):
                    px, py = row
                    mean_error_t, _, _ = poke_ae.patch_average_error(
                        imgs_list[n], imgs_rec[n], win_height, win_width, px/240.0, 1-py/240.0)
                    mean_error = sess.run(mean_error_t)
                    errors.append(mean_error)
                title_l = ['Error %d: %.2f'%(n, item) for n, item in enumerate(errors)]
                title = ' '.join(title_l)

                #plot_multisample(imgs_list, imgs_rec, actions_plot, positions_plot, title)
                #plt.pause(0.5)
                #plt.savefig(path+'rnn%d.png'%step, format='png', dpi=600)
                #plt.waitforbuttonpress()
                #plt.close()

                with open(path+'error_multi_rnn_off.dat', 'ab') as f:
                    np.savetxt(f, np.array([errors]))

                print step
                step+=1

            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')

        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)


if __name__ == '__main__':
    ae_rnn_test()
    #ae_rnn_multi_test(6)

    #ae_rnn_sample()
    #ae_rnn_sanity_check()
