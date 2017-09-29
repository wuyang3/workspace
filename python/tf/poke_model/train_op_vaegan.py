# -*- coding: utf-8 -*-
"""
based on vae with BN, this script trains vaegan.
"""

import numpy as np
import tensorflow as tf
from poke_vaegan import PokeVAEGAN
from poke_vaewgan import PokeVAEWGAN
from poke_vaewgangp import PokeVAEWGANGP
from batch_operation import read_data_list_ae
from batch_operation import batch_images_actions_ae

def main():
    shuffle = True
    num_epochs = 6
    batch_size = 8
    num_threads = 4
    min_after_dequeue = 256

    type_img = 1 #1\0
    split_size = 256
    in_channels = 3 #3\1
    corrupted = 0 #1\0
    # Reads pathes of data and create a input queue.
    #input_queue = read_data_list_ae(
    #    '../../poke/train_gazebo_ae.txt', num_epochs, shuffle)
    #input_queue = read_data_list_ae(
    #    '../../poke/train_gazebo_vae.txt', num_epochs, shuffle)
    input_queue = read_data_list_ae(
        '../../poke/train_cube_ae.txt', num_epochs, shuffle)

    images_1, images_2, u_cs = batch_images_actions_ae(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img, normalized=1)

    # Initial training parameters.
    learning_rate = 0.0001
    epsilon = 1e-6
    step = 0

    train_mode = 0 #1\0 WGAN, GAN
    Discriminator_loop = 5

    with tf.Session() as sess:
        if train_mode:
            poke_ae = PokeVAEWGAN(learning_rate, epsilon, batch_size, split_size,
                                  in_channels, corrupted, is_training=True)
            #poke_ae = PokeVAEWGANGP(learning_rate, epsilon, batch_size, split_size,
            #                        in_channels, corrupted, is_training=True)
        else:
            poke_ae = PokeVAEGAN(learning_rate, epsilon, batch_size, split_size,
                                 in_channels, corrupted, is_training=True)
        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeAE/', sess.graph)

        #saver.restore(sess, tf.train.latest_checkpoint('../logs/pokeAE/rs_49/'))
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            if train_mode:
                while not coord.should_stop():
                    i1, i2, uc = sess.run([images_1, images_2, u_cs])
                    feed_dict = {poke_ae.i1: i1, poke_ae.i2: i2,
                                 poke_ae.u_labels: uc}

                    _ = sess.run(poke_ae.train_op_e, feed_dict=feed_dict)
                    for j in range(Discriminator_loop):
                        _, _ = sess.run([poke_ae.train_op_d, poke_ae.clip_d], feed_dict=feed_dict)
                    _, loss, summary = sess.run(
                        [poke_ae.train_op_g, poke_ae.loss, poke_ae.merged], feed_dict=feed_dict)

                    if step%10 == 0:
                        train_writer.add_summary(summary, step)
                        print('step %d: loss E->%.4f loss G->%.4f loss D->%.4f'
                              %(step, loss['E'], loss['G'], loss['D']))

                    if step%1000 == 0:
                        saver.save(sess, '../logs/pokeAE/', global_step=step)
                    step+=1
                train_writer.close()
            else:
                while not coord.should_stop():
                    i1, i2, uc = sess.run([images_1, images_2, u_cs])

                    feed_dict = {poke_ae.i1: i1, poke_ae.i2: i2,
                                 poke_ae.u_labels: uc}

                    _ = sess.run(poke_ae.train_op_e, feed_dict=feed_dict)
                    _ = sess.run(poke_ae.train_op_g, feed_dict=feed_dict)
                    _ = sess.run(poke_ae.train_op_g, feed_dict=feed_dict)
                    _, loss, summary = sess.run(
                        [poke_ae.train_op_d, poke_ae.loss, poke_ae.merged],
                        feed_dict=feed_dict)

                    if step%10 == 0:
                        train_writer.add_summary(summary, step)
                        print('step %d: loss E->%.4f loss G->%.4f loss D->%.4f'
                              %(step, loss['E'], loss['G'], loss['D']))

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


if __name__ == '__main__':
    main()
