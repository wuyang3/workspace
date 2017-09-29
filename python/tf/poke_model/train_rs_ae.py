# -*- coding: utf-8 -*-
"""
Random hyper parameter search for auto-encoder training on object poking data.
"""
import numpy as np
import sys
import os
import tensorflow as tf
import random
from poke_autoencoder import PokeAE
from poke_vaegan import PokeVAEGAN
from batch_operation import read_data_list_ae
from batch_operation import batch_images_actions_ae

def generate_hyperparas():
    num_epochs = random.randint(1, 2)
    learning_rate = 10 ** np.random.uniform(-6, -3) # First search: (-5, -2)
    epsilon = 10 ** np.random.uniform(-7, -4)
    batch_size = 2 ** np.random.choice(np.array([3, 4, 5, 6]))
    idx = random.randint(0, 2)
    split_sizes = [128, 256, 512]
    split_size = split_sizes[idx]
    return num_epochs, batch_size, learning_rate, epsilon, split_size

def train_func(num_epochs, batch_size, learning_rate, epsilon, split_size, i):
    if not os.path.exists('../logs/pokeAE/rs_'+'%02d'%i+'/'):
        os.makedirs('../logs/pokeAE/rs_'+'%02d'%i+'/')

    tf.reset_default_graph()
    step = 0
    input_queue = read_data_list_ae(
        '../../poke/train_cube_ae.txt', num_epochs, shuffle=True)

    images_1, images_2, u_cs = batch_images_actions_ae(
        input_queue, batch_size, num_threads=4,
        min_after_dequeue=256, type_img=1, normalized=1)

    with tf.Session() as sess:
        #poke_ae = PokeAE(learning_rate, epsilon, batch_size, split_size)
        poke_ae = PokeVAEGAN(learning_rate, epsilon, batch_size, split_size,
                             in_channels=3, corrupted=0, is_training=True)

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/pokeAE/rs_'+'%02d'%i+'/', sess.graph)

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                i1, i2, uc = sess.run([images_1, images_2, u_cs])
                feed_dict = {poke_ae.i1: i1, poke_ae.i2: i2, poke_ae.u_labels: uc}

                _ = sess.run(poke_ae.train_op_e, feed_dict=feed_dict)
                _ = sess.run(poke_ae.train_op_d, feed_dict=feed_dict)
                _ = sess.run(poke_ae.train_op_g, feed_dict=feed_dict)
                _, loss, summary = sess.run(
                    [poke_ae.train_op_g, poke_ae.loss, poke_ae.merged], feed_dict=feed_dict)
                #_, loss, summary = sess.run(
                #    [poke_ae.train_op, poke_ae.loss, poke_ae.merged],
                #    feed_dict=feed_dict)

                if step%10 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%s' %(step, loss))

                if step%500 == 0:
                    saver.save(sess, '../logs/pokeAE/rs_'+'%02d'%i+'/', global_step=step)
                step+=1
            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')
        finally:
            coord.request_stop()
            coord.join(threads)
            print('steps of batch enqueuing is %d'%step)

def main():
    for i in range(40, 70):
        num_epochs, batch_size, learning_rate, epsilon, split_size = generate_hyperparas()
        print('\n'+'num_epochs=%d, batch size=%d, learning rate=%s, epsilon=%s, split_size=%s'
              %(num_epochs, batch_size,learning_rate, epsilon, split_size))
        train_func(num_epochs, batch_size, learning_rate, epsilon, split_size, i)
        with open('../logs/pokeAE/rs_'+'%02d'%i+'/info.txt','w') as f:
            f.writelines('num_epochs = %d\n'%num_epochs)
            f.writelines('batch_size = %d\n'%batch_size)
            f.writelines('learning rate = %s\n'%learning_rate)
            f.writelines('epsilon = %s\n'%epsilon)
            f.writelines('split_size = %s\n'%split_size)

if __name__ == '__main__':
    sys.exit(main())
