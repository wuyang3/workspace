import numpy as np
import sys
import os
import tensorflow as tf
import random
from poke import Poke
from batch_operation import read_data_list
from batch_operation import batch_images_actions
from batch_operation import feed

def generate_hyperparas():
    idx = random.randint(0,1)
    lambda_epoch = [[0, 2], [0, 1]]
    lamb, num_epochs = lambda_epoch[idx]
    learning_rate = 10 ** np.random.uniform(-6, -3) # First search: (-5, -2)
    epsilon = 10 ** np.random.uniform(-7, -4)
    batch_size = 2 ** np.random.choice(np.array([3, 4, 5, 6]))
    return num_epochs, batch_size, learning_rate, epsilon, lamb

def train_func(num_epochs, batch_size, learning_rate, epsilon, lamb, i):
    if not os.path.exists('../logs/poke/rs_'+'%02d'%i+'/'):
        os.makedirs('../logs/poke/rs_'+'%02d'%i+'/')
    train_layers = ['inverse', 'forward', 'siamese']
    include_type=1
    step = 0
    tf.reset_default_graph()
    input_queue = read_data_list(
        '../../poke/train_gazebo_extra_class.txt', num_epochs, shuffle=True, include_type)
    images_1, images_2, u_cs, p_ds, theta_ds, l_ds, type_ds = batch_images_actions(
        input_queue, batch_size, num_threads=4, min_after_dequeue=512, include_type)

    with tf.Session() as sess:
        poke_model = Poke(train_layers, include_type, learning_rate, epsilon, lamb)

        saver = tf.train.Saver(max_to_keep=2)
        train_writer = tf.summary.FileWriter('../logs/poke/rs_'+'%02d'%i+'/', sess.graph)

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

                if step % 20 == 0:
                    train_writer.add_summary(summary, step)
                    print('step %d: loss->%.4f inverse loss->%.4f forward loss->%.4f'
                          %(step, loss, loss_i, loss_f))
                if step % 500 == 0:
                    saver.save(sess, '../logs/poke/rs_'+'%02d'%i+'/', global_step=step)
                step += 1
            train_writer.close()
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')
        finally:
            coord.request_stop()
            coord.join(threads)

def main():
    for i in range(160, 200):
        num_epochs, batch_size, learning_rate, epsilon, lamb = generate_hyperparas()
        print('\n'+'num_epochs=%d, batch size=%d, learning rate=%s, epsilon=%s, lambda=%s'
              %(num_epochs, batch_size,learning_rate, epsilon, lamb))
        train_func(num_epochs, batch_size, learning_rate, epsilon, lamb, i)
        with open('../logs/poke/rs_'+'%02d'%i+'/info.txt','w') as f:
            f.writelines('num_epochs = %d\n'%num_epochs)
            f.writelines('batch_size = %d\n'%batch_size)
            f.writelines('learning rate = %s\n'%learning_rate)
            f.writelines('epsilon = %s\n'%epsilon)
            f.writelines('lambda = %s\n'%lamb)

if __name__ == '__main__':
    sys.exit(main())
