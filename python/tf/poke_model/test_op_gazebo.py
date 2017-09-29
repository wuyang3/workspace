# -*- coding: utf-8 -*-
"""
Generate test data batch and test the poke model. Test data are collected beforehand.
tf.train.latest_checkpoint('./')
Do we have to save explicitly with model.ckpt?

@author: wuyang
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage
import math
from poke import Poke
from batch_operation import (
    read_data_list,
    batch_dir_images_actions,
    feed,
)

# helper function for plotting poke.
def rect(ax, poke, c):
    x, y, t, l = poke
    dx = -400 * l * math.cos(t)
    dy = -400 * l * math.sin(t)
    ax.arrow(240-y, x, dy, dx, head_width=5, head_length=5, color=c)

def plot_sample(img_before, img_after, action, action_pred, loss):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_before.copy())
    rect(ax1, action, "green")
    rect(ax1, action_pred, "blue")
    ax2.imshow(img_after.copy())
    fig.suptitle('step %d: loss->%.4f inverse->%.4f forward->%.4f'
                 %(loss[0], loss[1], loss[2], loss[3]))

if __name__ == '__main__':
    # Input queue parameters
    include_type = 0
    num_epochs = 1
    shuffle = True
    batch_size = 1
    num_threads = 4
    min_after_dequeue = 512

    # Read pathes of test data and create a input queue.
    input_queue = read_data_list(
        '../../poke/test_gazebo.txt', num_epochs, shuffle, include_type)

    # Return batch of processed images and labels for testing.
    dir_1, dir_2, images_1, images_2, u_cs, p_ds, theta_ds, l_ds = batch_dir_images_actions(input_queue, batch_size, num_threads, min_after_dequeue, include_type)

    # Initial parameters.
    step=0

    with tf.Session() as sess:
        poke_model = Poke(include_type=0)

        saver = tf.train.Saver() # tf.trainable_variables()?
        saver.restore(sess, tf.train.latest_checkpoint('../logs/poke/rs_169/'))

        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                d1, d2, i1, i2, uc, pd, thetad, ld = sess.run(
                    [dir_1, dir_2, images_1, images_2, u_cs, p_ds, theta_ds, l_ds])

                feed_dict = feed(poke_model, (i1, i2, uc, pd, thetad, ld), include_type)

                loss, loss_i, loss_f, p_pred, theta_pred, l_pred = sess.run(
                    [poke_model.loss,
                     poke_model.loss_i,
                     poke_model.loss_f,
                     poke_model.classes[0],
                     poke_model.classes[1],
                     poke_model.classes[2]],
                    feed_dict=feed_dict)

                np.set_printoptions(suppress=True)
                print('Continuous: %s'%uc[0])
                print('Discrete: %s'%np.concatenate([pd, thetad, ld]))
                print('Prediction: %s'%np.concatenate(
                    [p_pred, theta_pred, l_pred]))

                # Plot real and predicted action on images.
                img_before = scipy.ndimage.imread(d1[0].decode("utf-8"))
                img_after = scipy.ndimage.imread(d2[0].decode("utf-8"))
                # Transfer discrete predictions together into pixels.
                p = p_pred[0]
                theta = theta_pred[0]
                l = l_pred[0]
                # For original data p%20 and p//20. For gazebo data x denotes rows
                # in the image and is actually y direction in the opencv image
                # sense. Moreover, x and y direction in the plotting function
                # are swapped. Mind that the y direction should be 240-y. This is
                # not corrected since the collected data is erroreous.
                ud_pred = np.array([p//20*12+6, p%20*12+6,
                                    theta*np.pi/18+np.pi/36,
                                    0.02+l*0.004+0.002])
                plot_sample(img_before, img_after, uc[0], ud_pred,
                            [step, loss, loss_i, loss_f])

                plt.pause(0.5)
                plt.waitforbuttonpress()
                plt.close()
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done queuing: epoch limit reached.')
        finally:
            coord.request_stop()
            coord.join(threads)
            print('epochs of batch enqueuing is %d' %step)
