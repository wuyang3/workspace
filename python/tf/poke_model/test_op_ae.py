# -*- coding: utf-8 -*-
"""
Test trained auto-encoder using collected offline image and action data.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage
import math

from poke_autoencoder import PokeAE
from batch_operation import read_data_list_ae
from batch_operation import batch_dir_images_actions_ae

# helper function for plotting poke.
def rect(ax, poke, c):
    x, y, t, l = poke
    dx = -400 * l * math.cos(t)
    dy = -400 * l * math.sin(t)
    ax.arrow(240-y, x, dy, dx, head_width=5, head_length=5, color=c)

def plot_sample(img_before, img_after, img_rec, action, loss):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(img_before.copy())
    rect(ax1, action, "green")
    ax1.set_title('before')
    ax2.imshow(img_after.copy())
    if action[3] == 0:
        ax2.set_title('after same')
    else:
        ax2.set_title('after')
    ax3.imshow(img_before.copy())
    ax3.set_title('before')
    ax4.imshow(img_rec.copy())
    ax4.set_title('reconstruction')
    fig.suptitle('step %d: loss->%.4f'%(loss[0], loss[1]))

def main():
     # Input queue parameters
    num_epochs = 1
    shuffle = True
    batch_size = 1
    num_threads = 4
    min_after_dequeue = 512

    type_img = 1 #0/1
    in_channels = 3 #1/3
    corrupted = 0 #1
    # Read pathes of test data and create a input queue.
    input_queue = read_data_list_ae(
        '../../poke/test_gazebo_ae.txt', num_epochs, shuffle)

    # Return batch of processed images and labels for testing.
    dir_1, dir_2, images_1, images_2, u_cs = batch_dir_images_actions_ae(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=type_img)

    # Initial parameters.
    step=0

    with tf.Session() as sess:
        poke_ae = PokeAE(batch_size=batch_size, in_channels=in_channels, corrupted=corrupted)

        saver = tf.train.Saver() # tf.trainable_variables()?
        saver.restore(sess, tf.train.latest_checkpoint('../logs/pokeAE/trial_16/'))

        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                d1, d2, i1, i2, uc = sess.run(
                    [dir_1, dir_2, images_1, images_2, u_cs])

                feed_dict = {poke_ae.i1: i1, poke_ae.i2: i2, poke_ae.u_labels: uc}

                i2_rec, loss = sess.run([poke_ae.decoding, poke_ae.loss],
                                        feed_dict = feed_dict)

                # Plot real and predicted action on images.
                img_before = scipy.ndimage.imread(d1[0].decode("utf-8"))
                img_after = scipy.ndimage.imread(d2[0].decode("utf-8"))

                px = uc[0][0]*12+6
                py = uc[0][1]*12+6
                if uc[0][2] == 0:
                    theta = 0
                else:
                    theta = uc[0][2]*np.pi/18 - np.pi/36 # - if testing 0 actions, otherwise +.
                if uc[0][3] == 0:
                    l = 0
                else:
                    l = uc[0][3]*0.004 - 0.002 + 0.02
                plot_sample(img_before, img_after, i2_rec[0]/i2_rec[0].max(),
                            [px, py, theta, l], [step, loss])
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

if __name__ == '__main__':
    main()
