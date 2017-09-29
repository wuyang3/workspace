#!/usr/bin/env python
"""
Test poke auto-encoder on real time poking images. Reconstruct on current image and action to
generate future images online.

For tensorflow tensor, index starts from top left including image tensor. [row, column]/[x, y]
For opencv, matplotlib image, index starts from top left. [column, row]/[x,y]
For image projection coordinates, index starts from top right. [row, column(right to left)]/[x, y]

@author: wuyang
"""
import rospy
import sys
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.insert(0, '/home/wuyang/workspace/python/tf/poke_model/')
from poke_autoencoder import PokeAE
from poke_bn_vae import PokeBnVAE
from poke_ae_rnn import PokeAERNN, PokeVAERNN, PokeAEFFRNN, PokeVAERNNC
from poke_ae_rnn_multi import PokeMultiRNN
from batch_operation import image_feed_ae

from poke_random_new import PokeRandomNew
from poke_random_new import (
    delete_gazebo_models,
    load_gazebo_models,
)

win_width = 10
win_height = 10

# helper function for plotting poke.
def rect(ax, poke, c):
    x, y, t, l = poke
    dx = -200.0 * l * math.cos(t)
    dy = -200.0 * l * math.sin(t)
    ax.arrow(64.0-y, x, dy, dx, head_width=5, head_length=5, color=c)

def plot_sample(img_before, img_after, img_rec, action,
                position, position_after, loss, patch_1, patch_2):
    """
    patches.Rectangle((top_left_x, top_left_y), win_width, win_height, ...)
    """
    def get_patch(px, py, color):
        return patches.Rectangle((64.0-py, px), win_width, win_height, edgecolor=color, fill=False)
    p_x, p_y = position
    p_x_after, p_y_after = position_after

    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)

    ax1.imshow(img_before.copy())
    ax1.add_patch(get_patch(p_x, p_y, 'b'))
    rect(ax1, action, "green")
    ax1.set_title('before')

    ax2.imshow(img_after.copy())
    ax2.add_patch(get_patch(p_x_after, p_y_after, 'r'))
    ax2.set_title('after')

    ax3.imshow(img_before.copy())
    ax3.add_patch(get_patch(p_x, p_y, 'b'))
    ax3.set_title('before')

    ax4.imshow(img_rec.copy())
    ax4.add_patch(get_patch(p_x, p_y, 'b'))
    ax4.add_patch(get_patch(p_x_after, p_y_after, 'r'))
    ax4.set_title('reconstruction')

    ax5.imshow(patch_1.copy())
    ax6.imshow(patch_2.copy())
    fig.suptitle('step %d: window pixel loss->%.4f'%(loss[0], loss[1]))


def center_to_corner(position):
    p_x, p_y = position
    p_x = 64.0*p_x/240.0-win_width/2.0
    p_y = 64.0*p_y/240.0+win_height/2.0
    return p_x, p_y

def plot_multisample(imgs, imgs_rec, actions, positions, title):
    """
    imgs/imgs_rec: list of backprop_step images.
    actions: list of backprop_step-1 actions (each as a list).
    positions: list of backprop_step positions (each as a list of x and y).
    title: a string.
    """
    num = len(imgs)
    assert (num == len(actions)+1 and num==len(imgs_rec)), (
        'number of images or actions are not matching.'
    )
    fig, axarr = plt.subplots(2, num)
    for i, ax in enumerate(axarr[0]):
        im = imgs[i][0]
        im = im/255.0
        p_x, p_y = center_to_corner(positions[i])
        ax.imshow(im)
        if i==0:
            ax.add_patch(patches.Rectangle(
                (64.0-p_y, p_x), win_height, win_width, edgecolor='b', fill=False))
        else:
            p_x0, p_y0 = center_to_corner(positions[i-1])
            ax.add_patch(patches.Rectangle(
                (64.0-p_y0, p_x0), win_height, win_width, edgecolor='b', fill=False))
            ax.add_patch(patches.Rectangle(
                (64.0-p_y, p_x), win_height, win_width, edgecolor='r', fill=False))

        if i < num-1:
            rect(ax, actions[i], "green")
    for i, ax in enumerate(axarr[1]):
        im = imgs_rec[i][0]
        im = im/im.max()
        p_x, p_y = center_to_corner(positions[i])
        ax.imshow(im)
        if i==0:
            ax.add_patch(patches.Rectangle(
                (64.0-p_y, p_x), win_height, win_width, edgecolor='b', fill=False))
        else:
            p_x0, p_y0 = center_to_corner(positions[i-1])
            ax.add_patch(patches.Rectangle(
                (64.0-p_y0, p_x0), win_height, win_width, edgecolor='b', fill=False))
            ax.add_patch(patches.Rectangle(
                (64.0-p_y, p_x), win_height, win_width, edgecolor='r', fill=False))

    fig.suptitle('%s'%title)

def ae_test():
    rospy.init_node('poke_test_tf')

    path = '/home/wuyang/workspace/python/poke/test_data/'
    i = 0
    k = 5
    poke_ros = PokeRandomNew(0.38)
    name = 'cube'
    rospy.sleep(3.0)
    poke_ros.save_image(path, i)
    rospy.sleep(2.0)
    current_img = image_feed_ae(path+'img%04d.jpg'%i, 1)

    poke_ros.move_to_start()

    with tf.Session() as sess:
        #poke_ae = PokeAE(batch_size=1, split_size = 256, in_channels=3,
        #                 corrupted=0, is_training=False)
        poke_ae = PokeBnVAE(batch_size=1, split_size = 256, in_channels=3, is_training=False)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(
            '/home/wuyang/workspace/python/tf/logs/pokeAE/trial_35/'))

        try:
            while (not rospy.is_shutdown() and i < 200):
                if i >= k:
                    delete_gazebo_models()
                    name = load_gazebo_models()
                    k += 5
                    rospy.sleep(3.0)
                    poke_ros.save_image(path, i)
                    rospy.sleep(2.0)
                    current_img = image_feed_ae(path+'img%04d.jpg'%i, 1)
                    poke_ros.move_to_start()

                poke_pointstamped, poke_action, _, _ = poke_ros.poke_generation(name)
                projected = poke_ros.model_image_projection(poke_pointstamped)
                block_position = poke_ros.model_pixel_location()

                position = poke_pointstamped.point
                poke_ros.poke(position, poke_action)
                rospy.sleep(1.0)
                poke_ros.save_image(path, i+1)
                block_position_after = poke_ros.model_pixel_location()

                after_img = image_feed_ae(path+'img%04d.jpg'%(i+1), 1)
                current_eval, after_eval = sess.run([current_img, after_img])
                #uc = np.array([[projected.x,
                #                projected.y,
                #                poke_action['theta'],
                #                poke_action['l']]])
                # for bn-vae, actions are normalized, sometimes plain ae, dae.
                uc = np.array([[projected.x/240.0,
                                projected.y/240.0,
                                poke_action['theta']/(np.pi*2),
                                (poke_action['l']-0.04)/0.04]])
                feed_dict = {poke_ae.i1: current_eval,
                             poke_ae.i2: after_eval,
                             poke_ae.u_labels: uc}
                i2_rec, loss = sess.run([poke_ae.decoding, poke_ae.loss],
                                        feed_dict = feed_dict)
                print('decoding pixel max %f min %f'%(i2_rec[0].max(), i2_rec[0].min()))

                mean_error_tensor, p1_tensor, p2_tensor = poke_ae.patch_average_error(
                    after_eval, i2_rec, win_height, win_width,
                    block_position_after.x/240.0,
                    1.0-(block_position_after.y/240.0))
                print("Center sent to tensorflow: row=%f, col=%f"
                      %(64.0*block_position_after.x/240.0,
                        64.0-64.0*block_position_after.y/240.0))
                mean_pixel_error, patch_1, patch_2 = sess.run(
                    [mean_error_tensor, p1_tensor, p2_tensor])

                plot_sample(current_eval[0]/255.0,
                            after_eval[0]/255.0,
                            i2_rec[0]/i2_rec[0].max(),
                            [projected.x*64.0/240.0, projected.y*64.0/240.0,
                             poke_action['theta'], poke_action['l']],
                            [64.0*block_position.x/240.0-win_width/2.0,
                             64.0*block_position.y/240.0+win_height/2.0],
                            [64.0*block_position_after.x/240.0-win_width/2.0,
                             64.0*block_position_after.y/240.0+win_height/2.0],
                            [i, mean_pixel_error],
                            patch_1/patch_1.max(),
                            patch_2/patch_2.max())

                plt.pause(0.5)
                plt.savefig(path+'vae%d.jpg'%i)
                plt.waitforbuttonpress()
                plt.close()

                current_img = after_img

                with open(path+'error_dae.dat', 'ab') as f:
                    np.savetxt(f, np.array([[mean_pixel_error]]))

                i += 1

        except Exception as e:
            print "Error: %s" %e

        finally:
            print('deleting model...')
            #delete_gazebo_models()
            sys.exit(1)

def ae_rnn_test():
    rospy.init_node('poke_test_tf')

    imgs = []
    actions_plot = []
    actions_feed = []
    positions = []

    path = '/home/wuyang/workspace/python/poke/test_data/'
    i = 0
    k = 5

    poke_ros = PokeRandomNew(0.38)
    name = 'cube'
    rospy.sleep(4.0)
    poke_ros.save_image(path, i)
    rospy.sleep(2.0)
    imgs.append(image_feed_ae(path+'img%04d.jpg'%i, 1))
    block_position = poke_ros.model_pixel_location()
    positions.append([block_position.x,
                      block_position.y])

    poke_ros.move_to_start()

    poke_pointstamped, poke_action, _, _ = poke_ros.poke_generation(name)
    projected = poke_ros.model_image_projection(poke_pointstamped)
    position = poke_pointstamped.point
    poke_ros.poke(position, poke_action)
    rospy.sleep(1.0)
    poke_ros.save_image(path, i+1)
    imgs.append(image_feed_ae(path+'img%04d.jpg'%(i+1), 1))
    block_position = poke_ros.model_pixel_location()
    positions.append([block_position.x,
                      block_position.y])
    actions_plot.append([projected.x*64.0/240.0, projected.y*64.0/240.0,
                        poke_action['theta'], poke_action['l']])
    u = np.array([[projected.x/240.0,
                   projected.y/240.0,
                   poke_action['theta']/(np.pi*2),
                   (poke_action['l']-0.04)/0.04]])
    actions_feed.append(u)

    lstm = 1
    vae = 1
    with tf.Session() as sess:
        #poke_ae = PokeAERNN(batch_size=1, split_size=512,
        #                    in_channels=3, corrupted=0, is_training=False, lstm=lstm)
        #poke_ae = PokeVAERNN(batch_size=1, split_size=512,
        #                     in_channels=3, corrupted=0, is_training=False, lstm=lstm)
        #poke_ae = PokeAEFFRNN(batch_size=1, in_channels=3,
        #                      corrupted=0, is_training=False, lstm=lstm, vae=vae)
        poke_ae = PokeVAERNNC(batch_size=1, split_size=512,
                              in_channels=3, corrupted=0, is_training=False, lstm=lstm)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(
            '/home/wuyang/workspace/python/tf/logs/pokeAERNN/lstm_vae_compare/'))

        try:
            while (not rospy.is_shutdown()):
                if i >= k:
                    delete_gazebo_models()
                    name = load_gazebo_models()
                    imgs = []
                    positions = []
                    actions_plot = []
                    actions_feed = []
                    k += 5
                    rospy.sleep(3.0)
                    poke_ros.save_image(path, i)
                    rospy.sleep(2.0)
                    imgs.append(image_feed_ae(path+'img%04d.jpg'%i, 1))
                    block_position = poke_ros.model_pixel_location()
                    positions.append([block_position.x,
                                      block_position.y])

                    poke_ros.move_to_start()

                    poke_pointstamped, poke_action, _, _ = poke_ros.poke_generation(name)
                    projected = poke_ros.model_image_projection(poke_pointstamped)
                    position = poke_pointstamped.point
                    poke_ros.poke(position, poke_action)
                    rospy.sleep(1.0)
                    poke_ros.save_image(path, i+1)
                    imgs.append(image_feed_ae(path+'img%04d.jpg'%(i+1), 1))
                    block_position = poke_ros.model_pixel_location()
                    positions.append([block_position.x,
                                      block_position.y])
                    actions_plot.append([projected.x*64.0/240.0, projected.y*64.0/240.0,
                                         poke_action['theta'], poke_action['l']])
                    u = np.array([[projected.x/240.0,
                                   projected.y/240.0,
                                   poke_action['theta']/(np.pi*2),
                                   (poke_action['l']-0.04)/0.04]])
                    actions_feed.append(u)

                poke_pointstamped, poke_action, _, _ = poke_ros.poke_generation(name)
                projected = poke_ros.model_image_projection(poke_pointstamped)
                position = poke_pointstamped.point
                poke_ros.poke(position, poke_action)
                rospy.sleep(1.0)
                poke_ros.save_image(path, i+2)
                imgs.append(image_feed_ae(path+'img%04d.jpg'%(i+2), 1))
                block_position = poke_ros.model_pixel_location()
                positions.append([block_position.x,
                                  block_position.y])
                actions_plot.append([projected.x*64.0/240.0, projected.y*64.0/240.0,
                                     poke_action['theta'], poke_action['l']])
                u = np.array([[projected.x/240.0,
                               projected.y/240.0,
                               poke_action['theta']/(np.pi*2),
                               (poke_action['l']-0.04)/0.04]])
                actions_feed.append(u)

                imgs_feed = sess.run(imgs)
                feed_dict = {poke_ae.i1: imgs_feed[0],
                             poke_ae.i2: imgs_feed[1],
                             poke_ae.i3: imgs_feed[2],
                             poke_ae.u2: actions_feed[0],
                             poke_ae.u3: actions_feed[1]}
                imgs_rec, loss_total = sess.run(
                    [[poke_ae.decoding_1, poke_ae.decoding_2, poke_ae.decoding_3], poke_ae.loss],
                    feed_dict = feed_dict)

                errors = []
                for n in range(1, 3):
                    p_x, p_y = positions[n]
                    mean_error_t, _, _ = poke_ae.patch_average_error(
                        imgs_feed[n], imgs_rec[n], win_height, win_width,
                        p_x/240.0, 1-p_y/240.0)
                    print("Center sent to tensorflow: row=%f, col=%f"
                          %(64.0*p_x/240.0,
                            64.0-64.0*p_y/240.0))
                    mean_error = sess.run(mean_error_t)
                    errors.append(mean_error)

                title_l = ['Error %d: %.2f'%(n, item) for n, item in enumerate(errors)]
                title = ' '.join(title_l)
                plot_multisample(imgs_feed, imgs_rec, actions_plot, positions, title)
                #plt.pause(0.5)
                plt.savefig(path+'rnn%d.eps'%i, format='eps', dpi=1000)
                #plt.waitforbuttonpress()
                #plt.close()

                imgs.pop(0)
                positions.pop(0)
                actions_plot.pop(0)
                actions_feed.pop(0)
                print('%d, %d, %d, %d'
                      %(len(imgs), len(positions),len(actions_plot), len(actions_feed)))

                if vae == 1:
                    f_name = path+'error_vae.dat'
                else:
                    f_name = path+'error_dae.dat'
                with open(f_name, 'ab') as f:
                    np.savetxt(f, np.array([errors]))

                i += 1

        except Exception as e:
            print "Error: %s" %e

        finally:
            print('stopped...')
            #delete_gazebo_models()
            sys.exit(1)

def ae_rnn_multi_test(bp_steps):
    rospy.init_node('poke_test_tf')

    imgs = []
    positions = []
    actions_plot = [] # bp_steps-1 of arraies for plotting.
    actions_feed = [np.array([[0, 0, 0, 0]])] # bp_steps of arraies for feeding.

    path = '/home/wuyang/workspace/python/poke/test_data/'
    i = 0
    k = 10

    poke_ros = PokeRandomNew(0.38)
    name = 'cube'

    def poke_start(num):
        rospy.sleep(3.0)
        poke_ros.save_image(path, num)
        rospy.sleep(2.0)
        imgs.append(image_feed_ae(path+'img%04d.jpg'%num, 1))
        block_position = poke_ros.model_pixel_location()
        positions.append([block_position.x, block_position.y])
        poke_ros.move_to_start()

    def poke_iterate(start, num):
        poke_pointstamped, poke_action, _, _ = poke_ros.poke_generation(name)
        projected = poke_ros.model_image_projection(poke_pointstamped)
        position = poke_pointstamped.point
        poke_ros.poke(position, poke_action)
        rospy.sleep(1.0)
        poke_ros.save_image(path, start+num)
        imgs.append(image_feed_ae(path+'img%04d.jpg'%(start+num), 1))
        block_position = poke_ros.model_pixel_location()
        positions.append([block_position.x, block_position.y])
        actions_plot.append([projected.x*64.0/240.0, projected.y*64.0/240.0,
                             poke_action['theta'], poke_action['l']])
        u = np.array([[projected.x/240.0,
                       projected.y/240.0,
                       poke_action['theta']/(np.pi*2),
                       (poke_action['l']-0.04)/0.04]])
        actions_feed.append(u)

    poke_start(i)
    for j in range(1, bp_steps-1):
        poke_iterate(i, j)

    lstm = 1
    vae = 1
    with tf.Session() as sess:
        poke_ae = PokeMultiRNN(batch_size=1, split_size=512, in_channels=3, corrupted=0,
                               is_training=False, bp_steps=bp_steps)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(
            '/home/wuyang/workspace/python/tf/logs/pokeAERNN/6_lstm_vae/'))

        try:
            while (not rospy.is_shutdown() and i<60):
                if i >= k:
                    delete_gazebo_models()
                    name = load_gazebo_models()
                    imgs = []
                    positions = []
                    actions_plot = []
                    actions_feed = [np.array([[0, 0, 0, 0]])]
                    k += 10

                    poke_start(i)
                    for j in range(1, bp_steps-1):
                        poke_iterate(i, j)

                poke_iterate(i, bp_steps-1)

                imgs_feed = sess.run(imgs) # list of 1x64x64x4
                imgs_t = np.stack(imgs_feed, axis=1) # 1xbp_stepsx64x64x3
                actions_t = np.stack(actions_feed, axis=1) # 1xbp_stepsx4
                feed_dict = {poke_ae.i: imgs_t, poke_ae.u: actions_t}

                imgs_rec, loss_total = sess.run([poke_ae.decoding_list, poke_ae.loss],
                                                feed_dict=feed_dict)

                errors = []
                for n in range(1, bp_steps):
                    p_x, p_y = positions[n]
                    mean_error_t, _, _ = poke_ae.patch_average_error(
                        imgs_feed[n], imgs_rec[n], win_height, win_width,
                        p_x/240.0, 1-p_y/240.0)
                    print("Center sent to tensorflow: row=%f, col=%f"
                          %(64.0*p_x/240.0, 64.0-64.0*p_y/240.0))
                    mean_error = sess.run(mean_error_t)
                    errors.append(mean_error)

                title_l = ['Error %d: %.2f'%(n, item) for n, item in enumerate(errors)]
                title = ' '.join(title_l)
                plot_multisample(imgs_feed, imgs_rec, actions_plot, positions, title)
                #plt.pause(0.5)
                plt.savefig(path+'rnn%d.png'%i, format='png', dpi=800)
                #plt.waitforbuttonpress()
                #plt.close()

                imgs.pop(0)
                positions.pop(0)
                actions_plot.pop(0)
                actions_feed.pop(1) # pop out the first non-zero four dimensional action.
                print('%d, %d, %d, %d'
                      %(len(imgs), len(positions),len(actions_plot), len(actions_feed)))

                if vae == 1:
                    f_name = path+'error_vae.dat'
                else:
                    f_name = path+'error_dae.dat'
                with open(f_name, 'ab') as f:
                    np.savetxt(f, np.array([errors]))

                i += 1

        except Exception as e:
            print "Error: %s" %e

        finally:
            print('stopped...')
            #delete_gazebo_models()
            sys.exit(1)

if __name__ == '__main__':
    #ae_test()
    #ae_rnn_test()
    ae_rnn_multi_test(bp_steps=6)
