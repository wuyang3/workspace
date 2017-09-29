#!/usr/bin/env python
"""
Jen's advice on testing the trained model:
Generating poke and executing poke; predicting poke and comparing poke.

@author: wuyang
"""
import rospy
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/wuyang/workspace/python/tf/poke_model/')
from poke import Poke
from batch_operation import feed, image_feed
from test_op_gazebo import plot_sample

from poke_random_new import PokeRandomNew
from poke_random_new import (
    delete_gazebo_models,
    load_gazebo_models,
)

if __name__ == '__main__':
    rospy.init_node('poke_test_tf')

    include_type = 0
    path = '/home/wuyang/workspace/python/poke/test_data/'
    i = 0
    k = 5
    poke_ros = PokeRandomNew(0.38)
    name = 'cube'
    rospy.sleep(3.0)
    poke_ros.save_image(path, i)
    rospy.sleep(2.0)
    current_img = image_feed(path+'img%04d.jpg'%i)

    poke_ros.move_to_start()

    with tf.Session() as sess:
        poke_tf = Poke(include_type=include_type)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(
            '/home/wuyang/workspace/python/tf/logs/poke/rs_169/'))

        try:
            while not rospy.is_shutdown():
                if i >= k:
                    delete_gazebo_models()
                    name = load_gazebo_models()
                    #i += 10
                    k += 5
                    rospy.sleep(3.0)
                    poke_ros.save_image(path, i)
                    rospy.sleep(2.0)
                    current_img = image_feed(path+'img%04d.jpg'%i)
                    poke_ros.move_to_start()

                poke_pointstamped, poke_action = poke_ros.poke_generation(name)
                projected = poke_ros.model_image_projection(poke_pointstamped)
                uc = np.array([[projected.x, projected.y,
                                poke_action['theta'], poke_action['l']]])
                dx = np.floor(projected.x/12)
                dy = np.floor(projected.y/12)
                pd = 20*dx + dy
                thetad = np.floor(poke_action['theta']/(np.pi/18))
                ld = np.floor((poke_action['l']-0.02)/0.004)

                position = poke_pointstamped.point
                poke_ros.poke(position, poke_action)
                rospy.sleep(1.0)
                poke_ros.save_image(path, i+1)
                after_img = image_feed(path+'img%04d.jpg'%(i+1))

                current_eval, after_eval = sess.run([current_img, after_img])
                feed_dict = feed(
                    poke_tf,
                    (current_eval, after_eval, uc,
                     np.array([pd]), np.array([thetad]), np.array([ld])),
                    include_type)

                loss, loss_i, loss_f, p_class, theta_class, l_class = sess.run(
                    [poke_tf.loss,
                     poke_tf.loss_i,
                     poke_tf.loss_f,
                     poke_tf.classes[0],
                     poke_tf.classes[1],
                     poke_tf.classes[2]],
                    feed_dict=feed_dict)

                p_x = p_class[0]//20*12+6
                p_y = p_class[0]%20*12+6
                theta = theta_class[0]*np.pi/18 + np.pi/36
                l = l_class[0]*0.004 + 0.002 + 0.02
                if i%5 != 0:
                    plot_sample(current_eval[0]/255.0,
                                after_eval[0]/255.0,
                                uc_old[0],
                                [p_x, p_y, theta, l],
                                [i, loss, loss_i, loss_f])

                plt.pause(3.0)
                plt.close()

                current_img = after_img
                # only do this since images are delayed! plot_sample(uc_old)
                uc_old = uc
                i += 1

        except KeyboardInterrupt:
            print('stop testing...')

        finally:
            print('deleting model...')
            delete_gazebo_models()
            sys.exit(1)
