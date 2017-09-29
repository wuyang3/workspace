#!/usr/bin/env python
"""
Using real time predicted action to poke in Gazebo simulation.
a cleaner solution to import a script from another directory? Make
a module! The module path still needs to be added to $PYTHONPATH
Using trained network to control the poking. To generate target image,
models are spawned once more before the actual poking.
"""
import rospy
import rospkg
import sys
import tensorflow as tf
#from tensorflow.python.framework import ops, dtypes
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from poke_random_new import PokeRandomNew
sys.path.insert(0, '/home/wuyang/workspace/python/tf/poke_model/')
from poke import Poke
from batch_operation import image_feed

from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion

class PokeNewExtend(PokeRandomNew):
    """
    Inheritate from class PokeNew.
    """
    def delete_gazebo_models(self, delete_block):
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            if delete_block:
                resp_delete = delete_model("block")
            else:
                resp_delete = delete_model("block")
                resp_delete = delete_model("block_1")
                resp_delete = delete_model("block_2")
        except rospy.ServiceException as e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))

    def load_gazebo_models(self, block_reference_frame="world"):
        # Get Models' Path
        model_path = rospkg.RosPack().get_path('poke_gazebo')+"/models/block/"
        file_list = ['block','block_blue','block_white','block_green',
                     'block_orange','block_l_1', 'block_l_2', 'block_l_3',
                     'block_l_4', 'cylinder_1', 'cylinder_2', 'cylinder_3',
                     'cylinder_4', 'cylinder_5']

        rospy.wait_for_service('/gazebo/spawn_urdf_model',2.0)
        spawn_urdf = rospy.ServiceProxy(
            '/gazebo/spawn_urdf_model', SpawnModel)
        num = 2#random.randint(1,2)
        chosen_list = random.sample(file_list, num)
        for i, item in enumerate(chosen_list):
            file_name = item+'.urdf'
            block_xml = ''
            with open (model_path + file_name, "r") as block_file:
                block_xml=block_file.read().replace('\n', '')
            if i == 0:
                block_pose = Pose(position=Point(x=random.uniform(0.3, 0.31),
                                                 y=random.uniform(0.53, 0.54),
                                                 z=0.815),
                                  orientation=Quaternion(x=0,
                                                         y=0,
                                                         z=0.14943813,
                                                         w=0.98877108))
                try:
                    resp_urdf = spawn_urdf("block", block_xml, "/",
                                           block_pose, block_reference_frame)
                    block_xml_first = block_xml
                    if item[0] == 'c':
                        name = 'cylinder'
                    elif item[:7] == 'block_l':
                        name = 'cuboid'
                    else:
                        name = 'cube'

                except rospy.ServiceException as e:
                    rospy.logerr("Spawn URDF service call failed: {0}".format(e))

            else:
                block_pose_rand = Pose(position=Point(
                    x=random.uniform(0.1, 0.7),
                    y=random.uniform(0.13, 0.73),
                    z=0.80))
                try:
                    resp_urdf = spawn_urdf("block"+'_%d'%i, block_xml, "/",
                                           block_pose_rand, block_reference_frame)
                except rospy.ServiceException as e:
                    rospy.logerr("Spawn URDF service call failed: {0}".format(e))

        self.save_image(
            '/home/wuyang/workspace/python/poke/test_data/',
            100)
        rospy.sleep(2.0)
        self.delete_gazebo_models(1)

        block_pose = Pose(position=Point(x=0.53,
                                         y=0.36,
                                         z=0.815),
                          orientation=Quaternion(x=0,
                                                 y=0,
                                                 z=-0.04997917,
                                                 w=0.99875026))
        resp_urdf = spawn_urdf("block", block_xml_first, "/",
                               block_pose, block_reference_frame)
        self.save_image(
            '/home/wuyang/workspace/python/poke/test_data/',
            0)
        return name

def plot_sample(img_before, img_after, action, step):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_before)
    x, y, t, l = action
    dx = -400*l*math.cos(t)
    dy = -400*l*math.sin(t)
    ax1.arrow(240-y, x, dy, dx, head_width=5, head_length=5, color='green')
    ax2.imshow(img_after)
    fig.suptitle('step %d: x=%.2f, y=%.2f, t=%.2f, l=%.2f)'
                 %(step, x, y, t/np.pi*180, l))

if __name__ == '__main__':
    rospy.init_node('poke_test')

    include_type=0
    path= '/home/wuyang/workspace/python/poke/test_data/'
    i = 0
    k = 10
    poke_ros = PokeNewExtend(0.38)
    name = 'cube'
    poke_ros.delete_gazebo_models(0)
    name = poke_ros.load_gazebo_models()
    rospy.sleep(1.0)
    target_img = image_feed(path+'img0100.jpg')

    poke_ros.move_to_start()

    with tf.Session() as sess:
        poke_tf = Poke(include_type=include_type)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(
            '/home/wuyang/workspace/python/tf/logs/poke/rs_165_4/'))

        try:
            while not rospy.is_shutdown():
                current_img= image_feed(path+'img%04d.jpg'%i)
                target_eval, current_eval = sess.run([target_img, current_img])
                feed_dict = {poke_tf.i1: current_eval, poke_tf.i2: target_eval}
                p_class, theta_class, l_class = sess.run(
                    [poke_tf.classes[0],
                     poke_tf.classes[1],
                     poke_tf.classes[2]],
                    feed_dict=feed_dict)

                p_x = p_class[0]//20*12+6
                p_y = p_class[0]%20*12+6
                theta = theta_class[0]*np.pi/18 + np.pi/36
                l = l_class[0]*0.004 + 0.002 + 0.02
                # for float type array, plt treats color ranges as [0,1]
                plot_sample(current_eval[0]/255.0,
                            target_eval[0]/255.0,
                            [p_x, p_y, theta, l], i)
                plt.pause(0.5)
                plt.waitforbuttonpress()
                plt.close()
                #generated from predicted pixel location, theta and l rather than action service.
                position_world = poke_ros.model_world_projection(p_x, p_y)
                poke_ros.poke(position_world,
                              {'theta': theta, 'l':l})

                rospy.sleep(1.0)
                poke_ros.save_image(path, i+1)
                i += 1

        except KeyboardInterrupt:
            print('stop testing...')

        finally:
            print('deleting model...')
            poke_ros.delete_gazebo_models(0)
            sys.exit(1)
