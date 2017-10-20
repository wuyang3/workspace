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
import sys, traceback
import tensorflow as tf
#from tensorflow.python.framework import ops, dtypes
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from poke_random_new import PokeRandomNew
sys.path.insert(0, '/home/wuyang/workspace/python/tf/poke_model/')
from poke import Poke
from poke_depth import PokeDepth, PokeTotal
from batch_operation import image_feed, image_feed_total

from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped

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
                     'block_orange']
                     # ,'block_l_1', 'block_l_2', 'block_l_3',
                     # 'block_l_4', 'cylinder_1', 'cylinder_2', 'cylinder_3',
                     # 'cylinder_4', 'cylinder_5'];

        rospy.wait_for_service('/gazebo/spawn_urdf_model',2.0)
        spawn_urdf = rospy.ServiceProxy(
            '/gazebo/spawn_urdf_model', SpawnModel)
        num = 1#random.randint(1,2)
        chosen_list = random.sample(file_list, num)
        for i, item in enumerate(chosen_list):
            file_name = item+'.urdf'
            block_xml = ''
            with open (model_path + file_name, "r") as block_file:
                block_xml=block_file.read().replace('\n', '')
            if i == 0:
                block_pose = Pose(position=Point(x=0.105,
                                                 y=0.255,
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
            '/home/wuyang/workspace/python/poke/test_poke/',
            100)
        rospy.sleep(2.0)
        self.delete_gazebo_models(1)

        block_pose = Pose(position=Point(x=random.uniform(0.39, 0.51),
                                         y=random.uniform(0.39, 0.51),
                                         z=0.815),
                          orientation=Quaternion(x=0,
                                                 y=0,
                                                 z=-0.04997917,
                                                 w=0.99875026))
        resp_urdf = spawn_urdf("block", block_xml_first, "/",
                               block_pose, block_reference_frame)
        self.save_image(
            '/home/wuyang/workspace/python/poke/test_poke/',
            0)
        return name

def plot_sample(img_before, img_after, action, step):
    h, w, channels = img_before.shape
    fig, (ax1, ax2) = plt.subplots(1, 2)
    x, y, t, l = action
    x = x*h/240
    y = y*w/240
    dx = -400*l*math.cos(t)
    dy = -400*l*math.sin(t)
    ax1.arrow(w-y, x, dy, dx, head_width=5, head_length=5, color='green')
    if channels == 3:
        ax1.imshow(img_before)
        ax2.imshow(img_after)
    elif channels == 1:
        ax1.imshow(img_before.reshape((h, w)), 'gray')
        ax2.imshow(img_after.reshape((h, w)), 'gray')
    elif channels == 4:
        ax1.imshow(img_before[:, :, :3])
        ax2.imshow(img_after[:, :, :3])
    else:
        pass
    fig.suptitle('step %d: x=%.2f, y=%.2f, t=%.2f, l=%.2f)'
                 %(step, x, y, t/np.pi*180, l))

def query_image(img_num, target_shape, q_type, normalized):
    """
    q_type: 0->depth, 1->rgb, 2->total
    """
    if q_type == 1:
        output = image_feed(
            path+'img%04d.jpg'%img_num, target_shape, type_img=1, normalized=normalized)
    elif q_type == 0:
        output = image_feed(
            path+'depth%04d.png'%img_num, target_shape, type_img=0, normalized=normalized)
    else:
        output = image_feed_total(path+'img%04d.jpg'%img_num, path+'depth%04d.png'%img_num,
                                  target_shape, normalized=normalized)
    return output

if __name__ == '__main__':
    rospy.init_node('poke_test')

    init_pose = np.array([[0.105, 0.255, 0, 0, 0.14943813, 0.98877108]])
    path= '/home/wuyang/workspace/python/poke/test_poke/'
    i = 0
    k = 0
    poke_ros = PokeNewExtend(0.38)
    name = 'cube'
    poke_ros.delete_gazebo_models(0)
    name = poke_ros.load_gazebo_models()
    rospy.sleep(1.0)

    include_type=0
    target_shape = 227
    q_type = 2

    target_img = query_image(100, target_shape, q_type, normalized=0)
    poke_ros.move_to_start()

    with tf.Session() as sess:
        # poke_tf = Poke(include_type=include_type)
        # #restore_path = '/home/wuyang/workspace/python/tf/logs/poke/rs_165_4/'
        # restore_path = '/home/wuyang/workspace/python/tf/logs/poke_inv/rgb_1/'

        # poke_tf = PokeDepth(include_type=include_type, corrupted=0, target_size=target_shape)
        # restore_path = '/home/wuyang/workspace/python/tf/logs/poke_inv/depth_alex/'

        poke_tf = PokeTotal(include_type=include_type, target_size=target_shape)
        restore_path = '/home/wuyang/workspace/python/tf/logs/poke_inv/total_alex/'


        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(restore_path))

        try:
            while not rospy.is_shutdown():
                if i > 15:
                    _, _, bp, bo = poke_ros.poke_generation(name)
                    pose = np.array([[bp.x, bp.y, bo.x, bo.y, bo.z, bo.w]])
                    error = pose - init_pose
                    with open(path+'pose_type_%d.dat'%q_type, 'ab') as f_handle:
                        np.savetxt(f_handle, error)

                    poke_ros.delete_gazebo_models(0)
                    name = poke_ros.load_gazebo_models()
                    rospy.sleep(1.0)
                    target_img = query_image(100, target_shape, q_type, normalized=0)
                    i = 0
                    k +=1
                    if k > 10:
                        break

                current_img = query_image(i, target_shape, q_type, normalized=0)

                target_eval, current_eval = sess.run([target_img, current_img])
                feed_dict = {poke_tf.i1: current_eval, poke_tf.i2: target_eval}
                p_class, theta_class, l_class = sess.run(
                    [poke_tf.classes[0],
                     poke_tf.classes[1],
                     poke_tf.classes[2]],
                    feed_dict=feed_dict)

                p_x = p_class[0]//20*12+random.uniform(0, 12)
                p_y = p_class[0]%20*12+random.uniform(0, 12)
                theta = theta_class[0]*np.pi/18 + np.pi/36
                l = l_class[0]*0.004 + 0.002 + 0.04
                # plot_sample(current_eval[0]/255.0,
                #             target_eval[0]/255.0,
                #             [p_x, p_y, theta, l], i)
                # plt.pause(0.5)
                # plt.waitforbuttonpress()
                # plt.close()
                #generated from predicted pixel location, theta and l rather than action service.
                position_world = poke_ros.model_world_projection(p_x, p_y)

                poke_ros.poke(position_world,
                              {'theta': theta, 'l':l})

                rospy.sleep(1.0)
                poke_ros.save_image(path, i+1)
                i += 1

        except Exception as e:
            traceback.print_exc()
        finally:
            sys.exit(1)
