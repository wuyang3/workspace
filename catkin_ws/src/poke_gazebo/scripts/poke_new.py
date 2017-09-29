#!/usr/bin/env python
"""
Poke demo with three links robot to collect data. Here we always poke through the center of the object. So it starts from somewhere behind the object center and poke through it by randomly generated angle and length.
"""
import sys
import os
import random
import math
import copy
import numpy as np

import rospy
import rospkg

from gazebo_msgs.srv import (
    DeleteModel,
    SpawnModel,
    GetModelState,
    GetJointProperties,
)

from std_msgs.msg import (
    Float64,
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    PointStamped,
)

from poke_gazebo.srv import (
    pointTrans,
    imageProj,
    worldProj,
    imageSave,
)

class PokeNew(object):
    def __init__(self, hover_distance=0.05):
        self._hover_distance = hover_distance
        ns_model  = '/gazebo/get_model_state'
        rospy.wait_for_service(ns_model, 2.0)
        self._get_model_srv = rospy.ServiceProxy(ns_model, GetModelState)

        ns_joint = '/gazebo/get_joint_properties'
        rospy.wait_for_service(ns_joint, 2.0)
        self._get_joint_srv = rospy.ServiceProxy(ns_joint, GetJointProperties)

        ns_transform = '/poke_tf_listener/coordinates_transform'
        rospy.wait_for_service(ns_transform, 2.0)
        self._get_coords_transform = rospy.ServiceProxy(ns_transform, pointTrans)

        ns_image = '/model_image_projection/image_projection'
        rospy.wait_for_service(ns_image, 2.0)
        self._get_image_projection = rospy.ServiceProxy(ns_image, imageProj)

        ns_world = '/model_world_projection/world_projection'
        rospy.wait_for_service(ns_world, 2.0)
        self._get_world_projection = rospy.ServiceProxy(ns_world, worldProj)

        ns_save = '/model_image_saver/image_save'
        #ns_save = '/model_depth_saver/depth_save'
        rospy.wait_for_service(ns_save, 2.0)
        self.save_image = rospy.ServiceProxy(ns_save, imageSave)

        self.pub_controller1 = rospy.Publisher(
            '/my_gripper/joint1_position_controller/command',
            Float64,
            queue_size=10)
        self.pub_controller2 = rospy.Publisher(
            '/my_gripper/joint2_position_controller/command',
            Float64,
            queue_size=10)
        self.pub_controller3 = rospy.Publisher(
            '/my_gripper/joint3_position_controller/command',
            Float64,
            queue_size=10)
        self.pub_base_controller = rospy.Publisher(
            '/my_gripper/gripper_base_controller/command',
            Float64,
            queue_size=10)

    def poke_generation(self):
        resp = self._get_model_srv('block', 'world')
        position = resp.pose.position
        theta = random.uniform(0, 2*math.pi)
        l = random.uniform(0.01, 0.05)
        poke_action = {'theta': theta, 'l': l}
        return position, poke_action

    def world_to_joint(self, position):
        position_wj = copy.deepcopy(position)
        # joint 3 moves in x direction, corresponds to controller3.
        position_wj.x = 0.8 - position_wj.x
        # joint 2 moves in y direction, corresponds to controller2.
        position_wj.y = 0.82 - position_wj.y
        return position_wj

    def model_image_projection(self):
        """
        pose_world = PoseStamped()
        pose_world.header.frame_id = 'world'
        # transform to camera frame
        resp = self._get_model_srv('block', 'world')
        pose_world.header.stamp = rospy.Time.now()
        pose_world.pose = resp.pose
        pose_camera = self._get_coords_transform(pose_world, 'camera1')
        # project on image plane
        position = pose_camera.poseOut.pose.position
        projected = self._get_image_projection(position.x,
                                               position.y,
                                               position.z)
        rospy.sleep(0.5)
        """
        point_world = PointStamped()
        point_world.header.frame_id = 'world'
        resp = self._get_model_srv('block', 'world')
        point_world.header.stamp = rospy.Time.now()
        point_world.point = resp.pose.position

        point_camera = self._get_coords_transform(point_world, 'camera1')
        position = point_camera.pointOut.point
        projected = self._get_image_projection(position.x,
                                               position.y,
                                               position.z)

        return projected

    def model_world_projection(self, x_pixel, y_pixel):
        camera_loc = [0.26, 0.43, 1.2]
        point_w = self._get_world_projection(x_pixel, y_pixel)
        x_w = point_w.x
        y_w = point_w.y
        z_w = point_w.z

        z_obj = 0.8
        k = (z_obj - camera_loc[2])/(z_w - camera_loc[2])
        x_obj = camera_loc[0] + k*(x_w - camera_loc[0])
        y_obj = camera_loc[1] + k*(y_w - camera_loc[1])
        point_world = Point()
        point_world.x = x_obj
        point_world.y = y_obj
        point_world.z = z_obj

        return point_world

    def move_to_start(self):
        self.pub_controller1.publish(self._hover_distance)
        rospy.sleep(0.5)
        resp = self._get_model_srv('block', 'world')
        position = resp.pose.position
        position = self.world_to_joint(position)
        self.pub_controller2.publish(position.y)
        self.pub_controller3.publish(position.x)
        print("Moved to starting position...")
        rospy.sleep(0.5)

    def _approach_direction(self, poke_action):
        if poke_action['theta'] < math.pi:
            theta = -poke_action['theta']
        else:
            theta = 2*math.pi - poke_action['theta']
        self.pub_base_controller.publish(theta)

    def _approach_planar(self, position, poke_action):
        position_p = copy.deepcopy(position)
        position_p.x = position_p.x + math.cos(poke_action['theta']-math.pi)*0.055
        position_p.y = position_p.y + math.sin(poke_action['theta']-math.pi)*0.055
        position_p = self.world_to_joint(position_p)

        self.pub_controller2.publish(position_p.y)
        self.pub_controller3.publish(position_p.x)

    def _approach_vertical(self):
        self.pub_controller1.publish(0.0)

    def _retract(self):
        self.pub_controller1.publish(self._hover_distance)

    def _trajectory_tracking(self, position, poke_action):
        position_t = copy.deepcopy(position)
        joint2 = self._get_joint_srv('joint2')
        joint2_position = joint2.position[0]

        joint3 = self._get_joint_srv('joint3')
        joint3_position = joint3.position[0]

        poke_x = position_t.x + \
                 poke_action['l']*math.cos(poke_action['theta']) - \
                 (0.8-joint3_position)
        poke_y = position_t.y + \
                 poke_action['l']*math.sin(poke_action['theta']) - \
                 (0.82-joint2_position)

        joint2_temp = joint2_position
        joint3_temp = joint3_position
        for i in range(0, 19):
            joint2_temp = joint2_temp - poke_y/20
            joint3_temp = joint3_temp - poke_x/20
            self.pub_controller2.publish(joint2_temp)
            self.pub_controller3.publish(joint3_temp)
            rospy.sleep(0.1)

        """
        resp = self._get_model_srv('block', 'world')
        position = resp.pose.position
        position.x = position.x + \
                     math.cos(poke_action['theta'])*poke_action['l']
        position.y = position.y + \
                     math.sin(poke_action['theta'])*poke_action['l']
        position = self.world_to_joint(position)
        self.pub_controller2.publish(position.y)
        self.pub_controller3.publish(position.x)
        """
        rospy.sleep(0.5)
        print('    back  off...')
        self.pub_controller2.publish(joint2_position)
        self.pub_controller3.publish(joint3_position)

    def poke(self, position, poke_action):
        print(("\ntheta={0:4.2f}, l={1:4.3f}").format(
            poke_action["theta"], poke_action["l"]))
        print("Approaching above...")
        self._approach_planar(position, poke_action)
        rospy.sleep(1.0)
        print("Rotate...")
        self._approach_direction(poke_action)
        rospy.sleep(1.5)
        print("Approaching vertically...")
        self._approach_vertical()
        rospy.sleep(0.5)
        print("poke...")
        self._trajectory_tracking(position, poke_action)
        rospy.sleep(0.5)
        print("retract...\n")
        self._retract()
        rospy.sleep(0.5)

def load_gazebo_models(block_pose=Pose(position=Point(x=0.4, y=0.43, z=0.80)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('poke_gazebo')+"/models/block/"
    file_list = ['block','block_blue','block_white','block_green','block_orange',
                 'block_l_1', 'block_l_2', 'block_l_3', 'block_l_4',
                 'cylinder_1', 'cylinder_2', 'cylinder_3', 'cylinder_4', 'cylinder_5']

    rospy.wait_for_service('/gazebo/spawn_urdf_model',2.0)
    spawn_urdf = rospy.ServiceProxy(
                    '/gazebo/spawn_urdf_model', SpawnModel)
    num = random.randint(1,3)
    chosen_list = random.sample(file_list, num)
    for i, item in enumerate(chosen_list):
        file_name = item+'.urdf'
        block_xml = ''
        with open (model_path + file_name, "r") as block_file:
            block_xml=block_file.read().replace('\n', '')
        if i == 0:
            try:
                resp_urdf = spawn_urdf("block", block_xml, "/",
                                       block_pose, block_reference_frame)
            except rospy.ServiceException as e:
                rospy.logerr("Spawn URDF service call failed: {0}".format(e))
        else:
            block_pose_rand = Pose(position=Point(x=random.uniform(0.1, 0.7),
                                                  y=random.uniform(0.13, 0.73),
                                                  z=0.80))
            try:
                resp_urdf = spawn_urdf("block"+'_%d'%i, block_xml, "/",
                                       block_pose_rand, block_reference_frame)
            except rospy.ServiceException as e:
                rospy.logerr("Spawn URDF service call failed: {0}".format(e))
    """
    ind = random.randint(0,4)
    file_name = file_list[ind]+'.urdf'
    # Load Block URDF
    block_xml = ''
    with open (model_path + file_name, "r") as block_file:
        block_xml=block_file.read().replace('\n', '')

    # Spawn Block URDF all with name block
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))
    """

def delete_gazebo_models():
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("block")

        resp_delete = delete_model("block_1")
        resp_delete = delete_model("block_2")

    except rospy.ServiceException as e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

def main():
    """
    Main function for executing the poke action.
    Python open option ab: append and binary mode.
    """
    hover_distance = 0.38
    k = 266
    path = '/home/wuyang/workspace/catkin_ws/src/poke_gazebo/data/run_'+'%03d/'%k
    if not os.path.exists(path):
        os.makedirs(path)
    rospy.init_node('poke_new')
    poke_new = PokeNew(hover_distance)

    i = 0
    rospy.sleep(3.0)
    success_flag = poke_new.save_image(path, i)
    rospy.sleep(2.0)
    poke_new.move_to_start()

    while not rospy.is_shutdown():
        resp = poke_new._get_model_srv('block', 'world')
        if resp.pose.position.z < 0.7:
            delete_gazebo_models()
            load_gazebo_models()
            k += 1
            path = '/home/wuyang/workspace/catkin_ws/src/poke_gazebo/data/run_' \
                   +'%03d/'%k
            if not os.path.exists(path):
                os.makedirs(path)
            i = 0
            rospy.sleep(2.0)
            success_flag = poke_new.save_image(path, i)
            rospy.sleep(2.0)
            poke_new.move_to_start()

        print("----------\nRound {0:d}:".format(i))
        position, poke_action = poke_new.poke_generation()
        projected = poke_new.model_image_projection()
        #world = poke_new.model_world_projection(projected.x, projected.y)
        #print('projected: x=%4.2f y=%4.2f, world: x=%4.2f, y=%4.2f, z=%4.2f'
        #      %(projected.x, projected.y, world.x, world.y, world.z))

        poke = np.array([[position.x, position.y, position.z,
                          projected.x, projected.y,
                          poke_action['theta'],poke_action['l']]])
        with open(path+'actions.dat', 'ab') as f_handle:
            np.savetxt(f_handle, poke)

        poke_new.poke(position, poke_action)
        rospy.sleep(1.0)
        success_flag = poke_new.save_image(path, i+1)
        rospy.sleep(2.0)
        i += 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
