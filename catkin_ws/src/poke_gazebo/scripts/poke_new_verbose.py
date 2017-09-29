#!/usr/bin/env python
"""
Poke demo with three links robot.
more call on GetModelState service in each substeps.
"""
import sys
import os
import random
import math
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
)

from poke_gazebo.srv import (
    poseTrans,
    imageProj,
    imageSave,
)

class PokeNew(object):
    def __init__(self, hover_distance=0.05):
        self._hover_distance = hover_distance
        ns_model  = '/gazebo/get_model_state'
        self._get_model_srv = rospy.ServiceProxy(ns_model, GetModelState)
        rospy.wait_for_service(ns_model, 5.0)

        ns_joint = '/gazebo/get_joint_properties'
        self._get_joint_srv = rospy.ServiceProxy(ns_joint, GetJointProperties)
        rospy.wait_for_service(ns_joint, 5.0)

        ns_transform = '/poke_tf_listener/coordinates_transform'
        self._get_coords_transform = rospy.ServiceProxy(ns_transform, poseTrans)
        rospy.wait_for_service(ns_transform, 5.0)

        ns_image = '/model_image_projection/image_projection'
        self._get_image_projection = rospy.ServiceProxy(ns_image, imageProj)
        rospy.wait_for_service(ns_image, 5.0)

        ns_save = '/model_image_saver/image_save'
        self.save_image = rospy.ServiceProxy(ns_save, imageSave)
        rospy.wait_for_service(ns_save, 5.0)

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
        """
        self.pub_gripper_l_controller = rospy.Publisher(
            '/my_gripper/gripper_l_finger_controller/command',
            Float64,
            queue_size=10)
        self.pub_gripper_r_controller = rospy.Publisher(
            '/my_gripper/gripper_r_finger_controller/command',
            Float64,
            queue_size=10)
        self.pub_gripper_l_controller.publish(0.0)
        self.pub_gripper_r_controller.publish(0.0)
        """

    def poke_generation(self):
        theta = random.uniform(0, 2*math.pi)
        l = random.uniform(0.01, 0.05)
        poke_action = {'theta': theta, 'l': l}
        return poke_action

    def world_to_joint(self, position):
        # joint 3 moves in x direction, corresponds to controller3.
        position.x = 0.8 - position.x
        # joint 2 moves in y direction, corresponds to controller2.
        position.y = 0.82 - position.y
        return position

    def model_image_projection(self):
        pose_world = PoseStamped()
        pose_world.header.frame_id = 'world'
        # transform to camera frame
        resp = self._get_model_srv('block', 'world')
        pose_world.header.stamp = rospy.Time.now()
        pose_world.pose = resp.pose
        pose_camera = self._get_coords_transform(pose_world)
        # project on image plane
        position = pose_camera.poseOut.pose.position
        projected = self._get_image_projection(position.x,
                                               position.y,
                                               position.z)
        rospy.sleep(0.5)
        return projected

    def move_to_start(self):
        rospy.sleep(0.5)
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

    def _approach_planar(self, poke_action):
        resp = self._get_model_srv('block', 'world')
        position = resp.pose.position

        position.x = position.x + math.cos(poke_action['theta']-math.pi)*0.055
        position.y = position.y + math.sin(poke_action['theta']-math.pi)*0.055
        position = self.world_to_joint(position)

        self.pub_controller2.publish(position.y)
        self.pub_controller3.publish(position.x)

    def _approach_vertical(self):
        self.pub_controller1.publish(0.0)

    def _retract(self):
        self.pub_controller1.publish(self._hover_distance)

    def _trajectory_tracking(self, poke_action):
        joint2 = self._get_joint_srv('joint2')
        joint2_position = joint2.position[0]

        joint3 = self._get_joint_srv('joint3')
        joint3_position = joint3.position[0]

        resp = self._get_model_srv('block', 'world')
        position = resp.pose.position
        poke_x = position.x + \
                 poke_action['l']*math.cos(poke_action['theta']) - \
                 (0.8-joint3_position)
        poke_y = position.y + \
                 poke_action['l']*math.sin(poke_action['theta']) - \
                 (0.82-joint2_position)

        joint2_temp = joint2_position
        joint3_temp = joint3_position
        for i in range(0, 19):
            joint2_temp = joint2_temp - poke_y/20
            joint3_temp = joint3_temp - poke_x/20
            self.pub_controller2.publish(joint2_temp)
            self.pub_controller3.publish(joint3_temp)
            rospy.sleep(0.2)

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
        rospy.sleep(4.0)
        print('    back  off...')
        self.pub_controller2.publish(joint2_position)
        self.pub_controller3.publish(joint3_position)

    def poke(self, poke_action, projected):
        print(("\nPoke action: pixel at ({0:4.2f},{1:4.2f}), "
              "theta={2:4.2f}, l={3:4.2f}").format(
                  projected.x, projected.y,
                  poke_action["theta"], poke_action["l"]))
        print("Approaching above...")
        self._approach_planar(poke_action)
        rospy.sleep(1.0)
        print("Rotate...")
        self._approach_direction(poke_action)
        rospy.sleep(1.5)
        print("Approaching vertically...")
        self._approach_vertical()
        rospy.sleep(0.5)
        print("poke...")
        self._trajectory_tracking(poke_action)
        rospy.sleep(1.0)
        print("retract...\n")
        self._retract()
        rospy.sleep(1.0)

def load_gazebo_models(block_pose=Pose(position=Point(x=0.4, y=0.43, z=0.80)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('poke_gazebo')+"/models/block/"
    file_list = ['model','model_blue','model_brown','model_green','model_orange']
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

def delete_gazebo_models():
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

def main():
    """
    Main function for executing the poke action.
    Python open option ab: append and binary mode.
    """
    hover_distance = 0.38
    k = 12
    path = '/home/wuyang/workspace/catkin_ws/src/poke_gazebo/data/run_'+'%02d/'%k
    if not os.path.exists(path):
        os.makedirs(path)
    rospy.init_node('poke_new')
    poke_new = PokeNew(hover_distance)
    poke_new.move_to_start()

    i = 0
    poke_new.save_image(path ,i)
    while not rospy.is_shutdown():
        resp = poke_new._get_model_srv('block', 'world')
        if resp.pose.position.z < 0.775:
            delete_gazebo_models()
            load_gazebo_models()
            rospy.sleep(10.0)
            k += 1
            path = '/home/wuyang/workspace/catkin_ws/src/poke_gazebo/data/run_' \
                   +'%02d/'%k
            if not os.path.exists(path):
                os.makedirs(path)
            i = 0
            poke_new.save_image(path, i)

        print("----------\nRound {0:d}:".format(i))
        poke_action = poke_new.poke_generation()
        projected = poke_new.model_image_projection()

        poke = np.array([[projected.x, projected.y,
                          poke_action['theta'],poke_action['l']]])
        with open(path+'actions.dat', 'ab') as f_handle:
            np.savetxt(f_handle, poke)

        poke_new.poke(poke_action, projected)
        poke_new.save_image(path ,i+1)
        i += 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
