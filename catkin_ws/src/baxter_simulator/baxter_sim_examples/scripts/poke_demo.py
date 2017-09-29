#!/usr/bin/env python
"""
Perform poking object demo.
"""
import argparse
import struct
import sys
import copy

import random
import math

import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
    GetModelState,
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface

class PokeDemo(object):
    def __init__(self, limb, hover_distance = 0.05, verbose=False):
        self._limb_name = limb
        self._hover_distance = hover_distance
        self._verbose = verbose
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns_ik = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns_ik, SolvePositionIK)
        rospy.wait_for_service(ns_ik, 5.0)
        ns_gs = "/gazebo/get_model_state"
        self._gssvc = rospy.ServiceProxy(ns_gs, GetModelState)
        rospy.wait_for_service(ns_gs, 5.0)
        print("Getting robot state...")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot...")
        self._rs.enable()

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._limb.joint_names(), [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_close()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("IK Service call failed: %s" %(e,))
            return False
        resp_seeds = struct.unpack('<%dB' %len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("SUCCESS - Valid Joint Solution Found from Seed Type: {0}"
                      .format((seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def poke_generation(self):
        theta = random.uniform(0, 2*math.pi)
        l = random.uniform(0.01, 0.05)
        poke_action = {'theta':theta, 'l':l}
        return poke_action

    def _approach(self, poke_action):
        resp = self._gssvc('block', 'world')
        approach = resp.pose
        # approach with a pose prepared for the poke action.
        approach.position.x = approach.position.x + \
                              math.cos(poke_action['theta']-math.pi)*0.04
        approach.position.y = approach.position.y + \
                              math.sin(poke_action['theta']-math.pi)*0.04
        approach.position.z = -0.129 + self._hover_distance
        approach.orientation.x = -0.0249590815779
        approach.orientation.y = 0.999649402929
        approach.orientation.z = 0.00737916180073
        approach.orientation.w = 0.00486450832011
        joint_angles = self.ik_request(approach)
        print('Approach 1...')
        self._guarded_move_to_joint_position(joint_angles)

        approach.position.z = -0.129
        joint_angles = self.ik_request(approach)
        print('Approach 2...')
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        print('Retract...')
        self._guarded_move_to_joint_position(joint_angles)

    def _servo_to_pose(self, poke_action):
        resp = self._gssvc('block', 'world')
        pose = resp.pose
        # pose modified to the poked location
        pose.position.x = pose.position.x + \
                          math.cos(poke_action['theta'])*poke_action['l']
        pose.position.y = pose.position.y + \
                          math.sin(poke_action['theta'])*poke_action['l']
        pose.position.z = -0.129
        pose.orientation.x = -0.0249590815779
        pose.orientation.y = 0.999649402929
        pose.orientation.z = 0.00737916180073
        pose.orientation.w = 0.00486450832011
        # servo to the expected poke location
        joint_angles = self.ik_request(pose)
        print('Poke...')
        self._guarded_move_to_joint_position(joint_angles)

    def poke(self):
        # generate poke action
        poke_action = self.poke_generation()
        # servo besides the object location to prepare for poking
        self._approach(poke_action)
        # servo to poke the object
        self._servo_to_pose(poke_action)
        # retract the gripper and prepare for the next poking.
        self._retract()
        rospy.sleep(1.0)

def load_gazebo_models(table_pose=Pose(position=Point(x=0.9, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.7, y=0.0, z=0.7825)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('baxter_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

def main():
    """
    Poke block demo.

    get_model_state and IKService are called. The performance can be improved by
    better ik solver and feedback control of the object.
    """
    rospy.init_node("poke_demo")
    # Spawning services. IK operates w.r.t /base frame while model reference is
    # /world frame.
    load_gazebo_models()
    # Remove model on shutdown
    rospy.on_shutdown(delete_gazebo_models)
    # Wait for the All Clear from emulator starup
    rospy.wait_for_message("/robot/sim/started", Empty)

    limb = 'left'
    hover_distance = 0.1
    # Starting joint angles for the left arm
    starting_joint_angles = {'left_w0': 0.6699952259595108,
                             'left_w1': 1.030009435085784,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.189968899785275,
                             'left_e1': 1.9400238130755056,
                             'left_s0': -0.08000397926829805,
                             'left_s1': -0.9999781166910306}

    poke_demo = PokeDemo(limb, hover_distance, True)
    # Move to starting position
    poke_demo.move_to_start(starting_joint_angles)
    i = 1
    while not rospy.is_shutdown():
        print("\nRound {0:d}...".format(i))
        poke_demo.poke()
        i += 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
