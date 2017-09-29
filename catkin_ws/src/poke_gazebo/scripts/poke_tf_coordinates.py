#!/usr/bin/env python
import roslib
import rospy
import tf
from geometry_msgs.msg import (
    PoseStamped,
    PointStamped,
)

from poke_gazebo.srv import imageProj

class poke_tf_coordinates(object):
    def __init__(self):
        ns = '/model_image_projection/image_projection'
        self.imgProj = rospy.ServiceProxy(ns, imageProj)
        rospy.wait_for_service(ns, 5.0)

        self.listener = tf.TransformListener()

        rospy.Subscriber('/my_gripper/model_state', PoseStamped,
                         self.handle_model_pose)

    def handle_model_pose(self, msg):
        """
        # here PointStamped can be transfered to PoseStamped for easy
        # transform to camera frame. Don't need to transform every msg!
        model_state = PointStamped()
        model_state.header = msg.header
        model_state.point.x = msg.pose.position.x
        model_state.point.y = msg.pose.position.y
        model_state.point.z = msg.pose.position.z
        """
        model_state_camera = self.listener.transformPose('camera1', msg)
        position = model_state_camera.pose.position
        #print position
        projected = self.imgProj(position.x, position.y, position.z)

        print projected


if __name__ == '__main__':
    rospy.init_node('poke_tf_coordinates')
    tf_coords = poke_tf_coordinates()
    rospy.spin()
