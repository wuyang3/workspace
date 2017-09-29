#!/usr/bin/env python
"""
def handle_model_pose(msg, pub):
    model_state = PoseStamped()
    model_state.header.frame_id = 'world'
    model_ind = msg.name.index('block')
    pose = msg.pose[model_ind]
    model_state.pose = pose
    model_state.header.stamp = rospy.Time.now()
    pub.publish(model_state)

def model_publisher():
    rospy.init_node('poke_model_publisher')
    #m_name = rospy.get_param('~model_name')
    pub = rospy.Publisher('/my_gripper/model_state', PoseStamped, queue_size=10)

    rospy.Subscriber('/gazebo/model_states', ModelStates,
                     handle_model_pose, pub)
    rospy.spin()
"""
import rospy
import roslib
from geometry_msgs.msg import (
    PoseStamped,
    PointStamped,
)
from gazebo_msgs.srv import (
    GetModelState,
)

from gazebo_msgs.msg import (
    ModelStates,
)

def model_publisher():
    rospy.init_node('model_state_publisher')
    model_name = rospy.get_param('~model_name')
    pub = rospy.Publisher('/my_gripper/model_state', PoseStamped, queue_size=10)

    # gazebo get_model_state service
    ns='/gazebo/get_model_state'
    gms = rospy.ServiceProxy(ns, GetModelState)
    rospy.wait_for_service(ns, 2.0)

    model_state = PoseStamped()
    model_state.header.frame_id = 'world'

    rate = rospy.Rate(50.0)
    while not rospy.is_shutdown():
        resp = gms(model_name, 'world')
        model_state.header.stamp = rospy.Time.now()
        model_state.pose = resp.pose # equivalent
        pub.publish(model_state)
        rate.sleep()

if __name__ == '__main__':
    model_publisher()
