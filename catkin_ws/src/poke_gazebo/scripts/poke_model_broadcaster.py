#!/usr/bin/env python
import roslib
import rospy

import tf
from geometry_msgs.msg import (
    PoseStamped,
)
from gazebo_msgs.srv import (
    GetModelState,
)
from gazebo_msgs.msg import (
    ModelStates,
)

def handle_model_pose(msg, model_name):
    model_ind = msg.name.index('block')
    pose = msg.pose[model_ind]
    br = tf.TransformBroadcaster()
    br.sendTransform((pose.position.x, pose.position.y, pose.position.z),
                     (pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w),
                     rospy.Time.now(),
                     model_name,
                     "world")

if __name__ == '__main__':
    rospy.init_node('model_tf_broadcaster')
    # model_name will be specified in node of launch file.
    m_name = rospy.get_param('~model_name')

    # option1: pick out block state from gazebo topic and send transform.
    rospy.Subscriber('/gazebo/model_states', ModelStates,
                     handle_model_pose, m_name)

    # option2: subscribe to block state topic which is published manually and
    # send transform. Block state is published in poke_model_publisher.py
    #rospy.Subscriber('/my_gripper/model_state', PoseStamped,
    #                 handle_model_pose, m_name)
    rospy.spin()
