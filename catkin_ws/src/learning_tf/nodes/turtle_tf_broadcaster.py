#!/usr/bin/env python
import roslib
roslib.load_manifest('learning_tf')
import rospy

import tf
import turtlesim.msg

def handle_turtle_pose(msg, turtlename):
    br = tf.TransformBroadcaster()
    # tf.TransformBroadcaster.sendTransform(
    # translation, rotation, time, child, parent) broadcast transform from
    # parent to child/ pose of child in parent. The transform is published to /tf.
    # child frame is the turtle and parent frame is the world.
    br.sendTransform((msg.x, msg.y, 0),
                     tf.transformations.quaternion_from_euler(0, 0, msg.theta),
                     rospy.Time.now(),
                     turtlename,
                     "world"
    )

if __name__ == '__main__':
    rospy.init_node('turtle_tf_broadcaster')
    turtlename = rospy.get_param('~turtle')
    rospy.Subscriber('/%s/pose'%turtlename,
                     turtlesim.msg.Pose,
                     handle_turtle_pose,
                     turtlename
    )
    rospy.spin()
