#!/usr/bin/env python
import roslib
import rospy
import tf
import math

if __name__ == '__main__':
    rospy.init_node('camera_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(50.0)
    while not rospy.is_shutdown():
        # Pose of camera1 frame in world frame.
        # In other words, pose of child frame in parent frame.
        # The euler angle is not correct here. The camera has its z axis
        # pointing forward, the actual pitch is 90 degrees more.
        # So it should be 4*pi/9+pi/2.
        br.sendTransform((0.26, 0.43, 1.2),
                         tf.transformations.quaternion_from_euler(
                             0, 4*math.pi/9, 0),
                         rospy.Time.now(),
                         "camera1",
                         "world")
        rate.sleep()
