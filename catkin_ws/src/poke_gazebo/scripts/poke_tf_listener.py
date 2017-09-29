#!/usr/bin/env python
"""
source frame id contained in req.poseIn (type poseStamped). Target frame is specified by req.targetFrame. return response.poseOut.

Pose transform is changed into point transform.

rate = rospy.Rate(50.0)
while not rospy.is_shutdown():
    try:
        # tf.TransformListener.lookupTransform(target, source frame...)
        # the pose of source frame in target frame. This pose as
        # rigid transform can transform coordinates in source frame into
        #  target frame.
        (trans, rot) = listener.lookupTransform('world',
                                                'camera1',
                                                rospy.Time(0))
        rospy.loginfo("transform found\n")
        rospy.loginfo(trans)
     except (tf.LookupException, tf.ConnectivityException,
             tf.ExtrapolationException):
        #rospy.loginfo("transform not found\n")
        continue
    rate.sleep()
"""
import numpy as np
import roslib
import rospy
import tf
from geometry_msgs.msg import PointStamped, Point
from poke_gazebo.srv import pointTrans

class poke_tf_listener(object):
    def __init__(self):
        self.listener = tf.TransformListener()
        self.s = rospy.Service('/poke_tf_listener/coordinates_transform',
                               pointTrans,
                               self.handle_pointTrans)
        print("coordinates transform service ready...")

    # return pointTransResponse object which has attribute pointOut
    # defined in srv file.
    def handle_pointTrans(self, req):
        # option1: wait for transform
        #self.listener.waitForTransform(req.targetFrame,
        #                               req.pointIn.header.frame_id,
        #                               req.pointIn.header.stamp,
        #                               rospy.Duration(0.05))
        #point = self.listener.transformPoint(req.targetFrame, req.pointIn)

        # option2: take the latest transform. For static transform in camera
        # frame this is handy and save time. The transform stays the same and
        # there is no need to wait. For block frame transform, a little
        # time difference does not matter when generating poke action data,
        # meaning point in block -> point in world -> point in camera
        # (same if using rospy.Time.now()) -> projected.
        # The first transform won't be much different. Especially when saving
        # poke position data, the block is static before the poking.
        translation,rotation = self.listener.lookupTransform(
            req.targetFrame, req.pointIn.header.frame_id, rospy.Time(0))
        mat44 = self.listener.fromTranslationRotation(translation, rotation)
        xyz = tuple(np.dot(mat44, np.array([
            req.pointIn.point.x,
            req.pointIn.point.y,
            req.pointIn.point.z, 1.0])))[:3]
        point = PointStamped()
        point.header.stamp = req.pointIn.header.stamp
        point.header.frame_id = req.targetFrame
        point.point = Point(*xyz)

        return {'pointOut': point}

if __name__ == '__main__':
    rospy.init_node('poke_tf_listener')
    tf_listener = poke_tf_listener()
    rospy.spin()
