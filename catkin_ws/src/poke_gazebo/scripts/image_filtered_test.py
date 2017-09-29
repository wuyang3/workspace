#!/usr/bin/env python
"""
Test the correctness of random poke point picking. Message filter works on two
topics, namely /my_gripper/poke_point and /my_gripper/camera1/rgb/image_raw
"""
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import (
    Image,
    CameraInfo,
)
from cv_bridge import CvBridge, CvBridgeError
import image_geometry

from geometry_msgs.msg import PoseStamped, PointStamped
from gazebo_msgs.srv import GetModelState
from poke_gazebo.srv import (
     pointTrans,
     imageProj,
)
import message_filters

class ImageFilteredTest(object):
    def __init__(self):
        ns_transform = '/poke_tf_listener/coordinates_transform'
        rospy.wait_for_service(ns_transform, 2.0)
        self._get_coords_transform = rospy.ServiceProxy(ns_transform, pointTrans)

        ns_image = '/model_image_projection/image_projection'
        rospy.wait_for_service(ns_image, 2.0)
        self._get_image_projection = rospy.ServiceProxy(ns_image, imageProj)

        self.bridge = CvBridge()

        self.img_sub = message_filters.Subscriber(
            '/my_gripper/camera1/rgb/image_raw', Image)
        self.point_sub = message_filters.Subscriber(
            '/my_gripper/poke_point', PointStamped)

        self.tss = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.point_sub], 10, 0.02)
        self.tss.registerCallback(self.handle_image_poke)

    def handle_image_poke(self, image, pointstamped):
        point_camera = self._get_coords_transform(pointstamped, 'camera1')
        position = point_camera.pointOut.point
        projected = self._get_image_projection(position.x,
                                               position.y,
                                               position.z)
        try:
            cv_img = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        x = int(round(projected.x))
        y = int(round(projected.y))
        cv2.circle(cv_img, (240-y, x), 5, (0, 255, 0), -1)
        cv2.imshow('model projection', cv_img)
        cv2.waitKey(1)

def main(args):
    rospy.init_node('image_filtered_test')
    image_filtered_test = ImageFilteredTest()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "shutting down..."
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
