#!/usr/bin/env python
"""
Test the correctness of model position projection on the image plane.
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

class image_proj_test(object):
    def __init__(self):
        ns_model  = '/gazebo/get_model_state'
        self._get_model_srv = rospy.ServiceProxy(ns_model, GetModelState)
        rospy.wait_for_service(ns_model, 2.0)

        ns_transform = '/poke_tf_listener/coordinates_transform'
        self._get_coords_transform = rospy.ServiceProxy(ns_transform, pointTrans)
        rospy.wait_for_service(ns_transform, 2.0)

        ns_image = '/model_image_projection/image_projection'
        self._get_image_projection = rospy.ServiceProxy(ns_image, imageProj)
        rospy.wait_for_service(ns_image, 2.0)

        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/my_gripper/camera1/rgb/image_raw',
                                           Image,
                                           self.image_test_handle)

        #cam_info = rospy.wait_for_message('/my_gripper/camera1/rgb/camera_info',
        #                                  CameraInfo)
        #self.pm = image_geometry.PinholeCameraModel()
        #self.pm.fromCameraInfo(cam_info)

    def image_test_handle(self, image):
        point_world = PointStamped()
        point_world.header.frame_id = 'world'
        point_world.header.stamp = rospy.Time.now()
        resp = self._get_model_srv('block', 'world')
        point_world.point = resp.pose.position

        point_camera = self._get_coords_transform(point_world, 'camera1')

        position = point_camera.pointOut.point
        projected = self._get_image_projection(position.x,
                                               position.y,
                                               position.z)
        #print 'projected x=%.2f, y=%d.2f'%(projected.x, projected.y)
        try:
            cv_img = self.bridge.imgmsg_to_cv2(image, 'bgr8')
            #cv_img_r = np.zeros((240, 240, 3), dtype='uint8')
            #self.pm.rectifyImage(cv_img, cv_img_r)
        except CvBridgeError as e:
            print(e)

        # Mind the Mat and Point difference in opencv and coordinate system.
        # For Mat, the index is (row, column) starting from top-left.
        # For image, the coordinate x would be column and y would be row.
        # However, one want to count pixels from top-left, the convention index
        # image is (column, row) from top-left. (ordinary x-y coordinates if
        # from bottom left)
        x = int(round(projected.x))
        y = int(round(projected.y))
        cv2.circle(cv_img, (240-y, x), 5, (0, 255, 0), -1)
        cv2.imshow('model projection', cv_img)
        cv2.waitKey(1)

def main(args):
    rospy.init_node('image_proj_test')
    image_test = image_proj_test()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "shutting down..."
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
