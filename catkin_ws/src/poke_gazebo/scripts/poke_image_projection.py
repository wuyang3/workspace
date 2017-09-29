#!/usr/bin/env python
"""
PoseStamped.pose.Point or Quaternion has float64 attributes.
Unsuccessful trial includes explicitly passing in PinholeCameraModel instance
to the subscriber callback (which is probably fine) and service callback
(problematic). Using class here makes it unneccessary to pass in additional
methods/instance.
Service returns imageProjResponse object which has attributes x and y.
"""
import rospy
import image_geometry

from sensor_msgs.msg import (
     CameraInfo,
)

from poke_gazebo.srv import imageProj

class image_projection(object):
    def __init__(self):
        self.pm = image_geometry.PinholeCameraModel()
        cam_info = rospy.wait_for_message('/my_gripper/camera1/rgb/camera_info',
                                          CameraInfo)
        self.pm.fromCameraInfo(cam_info)
        # try modifying the P matrix.
        #P = cam_info.P
        #P_l = list(P)
        #P_l[3] = 0
        #cam_info.P = tuple(P_l)

        self.s = rospy.Service('/model_image_projection/image_projection',
                               imageProj,
                               self.handle_image_projection)
        print("image projection service ready...")

    def handle_image_projection(self, req):
        (x_proj, y_proj) = self.pm.project3dToPixel((req.x, req.y, req.z))
        return (x_proj, y_proj)

if __name__ == '__main__':
    rospy.init_node('model_image_projection')
    poke_image_projection = image_projection()
    rospy.spin()
