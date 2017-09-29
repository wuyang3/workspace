#!/usr/bin/env python
"""
Service from transforming 2d pixels to 3d points.
"""
import rospy
import image_geometry

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import (
    PoseStamped,
    PointStamped,
    Point,
)
from poke_gazebo.srv import (
    worldProj,
    pointTrans,
)

class world_projection(object):
    def __init__(self):
        self.pm = image_geometry.PinholeCameraModel()
        cam_info = rospy.wait_for_message('/my_gripper/camera1/rgb/camera_info',
                                          CameraInfo)
        self.pm.fromCameraInfo(cam_info)

        rospy.wait_for_service('/poke_tf_listener/coordinates_transform', 3.0)
        self._transform_to_world = rospy.ServiceProxy(
            '/poke_tf_listener/coordinates_transform',
            pointTrans)

        self.s = rospy.Service('/model_world_projection/world_projection',
                               worldProj,
                               self.handle_world_projection)

    def handle_world_projection(self, req):
        (x_proj, y_proj, z_proj) = self.pm.projectPixelTo3dRay((req.x, req.y))
        point_camera = PointStamped()
        point_camera.header.frame_id = 'camera1'
        point_camera.header.stamp = rospy.Time.now()
        point_camera.point = Point(x_proj, y_proj, z_proj)

        point_world = self._transform_to_world(point_camera, 'world')

        x_world = point_world.pointOut.point.x
        y_world = point_world.pointOut.point.y
        z_world = point_world.pointOut.point.z

        return (x_world, y_world, z_world)

if __name__ == '__main__':
    rospy.init_node('model_world_projection')
    poke_world_projection = world_projection()
    rospy.spin()
