#!/usr/bin/env python
"""
Test: pixel locations -> camera coords -> world coords ->
      camera coords -> pixel locations

Check if pixels to world mapping is correct.
"""
import rospy
import random

from poke_gazebo.srv import (
    imageProj,
    worldProj,
    pointTrans,
)

from geometry_msgs.msg import (
    Point,
    PointStamped,
)


def random_location():
    x = random.uniform(0, 240)
    y = random.uniform(0, 240)
    return x, y

if __name__ == '__main__':
    rospy.init_node('inverse_proj')

    camera_loc = [0.26, 0.43, 1.2] # correct location?

    rospy.wait_for_service('/poke_tf_listener/coordinates_transform', 1.0)
    coord_trans = rospy.ServiceProxy('/poke_tf_listener/coordinates_transform',
                                     pointTrans)

    rospy.wait_for_service('/model_image_projection/image_projection', 1.0)
    img_proj = rospy.ServiceProxy('/model_image_projection/image_projection',
                                  imageProj)

    rospy.wait_for_service('/model_world_projection/world_projection', 1.0)
    world_proj = rospy.ServiceProxy('/model_world_projection/world_projection',
                                    worldProj)


    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        # Coordinates transform and image projection test.
        # To world.
        x_pixel, y_pixel = random_location()
        point_temp = world_proj(x_pixel, y_pixel)
        x_w = point_temp.x
        y_w = point_temp.y
        z_w = point_temp.z

        point_world = PointStamped()
        point_world.header.frame_id = 'world'

        z_obj = 0.8
        k = (z_obj - camera_loc[2])/(z_w - camera_loc[2])
        x_obj = camera_loc[0] + k*(x_w - camera_loc[0])
        y_obj = camera_loc[1] + k*(y_w - camera_loc[1])

        point_world.header.stamp = rospy.Time.now()
        point_world.point = Point(x_obj, y_obj, z_obj)

        # To pixel.
        point_camera = coord_trans(point_world, 'camera1')
        position = point_camera.pointOut.point

        proj = img_proj(position.x,
                        position.y,
                        position.z)

        print("Generated pixel location: (%.4f, %.4f)\n"%(x_pixel, y_pixel)+
              "Projected object location: (%.4f, %.4f)\n"%(proj.x, proj.y))

        """
        # image projection test
        z_cam = random.uniform(0,20)
        proj = img_proj(0, 0, z_cam)
        print("Projected location: (%.4f, %.4f)"%(proj.x, proj.y))
        """

        """
        # Coordinates transform test
        x_cam = random.uniform(0,12)
        y_cam = random.uniform(0,12)
        z_cam = random.uniform(0,12)

        point_cam = PointStamped()
        point_cam.header.frame_id = 'camera1'
        point_cam.header.stamp = rospy.Time.now()
        point_cam.point = Point(x_cam, y_cam, z_cam)

        point_w = coord_trans(point_cam, 'world')
        point_cam_back = coord_trans(point_w.pointOut, 'camera1')

        position_c = point_cam.point
        position_cb = point_cam_back.pointOut.point
        print('generated point: (%.8f, %.8f, %.8f)\n'
              %(position_c.x, position_c.y, position_c.z)+
              'projected point: (%.8f, %.8f, %.8f)\n'
              %(position_cb.x, position_cb.y, position_cb.z))
        """

        rate.sleep()
