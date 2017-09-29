#!/usr/bin/env python
"""
Randomly pick a point on the surface of the object. First pick it in its local frame and then transform it into the world frame. The poke direction is restricted to the side of objects where poking is possible. Theta is pointed to the direction of poking and within the range of (0, 2*pi) in the object body x-y frame.
theta range at each side of the object needs to be carefully selected.
header.frame_id is block. The transform for frame block is sent in poke_model_broadcaster.py
"""
import rospy
import random
import math
from geometry_msgs.msg import Point, PointStamped
import tf
from poke_gazebo.srv import pointTrans, actionGen
from gazebo_msgs.srv import GetModelState

class poke_action_generation(object):
    def __init__(self):
        self.cube_l = 0.08
        self.cuboid_l = 0.12
        self.cuboid_w = 0.06
        self.cuboid_h = 0.08
        self.cylin_r = 0.04
        self.cylin_h = 0.08

        rospy.wait_for_service('/gazebo/get_model_state', 3.0)
        self._get_model_srv = rospy.ServiceProxy(
            '/gazebo/get_model_state',
            GetModelState)
        rospy.wait_for_service('/poke_tf_listener/coordinates_transform', 3.0)
        self.transform_to_world = rospy.ServiceProxy(
            '/poke_tf_listener/coordinates_transform',
            pointTrans)

        self.ps = PointStamped()
        self.ps.header.frame_id = 'block'
        self.ps.header.stamp = rospy.Time.now()

        self.s = rospy.Service('/model_action_generation/action_generation',
                               actionGen,
                               self.handle_action_generation)
        print("action generation service ready...")

    def handle_action_generation(self, req):
        """
        original paper data collection: l range 0.01 - 0.05
        Last data collection: l range 0.02 - 0.06
                              angle to sie range 15 - 165 degrees
        Now data collection: only cude
                             l range 0.04 - 0.08
                             angle to side range 45 -135

        """
        poke_temp = PointStamped()
        poke_temp.header.frame_id = 'block'
        poke_temp.point.z = 0
        poke_temp.header.stamp = rospy.Time.now()
        if req.name == 'cube':
            poke_option = [[random.uniform(-self.cube_l/2, self.cube_l/2),
                            -self.cube_l/2,
                            random.uniform(math.pi/4, 3*math.pi/4)],
                           [self.cube_l/2,
                            random.uniform(-self.cube_l/2, self.cube_l/2),
                            random.uniform(3*math.pi/4, 5*math.pi/4)],
                           [random.uniform(-self.cube_l/2, self.cube_l/2),
                            self.cube_l/2,
                            random.uniform(5*math.pi/4, 7*math.pi/4)],
                           [-self.cube_l/2,
                            random.uniform(-self.cube_l/2, self.cube_l/2),
                            random.uniform(7*math.pi/4, 9*math.pi/4)]]
            side = random.choice([0, 1, 2, 3])
            poke_chosen = poke_option[side]
            poke_temp.point.x = poke_chosen[0]
            poke_temp.point.y = poke_chosen[1]
            theta = poke_chosen[2]

        elif req.name == 'cuboid':
            poke_option = [[random.uniform(-self.cuboid_l/2, self.cuboid_l/2),
                            -self.cuboid_w/2,
                            random.uniform(math.pi/12, 11*math.pi/12)],
                           [self.cuboid_l/2,
                            random.uniform(-self.cuboid_w/2, self.cuboid_w/2),
                            random.uniform(7*math.pi/12, 17*math.pi/12)],
                           [random.uniform(-self.cuboid_l/2, self.cuboid_l/2),
                            self.cuboid_w/2,
                            random.uniform(13*math.pi/12, 23*math.pi/12)],
                           [-self.cuboid_l/2,
                            random.uniform(-self.cuboid_w/2, self.cuboid_w/2),
                            random.uniform(19*math.pi/12, 29*math.pi/12)]]
            side = random.choice([0, 1, 2, 3])
            poke_chosen = poke_option[side]
            poke_temp.point.x = poke_chosen[0]
            poke_temp.point.y = poke_chosen[1]
            theta = poke_chosen[2]

        elif req.name == 'cylinder':
            theta_temp = random.uniform(0, 2*math.pi)
            poke_temp.point.x = self.cylin_r*math.cos(theta_temp)
            poke_temp.point.y = self.cylin_r*math.sin(theta_temp)
            theta = random.uniform(
                theta_temp+7*math.pi/12, theta_temp+17*math.pi/12)

        else:
            print('no such model %s'%req.name)
            raise rospy.ServiceException


        # update the chosen random point in block frame as one class attribute.
        # thus it is modified each time this service is called and transformed
        # into world frame.
        self.ps = poke_temp
        # only lookup for the latest transform.
        poke_point = self.transform_to_world(poke_temp, 'world')
        pointOut = poke_point.pointOut

        resp = self._get_model_srv('block', 'world')
        q = resp.pose.orientation
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(
            (q.x, q.y, q.z, q.w))

        theta = theta + yaw
        if theta > 2*math.pi or theta < 0:
            m = theta // (2*math.pi)
            theta = theta - m*2*math.pi

        l = random.uniform(0.04, 0.08)
        #l = 0.078

        return {'pointOut': pointOut, 'theta': theta, 'l': l}

if __name__ == '__main__':
    rospy.init_node('model_action_generation')
    action_generation = poke_action_generation()
    rospy.spin()
    """
    rospy.sleep(3.0)
    pub = rospy.Publisher('/my_gripper/poke_point', PointStamped, queue_size=10)
    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():
        action_generation.ps.header.stamp = rospy.Time.now()
        point_world = action_generation.transform_to_world(
            action_generation.ps,
            'world')
        pub.publish(point_world.pointOut)
        rate.sleep()
    """
