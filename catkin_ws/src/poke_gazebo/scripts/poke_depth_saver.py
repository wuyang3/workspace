#!/usr/bin/env python
"""
Provide image saving service. The callback is function wait_for_message. The message is transferred to cv and then saved.
imageSave:
request has two attributes, path and number (sequence number)
response has a single bool type of whether image saving success.
When passing request to the service proxy, one can just pass in each request variable of request. The return will be imageSaveResponse. The result is in imageSaveResponse.success  !
Saving numpy array costs to many spaces. Somehow the stored vale doesnot seem to be
correct. Saving the array as 16 bit png with properly chosen range might be a good option. Irrelavent depth point around the table are ruled out. Mind that the depth image array from cvbridge is not writable.
"""
import rospy
import numpy as np

from sensor_msgs.msg import Image
from poke_gazebo.srv import imageSave

from cv_bridge import CvBridge, CvBridgeError
import cv2

class poke_depth_saver(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.s = rospy.Service('/model_depth_saver/depth_save',
                               imageSave,
                               self.handle_depth_save)
        print "depth saving service ready..."

    def handle_depth_save(self, req):
        try:
            depth_msg = rospy.wait_for_message(
                '/my_gripper/camera1/depth/image_raw',
                Image,
                0.075)
            cv2_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        except (rospy.ROSException, CvBridgeError), e:
            print e
            success = False

        else:
            #np.save(req.path+'depth'+'%04d'%req.number+'.npy', cv2_depth)
            # normalization range 1.
            #cv2_depth_U16 = np.array(cv2_depth*65535.0/1.50, dtype=np.uint16)
            #cv2.imwrite(req.path+'depth'+'%04d'%req.number+'.png', cv2_depth_U16)
            # normalization range 2.
            depth_temp = cv2_depth.copy()
            depth_temp[depth_temp>0.8] = 0.512
            #depth_temp[depth_temp<0.324] = 0.324
            depth_temp = (depth_temp - 0.324)/(0.512-0.324)*65535
            depth_temp_int = np.array(depth_temp, np.uint16)
            cv2.imwrite(
                req.path+'depth'+'%04d'%req.number+'.png', depth_temp_int)
            success = True
        finally:
            return {'success':success}

if __name__ == '__main__':
    rospy.init_node('model_depth_saver')
    depth_saver = poke_depth_saver()
    rospy.spin()
