#!/usr/bin/env python
"""
Provide image saving service. The callback is function wait_for_message. The message is transferred to cv and then saved.
imageSave:
request has two attributes, path and number (sequence number)
response has a single bool type of whether image saving success.
When passing request to the service proxy, one can just pass in each request variable of request. The return will be imageSaveResponse. The result is in imageSaveResponse.success  !
Saving numpy array costs to many spaces. Somehow the stored vale doesnot seem to be
correct. Saving the array as 16 bit png with properly chosen range might be a good option. Irrelavent depth point around the table are ruled out. Mind that the depth image array from cvbridge is not writable.

#cv2_img = self.bridge.imgmsg_to_cv2(image_msg, "passthrough")
#n_channels = 3
#dtype = np.dtype('uint8')
#dtype.newbyteorder('>' if image_msg.is_bigendian else '<')
#img = np.ndarray(
#    shape=(image_msg.height, image_msg.width, n_channels),
#    dtype=dtype,
#    buffer=image_msg.data)

If no baseline or distortion coefficient is set, the image does not need to be rectified or undistorted.
"""
import rospy
import numpy as np

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from poke_gazebo.srv import imageSave

from cv_bridge import CvBridge, CvBridgeError
import cv2

class poke_image_saver(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.s = rospy.Service('/model_image_saver/image_save',
                               imageSave,
                               self.handle_image_save)

        print("image saving service ready...")
        """
        cam_info = rospy.wait_for_message('/my_gripper/camera1/rgb/camera_info',
                                          CameraInfo)
        self.K = np.array(cam_info.K)
        self.D = np.array(cam_info.D)
        """

    def handle_image_save(self, req):
        try:
            image_msg = rospy.wait_for_message(
                '/my_gripper/camera1/rgb/image_raw/compressed',
                CompressedImage,
                0.1)
            depth_msg = rospy.wait_for_message(
                '/my_gripper/camera1/depth/image_raw',
                Image,
                0.1)
            image_msg_2 = rospy.wait_for_message(
                '/my_gripper/camera1/rgb/image_raw/compressed',
                CompressedImage,
                0.1)

            cv2_img = self.bridge.compressed_imgmsg_to_cv2(image_msg_2, "bgr8")
            cv2_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        except (rospy.ROSException, CvBridgeError) as e:
            print(e)
            success = False
        # if no exceptions, else clause is executed.
        else:
            #img_rectified = cv2.undistort(cv2_img, self.K.reshape(3,3), self.D)
            cv2.imwrite(req.path+'img'+'%04d'%req.number+'.jpg', cv2_img)

            # bigger objects and surrounded table: 0.315 - 0.532
            depth_temp = cv2_depth.copy()
            depth_temp[depth_temp>0.9] = 0.532
            depth_temp = (depth_temp - 0.315)/(0.532-0.315)*65535
            depth_temp_int = np.array(depth_temp, np.uint16)
            cv2.imwrite(
                req.path+'depth'+'%04d'%req.number+'.png', depth_temp_int)
            success = True
        # finally clause is executed regardless of exceptions.
        finally:
            return {'success':success}

if __name__ == '__main__':
    rospy.init_node('model_image_saver')
    image_saver = poke_image_saver()
    rospy.spin()
