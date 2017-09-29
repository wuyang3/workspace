# pointCloudCutoff matters since distance lower than this will be
# discarded. As for the depth image, /my_gripper/camera1/depth/image_raw,
# it is there. You need to convert it properly to keep informations.
# somehow the cutoff range also influence depth image.
# in cmap visualization, low value of pixel corresponds to black. In converted cv image(passthrough), value at around 0.4-min/1.47-min are relatively small since they are automatically normalized. That's why you see table region are black. Moreover, if will be totally black if you normalize it with 0 and 255. A good way to go is to normalize by 0 to max(value to be shown as white).
# 32FC1: 32 bits floating point one channel. After transform, <f4 in numpy.ndarray
# note that plt.imshow will automatically normalize the image when change the color(how gray).s
import rospy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import PointCloud2

rospy.init_node("trial")
bridge = CvBridge()

depthmsg=rospy.wait_for_message('/my_gripper/camera1/depth/image_raw', Image, 0.25)
rgbmsg = rospy.wait_for_message('/my_gripper/camera1/rgb/image_raw', Image, 0.25)
rgbcompressedmsg = rospy.wait_for_message('/my_gripper/camera1/rgb/image_raw/compressed', CompressedImage, 0.25)
points = rospy.wait_for_message('/my_gripper/camera1/depth/points', PointCloud2, 0.25)

depthcv = bridge.imgmsg_to_cv2(depthmsg, "16UC1")
depthcv = bridge.imgmsg_to_cv2(depthmsg, "32FC1")
depthcv = bridge.imgmsg_to_cv2(depthmsg, "passthrough")

plt.imshow(depthcv)
plt.show()

depthcv16 = depthcv*65535.0/1.50
depthcvU16 =np.array(depthcv16, dtype=np.uint16) # floor on float
cv2.imwrite('depthcvU16.png',depthcvU16)

depthcv8 = depthcv*256.0/1.50
depthcvU8 = np.array(depthcv8, dtype=np.uint8)
cv2.imwrite('depthcvU8.png', depthcvU8)

# read to check the precision
depthcvU16read = cv2.imread('depthcvU16.png',cv2.IMREAD_ANYDEPTH)
# the following should be the same
plt.imshow(depthcvU16read,cmap='gray',vmin=0, vmax=65535)
plt.show() # as the png

depthcv16.max()
depthcvU16.max()
depthcvU16read.max()

# 1.4780619
# min max range not considering depth>10 is approximately 0.373632 to 
# 0.511621. Let's normalize it to 0.37 to 0.52. 0.3243(closest depth).
# let's normalize it from 0.324 to 0.512.
depthcv16.max()/65535.0*1.50
depthcvU16.max()/65535.0*1.50
depthcvU16read.max()/65535.0*1.50

import tensorflow as tf
img = tf.read_file('depthcvU16.png')
png = tf.image.decode_png(img,channels=0, dtype=tf.uint16)
png = tf.cast(png, tf.float32)
png = tf.image.resize_images(png, [240,240])
png = tf.squeeze(png) # whether or not to keep the extra dimension.
d = tf.constant([1.50/65535.0],dtype=tf.float32)
png_normalized = tf.multiply(png, d)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    png_normalized_array, pngarray = sess.run([png_normalized, png])

