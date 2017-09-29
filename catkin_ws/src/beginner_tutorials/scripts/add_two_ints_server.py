#!/usr/bin/env python
# srv is defined beforehand and included in CMakeLists.txt
# typeRespnse(something).
# [srv/AddTwoInts.srv]:
# int64 a
# int64 b
# ---
# int64 sum

from beginner_tutorials.srv import *
import rospy

def handle_add_two_ints(req):
    print "Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b))
    return AddTwoIntsResponse(req.a+req.b)

def add_two_ints_server():
    rospy.init_node('add_two_ints_server') # Declare node.
    # Declare service add_two_ints with type AddTwoInts.
    # All requests are passed to handle_add_two_ints function.
    # handle_add_two_ints is called with instance of AddTwoIntsRequest
    # and response with instance of AddTwIntsResponse.
    s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    # Mind the service name 'add_two_ints' here in client script.
    print "Ready to add two int."
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()
