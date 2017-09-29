#!/usr/bin/env python
# About resp1.sum, mind that a srv that describe the service is defined
# before hand. The service has fields of a, b (requests) and sum (response).

import sys
import rospy
from beginner_tutorials.srv import *

def add_two_ints_client(x, y):
    rospy.wait_for_service('add_two_ints') # Service name.
    try:
        add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)
        # Use this handle and call the function by service.
        resp1 = add_two_ints(x, y)
        # AddTwoIntsRequest is automaticly generated and passed into.
        # AddTwoIntResponse object is returned. See server.
        return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
    else:
        print usage()
        sys.exit(1)
    print "Requesting %s+%s"%(x, y)
    print "%s + %s = %s"%(x, y, add_two_ints_client(x, y))
