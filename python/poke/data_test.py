#!/usr/bin/env python
import numpy as np
import glob
import os
import math

dir = sorted(glob.glob('temp/run_*'))
poke_num = 0

for i, i_dir in enumerate(dir):
    num_dir = len([name for name in os.listdir(i_dir+'/')])
    depth_dir = sorted(glob.glob(i_dir+'/*.png'))
    rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

    with open(i_dir+'/actions.dat','r') as fhandle:
        dat = fhandle.readlines()

    poke_num += len(dat)

    if (len(depth_dir)!=len(rgb_dir) or
        len(dat) != len(depth_dir)-1):
        #(num_dir-1)/2 != len(depth_dir)):

        depth_num = [int(i[-8:-4]) for i in depth_dir]
        depth_missing = []
        for j, item in enumerate(depth_num[:-1]):
            diff = depth_num[j+1] - depth_num[j]
            if diff > 1:
                if diff == 2:
                    depth_missing.append([item+1])
                else:
                    depth_missing_temp = []
                    for k in range(diff-1):
                        depth_missing_temp.append(item+k+1)
                    depth_missing.append(depth_missing_temp)
            else:
                pass
        rgb_num = [int(i[-8:-4]) for i in rgb_dir]
        rgb_missing = []
        for j, item in enumerate(rgb_num[:-1]):
            diff = rgb_num[j+1] - rgb_num[j]
            if diff > 1:
                if diff == 2:
                    rgb_missing.append([item+1])
                else:
                    rgb_missing_temp = []
                    for k in range(diff-1):
                        rgb_missing_temp.append(item+k+1)
                    rgb_missing.append(rgb_missing_temp)
            else:
                pass

        print i_dir
        print ("total: %d \nrgb: %d depth: %d\n"
               %(num_dir, len(depth_dir), len(rgb_dir))+
               "action lines: %d"
               %(len(dat)))
        if rgb_missing != depth_missing:
            print("missing rgb and depth inequal")

        rgb_missing_string = []
        for item in rgb_missing:
            if len(item)==1:
                rgb_missing_string.append("img%04d.jpg"%(item[0]))
            else:
                rgb_missing_string.append("img%04d-%04d.jpg"%(item[0], item[-1]))
        rgb_m = ' '.join(rgb_missing_string)

        depth_missing_string = []
        for item in depth_missing:
            if len(item)==1:
                depth_missing_string.append("depth%04d.png"%(item[0]))
            else:
                depth_missing_string.append("depth%04d-%04d.png"%(item[0], item[-1]))
        depth_m = ' '.join(depth_missing_string)

        print(depth_m+'\n'+rgb_m+'\n')

    #else:
        #print("%s: %s %s actions %d"
              #%(i_dir, rgb_dir[-1], depth_dir[-1], len(dat)))
print 'number of poke %d'%poke_num
