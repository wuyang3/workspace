#!/usr/bin/env python
import numpy as np
import glob
import os
import math

dir = sorted(glob.glob('run_*'))
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
        depth_missing = [item for i, item in enumerate(depth_num[:-1])
                         if depth_num[i+1]-depth_num[i]>1]
        rgb_num = [int(i[-8:-4]) for i in rgb_dir]
        rgb_missing = [item for i, item in enumerate(rgb_num[:-1])
                        if rgb_num[i+1]-rgb_num[i]>1]

        print i_dir
        print ("total: %d \nrgb: %d depth: %d\n"
               %(num_dir, len(depth_dir), len(rgb_dir))+
               "action lines: %d"
               %(len(dat)))

        rgb_missing_string = ["img%04d.jpg"%(i+1) for i in rgb_missing]
        rgb_m = ' '.join(rgb_missing_string)

        depth_missing_string = ["depth%04d.png"%(i+1) for i in depth_missing]
        depth_m = ' '.join(depth_missing_string)

        print(depth_m+'\n'+rgb_m+'\n')

    #else:
        #print("%s: %s %s actions %d"
              #%(i_dir, rgb_dir[-1], depth_dir[-1], len(dat)))
print 'number of poke %d'%poke_num
