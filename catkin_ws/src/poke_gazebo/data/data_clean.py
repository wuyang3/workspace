#!/usr/bin/env python
import numpy as np
import glob
import os
import math

dir = sorted(glob.glob('run_*'))

# remember to update dir in which data needs to be processed.
dir = []

for i, i_dir in enumerate(dir):
    num_dir = len(glob.glob(i_dir+'/'))
    depth_dir = sorted(glob.glob(i_dir+'/*.png'))
    rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

    depth_num = [int(j[-8:-4]) for j in depth_dir]
    rgb_num = [int(j[-8:-4]) for j in rgb_dir]

    print("last depth image %d"%max(depth_num))
    print("first of rgb image %d\n"%min(rgb_num))

    os.remove(i_dir+'/depth%04d.png'%max(depth_num))
    os.remove(i_dir+'/img%04d.jpg'%min(rgb_num))

    rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))
    for k, filename in enumerate(rgb_dir):
        file_number = int(filename[-8:-4])
        if k == file_number-1:
            new_name = i_dir+'/img%04d.jpg'%(file_number-1)
            os.rename(filename, new_name)

    with open(i_dir+'/actions.dat', 'rb') as fhandle:
        dat = fhandle.readlines()
    dat_clean = dat[:-1]
    with open(i_dir+'/actions_clean.dat', 'wb') as f:
        f.writelines(dat_clean)
