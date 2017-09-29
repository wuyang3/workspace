#!/usr/bin/env python
"""
Preprocessing on the image data.
Second time collection of single object data are not shifted. RGB and depth are aligned. So the cleanup routine is a bit different.
"""
import numpy as np
import glob
import os
import math

dir = sorted(glob.glob('temp/run*'))

for i, i_dir in enumerate(dir):
    num_dir = len(glob.glob(i_dir+'/'))
    depth_dir = sorted(glob.glob(i_dir + '/*.png'))
    rgb_dir = sorted(glob.glob(i_dir + '/*.jpg'))

    depth_num = [int(j[-8:-4]) for j in depth_dir]
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

    rgb_num = [int(j[-8:-4]) for j in rgb_dir]
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

    print('missing depth:', depth_m)
    print('missing rgb', rgb_m)
    print("last depth image %d"%max(depth_num))
    print("last of rgb image %d"%max(rgb_num))

    with open(i_dir+'/actions.dat', 'rb') as fhandle:
        dat_clean = fhandle.readlines()
    os.remove(i_dir+'/actions_clean.dat')
    with open(i_dir+'/actions_clean.dat', 'wb') as f:
        f.writelines(dat_clean)

    if (depth_missing == rgb_missing and depth_missing == []):
        print(len(depth_missing)-1==len(dat_clean))
        print(i_dir + ' done\n')

    # processing for missing rgb and depth image. # else for normal folder!
    if (depth_missing == rgb_missing and depth_missing != []):
        actions_new = []
        with open(i_dir + '/actions_clean.dat', 'rb') as fhandle:
            actions_c = fhandle.readlines()
        actions_new.extend(actions_c[:depth_missing[0][0]-1])
        for j, item in enumerate(depth_missing):
            if j == len(depth_missing) - 1:
                actions_new.extend(actions_c[item[-1]+1:])
            else:
                actions_new.extend(actions_c[item[-1]+1:depth_missing[j+1][0]-1])

        with open(i_dir + '/actions_clean.dat', 'wb') as fhandle:
            fhandle.seek(0)
            fhandle.writelines(actions_new)
            fhandle.truncate()
        print(i_dir + ' cleaned\n')
