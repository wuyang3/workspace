#!/usr/bin/env python
"""
Preprocessing on the image data, processing steps are listed below.
1. Find missing rgb and depth image. The current version neglects a sequence of
missing image and assume that there aren't any sequence of images missing,
e.g. two in consecutive. This can be a problem when a sequence of images
are deleted manually.
2. Deleting the first rgb and the last depth image.
3. shift all rgb image number by one since the original data are delayed one step.
4. Deleting the last line of actions.dat and save in actions_clean.dat.
5. Additionally, for every missing image number k, deleting k-1 th depth image and k th rgb images.
6. Deleting k-2, k-1 and k th actions.

The above processings are not enough if I manually delete a series of wrong images
e.g. cylinders rolling back and forth. Alternative: keep each block of missing
images in a sublist and parsing based every missing sublist.
Aug 20: clean directories from 115 to 130.
"""
import numpy as np
import glob
import os
import math

dir = sorted(glob.glob('train_cube/run*'))
#dir_o = sorted(glob.glob('train_gazebo/run_1*'))
#dir_num = [i for i, item in enumerate(dir_o) if 114<int(item[17:])<131]
#dir = [dir_o[i] for i in dir_num]


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
    #depth_missing = [item+1 for j, item in enumerate(depth_num[:-1])
    #                 if depth_num[j+1] - depth_num[j] > 1]

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
    print("first of rgb image %d"%min(rgb_num))
    os.remove(i_dir+'/depth%04d.png'%max(depth_num))
    os.remove(i_dir+'/img%04d.jpg'%min(rgb_num))

    rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))
    for filename in rgb_dir:
        file_number = int(filename[-8:-4])
        new_name = i_dir+'/img%04d.jpg'%(file_number-1)
        os.rename(filename, new_name)

    with open(i_dir+'/actions.dat', 'rb') as fhandle:
        dat = fhandle.readlines()
    dat_clean = dat[:-1]
    with open(i_dir+'/actions_clean.dat', 'wb') as f:
        f.writelines(dat_clean)

    if (depth_missing == rgb_missing and depth_missing == []):
        print(i_dir + ' shifted\n')

    # processing for missing rgb and depth image. # else for normal folder!
    if (depth_missing == rgb_missing and depth_missing != []):
        for j, item in enumerate(depth_missing):
            if len(item) == 1:
                os.remove(i_dir + '/depth%04d.png'%(item[0]-1))
                os.remove(i_dir + '/img%04d.jpg'%item[0])
            elif len(item) > 1:
                os.remove(i_dir + '/depth%04d.png'%(item[0]-1))
                os.remove(i_dir + '/img%04d.jpg'%item[-1])
            else:
                pass

        actions_new = []
        with open(i_dir + '/actions_clean.dat', 'rb') as fhandle:
            actions_c = fhandle.readlines()
        # until index depth_missing[0][0]-2, number depth_missing[0][0]-3
        # depth0011 has an index of 11.
        # a[5:9] means index 5 to index 8 and index starts from 0.
        actions_new.extend(actions_c[:depth_missing[0][0]-2])
        for j, item in enumerate(depth_missing):
            if j == len(depth_missing) - 1:
                actions_new.extend(actions_c[item[-1]+1:])
            else:
                actions_new.extend(actions_c[item[-1]+1:depth_missing[j+1][0]-2])

        with open(i_dir + '/actions_clean.dat', 'wb') as fhandle:
            fhandle.seek(0)
            fhandle.writelines(actions_new)
            fhandle.truncate()
        print(i_dir + ' cleaned\n')
