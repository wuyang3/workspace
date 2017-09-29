#!/usr/bin/env python
"""
From cleaned image and depth image, adding object class to each line of actions_clean_classes.
This requires adatped write_dat_value.py
"""

import glob

dir = sorted(glob.glob('train_gazebo/run*'))
with open('object_type.txt', 'rb') as f:
    object_type = [line.strip().split() for line in f.readlines()]

dir_num = [item[0] for item in object_type]
class_num = [item[1] for item in object_type]

for i, i_dir in enumerate(dir):
    with open(i_dir + '/actions_clean.dat', 'rb') as f:
        actions_clean = f.readlines()

    dir_num_i = i_dir[13:]
    idx = dir_num.index(dir_num_i)
    class_i = class_num[idx]
    print('%s: object type %s'%(dir_num_i, class_i))

    actions_clean_classes = [line.strip()+' '+class_i+'\n' for line in actions_clean]
    with open(i_dir + '/actions_clean_classes.dat', 'wb') as f:
        f.writelines(actions_clean_classes)
