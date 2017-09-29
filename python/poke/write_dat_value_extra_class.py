#!/usr/bin/env python
"""
Post processing on poking data for training. Note that this one also saves object class information.
Pipeline:
data_test.py to figure out missing images.
data_clean_missing_pics.py cleans out missing images and sync rgb&depth image and save actions_clean.
write_dat_value.py writes out image pairs with corresponding actions.
data_write_extra_classes.py appends object type informaton and write actons_clean_extra_class.dat
write_dat_value_extra_class.py writes out image pairs with corresponding actions with object type.
data_processing_result_test.py tests the correctness of train_gazebo_extra_class.txt.
"""
import glob
import numpy as np

def round_by_half(s):
    value = float(s)
    rounded = round(value*2.0)/2.0
    s_rounded = '%.3e'%rounded
    return s_rounded

def bound_to_float(s):
    value = float(s)
    if value < 0.0:
        value = 0.0
    elif value > 240.0:
        value = 239.9
    else:
        pass
    return value

def main():
    train_dir = sorted(glob.glob('train_gazebo/run*'))
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        train_dir_aug = []

        with open(i_dir+'/actions_clean_classes.dat','r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        k = 0
        for j, item in enumerate(actions):
            dx_i = bound_to_float(item[3])
            dy_i = bound_to_float(item[4])
            # discrete class should be within the range. However, some continuous actions
            # are still below 0 or above 240.
            dx = np.floor(dx_i/12)
            dy = np.floor(dy_i/12)
            dxy = 20*dx+dy

            dtheta = np.floor(float(item[5])/(np.pi/18))
            dl = np.floor((float(item[6])-0.02)/0.004)

            obj_type = item[7]

            while (int(rgb_dir[k+1][-8:-4]) - int(rgb_dir[k][-8:-4])) != 1:
                k += 1
            # add depth directory to it when neccesary.
            train_dir_aug.append(rgb_dir[k]+' '+rgb_dir[k+1]+' '
                                 +round_by_half(item[3])+' '+round_by_half(item[4])
                                 +' '+item[5]+' '+item[6]+' '
                                 +'%d'%dxy+' '+'%d'%dtheta+' '+'%d'%dl
                                 +' '+obj_type)
            k += 1

        print(i_dir+' :image until %s included'%rgb_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open('train_gazebo_extra_class.txt', 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n' %item)

if __name__ == '__main__':
    main()
