#!/usr/bin/env python
"""
Post processing on poking data. Put image paths and action values read from dat file into a total txt prepared for the network training.
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
    train_dir_total = [] #for filling up the training data.

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        train_dir_aug = []

        with open(i_dir+'/actions_clean.dat','r') as f:
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

            while (int(rgb_dir[k+1][-8:-4]) - int(rgb_dir[k][-8:-4])) != 1:
                k += 1
            # add depth directory to it when neccesary.
            train_dir_aug.append(rgb_dir[k]+' '+rgb_dir[k+1]+' '
                                 +round_by_half(item[3])+' '+round_by_half(item[4])
                                 +' '+item[5]+' '+item[6]+' '
                                 +'%d'%dxy+' '+'%d'%dtheta+' '+'%d'%dl)
            k += 1

        print(i_dir+' :image until %s included'%rgb_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open('train_gazebo.txt', 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n' %item)

def cube_table():
    train_dir = sorted(glob.glob('train_cube_table/run*'))
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        rgb_num = [int(j[-8:-4]) for j in rgb_dir]
        train_dir_aug = []

        with open(i_dir+'/actions.dat','r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        k = 0
        for j, item in enumerate(rgb_num[:-1]):
            if (rgb_num[j+1]-item==1):
                action = actions[item]
                dx_i = bound_to_float(action[2])
                dy_i = bound_to_float(action[3])
                dx = np.floor(dx_i/12)
                dy = np.floor(dy_i/12)
                dxy = 20*dx+dy

                dtheta = np.floor(float(action[4])/(np.pi/18))
                dl = np.floor((float(action[5])-0.04)/0.004)

                train_dir_aug.append(rgb_dir[j]+' '+rgb_dir[j+1]+' '
                                     +round_by_half(action[2])+' '+round_by_half(action[3])
                                     +' '+action[4]+' '+action[5]+' '
                                     +'%d'%dxy+' '+'%d'%dtheta+' '+'%d'%dl)
                k = j+1
            else:
                pass

        print(i_dir+' :image until %s included'%rgb_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open('train_cube_table_inv.txt', 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n' %item)

def cube_table_depth():
    train_dir = sorted(glob.glob('train_cube_table/run*'))
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))

        depth_num = [int(j[-8:-4]) for j in depth_dir]
        train_dir_aug = []

        with open(i_dir+'/actions.dat','r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        k = 0
        for j, item in enumerate(depth_num[:-1]):
            if (depth_num[j+1]-item==1):
                action = actions[item]
                dx_i = bound_to_float(action[2])
                dy_i = bound_to_float(action[3])
                dx = np.floor(dx_i/12)
                dy = np.floor(dy_i/12)
                dxy = 20*dx+dy

                dtheta = np.floor(float(action[4])/(np.pi/18))
                dl = np.floor((float(action[5])-0.04)/0.004)

                train_dir_aug.append(depth_dir[j]+' '+depth_dir[j+1]+' '
                                     +round_by_half(action[2])+' '+round_by_half(action[3])
                                     +' '+action[4]+' '+action[5]+' '
                                     +'%d'%dxy+' '+'%d'%dtheta+' '+'%d'%dl)
                k = j+1
            else:
                pass

        print(i_dir+' :image until %s included'%depth_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open('train_cube_table_inv_d.txt', 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n' %item)

def cube_table_t():
    train_dir = sorted(glob.glob('train_cube_table/run*'))
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))

        rgb_num = [int(j[-8:-4]) for j in rgb_dir]
        depth_num = [int(j[-8:-4]) for j in depth_dir]
        assert rgb_num == depth_num, (
            'rgb and depth not matching at run_%02d'%i
        )
        train_dir_aug = []

        with open(i_dir+'/actions.dat','r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        k = 0
        for j, item in enumerate(depth_num[:-1]):
            if (depth_num[j+1]-item==1):
                action = actions[item]
                dx_i = bound_to_float(action[2])
                dy_i = bound_to_float(action[3])
                dx = np.floor(dx_i/12)
                dy = np.floor(dy_i/12)
                dxy = 20*dx+dy

                dtheta = np.floor(float(action[4])/(np.pi/18))
                dl = np.floor((float(action[5])-0.04)/0.004)

                train_dir_aug.append(' '.join(
                    [rgb_dir[j], rgb_dir[j+1], depth_dir[j], depth_dir[j+1],
                     round_by_half(action[2]), round_by_half(action[3]),
                     action[4], action[5], '%d'%dxy, '%d'%dtheta, '%d'%dl]))

                k = j+1
            else:
                pass

        print(i_dir+' :image until %s included'%depth_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open('train_cube_table_inv_t.txt', 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n' %item)

if __name__ == '__main__':
    #main()
    #cube_table()
    #cube_table_depth()
    cube_table_t()
