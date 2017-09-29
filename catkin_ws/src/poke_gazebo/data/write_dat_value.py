#!/usr/bin/env python
"""
Post processing on poking data. Deleting delayed image at the beginning and empth depth image at the end. Put action values read from dat file into a total dat prepared for the network training.
Mind that the generated projected pixels are within the matrix index style, meaning rows are x and columns are y starting from the top left corner.
"""
import glob
import numpy as np

def round_by_half(s):
    value = float(s)
    rounded = round(value*2.0)/2.0
    s_rounded = '%.3e'%rounded
    return s_rounded

def main():
    train_dir = sorted(glob.glob('run*'))
    train_dir_total = [] #for filling up the training data.

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        train_dir_aug = []

        with open(i_dir+'/actions_clean.dat','r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        for i, item in enumerate(actions):
            dx = np.floor(float(item[3])/12)
            dy = np.floor(float(item[4])/12)
            dxy = 20*dx+dy

            dtheta = np.floor(float(item[5])/(np.pi/18))
            dl = np.floor((float(item[6])-0.01)/0.004)

            # add depth directory to it when neccesary.
            train_dir_aug.append(rgb_dir[i]+' '+rgb_dir[i+1]+' '
                                 +round_by_half(item[3])+' '+round_by_half(item[4])
                                 +' '+item[5]+' '+item[6]+' '
                                 +'%d'%dxy+' '+'%d'%dtheta+' '+'%d'%dl)

        train_dir_total.extend(train_dir_aug)

    with open('train_new.txt', 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n' %item)

if __name__ == '__main__':
    main()
