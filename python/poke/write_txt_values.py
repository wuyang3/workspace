# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29.
Thie script reads directories of images and action arrays. Invalid poke action
with x pixels larger than 240 are ignored.

Image directories and  action values are wrritten in a text file.

@author: wuyang
"""

import glob
import numpy as np

def format_f(value):
    return '%.16f' %value
def format_d(value):
    return str(value)

train_dir = sorted(glob.glob('train/*'))

train_dir_total = []
for i, i_dir in enumerate(train_dir):
    # Operate in each episode directory.
    img_dir = sorted(glob.glob(i_dir+'/*.jpg'))
    img_dir_aug = []
    actions = np.load(i_dir+'/actions.npy')

    for j in range(len(img_dir)-1):
        # Operate in the episode directory.
        action = actions[j]
        if np.floor(action[0]/12) > 19:
            continue

        a0 = format_d(action[0])
        a1 = format_d(action[1])
        a2 = format_f(action[2])
        a3 = format_f(action[3])

        adx = np.floor(action[0]/12)
        ady = np.floor(action[1]/12)
        ad0 = format_d(adx+20*ady)

        adtheta = np.floor(action[2]/(np.pi/18))
        ad1 = format_d(adtheta)

        adl = np.floor((action[3]-0.01)/0.004)
        ad2 = format_d(adl)

        img_dir_aug.append(img_dir[j]+' '+img_dir[j+1]+' '
                           +a0+' '+a1+' '+a2+' '+a3+' '+ad0+' '+ad1+' '+ad2)

    train_dir_total.extend(img_dir_aug)

# with keyword ensures the file is closed after the suite.
with open('train_new.txt', 'w') as train_file:
    for item in train_dir_total:
        train_file.write('%s\n' %item)
