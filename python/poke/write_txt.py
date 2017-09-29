# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:30:46 2017
Thie script reads directories of images and action arraies and write the directories
in a text file which is prepared for the tensorflow queue pipeline.

@author: ibm
"""

import glob

train_dir = sorted(glob.glob('train/*'))

train_dir_total = []
for i, i_dir in enumerate(train_dir):
    img_dir = sorted(glob.glob(i_dir+'/*.jpg'))
    img_dir_aug = []
    for j in range(len(img_dir)-1):
        img_dir_aug.append(img_dir[j]+' '+img_dir[j+1]+' '+i_dir+' '+str(j))
    train_dir_total.extend(img_dir_aug)
    
train_dir_file = open('train_dir.txt','w')

for item in train_dir_total:
    train_dir_file.write('%s\n' %item)