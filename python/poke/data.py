# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:06:36 2017

Analyze action value array, find the minimum and maximum, discretize the action,
store discrete actions in array and then evaluate the discrete actions.
@author: wuyang
"""

import numpy as np
import glob
import os
import math

train_dir = sorted(glob.glob('train/*'))
cnt_invalid = 0

for i_0, i_dir in enumerate(train_dir):
    num_dir = len([name for name in os.listdir(i_dir+'/')])
    actions = np.load(i_dir+'/actions.npy')
    print(num_dir-1, len(actions))
    valid_idx = (actions[:,4] == 1)
    if sum(actions[:,4]==0) > 0:
        cnt_invalid += sum(actions[:,4]==0)
        print('\n run',i_0, 'has invalid poke data.\n')
    actions_valid = actions[valid_idx]
    t_max_a = np.amax(actions_valid, axis=0)
    #print('maximum location %d'%np.argmax(actions_valid[:,0], axis=0))
    t_min_a = np.amin(actions_valid, axis = 0)
    if i_0 == 0:
        max_a = t_max_a
        min_a = t_min_a
    idx_max = max_a < t_max_a
    max_a[idx_max] = t_max_a[idx_max]
    idx_min = min_a > t_min_a
    min_a[idx_min] = t_min_a[idx_min]
print(max_a)
print(min_a)
print('number of invalid poke: ',cnt_invalid)

#%% discretize the action data.
for i, i_dir in enumerate(train_dir):
    actions = np.load(i_dir + '/actions.npy')
    position_x = np.floor(actions[:,[0]]/12)
    i_exceed = position_x > 19
    position_x[i_exceed] = 19
    position_y = np.floor(actions[:,[1]]/12)
    position = position_x + 20 * position_y
    angle = np.floor(actions[:,[2]]/(np.pi/18))
    length = np.floor((actions[:,[3]]-0.01)/0.004)
    discretized = np.concatenate((position, angle, length),axis=1)
    #np.save(i_dir+'/actions_discrete',discretized)

#%% Evaluate processed action data
for i, i_dir in enumerate(train_dir):
    num_dir = len([name for name in os.listdir(i_dir+'/')])
    actions = np.load(i_dir+'/actions.npy')
    actions_discrete = np.load(i_dir+'/actions_discrete.npy')
    print(num_dir-2, len(actions), len(actions_discrete))
    position = actions_discrete[:,0]
    i_max_position = position.argmax(axis=0)
    print('max position: c->%s d->%s'%(actions[i_max_position,0:2], position[i_max_position]))
    print('max angle: c->%s d->%s'%(max(actions[:,2]), max(actions_discrete[:,1])))
    print('max length: c->%s d->%s\n'%(max(actions[:,3]), max(actions_discrete[:,2])))
    i_min_position = position.argmin(axis=0)
    print('min position: c->%s d->%s'%(actions[i_min_position,0:2], position[i_min_position]))
    print('min angle: c->%s d->%s'%(min(actions[:,2]), min(actions_discrete[:,1])))
    print('min length: c->%s d->%s\n'%(min(actions[:,3]), min(actions_discrete[:,2])))
