#!/usr/bin/env python
"""
Test the correctness of image and action preprocessing.
"""
import numpy as np
import random
import glob
import sys
import math
import matplotlib.pyplot as plt
import scipy.ndimage

def round_by_half(s):
    value = float(s)
    rounded = round(value*2.0)/2.0
    return rounded

def bound_to_float(s):
    value = float(s)
    if value < 0.0:
        value = 0.0
    elif value > 240.0:
        value = 239.9
    else:
        pass
    return value

def rect(ax, poke, c):
    x, y, t, l = poke
    dx = -600 * l * math.cos(t)
    dy = -600 * l * math.sin(t)
    ax.arrow(240-y, x, dy, dx, head_width=5, head_length=5, color=c)

def plot_sample(img_before, img_after, action, action_processed, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_before.copy())
    rect(ax1, action, "green")
    rect(ax1, action_processed, "blue")
    ax2.imshow(img_after.copy())
    #fig.suptitle('%s: object %d %s'%(title[0][-7:], title[1], title[2]))
    fig.suptitle('%s'%(title))

def plot_multisample(imgs, actions_original, actions_read, title):
    num = len(imgs)
    assert num == len(actions_original)+1, (
        'number of images and actions are not matching.'
    )
    fig, axes = plt.subplots(1, num)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        if i < num-1:
            rect(ax, actions_read[i], "blue")
            rect(ax, actions_original[i], "green")
    fig.suptitle('%s'%title)

def ae_test():
    #with open('train_gazebo_extra_class.txt', 'rb') as f:
    with open('train_cube_ae.txt', 'rb') as f:
        data = f.readlines()

    length = len(data)

    for i in range(100):
        idx = random.randint(0, length)
        line = data[idx]
        line_action = line.strip().split()
        #print line_action
        img1_name = line_action[0]
        img2_name = line_action[1]
        x_t = float(line_action[2])*240.0
        y_t = float(line_action[3])*240.0
        theta_t = float(line_action[4])*np.pi*2
        l_t = float(line_action[5])*0.04+0.04
        #type_t = int(line_action[-1])

        num = int(img1_name[-8:-4])
        i_dir = img1_name[:-11]

        with open(i_dir + 'actions.dat', 'rb') as f_handle:
            actions = f_handle.readlines()

        item = actions[num]
        item_action = item.strip().split()
        #print item_action
        #x = round_by_half(item_action[3])
        #y = round_by_half(item_action[4])
        #theta = float(item_action[5])
        #l = float(item_action[6])
        # new collected cube data.
        x = float(item_action[2])
        y = float(item_action[3])
        theta = float(item_action[4])
        l = float(item_action[5])
        img1 = scipy.ndimage.imread(img1_name)
        img2 = scipy.ndimage.imread(img2_name)
        #title = [i_dir, type_t, [x_t, y_t, theta_t, l_t] == [x, y, theta, l]]
        print 'same %d'%(img1_name==img2_name), [x_t, y_t, theta_t, l_t], [x, y, theta, l]
        if img1_name==img2_name:
            title = i_dir
        else:
            title = i_dir + ': ' + '%.2f, %.2f'%(x_t, x)
        plot_sample(img1, img2, [x_t, y_t, theta_t, l_t], [x, y, theta, l], title)
        plt.waitforbuttonpress()
        plt.close()

def ae_rnn_test():
    with open('train_cube_ae_rnn_1.txt', 'rb') as f:
        data = f.readlines()

    length = len(data)
    for i in range(100):
        idx = random.randint(0, length)
        line = data[idx]
        line_action = line.strip().split()
        i1_name = line_action[0]
        i2_name = line_action[1]
        i3_name = line_action[2]
        x1_t = float(line_action[3])*240.0
        y1_t = float(line_action[4])*240.0
        theta1_t = float(line_action[5])*np.pi*2
        l1_t = float(line_action[6])*0.04+0.04

        x2_t = float(line_action[7])*240.0
        y2_t = float(line_action[8])*240.0
        theta2_t = float(line_action[9])*np.pi*2
        l2_t = float(line_action[10])*0.04+0.04

        num1 = int(i1_name[-8:-4])
        num2 = int(i2_name[-8:-4])
        num3 = int(i3_name[-8:-4])
        assert num2 == num1+1, (
            'Not consecutive'
        )
        assert num3 == num2+1, (
            'Not consecutive'
        )
        i_dir = i1_name[:-11]

        with open(i_dir+'actions.dat', 'rb') as f_handle:
            actions = f_handle.readlines()
        item_action1 = actions[num1].strip().split()
        item_action2 = actions[num2].strip().split()

        x1 = float(item_action1[2])
        y1 = float(item_action1[3])
        theta1 = float(item_action1[4])
        l1 = float(item_action1[5])

        x2 = float(item_action2[2])
        y2 = float(item_action2[3])
        theta2 = float(item_action2[4])
        l2 = float(item_action2[5])
        print('action 1', [x1_t, y1_t, theta1_t, l1_t], [x1, y1, theta1, l1])
        print('action 2', [x2_t, y2_t, theta2_t, l2_t], [x2, y2, theta2, l2])

        img1 = scipy.ndimage.imread(i1_name)
        img2 = scipy.ndimage.imread(i2_name)
        img3 = scipy.ndimage.imread(i3_name)

        title = i_dir + ': ' + '%.2f, %.2f, %.2f, %.2f'%(x1, x1_t, x2, x2_t)
        plot_multisample([img1, img2, img3],
                         [[x1, y1, theta1, l1], [x2, y2, theta2, l2]],
                         [[x1_t, y1_t, theta1_t, l1_t], [x2_t, y2_t, theta2_t, l2_t]],
                         title)
        plt.waitforbuttonpress()
        plt.close()

if __name__ == '__main__':
    #sys.exit(ae_test())
    sys.exit(ae_rnn_test())
