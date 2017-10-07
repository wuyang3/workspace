#!/usr/bin/env python
"""
Processing on poking data. Put image paths and action values from dat file into a total txt
for training. This one prepares data for the training of convolutional autoencoder. It is still
unclear whether to feed continuous or discrete actions to the bottleneck of the auto-encoder.
For simplicity, it is started with discretized actions that are normalized.
Aug 20: write txt for training for directories with only one object.

Use separate function for writing data for training and testing is because that testing requires
extra pixel positions for plotting. Additionally, a function write image paths and images are used for t-sne.

main(): write for auto-encoder without rnn
write_rnn_three(_test): write for AE with three recurrent steps.
write_rnn(_test): write for AE for arbitrary number of steps.
write_for_tsne: write image path for generating image embedding.
"""
import glob
import numpy as np
import random

def round_by_half(s):
    value = float(s)
    rounded = round(value*2.0)/2.0
    s_rounded = '%.3e'%rounded
    return s_rounded

def bound_to_float(s):
    value = float(s)
    if value < 0.0:
        value = 0.0
    elif value >= 240.0:
        value = 239.9
    else:
        pass
    return value

def write_for_tsne():
    train_dir = sorted(glob.glob('test_trans/run_*'))
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        train_dir_aug = []
        for item in rgb_dir:
            train_dir_aug.append(item)

        train_dir_total.extend(train_dir_aug)

    with open('embedding_newtrans_generate.txt', 'wb') as f:
        for item in train_dir_total:
            f.write('%s\n'%item)

def main():
    """
    Pick out folder with only one object.
    obj_dir_nums = [10, 11, 14, 15, 23, 26, 29, 31, 40,
                    46, 50, 52, 55, 56, 57, 60, 66, 67,
                    77, 90, 108, 110, 115, 116, 117, 118,
                    119, 120, 125, 126, 127, 128, 129, 130] # directory of single objects.
    train_dir_o = sorted(glob.glob('train_gazebo/run*'))
    train_num = [i for i, item in enumerate(train_dir_o) if int(item[17:]) in obj_dir_nums]
    train_dir = [train_dir_o[i] for i in train_num]
    """
    train_dir = sorted(glob.glob('train_cube/run*'))
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        train_dir_aug = []

        with open(i_dir+'/actions_clean.dat', 'r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        k = 0
        for j, item in enumerate(actions):
            """ for training when actions are discrete on old collected data.
            dx_i = bound_to_float(item[3])
            dy_i = bound_to_float(item[4])
            #dx = dx_i/240.0*2.0 - 1.0
            #dy = dy_i/240.0*2.0 - 1.0
            #dtheta = float(item[5])/(2*np.pi)*2.0 - 1.0
            #dl = (float(item[6]) - 0.02)/0.04*2.0 - 1.0
            dx = np.floor(dx_i/12.0)
            dy = np.floor(dy_i/12.0)
            dtheta = np.ceil(float(item[5])/(np.pi/18))
            dl = np.ceil((float(item[6])-0.02)/0.004)
            """
            """
            # for training when actions are continuous on old collected data.
            # option: normalized used for BN vae; not normalized.
            dx = bound_to_float(item[3])/240.0
            dy = bound_to_float(item[4])/240.0
            dtheta = float(item[5])/(2*np.pi)
            dl = (float(item[6])-0.02)/0.04
            #dx = bound_to_float(item[3])
            #dy = bound_to_float(item[4])
            #dtheta = float(item[5])
            #dl = float(item[6])
            """

            # for training on new data where x, y, theta and l locates differently. Continuous.
            # option: normalized for vae; not normalized.
            #dx = bound_to_float(item[2])
            #dy = bound_to_float(item[3])
            #dtheta = float(item[4])
            #dl = float(item[5])
            dx = bound_to_float(item[2])/240.0
            dy = bound_to_float(item[3])/240.0
            dtheta = float(item[4])/(2*np.pi)
            dl = (float(item[5])-0.04)/0.04

            while (int(rgb_dir[k+1][-8:-4]) - int(rgb_dir[k][-8:-4])) != 1:
                k += 1

            train_dir_aug.append(' '.join([rgb_dir[k], rgb_dir[k+1],
                                           '%.4e'%dx, '%.4e'%dy,
                                           '%.4e'%dtheta, '%.4e'%dl]))
            train_dir_aug.append(' '.join([rgb_dir[k], rgb_dir[k],
                                           '%.4e'%0, '%.4e'%0,
                                           '%.4e'%0, '%.4e'%0]))
            #train_dir_aug.append(' '.join([depth_dir[k], depth_dir[k+1],
            #                               '%.3e'%dx, '%.3e'%dy,
            #                               '%.3e'%dtheta, '%.3e'%dl]))

            k += 1

        print(i_dir+' :image until %s included'%rgb_dir[k])
        #print(i_dir+' :image until %s included'%depth_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open('train_cube_ae.txt', 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n'%item)

def write_rnn_three(CONSECUTIVE = 1):
    path = 'train_cube_table/run*'#'train_cube/run*'
    train_dir = sorted(glob.glob(path))
    #name = 'train_cube_ae_rnn_1.txt'
    name = 'train_cube_table_rnn.txt'
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        # image numbers that are actually there.
        rgb_num = [int(j[-8:-4]) for j in rgb_dir]

        train_dir_aug = []

        with open(i_dir+'/actions.dat', 'r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        if CONSECUTIVE:
            k = 0
            for j, item in enumerate(rgb_num[:-2]):
                if (rgb_num[j+1]-item==1 and rgb_num[j+2]-rgb_num[j+1]==1):
                    action1 = actions[item]
                    dx1 = bound_to_float(action1[2])/240.0
                    dy1 = bound_to_float(action1[3])/240.0
                    dtheta1 = float(action1[4])/(2*np.pi)
                    dl1 = (float(action1[5])-0.04)/0.04

                    action2 = actions[rgb_num[j+1]]
                    dx2 = bound_to_float(action2[2])/240.0
                    dy2 = bound_to_float(action2[3])/240.0
                    dtheta2 = float(action2[4])/(2*np.pi)
                    dl2 = (float(action2[5])-0.04)/0.04

                    train_dir_aug.append(
                        ' '.join([rgb_dir[j], rgb_dir[j+1], rgb_dir[j+2],
                                  '%.4e'%dx1, '%.4e'%dy1, '%.4e'%dtheta1, '%.4e'%dl1,
                                  '%.4e'%dx2, '%.4e'%dy2, '%.4e'%dtheta2, '%.4e'%dl2]))
                    k = j+2
                else:
                    pass

        else:
            k = 0
            j = 0
            while j < len(rgb_num)-2:
                if rgb_num[j+1]-rgb_num[j]==1 and rgb_num[j+2]-rgb_num[j]==2:
                    action1 = actions[rgb_num[j]]
                    dx1 = bound_to_float(action1[2])/240.0
                    dy1 = bound_to_float(action1[3])/240.0
                    dtheta1 = float(action1[4])/(2*np.pi)
                    dl1 = (float(action1[5])-0.04)/0.04

                    action2 = actions[rgb_num[j+1]]
                    dx2 = bound_to_float(action2[2])/240.0
                    dy2 = bound_to_float(action2[3])/240.0
                    dtheta2 = float(action2[4])/(2*np.pi)
                    dl2 = (float(action2[5])-0.04)/0.04

                    train_dir_aug.append(
                        ' '.join([rgb_dir[j], rgb_dir[j+1], rgb_dir[j+2],
                                  '%.4e'%dx1, '%.4e'%dy1, '%.4e'%dtheta1, '%.4e'%dl1,
                                  '%.4e'%dx2, '%.4e'%dy2, '%.4e'%dtheta2, '%.4e'%dl2]))
                    k = j+2
                    j+=3
                else:
                    j+=1
                    pass

        print(i_dir+' :image until %s included'%rgb_dir[k])
        #print(i_dir+' :image until %s included'%depth_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open(name, 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n'%item)

def write_rnn(bp_steps):
    """
    In main function, actions_clean.dat are cleaned in a way that every actions corresponds to
    cleaned rgb images and you can just iterate through every action line and choose the write
    rgb images, meaning to jump over missing images. However, if you want to take more than two
    images in every written line in the final data text, every two actions have to be taken
    every time. But if you execute the loop from the action file, there is no way to guarantee
    that two actions are always consecutive from actions_clean.dat. You can do that in actions.dat.
    However, it is easier to do it from rgb_num and pick the right action from actions.dat.
    """
    path = 'train_cube_table/run*'#'train_cube/run*'
    train_dir = sorted(glob.glob(path))
    #name = 'train_cube_ae_rnn_6.txt'
    name = 'train_cube_table_rnn_6.txt'

    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        # image numbers that are actually there.
        rgb_num = [int(j[-8:-4]) for j in rgb_dir]

        train_dir_aug = []

        with open(i_dir+'/actions.dat', 'r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        #When picking up actions, we should use the image number rather than the image index since
        #there are images missing. When picking up image, we should use image index.
        #rgb_dir and rgb_num are totally corresponding to each other.
        # when append, you append whatever. when extend, you add the element in side the list into
        # the new list. Btw, a[:3] means all elements before a[3]/ before index 3.
        k = 0
        for j, item in enumerate(rgb_num[:-(bp_steps-1)]):
            img_list = []
            action_list_list = []

            img_number_correct = 1
            # 0, 2, 3, ..., bp_steps-2
            # 1-0, 2-1, ..., bp_steps-1 - bp_steps-2
            for m in range(bp_steps-1):
                if rgb_num[j+m+1] - rgb_num[j+m] != 1:
                    img_number_correct = 0

            if img_number_correct:
                for n in range(bp_steps):
                    img_list.append(rgb_dir[j+n]) # get a list of strings.
                for n in range(bp_steps-1):
                    action = actions[rgb_num[j+n]]
                    dx = bound_to_float(action[2])/240.0
                    dy = bound_to_float(action[3])/240.0
                    dtheta = float(action[4])/(2*np.pi)
                    dl = (float(action[5])-0.04)/0.04
                    # list of strings added to a list.
                    action_list_list.append(['%.4e'%dx, '%.4e'%dy, '%.4e'%dtheta, '%.4e'%dl])

                img_list_string = [' '.join(img_list)] # list of one string
                action_list_string = [' '.join(sub_list) for sub_list in action_list_list]
                total_list = img_list_string + action_list_string
                train_dir_aug.append(';'.join(total_list))

                k = j+bp_steps-1
            else:
                pass

        print(i_dir+' :image until %s included'%rgb_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open(name, 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n'%item)

def write_rnn_three_test(CONSECUTIVE = 1):
    #path = 'test_cube/run*'
    #name = 'test_cube_3.txt'
    path = 'test_cube_table/run*'
    name = 'test_cube_table_3.txt'

    train_dir = sorted(glob.glob(path))
    train_dir = train_dir[:10]
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        # image numbers that are actually there.
        rgb_num = [int(j[-8:-4]) for j in rgb_dir]

        train_dir_aug = []

        with open(i_dir+'/actions.dat', 'r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        if CONSECUTIVE:
            k = 0
            # change -2 into -3 since actions are one less than images. action3 out of range.
            for j, item in enumerate(rgb_num[:-3]):
                if (rgb_num[j+1]-item==1 and rgb_num[j+2]-rgb_num[j+1]==1):
                    action1 = actions[item]
                    dx1 = bound_to_float(action1[2])/240.0
                    dy1 = bound_to_float(action1[3])/240.0
                    dtheta1 = float(action1[4])/(2*np.pi)
                    dl1 = (float(action1[5])-0.04)/0.04
                    bx1 = float(action1[6])
                    by1 = float(action1[7])

                    action2 = actions[rgb_num[j+1]]
                    dx2 = bound_to_float(action2[2])/240.0
                    dy2 = bound_to_float(action2[3])/240.0
                    dtheta2 = float(action2[4])/(2*np.pi)
                    dl2 = (float(action2[5])-0.04)/0.04
                    bx2 = float(action2[6])
                    by2 = float(action2[7])

                    action3= actions[rgb_num[j+2]]
                    bx3 = float(action3[6])
                    by3 = float(action3[7])

                    train_dir_aug.append(
                        ' '.join([rgb_dir[j], rgb_dir[j+1], rgb_dir[j+2],
                                  '%.4e'%dx1, '%.4e'%dy1, '%.4e'%dtheta1, '%.4e'%dl1,
                                  '%.4e'%dx2, '%.4e'%dy2, '%.4e'%dtheta2, '%.4e'%dl2,
                                  '%.4e'%bx1, '%.4e'%by1, '%.4e'%bx2, '%.4e'%by2,
                                  '%.4e'%bx3, '%.4e'%by3]))
                    k = j+2
                else:
                    pass

        else:
            k = 0
            j = 0
            while j < len(rgb_num)-3:
                if rgb_num[j+1]-rgb_num[j]==1 and rgb_num[j+2]-rgb_num[j]==2:
                    action1 = actions[rgb_num[j]]
                    dx1 = bound_to_float(action1[2])/240.0
                    dy1 = bound_to_float(action1[3])/240.0
                    dtheta1 = float(action1[4])/(2*np.pi)
                    dl1 = (float(action1[5])-0.04)/0.04
                    bx1 = float(action1[6])
                    by1 = float(action1[7])

                    action2 = actions[rgb_num[j+1]]
                    dx2 = bound_to_float(action2[2])/240.0
                    dy2 = bound_to_float(action2[3])/240.0
                    dtheta2 = float(action2[4])/(2*np.pi)
                    dl2 = (float(action2[5])-0.04)/0.04
                    bx2 = float(action2[6])
                    by2 = float(action2[7])

                    action3= actions[rgb_num[j+2]]
                    bx3 = float(action3[6])
                    by3 = float(action3[7])
                    train_dir_aug.append(
                        ' '.join([rgb_dir[j], rgb_dir[j+1], rgb_dir[j+2],
                                  '%.4e'%dx1, '%.4e'%dy1, '%.4e'%dtheta1, '%.4e'%dl1,
                                  '%.4e'%dx2, '%.4e'%dy2, '%.4e'%dtheta2, '%.4e'%dl2,
                                  '%.4e'%bx1, '%.4e'%by1, '%.4e'%bx2, '%.4e'%by2,
                                  '%.4e'%bx3, '%.4e'%by3]))
                    k = j+2
                    j+=3
                else:
                    j+=1
                    pass

        print(i_dir+' :image until %s included'%rgb_dir[k])
        #print(i_dir+' :image until %s included'%depth_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open(name, 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n'%item)

def write_rnn_test(bp_steps):
    #path = 'test_cube/run*'
    #name = 'test_multi_cube_%d.txt'%bp_steps

    path = 'test_cube_table/run*'
    name = 'test_multi_cube_table_%d.txt'%bp_steps

    train_dir = sorted(glob.glob(path))
    #train_dir = train_dir[:10]
    train_dir_total = []

    for i, i_dir in enumerate(train_dir):
        depth_dir = sorted(glob.glob(i_dir+'/*.png'))
        rgb_dir = sorted(glob.glob(i_dir+'/*.jpg'))

        # image numbers that are actually there.
        rgb_num = [int(j[-8:-4]) for j in rgb_dir]

        train_dir_aug = []

        with open(i_dir+'/actions.dat', 'r') as f:
            actions = [line.strip().split() for line in f.readlines()]

        k = 0
        # change -(bp_steps-1) in -bp_steps
        for j, item in enumerate(rgb_num[:-bp_steps]):
            img_list = []
            action_list_list = []
            position_list_list = []

            img_number_correct = 1
            # 0, 2, 3, ..., bp_steps-2
            # 1-0, 2-1, ..., bp_steps-1 - bp_steps-2
            for m in range(bp_steps-1):
                if rgb_num[j+m+1] - rgb_num[j+m] != 1:
                    img_number_correct = 0

            if img_number_correct:
                for n in range(bp_steps):
                    img_list.append(rgb_dir[j+n]) # get a list of strings.
                    action = actions[rgb_num[j+n]]
                    bx = float(action[6])
                    by = float(action[7])
                    position_list_list.append(['%.4e'%bx, '%.4e'%by])
                for n in range(bp_steps-1):
                    action = actions[rgb_num[j+n]]
                    dx = bound_to_float(action[2])/240.0
                    dy = bound_to_float(action[3])/240.0
                    dtheta = float(action[4])/(2*np.pi)
                    dl = (float(action[5])-0.04)/0.04
                    # list of strings added to a list.
                    action_list_list.append(['%.4e'%dx, '%.4e'%dy, '%.4e'%dtheta, '%.4e'%dl])

                img_list_string = [' '.join(img_list)] # list of one string
                action_list_string = [' '.join(sub_list) for sub_list in action_list_list]
                position_list_string = [' '.join(sub_list) for sub_list in position_list_list]
                total_list = img_list_string + action_list_string + position_list_string
                train_dir_aug.append(';'.join(total_list))

                k = j+bp_steps-1
            else:
                pass

        print(i_dir+' :image until %s included'%rgb_dir[k])
        train_dir_total.extend(train_dir_aug)

    with open(name, 'wb') as train_file:
        for item in train_dir_total:
            train_file.write('%s\n'%item)


if __name__ == '__main__':
    #main()

    #write_rnn_three(CONSECUTIVE=1)
    #write_rnn(6)

    #write_rnn_three_test(CONSECUTIVE=1)
    #write_rnn_test(6)

    write_for_tsne()

