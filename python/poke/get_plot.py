# -*- coding: utf-8 -*-
"""
Seaborn supports grouped boxplo but requires pandas data frame.
#import seaborn as sns
# using seaborn requires a panda data frame.
#sns.boxplot(data=box_groups_list, palette="PRGn", order=['t0','t1','t2'])
#plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt

def box_plot(path):
    with open(path+'error_rnn_offline.dat', 'rb') as f:
        dat = f.readlines()

    path_list = path.split('/')
    p = path_list[-2]

    dat_list_s = [item.strip().split() for item in dat]
    dat_list_f = [[float(s) for s in item] for item in dat_list_s]

    dat_array = np.array(dat_list_f)
    bp_steps = dat_array.shape[1]
    dat_array_list = np.split(dat_array, bp_steps, axis=1)

    mean = []
    std = []

    for item in dat_array_list:
        mean.append(np.mean(item))
        std.append(np.std(item))

    statistics = zip(mean, std)
    title_l = ['(%.2f, %.2f) '%item for item in statistics]
    title = p+' ($\mu$, $\sigma$)=' +' '.join(title_l)

    fig, ax = plt.subplots()
    ax.boxplot(dat_array)
    ax.set_xticklabels(['$t_0$']+['$t_0+%i$'%i for i in range(1, bp_steps)])
    ax.set_xticks(range(1, bp_steps+1))
    ax.set_xlabel('time steps')
    ax.set_ylabel('squared pixel errors')
    ax.set_title(title)
    #plt.show()
    plt.waitforbuttonpress()
    plt.savefig(
        '/home/wuyang/workspace/python/poke/test_data/box_plot.png', format='png', dpi=600)
    plt.close()

def set_bp_color(bp, color_list, num_per_group):
    for i in range(num_per_group):
        plt.setp(bp['caps'][2*i], color=color_list[i])
        plt.setp(bp['caps'][2*i+1], color=color_list[i])
        plt.setp(bp['whiskers'][2*i], color=color_list[i])
        plt.setp(bp['whiskers'][2*i+1], color=color_list[i])
        plt.setp(bp['fliers'][i], markeredgecolor=color_list[i])

def set_bp_label(bp, label_list, num_per_group):
    # for automatically generating legend. Only to the last group of boxes.
    for i in range(num_per_group):
        bp['whiskers'][2*i].set_label(label_list[i])

def group_box_plot(list_of_path):
    color_list = ['forestgreen', 'orangered', 'orchid', 'deepskyblue', 'springgreen',
                  'peachpuff', 'darkcyan', 'black', 'magenta', 'turquoise']
    label_list = [
        'no prediction',
        'rnn dae', 'lstm dae', 'rnn vae',
        'lstm vae', 'lstm vae s',
        'ff rnn dae', 'ff lstm dae', 'ff lstm vae',
        'lstm vae 1', 'lstm vae 2'
    ]
    num_per_group = len(list_of_path)
    dats_array_list = []

    for item in list_of_path:
        with open(item+'error_rnn_offline.dat', 'rb') as f:
            lines = f.readlines()

        path_list = item.split('/')
        p = path_list[-2]

        dat_list_s = [line.strip().split() for line in lines]
        dat_list_f = [[float(i) for i in sub_list] for sub_list in dat_list_s]

        dat_array = np.array(dat_list_f)
        num_groups = dat_array.shape[1]
        dat_array_list = np.split(dat_array, num_groups, axis=1)

        dats_array_list.append(dat_array_list) #[[col1, col2, col3, ...], [], [], ...]

    box_groups_list = []
    for i in range(num_groups):
        temp_single_group = []
        for sub_list in dats_array_list:
            temp_single_group.append(sub_list[i])
        box_groups_list.append(temp_single_group)

    fig, ax = plt.subplots()
    groups_mean = []
    for i, group in enumerate(box_groups_list):
        bp = ax.boxplot(group,
                        positions=range(i+i*num_per_group+1, i+(i+1)*num_per_group+1),
                        showmeans=False)
        set_bp_color(bp, color_list, num_per_group)
        groups_mean.append(i*(num_per_group+1)+(num_per_group+1)/2.0)
    set_bp_label(bp, label_list, num_per_group)

    ax.set_xticklabels(['$t_0$']+['$t_0+%i$'%i for i in range(1, num_groups)])
    ax.set_xticks(groups_mean)
    ax.set_xlim(0, num_groups*(num_per_group+1))
    ax.set_xlabel('time steps')
    ax.set_ylabel('squared pixel errors')
    ax.legend(loc=2)
    #plt.show()
    plt.waitforbuttonpress()
    plt.savefig(
        '/home/wuyang/workspace/python/poke/test_data/box_plot_compare_2.png', format='png', dpi=600)
    plt.close()

if __name__ == '__main__':
    path0 = '/home/wuyang/workspace/python/poke/test_data/offline/'
    path_list = [
        'sanity_check/',
        'rnn_dae(bn)/', 'lstm_dae(bn)/', 'rnn_vae/',
        'lstm_vae/', 'lstm_vae_little/',
        'ff_rnn_dae(bn)/', 'ff_lstm_dae(bn)/', 'ff_lstm_vae/',
        '6_lstm_vae_8ep/', '6_lstm_vae_new/'
    ]

    # box_plot(path0+path_list[0])
    # box_plot(path0+path_list[1])
    # box_plot(path0+path_list[2])
    # box_plot(path0+path_list[3])
    # box_plot(path0+path_list[4])
    # box_plot(path0+path_list[5])
    # box_plot(path0+path_list[6])
    # box_plot(path0+path_list[7])
    # box_plot(path0+path_list[8])
    # box_plot(path0+path_list[9])
    # box_plot(path0+path_list[10])

    group_box_plot([path0+item for item in path_list[:-2]])
