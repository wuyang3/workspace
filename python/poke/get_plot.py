# -*- coding: utf-8 -*-
"""
Seaborn supports grouped boxplo but requires pandas data frame.
#import seaborn as sns
# using seaborn requires a panda data frame.
#sns.boxplot(data=box_groups_list, palette="PRGn", order=['t0','t1','t2'])
#plt.show()
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.style.use(['seaborn', 'presentation']) # for loss plots
mpl.rcParams.update({'font.size': 24}) # for boxplots
import pandas as pd
import seaborn as sns

def get_loss(paths, labels):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ['m', 'b', 'g', 'r']
    df_list = []
    for i, path in enumerate(paths):
        df_temp = pd.read_csv(path)
        df_temp.rename(columns = {'Value':'Loss'}, inplace = True)
        length = df_temp.shape[0]
        label_col = [labels[i] for j in range(length)]
        df_temp['labels'] = label_col

        #df_list.append(df_temp)
        ax.semilogy(df_temp['Step'], df_temp['Loss'], color=colors[i], label=labels[i])
    #df_result = pd.concat(df_list, axis=0, ignore_index=True)
    #sns.tsplot(time='Step', unit='labels', value='Loss', condition='labels', data=df_result)
    #ax.set_yscale('log')

    ax.legend(loc=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('loss')
    plt.subplots_adjust(bottom=0.15, top=0.95)
    plt.savefig('test_data/loss_2.pdf', format='pdf', dpi=600)

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
    #title = p +' ($\mu$, $\sigma$)=' +' '.join(title_l)
    title = 'lstm vae: 6 time steps'

    fig, ax = plt.subplots()
    ax.boxplot(dat_array, widths=0.25)
    ax.set_xticklabels(['$t_0$']+['$t_0+%i$'%i for i in range(1, bp_steps)])
    ax.set_xticks(range(1, bp_steps+1))
    ax.set_xlabel('time steps')
    ax.set_ylabel('squared pixel errors')
    ax.set_title(title)
    #plt.show()
    plt.waitforbuttonpress()
    plt.savefig(
        '/home/wuyang/workspace/python/poke/test_data/box_plot.pdf', format='pdf', dpi=600)
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
        'Jordan rnn dae', 'Jordan lstm dae', 'Jordan lstm vae',
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
        '/home/wuyang/workspace/python/poke/test_data/box_plot_compare_8.pdf', format='pdf', dpi=600)
    plt.close()

if __name__ == '__main__':
    # get single box plot.
    path0 = '/home/wuyang/workspace/python/poke/test_data/offline/'
    path_list = [
        'sanity_check/',
        'rnn_dae(bn)/', 'lstm_dae(bn)/', 'rnn_vae/',
        'lstm_vae/', 'lstm_vae_little/',
        'ff_rnn_dae(bn)/', 'ff_lstm_dae(bn)/', 'ff_lstm_vae/',
        '6_lstm_vae_8ep/', '6_lstm_vae_new/',
        'newset_6_lstm_vae_12/'
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
    # box_plot(path0+path_list[11])
    # box_plot('/home/wuyang/workspace/python/poke/test_data/')

    # get group box plots.
    #group_box_plot([path0+item for item in path_list[:5]])
    # group_box_plot(['/home/wuyang/workspace/python/poke/test_data/', '/home/wuyang/workspace/python/poke/test_data/offline/lstm_vae/'])

    paths = ['run_rnn_dae(bn),tag_loss_rec.csv', 'run_lstm_dae(bn),tag_loss_rec.csv',
             'run_rnn_vae,tag_loss_loss_vae.csv', 'run_lstm_vae,tag_loss_loss_vae.csv']
    paths = ['test_data/misc/'+item for item in paths]
    labels = ['rnn dae', 'lstm dae', 'rnn vae', 'lstm vae']

    # get loss plot.
    #get_loss(paths[2:], labels[2:])
