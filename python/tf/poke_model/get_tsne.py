# -*- coding: utf-8 -*-
"""
Generate embeding from rnn layers and use them for t-sne embedding.
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import math
import matplotlib as mpl
mpl.rcParams.update({'font.size': 24}) # for boxplots
import matplotlib.pyplot as plt

from poke_bn_vae import PokeBnVAE
from poke_ae_rnn import PokeVAERNN

import time
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.ndimage
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ggplot import *
import seaborn as sns

def read_data_embedding(data_list, num_epochs, shuffle):
    assert type(shuffle)==bool
    image_paths = []
    with open(data_list, 'r') as f:
        for item in f:
            item = item.strip()
            image_paths.append('../../poke/'+item)
    i_t = ops.convert_to_tensor(image_paths, dtype=dtypes.string)
    input_queue = tf.train.slice_input_producer(
        [i_t], num_epochs=num_epochs, shuffle=shuffle)
    return input_queue

def batch_dir_images(
        input_queue, batch_size, num_threads, min_after_dequeue, type_img=1, normalized=0):
    img_file = tf.read_file(input_queue[0])
    if type_img:
        img = tf.image.decode_jpeg(img_file)
        img_float = tf.cast(img, tf.float32)
        if normalized:
            img_float = img_float/255.0*2.0-1.0
        img_resized = tf.image.resize_images(img_float, [64, 64])
        img_resized.set_shape((64, 64, 3))
    else:
        img = tf.image.decode_png(img_file, dtype=tf.uint8)
        img_float = tf.cast(img, tf.float32)
        if normalized:
            tmg_float = img_float/255.0*2.0-1.0
        img_resized = tf.image.resize_images(img_float, [64, 64])
        img_resized.set_shape((64, 64, 1))

    dirs, images = tf.train.batch([input_queue[0], img_resized],
                                 batch_size=batch_size,
                                 num_threads=num_threads,
                                 capacity=min_after_dequeue+3*batch_size)
    output = (dirs, images)
    return output

def get_embeddings():
    shuffle = True
    num_epochs = 1

    batch_size = 1
    num_threads = 1
    min_after_dequeue = 4

    type_img = 1
    in_channels = 3
    normalized = 0
    split_size = 512

    paths_name = '../../poke/embedding_newtrans_generate.txt'
    path = '../../poke/test_trans/'
    #restore_path = '../logs/pokeAERNN/lstm_vae/'
    restore_path = '../logs/pokeAERNN_new/lstm_vae_14/'

    input_queue = read_data_embedding(paths_name, num_epochs, shuffle)
    dirs, imgs = batch_dir_images(input_queue, batch_size, num_threads, min_after_dequeue,
                                  type_img, normalized=0)

    step = 0
    with tf.Session() as sess:
        poke_ae = PokeVAERNN(batch_size=batch_size, split_size=split_size, in_channels=in_channels,
                             corrupted=0, is_training=False, lstm=1)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(restore_path))
        tf.local_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        identity_list = []
        transit_list = []
        path_list = []
        try:
            while not coord.should_stop():
                dir, img = sess.run([dirs, imgs])
                feed_dict = {poke_ae.i1: img,
                             poke_ae.u2: np.zeros([batch_size, 4]),
                             poke_ae.u3: np.zeros([batch_size, 4])}

                identity, transit = sess.run([poke_ae.identity, poke_ae.transit_series[0]],
                                                   feed_dict=feed_dict)
                identity_list.append(identity)
                transit_list.append(transit)
                path_list.append(dir)
                step+=1
                print step
        except tf.errors.OutOfRangeError:
            print('Done at step %d'%step)
        finally:
            iden_arr = np.concatenate(identity_list, axis=0)
            transit_arr = np.concatenate(transit_list, axis=0)
            path_arr = np.array(path_list)
            np.save(path+'iden_arr.npy', iden_arr)
            np.save(path+'tran_arr.npy', transit_arr)
            np.save(path+'path_arr.npy', path_arr)
            print iden_arr.shape
            print transit_arr.shape
            print path_arr.shape
            coord.request_stop()
            coord.join(threads)

def tsne_mnist():
    mnist = fetch_mldata("MNIST original")
    X = mnist.data/255.0
    y = mnist.target

    feat_cols = ['pixel%d'%i for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feat_cols)
    df['label'] = y
    df['label'] = df['label'].apply(lambda i: str(i))

    rndperm = np.random.permutation(df.shape[0])

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
    print('Explained variance per component {}'.format(np.sum(pca_50.explained_variance_ratio_)))

    n_sne = 10000

    time_start = time.time()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=600)
    tsne_pca_results = tsne.fit_transform(pca_result_50[rndperm[:n_sne]])
    print('t-sne done. Time elapsed: {} seconds'.format(time.time()-time_start))
    df_tsne = None
    df_tsne = df.loc[rndperm[:n_sne], :].copy()
    df_tsne['x-tsne-pca'] = tsne_pca_results[:, 0]
    df_tsne['y-tsne-pca'] = tsne_pca_results[:, 1]

    chart = ggplot( df_tsne, aes(x='x-tsne-pca', y='y-tsne-pca', color='label') ) \
                                    + geom_point(size=70,alpha=0.1) \
                                    + ggtitle("tSNE dimensions colored by Digit (PCA)")
    print(chart)

def imscatter(x, y, image_paths, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()

    im_list = []
    for path in image_paths:
        img = scipy.ndimage.imread(path)
        im = OffsetImage(img, zoom=zoom)
        im_list.append(im)
    x, y = np.atleast_1d(x, y)
    artists = []
    for i, (x0, y0) in enumerate(zip(x, y)):
        ab = AnnotationBbox(im_list[i], (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def tsne_embedding():
    path = '../../poke/test_trans/'
    iden = np.load(path+'iden_arr.npy')
    trans = np.load(path+'tran_arr.npy')
    path = np.load(path+'path_arr.npy')

    size = trans.shape[1]
    feat_cols = ['i%d'%i for i in range(size)]

    df = pd.DataFrame(trans[:, :], columns=feat_cols)
    df['path'] = path

    pca_50 = PCA(n_components=50)
    pca_50_r = pca_50.fit_transform(df[feat_cols].values)
    print('Explained variance {}'.format(np.sum(pca_50.explained_variance_ratio_)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
    tsne_r = tsne.fit_transform(pca_50_r)
    print('Time elapsed {}'.format(time.time()-time_start))

    df['x_tsne'] = tsne_r[:, 0]
    df['y_tsne'] = tsne_r[:, 1]

    #sns.lmplot('x_tsne', # Horizontal axis
    #           'y_tsne', # Vertical axis
    #           data=df, # Data source
    #           fit_reg=False, # Don't fix a regression line
    #           scatter_kws={"marker": "D", "s": 100})

    #rndperm = np.random.permutation(df.shape[0])
    df_tsne = df.iloc[:, :].copy()

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(df['x_tsne'], df['y_tsne'], c='m', marker='o', s=60)

    imscatter(df_tsne['x_tsne'], df_tsne['y_tsne'], df_tsne['path'], ax=ax, zoom=0.2)
    plt.title('tsne')
    plt.xlabel('x component')
    plt.ylabel('y component')
    #plt.waitforbuttonpress()
    #plt.savefig(path+'tsne_trans.pdf', format='pdf', dpi=600)
    plt.show()

if __name__ == '__main__':
    #get_embeddings()
    tsne_embedding()
