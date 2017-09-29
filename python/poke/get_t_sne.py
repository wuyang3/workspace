import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ggplot import *

mnist = fetch_mldata("MNIST original")
X = mnist.data/255.0
y = mnist.target

feat_cols = ['pixel%d'%i for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

rndperm = np.random.permutation(df.shape[0])

# # Starting PCA.
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df[feat_cols].values)

# df['pca-one'] = pca_result[:, 0]
# df['pca-two'] = pca_result[:, 1]
# df['pca-three'] = pca_result[:, 2]

# print('Explained variance per component {}'.format(pca.explained_variance_ratio_))

# chart = ggplot(df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label'))\
#                                                 + geom_point(size=75,alpha=0.8)\
#                                                 + ggtitle("First and Second Principal Components colored by digit")
# print(chart)

# start t-sne

# n_sne = 7000
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne], feat_cols].values)

# print('t-sne done. Time elapsed: {} seconds'.format(time.time()-time_start))

# df_tsne = df.loc[rndperm[:n_sne], :].copy()
# df_tsne['x-tsne'] = tsne_results[:, 0]
# df_tsne['y-tsne'] = tsne_results[:, 1]

# chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
#                                 + geom_point(size=70,alpha=0.1) \
#                                 + ggtitle("tSNE dimensions colored by digit")
# print(chart)

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
