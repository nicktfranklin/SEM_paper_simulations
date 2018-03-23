import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA

class TestData(object):
    """
    a base class for test datasets
    """
    D = 0 # dimension of scene vectors
    X = [[]] # sequence of scene vectors
    y = [] # sequence of event labels for the corresponding scenes

    def scenes(self):
        return self.X

    def events(self):
        return self.y

    def plot_scenes(self):
        if self.D == 2:
            X = self.X
        else:
            # if scenes are > 2 dimensional, run PCA first (this is for plotting only)
            pca = PCA(n_components=2)
            pca.fit(self.X)
            X = pca.transform(self.X)

        plt.plot(X[:, 0], X[:, 1])
        plt.show()

    def performance(self, post):
        y_hat = np.argmax(post, axis=1)
        mi = metrics.adjusted_mutual_info_score(self.y, y_hat)
        print "Adjusted Mutual Information:", mi
        r = metrics.adjusted_rand_score(self.y, y_hat)
        print "Adjusted Rand Score:", r
        print 
        print np.argmax(post, axis=1)
        return mi, r

    def plot_segmentation(self, post):
        if self.D == 2:
            X = self.X
        else:
            # if scenes are > 2 dimensional, run PCA first (this is for plotting only)
            pca = PCA(n_components=2)
            pca.fit(self.X)
            X = pca.transform(self.X)

        cluster_id = np.argmax(post, axis=1)
        cc = sns.color_palette('Dark2', post.shape[1])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1, 2]))
        for clt in cluster_id:
            idx = np.nonzero(cluster_id == clt)[0]
            axes[0].scatter(X[idx, 0], X[idx, 1], color=cc[clt], alpha=.5)
            
        axes[1].plot(post)
        plt.show()

    def plot_max_cluster(self, post):
	fig, axes = plt.subplots(2, 1)

	max_post = np.zeros(post.shape)
	for t in range(post.shape[0]):
	    max_post[t, :] = post[t, :] == post[t, :].max()
	axes[0].plot(max_post)
	axes[0].set_title('SEM max a posterior cluster')

	y_clust = np.zeros((self.y.shape[0], 13))
	for ii, y0 in enumerate(self.y):
	    y_clust[ii, y0] = 1.0
	axes[1].plot(y_clust)
	axes[1].set_title('True Clusters')
	axes[1].set_xlabel('Time')
	plt.subplots_adjust(hspace=0.4)
	plt.show()
