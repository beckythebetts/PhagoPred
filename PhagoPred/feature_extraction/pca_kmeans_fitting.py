from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

from PhagoPred.utils import mask_funcs
from PhagoPred import SETTINGS

class PCAKmeansFit:

    def __init__(self, h5py_file):
        self.h5py_file = h5py_file
        self.h5py_group = None
        
        self.num_training_images = None
        self.num_pca_componeents = None
        self.num_kmeans_clusters = None

        self.train_pcs = None
        self.pca = None
        self.kmeans = None

    def preprocess(self):
        pass

    def feature_func(self, frame):
        pass

    def fit_pca(self):
        if self.num_training_images is None:
            self.training_frames = np.arange(0, SETTINGS.NUM_FRAMES)
        else:
            self.training_frames = np.random.randint(0, SETTINGS.NUM_FRAMES, size=self.num_training_images)

        for i, frame in enumerate(self.training_frames):
            sys.stdout.write(f'\rProcessing Frame {i+1} / {SETTINGS.NUM_TRAINING_FRAMES}')
            sys.stdout.flush()
            new_feature = self.feature_func(frame)
            if i==0:
                features = new_feature
            else:
                features = np.concatenate((features, new_feature), axis=0)
            # features.append(self.feature_func(frame))

        features = np.array(features)
        print('Fitting PCA ...')
        self.pca = PCA(self.num_pca_componeents, svd_solver='full')
        self.train_pcs = self.pca.fit_transform(features)

        with h5py.File(self.h5py_file, 'r+') as f:
            pca_group = f.create_group(f'{self.h5py_group}/PCA')
            pca_group.create_dataset('components_', data=self.pca.components_)
            pca_group.create_dataset("mean_", data=self.pca.mean_)
            pca_group.create_dataset("explained_variance_", data=self.pca.explained_variance_)
            pca_group.create_dataset("explained_variance_ratio_", data=self.pca.explained_variance_ratio_)
            pca_group.create_dataset("train_pcs", data=self.train_pcs)

            pca_group.attrs["n_components"] = self.pca.n_components_
    
    def fit_kmeans(self):

        print('Fitting KMeans clustering ...')
        self.kmeans = KMeans(self.num_kmeans_clusters)
        train_labels = self.kmeans.fit_predict(self.train_pcs)

        with h5py.File(self.h5py_file, 'r+') as f:
            kmeans_group = f.create_group(f'{self.h5py_group}/KMeans')
            kmeans_group.create_dataset('cluster_centers_', data=self.kmeans.cluster_centers_)
            kmeans_group.create_dataset('train_labels', data=train_labels)

            kmeans_group.attrs['n_clusters'] = self.num_kmeans_clusters

    def fit(self):
        self.fit_pca()
        self.fit_kmeans()

    def load_pca(self):
        with h5py.File(self.h5py_file, 'r') as f:
            pca_group = f[self.h5py_group]['PCA']
            components_ = pca_group["components_"][:]
            mean_ = pca_group["mean_"][:]
            explained_variance_ = pca_group["explained_variance_"][:]
            explained_variance_ratio_ = pca_group["explained_variance_ratio_"][:]
            n_components = pca_group.attrs["n_components"]

            # Reconstruct the PCA object
            self.pca = PCA(n_components=n_components)
            self.pca.components_ = components_
            self.pca.mean_ = mean_
            self.pca.explained_variance_ = explained_variance_
            self.pca.explained_variance_ratio_ = explained_variance_ratio_

    def load_kmeans(self):
        with h5py.File(self.h5py_file, 'r') as f:
            kmeans_group = f[self.h5py_group]['KMeans']
            cluster_centers_ = kmeans_group["cluster_centers_"][:]
            n_clusters = kmeans_group.attrs["n_clusters"]

            self.kmeans = KMeans(n_clusters=n_clusters)
            self.kmeans.cluster_centers_ = cluster_centers_
        
    def apply(self, frame):
        if self.pca is None:
            self.load_pca()
        if self.kmeans is None:
            self.load_kmeans()

        features = self.feature_func(frame)

        pcs = self.pca.transform(features)

        kmeans_distances = self.kmeans.transform(pcs)

        return kmeans_distances
    
    def view_clusters(self):
        plt.rcParams["font.family"] = 'serif'
        with h5py.File(self.h5py_file, 'r') as f:
            clusters = f['KMeans']['train_labels'][:]
            pcs = f['PCA']['train_pcs'][:]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            colours = ['blue', 'orange', 'green', 'red']
            for cluster in range(self.num_kmeans_clusters):
                idxs = np.nonzero(clusters==cluster)
                ax.scatter(pcs[idxs, 0], pcs[idxs, 1], pcs[idxs, 2], label=str(cluster), color=colours[cluster])
        plt.legend()
        plt.show()



