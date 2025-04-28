from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt

from PhagoPred.utils import mask_funcs
from PhagoPred import SETTINGS

plt.rcParams["font.family"] = 'serif'

def update_or_create(group, name, data):
    if name in group:
        dset = group[name]
        # Resize if shapes don't match
        if dset.shape != data.shape:
            dset.resize(data.shape)
        dset[...] = data
    else:
        group.create_dataset(name, data=data, maxshape=tuple([None]*data.ndim))

class PCAKmeansFit:

    def __init__(self, h5py_file):
        self.h5py_file = h5py_file
        self.h5py_group = None
        self.feature_names = None
        
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
            sys.stdout.write(f'\rProcessing Frame {i+1} / {self.num_training_images}')
            sys.stdout.flush()
            new_feature = self.feature_func(frame)
            if i==0:
                features = new_feature
            else:
                features = np.concatenate((features, new_feature), axis=0)
            # features.append(self.feature_func(frame))
        # remove np.nan features
        features = features[~np.isnan(features).any(axis=1)]
        # features = np.array(features)
        print('Fitting PCA ...')
        self.pca = PCA(self.num_pca_componeents, svd_solver='full')
        self.train_pcs = self.pca.fit_transform(features)

        with h5py.File(self.h5py_file, 'r+') as f:
            pca_path = f'{self.h5py_group}/PCA'
            if pca_path in f:
                pca_group = f[pca_path]
            else:
                pca_group = f.create_group(pca_path)

            update_or_create(pca_group, 'components_', self.pca.components_)
            update_or_create(pca_group, 'mean_', self.pca.mean_)
            update_or_create(pca_group, 'explained_variance_', self.pca.explained_variance_)
            update_or_create(pca_group, 'explained_variance_ratio_', self.pca.explained_variance_ratio_)
            update_or_create(pca_group, 'train_pcs', self.train_pcs)

            # pca_group = f.create_group(f'{self.h5py_group}/PCA')

            # pca_group.create_dataset('components_', data=self.pca.components_)
            # pca_group.create_dataset("mean_", data=self.pca.mean_)
            # pca_group.create_dataset("explained_variance_", data=self.pca.explained_variance_)
            # pca_group.create_dataset("explained_variance_ratio_", data=self.pca.explained_variance_ratio_)
            # pca_group.create_dataset("train_pcs", data=self.train_pcs)

            pca_group.attrs["n_components"] = self.pca.n_components_
    
    def fit_kmeans(self):

        print('Fitting KMeans clustering ...')
        self.kmeans = KMeans(self.num_kmeans_clusters)
        train_labels = self.kmeans.fit_predict(self.train_pcs)

        with h5py.File(self.h5py_file, 'r+') as f:

            if f.get(self.h5py_group) is None:
                f.create_group(self.h5py_group)
            
            # Create or get the KMeans group
            kmeans_path = f'{self.h5py_group}/KMeans'
            if kmeans_path in f:
                kmeans_group = f[kmeans_path]
            else:
                kmeans_group = f.create_group(kmeans_path)

            update_or_create(kmeans_group, 'cluster_centers_', self.kmeans.cluster_centers_)
            update_or_create(kmeans_group, 'train_labels', train_labels)

            # kmeans_group = f.create_group(f'{self.h5py_group}/KMeans')
            # kmeans_group.create_dataset('cluster_centers_', data=self.kmeans.cluster_centers_)
            # kmeans_group.create_dataset('train_labels', data=train_labels)

            # kmeans_group.attrs['n_clusters'] = self.num_kmeans_clusters

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
        #only apply to cells with no np.nan values
        valid_mask = ~np.isnan(features).any(axis=1)

        pcs = self.pca.transform(features[valid_mask])

        kmeans_distances = self.kmeans.transform(pcs)
        
        #refill np.nan values to ensure consistaent indexing
        results = np.full((features.shape[0], self.num_kmeans_clusters), np.nan)
        results[valid_mask] = kmeans_distances

        return results
    
    def view_clusters(self):
        plt.rcParams["font.family"] = 'serif'
        with h5py.File(self.h5py_file, 'r') as f:
            clusters = f[self.h5py_group]['KMeans']['train_labels'][:]
            pcs = f[self.h5py_group]['PCA']['train_pcs'][:]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            colours = ['blue', 'orange', 'green', 'red']
            for cluster in range(self.num_kmeans_clusters):
                idxs = np.nonzero(clusters==cluster)
                ax.scatter(pcs[idxs, 0], pcs[idxs, 1], pcs[idxs, 2], label=str(cluster), color=colours[cluster])
        plt.legend()
        plt.show()

    def pc_heatmap(self):
        plt.rcParams["font.family"] = 'serif'
        with h5py.File(self.h5py_file, 'r') as f:

            plt.matshow(f[self.h5py_group]['PCA']['components_'])
            plt.xlabel('Input features')
            plt.ylabel('PCs')
        plt.show()

    def rank_input_features(self, colours=None):
        "rank all input features based on total contribution to PCs, 'coefficients/loadings'"


        if self.pca is None:
            self.load_pca()

        total_contributions = np.sum(np.abs(self.pca.components_), axis=0)

        order = np.argsort(total_contributions)[-15:][::-1]
        # sorted_values, sorted_names = zip(*np.sort(zip(total_contributions, self.feature_names.flatten()), reverse=True))
        bar_values, names = total_contributions[order], self.feature_names.flatten()[order]
        if colours is None:
            colour='k'
        else:
            # cmap = plt.get_cmap('gist_rainbow')
            colour = [tuple(colours[col_key]) for name in names for col_key in colours.keys() if name.startswith(col_key)]
            # colour = [colours[col_key] if any(name.startswith(col_key) for col_key in colours.keys()) for name in names]

        plt.bar(np.arange(len(bar_values)), bar_values, color=colour)
        plt.xlabel('Input Features')
        plt.xticks(ticks=np.arange(len(bar_values)), labels=names, rotation=90)
        plt.ylabel('Absolute sum of loadings')
        plt.tight_layout()
        plt.savefig('temp/example.png')

