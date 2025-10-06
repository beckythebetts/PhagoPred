from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    
    def training_preprocess(self, features):
        """OPtionall yoverwirte to apply a funciton to all trainign features."""
        return features

    def fit_pca(self):
        if self.num_training_images is None:
            self.training_frames = np.arange(0, SETTINGS.NUM_FRAMES)
        else:
            self.training_frames = np.random.randint(0, SETTINGS.NUM_FRAMES, size=self.num_training_images)
            self.training_frames = np.random.choice(np.arange(0, SETTINGS.NUM_FRAMES), size=self.num_training_images, replace=False)

        for i, frame in enumerate(tqdm(self.training_frames)):
            # sys.stdout.write(f'\rProcessing Frame {i+1} / {self.num_training_images}')
            # sys.stdout.flush()
            new_feature = self.feature_func(frame)
            if i==0:
                features = new_feature
            else:
                features = np.concatenate((features, new_feature), axis=0)
            # features.append(self.feature_func(frame))
        # remove np.nan features
        features = features[~np.isnan(features).any(axis=1)]
        self.features = self.training_preprocess(features)
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
            # kmeans_group.attrs['n_clusters'] = self.num_kmeans_clusters 
            n_clusters = kmeans_group.attrs["n_clusters"]

            self.kmeans = KMeans(n_clusters=n_clusters)
            self.kmeans.cluster_centers_ = cluster_centers_

    def load_model(self):
        self.load_pca()
        self.load_kmeans()
        
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
    
    def apply_expanded_mask(self, expanded_mask):
        if self.pca is None:
            self.load_pca()
        if self.kmeans is None:
            self.load_kmeans()

        features = self.feature_func_expanded_mask(expanded_mask)
        #only apply to cells with no np.nan values
        valid_mask = ~np.isnan(features).any(axis=1)

        results = np.full((features.shape[0], self.num_kmeans_clusters), np.nan)

        try:
            pcs = self.pca.transform(features[valid_mask])
        
        except ValueError:
            return results

        kmeans_distances = self.kmeans.transform(pcs)
        
        #refill np.nan values to ensure consistaent indexing
        results[valid_mask] = kmeans_distances

        return results
    
    def view_clusters(self):
        plt.rcParams["font.family"] = 'serif'
        with h5py.File(self.h5py_file, 'r') as f:
            clusters = f[self.h5py_group]['KMeans']['train_labels'][:]
            pcs = f[self.h5py_group]['PCA']['train_pcs'][:]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # colours = ['blue', 'orange', 'green', 'red']
            cmap = plt.get_cmap('Set1')
            num_clusters = len(self.kmeans.cluster_centers_)
            # colours = [cmap(i / (num_clusters - 1)) for i in range(num_clusters)]
            colours = [cmap(i) for i in range(num_clusters)]
            for cluster in range(self.num_kmeans_clusters):
                idxs = np.nonzero(clusters==cluster)
                ax.scatter(pcs[idxs, 0], pcs[idxs, 1], pcs[idxs, 2], label=str(cluster), color=colours[cluster])
        plt.legend()
        plt.savefig('temp/example.png')
        plt.show()

    def view_clusters_2d(self):
        plt.rcParams["font.family"] = 'serif'
        with h5py.File(self.h5py_file, 'r') as f:
            clusters = f[self.h5py_group]['KMeans']['train_labels'][:]
            pcs = f[self.h5py_group]['PCA']['train_pcs'][:]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            colours = ['blue', 'orange', 'green', 'red']
            for cluster in range(self.num_kmeans_clusters):
                idxs = np.nonzero(clusters==cluster)
                ax.scatter(pcs[idxs, 0], pcs[idxs, 1], label=str(cluster), color=colours[cluster])
        plt.legend()
        plt.savefig('temp/example.png')


    def pc_heatmap(self):
        plt.rcParams["font.family"] = 'serif'
        with h5py.File(self.h5py_file, 'r') as f:

            plt.matshow(f[self.h5py_group]['PCA']['components_'])
            plt.xlabel('Input features')
            plt.ylabel('PCs')
        plt.show()

    def plot_explained_variance_ratio(self):
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of PCs')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()
        
    def umap_embedding(self):
        import umap.umap_ as umap
        import matplotlib.pyplot as plt
        import numpy as np

        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
        embedding = reducer.fit_transform(self.features)

        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], c='gray', s=20, alpha=0.1, label='UMAP embeddings')
        plt.title('UMAP embedding with mean contour overlays')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')

        def plot_contour_at_xy(ax, contour, x, y, scale=0.1, colour='black', linestyle='-', alpha=1, label=''):
            contour = contour.reshape(-1, 2)
            contour = contour * scale
            contour[:, 0] += x
            contour[:, 1] += y
            ax.plot(contour[:, 0], contour[:, 1], color=colour, linewidth=1, linestyle=linestyle, alpha=alpha, label=label)

        ax = plt.gca()

        num_bins_x = 10
        num_bins_y = 10

        x_bins = np.linspace(np.min(embedding[:, 0]), np.max(embedding[:, 0]), num_bins_x + 1)
        y_bins = np.linspace(np.min(embedding[:, 1]), np.max(embedding[:, 1]), num_bins_y + 1)

        for i in range(num_bins_x):
            for j in range(num_bins_y):
                # Find points in current bin
                in_bin = np.where(
                    (embedding[:, 0] >= x_bins[i]) & (embedding[:, 0] < x_bins[i + 1]) &
                    (embedding[:, 1] >= y_bins[j]) & (embedding[:, 1] < y_bins[j + 1])
                )[0]

                if len(in_bin) == 0:
                    continue  # skip empty bins
                
                  # Compute bin center for plotting
                x_centre = (x_bins[i] + x_bins[i + 1]) / 2
                y_centre = (y_bins[j] + y_bins[j + 1]) / 2
                
                # Average PCA scores in bin
                # mean = np.mean(np.array(self.features)[in_bin], axis=0)
                bin_features = np.array(self.features)[in_bin]
                colours = ('b', 'k', 'r')
                for percentile, colour in zip((5, 50, 95), colours):
                    percentile_features = np.percentile(bin_features, percentile, axis=0)
                    # linestyle = '-' if percentile == 50 else '--'
                    # colour = 'k' if percentile==50 else 'gray'
                    alpha = 1 if percentile==50 else 0.5
                    plot_contour_at_xy(ax, percentile_features, x_centre, y_centre, scale=5, colour=colour, alpha=alpha, label=f'{percentile}th pecentile contour in region')
                    
                    
                # Reconstruct mean contour
                # mean_contour = self.pca.inverse_transform(mean_pca)

                # Compute bin center for plotting
                # x_center = (x_bins[i] + x_bins[i + 1]) / 2
                # y_center = (y_bins[j] + y_bins[j + 1]) / 2

                # plot_contour_at_xy(ax, mean, x_center, y_center, scale=2.5, color='k')

        plt.title('UMAP embedding with example shape overlays')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.axis('equal')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

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

