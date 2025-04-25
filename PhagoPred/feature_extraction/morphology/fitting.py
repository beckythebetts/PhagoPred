from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PhagoPred.feature_extraction.morphology.pre_processing import PreProcessing
from PhagoPred.utils import mask_funcs
from PhagoPred.feature_extraction.pca_kmeans_fitting import PCAKmeansFit
from PhagoPred import SETTINGS

## TESTING
class MorphologyFit(PCAKmeansFit):

    def __init__(self, h5py_file=SETTINGS.DATASET):
        super().__init__(h5py_file)
        self.h5py_group = 'Morphology'
        self.num_training_images = 50
        self.num_pca_componeents = 10
        self.num_kmeans_clusters = 4

    def feature_func(self, frame):
        """
        
        """
        results = []
        with h5py.File(self.h5py_file, 'r') as f:
            mask = f['Segmentations']['Phase'][f'{frame:04}'][:]
            contours = mask_funcs.get_border_representation(mask, f['Cells']['Phase'][:].shape[1], f, frame)
            for contour in contours:
                if len(contour) > 2:
                    results.append(PreProcessing(contour).pre_process().flatten())
        return np.array(results)
    
    def view_shape_modes(self):
        if self.pca is None:
            self.load_pca()
        if self.kmeans is None:
            self.load_kmeans()

        plt.rcParams["font.family"] = 'serif'
        # fig, axs = plt.subplots(1, SETTINGS.KMEANS_CLUSTERS)
        colours = ['blue', 'orange', 'green', 'red']
        for i, cluster in enumerate(self.kmeans.cluster_centers_):
            original_contour = self.pca.inverse_transform(cluster)
            original_contour = np.reshape(original_contour, (-1, 2))
            plt.plot(original_contour[:, 0], original_contour[:, 1], color=colours[i])
            # plt.set_aspect('equal')
            # axs[i].set_title(str(i))

        plt.show()
    



# def fit_pca_kmeans():
#     with h5py.File(SETTINGS.DATASET, 'r+') as f:
#         contours = []
#         train_frames = np.random.randint(0, SETTINGS.NUM_FRAMES, size=SETTINGS.NUM_TRAINING_FRAMES)

#         f['Cells']['Phase'].attrs['Num countour points'] = SETTINGS.NUM_CONTOUR_POINTS
#         f['Cells']['Phase'].attrs['Num training images'] = SETTINGS.NUM_TRAINING_FRAMES

#         for i, frame in enumerate(train_frames):
#             sys.stdout.write(f'\rFrame {i+1} / {SETTINGS.NUM_TRAINING_FRAMES}')
#             sys.stdout.flush()
#             mask = f['Segmentations']['Phase'][f'{frame:04}'][:]
#             raw_contours = mask_funcs.get_border_representation(mask, f['Cells']['Phase'][:].shape[1], f, i)
#             for c in raw_contours:
#                 new_contour = PreProcessing(c).pre_process()
#                 if len(new_contour) > 0:
#                     contours.append(new_contour.flatten())
#         pca = PCA(n_components=SETTINGS.PCA_COMPONENTS, svd_solver='full')
#         train_pcs = pca.fit_transform(np.array(contours))

#         pca_group = f.create_group('PCA')
#         pca_group.create_dataset('components_', data=pca.components_)
#         pca_group.create_dataset("mean_", data=pca.mean_)
#         pca_group.create_dataset("explained_variance_", data=pca.explained_variance_)
#         pca_group.create_dataset("explained_variance_ratio_", data=pca.explained_variance_ratio_)
#         pca_group.create_dataset("train_pcs", data=train_pcs)
#         pca_group.attrs["n_components"] = pca.n_components_

#         kmeans = KMeans(n_clusters=SETTINGS.KMEANS_CLUSTERS)
#         train_labels = kmeans.fit_predict(train_pcs)
#         kmeans_group = f.create_group('KMeans')
#         kmeans_group.create_dataset('cluster_centers_', data=kmeans.cluster_centers_)
#         kmeans_group.create_dataset('train_labels', data=train_labels)
#         kmeans_group.attrs['n_clusters'] = kmeans.n_clusters

# def load_pca():
#      with h5py.File(SETTINGS.DATASET, 'r') as f:
#         pca = f['PCA']
#         components_ = pca["components_"][:]
#         mean_ = pca["mean_"][:]
#         explained_variance_ = pca["explained_variance_"][:]
#         explained_variance_ratio_ = pca["explained_variance_ratio_"][:]
#         n_components = pca.attrs["n_components"]

#         # Reconstruct the PCA object
#         pca_loaded = PCA(n_components=n_components)
#         pca_loaded.components_ = components_
#         pca_loaded.mean_ = mean_
#         pca_loaded.explained_variance_ = explained_variance_
#         pca_loaded.explained_variance_ratio_ = explained_variance_ratio_
#         return pca_loaded
     
# def load_kmeans():
#     with h5py.File(SETTINGS.DATASET, 'r') as f:
#         kmeans = f['KMeans']
#         cluster_centers_ = kmeans["cluster_centers_"][:]
#         n_clusters = kmeans.attrs["n_clusters"]

#         kmeans_loaded = KMeans(n_clusters=n_clusters)
#         kmeans_loaded.cluster_centers_ = cluster_centers_

#         return kmeans_loaded

# # def view_shape_modes():
# #     kmeans = load_kmeans()
# #     pca = load_pca()

# #     plt.rcParams["font.family"] = 'serif'
# #     fig, axs = plt.subplots(1, SETTINGS.KMEANS_CLUSTERS)
# #     colours = ['blue', 'orange', 'green', 'red']
# #     for i, cluster in enumerate(kmeans.cluster_centers_):
# #         original_contour = pca.inverse_transform(cluster)
# #         original_contour = np.reshape(original_contour, (-1, 2))
# #         axs[i].plot(original_contour[:, 0], original_contour[:, 1], color=colours[i])
# #         axs[i].set_aspect('equal')
# #         axs[i].set_title(str(i))

# #     plt.show()

# def view_shape_modes():
#     kmeans = load_kmeans()
#     pca = load_pca()

#     plt.rcParams["font.family"] = 'serif'
#     # fig, axs = plt.subplots(1, SETTINGS.KMEANS_CLUSTERS)
#     colours = ['blue', 'orange', 'green', 'red']
#     for i, cluster in enumerate(kmeans.cluster_centers_):
#         original_contour = pca.inverse_transform(cluster)
#         original_contour = np.reshape(original_contour, (-1, 2))
#         plt.plot(original_contour[:, 0], original_contour[:, 1], color=colours[i])
#         # plt.set_aspect('equal')
#         # axs[i].set_title(str(i))

#     plt.show()
    

# def view_clusters():
#     plt.rcParams["font.family"] = 'serif'
#     with h5py.File(SETTINGS.DATASET, 'r') as f:
#         clusters = f['KMeans']['train_labels'][:]
#         pcs = f['PCA']['train_pcs'][:]

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         colours = ['blue', 'orange', 'green', 'red']
#         for cluster in range(SETTINGS.KMEANS_CLUSTERS):
#             idxs = np.nonzero(clusters==cluster)
#             ax.scatter(pcs[idxs, 0], pcs[idxs, 1], pcs[idxs, 2], label=str(cluster), color=colours[cluster])
#     plt.legend()
#     plt.show()

# def apply_pca_kmeans(frame):
#     contours = []
#     cell_idxs = []
#     with h5py.File(SETTINGS.DATASET, 'r') as f:
#         mask = f['Segmentations']['Phase'][f'{frame:04}'][:]
#         raw_contours = mask_funcs.get_border_representation(mask, f['Cells']['Phase'][:].shape[1], f, frame)
#         for i, c in enumerate(raw_contours):
#             new_contour = PreProcessing(c).pre_process()
#             if len(new_contour) > 0:
#                 contours.append(new_contour.flatten())
#                 cell_idxs.append(i)

#         contours = np.array(contours)

#         pca = load_pca()
#         kmeans = load_kmeans()

#         pcs = pca.transform(contours)
#         cluster_distances = kmeans.transform(pcs)

#         full_cluster_distances = np.full((f['Cells']['Phase'][:].shape[1], SETTINGS.KMEANS_CLUSTERS), np.nan)
#         full_cluster_distances[cell_idxs] = cluster_distances
        
#         return full_cluster_distances
        

if __name__ == '__main__':
    # fit_pca_kmeans()
    view_shape_modes()
    view_clusters()
    # apply_pca_kmeans(40)