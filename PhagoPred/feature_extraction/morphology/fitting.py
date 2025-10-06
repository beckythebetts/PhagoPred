from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

from PhagoPred.feature_extraction.morphology.pre_processing import PreProcessing, generalised_procrustes_analysis
from PhagoPred.utils import mask_funcs
from PhagoPred.feature_extraction.pca_kmeans_fitting import PCAKmeansFit
from PhagoPred import SETTINGS

## TESTING
class MorphologyFit(PCAKmeansFit):

    def __init__(self, h5py_file=SETTINGS.DATASET):
        super().__init__(h5py_file)
        self.h5py_group = 'Morphology'
        self.num_training_images = SETTINGS.NUM_TRAINING_FRAMES
        self.num_pca_componeents = SETTINGS.PCA_COMPONENTS
        self.num_kmeans_clusters = SETTINGS.KMEANS_CLUSTERS
        
        self.mean_contour = None

    # def feature_func(self, frame):
    #     """
    #     Gets contours of cells for given frame index
    #     """
    #     # results = []
        
    #     with h5py.File(self.h5py_file, 'r') as f:
    #         results = np.full((f['Cells']['Phase'][:].shape[1], SETTINGS.NUM_CONTOUR_POINTS*2), np.nan)

    #         mask = f['Segmentations']['Phase'][f'{frame:04}'][:]
    #         contours = mask_funcs.get_border_representation(mask, f['Cells']['Phase'][:].shape[1], f, frame)

    #         for i, contour in enumerate(contours):
    #             if len(contour) > 2:
    #                 # results.append(PreProcessing(contour).pre_process().flatten())
    #                 results[i] = PreProcessing(contour).pre_process().flatten()
    #     return np.array(results)

    def training_preprocess(self, features):
        features, mean_feature = generalised_procrustes_analysis(features)
        self.mean_contour = mean_feature
        return features
    
    def feature_func(self, frame_idx):
        with h5py.File(self.h5py_file, 'r') as f:
            mask = f['Segmentations']['Phase'][frame_idx][:]
            num_cells = f['Cells']['Phase']['X'].shape[1]
            circularities = f['Cells']['Phase']['Circularity'][frame_idx][:]

        expanded_mask = (np.expand_dims(mask, 0) == np.expand_dims(np.arange(num_cells), (1,2)))

        return self.feature_func_expanded_mask(expanded_mask)
    

    def feature_func_expanded_mask(self, expanded_mask):
        """
        Gets contours of cells for expanded frame mask dims [num_cells, x, y]
        """
        results = np.full((expanded_mask.shape[0], SETTINGS.NUM_CONTOUR_POINTS*2), np.nan)

        contours = mask_funcs.get_border_representation(expanded_mask)

        for i, contour in enumerate(contours):
            if len(contour) > 2:
                # results.append(PreProcessing(contour).pre_process().flatten())
                results[i] = PreProcessing(contour, self.mean_contour).pre_process().flatten()
                
        return np.array(results)

    # def feature_func(self, frame):
    #     with h5py.File(self.h5py_file, 'r') as f:
    #         mask = f['Segmentations']['Phase'][f'{frame:04}'][:]
    #         contours = mask_funcs.get_border_representation(mask, f['Cells']['Phase'][:].shape[1], f, frame)
        
    #     return contours
    
    def view_shape_modes(self):
        if self.pca is None:
            self.load_pca()
        if self.kmeans is None:
            self.load_kmeans()

        plt.rcParams["font.family"] = 'serif'
        # colours = ['blue', 'orange', 'green', 'red']
        cmap = plt.get_cmap('rainbow')
        num_clusters = len(self.kmeans.cluster_centers_)
        colours = [cmap(i / (num_clusters - 1)) for i in range(num_clusters)]
        for i, cluster in enumerate(self.kmeans.cluster_centers_):
            original_contour = self.pca.inverse_transform(cluster.reshape(1, -1))
            original_contour = np.reshape(original_contour, (-1, 2))

            plt.plot(original_contour[:, 0], original_contour[:, 1], color=colours[i])

        plt.savefig('temp/shape_modes.png')
        plt.show()
        
    def view_shape_modes_sep(self):
        plt.rcParams["font.family"] = 'serif'
        cmap = plt.get_cmap('Set1')
        num_clusters = len(self.kmeans.cluster_centers_)
        colours = [cmap(i) for i in range(num_clusters)]

        # Determine grid size (rows and cols) for subplots
        cols = math.ceil(math.sqrt(num_clusters))
        rows = math.ceil(num_clusters / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axs = np.array(axs).reshape(-1)  # flatten axs array for easy iteration

        for i, cluster in enumerate(self.kmeans.cluster_centers_):
            original_contour = self.pca.inverse_transform(cluster.reshape(1, -1))
            original_contour = np.reshape(original_contour, (-1, 2))

            axs[i].plot(original_contour[:, 0], original_contour[:, 1], color=colours[i])
            axs[i].set_title(f'Cluster {i}')
            axs[i].axis('equal')
            axs[i].axis('off')  # optional: hide axis ticks for cleaner look

        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.savefig('temp/shape_modes_separate.png')
        plt.show()

def main():
    test = MorphologyFit()
   
    test.fit()
    test.umap_embedding()
    test.view_shape_modes_sep()
    test.view_clusters()
    # test.pc_heatmap()
    # test.plot_explained_variance_ratio()
    # test.load_pca()
    # test.load_kmeans()
    # test.view_shape_modes()
    # fit_pca_kmeans()
    # view_shape_modes()
    # view_clusters()
    # apply_pca_kmeans(40)
    # test.pc_heatmap()
    # test.rank_input_features()


if __name__ == '__main__':
    main()
   