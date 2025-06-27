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
        self.num_training_images = SETTINGS.NUM_TRAINING_FRAMES
        self.num_pca_componeents = SETTINGS.PCA_COMPONENTS
        self.num_kmeans_clusters = SETTINGS.KMEANS_CLUSTERS

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

    def feature_func(self, frame_idx):
        with h5py.File(self.h5py_file, 'r') as f:
            mask = f['Segmentations']['Phase'][frame_idx][:]
            num_cells = f['Cells']['Phase']['X'].shape[1]

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
                results[i] = PreProcessing(contour).pre_process().flatten()
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
        # fig, axs = plt.subplots(1, SETTINGS.KMEANS_CLUSTERS)
        colours = ['blue', 'orange', 'green', 'red']
        for i, cluster in enumerate(self.kmeans.cluster_centers_):
            original_contour = self.pca.inverse_transform(cluster)
            original_contour = np.reshape(original_contour, (-1, 2))
            plt.plot(original_contour[:, 0], original_contour[:, 1], color=colours[i])
            # plt.set_aspect('equal')
            # axs[i].set_title(str(i))
        plt.savefig('temp/shape_modes.png')
        plt.show()
        
def main():
    test = MorphologyFit()
    test.view_shape_modes()
    test.view_clusters()
    # test.fit()
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
   