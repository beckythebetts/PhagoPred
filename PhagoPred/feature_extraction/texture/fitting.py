import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from PhagoPred.feature_extraction.pca_kmeans_fitting import PCAKmeansFit
from PhagoPred.feature_extraction.texture import generate_example_texture
from PhagoPred.utils import mask_funcs
from PhagoPred import SETTINGS

class TextureFit(PCAKmeansFit):

    def __init__(self, h5py_file=SETTINGS.DATASET):
        super().__init__(h5py_file)
        self.h5py_group = 'Texture'
        self.num_training_images = SETTINGS.NUM_TRAINING_FRAMES
        self.num_pca_componeents = 10
        self.num_kmeans_clusters = 4
        self.haralick_texture_distances = [1, 2, 3, 4, 5]

        self.haralick_features_names = [
            'Angular Second Moment',     # aka Energy or Uniformity
            'Contrast',
            'Correlation',
            'Sum of Squares: Variance',
            'Inverse Difference Moment', # aka Homogeneity
            'Sum Average',
            'Sum Variance',
            'Sum Entropy',
            'Entropy',
            'Difference Variance',
            'Difference Entropy',
            'Information Measure of Correlation 1',
            'Information Measure of Correlation 2'
        ]
        self.feature_names = np.array([[f'{feature}_{distance}' for feature in self.haralick_features_names] 
                              for distance in self.haralick_texture_distances])

    def feature_func(self, frame):
    
        with h5py.File(self.h5py_file, 'r') as f:
            results = np.full((f['Cells']['Phase'][:].shape[1], len(self.haralick_texture_distances)*13), np.nan)
                              
            mask = f['Segmentations']['Phase'][f'{frame:04}'][:]
            image = f['Images']['Phase'][f'{frame:04}']
            for cell_idx in np.unique(mask[mask>0]):
                results[cell_idx] = mask_funcs.get_haralick_texture_features(image, mask==cell_idx).flatten()
        return results

    def view_texture_modes(self):
        if self.pca is None:
            self.load_pca()
        if self.kmeans is None:
            self.load_kmeans()

        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(1, self.num_kmeans_clusters)
        # colours = ['blue', 'orange', 'green', 'red']
        for i, cluster in enumerate(self.kmeans.cluster_centers_):
            sys.stdout.write(f'{i} / {self.num_kmeans_clusters}')
            sys.stdout.flush()
            _ = self.pca.inverse_transform(cluster)
            original_haralick_features = np.reshape(_, (len(self.haralick_texture_distances), -1))
            axs[i].matshow(generate_example_texture.example_texture(original_haralick_features), cmap='gray')

        plt.show()

    def rank_input_features(self):
        cmap = plt.get_cmap('brg')
        colours = {har_feat: cmap(i/len(self.haralick_features_names)) for i, har_feat in enumerate(self.haralick_features_names)}
        return super().rank_input_features(colours)

    

    

if __name__ == '__main__':
    test = TextureFit()
    # test.fit()
    # test.load_pca()
    # test.load_kmeans()
    # test.view_texture_modes()
    # test.view_clusters()
    # test.pc_heatmap()
    test.rank_input_features()
