import h5py
import numpy as np

from PhagoPred.feature_extraction.pca_kmeans_fitting import PCAKmeansFit
from PhagoPred.utils import mask_funcs
from PhagoPred import SETTINGS

class TextureFit(PCAKmeansFit):

    def __init__(self, h5py_file=SETTINGS.DATASET):
        super().__init__(h5py_file)
        self.h5py_group = 'Texture'
        self.num_training_images = 50
        self.num_pca_componeents = 10
        self.num_kmeans_clusters = 4
        self.haralick_texture_distances = [1, 2, 3, 4, 5]

    def feature_func(self, frame):
    
        with h5py.File(self.h5py_file, 'r') as f:
            results = np.full((f['Cells']['Phase'][:].shape[1], len(self.haralick_texture_distances), 13), np.nan)
                              
            mask = f['Segmentations']['Phase'][f'{frame:04}'][:]
            image = f['Images']['Phase'][f'{frame:04}']
            for cell_idx in np.unique(mask[mask>0]):
                results[cell_idx] = mask_funcs.get_haralick_texture_features(image, mask==cell_idx)
        return results
    

if __name__ == '__main__':
    test = TextureFit()
    test.fit()
