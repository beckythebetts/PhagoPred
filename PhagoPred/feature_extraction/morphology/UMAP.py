import h5py

from PhagoPred.feature_extraction.morphology import pre_processing
from PhagoPred import SETTINGS

class UMAP_embedding:
    def __init__(self):
        self.mean_contour = None
        self.num_training_images = SETTINGS.NUM_TRAINING_FRAMES
        
    def fit(self, hdf5_file=SETTINGS.DATASET) -> None:
        """Fit the UMAP embedding based on a {self.num_training_images} random sample of frames from hdf5_file."""
        with h5py.File
        total_num_frames = 
        self.training_frames = np.random.choice(np.arange(0, SETTINGS.NUM_FRAMES), size=self.num_training_images, replace=False)