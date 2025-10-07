import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import umap.umap_ as umap
import matplotlib.pyplot as plt

from PhagoPred.feature_extraction.morphology.pre_processing import PreProcessing, generalised_procrustes_analysis
from PhagoPred.utils import mask_funcs
from PhagoPred import SETTINGS

class UMAP_embedding:
    def __init__(self):
        self.training_contours = []
        self.mean_contour = None
        self.num_training_images = SETTINGS.NUM_TRAINING_FRAMES
        self.num_contour_points = SETTINGS.NUM_CONTOUR_POINTS
        self.embedding = None
    
    def get_contours_from_mask(self, mask:np.ndarray) -> list:
        """Get a list of contours from a segmentation mask."""
        num_cells = np.max(mask)+1
        expanded_mask = (np.expand_dims(mask, 0) == np.expand_dims(np.arange(num_cells), (1,2)))
        results = np.full((expanded_mask.shape[0], self.num_contour_points*2), np.nan)

        contours = mask_funcs.get_border_representation(expanded_mask)

        for i, contour in enumerate(contours):
            if len(contour) > 2:
                # results.append(PreProcessing(contour).pre_process().flatten())
                results[i] = PreProcessing(contour, self.mean_contour).pre_process().flatten()
                
        return results
    
    def fit(self, hdf5_files:list | Path = [SETTINGS.DATASET], save_as: Path) -> None:
        """Fit the UMAP embedding based on a {self.num_training_images} random sample of frames from hdf5_file / s."""
        
        if isinstance(hdf5_files, Path):
            hdf5_files = [hdf5_files]
            
        training_ims_per_file = self.num_training_images // len(hdf5_files)
        
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, 'r+') as f:
                seg_dataset = f['Segmentations']['Phase']
                training_frames = np.random.choice(np.arange(0, seg_dataset.shape[0], size=training_ims_per_file, replace=False))
                for frame in tqdm(training_frames, desc=f'Getting sample outlines from {hdf5_file.name}:'):
                    self.training_contours.append(self.get_contours_from_mask(seg_dataset[frame]))
        
        self.training_contours, self.mean_contour = generalised_procrustes_analysis(self.training_contours, max_iter=20, tol=1e-7)
        
        self.umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        
        self.training_embeddings = self.embedding.fit_transform(self.training_contours)
        
        with open(save_as, 'wb') as f:
            pickle.dump(self, f)
            
    def plot_training(self, save_as:Path):
        """Plot the training embedding as well as example contours in each region."""
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.training_embeddings[:, 0], self.training_embeddings[:, 1], c='gray', s=20, alpha=0.1, label='UMAP embeddings')
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

        x_bins = np.linspace(np.min(self.training_embeddings[:, 0]), np.max(self.training_embeddings[:, 0]), num_bins_x + 1)
        y_bins = np.linspace(np.min(self.training_embeddings[:, 1]), np.max(self.training_embeddings[:, 1]), num_bins_y + 1)

        for i in range(num_bins_x):
            for j in range(num_bins_y):
                # Find points in current bin
                in_bin = np.where(
                    (self.training_embeddings[:, 0] >= x_bins[i]) & (self.training_embeddings[:, 0] < x_bins[i + 1]) &
                    (self.training_embeddings[:, 1] >= y_bins[j]) & (self.training_embeddings[:, 1] < y_bins[j + 1])
                )[0]

                if len(in_bin) == 0:
                    continue  # skip empty bins
                
                  # Compute bin center for plotting
                x_centre = (x_bins[i] + x_bins[i + 1]) / 2
                y_centre = (y_bins[j] + y_bins[j + 1]) / 2
                
                # Average PCA scores in bin
                # mean = np.mean(np.array(self.features)[in_bin], axis=0)
                bin_features = np.array(self.training_contours)[in_bin]
                colours = ('b', 'k', 'r')
                for percentile, colour in zip((5, 50, 95), colours):
                    percentile_features = np.percentile(bin_features, percentile, axis=0)
                    alpha = 1 if percentile==50 else 0.5
                    plot_contour_at_xy(ax, percentile_features, x_centre, y_centre, scale=5, colour=colour, alpha=alpha, label=f'{percentile}th pecentile contour in region')
        
                plt.title('UMAP embedding with example shape overlays')
                
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.axis('equal')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.savefig(save_as)
        plt.show()
        
        
def main():
    umap_embedding = UMAP_embedding()
    umap_embedding.fit(Path('PhagoPred') / 'Datasets' / '?', Path('PhagoPred') / 'feature_extraction' / 'morphology' / 'UMAP_model.pickle')
    umap_embedding.plot_training(Path('temp') / 'UMAP_embedding_plot.png')