import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import umap.umap_ as umap
import os
import matplotlib.pyplot as plt
import itertools

from PhagoPred.feature_extraction.morphology.pre_processing import PreProcessing, generalised_procrustes_analysis
from PhagoPred.utils import mask_funcs, tools
from PhagoPred import SETTINGS

plt.rcParams["font.family"] = 'serif'

class UMAP_embedding:
    def __init__(self):
        
        self.mean_contour = None
        self.num_training_images = SETTINGS.NUM_TRAINING_FRAMES
        self.num_contour_points = SETTINGS.NUM_CONTOUR_POINTS
        self.umap_model = None
        self.training_contours = np.empty((0, self.num_contour_points*2))
        self.training_metadata = {'File': [], 'Frame': [], 'Idx': []}
    
    def get_contours_from_mask(self, mask:np.ndarray, num_cells) -> list:
        """Get a list of contours from a segmentation mask.
        """
        # num_cells = np.max(mask)+1
        expanded_mask = (np.expand_dims(mask, 0) == np.expand_dims(np.arange(num_cells), (1,2)))
        results = np.full((expanded_mask.shape[0], self.num_contour_points*2), np.nan)

        contours = mask_funcs.get_border_representation(expanded_mask)

        for i, contour in enumerate(contours):
            if len(contour) > 2:
                results[i] = PreProcessing(contour, self.mean_contour).pre_process().flatten()
          
        return results
    
    def get_training_contours(self, hdf5_files: list):
        if isinstance(hdf5_files, Path):
            hdf5_files = [hdf5_files]
            
        training_ims_per_file = self.num_training_images // len(hdf5_files)
        
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, 'r+') as f:
                
                seg_dataset = f['Segmentations']['Phase']
                training_frames = np.random.choice(np.arange(0, seg_dataset.shape[0]), size=training_ims_per_file, replace=False)
                cell_death_ds = f['Cells']['Phase']['CellDeath'][0]
                num_cells = len(cell_death_ds)
                
                for frame in tqdm(training_frames, desc=f'Getting sample outlines from {hdf5_file.name}:'):
                    # self.training_contours.append(self.get_contours_from_mask(seg_dataset[frame]))
                    new_contours = self.get_contours_from_mask(seg_dataset[frame], num_cells)
                    
                    # For contours corresponding to dead cells, remove
                    alive_cells = np.nonzero((cell_death_ds > frame) | np.isnan(cell_death_ds))[0]

                    # Select contours for alive cells only
                    new_contours = new_contours[alive_cells]

                    # Filter out invalid contours (NaNs)
                    valid_mask = ~np.isnan(new_contours).any(axis=1)
                    new_contours = new_contours[valid_mask]
                    alive_cells = alive_cells[valid_mask]
                    
                    self.training_metadata['File'].extend([hdf5_file]*len(alive_cells))
                    self.training_metadata['Frame'].extend([frame]*len(alive_cells))
                    self.training_metadata['Idx'].extend(alive_cells.tolist())
                    
                    # new_contours = new_contours[alive_cells]
                    self.training_contours = np.concatenate((self.training_contours, new_contours))
                    
        # self.training_contours = self.training_contours[~np.isnan(self.training_contours).any(axis=1)]
        for key, value in self.training_metadata.items():
            self.training_metadata[key] = np.array(value)
            
        self.training_contours, self.mean_contour = generalised_procrustes_analysis(self.training_contours, max_iter=20, tol=1e-7)
        
        
    def fit(self, save_as: Path, umap_hyperparameters: dict = {}) -> None:
        """Fit the UMAP embedding based on a {self.num_training_images} random sample of frames from hdf5_file / s."""
        
        self.umap_model = umap.UMAP(**umap_hyperparameters)
        
        self.training_embeddings = self.umap_model.fit_transform(self.training_contours)
        
        with open(save_as, 'wb') as f:
            pickle.dump(self, f)
    
    def get_cell_metric(self, metric: str) -> np.ndarray:
        """Extract the specified cell metric from HDF5 datasets in order matching the embeddings."""

        metric_values = []

        current_file = None
        f = None  # current open HDF5 file object

        try:
            for file, frame, idx in zip(self.training_metadata['File'],
                                        self.training_metadata['Frame'],
                                        self.training_metadata['Idx']):

                # Only open a file if it changes
                if current_file != file:
                    if f:
                        f.close()
                    f = h5py.File(file, 'r')
                    current_file = file

                value = f['Cells']['Phase'][metric][frame][idx]
                metric_values.append(value)

        finally:
            if f:
                f.close()

        return np.array(metric_values)

            
    def plot_training_embedding(self, save_as:Path = None, metric: str = None):
        """Plot the training embeddings.
        OPtionally colour code according to metric."""
        fig, ax = plt.subplots(figsize=(10, 8))
        if metric is not None:
            metric_values = self.get_cell_metric(metric)
            print(min(metric_values), max(metric_values))

            scatter = ax.scatter(self.training_embeddings[:, 0],
                                self.training_embeddings[:, 1],
                                c=metric_values,
                                cmap='viridis',
                                s=20, alpha=0.7)

            ax.set_title(f'UMAP embedding coloured by {metric}')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(metric)
            
        else:
            ax.scatter(self.training_embeddings[:, 0],
                    self.training_embeddings[:, 1],
                    color='gray',
                    s=20, alpha=0.3,
                    label='UMAP embeddings')
            ax.set_title('UMAP embedding')

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        if save_as is not None:
            fig.savefig(save_as)

        return fig, ax

    def plot_contour_at_xy(self, ax, contour, x, y, scale=0.1, colour='black', linestyle='-', alpha=1, label=''):
        # contour = contour.reshape(-1, 2)
        contour = contour * scale
        contour[:, 0] += x
        contour[:, 1] += y
        ax.plot(contour[:, 0], contour[:, 1], color=colour, linewidth=1, linestyle=linestyle, alpha=alpha, label=label)
        
    def plot_sample_contours(self, save_as: Path):

        fig, ax = self.plot_training_embedding()

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
                
                bin_width = x_bins[i+1] - x_bins[i]
                bin_height = y_bins[j+1] - y_bins[j]

                # Choose scale based on smaller bin dimension, with some padding (e.g. 80%)
                scale = 0.4 * min(bin_width, bin_height)
                
                # Average PCA scores in bin
                # mean = np.mean(np.array(self.features)[in_bin], axis=0)
                bin_features = np.array(self.training_contours)[in_bin]
                colours = ('b', 'k', 'r')
                for percentile, colour in zip((5, 50, 95), colours):
                    percentile_features = np.percentile(bin_features, percentile, axis=0)
                    percentile_features = self.postprocess_contour(percentile_features)
                    alpha = 1 if percentile==50 else 0.5
                    self.plot_contour_at_xy(ax, percentile_features, x_centre, y_centre, scale=scale, colour=colour, alpha=alpha, label=f'{percentile}th pecentile contour in region')
        
        ax.set_title('UMAP embedding with example shape overlays')
                
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        fig.savefig(save_as)
        
        return fig, ax
    
    def postprocess_contour(self, raw):
        """Reshape, center, normalize a contour."""
        contour = raw.reshape(-1, 2)
        
        # Center
        contour -= contour.mean(axis=0)
        
        # Normalize scale (RMS radius)
        R = np.sqrt(np.mean(contour[:, 0]**2 + contour[:, 1]**2))
        if R > 0:
            contour /= R
        
        return contour
    
    def plot_inverse_embeddings(self, save_as: Path, rotate:bool=False) -> None:
        fig, ax = self.plot_training_embedding()
        
        num_bins_x = 10
        num_bins_y = 10
        
        x_bins = np.linspace(np.min(self.training_embeddings[:, 0]), np.max(self.training_embeddings[:, 0]), num_bins_x + 1)
        y_bins = np.linspace(np.min(self.training_embeddings[:, 1]), np.max(self.training_embeddings[:, 1]), num_bins_y + 1)
        
        x_centres = 0.5* (x_bins[:-1] + x_bins[1:])
        y_centres = 0.5 * (y_bins[:-1] + y_bins[1:])
        
        xx, yy = np.meshgrid(x_centres, y_centres)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        inverse_embeddings = self.umap_model.inverse_transform(coords)
        

        
        for i in range(num_bins_x):
            for j in range(num_bins_y):
                # Check if training data points fall inside this bin
                in_bin = np.where(
                    (self.training_embeddings[:, 0] >= x_bins[i]) & (self.training_embeddings[:, 0] < x_bins[i + 1]) &
                    (self.training_embeddings[:, 1] >= y_bins[j]) & (self.training_embeddings[:, 1] < y_bins[j + 1])
                )[0]

                if len(in_bin) == 0:
                    continue  # Skip bins with no data

                x_centre = x_centres[i]
                y_centre = y_centres[j]
                coord = np.array([[x_centre, y_centre]])

                # inverse_embedding = self.umap_model.inverse_transform(coord)[0]
                inverse_embedding = inverse_embeddings[j*num_bins_x + i]
                contour = self.postprocess_contour(inverse_embedding)
                if rotate:
                    contour = -contour
                        # Calculate bin dimensions
                bin_width = x_bins[i+1] - x_bins[i]
                bin_height = y_bins[j+1] - y_bins[j]

                # Choose scale based on smaller bin dimension, with some padding (e.g. 80%)
                scale = 0.4 * min(bin_width, bin_height)

                self.plot_contour_at_xy(ax, contour, x_centre, y_centre, scale=scale, colour='black')


        ax.set_title("Inverse-mapped contours across UMAP space")
        fig.savefig(save_as)
        
        return fig, ax
    
    def plot_trajectory(self, hdf5_file: Path, cell_idx: int):
        with h5py.File(hdf5_file, 'r') as f:
            first_frame, last_frame = f['Cells']['Phase']['First Frame'][cell_idx], f['Cells']['Phase']['Last Frame'][cell_idx]
            trajectory = np.empty(shape=(last_frame - first_frame, 2))
            for i, frame in enumerate(first_frame, last_frame+1):
                mask = f['Segmentations']['Phase'][frame][:] == cell_idx
                if mask.any():
                    contour = self.get_contours_from_mask(mask, 1)[0]
                    trajectory[i] = self.umap_model.transform(contour)[0]
        
        fig, axs = self.plot_training_embedding()
        

def hyperparamter_search(savedir: Path):
    
    # umap_embedding = UMAP_embedding()
    # umap_embedding.get_training_contours([Path('PhagoPred') / 'Datasets' / '13_06.h5',
    #                    Path('PhagoPred') / 'Datasets' / '16_09_3.h5'])
    param_grid = {
        'n_neighbors': [5, 10, 30, 50],
        'min_dist': [0.01, 0.1, 0.3, 0.5],
        'metric': ['euclidean', 'cosine', 'correlation']
    }
    
    # tools.remake_dir(savedir)
    
    for n_neighbors, min_dist, metric in tqdm(list(itertools.product(param_grid['n_neighbors'],
                                                          param_grid['min_dist'],
                                                          param_grid['metric']))):
        params = {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'random_state': 42,
        }
        print(f"Testing params: {params}")
        # param_dir = tools.remake_dir(savedir / f'{n_neighbors}_{min_dist}_{metric}')
        param_dir = savedir / f'{n_neighbors}_{min_dist}_{metric}'
        
        with open(param_dir / 'model.pickle', 'rb') as f:
            umap_embedding = pickle.load(f)
        
        if params['metric'] == 'cosine':
            rotate=True
        else:
            rotate = False
        umap_embedding.plot_inverse_embeddings(save_as = param_dir / 'inverse_transforms.png', rotate=rotate)
        # umap_embedding.fit(save_as = param_dir / 'model.pickle', umap_hyperparameters = params)
        # umap_embedding.plot_sample_contours(save_as = param_dir / 'sample_contours.png')
        # umap_embedding.plot_inverse_embeddings(save_as = param_dir / 'inverse_transforms.png')
        
    
def main():
    # hyperparamter_search(Path('PhagoPred') / 'feature_extraction' / 'morphology' / 'UMAP_model_search')
    # umap_embedding = UMAP_embedding()
    # umap_embedding.get_training_contours([Path('PhagoPred') / 'Datasets' / '13_06.h5',
    #                                       Path('PhagoPred') / 'Datasets' / '16_09_1.h5'])
    # umap_embedding.fit(Path('PhagoPred') / 'feature_extraction' / 'morphology' / 'UMAP_model.pickle', {'metric': 'correlation'})
    with open(Path(Path('PhagoPred') / 'feature_extraction' / 'morphology' / 'UMAP_model.pickle'), 'rb') as f:
        umap_embedding = pickle.load(f)
    umap_embedding.plot_training_embedding(Path('temp') / 'UMAP_embedding_plot_circ.png', 'Circularity')
    # umap_embedding.plot_training_embedding(Path('temp') / 'UMAP_embedding_plot_area.png', 'Area')
    # umap_embedding.plot_training_embedding(Path('temp') / 'UMAP_embedding_plot_per.png', 'Perimeter')
    # umap_embedding.plot_training_embedding(Path('temp') / 'UMAP_embedding_plot_spe.png', 'Speed')
    # umap_embedding.plot_sample_contours(Path('temp') / 'UMAP_embedding_plot.png')
    # umap_embedding.plot_inverse_embeddings(Path('temp') / 'UMAP_embedding_plot_inverse.png')
    
if __name__ == '__main__':
    main()
    