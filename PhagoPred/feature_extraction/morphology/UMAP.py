import random

import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import umap.umap_ as umap
# from umap.parametric_umap import ParametricUMAP
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import itertools
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
from scipy.spatial import procrustes
import numba
from numba import jit, prange

from PhagoPred.feature_extraction.morphology.pre_processing import PreProcessing, generalised_procrustes_analysis
from PhagoPred.utils import mask_funcs, tools
from PhagoPred import SETTINGS

plt.rcParams["font.family"] = 'serif'

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from scipy.spatial import procrustes
from contextlib import contextmanager
import joblib

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()
        
@numba.njit()
def procrustes_align(a, b, scaling=True):
    """
    Perform Procrustes alignment of shape a to shape b.

    Both a and b are (N, 2) arrays.
    Returns aligned a, disparity.
    """
    # Center shapes
    a_mean = numba_mean(a)
    b_mean = numba_mean(b)
    a_centered = a - a_mean
    b_centered = b - b_mean

    # Compute covariance matrix
    # C = a_centered.T @ b_centered
    C = np.zeros((2, 2))
    for i in range(a.shape[0]):
        for j in range(2):
            for k in range(2):
                C[j, k] += a_centered[i, j] * b_centered[i, k]

    # SVD decomposition
    U, s, Vt = np.linalg.svd(C)

    # Rotation matrix
    R = Vt.T @ U.T

    # Reflection correction (determinant)
    det = np.linalg.det(R)
    if det < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    # Scale factor
    if scaling:
        norm_a = 0.0
        for i in range(a_centered.shape[0]):
            for j in range(2):
                norm_a += a_centered[i, j] ** 2
        scale = np.sum(s) / norm_a
    else:
        scale = 1.0

    # Align a to b
    a_aligned = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(2):
            a_aligned[i, j] = scale * (R[j, 0] * a_centered[i, 0] + R[j, 1] * a_centered[i, 1]) + b_mean[j]

    # Compute disparity (sum of squared distances)
    disparity = 0.0
    for i in range(a.shape[0]):
        dx = a_aligned[i, 0] - b[i, 0]
        dy = a_aligned[i, 1] - b[i, 1]
        disparity += dx * dx + dy * dy
    disparity /= a.shape[0]

    return a_aligned, disparity

@numba.njit()
def numba_mean(arr):
    """Compute mean along axis 0 for 2D array."""
    n = arr.shape[0]
    sum0 = 0.0
    sum1 = 0.0
    for i in range(n):
        sum0 += arr[i, 0]
        sum1 += arr[i, 1]
    return np.array([sum0 / n, sum1 / n])

@numba.njit()
def procrustes_metric(a_flat, b_flat):
    # reshape to (N_points, 2)
    N = a_flat.shape[0] // 2
    a = a_flat.reshape((N, 2))
    b = b_flat.reshape((N, 2))

    _, disparity = procrustes_align(a, b, scaling=True)
    return disparity


class UMAP_embedding:
    def __init__(self):
        
        self.mean_contour = None
        self.num_training_images = SETTINGS.NUM_TRAINING_FRAMES
        self.num_contour_points = SETTINGS.NUM_CONTOUR_POINTS
        self.umap_model = None
        self.training_contours = np.empty((0, self.num_contour_points*2))
        self.training_metadata = {'File': [], 'Frame': [], 'Idx': []}
        self.save_as = Path('PhagoPred') / 'feature_extraction' / 'morphology' / 'UMAP_model.pickle'
        
    def apply(self, mask: np.ndarray) -> None:
        """Transfrom an expanded masks contour to UMAP embedding."""
        contours = self.get_contours_from_mask(mask, expanded = True)
        self.plot_processed_contour(contours=contours, save_as=(Path('temp/processed_contour.png')))
        all_embeddings = np.full((len(contours), 2), np.nan)
        valid_mask = ~np.isnan(contours).any(axis=1)
        contours = contours[valid_mask]
        if contours.shape[0] > 0:
            embeddings = self.umap_model.transform(contours)
            all_embeddings[valid_mask] = embeddings
        return all_embeddings
    
    def get_contours_from_mask(self, mask:np.ndarray, num_cells: int = None, expanded: bool = False) -> list:
        """Get a list of contours from a segmentation mask.
        """
        num_cells = np.max(mask)+1 if num_cells is None else num_cells
        if not expanded:
            expanded_mask = (np.expand_dims(mask, 0) == np.expand_dims(np.arange(num_cells), (1,2)))
        else:
            expanded_mask = mask
        results = np.full((expanded_mask.shape[0], self.num_contour_points*2), np.nan)

        contours = mask_funcs.get_border_representation(expanded_mask)

        for i, contour in enumerate(contours):
            if len(contour) > 3:
                results[i] = PreProcessing(contour, self.mean_contour).pre_process(smoothing=20).flatten()
          
        return results
    
    def plot_processed_contour(self, contours: np.ndarray, save_as: Path) -> None:
        processed = contours[0]
        contour_xy = processed.reshape(-1, 2)

        # Center for visualization
        centered = contour_xy - np.mean(contour_xy, axis=0)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(centered[:, 0], centered[:, 1], 'k-')
        ax.set_aspect('equal')
        ax.axis('off')
        fig.savefig(save_as)
        plt.close(fig)
        
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
            
        self.pickle_embedding()
    
    def filter_training_contours(self):
        # filter out highly circular shapes
        circularities = self.get_cell_metric('Circularity')
        mask = circularities < 0.5
        # print(self.training_contours.shape)
        self.training_contours = np.array(self.training_contours)[mask]
        self.training_metadata['File'] = np.array(self.training_metadata['File'])[mask]
        self.training_metadata['Frame'] = np.array(self.training_metadata['Frame'])[mask]
        self.training_metadata['Idx'] = np.array(self.training_metadata['Idx'])[mask]
        
        self.training_contours, self.mean_contour = generalised_procrustes_analysis(self.training_contours, max_iter=20, tol=1e-7)
        

        
        
    def fit(self, save_as: Path, umap_hyperparameters: dict = {}) -> None:
        """Fit the UMAP embedding based on a {self.num_training_images} random sample of frames from hdf5_file / s."""
        
        @numba.njit
        def contour_distance(x, y):
            """
            Compute the Procrustes distance between two 2D contours.
            x and y should be (N_points, 2) arrays.
            """
            x = x.reshape(-1, 2)
            y = y.reshape(-1, 2)
            
            diff = x - y
            return np.sqrt(np.sum(diff ** 2))
        
        
        # circularities = self.get_cell_metric('Circularity')
        # circularity_bins = pd.qcut(circularities, q=2, labels=False)
        self.umap_model = umap.UMAP(**umap_hyperparameters)
        # self.umap_model = ParametricUMAP()
        
        self.training_embeddings = self.umap_model.fit_transform(self.training_contours
            # self.procrustes_matrix, 
            #                                                     verbose=True
                                                                 )
        
        # with open(save_as, 'wb') as f:
        #     pickle.dump(self, f)
        self.pickle_embedding()
            
    def pickle_embedding(self):
        with open(self.save_as, 'wb') as f:
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
                    color='lightblue',
                    s=20, alpha=0.3,
                    label='UMAP embeddings')
            ax.set_title('UMAP embedding')

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_aspect('equal')

        if save_as is not None:
            fig.savefig(save_as)

        return fig, ax
    
    def plot_training_embedding_3d(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.training_embeddings[:, 0], self.training_embeddings[:, 1], self.training_embeddings[:, 2], c='black', s=5, alpha=0.5)
        ax.scatter(*self.training_embeddings.T, color='lightgray', s=3)
        ax.set_title("3D UMAP Embedding of Contours")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        plt.tight_layout()
        # plt.show()
        
        return fig, ax
    
    def plot_sample_contours_3d(self, num_bins=10, min_points_per_bin=1):
        """
        Plot representative contours in 3D UMAP space by dividing it into bins,
        and displaying the contour closest to the center of each non-empty bin.
        """

        # fig = plt.figure(figsize=(12, 10))
        # ax = fig.add_subplot(111, projection='3d')
        fig, ax = self.plot_training_embedding_3d()
        
        embeddings = self.training_embeddings  # shape (N, 3)
        contours = self.training_contours      # shape (N, P)

        # Create 3D bin edges
        x_bins = np.linspace(np.min(embeddings[:, 0]), np.max(embeddings[:, 0]), num_bins + 1)
        y_bins = np.linspace(np.min(embeddings[:, 1]), np.max(embeddings[:, 1]), num_bins + 1)
        z_bins = np.linspace(np.min(embeddings[:, 2]), np.max(embeddings[:, 2]), num_bins + 1)

        for i in range(num_bins):
            for j in range(num_bins):
                for k in range(num_bins):
                    # Select indices of points in this bin
                    in_bin = np.where(
                        (embeddings[:, 0] >= x_bins[i]) & (embeddings[:, 0] < x_bins[i + 1]) &
                        (embeddings[:, 1] >= y_bins[j]) & (embeddings[:, 1] < y_bins[j + 1]) &
                        (embeddings[:, 2] >= z_bins[k]) & (embeddings[:, 2] < z_bins[k + 1])
                    )[0]

                    if len(in_bin) < min_points_per_bin:
                        continue  # skip sparse bins

                    # Compute bin center
                    x_c = (x_bins[i] + x_bins[i + 1]) / 2
                    y_c = (y_bins[j] + y_bins[j + 1]) / 2
                    z_c = (z_bins[k] + z_bins[k + 1]) / 2
                    bin_center = np.array([x_c, y_c, z_c])

                    # Find closest embedding to the bin center
                    bin_embeddings = embeddings[in_bin]
                    dists = np.linalg.norm(bin_embeddings - bin_center, axis=1)
                    closest_idx = in_bin[np.argmin(dists)]

                    # Get and post-process the corresponding contour
                    contour = self.postprocess_contour(contours[closest_idx])  # shape: (num_points, 2)

                    # Project into XY plane and offset at (x_c, y_c, z_c)
                    scale = 0.3 * min(
                        x_bins[i + 1] - x_bins[i],
                        y_bins[j + 1] - y_bins[j],
                        z_bins[k + 1] - z_bins[k],
                    )

                    xs = x_c + contour[:, 0] * scale
                    ys = y_c + contour[:, 1] * scale
                    zs = np.full_like(xs, z_c)

                    # Close the contour
                    xs = np.append(xs, xs[0])
                    ys = np.append(ys, ys[0])
                    zs = np.append(zs, zs[0])

                    segments = [[(xs[m], ys[m], zs[m]), (xs[m+1], ys[m+1], zs[m+1])] for m in range(len(xs)-1)]
                    lc = Line3DCollection(segments, colors='black', linewidths=1.0, alpha=0.8)
                    ax.add_collection3d(lc)

        # Optional: scatter the actual points in background for context
        # ax.scatter(*embeddings.T, color='lightgray', s=3, alpha=0.2)

        ax.set_title('3D UMAP Embedding with Sample Contours')
        # ax.set_xlabel('UMAP-1')
        # ax.set_ylabel('UMAP-2')
        # ax.set_zlabel('UMAP-3')
        ax.view_init(elev=30, azim=45)

        plt.show()
        
        fig.tight_layout()
        # fig.savefig(save_as)
        plt.close(fig)
        
    


    def plot_contour_at_xy(self, ax, contour, x, y, scale=0.1, colour='black', linestyle='-', alpha=1, label=''):
        # contour = contour.reshape(-1, 2)
        contour = contour - np.mean(contour, axis=0)
        contour = contour * scale
        contour[:, 0] += x
        contour[:, 1] += y
        ax.plot(contour[:, 0], contour[:, 1], color=colour, linewidth=1, linestyle=linestyle, alpha=alpha, label=label)
        
    def plot_sample_contours(self, save_as: Path):

        fig, ax = self.plot_training_embedding()

        num_bins_x = 15
        num_bins_y = 15
        
        num_samples_per_bin = 10

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
                bin_center = np.array([x_centre, y_centre])
                
                bin_width = x_bins[i+1] - x_bins[i]
                bin_height = y_bins[j+1] - y_bins[j]

                # Choose scale based on smaller bin dimension, with some padding 
                scale = 0.3 * min(bin_width, bin_height)
                
                # Pot 5th, 50th and 95th percentile contour values
                # mean = np.mean(np.array(self.features)[in_bin], axis=0)
                
                # bin_features = np.array(self.training_contours)[in_bin]
                # colours = ('b', 'k', 'r')
                # for percentile, colour in zip((0, 50, 100), colours):
                #     percentile_features = np.percentile(bin_features, percentile, axis=0)
                #     percentile_features = self.postprocess_contour(percentile_features)
                #     alpha = 1 if percentile==50 else 0.5
                #     self.plot_contour_at_xy(ax, percentile_features, x_centre, y_centre, scale=scale, colour=colour, alpha=alpha, label=f'{percentile}th pecentile contour in region')
                
                 # Randomly sample N contours from this bin
                # sample_indices = random.sample(list(in_bin), min(num_samples_per_bin, len(in_bin)))
                # for k, idx in enumerate(sample_indices):
                #     contour = self.training_contours[idx]
                #     contour = self.postprocess_contour(contour)

                #     colour = plt.cm.hsv(k / num_samples_per_bin)  # Use colormap for variation
                #     alpha = 1.0 if num_samples_per_bin == 1 else 0.7

                #     self.plot_contour_at_xy(ax, contour, x_centre, y_centre, scale=scale, colour=colour, alpha=alpha,
                #                             label=f'Contour sample' if (i == 0 and j == 0 and k == 0) else '')
                    
                # Plot most central contour in bin
                # Find contour closest to center
                bin_embeddings = self.training_embeddings[in_bin]
                dists = np.linalg.norm(bin_embeddings - bin_center, axis=1)
                closest_idx = in_bin[np.argmin(dists)]

                # Plot the corresponding contour
                contour = self.training_contours[closest_idx]
                contour = self.postprocess_contour(contour)
                self.plot_contour_at_xy(ax, contour, x_centre, y_centre, scale=scale, colour='black', alpha=0.9)
        
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
    
    def plot_trajectory(self, hdf5_file: Path, cell_idx: int, save_as: Path):
        with h5py.File(hdf5_file, 'r') as f:
            first_frame, last_frame = f['Cells']['Phase']['First Frame'][0, cell_idx], f['Cells']['Phase']['Last Frame'][0, cell_idx]
            first_frame, last_frame = int(first_frame), int(last_frame)
            # first_frame, last_frame = 568, 573
            trajectory = np.full(shape=(last_frame - first_frame + 1, 2), fill_value=np.nan)
            for i, frame in enumerate(range(first_frame, last_frame+1)):
                mask = f['Segmentations']['Phase'][frame][:] == cell_idx
                if mask.any():
                    trajectory[i] = self.apply(mask[np.newaxis])
                    # contour = self.get_contours_from_mask(mask, 1)[0]
                    # print(contour)
                    # trajectory[i] = self.umap_model.transform(contour)[0]
        
        print(trajectory.shape)
            # Create figure and axes
        fig, axs = self.plot_training_embedding()

        # Generate time values (frame indices) for the colormap
        times = np.arange(first_frame, last_frame + 1)

        # Only keep valid (non-NaN) segments
        valid = ~np.isnan(trajectory).any(axis=1)
        valid_traj = trajectory[valid]
        valid_times = times[valid]

        # Build line segments between consecutive points
        segments = []
        colors = []

        for i in range(len(valid_traj) - 1):
            # Only add if both points are consecutive in time
            if valid_times[i + 1] - valid_times[i] == 1:
                seg = [valid_traj[i], valid_traj[i + 1]]
                segments.append(seg)
                colors.append(valid_times[i])

        if segments:
            # Create line collection with color mapping
            lc = LineCollection(segments, cmap='viridis', norm=Normalize(times[0], times[-1]))
            lc.set_array(np.array(colors))
            lc.set_linewidth(2)
            axs.add_collection(lc)
            fig.colorbar(lc, ax=axs, label='Frame')

        axs.autoscale()
        axs.set_aspect('equal')
        fig.savefig(save_as)
        plt.close(fig)
        
    def compute_spearman_correlation(self, subset_size=None, random_state=42, metric='euclidean'):
        """
        Compute Spearman rank correlation between pairwise distances
        in contour space and UMAP embedding space.
        If subset_size is given, compute on a random subset of data points.
        """
        n_samples = len(self.training_contours)
        self.training_contours = np.array(self.training_contours)
        self.training_embeddings = np.array(self.training_embeddings)
        if subset_size is not None and subset_size < n_samples:
            rng = np.random.default_rng(random_state)
            subset_indices = rng.choice(n_samples, size=subset_size, replace=False)
            
            # Subset the contours and embeddings
            contours_subset = self.training_contours[subset_indices]
            embeddings_subset = self.training_embeddings[subset_indices]
        else:
            contours_subset = self.training_contours
            embeddings_subset = self.training_embeddings

        # Pairwise distances in original contour space
        dist_contours = pairwise_distances(contours_subset, metric=metric)

        # Pairwise distances in UMAP embedding space
        dist_embeddings = pairwise_distances(embeddings_subset, metric='euclidean')

        # Flatten upper triangles (excluding diagonal) to 1D vectors for correlation
        triu_indices = np.triu_indices_from(dist_contours, k=1)

        dist_contours_flat = dist_contours[triu_indices]
        dist_embeddings_flat = dist_embeddings[triu_indices]

        # Spearman rank correlation
        corr, p_value = spearmanr(dist_contours_flat, dist_embeddings_flat)

        return corr, p_value     
    
    def get_procrustes_distances(self):
        """Calculate pairwise procrustes disatcne between all training contours."""
        self.training_contours = np.array(self.training_contours)
        N, dim = self.training_contours.shape
        assert dim % 2 == 0, "Each shape must have an even number of elements (x/y pairs)."
        P = dim // 2

        dist_matrix = np.zeros((N, N), dtype=np.float32)

        for i in tqdm(range(N), desc="Computing Procrustes distances"):
            shape_i = self.training_contours[i].reshape(P, 2)
            for j in range(i + 1, N):
                shape_j = self.training_contours[j].reshape(P, 2)
                _, _, disparity = procrustes(shape_i, shape_j)
                dist_matrix[i, j] = disparity
                dist_matrix[j, i] = disparity  # symmetry

        self.procrustes_matrix = dist_matrix
        
        self.pickle_embedding()
        


#     def get_procrustes_distances(self, n_jobs=-1):
#         """Calculate pairwise Procrustes distances in parallel."""
#         self.training_contours = np.array(self.training_contours)
#         N, dim = self.training_contours.shape
#         assert dim % 2 == 0, "Each shape must have an even number of elements (x/y pairs)."
#         P = dim // 2

#         # Generate list of unique i, j pairs (upper triangle)
#         index_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

#         # Parallel processing
#         with tqdm_joblib(tqdm(desc="Procrustes distances", total=len(index_pairs))) as progress_bar:
#             results = Parallel(n_jobs=n_jobs)(
#                 delayed(compute_procrustes_distance_pair)(i, j, self.training_contours, P)
#                 for i, j in tqdm(index_pairs, desc="Computing Procrustes distances")
#             )

#         # Fill distance matrix
#         dist_matrix = np.zeros((N, N), dtype=np.float32)
#         for i, j, dist in results:
#             dist_matrix[i, j] = dist
#             dist_matrix[j, i] = dist  # symmetric

#         self.procrustes_matrix = dist_matrix
#         self.pickle_embedding()

# def compute_procrustes_distance_pair(i, j, contours, P):
#     shape_i = contours[i].reshape(P, 2)
#     shape_j = contours[j].reshape(P, 2)
#     _, _, disparity = procrustes(shape_i, shape_j)
#     return (i, j, disparity)      

def hyperparamter_search(savedir: Path, umap_embedding: UMAP_embedding):
    
    # umap_embedding = UMAP_embedding()
    # umap_embedding.get_training_contours([Path('PhagoPred') / 'Datasets' / '13_06.h5',
    #                    Path('PhagoPred') / 'Datasets' / '16_09_3.h5'])
    param_grid = {
        'n_neighbors': [5, 50, 250, 500],
        'min_dist': [0.0, 0.1, 0.3, 0.5],
        'metric': ['euclidean', 'cosine', 'correlation']
    }
    
    # tools.remake_dir(savedir)
    results = []
    
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
        
        try:
            with open(param_dir / 'model.pickle', 'rb') as f:
                umap_embedding = pickle.load(f)
        except:
            continue
        
        if params['metric'] == 'cosine':
            rotate=True
        else:
            rotate = False
        # umap_embedding.plot_inverse_embeddings(save_as = param_dir / 'inverse_transforms.png', rotate=rotate)
        # umap_embedding.fit(save_as = param_dir / 'model.pickle', umap_hyperparameters = params)
        # umap_embedding.plot_sample_contours(save_as = param_dir / 'sample_contours.png')
        # umap_embedding.plot_inverse_embeddings(save_as = param_dir / 'inverse_transforms.png', rotate=rotate)
        corr, p = umap_embedding.compute_spearman_correlation(subset_size=100, metric=metric)
        results.append({
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'spearman_corr': corr,
            'spearman_p_value': p,
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(savedir / 'hyperparameter_spearman_results.csv', index=False)
    
    best_row = results_df.loc[results_df['spearman_corr'].idxmax()]
    print("Best hyperparameters based on Spearman correlation:")
    print(f"n_neighbors: {best_row['n_neighbors']}")
    print(f"min_dist: {best_row['min_dist']}")
    print(f"metric: {best_row['metric']}")
    print(f"Spearman correlation: {best_row['spearman_corr']:.4f}")
    print(f"p-value: {best_row['spearman_p_value']:.4g}")

def plot_spearman_heatmaps(csv_path):
    # Load results
    df = pd.read_csv(csv_path)

    # Filter for just 'euclidean' and 'cosine' metrics
    metrics_to_plot = ['euclidean', 'cosine', 'correlation']
    df = df[df['metric'].isin(metrics_to_plot)]

    # Prepare figure with subplots for each metric
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 6), sharey=True)

    if len(metrics_to_plot) == 1:
        axes = [axes]  # Make it iterable if only one subplot

    for ax, metric in zip(axes, metrics_to_plot):
        sub_df = df[df['metric'] == metric]

        # Pivot table: rows = n_neighbors, columns = min_dist, values = abs(spearman_corr)
        heatmap_data = sub_df.pivot(index='n_neighbors', columns='min_dist', values='spearman_corr')
        import seaborn as sns
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap='viridis',
            ax=ax,
            cbar=True
        )
        ax.set_title(f"Spearman Corr - {metric}")
        ax.set_xlabel("min_dist")
        ax.set_ylabel("n_neighbors")

    plt.tight_layout()
    # plt.show()
    plt.savefig(Path('temp') / 'speamrnas.png')

def load_UMAP(umap_path: Path) -> UMAP_embedding:
    # from PhagoPred.feature_extraction.morphology.UMAP import UMAP_embedding
    with open(umap_path, 'rb') as f:
        return pickle.load(f)   
    
def main():
    
    # # hyperparamter_search(Path('PhagoPred') / 'feature_extraction' / 'morphology' / 'UMAP_model_search')
    # umap_embedding = UMAP_embedding()
    # umap_embedding.get_training_contours([
    #     #       Path('PhagoPred') / 'Datasets' / 'Prelims' / '16_09_3.h5',
    # #                                       Path('PhagoPred') / 'Datasets' / 'Prelims' / '16_09_1.h5',
    #                                       Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
    #                                     #   Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
    #                                     #   Path('PhagoPred') / 'Datasets' / 'ExposureTest' / 'old' / '03_10_2500.h5'
    #                                     ])
    with open(Path(Path('PhagoPred') / 'feature_extraction' / 'morphology' / 'UMAP_model.pickle'), 'rb') as f:
        umap_embedding = pickle.load(f)
    
    # umap_embedding.filter_training_contours()
    # umap_embedding.get_procrustes_distances()
    # umap_embedding.fit(Path('PhagoPred') / 'feature_extraction' / 'morphology' / 'UMAP_model.pickle', {'metric': 'euclidean', 'n_neighbors': 50, 'min_dist': 0.0, 'n_components': 2})
    
    # hyperparamter_search(savedir=Path('/home/ubuntu/PhagoPred/PhagoPred/feature_extraction/morphology/UMAP_model_search'), umap_embedding=umap_embedding)
    # plot_spearman_heatmaps('/home/ubuntu/PhagoPred/PhagoPred/feature_extraction/morphology/UMAP_model_search/hyperparameter_spearman_results.csv')

    # umap_embedding.smooth_embeddings()
    
    # umap_embedding.plot_training_embedding_3d()
    # umap_embedding.plot_sample_contours_3d()
    umap_embedding.plot_training_embedding(Path('temp') / 'UMAP_embedding_plot_skel.png', 'Skeleton Branch Length Max')
    # umap_embedding.plot_training_embedding(Path('temp') / 'UMAP_embedding_plot_area.png', 'Area')
    # umap_embedding.plot_training_embedding(Path('temp') / 'UMAP_embedding_plot_per.png', 'Perimeter')
    # umap_embedding.plot_training_embedding(Path('temp') / 'UMAP_embedding_plot_spe.png', 'Speed')
    # umap_embedding.plot_sample_contours(Path('temp') / 'UMAP_embedding_plot.png')
    # umap_embedding.plot_inverse_embeddings(Path('temp') / 'UMAP_embedding_plot_inverse.png')
    # umap_embedding.plot_trajectory(Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5', 956, Path('temp') / 'UMAP_trajectory.png')
    # print(umap_embedding.compute_spearman_correlation(subset_size=100))
    print(len(umap_embedding.training_contours))
if __name__ == '__main__':
    main()
    