import xarray as xr
import torch
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, box
from scipy.signal import convolve
from scipy import ndimage
from skimage.morphology import skeletonize
import pickle
from skimage.measure import label, regionprops
import time
import kornia
import matplotlib.pyplot as plt

from PhagoPred.feature_extraction.morphology.fitting import MorphologyFit
from PhagoPred.feature_extraction.morphology.UMAP import load_UMAP, UMAP_embedding
from PhagoPred.feature_extraction.texture.gabor import Gabor
from PhagoPred import SETTINGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseFeature:

    primary_feature = False
    derived_feature = False
    crop = False

    def __init__(self):
        self.name = [self.__class__.__name__]
        # self.index_positions = None

    def get_names(self):
        return self.name
    
    def compute(self):
        """
        returns:
            np.array [num_frames, num_cells, num_features]
        """
        raise NotImplementedError(f"{self.get_name()} has no compute method implemented")
    
    # def set_index_positions(self, start: int, end: int = None) -> None:
    #     self.index_positions = (start, end)
    
    # def get_index_positions(self) -> list[int]:
    #     return self.index_positions
    
class CellDeath(BaseFeature):
    """
    Determine frame at which cell dies, or np.nan if no cell death.
    Note: unlike all other cell features, this returns a single value for each cell (not each frame of each cell).
    
    Use no smoothing for classification (as )
    """
    derived_feature = True

    def __init__(self, threshold: float = 0.85, min_frames_dead: int = 5, smoothed_frames: float = 5):
        super().__init__()
        self.threshold = threshold
        # self.min_frames_dead = min_frames_dead
        self.smoothed_frames = smoothed_frames

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        dead = phase_xr['Dead Macrophage']
        alive = phase_xr['Macrophage']
        
        cell_state = xr.full_like(alive, np.nan, dtype=float)
        
        is_alive = alive.fillna(0) > 0.5
        is_dead  = dead.fillna(0) > 0.5

        cell_state = xr.where(is_alive, 1.0, cell_state)
        cell_state = xr.where(is_dead, 0.0, cell_state)

        # cell_state = cell_state.where(alive < 0.5, 1)
        # cell_state = cell_state.where(dead > 0.5, 0)

        # smooth
        smoothed_cell_state = cell_state.rolling(Frame=self.smoothed_frames, center=True).reduce(np.nanmean)
        # cell_state = cell_state.rolling(1, center=True).reduce(np.nanmean)

        # need to interpolate, as nan values will be seen as alive when thresholding
        smoothed_cell_state = smoothed_cell_state.chunk({'Frame': -1})
        smoothed_cell_state = smoothed_cell_state.interpolate_na(dim='Frame', method="linear", fill_value="extrapolate").values
        
        cell_state = cell_state.chunk({'Frame': -1})
        cell_state = cell_state.interpolate_na(dim='Frame', method="linear", fill_value="extrapolate").values
        # smoothed_cell_state = np.interp(smoothed_cell_state, )
        
        
        dead_frames = smoothed_cell_state < self.threshold
        # print(np.unique(smoothed_cell_state))
        reversed_dead_frames = dead_frames[::-1]
    
        reversed_dead_frames_cum_min = np.minimum.accumulate(reversed_dead_frames, axis=0)
        permanently_dead_frames = reversed_dead_frames_cum_min[::-1]
        
        death_frames = np.argmax(permanently_dead_frames, axis=0).astype(float) # take index of first permanently dead frame 
        death_frames[np.all(permanently_dead_frames==0, axis=0)] = np.nan

        return death_frames[np.newaxis, :, np.newaxis]

class FirstLastFrame(BaseFeature):
    """Computes frame of each cells first and last appearances."""
    derived_feature = True

    def get_names(self):
        return ['First Frame', 'Last Frame']

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        # Use areas dataset to find nan frames
        areas = phase_xr['Area']

        not_nan_mask = ~np.isnan(areas)
        start = not_nan_mask.argmax(axis=0)
        reversed_mask = not_nan_mask[::-1, :]
        last = areas.shape[0] - 1 - reversed_mask.argmax(axis=0)

        return np.stack([start, last], axis=1)[np.newaxis, :, :]


class Coords(BaseFeature):
    """
    Centroid x, y coordinates and area
    """
    primary_feature = True

    def __init__(self):
        self.x_grid, self.y_grid = None, None

    def get_names(self):
        return ['X', 'Y', 'Area']
    
    def compute(self, mask: torch.tensor, image: torch.tensor, epi_image: torch.tensor, binary_mask: torch.tensor) -> np.array:
        if self.x_grid is None or self.y_grid is None:
            self.x_grid, self.y_grid = torch.meshgrid(
                torch.arange(mask.shape[1]).to(device),
                torch.arange(mask.shape[2]).to(device),
                indexing='ij')
            self.x_grid = self.x_grid.unsqueeze(0)
            self.y_grid = self.y_grid.unsqueeze(0)
            # for grid in (self.x_grid, self.y_grid): grid = grid.unsqueeze(0) 

        areas = torch.sum(mask, dim=(1, 2))
        xs = torch.sum(self.x_grid * mask, dim=(1, 2)) / areas
        ys = torch.sum(self.y_grid * mask, dim=(1, 2)) / areas

        # Set values for cells with area 0 to nan
        results = torch.stack((xs, ys, areas), dim=-1)
        nan_mask = results[:, -1] == 0.0
        nan_mask = nan_mask.unsqueeze(-1).expand_as(results)
        results[nan_mask] = float('nan')
        
        return results.cpu().numpy()
    
# class MorphologyModes(BaseFeature):

#     primary_feature = True

#     def __init__(self):
#         super().__init__()
#         self.morphology_model = MorphologyFit(SETTINGS.DATASET)

#         try:
#             self.morphology_model.load_model()

#         except:
#             print('Fitting Morpgology Model')
#             self.morphology_model.fit()


#     def get_names(self):
#         return [f'Mode {i}' for i in range(self.morphology_model.num_kmeans_clusters)]
    
#     def compute(self, mask: torch.tensor, image: torch.tensor, epi_image: torch.tensor) -> np.array:
#         """
#         Results are of dims [num_cells, num_clusters]
#         """
#         expanded_frame_mask = mask.cpu().numpy()
#         results = self.morphology_model.apply_expanded_mask(expanded_frame_mask)
#         # print(results, results.shape)
#         return results
    
# class UmapEmbedding(BaseFeature):
    
#     primary_feature = True
    
#     def __init__(self):
#         super().__init__()
#         self.umap_embedding = load_UMAP(SETTINGS.UMAP_MODEL)
#         # with open(SETTINGS.UMAP_MODEL, 'rb') as f:
#         #     self.umap_embedding = pickle.load(f)
    
#     def get_names(self):
#         return ['UMAP 1', 'UMAP 2']
    
#     def compute(self, mask: torch.tensor, image: torch.tensor, epi_image: torch.tensor) -> np.ndarray:
#         """
#         return [num_cells, num_features]"""
#         expanded_frame_mask = mask.cpu().numpy()
#         embeddings = self.umap_embedding.apply(expanded_frame_mask)
#         return embeddings
        
class Skeleton(BaseFeature):
    
    primary_feature = True
    crop = True
    
    def get_names(self):
        return [
            'Skeleton Length', 
            'Skeleton Branch Points', 
            'Skeleton End Points', 
            'Skeleton Branch Length Mean', 
            'Skeleton Branch Length Std',
            'Skeleton Branch Length Max'
        ]
    
    def compute(self, mask: torch.tensor, image: torch.tensor, epi_image: torch.tensor, binary_mask: torch.tensor) -> np.ndarray:
        """
        return [num_cells, num_features]
        """
        expanded_mask = mask.cpu().numpy()
        feature_names = self.get_names()
        
        results = {name: [] for name in feature_names}
        
        kernel = np.array([[1,1,1],
                           [1,10,1],
                           [1,1,1]])
        
        n = expanded_mask.shape[0]
        
        for i, cell_mask in enumerate(expanded_mask, 1):
            if not cell_mask.any():
                for feature in results.keys():
                    results[feature].append(np.nan)
                continue

            coords = np.argwhere(cell_mask)
            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)
            cell_mask = cell_mask[min_row:max_row+1, min_col:max_col+1]
            cell_mask = np.ascontiguousarray(cell_mask.astype(bool))
            
            skeleton = skeletonize(cell_mask)
            
            length = np.sum(skeleton)

            neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
            
            branch_points = np.sum((neighbor_count >= 13) & skeleton)
            end_points = np.sum((neighbor_count == 11) & skeleton)
            
            labeled_skel = label(skeleton, connectivity=2)
            
            regions = regionprops(labeled_skel)
            branch_lengths = np.array([r.area for r in regions])
            
            branch_len_mean = branch_lengths.mean() if branch_lengths.size > 0 else np.nan
            branch_len_std = branch_lengths.std() if branch_lengths.size > 0 else np.nan
            branch_len_max = branch_lengths.max() if branch_lengths.size > 0 else np.nan
            
            results['Skeleton Length'].append(length)
            results['Skeleton Branch Points'].append(branch_points)
            results['Skeleton End Points'].append(end_points)
            results['Skeleton Branch Length Mean'].append(branch_len_mean)
            results['Skeleton Branch Length Std'].append(branch_len_std)
            results['Skeleton Branch Length Max'].append(branch_len_max)
        
        return np.stack([results[feature] for feature in feature_names], axis=1)
    
class Speed(BaseFeature):

    derived_feature = True
    
    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        # positions = phase_xr.sel(Feature=['x', 'y'])
        positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')

        speeds = positions.diff(dim='Frame')

        speeds = (speeds**2).sum(dim='Feature', keepdims=True)**0.5

        # speeds = speeds.expand_dims(dim={'Feature':1}, axis=-1)

        #insert np.nan at beginning
        nan_insert = xr.DataArray([0], dims='Frame')
        nan_insert = xr.full_like(speeds.isel(Frame=0), np.nan).expand_dims(Frame=1)
        speeds = xr.concat([nan_insert, speeds], dim='Frame')

        #insert nan if current or peviosu position unknown
        valid_positions = positions.notnull().all(dim='Feature')
        valid_positions_prev = valid_positions.shift(Frame=1).fillna(False)
        speeds = speeds.where(valid_positions & valid_positions_prev)

        speeds = speeds.transpose('Frame', 'Cell Index', 'Feature')
        return speeds.values
    
class Displacement(BaseFeature):

    derived_feature = True

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')

        not_nan = positions.notnull().all(dim='Feature')
        first_valid_frame = not_nan.argmax(dim='Frame').compute()

        positions_0 = positions.isel(Frame=first_valid_frame).expand_dims({'Frame': positions.sizes['Frame']})

        # positions = phase_xr.sel(Feature=['x', 'y'])

        # positions_0 = positions.isel(Frame=0).expand_dims({'Frame': positions.sizes['Frame']})

        displacements = ((positions-positions_0)**2).sum(dim='Feature')**0.5

        # mask = positions.sel(Feature='x').notnull()
        mask = phase_xr['X'].notnull()

        displacements = displacements.where(mask)

        displacements = displacements.expand_dims({'Feature':1}, axis=-1)

        displacements = displacements.transpose('Frame', 'Cell Index', 'Feature')
        return displacements.values

    
# class DensityPhase(BaseFeature):

#     derived_feature = True

#     radii = [100, 250, 500]

#     frame_batch_size = 10

#     def get_names(self):
#         return [f'{state} Phagocytes within {radius} pixels' for radius in self.radii for state in ['Alive', 'Dead']]
    
#     def circle_fraction_in_frame(self, x, y, r, H, W):
#         """Calculate the fraction of a circle that is within the frame."""
#         circle = Point(x, y).buffer(r)
#         frame = box(0, 0, W, H)
#         intersection = circle.intersection(frame)
#         return intersection.area / (np.pi * r * r)
    
#     def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
#         """Compute number of other phase cells within self.radii pixels (centroid based).
#         Manual batching becuase of probelms with O(n^2) batching with Dask arrays
#         """
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')
#         positions = positions.transpose('Frame', 'Cell Index', 'Feature').chunk({})

#         results = np.full((positions.sizes['Frame'], positions.sizes['Cell Index'], len(self.radii)), np.nan)

#         H, W = SETTINGS.IMAGE_SIZE

#         for first_frame in tqdm(range(0, positions.sizes['Frame'], self.frame_batch_size)):
#             last_frame = min(first_frame + self.frame_batch_size, positions.sizes['Frame'])
            
#             batch_positions_np = positions.isel(Frame=slice(first_frame, last_frame)).values
#             batch_positions = torch.tensor(batch_positions_np, device=device)


#             distances = batch_positions.unsqueeze(2) - batch_positions.unsqueeze(1)
#             distances = torch.norm(distances, dim=3)
#             distances[distances == 0] = float('nan')
#             for radius_idx, radius in enumerate(self.radii):
#                 counts = (distances < radius).sum(dim=2).cpu().numpy()

#                 xs = batch_positions_np[..., 0]
#                 ys = batch_positions_np[..., 1]
#                 fraction_in = np.ones_like(xs)

#                 # Only correct for cells near the edge
#                 edge_mask = (
#                     (xs - radius < 0) | (xs + radius > W) |
#                     (ys - radius < 0) | (ys + radius > H)
#                 ) & ~np.isnan(xs) & ~np.isnan(ys)

#                 if np.any(edge_mask):
#                     # Vectorized geometric correction using shapely
#                     edge_indices = np.argwhere(edge_mask)
#                     for idx in edge_indices:
#                         i, j = idx
#                         x, y = xs[i, j], ys[i, j]
#                         fraction_in[i, j] = self.circle_fraction_in_frame(x, y, radius, H, W)

#                 # Avoid division by zero or overcorrection
#                 fraction_in = np.clip(fraction_in, 1e-3, 1.0)
#                 corrected = counts / fraction_in
#                 results[first_frame:last_frame, :, radius_idx] = corrected

#         results = np.where(np.isnan(phase_xr['X'].values)[:, :, np.newaxis], np.nan, results)

#         return results

class DensityPhase(BaseFeature):

    derived_feature = True
    crop = True
    
    radii = [100, 250, 500]
    frame_batch_size = 10

    def get_names(self):
        return [
            f'{state} Phagocytes within {radius} pixels'
            for radius in self.radii
            for state in ['Alive', 'Dead']
        ]

    def circle_fraction_in_frame(self, x, y, r, H, W):
        circle = Point(x, y).buffer(r)
        frame = box(0, 0, W, H)
        intersection = circle.intersection(frame)
        return intersection.area / (np.pi * r * r)

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray):
        """
        phase_xr contains:
          - X, Y   (Frame × Cell)
          - CellDeath[0] (Cell,) = death frame or NaN
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        positions = xr.concat(
            [phase_xr[feat] for feat in ['X', 'Y']],
            dim='Feature'
        ).transpose('Frame', 'Cell Index', 'Feature').chunk({})

        n_frames = positions.sizes['Frame']
        n_cells = positions.sizes['Cell Index']
        n_radii = len(self.radii)

        death_frame = phase_xr['CellDeath'][0].values

        frame_idx = np.arange(n_frames)[:, None]
        death_expanded = np.broadcast_to(death_frame, (n_frames, n_cells))

        is_dead = (~np.isnan(death_expanded)) & (death_expanded <= frame_idx)
        is_alive = ~is_dead   # includes NaN → alive

        results = np.full((n_frames, n_cells, 2 * n_radii), np.nan)

        H, W = SETTINGS.IMAGE_SIZE

        for first_frame in tqdm(range(0, n_frames, self.frame_batch_size)):
            last_frame = min(first_frame + self.frame_batch_size, n_frames)

            batch_positions_np = positions.isel(Frame=slice(first_frame, last_frame)).values
            batch_positions = torch.tensor(batch_positions_np, device=device)

            distances = batch_positions.unsqueeze(2) - batch_positions.unsqueeze(1)
            distances = torch.norm(distances, dim=3)
            distances[distances == 0] = float('nan')

            alive_mask = torch.tensor(is_alive[first_frame:last_frame], device=device)
            dead_mask = torch.tensor(is_dead[first_frame:last_frame], device=device)

            for r_idx, radius in enumerate(self.radii):

                neighbors = (distances < radius)  # boolean adjacency

                alive_counts = (neighbors & alive_mask.unsqueeze(1)).sum(dim=2).cpu().numpy()

                dead_counts = (neighbors & dead_mask.unsqueeze(1)).sum(dim=2).cpu().numpy()

                xs = batch_positions_np[..., 0]
                ys = batch_positions_np[..., 1]

                fraction_in = np.ones_like(xs)

                edge_mask = (
                    (xs - radius < 0) | (xs + radius > W) |
                    (ys - radius < 0) | (ys + radius > H)
                ) & ~np.isnan(xs) & ~np.isnan(ys)

                if np.any(edge_mask):
                    for i, j in np.argwhere(edge_mask):
                        fraction_in[i, j] = self.circle_fraction_in_frame(
                            xs[i, j], ys[i, j], radius, H, W
                        )

                fraction_in = np.clip(fraction_in, 1e-3, 1.0)

                alive_corrected = alive_counts / fraction_in
                dead_corrected = dead_counts / fraction_in

                results[first_frame:last_frame, :, 2 * r_idx] = alive_corrected
                results[first_frame:last_frame, :, 2 * r_idx + 1] = dead_corrected

        valid_mask = ~np.isnan(phase_xr['X'].values)
        results = np.where(valid_mask[:, :, None], results, np.nan)

        return results
   
class Perimeter(BaseFeature):

    primary_feature = True
    crop = True

    def compute(self, mask: torch.tensor, image: torch.tensor, epi_image: torch.tensor, binary_mask: torch.tensor) -> np.array:

        if mask.shape[1] < 3 or mask.shape[2] < 3:
            return np.full((mask.shape[0],), np.nan)
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        kernel = torch.tensor(
            [[1, 1, 1],
             [1, 9, 1],
             [1, 1, 1]]
             ).to(mask.device)
        
        padded_masks = torch.nn.functional.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
                                                padding=0).squeeze(1)
        
        result = torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2)).float().cpu().numpy()
        result[result==0] = np.nan

        return result
        
class Circularity(BaseFeature):
    """
    C = 4*pi*(Area) / (Perimeter)**2
    """

    derived_feature = True

    def compute(self, phase_xr: xr.Dataset, epi_xr: xr.Dataset) -> np.array:
        areas = phase_xr['Area']
        perimeters = phase_xr['Perimeter']
        circularity = areas * 4 * np.pi / (perimeters**2)
        return circularity.values
    
# class GaborScale(BaseFeature):

#     primary_feature = True

#     def __init__(self):
#         super().__init__()
#         self.gabor = Gabor(gpu=True)

#     def compute(self, mask: torch.tensor, image: torch.tensor, epi_image: torch.tensor) -> np.array:
#         dominant_scales = self.gabor.get_dominant_scales_batch(image, mask)
#         return np.array(dominant_scales)

class ExternalFluorescence(BaseFeature):
    """Quantify intensity of fluorescence at different distances away from cells."""
    primary_feature = True
    distances = [10, 25, 50]
    crop= True

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernels = {d: self._circular_kernel(d) for d in self.distances}
    
    def _circular_kernel(self, radius: int) -> torch.Tensor:
        """Create a circular kernel for a given radius."""
        size = 2 * radius + 1
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        center = radius
        mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
        kernel = mask.float().to(self.device)  # float32 for conv2d
        return kernel.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    def get_names(self):
        return [f'External Fluorescence Intensity within {dist} pixels' for dist in self.distances]

    def compute(self, mask: torch.Tensor = None,
                image: torch.Tensor = None,
                epi_image: torch.Tensor = None,
                binary_mask: torch.Tensor = None) -> np.ndarray:

        num_cells, H, W = mask.shape
        results = torch.full((num_cells, len(self.distances)),
                             float("nan"), device=self.device)

        valid = torch.sum(mask, dim=(1, 2)) > 0
        if not valid.any():
            return results.cpu().numpy()

        temp_mask = mask.unsqueeze(1).float()  # (C,1,H,W)
        # epi = (epi_image * binary_mask).float()  # (C,H,W)

        for i, distance in enumerate(self.distances):
            kernel = self.kernels[distance]
            
            padding = distance
            
            dilated_mask = torch.nn.functional.conv2d(temp_mask, kernel, padding=padding).squeeze(1)
            
            dilated_mask = dilated_mask > 0
            dilated_mask = dilated_mask * ~binary_mask
            fluor_intensity = torch.sum(epi_image * dilated_mask, dim=(1, 2))
            
            mask_sum = torch.sum(dilated_mask, dim=(1, 2))
            # normalise_factor = torch.where(mask_sum == 0, torch.tensor(0, device=mask_sum.device), 1/mask_sum)
            
            normalised_fluor_intensity = torch.where(mask_sum > 0, fluor_intensity / mask_sum, torch.tensor(0, device=mask_sum.device))
            # print(normalised_fluor_intensity.shape)
            results[valid, i] = normalised_fluor_intensity[valid]

        return results.cpu().numpy()
            
            
class Fluorescence(BaseFeature):
    """Quantify and describe distribution of fluorescence wihtin cells"""
    primary_feature = True
    crop = True
    
    def __init__(self):
        super().__init__()
        # self.crop_size = 300
        # self.pad = self.crop_size // 2
    
    def get_names(self):
        return [
            'Total Fluorescence', 
            'Fluorescence Distance Mean',
            'Fluorescence Distance Variance',
            'Inner Total Fluorescnece',
            'Outer Total FLuorescence',
            ]
    
    def compute(self, mask: torch.tensor = None, image: torch.tensor = None, epi_image: torch.tensor = None, binary_mask: torch.tensor = None) -> np.ndarray:
        num_cells, H, W = mask.shape
        
        results = np.full((num_cells, 5), np.nan)
        
        valid = torch.sum(mask, dim=(1,2)) > 0
        num_valid = torch.sum(valid).item()
        if num_valid > 0:
            
            epi_image = epi_image[valid]
            mask = mask[valid]
            
            temp_mask = (~mask).unsqueeze(0).to(torch.float) # (1, num_cells_in_batch, H, W)
            # mask = mask.unsqueeze(0).to(torch.float) #(1, num_cells_in_batch, H, W)
            dist_transform = kornia.contrib.distance_transform(image = temp_mask)
            # mask = mask.squeeze(0)
            dist_transform = dist_transform.squeeze(0) # (num_cells, H, W)
            
            max_dist, _ = torch.max(dist_transform.view(num_valid, H*W), dim=1) # (num_cells)
            
            max_dist = max_dist.unsqueeze(1).unsqueeze(1).expand(-1, H, W)
            dist_transform = dist_transform / (max_dist + 1e-6)
            
            inner_mask = dist_transform > 0.5 
            outer_mask = (dist_transform <= 0.5) & (dist_transform > 0)
            
            total_fluorescence = torch.sum(epi_image*mask, dim=(1, 2))
            dist_mean = torch.sum(epi_image * mask * dist_transform, dim=(1, 2)) / total_fluorescence
            dist_variance = torch.sum(epi_image * mask * dist_transform**2, dim=(1, 2)) / total_fluorescence - dist_mean**2
            inner_fluorescence = torch.sum(epi_image*inner_mask, dim=(1, 2))
            outer_fluorescence = torch.sum(epi_image*outer_mask, dim=(1, 2))
            
            results = np.full((num_cells, 5), np.nan)
            valid_np = valid.cpu().numpy()
            results[valid_np, 0] = total_fluorescence.cpu().numpy()
            results[valid_np, 1] = dist_mean.cpu().numpy()
            results[valid_np, 2] = dist_variance.cpu().numpy()
            results[valid_np, 3] = inner_fluorescence.cpu().numpy()
            results[valid_np, 4] = outer_fluorescence.cpu().numpy()

        return results

def crop_masks_images(expanded_mask: torch.tensor, epi_image: torch.tensor, phase_image: torch.tensor, binary_mask: torch.tensor, x_centres: torch.tensor, y_centres=torch.tensor, crop_size=200) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    
    num_cells = expanded_mask.shape[0]
    
    shape = (num_cells, crop_size, crop_size)
    
    mask_crops = torch.full(shape, False, device=expanded_mask.device)
    epi_crops = torch.full(shape, 0, device=expanded_mask.device, dtype=torch.uint8)
    phase_crops = torch.full(shape, 0, device=expanded_mask.device, dtype=torch.uint8)
    binary_mask_crops = torch.full(shape, 0, device=expanded_mask.device, dtype=torch.bool)

    valid = ~torch.isnan(x_centres)
    num_valid = torch.sum(valid)
    
    if num_valid > 0:
    
        half_crop_size = crop_size // 2
        
        pad = (half_crop_size, half_crop_size, half_crop_size, half_crop_size)
        mask_padded = torch.nn.functional.pad(expanded_mask[valid], pad)
        epi_padded = torch.nn.functional.pad(epi_image, pad).unsqueeze(0).expand(num_valid, -1, -1)
        phase_padded = torch.nn.functional.pad(phase_image, pad).unsqueeze(0).expand(num_valid, -1, -1)
        binary_padded = torch.nn.functional.pad(binary_mask, pad).unsqueeze(0).expand(num_valid, -1, -1)
        
        xcv = x_centres[valid] + half_crop_size
        ycv = y_centres[valid] + half_crop_size

        N = xcv.shape[0]
        rel = torch.arange(-half_crop_size, half_crop_size, device=mask_padded.device)
        yy, xx = torch.meshgrid(rel, rel, indexing='ij')
        
        yy = yy.unsqueeze(0) + ycv.view(N, 1, 1)
        xx = xx.unsqueeze(0) + xcv.view(N, 1, 1)
        
        H_pad, W_pad = mask_padded.shape[-2:]
        yy = yy.clamp(0, H_pad - 1).long()
        xx = xx.clamp(0, W_pad - 1).long()
        
        batch_idx = torch.arange(N, device=mask_padded.device).view(N, 1, 1).expand_as(xx)
        mask_crops_valid = mask_padded[batch_idx, xx, yy]
        epi_crops_valid = epi_padded[batch_idx, xx, yy]
        phase_crops_valid = phase_padded[batch_idx, xx, yy]
        binary_mask_crops_valid = binary_padded[batch_idx, xx, yy]
        
        mask_crops[valid] = mask_crops_valid
        epi_crops[valid] = epi_crops_valid.to(torch.uint8)
        phase_crops[valid] = phase_crops_valid.to(torch.uint8)
        binary_mask_crops[valid] = binary_mask_crops_valid.to(torch.bool)
        

    return mask_crops, epi_crops, phase_crops, binary_mask_crops

