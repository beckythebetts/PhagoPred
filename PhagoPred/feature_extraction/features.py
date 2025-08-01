import xarray as xr
import torch
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, box
from scipy.signal import convolve

from PhagoPred.feature_extraction.morphology.fitting import MorphologyFit
from PhagoPred.feature_extraction.texture.gabor import Gabor
from PhagoPred import SETTINGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseFeature:

    primary_feature = False
    derived_feature = False

    def __init__(self):
        self.name = [self.__class__.__name__]
        # self.index_positions = None

    def get_names(self):
        return self.name
    
    def compute(self):
        raise NotImplementedError(f"{self.get_name()} has no compute method implemented")
    
    # def set_index_positions(self, start: int, end: int = None) -> None:
    #     self.index_positions = (start, end)
    
    # def get_index_positions(self) -> list[int]:
    #     return self.index_positions
    
class CellDeath(BaseFeature):
    """
    Determine frame at which cell dies, or np.nan if no cell death.
    Note: unlike all other cell features, this returns a single value for each cell (not each frame of each cell).
    """
    derived_feature = True

    threshold = 0.5
    min_frames_dead = 10

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        dead = phase_xr['Dead Macrophage']
        alive = phase_xr['Macrophage']
        
        cell_state = xr.full_like(alive, np.nan, dtype=float)

        cell_state = cell_state.where(alive != 1, 1)
        cell_state = cell_state.where(dead != 1, 0)

        # smooth
        smoothed_cell_state = cell_state.rolling(Frame=5, center=True).reduce(np.nanmean).values

        below_threshold = smoothed_cell_state < self.threshold

        kernel = np.ones(self.min_frames_dead, dtype=int)
        conv_result = np.apply_along_axis(
            lambda x: convolve(x.astype(int), kernel, mode='valid'),
            axis=0, arr=below_threshold
        )

        death_possible = conv_result == self.min_frames_dead  # shape: (frames - N + 1, cells)
        first_death_idx = np.argmax(death_possible, axis=0)  # (n_cells,)
        has_death = np.any(death_possible, axis=0)

        # Output array
        death_frames = np.full(smoothed_cell_state.shape, np.nan)
        death_frames[0, has_death] = first_death_idx[has_death]

        return death_frames

class Coords(BaseFeature):
    """
    Centroid x, y coordinates and area
    """
    primary_feature = True

    def __init__(self):
        self.x_grid, self.y_grid = None, None

    def get_names(self):
        return ['X', 'Y', 'Area']
    
    def compute(self, mask: torch.tensor, image: torch.tensor) -> np.array:
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
    
class MorphologyModes(BaseFeature):

    primary_feature = True

    def __init__(self):
        super().__init__()
        self.morphology_model = MorphologyFit(SETTINGS.DATASET)

        try:
            self.morphology_model.load_model()

        except:
            print('Fitting Morpgology Model')
            self.morphology_model.fit()


    def get_names(self):
        return [f'Mode {i}' for i in range(self.morphology_model.num_kmeans_clusters)]
    
    def compute(self, mask: torch.tensor, image: torch.tensor) -> np.array:
        """
        Results are of dims [num_cells, num_clusters]
        """
        expanded_frame_mask = mask.cpu().numpy()
        results = self.morphology_model.apply_expanded_mask(expanded_frame_mask)
        return results
    
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

    
class DensityPhase(BaseFeature):

    derived_feature = True

    radii = [100, 250, 500]

    frame_batch_size = 10

    def get_names(self):
        return [f'Phagocytes within {radius} pixels' for radius in self.radii]
    
    def circle_fraction_in_frame(self, x, y, r, H, W):
        """Calculate the fraction of a circle that is within the frame."""
        circle = Point(x, y).buffer(r)
        frame = box(0, 0, W, H)
        intersection = circle.intersection(frame)
        return intersection.area / (np.pi * r * r)
    
    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        """Compute number of other phase cells within self.radii pixels (centroid based).
        Manual batching becuase of probelms with O(n^2) batching with Dask arrays
        """
        # print("Calculating macropahge densities...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')
        positions = positions.transpose('Frame', 'Cell Index', 'Feature').chunk({})
        # results = xr.DataArray(np.full((positions.sizes['Frame'], positions.sizes['Cell Index'], len(self.radii)), np.nan),
        #                 dims=positions.dims)
        # results.coords['Feature'] = self.radii

        results = np.full((positions.sizes['Frame'], positions.sizes['Cell Index'], len(self.radii)), np.nan)

        H, W = SETTINGS.IMAGE_SIZE

        for first_frame in tqdm(range(0, positions.sizes['Frame'], self.frame_batch_size)):
            last_frame = min(first_frame + self.frame_batch_size, positions.sizes['Frame'])
            
            batch_positions_np = positions.isel(Frame=slice(first_frame, last_frame)).values
            batch_positions = torch.tensor(batch_positions_np, device=device)


            distances = batch_positions.unsqueeze(2) - batch_positions.unsqueeze(1)
            distances = torch.norm(distances, dim=3)
            distances[distances == 0] = float('nan')
            for radius_idx, radius in enumerate(self.radii):
                counts = (distances < radius).sum(dim=2).cpu().numpy()

                xs = batch_positions_np[..., 0]
                ys = batch_positions_np[..., 1]
                fraction_in = np.ones_like(xs)

                # Only correct for cells near the edge
                edge_mask = (
                    (xs - radius < 0) | (xs + radius > W) |
                    (ys - radius < 0) | (ys + radius > H)
                ) & ~np.isnan(xs) & ~np.isnan(ys)

                if np.any(edge_mask):
                    # Vectorized geometric correction using shapely
                    edge_indices = np.argwhere(edge_mask)
                    for idx in edge_indices:
                        i, j = idx
                        x, y = xs[i, j], ys[i, j]
                        fraction_in[i, j] = self.circle_fraction_in_frame(x, y, radius, H, W)

                # Avoid division by zero or overcorrection
                fraction_in = np.clip(fraction_in, 1e-3, 1.0)
                corrected = counts / fraction_in
                results[first_frame:last_frame, :, radius_idx] = corrected

        results = np.where(np.isnan(phase_xr['X'].values)[:, :, np.newaxis], np.nan, results)

        return results
   
class Perimeter(BaseFeature):

    primary_feature = True

    def compute(self, mask: torch.tensor, image: torch.tensor) -> np.array:

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
    
class GaborScale(BaseFeature):

    primary_feature = True

    def __init__(self):
        super().__init__()
        self.gabor = Gabor()

    def compute(self, mask: torch.tensor, image: torch.tensor) -> np.array:
        dominant_scales = self.gabor.get_dominant_scales(image, mask)
        return np.array(dominant_scales)
    
class Fluorescence(BaseFeature):
    """Fluorescence intensity of cells in the image.
    Use radial moments to describe distribution from centre.
    $M_n = \sum_{i} I_i * r_i^n$
    where $I_i$ is the intensity at pixel $i$ and $r_i$ is the distance from the cell centroid.
    Total fluorescence is $M_0$, mean fluorescence distance is $M_1 / M_0$, and variance is $M_2 / M_0 - (M_1 / M_0)^2$.
    """
    primary_feature = True
    derived_feature = True
    requires = ['X', 'Y']

    def __init__(self):
        super().__init__()
        self.crop_size = 300
        self.pad = self.crop_size // 2
        self.distance_map = None  # Placeholder for distance map, if needed

    def get_distance_grid(self, device):

        # Create coordinate grids
        y = torch.arange(self.crop_size, device=device)
        x = torch.arange(self.crop_size, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        # Compute distances from center
        dist = torch.sqrt((yy - self.pad)**2 + (xx - self.pad)**2)
        return dist  # shape: (crop_size, crop_size)

    def get_names(self):
        return [
            'Total Fluorescence', 
            'Fluorescence Distance Mean', 
            'Fluorescence Distance Variance'
            ]
    
    def crop_centered_batch(self, array_padded, x_centres_padded, y_centres_padded):
        
        N = x_centres_padded.shape[0]
        pad = self.crop_size // 2

        rel = torch.arange(-pad, pad, device=array_padded.device)
        grid_y = rel.view(1, self.crop_size).expand(N, self.crop_size) + y_centres_padded.view(N, 1)
        grid_x = rel.view(1, self.crop_size).expand(N, self.crop_size) + x_centres_padded.view(N, 1)

        H_pad, W_pad = array_padded.shape[1:]
        grid_y = grid_y.clamp(0, H_pad - 1)
        grid_x = grid_x.clamp(0, W_pad - 1)

        assert torch.all(torch.isfinite(grid_y)), "grid_y contains NaN or Inf"
        assert torch.all(torch.isfinite(grid_x)), "grid_x contains NaN or Inf"
        assert torch.all(grid_y >= 0), "grid_y contains negative indices"
        assert torch.all(grid_x >= 0), "grid_x contains negative indices"
        assert array_padded.device == grid_y.device == grid_x.device, "Device mismatch"

        crops = array_padded[
            torch.arange(N, device=array_padded.device).view(N, 1, 1).int(),
            grid_x.unsqueeze(1).expand(N, self.crop_size, self.crop_size).int(),
            grid_y.unsqueeze(2).expand(N, self.crop_size, self.crop_size).int(),
        ]
        return crops

    def compute(self, mask: torch.tensor = None, image: torch.tensor = None, epi_image: torch.tensor = None, phase_xr: xr.Dataset = None, epi_xr: xr.Dataset = None) -> np.array:
        if self.distance_map is None:
            self.distance_map = self.get_distance_grid(mask.device)
        
        epi_image = torch.from_numpy(epi_image).to(mask.device)
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if epi_image.ndim == 2:
            epi_image = epi_image.unsqueeze(0)

        self.pad = self.crop_size // 2

        x_centres = torch.from_numpy(phase_xr['X'].values).to(mask.device)
        y_centres = torch.from_numpy(phase_xr['Y'].values).to(mask.device)

        valid = (~torch.isnan(x_centres)) & (~torch.isnan(y_centres))
        
        x_centres_padded = (x_centres + self.pad)[valid]
        y_centres_padded = (y_centres + self.pad)[valid]

        self.mask_padded = torch.nn.functional.pad(mask, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)[valid]
        self.epi_image_padded = torch.nn.functional.pad(epi_image, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        self.epi_image_padded = self.epi_image_padded.expand(len(x_centres_padded), -1, -1)

        cropped_masks = self.crop_centered_batch(self.mask_padded, x_centres_padded, y_centres_padded)
        cropped_epi_image = self.crop_centered_batch(self.epi_image_padded, x_centres_padded, y_centres_padded)

        total_fluorescence = torch.sum(cropped_epi_image * cropped_masks, dim=(1, 2))
        dist_mean = torch.sum(cropped_epi_image * cropped_masks * self.distance_map, dim=(1, 2)) / total_fluorescence
        dist_variance = torch.sum(cropped_epi_image * cropped_masks * self.distance_map**2, dim=(1, 2)) / total_fluorescence - dist_mean**2

        results = np.full((mask.shape[0], 3), np.nan)
        results[valid.cpu().numpy(), 0] = total_fluorescence.cpu().numpy()
        results[valid.cpu().numpy(), 1] = dist_mean.cpu().numpy()
        results[valid.cpu().numpy(), 2] = dist_variance.cpu().numpy()

        return results