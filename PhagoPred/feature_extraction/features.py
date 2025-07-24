import xarray as xr
import torch
import numpy as np
from tqdm import tqdm

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

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        dead = phase_xr['Dead Macrophage']
        alive = phase_xr['Macrophage']
        
        cell_state = xr.full_like(alive, np.nan, dtype=float)

        cell_state = cell_state.where(alive != 1, 1)
        cell_state = cell_state.where(dead != 1, 0)

        # smooth
        cell_state = cell_state.rolling(Frame=5, center=True).reduce(np.nanmean)
        return cell_state.values

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
    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        """Compute number of other phase cells within self.radii pixels (centroid based).
        Manual batching becuase of probelms with O(n^2) batching with Dask arrays
        """
        print("Calculating macropahge densities...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')
        positions = positions.transpose('Frame', 'Cell Index', 'Feature').chunk({})
        results = xr.DataArray(np.full((positions.sizes['Frame'], positions.sizes['Cell Index'], len(self.radii)), np.nan),
                        dims=positions.dims)
        results.coords['Feature'] = self.radii

        for first_frame in tqdm(range(0, positions.sizes['Frame'], self.frame_batch_size)):
            last_frame = min(first_frame + self.frame_batch_size, positions.sizes['Frame'])
            
            batch_positions = positions.isel(Frame=slice(first_frame, last_frame))
            batch_positions = torch.tensor(batch_positions.values, device=device)

            distances = batch_positions.unsqueeze(2) - batch_positions.unsqueeze(1)
            distances = torch.norm(distances, dim=3)
            distances[distances == 0] = float('nan')
            for radius_idx, radius in enumerate(self.radii):
                results.values[first_frame:last_frame, :, radius_idx] = (distances < radius).sum(dim=2).cpu().numpy()
        results = results.where(phase_xr['X'].notnull())
        results = results.transpose('Frame', 'Cell Index', 'Feature')

        return results.values

    # def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
    #     """Compute number of other phase cells within self.radii pixels (centroid based).
    #     Manual batching becuase of probelms with O(n^2) batching with Dask arrays
    #     """

    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')
    #     positions = positions.transpose('Frame', 'Cell Index', 'Feature').chunk(None)

    #     results = xr.DataArray(np.full((positions.sizes['Frame'], positions.sizes['Cell Index'], len(self.radii)), np.nan),
    #                     dims=positions.dims)
    #     results.coords['Feature'] = self.radii

    #     for frame in tqdm(range(positions.sizes['Frame'])):
    #         batch_positions = positions.isel(Frame=frame).compute()
            
    #         valid_cells_mask = batch_positions.notnull().all(dim='Feature')
    #         valid_cell_indices = np.where(valid_cells_mask.values)[0]  # array of valid cell indices in this frame

    #         batch_positions = torch.tensor(batch_positions.isel({'Cell Index': valid_cell_indices}).values, device=device)
    #         distances = batch_positions.unsqueeze(1) - batch_positions.unsqueeze(0)
    #         distances = torch.norm(distances, dim=2)
    #         distances[distances == 0] = float('nan')

    #         # batch_positions = batch_positions.isel({'Cell Index': valid_cell_indices})

    #         # batch_positions_1 = batch_positions.expand_dims(dim={'Cell Index 1': batch_positions.sizes['Cell Index']}, axis=1)
    #         # batch_positions_2 = batch_positions_1.swap_dims({'Cell Index': 'Cell Index 1', 'Cell Index 1': 'Cell Index'})

    #         # distances = batch_positions_1 - batch_positions_2
    #         # distances = (distances**2).sum(dim='Feature')**0.5
    #         # distances = distances.where(distances!=0, np.nan)

    #         for radius_idx, radius in enumerate(self.radii):
    #             results.values[frame, valid_cell_indices, radius_idx] = (distances < radius).sum(dim=1).cpu().numpy()
    #     results = results.where(phase_xr['X'].notnull())
    #     results = results.transpose('Frame', 'Cell Index', 'Feature')

    #     return results.values


    # def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
    #     """Compute number of other phase cells within self.radii pixels (centroid based).
    #     Manual batching becuase of probelms with O(n^2) batching with Dask arrays
    #     """
    #     positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')
    #     positions = positions.transpose('Frame', 'Cell Index', 'Feature')

    #     results = xr.DataArray(np.full((positions.sizes['Frame'], positions.sizes['Cell Index'], len(self.radii)), np.nan),
    #                     dims=positions.dims)
    #     results.coords['Feature'] = self.radii

    #     for first_frame in tqdm(range(0, positions.sizes['Frame'], self.frame_batch_size)):
    #         last_frame = min(first_frame + self.frame_batch_size, positions.sizes['Frame'])
            
    #         batch_positions = positions.isel(Frame=slice(first_frame, last_frame))

    #         valid_cells_mask = batch_positions.not_null().all(dim='Feature')


    #         batch_positions_1 = batch_positions.expand_dims(dim={'Cell Index 1': batch_positions.sizes['Cell Index']}, axis=1)
    #         batch_positions_2 = batch_positions_1.swap_dims({'Cell Index': 'Cell Index 1', 'Cell Index 1': 'Cell Index'})

    #         distances = batch_positions_1 - batch_positions_2
    #         distances = (distances**2).sum(dim='Feature')**0.5
    #         distances = distances.where(distances!=0, np.nan)

    #         for radius in self.radii:
    #             results.loc[{'Feature': radius, 'Frame': slice(first_frame, last_frame)}] = (distances < radius).sum(dim='Cell Index 1')
    #     results = results.where(phase_xr['X'].notnull())
    #     results = results.transpose('Frame', 'Cell Index', 'Feature')

    #     return results.values



    # def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
    #     with ProgressBar():
    #         print('\tCalculating densities. Progress: ')
    #         positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')
    #         positions = positions.chunk({"Feature": 2, "Frame": 1, "Cell Index": -1})
    #         # print(positions.chunks)
    #         positions = positions.transpose('Frame', 'Cell Index', 'Feature')
    #         # print(positions.chunks)
    #         positions_1 = positions.expand_dims(dim={'Cell Index 1': positions.sizes['Cell Index']}, axis=1)
    #         # print("1",  positions_1.chunks)

    #         positions_2 = positions_1.swap_dims({'Cell Index': 'Cell Index 1', 'Cell Index 1': 'Cell Index'})

    #         # print("2", positions_2.chunks)
    #         distances = positions_1 - positions_2
    #         # print("dist", distances.chunks)
    #         distances = (distances**2).sum(dim='Feature')**0.5
    #         # print("dist", distances.chunks)
        
    #         distances = distances.where(distances!=0, np.nan)
    #         # print("dist", distances.chunks)
    #         # create xarray to store results, using full_like copies chunking
    #         # results = xr.DataArray(np.full((positions.sizes['Frame'], positions.sizes['Cell Index'], len(self.radii)), np.nan),
    #         #                     dims=positions.dims)
    #         # results.coords['Feature'] = self.radii
    #         # for radius in self.radii:
    #         #     results.loc[dict(Feature=radius)] = (distances < radius).sum(dim='Cell Index 1')
        
    #         # results = results.where(phase_xr['X'].notnull())

    #         # results = results.transpose('Frame', 'Cell Index', 'Feature')
    #         # return results.compute().values

    #         tasks = []
    #         for radius in self.radii:
    #             mask = (distances < radius)
    #             task = mask.sum(dim='Cell Index 1')
    #             tasks.append(task)

    #         results = xr.concat(tasks, dim='Feature')
    #         results.coords['Feature'] = self.radii
    #         results = results.where(phase_xr['X'].notnull())
    #         return results.transpose('Frame', 'Cell Index', 'Feature').compute().values

    # def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.ndarray:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     max_batch = 512  # tune based on GPU RAM

    #     # Extract positions from 'X' and 'Y'
    #     positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')
    #     positions = positions.transpose('Frame', 'Cell Index', 'Feature')

    #     num_frames = positions.sizes['Frame']
    #     num_cells = positions.sizes['Cell Index']
    #     num_radii = len(self.radii)

    #     # Output array to store results
    #     results = np.full((num_frames, num_cells, num_radii), np.nan, dtype=np.float32)

    #     # Loop over each frame independently (avoids huge tensors)
    #     for frame_idx in range(num_frames):
    #         pos_frame = positions.isel(Frame=frame_idx).values  # shape: (Cells, 2)
            
    #         if np.isnan(pos_frame).all():
    #             continue  # skip empty frame

    #         pos_tensor = torch.tensor(pos_frame, dtype=torch.float32, device=device)  # (N, 2)

    #         # Loop over cell batches
    #         for start in range(0, num_cells, max_batch):
    #             end = min(start + max_batch, num_cells)
    #             batch = pos_tensor[start:end]  # (B, 2)

    #             # Compute pairwise distances: (B, N)
    #             diffs = batch[:, None, :] - pos_tensor[None, :, :]  # (B, N, 2)
    #             dists = torch.linalg.norm(diffs, dim=-1)  # (B, N)

    #             dists[dists == 0] = float('nan')  # ignore self

    #             # For each radius, count neighbors
    #             for r_idx, radius in enumerate(self.radii):
    #                 count = torch.sum(dists < radius, dim=1)  # (B,)
    #                 results[frame_idx, start:end, r_idx] = count.cpu().numpy()

    #     # Mask results based on NaNs in original input
    #     x_valid = phase_xr['X'].values
    #     results[np.isnan(x_valid)] = np.nan

    #     return results



    # def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
    #     # Extract positions from 'X' and 'Y' coordinates
    #     positions = xr.concat([phase_xr[feat] for feat in ['X', 'Y']], dim='Feature')
    #     positions = positions.transpose('Frame', 'Cell Index', 'Feature')

    #     # Broadcast positions to compute pairwise differences
    #     pos1 = positions.expand_dims({'Cell Index 1': positions.sizes['Cell Index']})  # shape: (Frame, Cell Index, Cell Index 1, Feature)
    #     pos2 = positions.expand_dims({'Cell Index 2': positions.sizes['Cell Index']})  # shape: (Frame, Cell Index 2, Cell Index, Feature)

    #     # Align for broadcasting: swap and rename dims
    #     pos1 = pos1.transpose('Frame', 'Cell Index', 'Cell Index 1', 'Feature')
    #     pos2 = pos2.transpose('Frame', 'Cell Index 1', 'Cell Index', 'Feature')
    #     pos2 = pos2.rename({'Cell Index': 'Cell Index 1'})

    #     # Compute pairwise distances
    #     diffs = pos1 - pos2  # shape: (Frame, Cell Index, Cell Index 1, Feature)
    #     distances = np.sqrt((diffs ** 2).sum(dim='Feature'))

    #     # Avoid self-comparisons
    #     distances = distances.where(distances != 0, np.nan)

    #     # Prepare result array with same chunking
    #     radii = self.radii
    #     result = xr.zeros_like(phase_xr['X'].expand_dims(Feature=radii)).astype('float32')
    #     result.coords['Feature'] = radii
    #     result.name = 'counts'

    #     # Count neighbors within each radius
    #     for radius in radii:
    #         count = (distances < radius).sum(dim='Cell Index 1')
    #         result.loc[dict(Feature=radius)] = count

    #     # Mask out invalid cell entries
    #     result = result.where(phase_xr['X'].notnull())

    #     # Transpose to desired output shape
    #     result = result.transpose('Frame', 'Cell Index', 'Feature')

    #     # Return as NumPy array
    #     return result.compute().values

   
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