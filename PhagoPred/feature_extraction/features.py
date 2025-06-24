import xarray as xr
import torch
import numpy as np

from PhagoPred.feature_extraction.morphology.fitting import MorphologyFit
from PhagoPred.feature_extraction.texture.gabor import Gabor
from PhagoPred import SETTINGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseFeature:

    primary_feature = False
    derived_feature = False

    def __init__(self):
        self.name = [self.__class__.__name__]
        self.index_positions = None

    def get_names(self):
        return self.name
    
    def compute(self):
        raise NotImplementedError(f"{self.get_name()} has no compute method implemented")
    
    def set_index_positions(self, start: int, end: int = None) -> None:
        self.index_positions = (start, end)
    
    def get_index_positions(self) -> list[int]:
        return self.index_positions
    
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
        positions = phase_xr.sel(Feature=['x', 'y'])

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

        return speeds.values
    
class Displacement(BaseFeature):

    derived_feature = True

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        positions = phase_xr.sel(Feature=['x', 'y'])

        positions_0 = positions.isel(Frame=0).expand_dims({'Frame': positions.sizes['Frame']})

        displacements = ((positions-positions_0)**2).sum(dim='Feature')**0.5

        mask = positions.sel(Feature='x').notnull()

        displacements = displacements.where(mask)

        displacements = displacements.expand_dims({'Feature':1}, axis=-1)

        return displacements.values

    
class DensityPhase(BaseFeature):

    derived_feature = True

    radii = [100, 250, 500]

    def get_names(self):
        return [f'Phagocytes within {radius} pixels' for radius in self.radii]

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:

        positions = phase_xr.sel(Feature=['x', 'y'])

        positions_1 = positions.expand_dims(dim={'Cell Index 1': positions.sizes['Cell Index']}, axis=1)

        positions_2 = positions_1.swap_dims({'Cell Index': 'Cell Index 1', 'Cell Index 1': 'Cell Index'})

        distances = positions_1 - positions_2
        
        distances = (distances**2).sum(dim='Feature')**0.5
        # distances = distances.where(distances!=0, np.nan)

        # create xarray to store results, using full_like copies chunking
        results = xr.full_like(phase_xr, np.nan).isel(Feature=slice(0, len(self.radii)))
        results.coords['Feature'] = self.radii
        for radius in self.radii:
            results.loc[dict(Feature=radius)] = (distances < radius).sum(dim='Cell Index 1') - 1
    
        results = results.where(positions.sel(Feature='x').notnull())

        return results.values
    
class Perimeter(BaseFeature):

    primary_feature = True

    def compute(self, mask: torch.tensor, image: torch.tensor) -> np.array:
        kernel = torch.tensor(
            [[1, 1, 1],
             [1, 9, 1],
             [1, 1, 1]]
             ).to(device)
        
        padded_masks = torch.nn.functional.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
                                                padding=0).squeeze()
        
        result = np.array(torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2)).float())
        result[result==0] = np.nan

        return result
        
class Circularity(BaseFeature):
    """
    C = 4*pi*(Area) / (Perimeter)**2
    """

    derived_feature = True

    def compute(self, phase_xr: xr.DataArray, epi_xr: xr.DataArray) -> np.array:
        circularity = phase_xr.sel(Feature=['area']) * 4 * np.pi / phase_xr.sel(Feature='Perimeter')**2
        return circularity.values
    
class GaborScale(BaseFeature):

    primary_feature = True

    def __init__(self):
        super().__init__()
        self.gabor = Gabor()

    def compute(self, mask: torch.tensor, image: torch.tensor) -> np.array:
        dominant_scales = self.gabor.get_dominant_scales(image, mask)
        return np.array(dominant_scales)

