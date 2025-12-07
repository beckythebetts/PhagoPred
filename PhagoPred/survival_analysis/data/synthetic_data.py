import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

class Cell:
    def __init__(self, T: int = 0):
        self.T = T
        self.pmf = self._generate_base_pmf()
        self.features = self._generate_base_features()
    
    def _generate_base_pmf(self):
        pmf = np.ones(self.T) / self.T
        # end_peak = np.exp(-0.5 * ((np.arange(self.T) - (self.T-1)) / 50)**2)
        # pmf += end_peak
        return pmf
        
    def _generate_base_features(self):
        features = {
            '0': 2 + np.random.randn(self.T)*0.1,
            '1': np.abs(np.random.randn(self.T)),
            '2': np.arange(self.T) + np.random.randn(self.T)*5,
            '3': np.arange(self.T)**0.5 + np.random.randn(self.T),
        }
        return features
    
    def __getitem__(self, key):
        return self.features[key]

    def __setitem__(self, key, value):
        self.features[key] = value
        
    def normalise_pmf(self):
        total = np.sum(self.pmf)
        if total > 0:
            self.pmf /= total
    
    def apply_observation_window(self, start: int, end: int):
        """Apply observation window to features."""
        for key in self.features:
            self.features[key][:start] = np.nan
            self.features[key][end+1:] = np.nan



class Rule:
    def apply(self, cell: Cell) -> None:
        """Modify features and pmf"""
        pass
    
    
class VariationRule(Rule):
    """Increase hazard {delay} frames after an increase in std of {feature}."""
    def __init__(self, feature: str = '0', probability: float = 1.0, delay: int = 200, sigma: float = 10.0):
        self.feature = feature
        self.probability = probability
        self.delay = delay
        self.sigma = sigma
        
    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            
            strength = np.abs(np.random.rand())
            
            frame = np.random.randint(cell.T)
            start_idx = frame - self.delay
            if start_idx < 0:
                return
            slice_ = cell[self.feature][start_idx:] 
            slice_ += np.random.randn(len(slice_))*strength
            cell[self.feature][start_idx:] = slice_
            
            t = np.arange(cell.T)
            gaussian = np.exp(-0.5 * ((t - frame) / self.sigma)**2)*strength
            
            cell.pmf += gaussian


class GradualRampRule(Rule):
    def __init__(self, feature='1', probability=1.0, ramp_length=30, ramp_height=10.0, delay=100, sigma=10.0):
        self.feature = feature
        self.probability = probability
        self.ramp_length = ramp_length
        self.ramp_height = ramp_height
        self.delay = delay
        self.sigma = sigma

        
    def apply(self, cell: Cell):
        if np.random.rand() < self.probability:
            strength = np.abs(np.random.randn())
            start = np.random.randint(cell.T - self.ramp_length)
            ramp = np.linspace(0, self.ramp_height*strength, self.ramp_length)
            cell[self.feature][start:start+self.ramp_length] += ramp
            t = np.arange(cell.T)
            hazard_frame = start + self.ramp_length + self.delay
            cell.pmf += np.exp(-0.5*((t - hazard_frame)/self.sigma)**2) * strength


            
def create_synthetic_dataset(filename: Path, num_cells: int = 1000, num_frames: int = 1000):
    """Create synthetic dataset with given number of cells and frames."""
    
    rules = [
        # VariationRule(),
        GradualRampRule(),
    ]
    
    start_frames = np.random.randint(0, num_frames//2, size=num_cells)
    end_frames = np.random.randint(num_frames//2, num_frames, size=num_cells)
    
    features = Cell().features.keys()
    
    all_features = {name: np.empty((num_frames, num_cells), dtype=np.float32) for name in features}
    all_deaths = np.empty(num_cells, dtype=np.float32)
    pmfs = np.empty((num_frames, num_cells), dtype=np.float32)
    
    for c in tqdm(range(num_cells), desc='Generating cells'):
        cell = Cell(num_frames)
        start = start_frames[c]
        end = end_frames[c]
        
        # Apply rules
        for rule in rules:
            rule.apply(cell)
        cell.normalise_pmf()
        pmfs[:, c] = cell.pmf
    
        death_frame = np.random.choice(np.arange(num_frames), p=cell.pmf)
        if death_frame < start or death_frame > end:
            death_frame = np.nan
        
        all_deaths[c] = death_frame
        cell.apply_observation_window(start, end)
        
        # plt.plot(cell.pmf)
        # plt.axvline(start, color='k', linestyle='--')
        # plt.axvline(end, color='k', linestyle='--')
        # if not np.isnan(death_frame):
        #     plt.axvline(death_frame, color='r', linestyle='-')
        # plt.show()  
        
        for name in features:
            all_features[name][:, c] = cell[name]
            
    with h5py.File(filename, 'w') as f:
        grp = f.create_group('Cells/Phase')
        for name in features:
            grp.create_dataset(name, data=all_features[name], dtype=np.float32)
        grp.create_dataset('CellDeath', data=all_deaths[np.newaxis, :], dtype=np.float32)
        grp.create_dataset('PMFs', data=pmfs, dtype=np.float32)
            
if __name__ == '__main__':
    create_synthetic_dataset(
        filename = Path('PhagoPred') / 'Datasets' / 'val_synthetic.h5',
        num_cells=1000,
        num_frames=1000,
    )
    create_synthetic_dataset(
        filename = Path('PhagoPred') / 'Datasets' / 'synthetic.h5',
        num_cells=10000,
        num_frames=1000,
    )