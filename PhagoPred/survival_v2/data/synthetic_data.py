import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

class Cell:
    def __init__(self, T: int = 0, noise_level: float = 0.01):
        self.T = T
        self.noise_level = noise_level
        self.features = self._generate_base_features(noise_level)
        self.hazards = self._generate_base_hazards()
    
    # def _generate_base_pmf(self):
    #     # pmf = np.ones(self.T) / self.T
    #     pmf = np.zeros(self.T)
    #     # end_peak = np.exp(-0.5 * ((np.arange(self.T) - (self.T-1)) / 50)**2)
    #     # pmf += end_peak
    #     return pmf

    def _compute_pmf(self):
        assert (self.hazards >= 0.0).any(), f"Hazards contain negative values {self.hazards}"
        assert (self.hazards <= 1.0).any(), "Hazards contain values greater than 1.0"
        hazards = self.hazards
        sf = np.cumprod(1 - hazards)
        pmf = np.zeros(self.T)
        pmf[0] = hazards[0]
        for t in range(1, self.T):
            pmf[t] = sf[t-1] * hazards[t]
        # assert pmf.sum() <= 1.0, f"PMF sums to more than 1.0: {pmf.sum()}"
        return pmf
    
    def _generate_base_hazards(self):
        hazards = np.full(self.T, 0.0)
        return hazards
        
    def _generate_base_features(self, noise_level=0.01):
        features = {
            '0': 2 + np.random.randn(self.T)*noise_level,
            '1': np.abs(np.random.randn(self.T))*noise_level,
            '2': np.arange(self.T) + np.random.randn(self.T)*noise_level,
            '3': np.random.randn(self.T)*noise_level,
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
            
    def apply_masking(self, probability: float = 0.1):
        """Randomly mask features with NaNs."""
        for key in self.features:
            mask = np.random.rand(self.T) < probability
            self.features[key][mask] = np.nan



class Rule:
    def apply(self, cell: Cell) -> None:
        """Modify features and pmf"""
        pass
    
    
class VariationRule(Rule):
    """Increase hazard {delay} frames after an increase in std of {feature}."""
    def __init__(self, feature: str = '0', probability: float = 1.0, delay: int = 200, sigma: float = 20.0, max_strength=1.0):
        self.feature = feature
        self.probability = probability
        self.delay = delay
        self.sigma = sigma
        self.max_strength = max_strength
        
    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            
            strength = np.random.rand()*self.max_strength
            
            frame = np.random.randint(cell.T)
            start_idx = frame - self.delay
            if start_idx < 0:
                return
            slice_ = cell[self.feature][start_idx:start_idx+100] 
            slice_ += np.random.randn(len(slice_))*strength*100
            cell[self.feature][start_idx:start_idx+100] = slice_
            
            t = np.arange(cell.T)
            # gaussian = np.exp(-0.5 * ((t - frame) / self.sigma)**2)
            # gaussian = gaussian / np.sum(gaussian)
            # cell.pmf += gaussian * strength
            
            # gaussian = np.exp(-0.5 * ((t - frame) / self.sigma)**2)
            # gaussian = gaussian / np.max(gaussian)
            # cell.hazards += gaussian * strength
            cell.hazards += get_gaussian_curve(cell.T, frame, self.sigma) * strength
            # assert np.sum(cell.pmf) <= 1.0, f"PMF exceeds 1.0, {self.max_strength}, {strength}, {np.sum(cell.pmf)}"


class GradualRampRule(Rule):
    def __init__(self, feature='1', probability=1.0, ramp_length=30, ramp_height=10.0, delay=100, sigma=30.0, max_strength: float=1.0):
        self.feature = feature
        self.probability = probability
        self.ramp_length = ramp_length
        self.ramp_height = ramp_height
        self.delay = delay
        self.sigma = sigma
        self.max_strength = max_strength

    
    def apply(self, cell: Cell):
        if np.random.rand() < self.probability:
            strength = (np.random.rand()**0.1)*self.max_strength
            # start = np.random.randint(cell.T - self.ramp_length)
            start = np.random.randint(cell.T)
            ramp = np.linspace(0, self.ramp_height*strength, self.ramp_length)
            ramp_end = min(start+self.ramp_length, cell.T)
            cell[self.feature][start:ramp_end] += ramp[:ramp_end-start]
            t = np.arange(cell.T)
            hazard_frame = start + self.ramp_length + self.delay

            cell.hazards += get_gaussian_curve(cell.T, hazard_frame, self.sigma) * strength

def get_gaussian_curve(T: int, center: int, sigma: float):
    t = np.arange(T)
    gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
    gaussian /= (sigma * np.sqrt(2 * np.pi))
    return gaussian * 5

def pmf_from_hazards(hazards: np.ndarray):
    sf = np.cumprod(1 - hazards)
    pmf = np.zeros(len(hazards))
    pmf[0] = hazards[0]
    for t in range(1, len(hazards)):
        pmf[t] = sf[t-1] * hazards[t]
    # assert pmf.sum() <= 1.0, f"PMF sums to more than 1.0: {pmf.sum()}"
    return pmf


def create_synthetic_dataset(
    filename: Path,
    num_cells: int = 1000,
    num_frames: int = 1000,
    rules: list = None,
    noise_level: float = 0.01,
    late_entry_prob: float = 0.0,
    late_entry_range: tuple = (0, 100),
    feature_mask_prob: float = 0.0,
    seed: int = None
):
    """
    Create synthetic dataset with configurable parameters.

    Args:
        filename: Output HDF5 file path
        num_cells: Number of cells to generate
        num_frames: Number of time frames
        rules: List of Rule objects to apply (if None, uses default rules)
        noise_level: Standard deviation of Gaussian noise added to features
        late_entry_prob: Probability of late entry (0 to 1)
        late_entry_range: (min, max) frames for late entry start time
        feature_mask_prob: Probability of randomly masking feature values with NaN
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # Default rules if none provided
    if rules is None:
        rules = [
            VariationRule(feature='3', delay=150, sigma=5.0),
            GradualRampRule(feature='0', ramp_height=10.0, delay=200, sigma=5.0),
            GradualRampRule(feature='1', ramp_height=10.0, delay=100, sigma=5.0),
        ]

    num_rules = len(rules) if rules else 1
    for rule in rules:
        rule.max_strength = 1 / num_rules
        print(f"Rule {rule.__class__.__name__}: max_strength={rule.max_strength:.3f}")
    
    # Configure start frames (late entry)
    start_frames = np.zeros(num_cells, dtype=int)
    if late_entry_prob > 0:
        late_entry_mask = np.random.rand(num_cells) < late_entry_prob
        num_late_entry = np.sum(late_entry_mask)
        start_frames[late_entry_mask] = np.random.randint(
            late_entry_range[0], late_entry_range[1], size=num_late_entry
        )
        print(f"Late entry: {num_late_entry}/{num_cells} cells ({100*late_entry_prob:.1f}%)")
    else:
        start_frames[:] = np.random.randint(0, 10, size=num_cells)

    end_frames = np.random.randint(num_frames-num_frames//2, num_frames, size=num_cells)
    
    features = Cell().features.keys()
    
    all_features = {name: np.empty((num_frames, num_cells), dtype=np.float32) for name in features}
    all_deaths = np.empty(num_cells, dtype=np.float32)
    cifs = np.empty((num_frames, num_cells), dtype=np.float32)
    pmfs = np.empty((num_frames, num_cells), dtype=np.float32)
    hazards = np.empty((num_frames, num_cells), dtype=np.float32)
    
    for c in tqdm(range(num_cells), desc='Generating cells'):
        cell = Cell(num_frames, noise_level=noise_level)
        start = start_frames[c]
        end = end_frames[c]

        # Apply rules
        for rule in rules:
            rule.apply(cell)

        cell.hazards = np.clip(cell.hazards, a_min=0.0, a_max=1.0)
        pmf = cell._compute_pmf()
        cif = np.cumsum(pmf)

        cifs[:, c] = cif
        pmfs[:, c] = pmf
        hazards[:, c] = cell.hazards

        # Sample death time from cif
        u = np.random.rand()
        if u > np.max(cif):
            death_frame = np.nan
        else:
            death_frame = np.argmax(cif >= u)
            if death_frame > end:
                death_frame = np.nan

        # Ensure death is within observation window
        if death_frame < start or death_frame > end:
            death_frame = np.nan

        all_deaths[c] = death_frame
        cell.apply_observation_window(start, end)

        # Apply random feature masking if requested
        if feature_mask_prob > 0:
            cell.apply_masking(probability=feature_mask_prob)
        
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
        grp.create_dataset('CIFs', data=cifs, dtype=np.float32)
        grp.create_dataset('PMFs', data=pmfs, dtype=np.float32)
        grp.create_dataset('Hazards', data=hazards, dtype=np.float32)
        
        
            
if __name__ == '__main__':
    create_synthetic_dataset(
        filename = Path('PhagoPred') / 'Datasets' / 'val_synthetic.h5',
        num_cells=1000,
        num_frames=1000,
    )
    create_synthetic_dataset(
        filename = Path('PhagoPred') / 'Datasets' / 'synthetic.h5',
        num_cells=1000,
        num_frames=1000,
    )