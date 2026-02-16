import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

class Cell:
    def __init__(self, T: int = 0, noise_level: float = 0.01):
        self.T = T
        self.noise_level = noise_level
        self.features = self._generate_base_features(noise_level)
        self.hazards = self._generate_base_hazards()

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
        # features = {
        #     '0': 2 + np.random.randn(self.T)*noise_level,
        #     '1': np.abs(np.random.randn(self.T))*noise_level,
        #     '2': np.arange(self.T) + np.random.randn(self.T)*noise_level,
        #     '3': np.random.randn(self.T)*noise_level,
        # }
        features = {
            'random_walk': generate_random_walk(self.T, volatility=10),
            'oscillation': generate_stochastic_oscillation(self.T, noise_level=noise_level, base_freq=0.002, freq_volatility=0.005, amplitude=5.0),
            'linear_trend': generate_linear_trend(self.T, slope_range=(0.01, 0.1), noise_level=noise_level),
            'frame_count': np.arange(self.T),
            'polynomial_trend': generate_polynomial_trend(self.T, degrees=7, noise_level=noise_level),
            'oscillation + linear': generate_stochastic_oscillation(self.T, noise_level=noise_level, base_freq=0.02, freq_volatility=0.005, amplitude=5.0) + generate_linear_trend(self.T),
        }
        for key in features:
            features[key] = features[key].astype(np.float32)
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

# funciotns genertaing random features
def generate_random_walk(T: int, volatility: float = 0.01):
    walk = np.cumsum(np.random.randn(T) * volatility)
    return walk

def generate_stochastic_oscillation(T: int, noise_level: float = 0.01, base_freq: float = 0.05, freq_volatility: float = 0.01, amplitude: float = 1.0):
    freq = base_freq + np.cumsum(np.random.randn(T) * freq_volatility)
    freq = np.clip(freq, 0.01, 0.2)  # keep frequency reasonable
    phase = np.cumsum(freq)
    return amplitude * np.sin(2 * np.pi * phase) + np.random.randn(T) * noise_level

def generate_linear_trend(T: int, slope_range: tuple = (0.01, 0.1), noise_level: float = 0.01):
    slope = np.random.uniform(*slope_range)
    trend = slope * np.arange(T)
    return trend + np.random.randn(T) * noise_level

def generate_polynomial_trend(T: int, degrees: int = 7, noise_level: float = 0.01):
    x = np.linspace(0, 1, T)  # normalize to [0, 1] for stability
    coeffs = np.random.randn(degrees + 1)
    trend = np.polyval(coeffs, x)
    return trend + np.random.randn(T) * noise_level

class Rule:
    def apply(self, cell: Cell) -> None:
        """Modify features and pmf"""
        pass
    
# == Hazard only rules ==
class ThresholdRule(Rule):
    """Increase hazard by {amount} when {feature} exceeds {threshold}."""
    def __init__(self, feature: str = '0', threshold: float = 1.0, hazard_increase: float = 0.1, probability: float = 1.0):
        self.feature = feature
        self.threshold = threshold
        self.hazard_increase = hazard_increase
        self.probability = probability
        
    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            exceed_idxs = np.where(cell[self.feature] > self.threshold)[0]
            cell.hazards[exceed_idxs] += self.hazard_increase

class CumulativeEffectRule(Rule):
    """Increase hazard by {amount} for every {window} frames where {feature} exceeds {threshold}."""
    def __init__(self, feature: str = '0', threshold: float = 1.0, hazard_increase: float = 0.05, probability: float = 1.0):
        self.feature = feature
        self.threshold = threshold
        self.hazard_increase = hazard_increase
        self.probability = probability
        
    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            exceed_mask = cell[self.feature] > self.threshold
            cumulative_exceed = np.cumsum(exceed_mask)
            cell.hazards += (cumulative_exceed * self.hazard_increase)

class InteractionRule(Rule):
    """Increase hazard by {amount} when both {feature1} and {feature2} exceed their thresholds."""
    def __init__(self, feature1: str = '0', threshold1: float = 1.0, feature2: str = '1', threshold2: float = 1.0, hazard_increase: float = 0.1, probability: float = 1.0):
        self.feature1 = feature1
        self.threshold1 = threshold1
        self.feature2 = feature2
        self.threshold2 = threshold2
        self.hazard_increase = hazard_increase
        self.probability = probability
        
    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            interaction_mask = (cell[self.feature1] > self.threshold1) & (cell[self.feature2] > self.threshold2)
            cell.hazards[interaction_mask] += self.hazard_increase

class RandomSpikeRule(Rule):
    """Randomly add a hazard spike of {height} at a random time point."""
    def __init__(self, height: float = 0.5, probability: float = 0.1):
        self.height = height
        self.probability = probability
        
    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            spike_time = np.random.randint(cell.T)
            cell.hazards += get_gaussian_curve(cell.T, spike_time, sigma=5) * self.height

class GradientRule(Rule):
    """Increase hazard when graidient exceeds threshold."""
    def __init__(self, feature: str = '0', gradient_threshold: float = 1.0, max_increase: float = 0.5, probability: float = 1.0):
        self.feature = feature
        self.gradient_threshold = gradient_threshold
        self.max_increase = max_increase
        self.probability = probability
        
    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            feature_gradient = np.diff(cell[self.feature], prepend=cell[self.feature][0])
            mask = feature_gradient > self.gradient_threshold
            cell.hazards[mask] += feature_gradient[mask] / np.max(feature_gradient) * self.max_increase
            
# == Feature modifying rules (with delayed hazard effect) ==
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
            cell.hazards += get_gaussian_curve(cell.T, frame, self.sigma) * strength

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
        # rules = [
        #     VariationRule(feature='3', delay=300, sigma=5.0),
        #     GradualRampRule(feature='0', ramp_height=10.0, delay=450, sigma=5.0),
        #     GradualRampRule(feature='1', ramp_height=10.0, delay=400, sigma=5.0),
        # ]
        rules = [
            ThresholdRule(feature='random_walk', threshold=200, hazard_increase=5e-3, probability=1.0),
            CumulativeEffectRule(feature='oscillation', threshold=4.0, hazard_increase=5e-7, probability=1.0),
            InteractionRule(feature1='random_walk', threshold1=5.0, feature2='oscillation', threshold2=2.0, hazard_increase=-4e-2, probability=1.0),
            RandomSpikeRule(height=1e-3, probability=0.1),
            GradientRule(feature='polynomial_trend', gradient_threshold=0.5, max_increase=5e-3, probability=1.0),
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

        # Apply rules and track per-rule hazard contribution
        for rule in rules:
            hazards_before = cell.hazards.copy()
            rule.apply(cell)
            delta = cell.hazards - hazards_before
            rule_name = rule.__class__.__name__
            if not hasattr(rule, '_hazard_deltas'):
                rule._hazard_deltas = []
            rule._hazard_deltas.append(delta.mean())

        cell.hazards = gaussian_filter1d(cell.hazards, sigma=10)
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
            
    # Print per-rule hazard contribution summary
    print("\n--- Per-rule mean hazard contribution ---")
    for rule in rules:
        deltas = np.array(rule._hazard_deltas)
        name = rule.__class__.__name__
        print(f"  {name:30s}  mean={deltas.mean():.2e}  std={deltas.std():.2e}  max={deltas.max():.2e}")
        del rule._hazard_deltas  # clean up

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