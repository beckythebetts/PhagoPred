import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt


def causal_gaussian_smooth(signal: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply causal Gaussian smoothing to a 1-D signal.

    Unlike `gaussian_filter1d` (symmetric kernel), this only uses current and
    past values, so a hazard spike at frame t cannot leak backwards to t-k.
    The kernel weight for a lag of k frames is proportional to exp(-k²/2σ²),
    with maximum weight at lag=0 (current frame) decaying into the past.

    Implementation: convolve with a one-sided half-Gaussian kernel of length
    ~4σ, then take the first len(signal) outputs of the 'full' convolution.

    Args:
        signal (np.ndarray): 1-D input array (e.g. hazard rates), shape (T,).
        sigma (float): Gaussian standard deviation in samples (frames).

    Returns:
        np.ndarray: Smoothed signal of the same shape as input.
    """
    kernel_radius = int(4 * sigma) + 1
    t = np.arange(kernel_radius)
    kernel = np.exp(-0.5 * (t / sigma)**2)
    kernel /= kernel.sum()
    return np.convolve(signal, kernel, mode='full')[:len(signal)]


def causal_exponential_smooth(signal, decay):
    """IIR, exponential decay smoothing"""
    smoothed = np.zeros_like(signal)
    alpha = 1.0 / decay
    for t in range(len(signal)):
        smoothed[t] = alpha * signal[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def forward_shifted_gaussian_smooth(signal: np.ndarray,
                                    sigma: float,
                                    k: float = 2.0) -> np.ndarray:
    """Gaussin funciotn shifted forward by k*sigma"""
    shift = int(k * sigma)
    tail = int(4 * sigma)
    t = np.arange(shift + tail + 1)
    kernel = np.exp(-0.5 * ((t - shift) / sigma)**2)
    kernel /= kernel.sum()
    return np.convolve(signal, kernel, mode='full')[:len(signal)]


class Cell:
    """
    Represents a single synthetic cell with temporal features and a per-timestep hazard function.

    Features are generated stochastically at construction time. Rules are applied
    externally to modify `hazards`, after which `_compute_pmf` derives the PMF and
    CIF from the discrete-time survival formula: PMF(t) = S(t-1) * h(t).

    Attributes:
        T (int): Number of time frames.
        noise_level (float): Amplitude of Gaussian noise added to base features.
        features (dict[str, np.ndarray]): Named temporal feature arrays of shape (T,).
        hazards (np.ndarray): Per-timestep hazard rates of shape (T,), initialised to zero.
    """

    def __init__(self, T: int = 0, noise_level: float = 0.00):
        self.T = T
        self.noise_level = noise_level
        self.features = self._generate_base_features(noise_level)
        self.hazards = self._generate_base_hazards()

    def _compute_pmf(self):
        """
        Compute the probability mass function from the current hazard array.

        Uses the discrete-time survival identity:
            PMF(t) = S(t-1) * h(t),  where S(t) = prod_{s<=t}(1 - h(s))

        Returns:
            np.ndarray: PMF of shape (T,). Sum is <= 1; the remainder is the
                        probability of surviving beyond the observation window.
        """
        assert (
            self.hazards
            >= 0.0).any(), f"Hazards contain negative values {self.hazards}"
        assert (self.hazards
                <= 1.0).any(), "Hazards contain values greater than 1.0"
        hazards = self.hazards
        sf = np.cumprod(1 - hazards)
        pmf = np.zeros(self.T)
        pmf[0] = hazards[0]
        for t in range(1, self.T):
            pmf[t] = sf[t - 1] * hazards[t]
        # assert pmf.sum() <= 1.0, f"PMF sums to more than 1.0: {pmf.sum()}"
        return pmf

    def _generate_base_hazards(self):
        """Return a zero hazard array of shape (T,). Rules add to this."""
        hazards = np.full(self.T, 0.0)
        return hazards

    def _generate_base_features(self, noise_level=0.00):
        """
        Generate a dictionary of named temporal features for this cell.

        Each feature is an independent stochastic process of length T:
            - random_walk: Brownian motion with high volatility.
            - oscillation: Sinusoidal signal with slowly drifting frequency.
            - linear_trend: Random-slope linear ramp with noise.
            - frame_count: Deterministic index [0, T).
            - polynomial_trend: Random degree-7 polynomial evaluated over [0, 1].
            - oscillation + linear: Superposition of oscillation and a linear trend.

        Args:
            noise_level (float): Gaussian noise amplitude applied to each feature.

        Returns:
            dict[str, np.ndarray]: Feature arrays of shape (T,), dtype float32.
        """
        # features = {
        #     '0': 2 + np.random.randn(self.T)*noise_level,
        #     '1': np.abs(np.random.randn(self.T))*noise_level,
        #     '2': np.arange(self.T) + np.random.randn(self.T)*noise_level,
        #     '3': np.random.randn(self.T)*noise_level,
        # }
        features = {
            'random_walk':
            generate_random_walk(self.T, volatility=10),
            'oscillation':
            generate_stochastic_oscillation(self.T,
                                            noise_level=noise_level,
                                            base_freq=0.002,
                                            freq_volatility=0.005,
                                            amplitude=5.0),
            'linear_trend':
            generate_linear_trend(self.T,
                                  slope_range=(0.01, 0.1),
                                  noise_level=noise_level),
            'frame_count':
            np.arange(self.T),
            'polynomial_trend':
            generate_polynomial_trend(self.T, degrees=7),
            'oscillation + linear':
            generate_stochastic_oscillation(self.T,
                                            noise_level=noise_level,
                                            base_freq=0.02,
                                            freq_volatility=0.005,
                                            amplitude=5.0) +
            generate_linear_trend(self.T),
        }
        for key in features:
            features[key] = features[key].astype(np.float32)
        return features

    def __getitem__(self, key):
        return self.features[key]

    def __setitem__(self, key, value):
        self.features[key] = value

    # def normalise_pmf(self):
    #     """Normalise self.pmf to sum to 1 in-place (no-op if already zero)."""
    #     total = np.sum(self.pmf)
    #     if total > 0:
    #         self.pmf /= total

    def apply_observation_window(self, start: int, end: int):
        """
        Mask features outside the observation window [start, end] with NaN.

        Frames before `start` and after `end` are set to NaN to simulate
        late entry and right-censoring in a real experiment.
        """
        for key in self.features:
            self.features[key][:start] = np.nan
            self.features[key][end + 1:] = np.nan
        # for key, val ins elf.f
    def apply_masking(self, probability: float = 0.1):
        """
        Randomly set feature values to NaN to simulate missing measurements.

        Each time point is independently masked with the given probability,
        independently across all features.

        Args:
            probability (float): Per-frame probability of masking (0–1).
        """
        for key in self.features:
            mask = np.random.rand(self.T) < probability
            self.features[key][mask] = np.nan


def generate_random_walk(T: int, volatility: float = 0.01):
    """
    Generate a Gaussian random walk (discrete Brownian motion).

    Args:
        T (int): Number of time steps.
        volatility (float): Standard deviation of each step increment.

    Returns:
        np.ndarray: Cumulative sum of N(0, volatility) increments, shape (T,).
    """
    walk = np.cumsum(np.random.randn(T) * volatility)
    return walk


def generate_stochastic_oscillation(T: int,
                                    noise_level: float = 0.01,
                                    base_freq: float = 0.05,
                                    freq_volatility: float = 0.01,
                                    amplitude: float = 1.0):
    """
    Generate a sinusoidal oscillation with a slowly drifting frequency.

    The instantaneous frequency performs a random walk around `base_freq`,
    clipped to [0.01, 0.2] cycles per frame. Phase is the cumulative sum of
    this frequency, giving a non-stationary oscillation.

    Args:
        T (int): Number of time steps.
        noise_level (float): Amplitude of additive Gaussian observation noise.
        base_freq (float): Starting frequency in cycles per frame.
        freq_volatility (float): Std of per-step frequency drift.
        amplitude (float): Peak amplitude of the sinusoid.

    Returns:
        np.ndarray: Oscillation signal of shape (T,).
    """
    freq = base_freq + np.cumsum(np.random.randn(T) * freq_volatility)
    freq = np.clip(freq, 0.01, 0.2)  # keep frequency reasonable
    phase = np.cumsum(freq)
    return amplitude * np.sin(
        2 * np.pi * phase) + np.random.randn(T) * noise_level


def generate_linear_trend(T: int,
                          slope_range: tuple = (0.01, 0.1),
                          noise_level: float = 0.01):
    """
    Generate a linear ramp with a random slope and additive noise.

    Args:
        T (int): Number of time steps.
        slope_range (tuple[float, float]): (min, max) range for the slope drawn uniformly.
        noise_level (float): Std of additive Gaussian noise.

    Returns:
        np.ndarray: Linear trend of shape (T,).
    """
    slope = np.random.uniform(*slope_range)
    trend = slope * np.arange(T)
    return trend + np.random.randn(T) * noise_level


def generate_polynomial_trend(
    T: int,
    degrees: int = 7,
):
    """
    Generate a random polynomial trend evaluated over [0, 1].

    Coefficients are drawn i.i.d. from N(0, 1). The input is normalised to
    [0, 1] for numerical stability before evaluating with `np.polyval`.

    Args:
        T (int): Number of time steps.
        degrees (int): Degree of the polynomial (number of coefficients = degrees + 1).
        noise_level (float): Std of additive Gaussian noise.

    Returns:
        np.ndarray: Polynomial trend of shape (T,).
    """
    x = np.linspace(0, 1, T)  # normalize to [0, 1] for stability
    coeffs = np.random.randn(degrees + 1)
    trend = np.polyval(coeffs, x)
    return trend


class Rule:
    """
    Abstract base class for rules that modify a Cell's hazard array (and optionally features).

    Subclasses implement `apply`, which receives a Cell and modifies `cell.hazards`
    in-place. Rules may also modify `cell.features` to inject feature signals that
    are causally linked to the hazard change.
    """

    def apply(self, cell: Cell) -> None:
        """Modify cell hazards (and optionally features) in-place."""
        pass


# == Hazard only rules ==
class ThresholdRule(Rule):
    """
    Increase hazard at all frames where a feature exceeds a fixed threshold.

    Applied with probability `probability` per cell, so only a fraction of
    cells will be affected by this rule.

    Args:
        feature (str): Feature name in cell.features to threshold.
        threshold (float): Value above which hazard is increased.
        hazard_increase (float): Amount added to hazard at each exceeding frame.
        probability (float): Probability that this rule applies to a given cell.
    """

    def __init__(self,
                 feature: str = '0',
                 threshold: float = 1.0,
                 hazard_increase: float = 0.1,
                 probability: float = 1.0):
        self.feature = feature
        self.threshold = threshold
        self.hazard_increase = hazard_increase
        self.probability = probability

    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            exceed_idxs = np.where(cell[self.feature] > self.threshold)[0]
            cell.hazards[exceed_idxs] += self.hazard_increase


class CumulativeEffectRule(Rule):
    """
    Apply a monotonically increasing hazard proportional to cumulative feature exceedance.

    The hazard contribution at frame t equals `hazard_increase * cumsum(feature > threshold)[t]`,
    so the hazard ramps up whenever the feature is above threshold and never decreases.
    This models cumulative damage or stress accumulation.

    Args:
        feature (str): Feature name to evaluate.
        threshold (float): Exceedance threshold.
        hazard_increase (float): Hazard increment per frame of exceedance.
        probability (float): Probability that this rule applies to a given cell.
    """

    def __init__(self,
                 feature: str = '0',
                 threshold: float = 1.0,
                 hazard_increase: float = 0.05,
                 probability: float = 1.0):
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
    """
    Modify hazard at frames where two features simultaneously exceed their thresholds.

    Models synergistic or antagonistic interactions between two signals. A negative
    `hazard_increase` is allowed (protective interaction) but will be clipped to
    zero during the final hazard clipping step in `create_synthetic_dataset`.

    Args:
        feature1 (str): First feature name.
        threshold1 (float): Threshold for feature1.
        feature2 (str): Second feature name.
        threshold2 (float): Threshold for feature2.
        hazard_increase (float): Hazard delta applied where both thresholds are exceeded.
            Can be negative to model protective interactions.
        probability (float): Probability that this rule applies to a given cell.
    """

    def __init__(self,
                 feature1: str = '0',
                 threshold1: float = 1.0,
                 feature2: str = '1',
                 threshold2: float = 1.0,
                 hazard_increase: float = 0.1,
                 probability: float = 1.0):
        self.feature1 = feature1
        self.threshold1 = threshold1
        self.feature2 = feature2
        self.threshold2 = threshold2
        self.hazard_increase = hazard_increase
        self.probability = probability

    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            interaction_mask = (cell[self.feature1] > self.threshold1) & (
                cell[self.feature2] > self.threshold2)
            cell.hazards[interaction_mask] += self.hazard_increase


class RandomSpikeRule(Rule):
    """
    Add a Gaussian hazard spike at a random time point with probability `probability`.

    The spike is placed at a uniformly sampled frame and shaped by `get_gaussian_curve`
    with sigma=5. This models sudden, unpredictable stress events with no associated
    feature signal, introducing irreducible noise into the hazard.

    Args:
        height (float): Scaling factor applied to the Gaussian spike.
        probability (float): Per-cell probability of a spike occurring.
    """

    def __init__(self, height: float = 0.5, probability: float = 0.1):
        self.height = height
        self.probability = probability

    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            spike_time = np.random.randint(cell.T)
            cell.hazards += get_gaussian_curve(cell.T, spike_time,
                                               sigma=5) * self.height


class GradientRule(Rule):
    """
    Increase hazard proportionally to the absolute gradient of a feature.

    The normalised gradient magnitude (divided by its maximum) is scaled by
    `max_increase` and added to the hazard at every frame. This models scenarios
    where rapid feature changes (not just high values) are predictive of death.

    Args:
        feature (str): Feature name whose gradient drives the hazard.
        gradient_threshold (float): 
        max_increase (float): Maximum hazard contribution (at the frame of steepest gradient).
        probability (float): Probability that this rule applies to a given cell.
    """

    def __init__(self,
                 feature: str = '0',
                 gradient_threshold: float = 0.0,
                 max_increase: float = 0.5,
                 probability: float = 1.0):
        self.feature = feature
        self.gradient_threshold = gradient_threshold
        self.max_increase = max_increase
        self.probability = probability

    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:
            feature_gradient = np.diff(cell[self.feature],
                                       prepend=cell[self.feature][0])
            mask = feature_gradient > self.gradient_threshold
            cell.hazards[mask] += np.clip(np.abs(feature_gradient[mask]), 0,
                                          self.max_increase)
            # cell.hazards += np.abs(
            #     (feature_gradient /
            #      np.max(np.abs(feature_gradient)))) * self.max_increase
            # mask = feature_gradient > self.gradient_threshold
            # cell.hazards[mask] += feature_gradient[mask] / np.max(feature_gradient) * self.max_increase


# == Feature modifying rules (with delayed hazard effect) ==
class VariationRule(Rule):
    """
    Inject noise into a feature window and raise hazard `delay` frames later.

    A 100-frame window ending `delay` frames before a randomly chosen event frame
    has its variance increased by additive Gaussian noise. The hazard then peaks
    at the event frame via a Gaussian curve. This models scenarios where increased
    feature variability precedes cell death with a fixed lag.

    Args:
        feature (str): Feature to perturb with added noise.
        probability (float): Per-cell probability of applying this rule.
        delay (int): Number of frames between the noise injection and the hazard peak.
        sigma (float): Width (std) of the Gaussian hazard peak in frames.
        max_strength (float): Upper bound on the randomly sampled noise/hazard amplitude.
    """

    def __init__(self,
                 feature: str = '0',
                 probability: float = 1.0,
                 delay: int = 200,
                 sigma: float = 20.0,
                 max_strength=1.0):
        self.feature = feature
        self.probability = probability
        self.delay = delay
        self.sigma = sigma
        self.max_strength = max_strength

    def apply(self, cell: Cell) -> None:
        if np.random.rand() < self.probability:

            strength = np.random.rand() * self.max_strength

            frame = np.random.randint(cell.T)
            start_idx = frame - self.delay
            if start_idx < 0:
                return
            slice_ = cell[self.feature][start_idx:start_idx + 100]
            slice_ += np.random.randn(len(slice_)) * strength * 100
            cell[self.feature][start_idx:start_idx + 100] = slice_

            t = np.arange(cell.T)
            cell.hazards += get_gaussian_curve(cell.T, frame,
                                               self.sigma) * strength


class GradualRampRule(Rule):
    """
    Add a linear ramp to a feature followed by a hazard peak after a fixed delay.

    A ramp of length `ramp_length` is added to the feature starting at a random
    frame. The hazard then peaks at `ramp_end + delay` frames via a Gaussian curve,
    simulating a gradual physiological change that culminates in a death event.

    Args:
        feature (str): Feature to which the ramp is added.
        probability (float): Per-cell probability of applying this rule.
        ramp_length (int): Duration in frames of the linear ramp.
        ramp_height (float): Maximum height of the ramp (before strength scaling).
        delay (int): Frames between the end of the ramp and the hazard peak.
        sigma (float): Width (std) of the Gaussian hazard peak in frames.
        max_strength (float): Upper bound for the strength multiplier; drawn as
            Uniform(0,1)^0.1 to bias towards high strength.
    """

    def __init__(self,
                 feature='1',
                 probability=1.0,
                 ramp_length=30,
                 ramp_height=10.0,
                 delay=100,
                 sigma=30.0,
                 max_strength: float = 1.0):
        self.feature = feature
        self.probability = probability
        self.ramp_length = ramp_length
        self.ramp_height = ramp_height
        self.delay = delay
        self.sigma = sigma
        self.max_strength = max_strength

    def apply(self, cell: Cell):
        if np.random.rand() < self.probability:
            strength = (np.random.rand()**0.1) * self.max_strength
            # start = np.random.randint(cell.T - self.ramp_length)
            start = np.random.randint(cell.T)
            ramp = np.linspace(0, self.ramp_height * strength,
                               self.ramp_length)
            ramp_end = min(start + self.ramp_length, cell.T)
            cell[self.feature][start:ramp_end] += ramp[:ramp_end - start]
            t = np.arange(cell.T)
            hazard_frame = start + self.ramp_length + self.delay

            cell.hazards += get_gaussian_curve(cell.T, hazard_frame,
                                               self.sigma) * strength


def get_gaussian_curve(T: int, center: int, sigma: float):
    """
    Return a scaled Gaussian curve of length T centred at `center`.

    The curve is computed as a normalised Gaussian density then multiplied by 5
    to give a usable hazard amplitude. The resulting peak height is approximately
    5 / (sigma * sqrt(2*pi)).

    Args:
        T (int): Length of the output array.
        center (int): Frame index of the Gaussian peak.
        sigma (float): Standard deviation in frames.

    Returns:
        np.ndarray: Gaussian curve of shape (T,).
    """
    t = np.arange(T)
    gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
    gaussian /= (sigma * np.sqrt(2 * np.pi))
    return gaussian * 5


def pmf_from_hazards(hazards: np.ndarray):
    """
    Compute a survival PMF from a per-timestep hazard array.

    Equivalent to Cell._compute_pmf but operates on a bare numpy array.

    Args:
        hazards (np.ndarray): Per-timestep hazard rates in [0, 1], shape (T,).

    Returns:
        np.ndarray: PMF of shape (T,). Sum is <= 1; the remainder represents
                    right-censored probability mass.
    """
    sf = np.cumprod(1 - hazards)
    pmf = np.zeros(len(hazards))
    pmf[0] = hazards[0]
    for t in range(1, len(hazards)):
        pmf[t] = sf[t - 1] * hazards[t]
    # assert pmf.sum() <= 1.0, f"PMF sums to more than 1.0: {pmf.sum()}"
    return pmf


def add_feature_noise(feature_array: np.ndarray,
                      noise_level: float) -> np.ndarray:
    """Add gaussian noise to feature"""
    return np.random.randn(*feature_array.shape) * noise_level + feature_array


# def causal_smooth(signal: np.ndarray, sigma: int) -> np.ndarray:
#     """Apply gaussian smoothing, shifted forards by 2*sigma to prevent increased hazard prior to cause of increase."""


def create_synthetic_dataset(filename: Path,
                             num_cells: int = 1000,
                             num_frames: int = 1000,
                             rules: list = None,
                             noise_level: float = 0.01,
                             late_entry_prob: float = 0.0,
                             late_entry_range: tuple = (0, 100),
                             feature_mask_prob: float = 0.0,
                             seed: int = None):
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
            ThresholdRule(feature='random_walk',
                          threshold=100,
                          hazard_increase=5e-3,
                          probability=1.0),
            CumulativeEffectRule(feature='oscillation',
                                 threshold=4.0,
                                 hazard_increase=1e-5,
                                 probability=1.0),
            InteractionRule(feature1='random_walk',
                            threshold1=5.0,
                            feature2='oscillation',
                            threshold2=2.0,
                            hazard_increase=-1e-2,
                            probability=1.0),
            # RandomSpikeRule(height=4e-2, probability=0.3),
            GradientRule(feature='polynomial_trend',
                         gradient_threshold=0.0,
                         max_increase=5e-3,
                         probability=1.0),
            # GradualRampRule(feature='oscillation',
            #                 ramp_length=50,
            #                 ramp_height=50,
            #                 delay=100,
            #                 sigma=30,
            #                 max_strength=0.5)
        ]

    num_rules = len(rules) if rules else 1
    # for rule in rules:
    #     rule.max_strength = 1 / num_rules
    #     print(f"Rule {rule.__class__.__name__}: max_strength={rule.max_strength:.3f}")

    # Configure start frames (late entry)
    start_frames = np.zeros(num_cells, dtype=int)
    if late_entry_prob > 0:
        late_entry_mask = np.random.rand(num_cells) < late_entry_prob
        num_late_entry = np.sum(late_entry_mask)
        start_frames[late_entry_mask] = np.random.randint(late_entry_range[0],
                                                          late_entry_range[1],
                                                          size=num_late_entry)
        print(
            f"Late entry: {num_late_entry}/{num_cells} cells ({100*late_entry_prob:.1f}%)"
        )
    else:
        start_frames[:] = np.random.randint(0, 10, size=num_cells)

    end_frames = np.random.randint(num_frames // 2, num_frames, size=num_cells)

    features = Cell().features.keys()

    all_features = {
        name: np.empty((num_frames, num_cells), dtype=np.float32)
        for name in features
    }
    all_deaths = np.empty(num_cells, dtype=np.float32)
    cifs = np.empty((num_frames, num_cells), dtype=np.float32)
    pmfs = np.empty((num_frames, num_cells), dtype=np.float32)
    hazards = np.empty((num_frames, num_cells), dtype=np.float32)
    hazard_contributions = {
        rule.__class__.__name__:
        np.empty((num_frames, num_cells), dtype=np.float32)
        for rule in rules
    }

    for c in tqdm(range(num_cells), desc='Generating cells'):
        cell = Cell(num_frames, noise_level=0.0)
        start = start_frames[c]
        end = end_frames[c]

        # rule_contributions = np.zeros((num_rules, cell.T), dtype=np.float32)
        # Apply rules and track per-rule hazard contribution
        for rule in rules:
            hazards_before = cell.hazards.copy()
            rule.apply(cell)
            hazard_delta = cell.hazards - hazards_before
            hazard_contributions[
                rule.__class__.__name__][:, c] = causal_gaussian_smooth(
                    hazard_delta, sigma=50)
            # rule_contributions[rules.index(rule)] = hazard_delta

        # cell.hazards = gaussian_filter1d(cell.hazards, sigma=50)
        cell.hazards = forward_shifted_gaussian_smooth(cell.hazards, 10)
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

    with h5py.File(filename, 'w') as f:
        grp = f.create_group('Cells/Phase')
        for name in features:
            grp.create_dataset(name, data=all_features[name], dtype=np.float32)
        for rule in rules:
            rule_dataset = grp.create_dataset(
                f"HazardContribution_{rule.__class__.__name__}",
                data=hazard_contributions[rule.__class__.__name__],
                dtype=np.float32)
            for attr, value in rule.__dict__.items():
                rule_dataset.attrs[attr] = value
        grp.create_dataset('CellDeath',
                           data=all_deaths[np.newaxis, :],
                           dtype=np.float32)
        grp.create_dataset('CIFs', data=cifs, dtype=np.float32)
        grp.create_dataset('PMFs', data=pmfs, dtype=np.float32)
        grp.create_dataset('Hazards', data=hazards, dtype=np.float32)


if __name__ == '__main__':
    causal_gaussian_smooth(np.array([0, 0]), sigma=50)
    # create_synthetic_dataset(
    #     filename=Path('PhagoPred') / 'Datasets' / 'val_synthetic.h5',
    #     num_cells=1000,
    #     num_frames=1000,
    # )
    # create_synthetic_dataset(
    #     filename=Path('PhagoPred') / 'Datasets' / 'synthetic.h5',
    #     num_cells=1000,
    #     num_frames=1000,
    # )
