from __future__ import annotations
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm

from PhagoPred.utils.logger import get_logger
from .graph import CausalGraph, Feature
from .noise_funcs import Noise, GaussianNoise, NoNoise
from .rules.rules import SEMRule
from .rules.structural_forms import StructuralForm, SingleIndex
from .rules.component_funcs import ComponentFunc, Linear
from .noise_funcs import Noise, GaussianNoise, LaplaceNoise, NoNoise
from .rules.component_funcs import Linear, Hill
from .rules.structural_forms import (
    SingleIndex,
    AdditiveNonlinear,
    MultiplicativeInteraction,
    AdditiveWithInteraction,
)

log = get_logger()


class HazardModel:

    def __init__(self, num_input_edges: int,
                 structural_rules: list[StructuralForm],
                 component_funcs: list[ComponentFunc], noise: Noise):
        self.num_input_edges = num_input_edges
        self.strcutural_rules = structural_rules
        self.component_funcs = component_funcs
        self.noise = noise

        self.hazard = None
        self.survival = None
        self.pmf = None
        self.cif = None
        self.cell_deaths = None

    def compute_all(self, in_hazard: np.ndarray) -> None:
        """Transform hazard to range (0, 1) with sigmoid and compute survival, pmf, CIF."""
        self.hazard = 1.0 / (1.0 + np.exp(-in_hazard))
        self.survival = np.cumprod(1.0 - self.hazard, axis=0)
        s_prev = np.vstack(
            [np.ones((1, self.survival.shape[1])), self.survival[:-1]])
        self.pmf = self.hazard * s_prev
        self.cif = 1.0 - self.survival

    def sample_death_times(self) -> np.ndarray:
        if self.pmf is None:
            return None
        _, samples = self.pmf.shape
        cell_death = np.full(samples, np.nan, dtype=float)

        for i in range(samples):
            u = np.random.rand()
            if u > np.max(self.cif):
                continue
            death_frame = np.argmax(self.cif >= u)
            cell_death[i] = death_frame

        return cell_death


@dataclass
class HazardModelOld:

    weights: dict[str, float] | None = None
    baseline: float = -4.0
    target_death_fraction: float = 0.5
    seed: int | None = None

    def get_weights(self, feature_names: list[str],
                    rng: np.random.Generator) -> dict[str, float]:
        if self.weights is not None:
            return self.weights
        w = rng.uniform(-1.0, 1.0, size=len(feature_names))
        return dict(zip(feature_names, w))

    def _linear_predictor(self,
                          signals: np.ndarray,
                          feature_names: list[str],
                          weights: dict[str, float],
                          offset: float = 0.0) -> np.ndarray:
        frames, samples, _ = signals.shape
        predictor = np.full((frames, samples), self.baseline + offset)
        for fi, name in enumerate(feature_names):
            if name in weights:
                predictor += weights[name] * signals[:, :, fi]
        return predictor

    def _expected_death_fraction(self, predictor: np.ndarray,
                                 last_frames: np.ndarray) -> float:
        """Mean CIF at each sample's last observed frame."""
        hazard = 1.0 / (1.0 + np.exp(-predictor))
        survival = np.cumprod(1.0 - hazard, axis=0)
        cif = 1.0 - survival
        return float(
            np.mean([cif[last_frames[i], i] for i in range(len(last_frames))]))

    def _calibrate_offset(self, signals: np.ndarray, feature_names: list[str],
                          weights: dict[str, float],
                          last_frames: np.ndarray) -> float:
        """Binary search on a baseline offset to hit target_death_fraction."""
        lo, hi = -20.0, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            pred = self._linear_predictor(signals, feature_names, weights, mid)
            frac = self._expected_death_fraction(pred, last_frames)
            if frac < self.target_death_fraction:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def compute(
        self,
        signals: np.ndarray,
        feature_names: list[str],
        last_frames: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute hazard, survival, PMF, CIF with a calibrated baseline offset
        so that approximately `target_death_fraction` of cells die within their
        observed window.

        Args:
            signals:       (frames, samples, features) — clean, pre-NaN signals.
            feature_names: feature names matching last axis of signals.
            last_frames:   (samples,) last observed frame per sample.
            rng:           random generator.

        Returns:
            hazard, survival, pmf, cif — each (frames, samples).
        """
        weights = self.get_weights(feature_names, rng)
        samples = signals.shape[1]

        offset = self._calibrate_offset(signals, feature_names, weights,
                                        last_frames)
        predictor = self._linear_predictor(signals, feature_names, weights,
                                           offset)

        hazard = 1.0 / (1.0 + np.exp(-predictor))
        survival = np.cumprod(1.0 - hazard, axis=0)
        s_prev = np.vstack([np.ones((1, samples)), survival[:-1]])
        pmf = hazard * s_prev
        cif = 1.0 - survival

        return hazard, survival, pmf, cif

    def sample_death_times(self, pmf: np.ndarray, first_frames: np.ndarray,
                           last_frames: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
        """
        Sample a death frame for each cell from its PMF within its observed window.
        Cells whose window PMF sums to p_die are censored with probability 1-p_die.

        Returns:
            cell_death: (samples,) float array, 0 = right-censored.
        """
        _, samples = pmf.shape
        cell_death = np.full(samples, np.nan, dtype=float)

        for i in range(samples):
            t0, t1 = int(first_frames[i]), int(last_frames[i])
            window_pmf = np.maximum(pmf[t0:t1 + 1, i], 0)
            p_die = window_pmf.sum()
            if p_die <= 0:
                continue
            u = rng.uniform(0, 1)
            if u > p_die:
                continue  # right-censored: survived the window
            # Conditional distribution given death in window
            chosen = rng.choice(t1 - t0 + 1, p=window_pmf / p_die)
            cell_death[i] = t0 + chosen

        return cell_death


@dataclass
class GraphGeneratedDataset:
    """Configuration for a CausalGraph-based synthetic dataset.

    Args:
        num_features:           number of feature nodes in the graph.
        num_samples:            number of independent cell trajectories to simulate.
        num_frames:             length of each full trajectory.
        process_noise_type:     list of Noise instances for pre_noise, cycled per feature.
        measurement_noise_type: list of Noise instances for post_noise, cycled per feature.
        num_edges:              number of directed causal edges to randomly wire.
        rules_component_funcs:  list of ComponentFunc instances cycled per edge,
                                used inside a SingleIndex structural form.
        rules_structural_forms: optional list of StructuralForm *classes* (not instances)
                                cycled per edge.  If provided, overrides
                                rules_component_funcs for structural form selection.
                                Each class is instantiated as cls({source: 1.0}).
        lag_range:              (min, max) uniform range for random edge lags.
        self_edge_prob:         probability each feature gets an AR(1) self-edge.
                                Coefficient is sampled uniformly in (0.3, 0.8)
                                to ensure stationarity.
        truncation_prob:        fraction of samples to truncate (late start or early end).
        truncation_range:       (min, max) frames removed from start or end.
        seed:                   random seed.
    """
    num_features: int = 5
    num_samples: int = 100
    num_frames: int = 200
    process_noise_type: list[Noise] = field(
        default_factory=lambda: [GaussianNoise(0.1)])
    measurement_noise_type: list[Noise] = field(
        default_factory=lambda: [NoNoise()])
    num_edges: int = 3
    rules_component_funcs: list[ComponentFunc] = field(
        default_factory=lambda: [Linear(slope=0.5)])
    rules_structural_forms: list[type[StructuralForm]] | None = None
    lag_range: tuple[int, int] = (1, 5)
    self_edge_prob: float = 0.8
    self_edge_weight_range: tuple[float] = (0.9, 0.95)
    # hazard_model: HazardModel | None = HazardModel({
    #     'feature_0': 0.5,
    #     'feature_1': 0.5
    # })
    truncation_prob: float = 0.3
    max_truncation: int = 50
    seed: int | None = None
    hazard_model: HazardModel | None = HazardModel(
        3,
        structural_rules=None,
        component_funcs=[Linear(), Hill()],
        noise=GaussianNoise(1),
    )

    def _make_features(self) -> list[Feature]:
        log.info('Creating features')
        pre_noise_cycle = cycle(self.process_noise_type)
        post_noise_cycle = cycle(self.measurement_noise_type)
        return [
            Feature(
                name=f'feature_{i}',
                pre_noise=next(pre_noise_cycle),
                post_noise=next(post_noise_cycle),
            ) for i in range(self.num_features)
        ]

    def _make_hazard(self) -> Feature:
        log.info('Adding hazard')
        return Feature('Hazard',
                       pre_noise=self.hazard_model.noise,
                       post_noise=NoNoise())

    def _make_rules(self, features: list[Feature]) -> list[SEMRule]:
        log.info('Creating rules')
        rng = np.random.default_rng(self.seed)
        names = [f.name for f in features]

        rules = []
        # Add self edges
        for name in names:
            if rng.random() < self.self_edge_prob:
                coeff = float(
                    rng.uniform(low=self.self_edge_weight_range[0],
                                high=self.self_edge_weight_range[1]))
                rules.append(
                    SEMRule(
                        inputs=[name],
                        target=name,
                        structural_form=SingleIndex(weights={name: coeff},
                                                    func=Linear()),
                        lag=1,
                    ))

        # Sample unique directed cross-edges (no self-loops)
        all_edges = [(s, t) for s in names for t in names if s != t]
        num_edges = min(self.num_edges, len(all_edges))
        chosen = [
            all_edges[i]
            for i in rng.choice(len(all_edges), size=num_edges, replace=False)
        ]

        form_cycle = cycle(self.rules_structural_forms) \
            if self.rules_structural_forms is not None else None
        func_cycle = cycle(self.rules_component_funcs)

        for source, target in chosen:
            lag = int(rng.integers(self.lag_range[0], self.lag_range[1] + 1))
            weight = float(rng.choice([-1, 1])) * rng.uniform(0.3, 1.0)
            func = next(func_cycle)
            form_cls = next(
                form_cycle) if form_cycle is not None else SingleIndex

            if form_cls is SingleIndex:
                structural_form = SingleIndex(weights={source: weight},
                                              func=func)
            else:
                structural_form = form_cls(funcs={source: func})
            rules.append(
                SEMRule(inputs=[source],
                        target=target,
                        structural_form=structural_form,
                        lag=lag))

        # Sample rules to hazard
        form_cycle = cycle(
            self.hazard_model.strcutural_rules
        ) if self.hazard_model.strcutural_rules is not None else None
        func_cycle = cycle(self.hazard_model.component_funcs)

        for _ in range(self.hazard_model.num_input_edges):
            input_feat = np.random.choice(names)
            func = next(func_cycle)
            form_cls = next(
                form_cycle) if form_cycle is not None else SingleIndex

            if form_cls is SingleIndex:
                structural_form = SingleIndex(weights={input_feat: weight},
                                              func=func)
            else:
                structural_form = form_cls(funcs={input_feat: func})

            rules.append(
                SEMRule([input_feat],
                        'Hazard',
                        structural_form,
                        lag=int(
                            rng.integers(self.lag_range[0],
                                         self.lag_range[1] + 1))))
        return rules

    def _sample_truncation_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a single (first_frame, last_frame) window per sample.
        The same window is applied to every feature for consistency.

        Returns:
            first_frame: (samples,) int array
            last_frame:  (samples,) int array
        """
        log.info('Sampling truncation windows')
        first_frames = np.random.uniform(0, self.max_truncation,
                                         self.num_samples)
        first_frames_mask = np.random.uniform(
            0, 1, self.num_samples) < self.truncation_prob
        first_frames[~first_frames_mask] = 0

        last_frames = np.random.uniform(self.num_frames - self.max_truncation,
                                        self.num_frames, self.num_samples)
        last_frames_mask = np.random.uniform(
            0, 1, self.num_samples) < self.truncation_prob
        last_frames[~last_frames_mask] = self.num_frames
        return first_frames.astype(int), last_frames.astype(int)

    def generate_and_save(self, save_path: Path) -> None:
        """Simulate `num_samples` trajectories and write to an h5 file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        features = self._make_features()
        hazard_feat = self._make_hazard()
        rules = self._make_rules(features)
        graph = CausalGraph(features=features + [hazard_feat],
                            rules=rules,
                            time_steps=self.num_frames)

        log.info('Generating signals')

        all_signals = {f.name: [] for f in features + [hazard_feat]}
        for _ in tqdm(range(self.num_samples), desc='Sampling graph'):
            sample_dict = graph.sample_graph()
            for f in features + [hazard_feat]:
                all_signals[f.name].append(sample_dict[f.name])
        for key, val in all_signals.items():
            all_signals[key] = np.stack(val, axis=-1)  # (frames, samples)
        hazard = all_signals['Hazard']

        first_frames, last_frames = self._sample_truncation_windows()

        self.hazard_model.hazard = hazard
        self.hazard_model.compute_all(hazard)
        death_frames = self.hazard_model.sample_death_times()

        frame_idxs = np.arange(self.num_frames)[:, np.newaxis]
        observed = (frame_idxs > first_frames) & (frame_idxs < last_frames)
        for feat in features:
            all_signals[feat.name][~observed] = np.nan

        death_observed = (death_frames > first_frames) & (death_frames
                                                          < last_frames)
        death_frames[~death_observed] = np.nan

        with h5py.File(save_path, 'w') as f:
            log.info(f'Writing cells to {save_path}')
            phase = f.require_group('Cells/Phase')
            for feature in features:
                phase.create_dataset(
                    feature.name,
                    data=all_signals[feature.name])  # (frames, samples)

            if self.hazard_model is not None:
                # Save survival quantities as extra features (frames, samples)
                phase.create_dataset('Hazard', data=self.hazard_model.hazard)
                phase.create_dataset('Survival',
                                     data=self.hazard_model.survival)
                phase.create_dataset('PMF', data=self.hazard_model.pmf)
                phase.create_dataset('CIF', data=self.hazard_model.cif)

            phase.create_dataset('First Frame',
                                 data=first_frames.reshape(1,
                                                           -1).astype(float))
            phase.create_dataset('Last Frame',
                                 data=last_frames.reshape(1, -1).astype(float))
            phase.create_dataset('CellDeath', data=death_frames.reshape(1, -1))

        print(f'Saved {self.num_samples} samples × {self.num_frames} frames × '
              f'{self.num_features} features → {save_path}')
        print(f'  Edges: {[(r.inputs[0], r.target, r.lag) for r in rules]}')


def generate_datasets(output_dir: Path = Path(
    'PhagoPred/Datasets/graph_synthetic')) -> None:
    """Generate a set of graph-synthetic datasets with varied configurations."""

    output_dir = Path(output_dir)

    configs: dict[str, GraphGeneratedDataset] = {
        'baseline':
        GraphGeneratedDataset(
            process_noise_type=[GaussianNoise(0.1)],
            measurement_noise_type=[NoNoise()],
            num_edges=4,
            rules_component_funcs=[Linear(slope=0.5)],
            lag_range=(1, 5),
            truncation_prob=0.3,
            max_truncation=50,
            seed=42,
        ),
        'laplace_noise':
        GraphGeneratedDataset(
            process_noise_type=[LaplaceNoise(0.1)],
            measurement_noise_type=[NoNoise()],
            num_edges=4,
            rules_component_funcs=[Linear(slope=0.5)],
            lag_range=(1, 5),
            truncation_prob=0.3,
            max_truncation=50,
            seed=42,
        ),
        'mixed_noise':
        GraphGeneratedDataset(
            num_features=6,

            # Cycles: features 0,2,4 get Gaussian; features 1,3,5 get Laplace
            process_noise_type=[GaussianNoise(1.0),
                                LaplaceNoise(1.0)],
            measurement_noise_type=[NoNoise()],
            num_edges=5,
            rules_component_funcs=[Linear(slope=0.5)],
            lag_range=(1, 5),
            truncation_prob=0.3,
            max_truncation=50,
            seed=43,
        ),
        'nonlinear_edges':
        GraphGeneratedDataset(
            process_noise_type=[GaussianNoise(0.1)],
            measurement_noise_type=[NoNoise()],
            num_edges=4,
            # Cycles: alternate linear and Hill (sigmoid-like) edges
            rules_component_funcs=[Linear(slope=0.5),
                                   Hill(k=1.0, n=2.0)],
            lag_range=(1, 5),
            truncation_prob=0.3,
            max_truncation=50,
            seed=44,
        ),
        'long_lags':
        GraphGeneratedDataset(
            process_noise_type=[GaussianNoise(0.1)],
            measurement_noise_type=[NoNoise()],
            num_edges=4,
            rules_component_funcs=[Linear(slope=0.5)],
            lag_range=(10, 20),
            truncation_prob=0.3,
            max_truncation=50,
            seed=45,
        ),
        'dense_graph':
        GraphGeneratedDataset(
            process_noise_type=[GaussianNoise(0.1)],
            measurement_noise_type=[NoNoise()],
            num_edges=10,
            rules_component_funcs=[Linear(slope=0.3)],
            lag_range=(1, 5),
            truncation_prob=0.3,
            max_truncation=50,
            seed=46,
        ),
        'heavy_truncation':
        GraphGeneratedDataset(
            process_noise_type=[GaussianNoise(0.1)],
            measurement_noise_type=[NoNoise()],
            num_edges=4,
            rules_component_funcs=[Linear(slope=0.5)],
            lag_range=(1, 5),
            truncation_prob=0.7,
            max_truncation=99,
            seed=47,
        ),
        # ── Structural form variants ──────────────────────────────────────────
        'additive_nonlinear':
        GraphGeneratedDataset(
            # Each edge applies an independent nonlinear function per parent:
            # effect = f(source), where f cycles through [Linear, Hill]
            process_noise_type=[GaussianNoise(0.1)],
            measurement_noise_type=[NoNoise()],
            num_edges=4,
            rules_structural_forms=[AdditiveNonlinear],
            rules_component_funcs=[Linear(slope=0.5),
                                   Hill(k=1.0, n=2.0)],
            lag_range=(1, 5),
            truncation_prob=0.3,
            max_truncation=50,
            seed=48,
        ),
        'multiplicative':
        GraphGeneratedDataset(
            # Each edge applies a product of per-parent functions:
            # effect = f(source)   (degenerate to single-input here,
            # but uses the multiplicative gate form)
            process_noise_type=[GaussianNoise(0.1)],
            measurement_noise_type=[NoNoise()],
            num_edges=4,
            rules_structural_forms=[MultiplicativeInteraction],
            rules_component_funcs=[Hill(k=1.0, n=2.0)],
            lag_range=(1, 5),
            truncation_prob=0.3,
            max_truncation=50,
            seed=49,
        ),
        'mixed_structural_forms':
        GraphGeneratedDataset(
            # Cycles: edge 0 → SingleIndex, edge 1 → AdditiveNonlinear,
            #          edge 2 → MultiplicativeInteraction, edge 3 → SingleIndex ...
            # Component funcs also cycle: Linear, Hill, Linear, Hill ...
            num_features=6,
            process_noise_type=[GaussianNoise(1.0),
                                LaplaceNoise(1.0)],
            measurement_noise_type=[NoNoise()],
            num_edges=6,
            rules_structural_forms=[
                SingleIndex, AdditiveNonlinear, MultiplicativeInteraction
            ],
            rules_component_funcs=[Linear(slope=0.5),
                                   Hill(k=1.0, n=2.0)],
            lag_range=(1, 5),
            truncation_prob=0.3,
            max_truncation=50,
            seed=50,
        ),
    }

    for name, config in tqdm(configs.items(), desc="Generating datasets"):
        print(f'\n{"="*60}')
        print(f'Generating: {name}')
        config.generate_and_save(output_dir / f'{name}.h5')

    print(f'\nAll datasets saved to {output_dir}')


if __name__ == '__main__':
    generate_datasets()
