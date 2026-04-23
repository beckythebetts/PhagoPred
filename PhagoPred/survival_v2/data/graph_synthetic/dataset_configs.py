from dataclasses import dataclass, fields
from pathlib import Path

from . import base_funcs, noise_funcs
from .generate_datasets import generate_dataset
from .graph import Feature
from .rules import Apply, Rule, ReLU, Var, threshold


@dataclass
class DatasetConfig:
    """All parameters needed to generate one dataset."""
    filename: str
    features: list
    rules: list
    num_cells: int = 1000
    num_frames: int = 500
    late_entry_prob: float = 0.0
    late_entry_range: tuple = (0, 100)
    feature_mask_prob: float = 0.0
    smooth_hazard: bool = True
    smooth_sigma: int = 10
    seed: int = None
    target_death_fraction: float = 0.5

    def generate(self, save_dir: Path) -> None:
        """Generate and save the dataset to save_dir/<filename>.h5."""
        kwargs = {
            f.name: getattr(self, f.name)
            for f in fields(self) if f.name != 'filename'
        }
        generate_dataset(filename=Path(save_dir) / f'{self.filename}.h5',
                         **kwargs)


# ---------------------------------------------------------------------------
# Config 1: simple linear chain  A -> B -> Hazard
# A drifts as an AR(1); B is driven by A at lag 1; Hazard tracks positive B.
# ---------------------------------------------------------------------------
_simple_chain = DatasetConfig(
    filename='simple_chain',
    features=[
        Feature('A',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(1.0)),
        Feature('B',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(0.5)),
        Feature('Hazard',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.NoNoise()),
    ],
    rules=[
        Rule(target='B', expr=0.8 * Var('B') + 0.5 * Var('A')),
        Rule(target='Hazard', expr=Apply(ReLU, Var('B'), thresh=0.0) * 0.1),
    ],
    seed=0,
)

# ---------------------------------------------------------------------------
# Config 2: interaction effect  (A AND B) -> Hazard
# Hazard only rises when both A and B simultaneously exceed their thresholds.
# ---------------------------------------------------------------------------
_interaction = DatasetConfig(
    filename='interaction',
    features=[
        Feature('A',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(1.0)),
        Feature('B',
                base_func=base_funcs.Oscillation(amplitude=2.0,
                                                 frequency=0.01),
                pre_noise=noise_funcs.GaussianNoise(0.3)),
        Feature('Hazard',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.NoNoise()),
    ],
    rules=[
        Rule(target='A', expr=0.9 * Var('A')),
        Rule(target='Hazard',
             expr=Apply(threshold, Var('A'), thresh=1.0) *
             Apply(threshold, Var('B'), thresh=0.5) * 0.5),
    ],
    seed=1,
)

# ---------------------------------------------------------------------------
# Config 3: lagged effect  A -> Hazard with a 10-frame lag
# Tests whether a model can recover a delayed causal link.
# ---------------------------------------------------------------------------
_lagged = DatasetConfig(
    filename='lagged',
    features=[
        Feature('A',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(1.0)),
        Feature('Hazard',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.NoNoise()),
    ],
    rules=[
        Rule(target='A', expr=0.8 * Var('A')),
        Rule(target='Hazard',
             expr=Apply(ReLU, Var('A', lag=10), thresh=0.5) * 0.2),
    ],
    seed=2,
)

# ---------------------------------------------------------------------------
# Config 4: censored version of simple_chain
# Same causal structure but with late entry and random feature masking.
# ---------------------------------------------------------------------------
_censored = DatasetConfig(
    filename='censored',
    features=[
        Feature('A',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(1.0)),
        Feature('B',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(0.5)),
        Feature('Hazard',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.NoNoise()),
    ],
    rules=[
        Rule(target='B', expr=0.8 * Var('B') + 0.5 * Var('A')),
        Rule(target='Hazard', expr=Apply(ReLU, Var('B'), thresh=0.0) * 0.1),
    ],
    late_entry_prob=0.4,
    late_entry_range=(50, 200),
    feature_mask_prob=0.05,
    seed=3,
)

# ---------------------------------------------------------------------------
# Config 5: confounded  C -> A and C -> Hazard
# A appears predictive of death but the true cause is the shared parent C.
# ---------------------------------------------------------------------------
_confounded = DatasetConfig(
    filename='confounded',
    features=[
        Feature('C',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(1.0)),
        Feature('A',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(0.2)),
        Feature('Hazard',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.NoNoise()),
    ],
    rules=[
        Rule(target='C', expr=0.9 * Var('C')),
        Rule(target='A', expr=0.5 * Var('C')),
        Rule(target='Hazard', expr=Apply(ReLU, Var('C'), thresh=1.0) * 0.15),
    ],
    seed=4,
)

ALL_CONFIGS: list[DatasetConfig] = [
    _simple_chain,
    _interaction,
    _lagged,
    _censored,
    _confounded,
]

if __name__ == '__main__':
    save_dir = Path('PhagoPred\\Datasets\\graph_synthetic')
    for cfg in ALL_CONFIGS:
        print(f"Generating '{cfg.filename}' ...")
        cfg.generate(save_dir)
    print('Done.')
