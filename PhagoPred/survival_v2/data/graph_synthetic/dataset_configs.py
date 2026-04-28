from dataclasses import dataclass
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
    train_num_cells: int = 1000
    val_num_cells: int = 200
    num_frames: int = 500
    late_entry_prob: float = 0.0
    late_entry_range: tuple = (0, 100)
    feature_mask_prob: float = 0.0
    smooth_hazard: bool = False
    smooth_sigma: int = 10
    seed: int = None
    target_death_fraction: float = 0.5

    def generate(self, save_dir: Path) -> None:
        """Generate and save train/val datasets to save_dir/<filename>_{train,val}.h5."""
        save_dir = Path(save_dir)
        generate_dataset(
            train_filename=save_dir / f'{self.filename}_train.h5',
            val_filename=save_dir / f'{self.filename}_val.h5',
            features=self.features,
            rules=self.rules,
            train_num_cells=self.train_num_cells,
            val_num_cells=self.val_num_cells,
            num_frames=self.num_frames,
            late_entry_prob=self.late_entry_prob,
            late_entry_range=self.late_entry_range,
            feature_mask_prob=self.feature_mask_prob,
            smooth_hazard=self.smooth_hazard,
            smooth_sigma=self.smooth_sigma,
            target_death_fraction=self.target_death_fraction,
            seed=self.seed,
        )


_linear = DatasetConfig(
    filename='linear',
    features=[
        Feature('A'),
        Feature('B'),
        Feature('C'),
        Feature('Hazard', pre_noise=noise_funcs.NoNoise()),
    ],
    rules=[
        Rule(target='A', expr=0.9 * Var('A')),
        Rule(target='B', expr=0.9 * Var('B')),
        Rule(target='C', expr=0.9 * Var('C')),
        Rule(
            target='Hazard',
            expr=Apply(ReLU,
                       0.99 * Var('Hazard') + 0.9 * Var('A') + 0.9 * Var('B') +
                       0.9 * Var('C'),
                       thresh=0.0),
        )
    ])
_l
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
    # _simple_chain,
    # _interaction,
    # _lagged,
    # _censored,
    # _confounded,
    _linear
]

if __name__ == '__main__':
    save_dir = Path('PhagoPred') / 'Datasets' / 'graph_synthetic'
    for cfg in ALL_CONFIGS:
        print(f"Generating '{cfg.filename}' ...")
        cfg.generate(save_dir)
    print('Done.')
