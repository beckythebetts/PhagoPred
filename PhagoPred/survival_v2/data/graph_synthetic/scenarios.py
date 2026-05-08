from dataclasses import dataclass, field
from pathlib import Path

from . import base_funcs, noise_funcs
from .generate_datasets import generate_dataset
from .graph import Feature
from .rules import (
    Apply,
    Rule,
    ReLU,
    Var,
    threshold,
    AutoCorrelationRule,
    MeanReversionRule,
    expand_to_noise,
)


@dataclass
class NoiseCfg:
    """Paramaters for adding noise to datasets"""
    pre_noise: noise_funcs.Noise = noise_funcs.GaussianNoise(sigma=1.0)
    post_noise: noise_funcs.Noise = noise_funcs.GaussianNoise(sigma=1.0)


@dataclass
class MissingnessCfg:
    """Paramaters for missing time steps"""
    prob_missing: float = 0.0
    late_entry_prob: float = 0.5
    max_late_entry_frac: float = 0.49
    early_exit_prob: float = 0.5
    max_early_exit_frac: float = 0.49


@dataclass
class ScenarioCfg:
    """Full paramaters fro creating a dataset"""
    filename: str

    observed_features: list[str] = field(
        default_factory=lambda: ['A', 'B', 'C', 'D', 'Hazard'])
    hidden_features: list[str] = field(default_factory=lambda: [])

    rules: list[Rule] = field(default_factory=lambda: [])

    noise_cfg: NoiseCfg = NoiseCfg()
    missingness_cfg: MissingnessCfg = MissingnessCfg()

    target_death_fraction: float = 0.5
    train_num_cells: int = 1000
    val_num_cells: int = 200
    num_frames: int = 500

    def _generate_features(self) -> list[Feature]:
        assert not bool(
            set(self.observed_features) & set(self.hidden_features))
        features = []
        for feature_name in self.observed_features:
            features.append(
                Feature(
                    feature_name,
                    pre_noise=self.noise_cfg.pre_noise,
                    post_noise=self.noise_cfg.post_noise,
                    hidden=False,
                ))
        for feature_name in self.hidden_features:
            features.append(
                Feature(
                    feature_name,
                    pre_noise=self.noise_cfg.pre_noise,
                    post_noise=self.noise_cfg.post_noise,
                    hidden=True,
                ))
        return features

    def generate(self, save_dir: Path) -> None:
        """Generate and save train/val datasets to save_dir/<filename>_{train,val}.h5."""
        save_dir = Path(save_dir)
        generate_dataset(
            train_filename=save_dir / f'{self.filename}_train.h5',
            val_filename=save_dir / f'{self.filename}_val.h5',
            features=self._generate_features(),
            rules=self.rules,
            train_num_cells=self.train_num_cells,
            val_num_cells=self.val_num_cells,
            num_frames=self.num_frames,
            late_entry_prob=self.missingness_cfg.late_entry_prob,
            late_entry_range=(0, self.missingness_cfg.max_late_entry_frac *
                              self.num_frames),
            early_exit_prob=self.missingness_cfg.early_exit_prob,
            early_exit_range=(
                self.num_frames -
                self.missingness_cfg.max_early_exit_frac * self.num_frames,
                self.num_frames),
            feature_mask_prob=self.missingness_cfg.prob_missing,
            target_death_fraction=self.target_death_fraction,
        )


# === NOISE CFGS ===
# pre_noise drives the causal dynamics (same across noise levels for fair comparison).
# post_noise is observation/measurement noise — varied to control SNR.
_pre_noise_sigma = 0.5
_no_noise = NoiseCfg(pre_noise=noise_funcs.GaussianNoise(_pre_noise_sigma),
                     post_noise=noise_funcs.NoNoise())
_low_noise = NoiseCfg(pre_noise=noise_funcs.GaussianNoise(_pre_noise_sigma),
                      post_noise=noise_funcs.GaussianNoise(0.1))
_high_noise = NoiseCfg(pre_noise=noise_funcs.GaussianNoise(_pre_noise_sigma),
                       post_noise=noise_funcs.GaussianNoise(1.0))

# === MISSINGNESS CFGS ===
_none_missing = MissingnessCfg(prob_missing=0.0,
                               late_entry_prob=0.0,
                               early_exit_prob=0.0)
_low_missingness = MissingnessCfg(prob_missing=0.0,
                                  late_entry_prob=0.2,
                                  early_exit_prob=0.2)


# === RULES ===
def auto_correlate(feature_coeffs: dict[str, float] = None) -> list[Rule]:
    if feature_coeffs is None:
        feature_coeffs = {
            feat_name: 1.0
            for feat_name in ('A', 'B', 'C', 'D', 'Hazard')
        }
    rules = []
    for key, val in feature_coeffs.items():
        rules.append(AutoCorrelationRule(key, val))

    return rules


def mean_reversion(feature_params: dict[str, tuple] = None) -> list[Rule]:
    if feature_params is None:
        feature_params = {
            feat_name: (0.0, 0.01)
            for feat_name in ('A', 'B', 'C', 'D')
        }
    rules = []
    for key, val in feature_params.items():
        rules.append(MeanReversionRule(key, *val))

    return rules


_linear = [
    Rule(target='Hazard',
         expr=Apply(ReLU,
                    0.9 * Var('A') + 0.9 * Var('B') + 0.9 * Var('C'),
                    thresh=0.0))
] + auto_correlate() + mean_reversion()

_chain = [
    Rule(target='B', expr=0.8 * Var('A')),
    Rule(target='C', expr=0.8 * Var('B')),
    Rule(target='Hazard', expr=Apply(ReLU, 0.9 * Var('C'), thresh=0.0))
] + auto_correlate({
    'A': 1.0,
    'B': 0.2,
    'C': 0.2,
    'D': 1.0,
    'Hazard': 1.0
}) + mean_reversion()

_multiplicative = [
    Rule(target='Hazard', expr=Apply(ReLU, Var('A') * Var('B'), thresh=0.0))
] + auto_correlate() + mean_reversion()

_ratio = [
    Rule(target='Hazard', expr=Apply(ReLU, Var('A') / Var('B'), thresh=0.5))
] + auto_correlate() + mean_reversion()

_resetting_accumulation = [
    Rule(target='Hazard',
         expr=Apply(threshold, Var('A'), thresh=0.0) *
         Apply(ReLU, Var('Hazard'), thresh=0.0))
] + auto_correlate({
    'A': 1.0,
    'B': 1.0,
    'C': 1.0,
    'D': 1.0,
}) + mean_reversion()

ALL_CFGS: list[ScenarioCfg] = [
    # ScenarioCfg('base_linear',
    #             rules=_linear,
    #             noise_cfg=_low_noise,
    #             missingness_cfg=_low_missingness),
    # ScenarioCfg('base_chain',
    #             rules=_chain,
    #             noise_cfg=_low_noise,
    #             missingness_cfg=_low_missingness),
    # ScenarioCfg('base_multiplicative',
    #             rules=_multiplicative,
    #             noise_cfg=_low_noise,
    #             missingness_cfg=_low_missingness),
    # ScenarioCfg('base_ratio',
    #             rules=_ratio,
    #             noise_cfg=_low_noise,
    #             missingness_cfg=_low_missingness),
    ScenarioCfg('base_restting_accumulation',
                rules=_resetting_accumulation,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness),
    # Noise-level comparison (linear rules)
    # ScenarioCfg('linear_no_noise',
    #             rules=_linear,
    #             noise_cfg=_no_noise,
    #             missingness_cfg=_low_missingness),
    # ScenarioCfg('linear_low_noise',
    #             rules=_linear,
    #             noise_cfg=_low_noise,
    #             missingness_cfg=_low_missingness),
    # ScenarioCfg('linear_high_noise',
    #             rules=_linear,
    #             noise_cfg=_high_noise,
    #             missingness_cfg=_low_missingness),
]

if __name__ == '__main__':
    # save_dir = Path('PhagoPred') / 'Datasets' / 'graph_synthetic'

    # for cfg in ALL_CFGS:
    #     print(f"Generating '{cfg.filename}' ...")
    #     cfg.generate(save_dir)
    # print('Done.')

    print(expand_to_noise(_chain, 'Hazard', 10, 10))
    print(expand_to_noise(_multiplicative, 'Hazard', 10, 10))
    print(expand_to_noise(_linear, 'Hazard', 10, 10))
