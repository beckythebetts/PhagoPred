from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from . import base_funcs, noise_funcs
import h5py

from .generate_datasets import generate_dataset, calibrate_hazard, load_hazard_func, estimate_variances
from .predictability import compute_h_var
from .graph import Feature, CausalGraph
from .rules import (
    Rule,
    ReLU,
    Threshold,
    Sigmoid,
    Var,
    AutoCorrelationRule,
    MeanReversionRule,
    expand_to_noise,
)


@dataclass
class VarianceResult:
    total: np.ndarray
    unobserved: np.ndarray


@dataclass
class ScenarioVariances:
    cdf: VarianceResult | None = None
    hazard: dict[str, VarianceResult] = field(default_factory=dict)
    # hazard keys: 'default' for no bins, '0_50_100' etc. for binned


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
    val_num_cells: int = 1000
    cal_num_cells: int = 200
    num_frames: int = 500

    hazard_calibration_func: callable | None = None
    variances: ScenarioVariances | None = field(default=None, init=False)

    def __post_init__(self):
        self.graph = CausalGraph(self._generate_features(), self.rules,
                                 self.num_frames)

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

    def calibrate_hazard(self,
                         target_death_fraction: float = 0.5,
                         num_samples: int = 200) -> None:
        calib_starts = np.zeros(num_samples)
        calib_ends = np.full(num_samples, self.num_frames)
        self.hazard_calibration_func = calibrate_hazard(
            self.graph,
            self.num_frames,
            calib_starts,
            calib_ends,
            target_death_fraction,
        )

    def load(self, save_dir: Path) -> None:
        """Load calibration and saved variances from a previously generated dataset."""
        h5_path = Path(save_dir) / f'{self.filename}_train.h5'
        with h5py.File(h5_path, 'r') as f:
            self.hazard_calibration_func = load_hazard_func(
                f['Cells']['Phase']['HazardRates'].attrs)
            self.variances = ScenarioVariances()
            if 'Cells/Phase/Variances' not in f:
                return
            vg = f['Cells/Phase/Variances']
            if 'CDF' in vg:
                ds = vg['CDF']
                self.variances.cdf = VarianceResult(ds[0][:], ds[1][:])
            for ds_name in vg:
                if ds_name == 'CDF':
                    continue
                ds = vg[ds_name]
                key = 'default' if ds_name == 'hazard' else ds_name[
                    len('hazard_'):]
                self.variances.hazard[key] = VarianceResult(ds[0][:], ds[1][:])

    def generate(self, save_dir: Path) -> None:
        """Generate and save train/val/cal datasets to save_dir/<filename>_{train,val,cal}.h5."""
        save_dir = Path(save_dir)
        generate_dataset(
            train_filename=save_dir / f'{self.filename}_train.h5',
            val_filename=save_dir / f'{self.filename}_val.h5',
            cal_filename=save_dir /
            f'{self.filename}_cal.h5' if self.cal_num_cells > 0 else None,
            graph=self.graph,
            train_num_cells=self.train_num_cells,
            val_num_cells=self.val_num_cells,
            cal_num_cells=self.cal_num_cells,
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
            hazard_calibration_func=self.hazard_calibration_func,
        )

    # def get_variances(self,
    #                   save_dir: Path,
    #                   max_horizon: int,
    #                   base_sample_size: int,
    #                   branch_sample_size: int,
    #                   hazard_bins: np.ndarray | None = None) -> None:
    #     save_variances(
    #         [
    #             save_dir / f'{self.filename}_train.h5',
    #             save_dir / f'{self.filename}_val.h5'
    #         ],
    #         self.graph,
    #         self.hazard_calibration_func,
    #         max_horizon,
    #         self.num_frames,
    #         base_sample_size,
    #         branch_sample_size,
    #         hazard_bins=hazard_bins,
    #     )

    def save_variances(self,
                       save_dir: Path,
                       max_horizon: int,
                       base_sample_size: int = 200,
                       branch_sample_size: int = 1000,
                       warm_up_steps: int = 0,
                       hazard_bins: np.ndarray | None = None,
                       force: bool = False,
                       landmark_distribution: np.ndarray | None = None):
        variances = estimate_variances(
            self.graph,
            self.hazard_calibration_func,
            max_horizon,
            self.num_frames - max_horizon,
            base_sample_size,
            branch_sample_size,
            warm_up_steps,
            hazard_bins,
            landmark_distribution,
        )
        hazard_ds = 'hazard' if hazard_bins is None else 'hazard_' + '_'.join(
            str(int(n)) for n in hazard_bins)
        name_map = {'cdf': 'CDF', 'hazard': hazard_ds}

        for split in ('train', 'val'):
            file = save_dir / f'{self.filename}_{split}.h5'
            with h5py.File(file, 'r+') as f:
                variances_group = f.require_group('Cells/Phase/Variances')
                for var_key, ds_name in name_map.items():
                    if ds_name in variances_group:
                        if not force and var_key == 'cdf':
                            continue
                        del variances_group[ds_name]
                    ds = variances_group.create_dataset(
                        ds_name,
                        data=np.stack([
                            variances[var_key]['Total'],
                            variances[var_key]['Unobserved'],
                        ]),
                        dtype=float,
                    )
                    ds.attrs['rows'] = ['Total', 'Unobserved']

    # def _save_variances(file_names: list[Path],
    #                     graph: CausalGraph,
    #                     hazard_calibration_func: callable,
    #                     max_horizon: int,
    #                     max_base_time_steps: int,
    #                     base_sample_size: int = 200,
    #                     branch_sample_size: int = 1000,
    #                     hazard_bins: np.ndarray | None = None):
    #     variances = estimate_variances(
    #         graph,
    #         hazard_calibration_func,
    #         max_horizon,
    #         max_base_time_steps,
    #         base_sample_size,
    #         branch_sample_size,
    #         hazard_bins=hazard_bins,
    #     )
    #     dataset_map = {'hazard': 'HazardRates', 'cdf': 'CIFs'}
    #     for file_name in file_names:
    #         with h5py.File(file_name, 'r+') as f:
    #             for name, dataset_name in dataset_map.items():
    #                 attrs = f['Cells']['Phase'][dataset_name].attrs
    #                 attrs['Total Variance'] = variances[name]['Total']
    #                 attrs['Unobserved Variance'] = variances[name][
    #                     'Unobserved']
    #             if hazard_bins is not None:
    #                 f['Cells']['Phase']['HazardRates'].attrs[
    #                     'Hazard Bins'] = hazard_bins


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

_default_auto_coefficient = 1 - 1e-5


# === RULES ===
def auto_correlate(feature_coeffs: dict[str, float] = None) -> list[Rule]:
    if feature_coeffs is None:
        feature_coeffs = {
            feat_name: _default_auto_coefficient
            for feat_name in ('A', 'B', 'C', 'D')
        }
        # feature_coeffs['Hazard'] = 0.8
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
         expr=ReLU(Var('A', 25) + Var('B', 50) + Var('C', 100), thresh=3.0))
] + auto_correlate()

_chain = [
    Rule(target='B', expr=0.8 * Var('A', 50)),
    Rule(target='C', expr=0.8 * Var('B', 50)),
    Rule(target='Hazard', expr=ReLU(Var('C', 50), thresh=0.0))
] + auto_correlate({
    'A': _default_auto_coefficient,
    'B': 0.2,
    'C': 0.2,
    'D': _default_auto_coefficient,
    # 'Hazard': 0.8
})

_multiplicative = [
    Rule(target='Hazard', expr=ReLU(Var('A', 50) * Var('B', 100), thresh=0.0))
] + auto_correlate()

_ratio = [Rule(target='Hazard', expr=Sigmoid(Var('A', 50) / Var('B', 100)))
          ] + auto_correlate()

# _resetting_accumulation = [
#     Rule(target='Hazard', expr=ReLU(Var('A') - )
# ] + auto_correlate({
#     'A': 0.999,
#     'B': 0.999,
#     'C': 0.999,
#     'D': 0.999,
# }) + mean_reversion()

ALL_CFGS: list[ScenarioCfg] = [
    ScenarioCfg('base_linear',
                rules=_linear,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness),
    # Learning-curve ablation: linear rules at varying training set sizes
    ScenarioCfg('base_linear_n100',
                rules=_linear,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness,
                train_num_cells=100),
    ScenarioCfg('base_linear_n500',
                rules=_linear,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness,
                train_num_cells=500),
    ScenarioCfg('base_linear_n1000',
                rules=_linear,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness,
                train_num_cells=1000),
    ScenarioCfg('base_linear_n5000',
                rules=_linear,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness,
                train_num_cells=5000),
    ScenarioCfg('base_chain',
                rules=_chain,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness),
    ScenarioCfg('base_multiplicative',
                rules=_multiplicative,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness),
    ScenarioCfg('base_ratio',
                rules=_ratio,
                noise_cfg=_low_noise,
                missingness_cfg=_low_missingness),
    # ScenarioCfg('base_restting_accumulation',
    #             rules=_resetting_accumulation,
    #             noise_cfg=_low_noise,
    #             missingness_cfg=_low_missingness),
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


def plot_variances(
    save_path: Path = None,
    quantity: Literal['hazard', 'cdf'] = 'hazard',
    hazard_key: str = 'default',
) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = 'serif'
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap('Set1')
    for i, cfg in enumerate(ALL_CFGS):
        if cfg.variances is None:
            continue
        result = cfg.variances.cdf if quantity == 'cdf' else cfg.variances.hazard.get(
            hazard_key)
        if result is None:
            continue
        label = cfg.filename.replace('_', ' ').capitalize()
        colour = cmap(i)
        horizons = np.arange(len(result.total))
        ax.plot(horizons,
                result.total,
                marker='.',
                markersize=3,
                linestyle='dotted',
                label=f'{label} Total',
                color=colour)
        ax.plot(horizons,
                result.unobserved,
                marker='.',
                markersize=3,
                label=f'{label} Unobserved',
                color=colour)

    ax.set_xlabel('Horizon (H)')
    ax.set_ylabel(f'Variance ({quantity})')
    ax.set_title(
        f'Variance contribution from unobserved timesteps by scenario ({quantity})'
    )
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    save_dir = Path('PhagoPred') / 'Datasets' / 'graph_synthetic'

    for cfg in ALL_CFGS:
        print(f"Generating '{cfg.filename}' ...")
        cfg.calibrate_hazard()
        cfg.generate(save_dir)
        # c
        cfg.load(save_dir)
        max_horizon = 100
        # cfg.save_variances(save_dir,
        #                    max_horizon,
        #                    base_sample_size=500,
        #                    warm_up_steps=50,
        #                    force=True)
        cfg.load(save_dir)
    print('Done.')

    # plot_variances(save_path=Path('temp') / 'variances_hazards.png',
    #                quantity='hazard')
    # plot_variances(save_path=Path('temp') / 'variances_cdf.png',
    #                quantity='cdf')
    # plot_variances(save_dir,
    #                save_path=Path('temp') / 'variances_pmf.png',
    #                quantity='pmf')
    # print(expand_to_noise(_linear, 'Hazard', 10, 10))
