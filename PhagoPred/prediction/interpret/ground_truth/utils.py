from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import h5py

from PhagoPred.utils.logger import get_logger
from PhagoPred.prediction.data.graph_synthetic.scenarios import ALL_CFGS, ScenarioCfg
from .generate_samples import generate_sample_with_importances

log = get_logger()


def infer_scenario(experiment_dir: Path) -> ScenarioCfg | None:
    """Match the experiment's dataset paths to a known ScenarioCfg by filename stem."""
    with open(experiment_dir / 'config.json') as f:
        cfg_raw = json.load(f)
    dataset_cfg = cfg_raw.get('dataset', {})
    all_paths = dataset_cfg.get('train_paths', []) + dataset_cfg.get(
        'val_paths', [])
    for path in all_paths:
        stem = Path(path).stem  # '<scenario.filename>_<split>'
        for suffix in ('_train', '_val', '_cal'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break

        for scenario_cfg in ALL_CFGS:
            if scenario_cfg.filename == stem:
                return scenario_cfg
    return None


def _params_match(f: h5py.File, output_type: str, horizon: int,
                  hazard_bins: None | np.ndarray, num_permutations: int,
                  min_horizon_cif: float, num_background_samples: int,
                  seg_attr: int) -> bool:
    """Whether a cached file was generated with the same (everything but
    ``num_samples``) parameters — a mismatch on any of these makes its
    samples wrong for this request, unlike ``num_samples`` (see
    ``get_samples``)."""
    return (f.attrs['Output type'] == output_type
            and f.attrs['Horizon'] == horizon and np.array_equal(
                f.attrs['Hazard Bins'],
                np.array([]) if hazard_bins is None else hazard_bins)
            and f.attrs['Num Permutations'] == num_permutations
            and f.attrs['Min Horizon CIF'] == min_horizon_cif
            and f.attrs['Num Background Samples'] == num_background_samples
            and f.attrs.get('Num Segments', -1) == seg_attr)


def get_samples(
    daatset_dir: Path,
    scenario: ScenarioCfg,
    num_samples: int,
    horizon: int,
    hazard_bins: None | np.ndarray,
    num_permutations: int,
    min_horizon_cif: float,
    output_type: str,
    num_background_samples: int,
    num_segments: int | None = None,
) -> Path:
    seg_attr = -1 if num_segments is None else int(num_segments)
    file_name = None
    # Smallest matching cache with >= num_samples: cheap to subset (each
    # ground-truth sample is a full permutation-Shapley run over the causal
    # graph, so generating fewer from scratch is not "cheap" just because
    # it's fewer — reusing an existing superset avoids that entirely).
    superset: tuple[int, Path] | None = None
    samples_dir = daatset_dir / 'shap_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    try:
        for file in samples_dir.iterdir():
            if scenario.filename not in file.name:
                continue
            with h5py.File(file, 'r') as f:
                if not _params_match(f, output_type, horizon, hazard_bins,
                                     num_permutations, min_horizon_cif,
                                     num_background_samples, seg_attr):
                    continue
                n = int(f.attrs['Num Samples'])
                if n == num_samples:
                    file_name = file
                    break
                if n > num_samples and (superset is None or n < superset[0]):
                    superset = (n, file)
    except KeyError:
        pass

    if file_name is None:
        idx = 0
        file_name = samples_dir / f'{scenario.filename}_{idx}.h5'
        while file_name.is_file():
            idx += 1
            file_name = samples_dir / f'{scenario.filename}_{idx}.h5'
        if superset is not None:
            _subset_samples(superset[1], file_name, num_samples)
        else:
            _generate_samples(
                file_name,
                scenario,
                num_samples,
                horizon,
                hazard_bins,
                num_permutations,
                min_horizon_cif,
                output_type,
                num_background_samples,
                num_segments,
            )
    return file_name


def _subset_samples(src_path: Path, dst_path: Path, num_samples: int) -> None:
    """Copy the first ``num_samples`` sample groups (and attrs, with ``Num
    Samples`` corrected) out of an existing larger matching cache, instead of
    rerunning ``_generate_samples`` for a smaller count from scratch."""
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        dst.attrs['Num Samples'] = num_samples
        for i in range(num_samples):
            src.copy(str(i), dst)


def _generate_samples(
    h5_path: Path,
    scenario: ScenarioCfg,
    num_samples: int,
    horizon: int,
    hazard_bins: None | np.ndarray,
    num_permutations: int,
    min_horizon_cif: float,
    output_type: str,
    num_background_samples: int,
    num_segments: int | None = None,
) -> None:

    with h5py.File(h5_path, 'w') as f:
        f.attrs['Scenario'] = scenario.filename
        f.attrs['Output type'] = output_type
        f.attrs['Horizon'] = horizon
        f.attrs['Hazard Bins'] = np.array(
            []) if hazard_bins is None else hazard_bins
        f.attrs['Num Permutations'] = num_permutations
        f.attrs['Min Horizon CIF'] = min_horizon_cif
        f.attrs['Num Samples'] = num_samples
        f.attrs['Num Background Samples'] = num_background_samples
        f.attrs['Num Segments'] = -1 if num_segments is None else int(
            num_segments)

    attempt_budget = num_samples * 200
    attempts = 0
    samples = 0
    while samples < num_samples and attempts < attempt_budget:
        sample = generate_sample_with_importances(
            scenario.graph,
            scenario.hazard_calibration_func,
            horizon,
            scenario.num_frames,
            100,
            num_permutations,
            hazard_bins,
            output_type,
            min_horizon_cif,
            num_background_samples,
            num_segments=num_segments,
        )
        if sample is not None:
            sample.write_h5(h5_path, samples)
            samples += 1
            log.info(
                f'Generated sample {samples} / {num_samples}, {attempts} total attempts'
            )

        attempts += 1
    return samples
