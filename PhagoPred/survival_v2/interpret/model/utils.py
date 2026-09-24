from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import torch

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.utils.io import load_dataset
from PhagoPred.survival_v2.data.dataset import collate_fn
from PhagoPred.survival_v2.interpret import SampleWithSHAP

log = get_logger()


def align_sample_features(sample: SampleWithSHAP,
                          model_feat_names: list[str]) -> np.ndarray:
    """The (model_feat_names, landmark_frame)-aligned view of a sample.

    Two adjustments, needed for a ground-truth ``SampleWithSHAP`` (real-data
    ones already satisfy both by construction, so this is a no-op there):

    - **Column subset/reorder**: ``sample.feature_names`` may include nodes the
      model never sees as input (e.g. ``Hazard`` — the latent quantity driving
      the outcome, not an observed feature) and needn't be in the model's own
      column order. Select and reorder to ``model_feat_names`` exactly.
    - **Crop to the observed past**: a ground-truth sample's ``feature_vals``
      spans ``[0, lf+horizon)`` — it knows the future, that's the point of it
      — while the model must only ever see ``[0, landmark_frame)``. Slicing to
      ``[:landmark_frame]`` is a safe no-op for a real-data sample, whose
      stored length is already that short or shorter.
    """
    idx = [sample.feature_names.index(name) for name in model_feat_names]
    return sample.feature_vals[idx, :sample.landmark_frame]


def load_background_pool(
        h5_path: Path,
        model_feat_names: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Read back every sample ``get_samples``/``_generate_samples``/the
    ground-truth generator wrote and stack them into a ``(N, T, K)``
    background pool for ``var_precision.fit_var``.

    ``model_feat_names`` aligns each sample first (see
    ``align_sample_features``) — required for a ground-truth results file,
    since its raw ``feature_vals``/``feature_names`` include ``Hazard`` and
    the future; harmless (identity) for a real-data one already in that shape.
    Samples have different observed lengths (different landmark frames), so
    shorter trajectories are padded with NaN — ``fit_var`` already drops any
    lag window touching a NaN, which is exactly "this cell doesn't cover this
    frame" rather than a real observation to fit against.
    """
    with h5py.File(h5_path, 'r') as f:
        indices = sorted(int(k) for k in f.keys() if k.isdigit())
        if not indices:
            raise ValueError(f'No samples found in {h5_path}')
        samples = [SampleWithSHAP.read_h5(f, i) for i in indices]

    feature_names = model_feat_names or samples[0].feature_names
    aligned = ([align_sample_features(s, feature_names) for s in samples]
              if model_feat_names else [s.feature_vals for s in samples])

    max_t = max(a.shape[1] for a in aligned)
    K = len(feature_names)
    pool = np.full((len(samples), max_t, K), np.nan, dtype=np.float32)
    for n, a in enumerate(aligned):
        t = a.shape[1]
        pool[n, :t] = a.T  # (K, t) -> (t, K)
    return pool, feature_names


def background_rows_for_length(pool: np.ndarray,
                               T: int,
                               num_rows: int = 8,
                               seed: int | None = None) -> np.ndarray:
    """Rows of ``pool`` (N, T_max, K) fully observed over the first ``T``
    frames (no NaN there), cropped to ``(num_rows, T, K)`` — an interventional
    background matching one particular explained sample's window length,
    since different real samples generally have different lengths.
    """
    valid = ~np.isnan(pool[:, :T]).any(axis=(1, 2))
    candidates = np.where(valid)[0]
    if len(candidates) == 0:
        raise ValueError(
            f'No background rows fully observed over the first {T} frames.')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates,
                        size=min(num_rows, len(candidates)),
                        replace=False)
    return pool[chosen, :T]


def get_samples(
    experiment_dir: Path,
    num_samples: int,
) -> Path:
    """Real (non-synthetic) counterpart of ``ground_truth.utils.get_samples``.

    Caches by ``num_samples`` under ``experiment_dir/shap_samples`` the same
    way the ground-truth version caches by scenario/config, so a rerun with an
    identical ``num_samples`` reuses the existing file instead of resampling.
    """
    samples_dir = Path(experiment_dir) / 'shap_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    file_name = None
    try:
        for file in samples_dir.iterdir():
            with h5py.File(file, 'r') as f:
                if f.attrs.get('Num Samples') == num_samples:
                    file_name = file
                    break
    except KeyError:
        pass

    if file_name is None:
        idx = 0
        file_name = samples_dir / f'model_samples_{idx}.h5'
        while file_name.is_file():
            idx += 1
            file_name = samples_dir / f'model_samples_{idx}.h5'
        _generate_samples(file_name, experiment_dir, num_samples)
    return file_name


def _generate_samples(
    h5_path: Path,
    experiment_dir: Path,
    num_samples: int,
) -> int:
    """Write ``num_samples`` real validation-set samples, with no SHAP values yet.

    Populates only ``feature_vals``/``landmark_frame``/``death_frame`` (no
    ``noise_vals`` — the generating mechanism is unknown for real data, unlike
    the synthetic ground truth). This file serves double duty: it's the record
    of which samples get explained, and it's the background pool a dependence-
    aware (observational) explainer would be fit from — both need to exist
    before any per-sample SHAP computation runs, so this is written first, with
    ``shap_vals`` filled in by a later pass via read-modify-write.
    """
    dataset = load_dataset(experiment_dir, 'val')

    with h5py.File(h5_path, 'w') as f:
        f.attrs['Num Samples'] = num_samples

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(num_samples, len(dataset)),
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, device='cpu'),
    )
    batch = next(iter(dataloader))

    n = len(batch['landmark_frame'])
    for i in range(n):
        landmark_frame = int(batch['landmark_frame'][i])
        start_frame = int(batch['start_frame'][i]) if 'start_frame' in batch \
            and batch['start_frame'] is not None else 0
        sample_len = landmark_frame - start_frame

        # features: (batch, max_seq_len, num_features), cropped to
        # [start_frame:landmark_frame] then zero-padded at the end by
        # collate_fn — valid data is exactly the first ``sample_len`` frames.
        feature_vals = batch['features'][i, :sample_len].cpu().numpy().T

        death_frame = None
        if ('event_indicator' in batch and batch['event_indicator'] is not None
                and 'time_to_event' in batch
                and batch['time_to_event'] is not None):
            if int(batch['event_indicator'][i]) == 1:
                death_frame = landmark_frame + float(
                    batch['time_to_event'][i])

        sample = SampleWithSHAP(
            feature_vals=feature_vals,
            feature_names=dataset.feature_names,
            landmark_frame=landmark_frame,
            death_frame=death_frame,
            noise_vals=None,
            shap_vals={},
        )
        sample.write_h5(h5_path, sample_idx=i)
        log.info(f'Wrote sample {i + 1} / {n}')

    return n
