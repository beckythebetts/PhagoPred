from __future__ import annotations
from pathlib import Path

import torch
import numpy as np

from PhagoPred.SETTINGS import BAD_TRACK_MODEL, DEATH_MODEL
from PhagoPred.utils.h5_file import H5File
from PhagoPred.death_classifier.dataset import CellWindowDataset
from PhagoPred.death_classifier.model import load_model
from PhagoPred.death_classifier.extract_crops import cache_name, extract_file
from PhagoPred.death_classifier.predict import predict_track


def predict_bad_tracks(h5_file: Path) -> None:
    run(h5_file, BAD_TRACK_MODEL)


def predict_death(h5_file: Path) -> None:
    run(h5_file, DEATH_MODEL)


def run(
    h5_file: Path,
    model_path: Path,
    cell_batch_size: int = 100,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_file = H5File(h5_file)

    model, config = load_model(model_path, device=device)

    num_cells = h5_file.num_cells
    results = np.full(num_cells, np.nan)
    for start in range(0, num_cells, cell_batch_size):
        cells = list(range(start, min(start + cell_batch_size, num_cells)))
        crops = extract_file(h5_file.path, cells, **config.crop_kwargs())
        dataset = CellWindowDataset(crops, **config.dataset_kwargs())
        for cell in cells:
            result = predict_track(model, config, dataset,
                                   cache_name(h5_file.path.name, cell), device)
            value = result['p_bad'] if config.loss == 'none' else result[
                'predicted']
            results[cell] = np.nan if value is None else value
    h5_file.write_feature(
        'Bad Track Probability' if config.loss == 'none' else 'Death Frame',
        results)


import shutil
import h5py

LOCAL = Path('temp/pipeline_scratch')
FEATURE_NAMES = {
    'Bad Track Probability': BAD_TRACK_MODEL,
    'Death Frame': DEATH_MODEL
}


def process(file: Path) -> None:
    """Predict on a local copy, then write just the new datasets back."""
    with h5py.File(file, 'r') as f:
        todo = [n for n in FEATURE_NAMES if n not in f['Cells/Phase']]
    if not todo:
        return
    LOCAL.mkdir(parents=True, exist_ok=True)
    local = LOCAL / file.name
    shutil.copy(file, local)
    try:
        for name in todo:
            print(f'{file}: {name}', flush=True)
            run(local, FEATURE_NAMES[name])
        with h5py.File(local, 'r') as src, h5py.File(file, 'r+') as dst:
            for name in todo:
                if name in dst['Cells/Phase']:
                    del dst['Cells/Phase'][name]
                src.copy(src['Cells/Phase'][name],
                         dst['Cells/Phase'],
                         name=name)
    finally:
        local.unlink(missing_ok=True)


if __name__ == '__main__':
    datasets_folder = Path('~/thor_server').expanduser() / 'MacrophageData'
    for folder in (datasets_folder / '02_10', datasets_folder / '16_10'):
        for file in sorted(folder.glob('*.h5')):
            try:
                process(file)
            except Exception as e:
                print(f'{file} failed: {e!r}', flush=True)
