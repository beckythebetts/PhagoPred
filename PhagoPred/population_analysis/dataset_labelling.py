from __future__ import annotations
from pathlib import Path
from tqdm import tqdm

import h5py


def label_datasets(directory_path: Path,
                   phase_only_names: list[str] | None = None,
                   moi_dict: dict | None = None):

    for h5_path in tqdm(directory_path.glob('*.h5'),
                        'Adding dataset metedata'):
        with h5py.File(h5_path, 'r+') as f:
            h5_path = Path(h5_path)
            phase_only = True if h5_path.stem in phase_only_names else False
            f.attrs['phase_only'] = phase_only

            moi = None
            for moi_val, moi_files in moi_dict.items():
                if h5_path.stem in moi_files:
                    moi = moi_val
                    break
            f.attrs['moi'] = moi


if __name__ == '__main__':
    directory_path = Path('/home/ubuntu/thor_server/MacrophageData/12_06')
    phase_only_names = [
        'A', 'B', 'C', 'L', 'J', 'K', 'M', 'N', 'O', 'W', 'X', 'Y'
    ]
    moi_dict = {
        10: ['A', 'B', 'C', 'D', 'E', 'F'],
        5: ['L', 'K', 'J', 'I', 'H', 'G'],
        2: ['M', 'N', 'O', 'P', 'Q', 'R'],
        0: ['W', 'X', 'Y', 'U', 'T', 'S']
    }

    label_datasets(directory_path, phase_only_names, moi_dict)
