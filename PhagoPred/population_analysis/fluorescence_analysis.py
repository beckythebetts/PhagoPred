from __future__ import annotations
from pathlib import Path
import json

import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars (e.g. large uint64) and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_json(data, json_path: Path) -> None:
    """Write ``data`` to ``json_path``, coercing numpy scalars/arrays to native types."""
    with Path(json_path).open('w') as f:
        json.dump(data, f, cls=_NumpyEncoder)


def read_json(json_path: Path):
    """Read JSON from ``json_path``."""
    with Path(json_path).open('r') as f:
        return json.load(f)


def get_total_fluors(dir_path: Path,
                     stride: int = 3,
                     json_path: Path | None = None):
    total_fluors = {}
    for h5_path in dir_path.glob('*.h5'):
        print(f'CHecking fluorescnec in file {h5_path}')
        with h5py.File(h5_path, 'r') as f:
            if not f.attrs['phase_only']:
                moi = int(f.attrs['moi'])
                fluors = []
                for frame_idx in tqdm(
                        np.arange(0, f['Images']['Epi'].shape[0], stride)):
                    fluors.append(np.sum(f['Images']['Epi'][frame_idx][:]))
                if moi not in list(total_fluors.keys()):
                    total_fluors[moi] = []
                total_fluors[moi].append(fluors)
    if json_path is not None:
        write_json(total_fluors, json_path)
    return total_fluors


def plot_fluors(total_fluors: dict, save_path: Path) -> None:
    cmap = plt.get_cmap('Set1')
    fig, axs = plt.subplots()
    for i, (moi, fluors) in enumerate(total_fluors.items()):
        colour = cmap(i)
        for fluors_i in fluors:
            fluors_i = np.array(fluors_i)
            time_steps = np.arange(len(fluors_i))

            mask = fluors_i > 0
            fluors_i = fluors_i[mask]
            time_steps = time_steps[mask]
            axs.plot(time_steps, fluors_i, color=colour, label=f'MOI = {moi}')
    axs.legend()
    plt.savefig(save_path)


if __name__ == '__main__':
    dir_path = Path('/home/ubuntu/thor_server/MacrophageData/12_06')
    # total_fluors = get_total_fluors(dir_path,
    #                                 json_path=Path('temp') / 'fluors.json')

    total_fluors = read_json(Path('temp') / 'fluors.json')
    plot_fluors(total_fluors, Path('temp') / 'fluors.png')
