from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm


def _time_from_filename(path: Path) -> float | None:
    """Extract the (last) numeric value from a tiff filename, used as its time point.

    Filenames look like ``frame_  0.tiff`` / ``frame_ 12.tiff`` (number right-padded
    with spaces), so the trailing integer is the time index.
    """
    nums = re.findall(r'\d+', path.stem)
    return float(nums[-1]) if nums else None


def get_total_fluors(directory_path: Path,
                     moi_dict: dict,
                     fluor_dirname: str = 'Fluor',
                     stride: int = 1) -> dict:
    """For each subfolder with a fluorescence directory, sum the intensity of every
    tiff over time.

    Parameters:
        directory_path: dir of per-sample folders, each optionally containing a
            ``fluor_dirname`` subdir of time-named tiffs.
        moi_dict: {moi_value: [folder_name, ...]} mapping each folder to its MOI.
        fluor_dirname: name of the fluorescence subdirectory within each folder.
        stride: subsample every ``stride``-th tiff (sorted by time).

    Returns:
        {moi: {folder_name: {'times': [...], 'sums': [...]}}}
    """
    folder_to_moi = {
        folder: moi
        for moi, folders in moi_dict.items()
        for folder in folders
    }

    total_fluors: dict = {}
    for folder in sorted(p for p in directory_path.iterdir() if p.is_dir()):
        fluor_dir = folder / fluor_dirname
        if not fluor_dir.is_dir():
            continue

        moi = folder_to_moi.get(folder.name)
        if moi is None:
            print(f'Skipping {folder.name}: no MOI in moi_dict')
            continue

        tiff_files = sorted(fluor_dir.glob('*.tif*'),
                            key=lambda p: _time_from_filename(p) or 0)
        tiff_files = tiff_files[::stride]
        if not tiff_files:
            continue

        times, sums = [], []
        for tiff_file in tqdm(tiff_files,
                              desc=f'Folder {folder.name} (MOI {moi})'):
            im = tifffile.imread(str(tiff_file))
            times.append(_time_from_filename(tiff_file))
            sums.append(float(np.sum(im, dtype=np.float64)))

        total_fluors.setdefault(moi, {})[folder.name] = {
            'times': times,
            'sums': sums,
        }

    return total_fluors


def plot_fluors(total_fluors: dict, save_path: Path) -> None:
    """Plot total fluorescence intensity over time, one line per folder, coloured by MOI."""
    cmap = plt.get_cmap('Set1')
    fig, ax = plt.subplots()
    for i, moi in enumerate(sorted(total_fluors, key=float)):
        colour = cmap(i)
        labelled = False
        for series in total_fluors[moi].values():
            ax.plot(series['times'],
                    series['sums'],
                    color=colour,
                    label=None if labelled else f'MOI = {moi}')
            labelled = True

    ax.set_xlabel('Time')
    ax.set_ylabel('Total fluorescence intensity')
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)


if __name__ == '__main__':
    directory_path = Path('/home/ubuntu/thor_server/29_06')
    moi_dict = {
        0.1: ['A', 'B', 'C', 'D', 'E', 'F'],
        0.5: ['L', 'K', 'J', 'I', 'H', 'G'],
        1: ['M', 'N', 'O', 'P', 'Q', 'R'],
        2: ['X', 'W', 'V', 'U', 'T', 'S'],
    }

    total_fluors = get_total_fluors(directory_path, moi_dict)
    plot_fluors(total_fluors, Path('temp') / 'fluors_tiffs.png')
