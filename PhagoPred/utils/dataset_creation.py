from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Iterator
import math
import shutil
import os

import tifffile
from tqdm import tqdm
import numpy as np
import cv2
import imagej_rolling_ball

from .h5_file import H5File
from .logger import get_logger
from .hot_pixel_removal import HOT_PIXEL_PATH, RESIDUAL_THRESHOLD, correct_hot_pixels, get_mask

log = get_logger()


def _natural_sort_key(s) -> list:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', str(s))
    ]


def find_ome_positions(tiff_dir: Path) -> dict[str, list[Path]]:
    well_site = re.compile(r'MMStack_(?P<well>[A-Za-z]+\d+)-Site_(?P<site>\d+)'
                           r'(?:_\d+)?\.ome\.tif$')
    pos_pattern = re.compile(r'_Pos(?P<pos>\d+)(?:[._].*)?\.ome\.tif$')
    groups = {}
    for path in tiff_dir.glob('*.ome.tif'):
        if match := well_site.search(path.name):
            key = (match['well'].upper(), int(match['site']))
        elif match := pos_pattern.search(path.name):
            key = (None, int(match['pos']))
        else:
            log.warning(f'Skipping {path.name}: no well/site or Pos in name')
            continue
        groups.setdefault(key, []).append(path)

    sites_per_well = {}
    for well, site in groups:
        sites_per_well.setdefault(well, set()).add(site)
    positions = {}
    for (well, site), files in sorted(groups.items(),
                                      key=lambda kv: _natural_sort_key(kv[0])):
        if well is None:
            name = str(site)
        elif len(sites_per_well[well]) > 1:
            name = f'{well}_site{site}'
        else:
            name = well
        positions[name] = sorted(files, key=_natural_sort_key)
    return positions


def process_ome_tiffs(
    tiff_files_path: Path,
    hdf5_dir: Path,
    phase_channel: int | None = 1,
    epi_channel: int | None = 2,
    num_channels: int = 3,
    frame_step: int = 1,
    batch_size: int = 50,
    overwrite: bool = False,
    positions: list[str] | str | None = None,
) -> None:
    """One h5 file per position (named by find_ome_positions, e.g. A3.h5).

    Each channel is read and written separately, so one channel can be
    (re)made without touching the rest of an existing file, e.g. raw 16 bit
    Epi only: phase_channel=None, overwrite=True.

    Args:
        phase_channel, epi_channel: channel index in the tiffs; None skips
            that channel.
        num_channels: channels interleaved in the tiff pages.
        frame_step: keep every frame_step'th frame.
        overwrite: replace existing Phase / Epi datasets, else raise.
    """
    log.info(f'Creating hdf5 files at {hdf5_dir} from {tiff_files_path}')
    found_positions = find_ome_positions(tiff_files_path)

    positions = [positions] if isinstance(positions, str) else positions
    if positions is not None:
        positions = {
            k: v
            for k, v in found_positions.items() if k in list(positions)
        }
    else:
        positions = found_positions
    log.info(f'Found {len(positions)} positions: {", ".join(positions)}')
    for pos, tiff_files in positions.items():
        h5 = H5File(hdf5_dir / f'{pos}.h5')
        log.info(f'Writing {h5.path} from {[f.name for f in tiff_files]}')

        def read(channel: int) -> Iterator[np.ndarray]:
            return _read_channel(tiff_files, channel, num_channels, frame_step,
                                 batch_size)

        if phase_channel is not None:
            h5.write_phase(_scaled_to_8bit(read(phase_channel)),
                           overwrite=overwrite)
        if epi_channel is not None:
            h5.write_epi(read(epi_channel), overwrite=overwrite)


def _read_channel(tiff_files: list[Path],
                  channel: int,
                  num_channels: int,
                  frame_step: int = 1,
                  batch_size: int = 50) -> Iterator[np.ndarray]:
    """(t, H, W) batches of one channel's raw frames, through a position's
    files in order (main file, then its continuation files)."""
    first_frame = 0  # of each file, counted over the whole position
    with tqdm(desc=f'Reading channel {channel}', unit='frame') as pbar:
        for file in tiff_files:
            with tifffile.TiffFile(str(file)) as tif:
                pages = tif.pages
                num_frames = len(pages) // num_channels
                # Step counted over the whole position, not restarting per file
                frames = np.arange(-first_frame % frame_step, num_frames,
                                   frame_step)
                first_frame += num_frames
                pbar.total = (pbar.total or 0) + len(frames)
                pbar.refresh()
                for start in range(0, len(frames), batch_size):
                    batch = []
                    for frame in frames[start:start + batch_size]:
                        batch.append(pages[int(frame) * num_channels +
                                           channel].asarray())
                        pbar.update()
                    yield np.stack(batch)


def _scaled_to_8bit(batches: Iterable[np.ndarray],
                    lower_percentile: float = 0,
                    upper_percentile: float = 100) -> Iterator[np.ndarray]:
    """Batches scaled to uint8, with limits from the first batch's
    percentiles applied to every batch."""
    limits = None
    for batch in batches:
        if limits is None:
            limits = np.percentile(batch, (lower_percentile, upper_percentile))
        yield to_8bit(batch, *limits)


def to_8bit(images: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Scale so min_val -> 0 and max_val -> 255, clipping outside."""
    scale = max_val - min_val if max_val != min_val else 1
    scaled = (images.astype(np.float32) - min_val) / scale
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)


def preprocessing(h5_file: Path) -> None:
    log.info(f'Applying preprocessing to {h5_file}')
    phase_preprocessing(h5_file)
    epi_preprocessing(h5_file)


def phase_preprocessing(h5_file: Path,
                        grid_size: int = 20,
                        clip_limit: float = 3.0) -> None:
    log.info(f'Running CLAHE on file {h5_file}')
    file = H5File(h5_file)
    Y, X = file.im_dims
    x_tiles = math.floor(X / (grid_size * 2)) * 2 + 1
    y_tiles = math.floor(Y / (grid_size * 2)) * 2 + 1
    clahe = cv2.createCLAHE(tileGridSize=(y_tiles, x_tiles),
                            clipLimit=clip_limit)

    def _processed_frames():
        for i in tqdm(range(file.num_frames), desc='Applying CLAHE'):
            yield clahe.apply(file.get_phase(i))

    file.write_phase(_processed_frames(), overwrite=True)
    file.write_attr('CLAHE grid size', grid_size)
    file.write_attr('CLAHE clip limit', clip_limit)


def epi_preprocessing(h5_file: Path) -> None:
    file = H5File(h5_file)

    def processed_frames():
        bg_subtractor = imagej_rolling_ball.BackgroundSubtracter(
            java_options='-Xmx2g')
        hot_pixel_mask = get_mask()
        for i in tqdm(range(file.num_frames), desc='Preprocessing Epi'):
            im = file.get_epi(i)
            im = correct_hot_pixels(im, hot_pixel_mask)
            bg_subtractor.rolling_ball_background(im,
                                                  50,
                                                  do_presmooth=True,
                                                  use_paraboloid=False,
                                                  inplace=True)

            yield im

    file.write_epi(processed_frames(), overwrite=True)
    file.delete_attr('Total Fluorescence')
    file.total_fluorescence
    file.write_attr('Hot pixel mask', HOT_PIXEL_PATH)
    file.write_attr('Hot pixel residual threshold', RESIDUAL_THRESHOLD)


def redo_epi_16bit(name: str) -> None:
    """Remake a MacrophageData folder's Epi as raw 16 bit from its tiffs (in
    _1), preprocess it, and replace each server file once it's done. Wells
    already with 16 bit Epi are skipped, so an interrupted run resumes."""
    orig_datasets_dir = Path(
        '~/thor_server/MacrophageData').expanduser() / name / '_1'
    local_datasets_dir = Path('PhagoPred') / 'Datasets' / name
    local_datasets_dir.mkdir(parents=True, exist_ok=True)
    remote_datasets_dir = Path(
        '~/thor_server/MacrophageData').expanduser() / name

    for remote_file in sorted(remote_datasets_dir.glob('*.h5')):
        if H5File(remote_file).ims_info(H5File.FLUOR)[1] == np.uint16:
            print(f'Skipping {remote_file}: Epi already 16 bit')
            continue
        local_file = local_datasets_dir / remote_file.name
        print(f'Copying {local_file.name} from remote')
        shutil.copy(remote_file, local_file)
        process_ome_tiffs(orig_datasets_dir,
                          local_datasets_dir,
                          phase_channel=None,
                          overwrite=True,
                          positions=remote_file.stem)

        # epi_preprocessing(local_file)
        H5File(local_file).repack()
        temp_file = remote_file.with_suffix('.h5.tmp')
        shutil.move(local_file, temp_file)
        os.replace(temp_file, remote_file)


if __name__ == '__main__':
    for name in ['02_10', '16_10']:
        redo_epi_16bit(name)
