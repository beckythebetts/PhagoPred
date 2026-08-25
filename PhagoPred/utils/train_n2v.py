from __future__ import annotations
from pathlib import Path
import ctypes
import math
import multiprocessing as mp

from tqdm import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt
from careamics import CAREamist
from careamics.config import create_n2v_configuration
import tifffile

# CuPy/cuCIM load libnvrtc.so.12 via ctypes at runtime, but the version pip
# installs (nvidia-cuda-nvrtc-cu12) sits in a package directory that isn't on
# the dynamic linker's search path. Preload it explicitly so CuPy finds it
# regardless of LD_LIBRARY_PATH.
import nvidia.cuda_nvrtc

ctypes.CDLL(
    str(Path(nvidia.cuda_nvrtc.__file__).parent / 'lib' / 'libnvrtc.so.12'),
    mode=ctypes.RTLD_GLOBAL,
)

import cucim.skimage.filters
import cupy as cp

from PhagoPred.utils.logger import get_logger

log = get_logger()


def load_epi_stack(hdf5_files: list[Path],
                   num_frames: int | None = None) -> np.ndarray:
    frames_per_file = math.ceil(num_frames / len(hdf5_files))
    all_ims = []
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, 'r') as f:
            frames = np.random.choice(np.arange)
            epi = f['Images']['Epi']
            frames = np.random.choice(np.arange(epi.shape[0]), frames_per_file)
            all_ims.append(epi[frames].astype(np.float32))
    all_ims = np.concatenate(all_ims, axis=0)
    return all_ims


def load_tiff_epi_stack(dirs: list[Path],
                        frames_per_file: int = 3,
                        bg_removal_radius: int | None = 200,
                        return_raw: bool = False) -> np.ndarray:
    # import imagej_rolling_ball
    all_ims = []
    raw_ims = []
    # bg_subtractor = imagej_rolling_ball.BackgroundSubtracter()
    for dir in tqdm(dirs, desc='Loading images'):

        fluor_ims = dir / 'Fluor'
        if fluor_ims.exists():
            log.info(f'Getting {frames_per_file} ims from {dir}')
            file_ims = list(fluor_ims.glob('*'))
            rnd_idxs = np.random.choice(np.arange(len(file_ims)),
                                        frames_per_file)
            for i in rnd_idxs:
                im = tifffile.imread(file_ims[i])
                raw_ims.append(im.copy())
                if bg_removal_radius is not None:
                    im = cp.asarray(im, dtype=cp.float32)
                    bg = cucim.skimage.filters.gaussian(
                        im, sigma=bg_removal_radius)
                    im -= bg
                    im = cp.asnumpy(im)
                all_ims.append(im)
    all_ims = np.stack(all_ims)
    log.info(f'Got {len(all_ims)} images.')
    if return_raw:
        return all_ims, np.stack(raw_ims)
    return all_ims


def _load_tiff_epi_stack_worker(dirs: list[Path], frames_per_file: int,
                                return_raw: bool, queue: mp.Queue) -> None:
    queue.put(load_tiff_epi_stack(dirs, frames_per_file,
                                  return_raw=return_raw))


def load_tiff_epi_stack_isolated(dirs: list[Path],
                                 frames_per_file: int = 3,
                                 return_raw: bool = False) -> np.ndarray:
    """Runs load_tiff_epi_stack (which starts an embedded JVM via
    imagej_rolling_ball) in a separate subprocess, so the JVM's signal
    handlers never share a process with CUDA/PyTorch - mixing the two
    in-process causes segfaults after training."""
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    process = ctx.Process(target=_load_tiff_epi_stack_worker,
                          args=(dirs, frames_per_file, return_raw, queue))
    process.start()
    result = queue.get()
    process.join()
    return result


def train(ims: np.ndarray,
          work_dir: Path,
          patch_size: int = 64,
          batch_size: int = 64,
          num_epochs: int = 100) -> None:

    config = create_n2v_configuration(
        experiment_name="epi_n2v",
        data_type="array",
        axes="SYX",
        patch_size=[patch_size, patch_size],
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    careamist = CAREamist(source=config, work_dir=str(work_dir))
    log.info('Starting NV2 training')
    careamist.train(train_source=ims)


def test_model(model_dir: Path,
               test_im: np.ndarray,
               raw_im: np.ndarray | None = None) -> None:
    checkpoints = (Path(model_dir) / 'checkpoints').glob('last*.ckpt')
    checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    careamist = CAREamist(source=str(checkpoint), work_dir=str(model_dir))

    denoised = careamist.predict(test_im.astype(np.float32),
                                 axes='YX',
                                 data_type='array')
    if isinstance(denoised, list):
        denoised = denoised[0]
    denoised = np.squeeze(denoised)

    test_vmin, test_vmax = np.percentile(test_im, [0.1, 99.9])
    denoised_vmin, densoised_vmax = np.percentile(denoised, [0.1, 99.9])

    num_panels = 3 if raw_im is not None else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 6))

    if raw_im is not None:
        raw_vmin, raw_vmax = np.percentile(raw_im, [0.1, 99.9])
        axes[0].imshow(raw_im, cmap='gray', vmin=raw_vmin, vmax=raw_vmax)
        axes[0].set_title('Raw')
        axes = axes[1:]

    axes[0].imshow(test_im, cmap='gray', vmin=test_vmin, vmax=test_vmax)
    axes[0].set_title('Background removed')
    axes[1].imshow(denoised,
                   cmap='gray',
                   vmin=denoised_vmin,
                   vmax=densoised_vmax)
    axes[1].set_title('Denoised')
    for ax in fig.axes:
        ax.axis('off')
    fig.tight_layout()
    plt.savefig(model_dir / 'example.png')


if __name__ == '__main__':

    model_dir = Path('/home/ubuntu/PhagoPred/PhagoPred/Datasets/nv2_model')

    # im_paths = list(
    #     Path('~/thor_server/MacrophageData/14_08/').expanduser().glob('*'))
    # im_paths = [p for p in im_paths if p.is_dir()]
    # ims = load_tiff_epi_stack_isolated(im_paths, 5)
    example_im, raw_im = load_tiff_epi_stack(
        [Path('~/thor_server/MacrophageData/14_08/A').expanduser()],
        1,
        return_raw=True)

    example_im, raw_im = example_im[0], raw_im[0]

    # train(
    #     ims,
    #     Path('/home/ubuntu/PhagoPred/PhagoPred/Datasets/nv2_model'),
    # )

    test_model(model_dir, example_im, raw_im)
