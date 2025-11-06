import numpy as np
import os
import sys
import tifffile
import h5py
from pathlib import Path
from tqdm import tqdm
import cv2
import scipy
from concurrent.futures import ThreadPoolExecutor

from PhagoPred import SETTINGS

def compute_percentile_limits(
        images: np.ndarray,
        lower_percentile: float = 0.1,
        upper_percentile: float = 99.9
        ) -> tuple[float, float]:
    
    """
    Compute the intensity limits on images according to lower_percentile, upper_percentile.
    Parameters:
        images: (T, X, Y)
        lower_percentile: %
        upper_percenrile: %
    Returns
        lower limit, upper limit
    """
    images = images.astype(np.float32)

    min_val = np.percentile(images, lower_percentile)
    max_val = np.percentile(images, upper_percentile)

    return min_val, max_val

def to_8bit(
        images: np.array,
        min_val: float = None,
        max_val: float = None
        ) -> np.ndarray:
    
    """ 
    Scale images to 8bit using min_val and max_val
    Parameters:
        images: (T, X, Y)
        min_val: value which will go to 0 in 8bit array
        max_val: value which will go to 2**8 - 1 in 8bit array
    Returns:
        np.array (T, X, Y) dtype=np.uint8
    """
    scale = max_val - min_val if max_val != min_val else 1
    scaled = (images - min_val) / scale
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)

def set_hdf5_metadata(dataset: h5py.Dataset, x_size: int, y_size: int) -> None:
    """
    Set metadats associated with hdf5 datset.
    Parameters:
        dataset: open hdf5 datset to assign metadats to
        x_size: image x dimension
        y_size: image y dimension
    """
    dataset.attrs['Camera'] = "Orca Flash 4.0, C11440"
    dataset.attrs['Pixel Size / um'] = 6.5
    dataset.attrs['Image size / pixels'] = [y_size, x_size]
    dataset.attrs['Objective magnification'] = 20
    res_um = dataset.attrs['Pixel Size / um'] / dataset.attrs['Objective magnification']
    dataset.attrs['Resolution / um'] = res_um
    dataset.attrs['FOV / um'] = [y_size * res_um, x_size * res_um]
    dataset.attrs['Phase exposure / ms'] = 500
    dataset.attrs['Epi exposure / ms'] = 0
    dataset.attrs['Time interval / s'] = 300
    dataset.attrs['Objective NA'] = 0.45
    dataset.attrs['Number of frames'] = 0  # to be updated later
    dataset.attrs['Notes'] = ("J774 macrophages, P11. Imaged in Flurobrite. 1*10**5 cells per dish. 07/10/2025")


def hdf5_from_tiffs(tiff_files_path: Path, hdf5_file: Path,
                    phase_channel: int = 1, epi_channel: int = 2,
                    frame_step_size: int = 1, batch_size: int = 50) -> None:
    """
    Convert multi-file, multi-page .ome.tif files to HDF5 with batched reading and
    global min/max scaling for quantitative uint8 conversion.
    """

    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)

    print(f'Tiff files found? {tiff_files_path.exists()}')
    print(f'HDF5 path found? {hdf5_file.parent.exists()}')

    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', str(s))]

    tiff_files = sorted(tiff_files_path.glob("*.ome.tif"), key=natural_sort_key)
    if not tiff_files:
        raise ValueError("No .ome.tif files found.")

    # Get shape info from first file
    with tifffile.TiffFile(str(tiff_files[0])) as tif:
        series = tif.series[0]
        shape = series.shape
        T, C, Y, X = shape

    with h5py.File(hdf5_file, 'w') as h:
        Images = h.create_group('Images')
        set_hdf5_metadata(Images, X, Y)

        phase_ds = Images.create_dataset('Phase', shape=(0, Y, X), maxshape=(None, Y, X),
                                         dtype='uint8', chunks=(1, Y, X))
        epi_ds = Images.create_dataset('Epi', shape=(0, Y, X), maxshape=(None, Y, X),
                                       dtype='uint8', chunks=(1, Y, X))

        phase_min, phase_max = None, None
        epi_min, epi_max = None, None

        frame_count = 0

        for i, file in enumerate(tiff_files):
            sys.stdout.write(f"\rProcessing file {i + 1} / {len(tiff_files)}, {frame_count} frames processed")
            sys.stdout.flush()

            with tifffile.TiffFile(str(file)) as tif:
                pages = tif.pages
                num_pages = len(pages)
                num_frames = num_pages // C

                selected_frames = np.arange(0, num_frames, frame_step_size)

                for batch_start in tqdm(range(0, len(selected_frames), batch_size), desc=f'Batch {i + 1}'):
                    batch_end = min(batch_start + batch_size, len(selected_frames))
                    batch_frames = selected_frames[batch_start:batch_end]

                    phase_page_idxs = batch_frames * C + phase_channel
                    epi_page_idxs = batch_frames * C + epi_channel

                    # Read one frame at a time to reduce memory usage
                    phase_batch = np.stack([pages[i].asarray() for i in phase_page_idxs])
                    epi_batch = np.stack([pages[i].asarray() for i in epi_page_idxs])

                    # Compute global scaling from first batch
                    if phase_min is None:
                        phase_min, phase_max = compute_percentile_limits(phase_batch, 0, 100)
                    if epi_min is None:
                        epi_min, epi_max = compute_percentile_limits(epi_batch, 1, 99)

                    phase_batch = to_8bit(phase_batch, phase_min, phase_max)
                    epi_batch = to_8bit(epi_batch, epi_min, epi_max)

                    # Resize and write to HDF5
                    batch_len = len(phase_batch)
                    phase_ds.resize((frame_count + batch_len, Y, X))
                    epi_ds.resize((frame_count + batch_len, Y, X))

                    phase_ds[frame_count:frame_count + batch_len] = phase_batch
                    epi_ds[frame_count:frame_count + batch_len] = epi_batch

                    frame_count += batch_len

        Images.attrs['Number of frames'] = frame_count
        print(f"\nHDF5 file created: {frame_count} frames")

# def hdf5_from_tiffs(tiff_files_path: Path, hdf5_file: Path,
#                     phase_channel: int = 1, epi_channel: int = 2, frame_step_size: int=1) -> None:
#     """Convert multi-file, multi-page .ome.tif files to HDF5 with batched reading and
#     global min/max scaling for quantitative uint8 conversion."""

#     if os.path.exists(hdf5_file):
#         os.remove(hdf5_file)

#     print(f'Tiff files found? {tiff_files_path.exists()}')
#     print(f'HDF5 path found? {hdf5_file.parent.exists()}')

#     def natural_sort_key(s):
#         import re
#         return [int(text) if text.isdigit() else text.lower()
#                 for text in re.split(r'(\d+)', str(s))]

#     tiff_files = sorted(tiff_files_path.glob("*.ome.tif"), key=natural_sort_key)
#     if not tiff_files:
#         raise ValueError("No .ome.tif files found.")

#     # Get shape info from first file
#     with tifffile.TiffFile(str(tiff_files[0])) as tif:
#         series = tif.series[0]
#         shape = series.shape
#         T, C, Y, X = shape
    
#     with h5py.File(hdf5_file, 'w') as h:

#         Images = h.create_group('Images')
#         set_hdf5_metadata(Images, X, Y)

#         phase_ds = Images.create_dataset('Phase', shape=(0, Y, X), maxshape=(None, Y, X),
#                                          dtype='uint8', chunks=(1, Y, X))
#         epi_ds = Images.create_dataset('Epi', shape=(0, Y, X), maxshape=(None, Y, X),
#                                          dtype='uint8', chunks=(1, Y, X))
    
#         phase_min, phase_max = None, None
#         epi_min, epi_max = None, None

#         frame_count = 0

#         for i, file in enumerate(tiff_files):
#             sys.stdout.write(f"\rProcessing file {i + 1} / {len(tiff_files)}, {frame_count} frames processed")
#             sys.stdout.flush()
#             with tifffile.TiffFile(str(file)) as tif:
#                 pages = tif.pages
#                 num_pages = len(pages)
#                 num_frames = num_pages // C

#                 selected_frames = np.arange(0, num_frames, frame_step_size)
                
#                 num_frames = len(selected_frames)
                
#                 phase_page_idxs = selected_frames * C + phase_channel
#                 epi_page_idxs = selected_frames * C + epi_channel

#                 phase_array = np.stack([pages[i].asarray() for i in phase_page_idxs])
#                 epi_array = np.stack([pages[i].asarray() for i in epi_page_idxs])

#                 if phase_min is None:
#                     phase_min, phase_max = compute_percentile_limits(phase_array, 0, 100)
#                 if epi_min is None:
#                     epi_min, epi_max = compute_percentile_limits(epi_array, 1, 99)
                
#                 phase_array = to_8bit(phase_array, phase_min, phase_max)
#                 epi_array = to_8bit(epi_array, epi_min, epi_max)

#                 phase_ds.resize((frame_count+num_frames, Y, X))
#                 epi_ds.resize((frame_count + num_frames, Y, X))

#                 phase_ds[frame_count:frame_count+num_frames] = phase_array
#                 epi_ds[frame_count:frame_count+num_frames] = epi_array

#                 frame_count += num_frames
        
#         Images.attrs['Number of frames'] = frame_count
#         print(f"\nHDF5 file created: {frame_count} frames")

def make_short_test_copy(orig_file: Path, short_file: Path, start_frame: int = 0, end_frame: int = 50) -> None:
    """
    Copy the images group from the orig_file, taking only frames from start_frame to end_frame
    """
    group_name = 'Images'
    with h5py.File(orig_file, 'r') as orig:
        with h5py.File(short_file, 'x') as short:
            orig_group = orig[group_name]
            short_group = short.create_group(group_name)

            for name, dataset in orig_group.items():
                if isinstance(dataset, h5py.Dataset):
                    print(f"Copying dataset '{name}'")
                    try:
                        # Extract the slice of data
                        sliced_data = dataset[start_frame:end_frame].copy()
                        print(f"{name} sliced data shape: {sliced_data.shape}, dtype: {sliced_data.dtype}")
                        # Get creation properties
                        kwargs = {
                            'dtype': dataset.dtype,
                            'compression': dataset.compression,
                            'compression_opts': dataset.compression_opts,
                            'chunks': dataset.chunks,
                            'shuffle': dataset.shuffle,
                            'fletcher32': dataset.fletcher32,
                            'maxshape': dataset.maxshape
                        }

                        # Create dataset in destination
                        short_group.create_dataset(
                            name,
                            data=sliced_data,
                            shape=sliced_data.shape,
                            **{k: v for k, v in kwargs.items() if v is not None}
                            )
                        short.flush()

                    except Exception as e:
                        print(e)

            for attr_name, attr_value in orig_group.attrs.items():
                        short_group.attrs[attr_name] = attr_value
                    
            short_group.attrs['Number of frames'] = end_frame - start_frame
            short.flush()

def keep_only_group(hdf5_path, keep_groups=['Images']):
    with h5py.File(hdf5_path, 'r+') as f:
        # List all top-level groups
        all_groups = list(f.keys())

        for group in all_groups:
            if group not in keep_groups:
                print(f"Deleting group: {group}")
                del f[group]  # Deletes the group and its contents

        # Ensure changes are written to disk
        f.flush()

    print(f"Only '{keep_groups}' group retained.")

def epi_background_correction(dataset=SETTINGS.DATASET):
    """Averge all epi images, then subtract avrgaed image from each frame. Gaussian blur then rescale to 8bit taking only top 10% of pixel values"""
    with h5py.File(dataset, 'r+') as f:
        epi_ds = f['Images']['Epi']
        d = epi_ds.shape[0]
        sum_image = np.zeros(epi_ds.shape[1:], dtype=np.float32)

        
        for i in tqdm(range(d), desc='Avergaing all epi images'):
            sum_image += epi_ds[i].astype(np.float32) / d

        print('Sampling pixel values...')
        sample_pixel_values = []
        for i in np.linspace(0, d-1, 50).astype(int):
            sample_pixel_values.append(np.clip((epi_ds[i] - sum_image).astype(np.float32), a_min=0, a_max=None))
        all_pixel_values = np.concatenate(sample_pixel_values)
        p0 = np.percentile(all_pixel_values, 0)
        p99_9 = np.percentile(all_pixel_values, 99.9)

        for i in tqdm(range(d), desc='Subtracting background'):
            corrected = np.clip(epi_ds[i].astype(np.float32) - sum_image, a_min=0, a_max=None)
            # corrected = cv2.medianBlur(corrected, 5)
            corrected = cv2.GaussianBlur(corrected, (31, 31), 0)
            epi_ds[i] = to_8bit(corrected, p0, p99_9)
            # epi_ds[i] = to_8bit(corrected, np.percentile(corrected, 90), np.percentile(corrected, 99.9))

        epi_ds.attrs['Background corrected'] = True

# def epi_background_correction_gaussian(dataset=SETTINGS.DATASET, sigma: int = 200) -> None:
#     """For each frame, aplpy gaussian smoothing to approximate background signal and subtract."""
#     with h5py.File(dataset, 'r+') as f:
#         epi_ds = f['Images']['Epi']
#         for frame in tqdm(range(epi_ds.shape[0])):
#             epi_im = epi_ds[frame]
#             bg_estimate = scipy.ndimage.gaussian_filter(epi_im, sigma=sigma)
#             # epi_ds[frame] = np.clip(epi_im / bg_estimate
#             epi_im = epi_im - bg_estimate
#             epi_im = np.clip(epi_im, 0, None)
#             epi_ds[frame] = epi_im

def epi_background_correction_gaussian(dataset=SETTINGS.DATASET, sigma: int = 100, max_workers: int = 4) -> None:
    """Apply Gaussian background correction to each Epi frame using multithreading."""
    
    with h5py.File(dataset, 'r+') as f:
        epi_ds = f['Images']['Epi']
        n_frames, height, width = epi_ds.shape

        def process_frame(i):
            epi_im = epi_ds[i].astype(np.float32)  # cast once to avoid precision issues
            bg_estimate = scipy.ndimage.gaussian_filter(epi_im, sigma=sigma)
            corrected = np.clip(epi_im - bg_estimate, 0, None).astype(np.uint8)
            return i, corrected

        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_frame, i) for i in range(n_frames)]

            for future in tqdm(futures, desc="Correcting frames"):
                i, corrected_frame = future.result()
                epi_ds[i] = corrected_frame
        
def create_survival_analysis_dataset(dataset_path: Path, new_path: Path = None) -> None:
    """Create a new datset with only 'Cells group' and with chunks set to (num_frames, 1)"""
    if new_path is None:
        new_path = dataset_path.parent / f"{dataset_path.stem}_survival{dataset_path.suffix}"
    if os.path.exists(new_path):
        os.remove(new_path)
    with h5py.File(dataset_path, 'r') as orig:
        with h5py.File(new_path, 'x') as new:
            orig_group = orig['Cells']['Phase']
            new_group = new.create_group('Cells/Phase')

            for name, dataset in tqdm(orig_group.items(), desc='Copying datasets'):
                if isinstance(dataset, h5py.Dataset):
                    print(f"Copying dataset '{name}'")
                    try:
                        # Get creation properties
                        kwargs = {
                            'dtype': dataset.dtype,
                            'compression': dataset.compression,
                            'compression_opts': dataset.compression_opts,
                            'chunks': (dataset.shape[0], 1),
                            'shuffle': dataset.shuffle,
                            'fletcher32': dataset.fletcher32,
                            'maxshape': dataset.maxshape
                        }

                        # Create dataset in destination
                        new_group.create_dataset(
                            name,
                            data=dataset[...],
                            shape=dataset.shape,
                            **{k: v for k, v in kwargs.items() if v is not None}
                            )
                        new.flush()

                    except Exception as e:
                        print(e)

            for attr_name, attr_value in orig_group.attrs.items():
                        new_group.attrs[attr_name] = attr_value
                    
            new.flush()

def split_by_radius(hdf5_file: Path) -> None:
    """Split cell dataset into two groups according to distance from image centre.
    Threshold radius will be $\frac{L}{\sqrt(2 \pi)}$, where L is height/width of image. (Assuming square image)."""
    with h5py.File(hdf5_file, 'r') as f:
        features_group = f['Cells']['Phase']
        frames, h, w =f['Images']['Phase'].shape
        centre_coord = np.array([w//2, w//2])
        threshold_radius = w / np.sqrt(2*np.pi)
        cell_positions = np.stack((features_group['X'], features_group['Y']), axis=-1)
        dist_from_centre = np.linalg.norm(cell_positions - centre_coord[np.newaxis, np.newaxis, :], axis=-1)
        inner_region_mask = np.nanmax(dist_from_centre, axis=0) < threshold_radius
        outer_region_mask = np.nanmin(dist_from_centre, axis=0) > threshold_radius
        
        def copy_dataset(src_group, dst_group, cell_mask):
            for feature in src_group.keys():
                src_dset = src_group[feature]
                # Preserve chunks, compression, dtype
                dst_dset = dst_group.create_dataset(
                    feature,
                    shape=(src_dset.shape[0], np.sum(cell_mask)),
                    dtype=src_dset.dtype,
                    chunks=True,
                    compression=src_dset.compression,
                    compression_opts=src_dset.compression_opts
                )
                # Copy data selectively
                dst_dset[:] = src_dset[:, cell_mask]

        # Save inner region
        inner_file = hdf5_file.parent / f'{hdf5_file.stem}_inner{hdf5_file.suffix}'
        with h5py.File(inner_file, 'w') as f_inner:
            inner_group = f_inner.create_group('Cells/Phase')
            copy_dataset(features_group, inner_group, inner_region_mask)

        # Save outer region
        outer_file = hdf5_file.parent / f'{hdf5_file.stem}_outer{hdf5_file.suffix}'
        with h5py.File(outer_file, 'w') as f_outer:
            outer_group = f_outer.create_group('Cells/Phase')
            copy_dataset(features_group, outer_group, outer_region_mask)
            
        # print(cell_positions.shape)
        
if __name__ == '__main__':
    # keep_only_group("/home/ubuntu/PhagoPred/PhagoPred/Datasets/ExposureTest/07_10_0.h5")
    # keep_only_group("/home/ubuntu/PhagoPred/PhagoPred/Datasets/ExposureTest/10_10_5000.h5")
    hdf5_from_tiffs(Path("~/thor_server/28_10_no_staph_1").expanduser(), 
                    # Path('D:/27_05.h5'),
                    Path("~/PhagoPred/PhagoPred/Datasets/ExposureTest/28_10_2500.h5").expanduser(),
                    phase_channel=2,
                    epi_channel=1,
                    frame_step_size=1,
                    )
    # epi_background_correction()
    # epi_background_correction_gaussian()
    # create_survival_analysis_dataset(Path('PhagoPred') / 'Datasets' / 'ExposureTest'/ '10_10_5000.h5', Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000_survival.h5')
    # create_survival_analysis_dataset(Path('PhagoPred') / 'Datasets' / 'ExposureTest'/ '07_10_0.h5', Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0_survival.h5')
    # make_short_test_copy(Path("C:/Users/php23rjb/Documents/PhagoPred/PhagoPred/Datasets/27_05.h5"),
    #                      Path("C:/Users/php23rjb/Documents/PhagoPred/PhagoPred/Datasets/27_05_500.h5"),
    #                      start_frame=3000,
    #                      end_frame=3500)
    # Create dummy data similar to your sliced data
    # data = np.random.randint(0, 256, size=(50, 2048, 2048), dtype=np.uint8)
    # split_by_radius(Path("/home/ubuntu/PhagoPred/PhagoPred/Datasets/ExposureTest/10_10_5000.h5"))
    # with h5py.File(Path("/home/ubuntu/PhagoPred/PhagoPred/Datasets/27_05_tets.h5"), "w") as f:
    #     dset = f.create_dataset("Epi", data=data)
    #     f.flush()

    # import os
    # print("File size:", os.path.getsize(Path("/home/ubuntu/PhagoPred/PhagoPred/Datasets/27_05_tets.h5")), "bytes")
        

