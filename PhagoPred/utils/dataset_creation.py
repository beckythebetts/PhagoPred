import numpy as np
import os
import sys
import tifffile
import h5py
from pathlib import Path
from tqdm import tqdm
import cv2

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
    dataset.attrs['Phase exposure / ms'] = 1000
    dataset.attrs['Epi exposure / ms'] = 10000
    dataset.attrs['Time interval / s'] = 60
    dataset.attrs['Objective NA'] = 0.45
    dataset.attrs['Number of frames'] = 0  # to be updated later
    dataset.attrs['Notes'] = ("J774 macrophages, SH1000 S.Aureus MCherry, grown in TSB + 0.1% Tet overnight. "
                            "Imaged in DMEM. 1*10**5 cells per dish 1:1 MOI.")
    
def hdf5_from_tiffs(tiff_files_path: Path, hdf5_file: Path,
                    phase_channel: int = 1, epi_channel: int = 2) -> None:
    """Convert multi-file, multi-page .ome.tif files to HDF5 with batched reading and
    global min/max scaling for quantitative uint8 conversion."""

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

                phase_page_idxs = np.arange(num_frames) * C + phase_channel
                epi_page_idxs = np.arange(num_frames) * C + epi_channel

                phase_array = np.stack([pages[i].asarray() for i in phase_page_idxs])
                epi_array = np.stack([pages[i].asarray() for i in epi_page_idxs])

                if phase_min is None:
                    phase_min, phase_max = compute_percentile_limits(phase_array, 0, 100)
                if epi_min is None:
                    epi_min, epi_max = compute_percentile_limits(epi_array, 1, 99)
                
                phase_array = to_8bit(phase_array, phase_min, phase_max)
                epi_array = to_8bit(epi_array, epi_min, epi_max)

                phase_ds.resize((frame_count+num_frames, Y, X))
                epi_ds.resize((frame_count + num_frames, Y, X))

                phase_ds[frame_count:frame_count+num_frames] = phase_array
                epi_ds[frame_count:frame_count+num_frames] = epi_array

                frame_count += num_frames
        
        Images.attrs['Number of frames'] = frame_count
        print(f"\nHDF5 file created: {frame_count} frames")

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

if __name__ == '__main__':
    # keep_only_group("/home/ubuntu/PhagoPred/PhagoPred/Datasets/27_05_500_seg.h5")
    # hdf5_from_tiffs(Path("~/thor_server/16_09_1").expanduser(), 
    #                 # Path('D:/27_05.h5'),
    #                 Path("~/PhagoPred/PhagoPred/Datasets/16_09_1.h5").expanduser(),
    #                 phase_channel=1,
    #                 epi_channel=2,
    #                 )
    epi_background_correction()
    # make_short_test_copy(Path("C:/Users/php23rjb/Documents/PhagoPred/PhagoPred/Datasets/27_05.h5"),
    #                      Path("C:/Users/php23rjb/Documents/PhagoPred/PhagoPred/Datasets/27_05_500.h5"),
    #                      start_frame=3000,
    #                      end_frame=3500)
    # Create dummy data similar to your sliced data
    # data = np.random.randint(0, 256, size=(50, 2048, 2048), dtype=np.uint8)

    # with h5py.File(Path("/home/ubuntu/PhagoPred/PhagoPred/Datasets/27_05_tets.h5"), "w") as f:
    #     dset = f.create_dataset("Epi", data=data)
    #     f.flush()

    # import os
    # print("File size:", os.path.getsize(Path("/home/ubuntu/PhagoPred/PhagoPred/Datasets/27_05_tets.h5")), "bytes")
        

