from pathlib import Path
import shutil

import h5py
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.segmentation.detectron_segmentation import segment
from PhagoPred.segmentation.cellpose_segmentation import segment as cellpose_segment
from PhagoPred.feature_extraction import extract_features, clean_features, features
from PhagoPred.feature_extraction.morphology.UMAP import UMAP_embedding
from PhagoPred.segmentation.detectron_segmentation import eval, fine_tune_class
from PhagoPred.utils.tools import fill_missing_cells, repack_hdf5, rechunk_hdf5, hdf5_needs_rechunk
from PhagoPred.utils.dataset_creation import epi_background_correction, keep_only_group, truncate_hdf5, hdf5_from_tiffs, rename_group, hdf5_from_ome_tiffs, preprocessing
import PhagoPred.display.GUI.main as GUI
from PhagoPred.tracking import trackpy_2_stage

if __name__ == '__main__':
    orig_datasets_dir = Path(
        '~/thor_server/MacrophageData/18_09/_1').expanduser()
    machine_dataset_dir = Path('PhagoPred') / 'Datasets' / '18_09'
    remote_datasets_dir = Path(
        '~/thor_server/MacrophageData/18_09').expanduser()

    # hdf5_from_ome_tiffs(orig_datasets_dir, remote_datasets_dir)
    for h5_file in remote_datasets_dir.glob('*.h5'):
        shutil.move(h5_file, machine_dataset_dir)
        h5_file = machine_dataset_dir / h5_file.name
        with h5py.File(h5_file, 'r') as f:
            SETTINGS.IMAGE_SIZE = f['Images'].attrs['Image size / pixels']
        if h5_file.name not in ['1.h5', '2.h5', '3.h5']:
            preprocessing(h5_file)
        cellpose_segment.seg_dataset(h5_file)
        trackpy_2_stage.run_tracking(h5_file)
        extract_features.extract_features(
            h5_file, phase_features=[features.FirstLastFrame()])

        fill_missing_cells(h5_file)

        extract_features.extract_features(h5_file)

        shutil.move(h5_file, remote_datasets_dir)
