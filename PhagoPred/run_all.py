from pathlib import Path
import shutil

import h5py
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.detectron_segmentation import segment, fine_tune_class, eval
from PhagoPred.cellpose_segmentation import segment as cellpose_segment
from PhagoPred.tracking import trackpy
from PhagoPred.display import save, plots, napari_GUI
from PhagoPred.feature_extraction import extract_features, clean_features, features
from PhagoPred.feature_extraction.morphology.UMAP import UMAP_embedding
from PhagoPred.prediction.decision_tree import model
from PhagoPred.utils.tools import fill_missing_cells, repack_hdf5, rechunk_hdf5, hdf5_needs_rechunk
from PhagoPred.utils.dataset_creation import epi_background_correction, keep_only_group, truncate_hdf5, hdf5_from_tiffs, rename_group, hdf5_from_ome_tiffs, preprocessing
import PhagoPred.display.GUI.main as GUI
from PhagoPred.tracking import trackpy_2_stage

from PhagoPred.survival_analysis.models import losses

import re


def natural_key(name):
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r'(\d+)', name)
    ]


def is_processed(h5_file):
    with h5py.File(h5_file, 'r') as f:
        return 'Cells' in f


# from PhagoPred.survival_analysis import train, validate

if __name__ == '__main__':
    orig_datasets_dir = Path(
        '~/thor_server/MacrophageData/18_09/_1').expanduser()
    machine_dataset_dir = Path('PhagoPred') / 'Datasets' / '18_09'
    remote_datasets_dir = Path(
        '~/thor_server/MacrophageData/18_09').expanduser()

    # hdf5_from_ome_tiffs(orig_datasets_dir, remote_datasets_dir)
    # Files are moved here while being processed and back when finished, so
    # a file left here is from an interrupted run and is resumed in place
    names = {p.name for p in remote_datasets_dir.glob('*.h5')}
    names |= {p.name for p in machine_dataset_dir.glob('*.h5')}
    for name in sorted(names, key=natural_key):
        h5_file = machine_dataset_dir / name
        if not h5_file.exists():
            if is_processed(remote_datasets_dir / name):
                print(f'Skipping {name}: already processed')
                continue
            shutil.move(remote_datasets_dir / name, machine_dataset_dir)
        print(f'Processing {name}')
        with h5py.File(h5_file, 'r') as f:
            SETTINGS.IMAGE_SIZE = f['Images'].attrs['Image size / pixels']
        if h5_file.name not in ['1.h5', '2.h5', '3.h5', '4.h5']:
            preprocessing(h5_file)
        cellpose_segment.seg_dataset(h5_file)
        trackpy_2_stage.run_tracking(h5_file)
        extract_features.extract_features(
            h5_file, phase_features=[features.FirstLastFrame()])

        fill_missing_cells(h5_file)

        extract_features.extract_features(h5_file)

        shutil.move(h5_file, remote_datasets_dir)

# if __name__ == '__main__':
#     datasets = Path(
#         '~/thor_server/MacrophageData/14_08/').expanduser().iterdir()
#     # dataset
#     h5_paths = [
#         Path('PhagoPred') / 'Datasets' / 'B.h5',
#         # "PhagoPred\\Datasets\\B.h5",
#         # "C:\\Users\\php23rjb\\Downloads\\A.h5",
#         # "C:\\Users\\php23rjb\\Downloads\\E.h5",
#         # "C:\\Users\\php23rjb\\Downloads\\C.h5",
#         # "C:\\Users\\php23rjb\\Downloads\\D.h5"
#     ]
#     for dataset in tqdm(
#             datasets
#             # h5_paths
#             # # [
#             # #     # Path('PhagoPred') / 'Datasets' / 'E.h5',
#             # #     # Path('~/thor_server/06_03/K').expanduser(),
#             # #     Path("C:\\Users\\php23rjb\\Downloads\\C.h5")
#             # # ]
#     ):
#         # dataset = Path(dataset)
#         # if dataset.name != '10_02_26_1':
#         #     continue
#         if not dataset.name == 'A.h5':
#             continue
#         # if dataset.name in ['Test', 'A', 'B', 'C', 'D', 'E']:
#         #     continue
#         # if not dataset.is_dir():
#         #     continue
#         # if dataset.stem in [
#         #         'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
#         #         'N', 'O'
#         # ]:
#         #     continue

#         print(f'Processing dataset: {dataset.name}')
#         h5_file = Path('PhagoPred') / 'Datasets' / f'{dataset.stem}.h5'
#         hdf5_from_tiffs(dataset, h5_file, frame_steps={'Phase': 3, 'Fluor': 1})
#         rename_group(h5_file,
#                      old_group_name='Images/Fluor',
#                      new_group_name='Images/Epi')

#         with h5py.File(h5_file, 'r') as f:
#             SETTINGS.IMAGE_SIZE = f['Images'].attrs['Image size / pixels']
#         dataset = h5_file

#         # segment.seg_dataset(dataset=dataset)
#         cellpose_segment.seg_dataset(dataset)
#         trackpy_2_stage.run_tracking(dataset=dataset)
#         extract_features.extract_features(
#             dataset=dataset, phase_features=[features.FirstLastFrame()])

#         fill_missing_cells(dataset=dataset)

#         extract_features.extract_features(dataset=dataset)
#         # extract_features.extract_features(dataset=h5_file,
#         #                                   phase_features=[
#         #                                       features.Fluorescence(),
#         #                                       features.ExternalFluorescence()
#         #                                   ])
#         # shutil.move(h5_file, dataset)
#         # shutil.move(
#         #     dataset,
#         #     Path('~/thor_server/MacrophageData/14_08/').expanduser() /
#         #     dataset.name)
#         extract_features.extract_features(dataset, [
#             features.RegionProps(),
#         ])
# GUI.run(dataset=dataset)
# truncate_hdf5(dataset, dataset.parent / f"truncated_{dataset.name}", start_frame = 0, end_frame=300)

# trackpy.run_tracking()
# extract_features.extract_features()
# segment.main()
# trackpy.main()
# # save.main()
# extract_features.main()
# plots.main()
# fitting.main()
# napari_GUI.main()
# model.main()
# fine_tune_class.main()
# eval.main()
# losses.main()
# train.main()
# validate.main()
