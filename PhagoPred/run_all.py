from pathlib import Path
import shutil

import h5py
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.detectron_segmentation import segment, fine_tune_class, eval
from PhagoPred.tracking import trackpy
from PhagoPred.display import save, plots, napari_GUI
from PhagoPred.feature_extraction import extract_features, clean_features, features
from PhagoPred.feature_extraction.morphology.UMAP import UMAP_embedding
from PhagoPred.prediction.decision_tree import model
from PhagoPred.utils.tools import fill_missing_cells, repack_hdf5
from PhagoPred.utils.dataset_creation import epi_background_correction, keep_only_group, truncate_hdf5, hdf5_from_tiffs, rename_group, hdf5_from_ome_tiffs
import PhagoPred.display.GUI.main as GUI
from PhagoPred.tracking import trackpy_2_stage

from PhagoPred.survival_analysis.models import losses
# from PhagoPred.survival_analysis import train, validate

if __name__ == '__main__':
    datasets = Path('~/thor_server/27_02/').expanduser().iterdir()
    # dataset
    for dataset in tqdm(
            # datasets
        [
            # Path('PhagoPred') / 'Datasets' / 'E.h5',
            Path('~/thor_server/06_03/K').expanduser(),
        ]):

        # if dataset.name != '10_02_26_1':
        #     continue

        # if not dataset.is_dir():
        #     continue
        # if dataset.name in ['A', 'B', 'C', 'D', 'E']:
        #     continue
        # print(f'Processing dataset: {dataset.name}')
        h5_file = Path('PhagoPred') / 'Datasets' / f'{dataset.stem}.h5'

        # hdf5_from_tiffs(dataset, h5_file, num_frames=100)
        # rename_group(h5_file,
        #              old_group_name='Images/Fluor',
        #              new_group_name='Images/Epi')
        # with h5py.File(h5_file, 'r') as f:
        #     SETTINGS.IMAGE_SIZE = f['Images'].attrs['Image size / pixels']
        dataset = h5_file
        # # keep_only_group(dataset, ['Images'])
        # # repack_hdf5(dataset)

        # # # epi_background_correction(dataset=dataset)
        # segment.seg_dataset(dataset=dataset)
        # # trackpy.run_tracking(dataset=dataset)
        # trackpy_2_stage.run_tracking(dataset=dataset)
        # extract_features.extract_features(
        #     dataset=dataset, phase_features=[features.FirstLastFrame()])
        # fill_missing_cells(dataset=dataset)
        # extract_features.extract_features(dataset=dataset)

        # shutil.move(dataset, Path('~/thor_server/24_02/').expanduser()/dataset.name)

        GUI.run(dataset=dataset)
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
