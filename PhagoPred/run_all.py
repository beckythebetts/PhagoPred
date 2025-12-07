from pathlib import Path
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.detectron_segmentation import segment, fine_tune_class, eval
from PhagoPred.tracking import trackpy
from PhagoPred.display import save, plots, napari_GUI
from PhagoPred.feature_extraction import extract_features, clean_features, features
from PhagoPred.feature_extraction.morphology.UMAP import UMAP_embedding
from PhagoPred.prediction.decision_tree import model
from PhagoPred.utils.tools import fill_missing_cells, repack_hdf5
from PhagoPred.utils.dataset_creation import epi_background_correction, keep_only_group, truncate_hdf5
import PhagoPred.display.GUI.main as GUI

from PhagoPred.survival_analysis.models import losses
# from PhagoPred.survival_analysis import train, validate


if __name__ == '__main__':
    datasets = (Path('PhagoPred')/'Datasets'/ 'ExposureTest').iterdir()
    for dataset in tqdm(
        datasets
        # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / 'old' / '03_10_2500.h5',
        # Path('PhagoPred')/'Datasets'/ 'Prel/
        # Path('PhagoPred')/'Datasets'/ 'Prelims' / '16_09_1 (another copy).h5',
        # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '21_10_2500.h5',
        # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '07_10_0.h5',
        # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '10_10_5000.h5',
    ):
        # keep_only_group(dataset, ['Images'])
        # repack_hdf5(dataset)
        # # epi_background_correction(dataset=dataset)
        # segment.seg_dataset(dataset=dataset)
        # trackpy.run_tracking(dataset=dataset)
        # extract_features.extract_features(dataset=dataset, phase_features=[features.FirstLastFrame()])
        # fill_missing_cells(dataset=dataset)
        # extract_features.extract_features(dataset=dataset)
        # extract_features.extract_features(dataset=dataset, phase_features = [features.FirstLastFrame()])
        # clean_features.remove_bad_frames(dataset=dataset)
        # fill_missing_cells(dataset)
        # GUI.run(dataset=dataset)
        truncate_hdf5(dataset, dataset.parent / f"truncated_{dataset.name}", start_frame = 0, end_frame=300)
        
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
