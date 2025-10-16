from pathlib import Path

from PhagoPred import SETTINGS
from PhagoPred.detectron_segmentation import segment, fine_tune_class, eval
from PhagoPred.tracking import trackpy
from PhagoPred.display import save, plots, napari_GUI
from PhagoPred.feature_extraction import extract_features
from PhagoPred.feature_extraction.morphology import fitting
from PhagoPred.prediction.decision_tree import model

from PhagoPred.survival_analysis.models import losses
from PhagoPred.survival_analysis import train, validate
if __name__ == '__main__':
    for dataset in (
        # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '03_10_2500.h5',
        # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '07_10_0.h5',
        Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '10_10_5000.h5',
    ):
        segment.seg_dataset(dataset=dataset)
        trackpy.run_tracking(dataset=dataset)
        extract_features.extract_features(dataset=dataset)
        
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
