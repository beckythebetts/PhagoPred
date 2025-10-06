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
    segment.main()
    trackpy.main()
    # save.main()
    extract_features.main()
    # plots.main()
    # fitting.main()
    # napari_GUI.main()
    # model.main()
    # fine_tune_class.main()
    # eval.main()
    # losses.main()
    # train.main()
    # validate.main()
