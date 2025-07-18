from PhagoPred import SETTINGS
from PhagoPred.detectron_segmentation import segment
from PhagoPred.tracking import track
from PhagoPred.display import save, plots, napari_GUI
from PhagoPred.feature_extraction import extract_features
from PhagoPred.feature_extraction.morphology import fitting

if __name__ == '__main__':
    # segment.main()
    # track.main()
    # save.main()
    # extract_features.main()
    # plots.main()
    # fitting.main()
    napari_GUI.main()
