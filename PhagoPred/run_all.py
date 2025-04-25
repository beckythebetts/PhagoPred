from PhagoPred import SETTINGS
from PhagoPred.detectron_segmentation import segment
from PhagoPred.tracking import track_all
from PhagoPred.feature_extraction import batch_extract_features

if __name__ == '__main__':
    segment.main()
    track_all.main()
    batch_extract_features.main()