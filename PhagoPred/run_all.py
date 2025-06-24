from PhagoPred import SETTINGS
from PhagoPred.detectron_segmentation import segment
from PhagoPred.tracking import tracker
from PhagoPred.display import save
# from PhagoPred.feature_extraction import batch_extract_features

if __name__ == '__main__':
    segment.main()
    tracker.main()
    save.main()
    # batch_extract_features.main()