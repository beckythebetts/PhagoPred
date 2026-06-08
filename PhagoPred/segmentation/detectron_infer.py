from pathlib import Path
import numpy as np


def load_model(model_dir: Path) -> tuple:
    import sys
    sys.path.insert(0, 'detectron2')
    from PhagoPred.detectron_segmentation.segment import get_model, get_predictor
    train_metadata, cfg = get_model(model_dir)
    cfg, predictor = get_predictor(cfg)
    return train_metadata, cfg, predictor


def seg_image(im: np.ndarray, model_dir: Path, model: tuple = None) -> dict | None:
    """
    Run detectron2 inference on a single image (H, W, 3).

    Returns:
        Dict of {category_name: integer mask} or None if no instances found.
    """
    import sys
    sys.path.insert(0, 'detectron2')
    from PhagoPred.detectron_segmentation.segment import seg_image as _seg_image
    if model is None:
        model = load_model(model_dir)
    train_metadata, cfg, predictor = model
    return _seg_image(model_dir, im, train_metadata, cfg, predictor)
