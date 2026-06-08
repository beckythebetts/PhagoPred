from pathlib import Path
import numpy as np


def load_model(model_dir: Path):
    from cellpose import models, core
    use_GPU = core.use_gpu()
    return models.CellposeModel(gpu=use_GPU, pretrained_model=str(model_dir / 'models' / 'model'))


def seg_image(im: np.ndarray, model=None, model_dir: Path = None) -> np.ndarray:
    """
    Run cellpose inference on a single image.

    Returns:
        Integer mask (H, W), 0 = background, 1..N = cell instances.
    """
    if model is None:
        model = load_model(model_dir)
    masks, _, _ = model.eval([im])
    return masks[0].astype(np.int16)
