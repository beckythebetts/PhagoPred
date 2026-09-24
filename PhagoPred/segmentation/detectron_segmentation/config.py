"""Custom detectron2 config options for the segmentation models.

Kept separate from train.py so that anything loading a written-out config.yaml
(segment.py, eval.py, fine_tune_class.py) can register the same keys without
importing the training module.
"""
import sys

sys.path.insert(0, 'detectron2')

from detectron2.config import CfgNode as CN


def add_validation_config(cfg):
    """Register the custom cfg.VALIDATION options on a detectron2 config.

    ENABLED    -- compute validation loss during training at all.
    PERIOD     -- iterations between validation loss evaluations.
    SAVE_BEST  -- checkpoint the weights whenever validation loss improves.
    BEST_NAME  -- basename (no extension) of that checkpoint in OUTPUT_DIR.

    Must be called on any cfg before merging a config.yaml written by train(),
    otherwise yacs rejects the unknown VALIDATION key.
    """
    cfg.VALIDATION = CN()
    cfg.VALIDATION.ENABLED = False
    cfg.VALIDATION.PERIOD = 200
    cfg.VALIDATION.SAVE_BEST = True
    cfg.VALIDATION.BEST_NAME = "model_best"
    return cfg
