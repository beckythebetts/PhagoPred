"""Run DeathNet on whole cell tracks and decode a death frame per cell."""
from typing import Optional

import numpy as np
import torch

from PhagoPred.death_classifier.config import TrainConfig
from PhagoPred.death_classifier.dataset import CellWindowDataset
from PhagoPred.death_classifier.model import (DeathNet, death_log_probs,
                                              decode_death)


def p_dead_from_outputs(cfg: TrainConfig,
                        frame_logits,
                        survive_logit,
                        valid,
                        dead_before_logit=None) -> torch.Tensor:
    """Per-frame probability the cell is dead, for either loss."""
    if cfg.loss == 'frame':
        return torch.sigmoid(frame_logits.float())
    # Cumulative probability of having died by each frame, plus the
    # probability it was already dead when the sequence started
    probs = death_log_probs(frame_logits, survive_logit, valid,
                            dead_before_logit).exp()
    num_frames = frame_logits.shape[1]
    p_dead = torch.cumsum(probs[:, :num_frames], dim=1)
    if dead_before_logit is not None:
        p_dead = p_dead + probs[:, num_frames + 1:num_frames + 2]
    return p_dead


def decode_outcome(probs: np.ndarray,
                   p_dead_before: Optional[float] = None) -> int:
    """Death index from a one-death outcome distribution, T for survives.

    Decides whether the cell dies at all from the total death probability,
    then picks the most likely frame. A plain argmax over frames + survives
    compares survives against each single frame, so predicts survives
    whenever death probability is spread over a few frames, even if death is
    much more likely overall.

    Args:
        probs: (T + 1,) P(death at each frame), then P(survives).
        p_dead_before: P(already dead when the track starts), for models
            with a dead_before output.
    """
    num_frames = len(probs) - 1
    frame_probs = probs[:num_frames]
    if probs[num_frames] >= 0.5:
        return num_frames
    if p_dead_before is not None and p_dead_before >= frame_probs.sum():
        # Already dead when the track starts, more likely than dying at any
        # point during it
        return 0
    return int(np.argmax(frame_probs))


@torch.no_grad()
def predict_track(model: DeathNet, cfg: TrainConfig,
                  dataset: CellWindowDataset, key: str,
                  device: torch.device) -> dict:
    """Run on a cell's whole track and decode a single death frame."""
    num_frames = len(dataset.crops[key])
    sample = dataset.get_window(key, 0, num_frames)
    x = sample['x'].unsqueeze(0).to(device).float()
    valid = sample['valid'].unsqueeze(0).to(device)
    with torch.autocast(device.type, enabled=cfg.amp):
        frame_logits, survive_logit, bad_logit, _, dead_before_logit = model(
            x, valid)
    p_dead = p_dead_from_outputs(cfg, frame_logits, survive_logit, valid,
                                 dead_before_logit)[0]

    if cfg.loss == 'frame':
        death_idx, _ = decode_death(p_dead)
    else:
        probs = death_log_probs(frame_logits, survive_logit, valid,
                                dead_before_logit)[0].exp().cpu().numpy()
        death_idx = decode_outcome(
            probs[:num_frames + 1],
            None if dead_before_logit is None else probs[num_frames + 1])

    first_frame = dataset.meta[key]['first_frame']
    return {
        'p_dead': p_dead.cpu().numpy(),
        'dead': sample['dead'].numpy(),
        'predicted':
        None if death_idx >= num_frames else first_frame + death_idx,
        'p_bad': float(torch.sigmoid(bad_logit.float())[0]),
    }


def _dihedral(x: torch.Tensor) -> torch.Tensor:
    """The 8 rotations / flips of (1, C, T, H, W) crops, as a batch of 8."""
    rotations = [torch.rot90(x, k, dims=(3, 4)) for k in range(4)]
    return torch.cat(rotations + [torch.flip(r, dims=(4, )) for r in rotations])


@torch.no_grad()
def outcome_probs(model: DeathNet,
                  cfg: TrainConfig,
                  dataset: CellWindowDataset,
                  key: str,
                  device: torch.device,
                  flips: bool = False) -> np.ndarray:
    """P(death at each frame of the track), then P(survives).

    With a dead_before output, P(dead before) is added to the first frame,
    which is where it is decoded to.

    Args:
        flips: average the outcome distribution over the 8 rotations / flips
            of the crops (as in training augmentation), for steadier
            predictions on borderline cells.

    Returns:
        (T + 1,) probabilities.
    """
    num_frames = len(dataset.crops[key])
    sample = dataset.get_window(key, 0, num_frames)
    x = sample['x'].unsqueeze(0).to(device).float()
    valid = sample['valid'].unsqueeze(0).to(device)
    if flips:
        x = _dihedral(x)
        valid = valid.expand(len(x), -1)
    with torch.autocast(device.type, enabled=cfg.amp):
        frame_logits, survive_logit, _, _, dead_before_logit = model(x, valid)
    probs = death_log_probs(frame_logits, survive_logit, valid,
                            dead_before_logit).exp().mean(0).cpu().numpy()
    if dead_before_logit is not None:
        probs[0] += probs[-1]
        probs = probs[:-1]
    return probs
