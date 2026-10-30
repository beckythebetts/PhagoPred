"""3D CNN + bidirectional GRU for detecting cell death from crop sequences.

The CNN looks at short windows of frames (temporal receptive field ~9 frames)
to pick up local changes such as loss of membrane movement, keeping the full
temporal resolution so there is one feature vector per frame. The GRU then
adds long range context along the whole track.

Two outputs:
    - frame_logits (B, T): per-frame logit that the cell is dead (or, for the
      one-death loss, unnormalised log probability that it dies at frame t).
    - survive_logit (B,): logit that the cell doesn't die within the
      sequence, for the one-death loss.
    - bad_logit (B,): logit that the sequence is a bad track (merge, ID
      switch, debris...), to filter these out of further analysis.
"""
from pathlib import Path
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from PhagoPred.death_classifier.config import TrainConfig, num_input_channels


class FrameGroupNorm(nn.GroupNorm):
    """GroupNorm with statistics computed separately for each frame.

    Plain GroupNorm on (B, C, T, H, W) pools statistics over all frames, so
    padding frames and sequence length change how real frames are
    normalised (a 64 frame crop padded to 287 frames is mostly zeros), which
    differs between training and inference on whole tracks. Same parameters
    as nn.GroupNorm, so checkpoints are interchangeable.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = super().forward(x.transpose(1, 2).reshape(b * t, c, h, w))
        return y.reshape(b, t, c, h, w).transpose(1, 2)


def _norm(channels: int, frame_norm: bool) -> nn.GroupNorm:
    # GroupNorm rather than BatchNorm as batches are small (~4)
    norm = FrameGroupNorm if frame_norm else nn.GroupNorm
    return norm(min(8, channels), channels)


class Conv2Plus1D(nn.Module):
    """Spatial 3x3 conv then temporal 3 conv, cheaper than a full 3x3x3.

    Downsamples spatially by stride, never temporally, so the output has one
    time step per input frame.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 2,
                 frame_norm: bool = False):
        super().__init__()
        self.spatial = nn.Conv3d(in_channels,
                                 out_channels, (1, 3, 3),
                                 stride=(1, stride, stride),
                                 padding=(0, 1, 1),
                                 bias=False)
        self.norm1 = _norm(out_channels, frame_norm)
        self.temporal = nn.Conv3d(out_channels,
                                  out_channels, (3, 1, 1),
                                  padding=(1, 0, 0),
                                  bias=False)
        self.norm2 = _norm(out_channels, frame_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.spatial(x)))
        return F.relu(self.norm2(self.temporal(x)))


class DeathNet(nn.Module):

    def __init__(self,
                 in_channels: int = 2,
                 widths: tuple[int, ...] = (16, 32, 64, 128),
                 hidden: int = 64,
                 bidirectional: bool = True,
                 dropout: float = 0.2,
                 bad_pool_r: float = 5.0,
                 frame_norm: bool = False,
                 dead_before: bool = False):
        """
        Args:
            in_channels: input channels (e.g. 2 for phase + mask).
            widths: channels of each Conv2Plus1D block, each halves the
                spatial size.
            hidden: GRU hidden size.
            bidirectional: let the GRU see future frames, appropriate for
                detecting death after the fact rather than predicting it.
            dropout: dropout on per-frame CNN features (GRU input) and on
                the GRU output before the heads.
            bad_pool_r: sharpness of the smooth max pooling per-frame bad
                track logits into one per track (see lse_pool).
            frame_norm: normalise each frame separately (FrameGroupNorm),
                so padding and sequence length don't affect features.
                False reproduces models trained before this option.
            dead_before: add an output for "already dead when the sequence
                starts", as an outcome separate from dying at frame 0 (see
                death_log_probs).
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels,
                      widths[0], (1, 5, 5),
                      stride=(1, 2, 2),
                      padding=(0, 2, 2),
                      bias=False),
            _norm(widths[0], frame_norm),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            Conv2Plus1D(c_in, c_out, frame_norm=frame_norm)
            for c_in, c_out in zip(widths[:-1], widths[1:])
        ])
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(widths[-1],
                          hidden,
                          batch_first=True,
                          bidirectional=bidirectional)
        gru_out = hidden * (2 if bidirectional else 1)
        self.frame_head = nn.Linear(gru_out, 1)
        self.survive_head = nn.Linear(gru_out, 1)
        # Per-frame, then pooled with a smooth max, as a track is bad if any
        # part of it is (a merge or ID switch may only affect a few frames)
        self.bad_head = nn.Linear(gru_out, 1)
        self.bad_pool_r = bad_pool_r
        # Only created when used, so older checkpoints still load
        self.dead_before_head = nn.Linear(gru_out, 1) if dead_before else None

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) crops.
            valid: (B, T) bool, False for padding at the end of sequences.

        Returns:
            frame_logits (B, T), survive_logit (B,), bad_logit (B,) and
            bad_frame_logits (B, T), the per-frame bad track logits pooled
            into bad_logit, which show where a track goes wrong, and
            dead_before_logit (B,), or None if the model has no dead_before
            output.
        """
        # Re-zero padding frames after each layer: temporal convs leak real
        # frames into the first padding frame, which per-frame normalisation
        # then scales up and the next temporal conv feeds back into the last
        # real frames. Zeroed, padding acts like a track's end at inference
        pad_mask = valid[:, None, :, None, None].to(x.dtype)
        features = self.stem(x) * pad_mask
        for block in self.blocks:
            features = block(features) * pad_mask  # (B, D, T, h, w)
        features = features.mean(dim=(3, 4)).transpose(1, 2)  # (B, T, D)
        features = self.dropout(features)

        # Pack so padding doesn't feed into the backward GRU direction
        lengths = valid.sum(dim=1).cpu()
        packed = pack_padded_sequence(features.float(),
                                      lengths,
                                      batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out,
                                     batch_first=True,
                                     total_length=valid.shape[1])

        # Same module as on the GRU input; no parameters, so old checkpoints
        # load and are unchanged at eval
        out = self.dropout(out)
        frame_logits = self.frame_head(out).squeeze(-1)
        pooled = (out * valid.unsqueeze(-1)).sum(1) / lengths.to(
            out.device).unsqueeze(-1)
        survive_logit = self.survive_head(pooled).squeeze(-1)
        bad_frame_logits = self.bad_head(out).squeeze(-1)
        bad_logit = lse_pool(bad_frame_logits, valid, self.bad_pool_r)
        dead_before_logit = (None if self.dead_before_head is None else
                             self.dead_before_head(pooled).squeeze(-1))
        return (frame_logits, survive_logit, bad_logit, bad_frame_logits,
                dead_before_logit)


def lse_pool(logits: torch.Tensor, valid: torch.Tensor,
             r: float) -> torch.Tensor:
    """Smooth max over valid frames: (1/r) log(mean(exp(r * logits))).

    Tends to the mean as r -> 0 and the max as r -> inf. Unlike a plain
    mean, a short bad segment isn't averaged away over a long track (e.g. 10
    bad frames lose ~0.3 logits going from a 64 frame window to a 288 frame
    track at r=5, vs ~0.7 for the mean), and taking the mean inside the log
    rather than a plain logsumexp stops long clean tracks scoring higher
    just for being long.

    Args:
        logits: (B, T) per-frame logits.
        valid: (B, T) bool.
        r: sharpness.

    Returns:
        (B,) pooled logits.
    """
    logits = logits.float().masked_fill(~valid, float('-inf'))
    num_valid = valid.sum(dim=1).float()
    return (torch.logsumexp(r * logits, dim=1) - torch.log(num_valid)) / r


def frame_loss(frame_logits: torch.Tensor, dead: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
    """Per-frame alive / dead binary cross entropy over valid frames."""
    loss = F.binary_cross_entropy_with_logits(frame_logits,
                                              dead,
                                              reduction='none')
    return (loss * valid).sum() / valid.sum()


def death_log_probs(
        frame_logits: torch.Tensor,
        survive_logit: torch.Tensor,
        valid: torch.Tensor,
        dead_before_logit: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Log P(death at frame t) for each frame, then log P(survives), then
    log P(already dead when the sequence starts) if dead_before_logit.

    Softmax over all the options, so exactly one outcome per sequence. P(dead
    by frame t) is P(dead before) plus the cumulative sum over frames, so is
    monotonic by construction.

    Returns:
        (B, T + 1) log probabilities, or (B, T + 2) with dead_before_logit.
    """
    logits = frame_logits.float().masked_fill(~valid, float('-inf'))
    extra = [survive_logit.float().unsqueeze(1)]
    if dead_before_logit is not None:
        extra.append(dead_before_logit.float().unsqueeze(1))
    return F.log_softmax(torch.cat([logits, *extra], dim=1), dim=1)


def one_death_loss(frame_logits: torch.Tensor,
                   survive_logit: torch.Tensor,
                   death_idx: torch.Tensor,
                   valid: torch.Tensor,
                   sigma: float = 0.0,
                   dead_before_logit: Optional[torch.Tensor] = None,
                   dead_before: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Negative log likelihood of the labelled outcome.

    Args:
        frame_logits, survive_logit: model outputs.
        death_idx: (B,) frame index the cell is dead from, T for no death.
        valid: (B, T) bool.
        sigma: if > 0, spread frame targets over neighbouring frames with a
            Gaussian of this width (frames), as manual labels are only
            accurate to a few frames.
        dead_before_logit: if given, sequences with dead_before True target
            the "already dead when the sequence starts" outcome instead of
            death at frame 0.
        dead_before: (B,) bool, needed with dead_before_logit.
    """
    log_probs = death_log_probs(frame_logits, survive_logit, valid,
                                dead_before_logit)
    num_frames = valid.shape[1]
    if dead_before_logit is not None:
        target_idx = torch.where(dead_before, num_frames + 1, death_idx)
    else:
        target_idx = death_idx
    if sigma <= 0:
        return F.nll_loss(log_probs, target_idx)

    frames = torch.arange(num_frames, device=valid.device)
    target = torch.exp(-0.5 * ((frames[None] - death_idx[:, None]) / sigma)**2)
    target = target * valid
    survives = death_idx == num_frames
    target[survives] = 0
    extra = [survives.float().unsqueeze(1)]
    if dead_before_logit is not None:
        target[dead_before] = 0
        extra.append(dead_before.float().unsqueeze(1))
    target = torch.cat([target, *extra], dim=1)
    target = target / target.sum(dim=1, keepdim=True)
    # Masked frames have -inf log prob and 0 target, avoid 0 * -inf = nan
    log_probs = log_probs.masked_fill(target == 0, 0)
    return -(target * log_probs).sum(dim=1).mean()


def decode_death(p_dead: torch.Tensor) -> tuple[int, float]:
    """Single death frame best explaining per-frame dead probabilities.

    Chooses the frame d (or no death) maximising
    sum_{t<d} log(1 - p_t) + sum_{t>=d} log(p_t), enforcing that a cell
    dies at most once.

    Args:
        p_dead: (T,) per-frame probability dead.

    Returns:
        Death frame index (T for no death) and its log likelihood.
    """
    p = p_dead.double().clamp(1e-6, 1 - 1e-6)
    log_alive, log_dead = torch.log1p(-p), torch.log(p)
    # Likelihood of dying at d for d = 0..T, where d = T is no death
    zero = torch.zeros(1, dtype=p.dtype, device=p.device)
    alive_before = torch.cat([zero, torch.cumsum(log_alive, 0)])
    dead_after = torch.cat([torch.flip(torch.cumsum(torch.flip(log_dead, [0]), 0), [0]), zero])
    scores = alive_before + dead_after
    best = int(torch.argmax(scores))
    return best, float(scores[best])


def load_model(checkpoint_path: Path,
               device: torch.device) -> tuple[DeathNet, TrainConfig]:
    """Model and its training config from a checkpoint, in eval mode."""
    state = torch.load(checkpoint_path, map_location=device)
    cfg = TrainConfig.from_dict(state['config'])
    model = DeathNet(in_channels=num_input_channels(cfg),
                     widths=cfg.widths,
                     hidden=cfg.hidden,
                     dropout=cfg.dropout,
                     bad_pool_r=cfg.bad_pool_r,
                     frame_norm=cfg.frame_norm,
                     dead_before=cfg.dead_before).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, cfg


# Config fields which set the model's input, so must match across an
# ensemble for its models to share one set of crops and one dataset
INPUT_FIELDS = ('channels', 'downsample', 'background_dim',
                'frame_diff') + TrainConfig._FROM_CACHE


def load_ensemble(
        run_dir: Path,
        checkpoint: str = 'last.pt',
        device: torch.device = torch.device('cpu')
) -> list[tuple[DeathNet, TrainConfig]]:
    """Every fold's model from a cross validation run."""
    paths = sorted(run_dir.glob(f'fold_*/{checkpoint}'))
    if not paths:
        raise FileNotFoundError(f'No fold_*/{checkpoint} in {run_dir}')
    ensemble = [load_model(path, device) for path in paths]
    for _, cfg in ensemble:
        for field in INPUT_FIELDS:
            if getattr(cfg, field) != getattr(ensemble[0][1], field):
                raise ValueError(f'Folds differ in {field}')
    return ensemble
