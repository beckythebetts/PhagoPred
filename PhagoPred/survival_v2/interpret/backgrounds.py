from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..data.dataset import CellDataset


@dataclass
class BackgroundSet:
    """A pool of background cells: each row's full history up to its last
    valid landmark, plus its valid landmarks.

    For a sample with window length L, a row is usable at any landmark
    ``lm >= L`` whose window is fully observed; ``align='landmark'`` takes
    the L frames ending at ``lm`` (histories of cells alive at a landmark,
    like the sample), ``align='start'`` takes frames [0, L) (for
    hazard_from_frame_0 scenarios)."""
    feature_names: list[str]
    vals: list[np.ndarray]  # [T_i, F], raw units
    landmarks: list[np.ndarray]  # valid landmarks of each row
    clean: bool
    align: Literal['start', 'landmark'] = 'landmark'

    @classmethod
    def from_dataset(cls,
                     dataset: CellDataset,
                     num_samples: int,
                     clean: bool,
                     seed: int = 0,
                     align: Literal['start', 'landmark'] = 'landmark'
                     ) -> BackgroundSet:
        """``num_samples`` distinct cells, weighted by number of landmarks
        (as the training sampler). The same seed gives the same cells for
        clean and noisy sets."""
        rng = np.random.default_rng(seed)
        weights = np.array(
            [len(lf) for lf in dataset.cell_metadata.landmark_frames], float)
        n = min(num_samples, int((weights > 0).sum()))
        idxs = rng.choice(len(dataset),
                          n,
                          replace=False,
                          p=weights / weights.sum())
        meta = dataset.cell_metadata
        vals, landmarks = [], []
        for idx in idxs:
            lms = dataset.valid_landmarks(idx)
            if align == 'start':
                # A [0, L) window only needs the cell alive at L, not a
                # known label there: valid landmarks also need the horizon
                # observed, which drops most cells for L near the end of the
                # recording (censoring, unrelated to the features).
                death = meta.death_frames[idx]
                last = (int(death) - 1
                        if not np.isnan(death) else int(meta.end_frames[idx]))
                lms = np.arange(lms.min(), max(last, lms.max()) + 1)
            vals.append(dataset.raw_features(idx, clean)[:lms.max()])
            landmarks.append(lms)
        return cls(dataset.feature_names, vals, landmarks, clean, align)

    def eligible(self, num_frames: int) -> np.ndarray:
        """Rows with at least one usable landmark for a window of
        ``num_frames``."""
        return np.array([
            i for i in range(len(self.vals))
            if len(self._usable_landmarks(i, num_frames))
        ], int)

    def rows(self, idxs: np.ndarray, num_frames: int,
             rng: np.random.Generator) -> np.ndarray:
        """[len(idxs), T, F]: each row's window at a random usable landmark."""
        out = []
        for i in idxs:
            lm = int(rng.choice(self._usable_landmarks(i, num_frames)))
            start = lm - num_frames if self.align == 'landmark' else 0
            out.append(self.vals[i][start:start + num_frames])
        return np.stack(out)

    def _usable_landmarks(self, i: int, num_frames: int) -> np.ndarray:
        lms = self.landmarks[i]
        lms = lms[lms >= num_frames]
        if not len(lms):
            return lms
        # bad[t] = number of frames with a NaN in [0, t)
        bad = np.concatenate(
            [[0], np.cumsum(np.isnan(self.vals[i]).any(axis=1))])
        start = lms - num_frames if self.align == 'landmark' else 0
        return lms[bad[lms] - bad[start] == 0]
