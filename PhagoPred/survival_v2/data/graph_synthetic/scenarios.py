from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
import json

import numpy as np

from PhagoPred.utils.logger import get_logger
from .noise_funcs import Noise
from .hazard import HazardFunc, calibrate_hazard
from .graph import Feature, CausalGraph
from .rules import Rule
from .rule_sets import RULE_SETS
from .dataset_file import read_scenario_dict

log = get_logger()


@dataclass
class Missingness:
    """Parameters for missing time steps.

    ``max_*_frac`` are fractions of the window: a late-entering cell starts in
    [0, max_late_entry_frac * num_frames), an early-exiting one ends in the
    last max_early_exit_frac of it."""
    prob_missing: float = 0.0  # per-frame probability a frame is masked
    late_entry_prob: float = 0.0
    max_late_entry_frac: float = 0.49
    early_exit_prob: float = 0.0
    max_early_exit_frac: float = 0.49

    @classmethod
    def none(cls) -> Missingness:
        return cls()

    @classmethod
    def low(cls) -> Missingness:
        return cls(late_entry_prob=0.2, early_exit_prob=0.2)

    @property
    def label(self) -> str:
        for preset in ('none', 'low'):
            if self == getattr(Missingness, preset)():
                return preset
        return 'custom'

    def sample_observed_ranges(
            self, num_frames: int,
            num_cells: int) -> tuple[np.ndarray, np.ndarray]:
        """First and last observed frame (inclusive) of each cell.

        Without late entry, cells start in the first 10 frames; late entrants
        start uniformly in [0, max_late_entry_frac * num_frames). Likewise for
        the end of the window and early exit."""
        if self.late_entry_prob > 0.0:
            start = np.zeros(num_cells, dtype=int)
            late = np.random.rand(num_cells) < self.late_entry_prob
            start[late] = np.random.randint(
                0,
                max(1, int(self.max_late_entry_frac * num_frames)),
                size=int(late.sum()))
        else:
            start = np.random.randint(0, 10, size=num_cells)

        if self.early_exit_prob > 0.0:
            end = np.full(num_cells, num_frames - 1, dtype=int)
            early = np.random.rand(num_cells) < self.early_exit_prob
            end[early] = np.random.randint(
                int(num_frames - self.max_early_exit_frac * num_frames),
                num_frames,
                size=int(early.sum()))
        else:
            end = np.random.randint(num_frames - 10,
                                    num_frames,
                                    size=num_cells)
        return start, end

    def sample_frame_mask(self, num_frames: int, num_cells: int) -> np.ndarray:
        """(num_frames, num_cells) True where a frame is randomly missing."""
        return np.random.rand(num_frames, num_cells) < self.prob_missing


@dataclass
class ScenarioCfg:
    """Everything needed to generate a synthetic dataset, as plain values.

    Saved as JSON in each generated file (see ``to_dict`` / ``load_scenario``),
    so a dataset can always be rebuilt with the exact settings that made it.
    The calibrated hazard function is generated, not configured: it is fitted
    by ``calibrate_hazard`` and saved separately with the dataset."""
    filename: str
    rules: list[Rule]

    pre_noise: Noise = field(default_factory=lambda: Noise.gaussian(0.5))
    post_noise: Noise = field(default_factory=Noise.none)
    hazard_pre_noise: Noise = field(default_factory=Noise.none)
    missingness: Missingness = field(default_factory=Missingness)

    target_death_fraction: float = 0.5
    num_files: int = 10
    cells_per_file: int = 500
    num_frames: int = 300
    burn_in: int = 0
    death_burn_in: int = 0
    hazard_spread: float | None = 3.0
    hazard_spread_start: int = 0
    hazard_from_frame_0: bool = False

    hazard_func: HazardFunc | None = field(default=None, compare=False)

    def __post_init__(self):
        self.graph = CausalGraph(self._features,
                                 self.rules,
                                 burn_in=self.burn_in,
                                 death_burn_in=self.death_burn_in,
                                 hazard_from_frame_0=self.hazard_from_frame_0)

    @property
    def _features(self) -> list[Feature]:
        names = sorted({n for r in self.rules for n in (r.target, *r.inputs)})
        return [
            Feature(n, pre_noise=self.hazard_pre_noise) if n == 'Hazard' else
            Feature(n, pre_noise=self.pre_noise, post_noise=self.post_noise)
            for n in names
        ]

    def to_dict(self) -> dict:
        """Plain-value dict of the scenario, including the fitted hazard
        function if it has been calibrated."""
        d = {
            f.name: asdict(v) if is_dataclass(v) else v
            for f in fields(self) if f.name != 'rules'
            for v in [getattr(self, f.name)]
        }

        d['rules'] = [r.to_dict() for r in self.rules]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ScenarioCfg:
        d = dict(d)
        d['rules'] = [Rule.from_dict(r) for r in d['rules']]
        for key in ('pre_noise', 'post_noise', 'hazard_pre_noise'):
            d[key] = Noise(**d[key])
        d['missingness'] = Missingness(**d['missingness'])
        if d.get('hazard_func') is not None:
            d['hazard_func'] = HazardFunc(**d['hazard_func'])
        return cls(**d)

    def calibrate_hazard(self, num_samples: int = 200) -> None:
        self.hazard_func = calibrate_hazard(
            self.graph,
            self.num_frames,
            np.zeros(num_samples),
            np.full(num_samples, self.num_frames),
            self.target_death_fraction,
            hazard_spread=self.hazard_spread,
            spread_start=self.hazard_spread_start,
        )


def load_scenario(h5_path: Path | str) -> ScenarioCfg | None:
    """The scenario that generated this dataset file, with its calibrated
    hazard function; None if the file is not a synthetic dataset."""
    h5_path = Path(h5_path)
    scenario = read_scenario_dict(h5_path)
    if scenario is not None:
        scenario = ScenarioCfg.from_dict(scenario)
    return scenario


# === PRESETS ===

BASE_CFGS: list[ScenarioCfg] = [
    ScenarioCfg(
        name,
        rules=rules_func(),
        pre_noise=Noise.gaussian(0.5),
        post_noise=Noise.gaussian(0.1),
        missingness=Missingness.none(),
        burn_in=1000,
        death_burn_in=200,
    ) for name, rules_func in RULE_SETS.items()
]

ALL_CFGS: list[ScenarioCfg] = BASE_CFGS
