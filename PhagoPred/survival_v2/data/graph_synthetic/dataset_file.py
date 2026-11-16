from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import h5py
import numpy as np

_MAIN = 'Cells'
_FEATURES = f'{_MAIN}/Phase'
_CLEAN_FEATURES = f'{_MAIN}/Clean'

# Cells/Phase matches real data exactly: per-frame features
# (num_frames, num_cells) plus these per-cell values, (1, num_cells).
# First/Last Frame are the first and last observed frame (0-based, inclusive).
_PER_CELL = ('CellDeath', 'First Frame', 'Last Frame')
# Synthetic-only ground truth, (num_frames, num_cells), kept on Cells so
# Cells/Phase stays the same as real data.
_TRUTH = ('Hazard', 'HazardRates', 'PMFs', 'CIFs')


@dataclass
class SyntheticCells:
    """One split of a synthetic dataset (see the module docstring)."""
    features: dict[str, np.ndarray]
    clean_features: dict[str, np.ndarray]
    hazard: np.ndarray
    hazard_rates: np.ndarray
    pmfs: np.ndarray
    cifs: np.ndarray
    deaths: np.ndarray
    start_frames: np.ndarray
    end_frames: np.ndarray

    @property
    def num_cells(self) -> int:
        return len(self.deaths)

    def save(self, file_path: Path | str, scenario: dict) -> None:
        """Cells/Phase has exactly the layout and names of real data, so the
        same readers work. Synthetic-only extras: ground truth (Hazard,
        HazardRates, PMFs, CIFs) on Cells, clean features in Cells/Clean,
        and the scenario as a Cells attribute."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(file_path, 'w') as f:
            phase = f.create_group(_FEATURES)  # also creates Cells
            for name, arr in self.features.items():
                phase.create_dataset(name, data=arr, dtype=np.float32)
            for name, arr in (('CellDeath', self.deaths), ('First Frame',
                                                           self.start_frames),
                              ('Last Frame', self.end_frames)):
                phase.create_dataset(name,
                                     data=np.asarray(arr)[np.newaxis, :],
                                     dtype=np.float32)

            clean = f.create_group(_CLEAN_FEATURES)
            for name, arr in self.clean_features.items():
                clean.create_dataset(name, data=arr, dtype=np.float32)

            main = f[_MAIN]
            for name, arr in zip(
                    _TRUTH,
                (self.hazard, self.hazard_rates, self.pmfs, self.cifs)):
                main.create_dataset(name, data=arr, dtype=np.float32)
            main.attrs['num_cells'] = self.num_cells
            main.attrs['scenario'] = json.dumps(scenario)

    @classmethod
    def load(cls, file_path: Path | str) -> SyntheticCells:
        with h5py.File(file_path, 'r') as f:
            phase, clean, main = f[_FEATURES], f[_CLEAN_FEATURES], f[_MAIN]
            return cls(
                features={
                    name: phase[name][:]
                    for name in phase if name not in _PER_CELL
                },
                clean_features={name: clean[name][:]
                                for name in clean},
                hazard=main['Hazard'][:],
                hazard_rates=main['HazardRates'][:],
                pmfs=main['PMFs'][:],
                cifs=main['CIFs'][:],
                deaths=phase['CellDeath'][0],
                start_frames=phase['First Frame'][0].astype(int),
                end_frames=phase['Last Frame'][0].astype(int),
            )


def read_scenario_dict(file_path: Path | str) -> dict | None:
    """The saved scenario, or None if the file has none (not synthetic, or
    generated before scenarios were saved)."""
    with h5py.File(file_path, 'r') as f:
        if _MAIN not in f or 'scenario' not in f[_MAIN].attrs:
            return None
        return json.loads(f[_MAIN].attrs['scenario'])
