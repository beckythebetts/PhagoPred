from __future__ import annotations
from pathlib import Path

import numpy as np

from .dataset_file import SyntheticCells
from .hazard import pmf_from_hazards, sample_death_frames
from .scenarios import ScenarioCfg, ALL_CFGS

_SAVE_DIRECTORY = Path('PhagoPred') / 'Datasets' / 'graph_synthetic'


def simulate_cells(scenario: ScenarioCfg) -> SyntheticCells:
    """Simulate one split: trajectories (with the graph's burn-in and
    survivor conditioning), observation noise, each cell's observed range and
    masked frames, and a death frame drawn from its hazard."""
    graph, hazard_func = scenario.graph, scenario.hazard_func
    if hazard_func is None:
        raise ValueError(
            'hazard_func is None, calibrate hazard before simulating cells')
    num_cells = scenario.cells_per_file
    num_frames = scenario.num_frames

    feature_names = [f.name for f in graph.features if f.name != 'Hazard']

    missingness = scenario.missingness
    start, end = missingness.sample_observed_ranges(num_frames, num_cells)
    frames = np.arange(num_frames)[:, None]  # (T, 1)
    outside = (frames < start) | (frames > end)  # (T, N)

    clean = graph.simulate(num_cells,
                           num_frames,
                           hazard_func=hazard_func,
                           post_noise=False)
    observed = graph.observe(clean)

    # Hazard is never observed with noise, so it comes from the clean run.
    hazard_rates = hazard_func(clean['Hazard']).astype(np.float32)
    pmfs = pmf_from_hazards(hazard_rates).astype(np.float32)
    cifs = np.cumsum(pmfs, axis=0).astype(np.float32)

    deaths = sample_death_frames(hazard_rates)
    deaths[(deaths < start) | (deaths > end)] = np.nan

    features, clean_features = {}, {}
    for name in feature_names:
        unobserved = outside | missingness.sample_frame_mask(
            num_frames, num_cells)
        features[name] = np.where(unobserved, np.nan,
                                  observed[name]).astype(np.float32)
        clean_features[name] = np.where(unobserved, np.nan,
                                        clean[name]).astype(np.float32)

    return SyntheticCells(
        features=features,
        clean_features=clean_features,
        hazard=np.where(outside, np.nan, clean['Hazard']).astype(np.float32),
        hazard_rates=hazard_rates,
        pmfs=pmfs,
        cifs=cifs,
        deaths=deaths.astype(np.float32),
        start_frames=start,
        end_frames=end,
    )


def generate_dataset(scenario: ScenarioCfg,
                     save_dir: Path | str,
                     seed: int | None = None) -> None:
    if scenario.hazard_func is None:
        scenario.calibrate_hazard()
    for i in range(scenario.num_files):
        if seed is not None:
            np.random.seed(seed + i)

        path = Path(
            save_dir
        ) / f'{scenario.filename}' / f'{scenario.filename}_{i:02d}.h5'
        print(f'Generating {str(path.name)}')
        simulate_cells(scenario).save(path, scenario.to_dict())


if __name__ == '__main__':

    for cfg in ALL_CFGS:
        generate_dataset(cfg, _SAVE_DIRECTORY)
    print('Done.')
