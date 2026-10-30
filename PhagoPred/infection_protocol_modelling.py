from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'serif'

WELL_AREA = 1.9  # cm^2
DOUBLING_TIME = 1  # hours
STATIONARY_DENSITY = 1e10
"""
Stokes law settling velocity
v = 2r^2(\rho_P - \rho_F)g / 9 \mu

r := partlicle radius ~1um
\rho_P := particle density ~1.1g/cm
\rho_F := flod density ~1g/cm^3
\mu := fluid viscosity ~0.7 mPa·s
"""
SETTLING_V = 0.28  # mm/hour
DT = 0.005  # time step /hours
MACROPHAGE_COUNT = 1e5

TOXIN_OFF = 0.01
TOXIN_ON = 50
AGR_THRESHOLD = 5e7  # CFU at which AGR swithc turns on
HILL_N = 3.0  # agr switch steepness

WASH_EFFICIENCY = 0.95

_COLOURS = {'suspended': 'r', 'deposited': 'b', 'toxin': 'g'}


@dataclass
class Scenario:
    MOI: int = 5
    infection_volume: int = 0.5  # mL
    infection_duration: int = 6  # hours
    wash_at: int = 6  # hours

    @property
    def time_steps(self) -> np.ndarray:
        return np.arange(0, (self.infection_duration), DT)

    def __str__(self) -> str:
        return f'MOI: {self.MOI}, infection_volume: {self.infection_volume}, infection-duration: {self.infection_duration}, wash_duration: {self.wash_at}'


@dataclass
class Results:
    scenario: Scenario
    suspended_bacteria: list[int] = field(default_factory=list)
    deposited_bacteria: list[int] = field(default_factory=list)
    toxin: list[int] = field(default_factory=list)

    @property
    def latest_step(self):
        return self.suspended_bacteria[-1], self.deposited_bacteria[
            -1], self.toxin[-1]

    def append_step(self, supsended: int, deposited: int, toxin: int):
        self.suspended_bacteria.append(supsended)
        self.deposited_bacteria.append(deposited)
        self.toxin.append(toxin)

    @property
    def toxin_exposure(self) -> list[float]:
        return np.cumsum(
            np.asarray(self.toxin) / self.scenario.infection_volume)


def _write_results(results: Results, save_as: Path) -> None:
    results = asdict(results)

    with open(save_as, 'w') as f:
        json.dump(results, f)


def _read_results(results: Results, file: Path) -> Results:
    with open(file, 'r') as f:
        results = json.load(f)

    results['scenario'] = Scenario(**results['scenario'])
    results = Results(**results)
    return results


def _plot_results(results: Results, ax: plt.Axes) -> plt.Axes:
    time_steps = results.scenario.time_steps
    ax.plot(time_steps,
            results.suspended_bacteria,
            color=_COLOURS['suspended'],
            label='Suspended Bacteria')
    ax.plot(time_steps,
            results.deposited_bacteria,
            color=_COLOURS['deposited'],
            label='Deposited Bacteria')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Bacteria (CFU)')

    toxin_ax = ax.twinx()
    toxin_ax.plot(time_steps,
                  results.toxin_exposure,
                  color=_COLOURS['toxin'],
                  label='Toxin exposure')
    toxin_ax.set_ylabel('Toxin exposure', color=_COLOURS['toxin'])

    ax.axvline(results.scenario.wash_at,
               linestyle='--',
               color='k',
               label='Media change')

    # ax.axhline(results.scenario.MOI * MACROPHAGE_COUNT,
    #            linestyle='--',
    #            color='k',
    #            label='Deposited MOI of 1')

    ax.legend()
    # toxin_ax.legend()

    ax.set_title(f'media change at: {results.scenario.wash_at} hours')
    # ax.set_title(f'MOI = {results.scenario.MOI}')
    # ax.set_title(f'Volume = {results.scenario.infection_volume} mL')
    return toxin_ax


# def _plot_final_results(results)


def make_plot(results: Results | list[Results], save_as: Path) -> None:
    if isinstance(results, Results):
        results = [results]
    fig, axs = plt.subplots(1,
                            len(results),
                            sharex=True,
                            sharey=True,
                            squeeze=False,
                            figsize=(4 * len(results), 4))
    toxin_axs = [
        _plot_results(result, ax) for result, ax in zip(results, axs[0])
    ]
    for toxin_ax in toxin_axs[1:]:
        toxin_ax.sharey(toxin_axs[0])
    # Limits were set before sharing, so rescale over every panel's data
    toxin_axs[0].autoscale(axis='y')

    # fig.suptitle('MOI = 5')
    fig.suptitle('MOI5, 0.5mL')
    fig.tight_layout()
    plt.savefig(save_as)
    plt.close(fig)


def simulate(scenario: Scenario) -> Results:
    well_depth = (scenario.infection_volume) / WELL_AREA * 10  #mm
    growth_rate = np.log(2) / DOUBLING_TIME  # /hour

    n_bacteria = scenario.MOI * MACROPHAGE_COUNT
    t = 0
    results = Results(scenario)

    cleared_depth = 0.0  # mm

    for t in scenario.time_steps:
        if t == 0.0:
            suspended = n_bacteria
            deposited = 0
            toxin = 0
            results.append_step(suspended, deposited, toxin)
            continue

        # Bacteria growth
        conc = (suspended + deposited) / scenario.infection_volume
        growth = growth_rate * (1 - conc / STATIONARY_DENSITY)
        suspended *= (1.0 + growth * DT)
        deposited *= (1.0 + growth * DT)

        # Depositing
        if cleared_depth < well_depth:
            settling_height = SETTLING_V * DT
            fraction = min(1.0, settling_height / (well_depth - cleared_depth))
            arrived = suspended * fraction

            suspended -= arrived
            deposited += arrived

            cleared_depth += settling_height

        # Toxin
        conc = (suspended + deposited) / scenario.infection_volume
        # hill = conc**HILL_N / (AGR_THRESHOLD**HILL_N + conc**HILL_N)

        # rate_per_cell = TOXIN_OFF + (TOXIN_ON - TOXIN_OFF) * hill
        rate_per_cell = 1
        toxin += rate_per_cell * (suspended + deposited) * DT

        # Wash
        if scenario.wash_at <= t < scenario.wash_at + DT:
            suspended *= (1.0 - WASH_EFFICIENCY)
            toxin *= (1.0 - WASH_EFFICIENCY)

        results.append_step(suspended, deposited, toxin)

    return results


def main():
    mois = [
        # 1,
        # 2,
        5,
        # 10,
    ]
    volumes = [
        # 0.1,
        # 0.25,
        0.5,
        # 1.0,
    ]
    infection_times = [
        # 1,
        # 2,
        # 4,
        6,
    ]
    wash_times = [
        # 0,
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        6,
    ]

    all_scenarios = [
        Scenario(moi, volume, infection_time, wash_time) for moi in mois
        for volume in volumes for infection_time in infection_times
        for wash_time in wash_times
    ]

    out_dir = Path('temp') / 'infection_protocol'
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, scenario in tqdm(enumerate(all_scenarios),
                            total=len(all_scenarios)):
        print(f'Running scenario {str(scenario)}')
        results = simulate(scenario)
        _write_results(results, out_dir / f'results_{i}.json')
        all_results.append(results)

    make_plot(all_results, out_dir / 'results.png')


if __name__ == '__main__':
    main()
