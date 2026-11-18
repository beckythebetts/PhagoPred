from pathlib import Path
from collections import defaultdict

from .survival import plot_num_alive_cells, plot_event_time_distributions, plot_censoring_vs_death
from .fluorescence import plot_tot_fluor, plot_fluor_hist
from .fluor_survival import plot_hazard_vs_burden, plot_hazard_by_infection
from .counts import plot_manual_counts_by_day
from ._common import label_to_files_mapping, generate_files_mapping

# BASEDIR = Path('~/thor_server').expanduser()
BASEDIR = Path(r'T:\\')


def main():
    duration_path = BASEDIR / 'MacrophageData' / '16_10'
    # duration_path = Path('T:')
    moi_path = BASEDIR / 'MacrophageData' / '02_10'
    control_path = BASEDIR / 'MacrophageData' / '06_11'
    mois_json = moi_path / 'mois.json'

    duration_dict = generate_files_mapping({
        0: 'A',
        5: 'B',
        3: 'C',
        1: 'D'
    }, duration_path)
    well_position_dict = generate_files_mapping(
        {
            0: ['C3', 'C4'],
            1: ['C2', 'C5'],
            2: ['C1', 'C6'],
        }, duration_path)
    moi_dict = label_to_files_mapping(mois_json)

    control_dict = generate_files_mapping({'Control': ['4', '5', '6']},
                                          control_path)
    # plot_hazard_by_infection(duration_dict,
    #                          label_prefix='Infection duration: ',
    #                          infected_thresh=500
    #                          # label_suffix=' hours',
    #                          )
    # plot_tot_fluor(
    #     duration_dict,
    #     colours={0: 'k'},
    #     # label_prefix='Infection duration: ',
    #     # label_suffix=' hours',
    # )
    # plot_manual_counts_by_day(
    #     {
    #         'Day 0': BASEDIR / 'MacrophageData' / 'control_test',
    #         'Day 3': BASEDIR / 'MacrophageData' / 'control_test_day3'
    #     },
    #     {
    #         'Control': ['A4', 'A5', 'A6', 'B4', 'B5', 'B6'],
    #         '+ Lysosostaphin': ['A1', 'A2', 'A3', 'B1', 'B2', 'B3'],
    #         '+ Gentamicin': ['C4', 'C5', 'C6', 'D4', 'D5', 'D6'],
    #         '+ Gentamicin \n+ Lysostaphin':
    #         ['C1', 'C2', 'C3', 'D1', 'D2', 'D3'],
    #     },  # text in file names
    # )
    # plot_fluor_hist(duration_dict)
    plot_num_alive_cells(
        control_dict,
        # every=40,
        # bar=True,
        # death_rate=True,
        # label_prefix='Distance from centre ',
        label_prefix='Infection duration: ',
        label_suffix=' hours',
        # colours={0: 'k'},
    )
    # plot_censoring_vs_death(duration_dict, num_bins=12)


if __name__ == '__main__':
    main()
