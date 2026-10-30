from pathlib import Path
from collections import defaultdict

from .survival import plot_num_alive_cells, plot_event_time_distributions
from .fluorescence import plot_tot_fluor
from ._common import label_to_files_mapping, generate_files_mapping


def main():
    duration_path = Path(
        '~/thor_server').expanduser() / 'MacrophageData' / '16_10'
    moi_path = Path('~/thor_server').expanduser() / 'MacrophageData' / '02_10'
    mois_json = moi_path / 'mois.json'

    duration_dict = generate_files_mapping({
        0: 'A',
        5: 'B',
        3: 'C',
        1: 'D'
    }, duration_path)
    moi_dict = label_to_files_mapping(mois_json)

    plot_tot_fluor(duration_dict,
                   colours={0: 'k'},
                   label_prefix='Infection duration: ',
                   label_suffix=' hours')
    # plot_num_alive_cells(duration_dict)


if __name__ == '__main__':
    main()
