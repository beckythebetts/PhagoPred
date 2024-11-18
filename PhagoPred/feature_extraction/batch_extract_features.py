import h5py
import numpy as np

from PhagoPred import SETTINGS
from PhagoPred.utils import tools

def get_phagocytosis(file=SETTINGS.DATASET):
    with h5py.File(file, 'r') as f:
        epi = tools.get_masks(f, 'Epi')
        phase = tools.get_masks(f, 'Phase')
    overlap = (epi>0) &

    phago_cells = np.unique(phase[overlap])
    print(phago_cells)

def main():
    get_phagocytosis()

if __name__ == '__main__':
    main()