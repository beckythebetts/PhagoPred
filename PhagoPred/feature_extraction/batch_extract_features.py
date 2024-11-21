import h5py
import numpy as np

from PhagoPred import SETTINGS
from PhagoPred.utils import tools

def get_phagocytosis(file=SETTINGS.DATASET):
    with h5py.File(file, 'r') as f:
        epi = tools.get_masks(f, 'Epi')
        phase = tools.get_masks(f, 'Phase')
    overlap = (epi>0) & (phase>0)
    epi_idxs = epi[overlap]
    phase_idxs = phase[overlap]
    frames = np.nonzero(overlap)[0]
    events = np.unique(np.stack((phase_idxs, epi_idxs, frames), axis=1), axis=0)
    print(events)

def main():
    get_phagocytosis()

if __name__ == '__main__':
    main()