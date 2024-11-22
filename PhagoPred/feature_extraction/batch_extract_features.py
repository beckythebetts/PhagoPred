import h5py
import numpy as np

from PhagoPred import SETTINGS
from PhagoPred.utils import tools

def get_phagocytosis(file=SETTINGS.DATASET):
    with h5py.File(file, 'r+') as f:
        phago_group = f.create_group('Cells/Phagocytosis')
        epi = tools.get_masks(f, 'Epi')
        phase = tools.get_masks(f, 'Phase')
        overlap = (epi>0) & (phase>0)
        epi_idxs = epi[overlap]
        phase_idxs = phase[overlap]
        frames = np.nonzero(overlap)[0]
        events, counts = np.unique(np.stack((phase_idxs, epi_idxs, frames), axis=1), axis=0, return_counts=True)
        idx_pairs = events[:, :2]
        frames = events[:, 2]
        for idx_pair in np.unique(events[:, :2], axis=0):
            mask = np.all(idx_pairs==idx_pair, axis=1)
            idx_pair_frames = frames[mask]
            idx_pair_counts = counts[mask]
            all_frames = np.arange(min(idx_pair_frames), max(idx_pair_frames)+1)
            all_counts = np.full(len(all_frames), np.nan)
            all_counts[np.searchsorted(all_frames, idx_pair_frames)] = idx_pair_counts
            phago_ds = phago_group.create_dataset(f'{idx_pair[0]:04}_{idx_pair[1]:04}', shape=(len(all_frames), 2), maxshape=(None, None), dtype=np.float32)
            phago_ds.attrs['columns'] = ['frame', 'number_of_pathogen_pixels']
            phago_ds[:, 0] = all_frames
            phago_ds[:, 1] = all_counts



def main():
    get_phagocytosis()

if __name__ == '__main__':
    main()