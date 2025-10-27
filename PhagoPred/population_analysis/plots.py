from typing import Optional, Union
import sys

from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.extract_features import CellType
from PhagoPred.population_analysis import fitting

def plot_num_alive_cells(files: list = [SETTINGS.DATASET], save_as: Optional[Path]=None, first_frame: Union[int, list] = None, last_frame: Union[int, list] = None, time_step: int = 5, labels: list = []) -> 'matplotlib.figure.Figure':
    """PLot number of alive cells at each frame."""
    plt.rcParams["font.family"] = 'serif'
    cmap = plt.get_cmap('Set1')
    colours = [cmap(i) for i in range(len(files))]
    fig, (ax, ax_resid) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
    if isinstance(last_frame, int):
        last_frame = [last_frame]*len(files)
    if isinstance(first_frame, int):
        first_frame = [first_frame]*len(files)

    for i, (file, colour, label) in enumerate(zip(files, colours, labels)):
        with h5py.File(file, 'r') as f:
            file_first_frame = 0 if first_frame is None else first_frame[i]
            file_last_frame = f['Cells']['Phase']['Area'].shape[0] - 80 if last_frame is None else last_frame[i]
            
            first_frames = f['Cells']['Phase']['First Frame'][0] - file_first_frame
            last_frames = f['Cells']['Phase']['CellDeath'][0] - file_first_frame
            last_frames = np.where(np.isnan(last_frames), f['Cells']['Phase']['Last Frame'][0] - file_first_frame, last_frames)

            # frames = np.arange(file_first_frame, file_last_frame)
            frames = np.arange(0, file_last_frame - file_first_frame)
            alive_mask = (first_frames[np.newaxis, :] <= frames[:, np.newaxis]) & (last_frames[np.newaxis, :] > frames[:, np.newaxis])
            
            num_alive = alive_mask.sum(axis=1)
            # times = np.arange(file_first_frame, file_last_frame) * time_step 
            times = frames * time_step
            
            # N0, r, perr = fitting.fit_exponential(times, num_alive)
            N0, r, K, perr = fitting.fit_logistic(times, num_alive)
            # r, t0, K, perr = fitting.fit_gompertz(times, num_alive)
            # print(perr)
            
            ax.plot(times, num_alive, label=label, color=colour)
            # ax.plot(times, fitting.exponential_model(times, N0, r), label=f'{label} fit: N0={N0:.1f} ± {perr[0]:.1f}, r={r:.5f} ± {perr[1]:.5f}', color=colour, linestyle='--')
            
            fit_values = fitting.logistic_model(times, N0, r, K)           
            ax.plot(times, fit_values, label=f'{label} fit: N0={N0:.1f} ± {perr[0]:.1f}, r={r:.2e} ± {perr[1]:.1e} K={K:.0f} ± {perr[2]:.0f}', color=colour, linestyle='--')

            # fit_values = fitting.gompertz_model(times, r, t0, K)
            # ax.plot(times, fit_values, label=f'{label} fit:  r={r:.2e} ± {perr[0]:.1e}, t0={t0:.2f} ± {perr[1]:.2f} K={K:.0f} ± {perr[2]:.0f}', color=colour, linestyle='--')

            
            residuals = num_alive - fit_values
            ax_resid.plot(times, residuals, color=colour, label=label)

            # ax.plot(range(first_frame, last_frame), total_cells, label='Total Cells', color='black', linestyle='--')
        ax.set_xlabel('Time / minutes')
        ax.set_ylabel('Number of Detected Alive Cells')
        ax.legend()
        ax.grid(True)
        
        ax_resid.axhline(0, color='black', lw=0.8, linestyle='--')
        ax_resid.set_xlabel('Time / minutes')
        ax_resid.set_ylabel('Residuals')
        ax_resid.grid(True)
    if save_as is not None:
        plt.savefig(save_as)

    return fig

def main():
    files = [
        Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '07_10_0.h5',
        Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '21_10_2500.h5',
        # Path('PhagoPred')/'Datasets'/ 'Prelims' / '16_09_3.h5',
        Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '10_10_5000.h5',
        # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '10_10_5000_inner.h5',
        # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '10_10_5000_outer.h5',
    ]
    labels = ['0s exposure', 
              '2.5s exposure', 
              '5s exposure',
            #   'Outer radius',
              ]
    # km_plot(files,
    #         labels,
    #         Path('temp') / 'km_curve.png',
    #         [5, 5, 5],
    #         num_frames = 600)
    plot_num_alive_cells(files, Path('temp') / 'dead_cells.png', labels=labels, last_frame=[700, 820, 900])
    # plot_num_alive_cells(files, Path('temp') / 'dead_cells.png', labels=labels, last_frame=[700, 800, 750], first_frame=[0, 100, 50])
    
if __name__ == '__main__':
    main()