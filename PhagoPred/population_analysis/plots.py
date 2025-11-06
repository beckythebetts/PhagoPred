from typing import Optional, Union
import sys

from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
import sklearn

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.extract_features import CellType
from PhagoPred.population_analysis import fitting

def plot_num_alive_cells(files: list = [SETTINGS.DATASET], save_as: Optional[Path]=None, first_frame: Union[int, list] = None, last_frame: Union[int, list] = None, time_step: int = 5, labels: list = [], true_count_fit_vals=[0.61, 1.11], std_fit_vals=[0.1, -5.59]) -> 'matplotlib.figure.Figure':
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
    
    r_values = []
    r_errors = []

    for i, (file, colour, label) in enumerate(zip(files, colours, labels)):
        with h5py.File(file, 'r') as f:
            file_first_frame = 0 if first_frame is None else first_frame[i]
            file_last_frame = f['Cells']['Phase']['Area'].shape[0]-30 if last_frame is None else last_frame[i]
            
            first_frames = f['Cells']['Phase']['First Frame'][0] - file_first_frame
            last_frames = f['Cells']['Phase']['CellDeath'][0] - file_first_frame
            last_frames = np.where(np.isnan(last_frames), f['Cells']['Phase']['Last Frame'][0] - file_first_frame, last_frames)

            # frames = np.arange(file_first_frame, file_last_frame)
            frames = np.arange(0, file_last_frame - file_first_frame)
            alive_mask = (first_frames[np.newaxis, :] <= frames[:, np.newaxis]) & (last_frames[np.newaxis, :] > frames[:, np.newaxis])
            
            num_alive = alive_mask.sum(axis=1)
            num_alive = true_count_fit_vals[0]*num_alive**true_count_fit_vals[1]
            # times = np.arange(file_first_frame, file_last_frame) * time_step 
            times = frames * time_step
            stds = num_alive*std_fit_vals[0]+std_fit_vals[1]
            # a, b, perr = fitting.fit_linear(times, num_alive, stds)
            # N0, r, perr = fitting.fit_exponential(times, num_alive, stds)
            # N0, r, K, perr = fitting.fit_logistic(times, num_alive, stds)
            r, t0, K, perr = fitting.fit_gompertz(times, num_alive, stds)
            
            r_values.append(r)
            r_errors.append(perr[0])
            
            ax.plot(times, num_alive, label=label, color=colour)
            ax.fill_between(times, num_alive-stds, num_alive+stds, color=colour, alpha=0.3, edgecolor=None)
            
            # fit_values = fitting.linear_model(times, a, b)
            # ax.plot(times, fit_values, label=f'{label} fit\na={a:.3f} ± {perr[0]:.4f}\nb={b:.0f} ± {perr[1]:.0f}', color=colour, linestyle='--')
            
            # fit_values = fitting.exponential_model(times, N0, r)
            # ax.plot(times, fit_values, label=f'{label} fit\nN0={N0:.1f} ± {perr[0]:.1f}\nr={r:.2e} ± {perr[1]:.1e}', color=colour, linestyle='--')
            
            # fit_values = fitting.logistic_model(times, N0, r, K)           
            # ax.plot(times, fit_values, label=f'{label} fit\nN0={N0:.1f} ± {perr[0]:.1f}\nr={r:.2e} ± {perr[1]:.1e}\nK={K:.0f} ± {perr[2]:.0f}', color=colour, linestyle='--')

            fit_values = fitting.gompertz_model(times, r, t0, K)
            ax.plot(times, fit_values, label=f'{label} fit\nr={r:.2e} ± {perr[0]:.1e}\nt0={t0:.2f} ± {perr[1]:.2f}\nK={K:.0f} ± {perr[2]:.0f}', color=colour, linestyle='--')

            
            residuals = num_alive - fit_values
            rss = np.sum(residuals**2)
            ax_resid.plot(times, residuals, color=colour, label=f'{label} residuals\nRSS={rss:.2e}')
            ax_resid.fill_between(times, residuals - stds, residuals + stds, color=colour, alpha=0.3, edgecolor=None)

            # ax.plot(range(first_frame, last_frame), total_cells, label='Total Cells', color='black', linestyle='--')
        # ax.set_xlabel('Time / minutes')
        ax.set_ylabel('Number of Detected Alive Cells')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(True)
        
        ax_resid.axhline(0, color='black', lw=0.8, linestyle='--')
        ax_resid.set_xlabel('Time / minutes')
        ax_resid.set_ylabel('Residuals')
        ax_resid.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax_resid.grid(True)
        

    # ax_r.grid(True)
    
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    if save_as is not None:
        plt.savefig(save_as)
    
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(files)), r_values, yerr=r_errors, color=colours, alpha=1.0, capsize=5, edgecolor='none')
    plt.xticks(range(len(files)), labels)
    plt.ylabel('Gompertz r parameter')
    plt.title('Growth rate (r)')
    plt.tight_layout()
    plt.savefig(Path('temp') / 'growth_rates.png')

    return fig


def plot_km(files: list = [SETTINGS.DATASET], save_as: Optional[Path] = None, last_frame: int = None, time_step: int = 5, labels: list = []) -> 'matplotlib.figure.Figure':
    cmap = plt.get_cmap('Set1')  # Choose colormap and number of discrete colors

    plt.figure(figsize=(12, 8))

    for i, (file) in enumerate(files):
        with h5py.File(file, 'r') as f:
            initial_cells_mask = f['Cells']['Phase']['First Frame'][0] == 0
            death_frames = f['Cells']['Phase']['CellDeath'][0][initial_cells_mask]
            end_frames = f['Cells']['Phase']['Last Frame'][0][initial_cells_mask]
            
        durations = np.where(np.isnan(death_frames), end_frames, death_frames)
        event_observed = ~np.isnan(death_frames)
        
        # Clip to last_frame
        event_observed[durations > last_frame] = False
        durations = np.clip(durations, None, last_frame)
        
        durations = durations * time_step
        colour = cmap(i)
        kmf = KaplanMeierFitter().fit(durations=durations, event_observed=event_observed)
        ax = kmf.plot_survival_function(ci_show=True, color=colour, label=labels[i])


    plt.title("Kaplan-Meier Survival Estimates")
    plt.xlabel("Time / minutes")
    plt.ylabel("Survival Probability")
    plt.legend(loc='best', fontsize='small', ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)

def plot_cell_count_errors(
    files: list[Path] = [SETTINGS.DATASET], 
    errors_txts: list[Path] = None, 
    save_as: Path = None
) -> 'matplotlib.figure.Figure':

    all_errors = []
    all_predicted_cell_counts = []

    for errors_txt, file in zip(errors_txts, files):
        errors_txt = np.genfromtxt(errors_txt, int, delimiter=', ')
        frames = errors_txt[:, 0]
        errors = errors_txt[:, 1]
        all_errors.append(errors)

        with h5py.File(file, 'r') as f:
            first_frames = f['Cells']['Phase']['First Frame'][0]
            last_frames = f['Cells']['Phase']['CellDeath'][0]
            last_frames = np.where(np.isnan(last_frames), f['Cells']['Phase']['Last Frame'][0], last_frames)

            alive_mask = (first_frames[np.newaxis, :] <= frames[:, np.newaxis]) & (last_frames[np.newaxis, :] > frames[:, np.newaxis])
            predicted_cell_count = alive_mask.sum(axis=1)
            all_predicted_cell_counts.append(predicted_cell_count)

    errors = np.concatenate(all_errors)
    predicted_cell_count = np.concatenate(all_predicted_cell_counts)

    # Fit linear regression
    # model = sklearn.linear_model.LinearRegression().fit(predicted_cell_count.reshape(-1, 1), predicted_cell_count + errors)
    # a = model.coef_[0]
    # b = model.intercept_
    a, b, perr = fitting.fit_power_law(predicted_cell_count, predicted_cell_count+errors, None)
    

    # predicted_manual = model.predict(predicted_cell_count.reshape(-1, 1))
    predicted_manual = fitting.power_law_model(predicted_cell_count, a, b)
    residuals = (predicted_cell_count + errors) - predicted_manual
    std_residuals = np.std(residuals)

    print(f'Standard deviation of residuals: {std_residuals:.3f}')

    # Create figure with two subplots vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

    label = f'Power law fit: y = {a:.2f} • x^{b:.2f}\nStd residuals: {std_residuals:.2f}'
    # Top plot: manual vs automated counts + fit line
    ax1.scatter(predicted_cell_count, predicted_cell_count + errors, color='k')
    ax1.plot(predicted_cell_count, predicted_manual, color='r', linestyle='-', label=label)
    ax1.set_ylabel('Manual Cell Count')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Manual vs Automated Cell Counts with Power Law Fit')

    # Bottom plot: residuals
    ax2.scatter(predicted_cell_count, residuals, color='k')
    ax2.axhline(0, color='k', linestyle='--')
    # ax2.set_xlabel('Automated Cell Count')
    ax2.set_ylabel('Residuals\n(Manual - Automated)')
    ax2.grid(True)
    ax2.set_title('Residuals')
    
    std_model = sklearn.linear_model.LinearRegression().fit(predicted_cell_count.reshape(-1, 1), np.abs(residuals))
    a = std_model.coef_[0]
    b = std_model.intercept_
    
    label = f'Std linear fit: y = {a/ np.sqrt(2/np.pi):.2f} x + {b / np.sqrt(2/np.pi):.2f}'
    ax3.scatter(predicted_cell_count, np.abs(residuals), color='k')
    ax3.plot(predicted_cell_count, a*predicted_cell_count+b, color='r', linestyle='-', label=label)
    ax3.set_xlabel('Automated Cell Count')
    ax3.set_ylabel('Abs(Residuals)')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Absolute Residuals with Linear Fit')
    

    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as)

    return fig
    

def main():
    files = [
        Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '07_10_0.h5',
        Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '28_10_2500.h5',
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


    plot_num_alive_cells(files, Path('temp') / 'dead_cells.png', labels=labels, 
                         last_frame=[600, 600, 600]
                         )
    # plot_km(files, Path('temp') / 'km_plot.png', labels=labels, last_frame=750)
    # plot_cell_count_errors(
    #     files = [
    #         Path('PhagoPred') / 'Datasets' / 'ExposureTest' / 'old' / '03_10_2500.h5', 
    #         Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
    #         ],
    #     errors_txts=[
    #         Path('temp') / '03_10_errors.txt', 
    #         Path('temp') / '07_10_errors.txt',
    #         ], 
    #     save_as=Path('temp') / 'count_errors.png'
    #     )
    
if __name__ == '__main__':
    main()