"""Checks of the automated cell counts against manual counts."""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model

from PhagoPred import SETTINGS
from PhagoPred.population_analysis import fitting


def plot_cell_count_errors(files: list[Path] = [SETTINGS.DATASET],
                           errors_txts: list[Path] = None,
                           save_as: Path = None) -> 'matplotlib.figure.Figure':

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
            last_frames = np.where(np.isnan(last_frames),
                                   f['Cells']['Phase']['Last Frame'][0],
                                   last_frames)

            alive_mask = (first_frames[np.newaxis, :]
                          <= frames[:,
                                    np.newaxis]) & (last_frames[np.newaxis, :]
                                                    > frames[:, np.newaxis])
            predicted_cell_count = alive_mask.sum(axis=1)
            all_predicted_cell_counts.append(predicted_cell_count)

    errors = np.concatenate(all_errors)
    predicted_cell_count = np.concatenate(all_predicted_cell_counts)

    # Fit linear regression
    # model = sklearn.linear_model.LinearRegression().fit(predicted_cell_count.reshape(-1, 1), predicted_cell_count + errors)
    # a = model.coef_[0]
    # b = model.intercept_
    a, b, perr = fitting.fit_power_law(predicted_cell_count,
                                       predicted_cell_count + errors, None)

    # predicted_manual = model.predict(predicted_cell_count.reshape(-1, 1))
    predicted_manual = fitting.power_law_model(predicted_cell_count, a, b)
    residuals = (predicted_cell_count + errors) - predicted_manual
    std_residuals = np.std(residuals)

    print(f'Standard deviation of residuals: {std_residuals:.3f}')

    # Create figure with two subplots vertically
    fig, (ax1, ax2,
          ax3) = plt.subplots(3,
                              1,
                              figsize=(12, 10),
                              gridspec_kw={'height_ratios': [3, 1, 1]},
                              sharex=True)

    label = f'Power law fit: y = {a:.2f} • x^{b:.2f}\nStd residuals: {std_residuals:.2f}'
    # Top plot: manual vs automated counts + fit line
    ax1.scatter(predicted_cell_count, predicted_cell_count + errors, color='k')
    ax1.plot(predicted_cell_count,
             predicted_manual,
             color='r',
             linestyle='-',
             label=label)
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

    std_model = sklearn.linear_model.LinearRegression().fit(
        predicted_cell_count.reshape(-1, 1), np.abs(residuals))
    a = std_model.coef_[0]
    b = std_model.intercept_

    label = f'Std linear fit: y = {a/ np.sqrt(2/np.pi):.2f} x + {b / np.sqrt(2/np.pi):.2f}'
    ax3.scatter(predicted_cell_count, np.abs(residuals), color='k')
    ax3.plot(predicted_cell_count,
             a * predicted_cell_count + b,
             color='r',
             linestyle='-',
             label=label)
    ax3.set_xlabel('Automated Cell Count')
    ax3.set_ylabel('Abs(Residuals)')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Absolute Residuals with Linear Fit')

    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as)

    return fig
