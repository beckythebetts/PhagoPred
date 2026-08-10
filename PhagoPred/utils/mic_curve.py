from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'serif'

vals = {
    0: [34, 38, 45],
    # 0.5: [41, 31, 35],
    # 1.0: [39, 33, 27],
    # 2.0: [29, 27, 38],
    5.0: [28, 44, 36],
    10.0: [28, 38, 39],
    20.0: [35, 27, 33],
    50.0: [3, 4, 1],
    100.0: [1, 0, 0],
    200.0: [1, 0, 0],
}


def get_cfu(val):
    return val * 100 * 100


def plot_mic_curve(vals: dict, save_as: Path):
    conc = list(vals.keys())
    counts = [get_cfu(np.array(c)) for c in vals.values()]
    means = [np.mean(c) for c in counts]
    stds = [np.std(c) for c in counts]
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.errorbar(
        conc,
        means,
        yerr=stds,
        marker='o',
        color='k',
        capsize=6,
    )
    ax.set_xlabel('Gentamicin concentration (ug/mL)')
    ax.set_ylabel('CFU / mL')
    ax.grid(True, which='both')
    ax.set_title('S. Aureus concentration after 1 hour Gentamicin exposure')
    plt.tight_layout()
    plt.savefig(save_as)


if __name__ == '__main__':
    plot_mic_curve(vals, Path('temp') / 'mic_curve.png')
