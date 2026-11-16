from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_med_range_on_ax(ax: plt.Axes,
                         x_vals: np.ndarray,
                         y_vals: list[np.ndarray],
                         color: Tuple[float, float, float, float],
                         percentile_range: Tuple[int, int],
                         label: str = None) -> None:
    """Plot median and given percentile ranges on axes.

    Curves of different lengths (e.g. loss histories from early-stopped runs)
    are NaN-padded, so later points summarise only the curves that reach them.
    """
    max_len = max(len(y) for y in y_vals)
    y_vals = np.stack([
        np.pad(np.asarray(y, dtype=float), (0, max_len - len(y)),
               constant_values=np.nan) for y in y_vals
    ])
    y_med = np.nanmedian(y_vals, axis=0)
    y_lower = np.nanpercentile(y_vals, percentile_range[0], axis=0)
    y_upper = np.nanpercentile(y_vals, percentile_range[1], axis=0)

    ax.plot(x_vals, y_med, color=color, label=label)
    ax.fill_between(x_vals,
                    y_lower,
                    y_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0)
