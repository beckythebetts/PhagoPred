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
    """
    y_med = np.median(y_vals, axis=0)
    y_lower = np.percentile(y_vals, percentile_range[0], axis=0)
    y_upper = np.percentile(y_vals, percentile_range[1], axis=0)

    ax.plot(x_vals, y_med, color=color, label=label)
    ax.fill_between(x_vals,
                    y_lower,
                    y_upper,
                    color=color,
                    alpha=0.3,
                    linewidth=0)
