from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_ma(coeffs: list[float], std: float, length: int,
            ax: plt.axes | None) -> None:
    vals = []
    coeffs = np.array([1] + coeffs)
    coeffs = coeffs[::-1]
    noise = np.random.normal(0, std, length + len(coeffs))
    for i in range(length):
        prev_noise = noise[i:i + len(coeffs)]
        vals.append(np.sum(coeffs * prev_noise))

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.arange(len(vals)), vals)


# def plot_ar(coeffs: list[float], std: float, length: int,
#             ax: plt.axes | None) -> None:
#     ar_coeffs = np.array(coeffs[::-1])
#     noise = np.random.normal(0, std, length + len(ar_coeffs))
#     vals = list(noise[:len(ar_coeffs)])

#     for i in range(length):
#         prev_vals = np.array(vals[i:i + len(ar_coeffs)])
#         vals.append(np.dot(ar_coeffs, prev_vals) + noise[i + len(ar_coeffs)])

#     ax.plot(np.arange(length), vals[-length:])


def plot_ar(coeffs: list[float], std: float, length: int,
            ax: plt.axes | None) -> None:
    ar_coeffs = np.array(coeffs[::-1] + [1])
    vals = np.random.normal(0, std, length + len(ar_coeffs))
    # vals = list(noise[:len(ar_coeffs)])

    for i in range(length):
        prev_vals = np.array(vals[i + 1:i + len(ar_coeffs) + 1])
        vals[i + len(ar_coeffs)] = np.dot(ar_coeffs, prev_vals)
        # vals.append()

    ax.plot(np.arange(length), vals[-length:])


def plot_grid(func: callable, args_1_name: str, args_1: list, args_2_name: str,
              args_2: list, **kwargs) -> plt.Figure:
    fig, axs = plt.subplots(len(args_1), len(args_2))
    for i, arg_1 in enumerate(args_1):
        for j, arg_2 in enumerate(args_2):
            all_args = {args_1_name: arg_1, args_2_name: arg_2} | kwargs
            func(ax=axs[i, j], **all_args)
            ax = axs[i, j]
            if i == 0:
                ax.set_title(f"{args_2_name}={arg_2}")
            if j == 0:
                ax.set_ylabel(f"{args_1_name}={arg_1}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    plot_grid(plot_ma,
              args_1_name='coeffs',
              args_1=[[0.95], [1], [1.05, 1.0, 1.0]],
              args_2_name='std',
              args_2=[0.1, 1, 5],
              length=200)
