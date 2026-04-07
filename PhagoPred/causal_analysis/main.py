from __future__ import annotations
from pathlib import Path
from copy import deepcopy

import tigramite.data_processing as data_processing
import tigramite.plotting as plotting
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.jpcmciplus import JPCMCIplus
# from tigramite.independence_tests.gpdc_torch import GPDCtorch
import numpy as np
import matplotlib.pyplot as plt
import h5py

from PhagoPred.survival_v2.data.graph_synthetic_2.graph import generate_signals
from . import fcit


class FCITWrapper(CondIndTest):
    measure = 'fcit'
    two_sided = False

    def __init__(self, **kwargs):
        kwargs.setdefault('significance', 'analytic')
        super().__init__(**kwargs)
        self.significance = 'analytic'

    def get_dependence_measure(self, array, xyz, data_type=None):
        x = array[xyz == 0].T
        y = array[xyz == 1].T
        z = array[xyz == 2].T

        if z.shape[1] == 0:
            z = None

        pval = fcit.test(x, y, z)

        return 1.0 - pval

    def get_significance(self, val, T, dim=0, xyz=None):
        """
        Convert the stored measure back to a p-value for Tigramite.
        val here is (1 - pval), so pval = 1 - val.
        """
        return 1.0 - val


def get_dataframe(signals: dict | np.ndarray,
                  feature_names: list[str]) -> data_processing.DataFrame:

    if isinstance(signals, dict):
        signals = np.array([signal for signal in signals.values()])
        signals = signals.T  # shape (T, N)

    signals: np.ndarray
    if signals.ndim == 2:
        return data_processing.DataFrame(data=signals,
                                         analysis_mode='single',
                                         var_names=feature_names)
    elif signals.ndim == 3:
        # assuming (smaples, frames, features)
        data_dict = {i: arr for i, arr in enumerate(signals)}
        return data_processing.DataFrame(data=data_dict,
                                         analysis_mode='multiple',
                                         var_names=feature_names)


def plot_signals(
    dataframe: data_processing.DataFrame,
    save_dir: Path = Path('temp')) -> None:
    if dataframe.analysis_mode == 'single':
        fig, axes = plotting.plot_timeseries(dataframe)
        plt.savefig(save_dir / 'tm_plot.png')
        plt.close()
    elif dataframe.analysis_mode == 'multiple':
        for i in range(10):
            fig, axes = plotting.plot_timeseries(dataframe, selected_dataset=i)
            plt.savefig(save_dir / f'tm_plot{i}.png')
            plt.close()
    # else:


def investigate_dependencies(dataframe: data_processing.DataFrame,
                             save_path: Path = Path('temp') /
                             'tm_dependencies.png'):
    dataframe = deepcopy(dataframe)

    matrix = plotting.setup_density_matrix(
        N=dataframe.values[0].shape[1],
        var_names=dataframe.var_names,
    )

    # Compute mean/std across all samples concatenated, then apply to each
    all_data = np.concatenate(list(dataframe.values.values()), axis=0)
    mean, std = data_processing.weighted_avg_and_std(
        all_data, axis=0, weights=np.ones_like(all_data))
    std = np.where(std == 0, 1, std)

    standardised_dict = {
        m: (deepcopy(arr) - mean) / std
        for m, arr in dataframe.values.items()
    }
    standardised_dataframe = data_processing.DataFrame(
        data=standardised_dict,
        analysis_mode='multiple',
        var_names=dataframe.var_names,
    )

    matrix.add_densityplot(
        standardised_dataframe,
        snskdeplot_args={
            'cmap': 'Reds',
            'alpha': 1.,
            'levels': 4
        },
    )

    normalised_dataframe = normalise_dataframe(dataframe)

    matrix.add_densityplot(normalised_dataframe,
                           snskdeplot_args={
                               'cmap': 'Greys',
                               'alpha': 1.,
                               'levels': 4
                           })

    plt.savefig(save_path)
    plt.close()


def normalise_signals(signals: np.ndarray) -> np.ndarray:
    """Rank-Gaussianise (samples, frames, features) array across all samples."""
    samples, frames, features = signals.shape
    flat = signals.reshape(-1, features)
    flat_normalised = data_processing.trafo2normal(
        flat, np.zeros_like(flat, dtype=bool))
    return flat_normalised.reshape(samples, frames, features)


def normalise_dataframe(
        dataframe: data_processing.DataFrame) -> data_processing.DataFrame:
    lengths = [arr.shape[0] for arr in dataframe.values.values()]
    all_data_concat = np.concatenate(list(dataframe.values.values()), axis=0)
    all_normalised = data_processing.trafo2normal(
        all_data_concat, np.zeros_like(all_data_concat, dtype=bool))
    splits = np.cumsum(lengths[:-1])
    normalised_dict = {
        m: arr
        for m, arr in zip(dataframe.values.keys(),
                          np.split(all_normalised, splits))
    }
    normalised_dataframe = data_processing.DataFrame(
        data=normalised_dict,
        analysis_mode='multiple',
        var_names=dataframe.var_names,
    )
    return normalised_dataframe


def apply_pcmci(dataframe: data_processing.DataFrame, save_dir=Path('temp')):
    # pcmci = PCMCI(dataframe, GPDC(), 1)
    pcmci = PCMCI(dataframe, cond_ind_test=ParCorr(), verbosity=1)

    node_classification = {
        i: 'system'
        for i in range(len(dataframe.var_names))
    }
    pcmci = JPCMCIplus(dataframe=dataframe,
                       cond_ind_test=ParCorr(),
                       verbosity=1,
                       node_classification=node_classification)
    # pcmci = PCMCI(dataframe, cond_ind_test=ParCorr())
    # correlations = pcmci.get_lagged_dependencies(tau_max=20,
    #                                              val_only=True)['val_matrix']
    # fig = plotting.plot_lagfuncs(correlations,
    #                              setup_args={'var_names': dataframe.var_names})
    # plt.savefig(save_dir / 'lagged_dependencies.png')
    # results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.2, alpha_level=0.05)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=5, pc_alpha=0.001)
    # results = pcmci.run
    plotting.plot_graph(
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=dataframe.var_names,
        # cmap_nodes='RdYlBu',
        # cmap_edges='RdYlBu',
        vmin_edges=-0.4,
        vmax_edges=0.4,
        vmin_nodes=-0.4,
        vmax_nodes=0.4)
    plt.savefig(save_dir / 'results_graph.png')
    plt.close()

    plotting.plot_time_series_graph(graph=results['graph'],
                                    val_matrix=results['val_matrix'],
                                    var_names=dataframe.var_names,
                                    node_classification=node_classification)
    plt.savefig(save_dir / 'results_tmeporal_graph.png')
    plt.close()


def get_synthetic_signals():
    signals = generate_signals()
    feature_names = list(signals.keys())
    return signals, feature_names


def load_signals(hdf5_paths: Path | list[Path], features: list[str] = None):
    if isinstance(hdf5_paths, Path):
        hdf5_paths = [hdf5_paths]
    all_data = []
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, 'r') as f:

            data = f['Cells']['Phase']
            if features is None:
                features = [
                    feat for feat in data.keys()
                    if feat not in ('Images', 'First Frame', 'Last Frame', 'X',
                                    'Y', 'CellDeath', 'Macrophage',
                                    'Dead Macrophage', 'Speed')
                ]
            data = np.array([data[feat] for feat in features
                             ])  # (featues, frame, samples)

            # only use samples observed for full tiem series
            nan_mask = np.any(np.isnan(data), axis=(0, 1))
            # print(nan_mask, np.unique(nan_mask))
            data = data[:, :, ~nan_mask]
            data = data.transpose(2, 1, 0)  # (samples, frame, features)
            all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    # all_data = all_data[:, 50:, :]
    return all_data, features


def main():
    # signals, feature_names = get_synthetic_signals()
    # feature_names = []
    hdf5_paths = [
        h for h in (Path('~/thor_server').expanduser() / '24_02').glob('*.h5')
        if h.stem in ('C', 'D', 'E', 'F')
    ]
    features = [
        # 'Area',
        'Total Fluorescence',
        # 'Displacement',
        # 'External Fluorescence Intensity within 50 pixels',
        # 'Perimeter',
        'Alive Phagocytes within 250 pixels',
        'Fluorescence Distance Mean',
        # 'Area',
    ]
    signals, feature_names = load_signals(hdf5_paths, features)
    print(f'{signals.shape[0]} Samples of length {signals.shape[1]}')
    # signals = generate_signals()
    # feature_names = list(signals.keys())

    dataframe = get_dataframe(signals, feature_names)
    dataframe = normalise_dataframe(dataframe)
    plot_signals(dataframe)
    investigate_dependencies(dataframe)
    apply_pcmci(dataframe)


if __name__ == '__main__':
    main()
