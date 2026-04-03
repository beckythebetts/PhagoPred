from pathlib import Path
from copy import deepcopy

import tigramite.data_processing as data_processing
import tigramite.plotting as plotting
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.independence_tests.gpdc import GPDC
# from tigramite.independence_tests.gpdc_torch import GPDCtorch
import numpy as np
import matplotlib.pyplot as plt

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


def get_dataframe(signals: dict,
                  feature_names: list[str]) -> data_processing.DataFrame:

    signals = np.array([signal for signal in signals.values()])
    signals = signals.T  # shape (T, N)

    print(feature_names)

    return data_processing.DataFrame(data=signals,
                                     analysis_mode='single',
                                     var_names=feature_names)


def plot_signals(dataframe: data_processing.DataFrame,
                 save_path: Path = Path('temp') / 'tm_plot.png') -> None:
    fig, axes = plotting.plot_timeseries(dataframe)
    plt.savefig(save_path)
    plt.close()


def investigate_dependencies(dataframe: data_processing.DataFrame,
                             save_path: Path = Path('temp') /
                             'tm_dependencies.png'):
    dataframe = deepcopy(dataframe)

    matrix = plotting.setup_density_matrix(
        N=dataframe.values[0].shape[1],
        var_names=dataframe.var_names,
    )

    standardised_data_array = deepcopy(dataframe.values[0])
    mean, std = data_processing.weighted_avg_and_std(
        standardised_data_array,
        axis=0,
        weights=np.ones_like(standardised_data_array))
    standardised_data_array -= mean
    standardised_data_array /= std

    standardised_dataframe = data_processing.DataFrame(
        data=standardised_data_array,
        analysis_mode='single',
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

    normalised_data_array = deepcopy(dataframe.values[0])
    normalised_data_array = data_processing.trafo2normal(
        normalised_data_array, np.zeros_like(normalised_data_array,
                                             dtype=bool))
    normalised_dataframe = data_processing.DataFrame(
        data=normalised_data_array,
        var_names=dataframe.var_names,
    )

    matrix.add_densityplot(normalised_dataframe,
                           snskdeplot_args={
                               'cmap': 'Greys',
                               'alpha': 1.,
                               'levels': 4
                           })

    plt.savefig(save_path)
    plt.close()


def apply_pcmci(dataframe: data_processing.DataFrame, save_dir=Path('temp')):
    # pcmci = PCMCI(dataframe, GPDC(), 1)
    pcmci = PCMCI(dataframe, cond_ind_test=CMIknn(knn=5), verbosity=1)
    # pcmci = PCMCI(dataframe, cond_ind_test=ParCorr())
    # correlations = pcmci.get_lagged_dependencies(tau_max=20,
    #                                              val_only=True)['val_matrix']
    # fig = plotting.plot_lagfuncs(correlations,
    #                              setup_args={'var_names': dataframe.var_names})
    # plt.savefig(save_dir / 'lagged_dependencies.png')
    results = pcmci.run_pcmci(tau_max=5, pc_alpha=0.05, alpha_level=0.01)
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


def main():
    signals = generate_signals()
    feature_names = list(signals.keys())

    dataframe = get_dataframe(signals, feature_names)
    plot_signals(dataframe)
    # investigate_dependencies(dataframe)
    apply_pcmci(dataframe)


if __name__ == '__main__':
    main()
