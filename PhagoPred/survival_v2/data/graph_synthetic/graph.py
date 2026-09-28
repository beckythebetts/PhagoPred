from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from pyvis.network import Network
from tqdm import tqdm

from . import noise_funcs, base_funcs
from .rules import (
    Var,
    Const,
    Min,
    Max,
    Rule,
    Abs,
    Threshold,
)

temp_save_path = Path(
    "C:\\Users\\php23rjb\\Documents\\PhagoPred\\temp\\graph_synthetic_data")
os.makedirs(temp_save_path, exist_ok=True)

node_format_atributes = {
    'shape': 'circle',
    'fillcolor': 'white',
    'fontname': 'Helvetica',
    'width': '1.0',
    'fixedsize': 'true'
}
edge_format_attributes = {'fontname': 'Helvetica', 'fontsize': '10'}


@dataclass
class Feature:
    """Time varying feature"""
    name: str

    base_func: base_funcs.BaseFunc = field(
        default_factory=base_funcs.Constant
    )  # Base function MUST be constant for staionarity condition!!!
    inital_value: float = 0.0
    pre_noise: noise_funcs.Noise = field(default=noise_funcs.GaussianNoise(1))
    post_noise: noise_funcs.Noise = field(default=noise_funcs.NoNoise())
    hidden: bool = False

    def generate_signal(self, time_steps: int) -> np.ndarray:
        """Generate base signal for node, with base_func and pre_noise"""
        signal = self.base_func.generate(time_steps).astype(float)
        # signal[0] += self.inital_value.sample(1).astype(float)[0]
        signal += self.pre_noise.sample(time_steps).astype(float)
        return signal


@dataclass
class Node:
    """Node of graph, with feature name and time step."""
    feature: Feature
    time_step: int

    def __str__(self):
        return f"{self.feature.name}_{self.time_step}"


class CausalGraph:
    """Directed graph of features with rules governing their interactions."""

    def __init__(self,
                 features: list[Feature],
                 rules: list[Rule],
                 time_steps: int = None,
                 burn_in: int = 0,
                 death_burn_in: int = 0):
        """
        Args:
            burn_in: frames simulated (and discarded) before frame 0, so every
                trajectory starts from the process's long-run behaviour rather
                than from zero.
            death_burn_in: the last ``death_burn_in`` burn-in frames run the
                death process; cells dying there are rejected, so frame 0 only
                holds cells that were alive when observation started.
        """
        self.features = features
        self.rules = rules
        if time_steps is None:
            time_steps = max(rule.get_delay() for rule in rules) + 1
        self.time_steps = time_steps
        if death_burn_in > burn_in:
            raise ValueError(
                f'death_burn_in ({death_burn_in}) > burn_in ({burn_in})')
        self.burn_in = burn_in
        self.death_burn_in = death_burn_in
        self.nx_graph = self._build_nx_graph()

        self.map = None  # Mapping of downstream children nodes for each parent node

    @property
    def max_lag(self) -> int:
        return max(rule.max_lag for rule in self.rules)

    def get_variances(
        self,
        target_feature: str,
        max_horizon: int,
        max_base_time_steps: int,
        hazard_calibration_func: callable,
        base_sample_size: int = 200,
        branch_sample_size: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute hazard variance and binary Bernoulli floor via branching.

        base_time_steps is sampled uniformly from [1, max_base_time_steps] each
        iteration to approximately match the landmark frame distribution of the test dataset.
        The Bernoulli floor is weighted by base survival to condition on the cell
        being alive at the branch point.

        Returns:
            hazard_branch_var: (max_horizon,) unobserved hazard variance
            hazard_total_var:  (max_horizon,) total hazard variance
            bernoulli_floor:   (max_horizon,) survival-weighted E[p̄*(1-p̄)]
        """
        hazard_branches = []
        cdf_branches = []
        survival_weights = []

        for _ in tqdm(range(base_sample_size)):
            base_time_steps = np.random.randint(1, max_base_time_steps + 1)
            base_signals = self.sample_graph(base_time_steps)

            base_hazard = np.clip(
                hazard_calibration_func(base_signals[target_feature]), 0.0,
                1.0)
            survival_weights.append(float(np.prod(1.0 - base_hazard)))

            base_hazard_results = []
            base_cdf_results = []
            for _ in range(branch_sample_size):
                signals = {
                    feature.name:
                    np.append(base_signals[feature.name],
                              feature.generate_signal(max_horizon))
                    for feature in self.features
                }
                for t in range(base_time_steps, base_time_steps + max_horizon):
                    for rule in self.rules:
                        signals = rule.apply_step(signals, t)

                branch_traj = signals[target_feature][
                    base_time_steps:base_time_steps + max_horizon]
                hazard = np.clip(hazard_calibration_func(branch_traj), 0.0,
                                 1.0)
                base_hazard_results.append(hazard)
                base_cdf_results.append(1.0 - np.cumprod(1.0 - hazard))

            hazard_branches.append(base_hazard_results)
            cdf_branches.append(base_cdf_results)

        h_arr = np.array(
            hazard_branches
        )  # (base_sample_size, branch_sample_size, max_horizon)
        c_arr = np.array(cdf_branches)

        weights = np.array(survival_weights)
        weights = weights / weights.sum()
        w = weights[:, np.newaxis]  # (base_sample_size, 1)

        # hazard variance — no survival weighting needed
        hazard_branch_var = np.mean(np.var(h_arr, axis=1), axis=0)
        hazard_total_var = hazard_branch_var + np.var(np.mean(h_arr, axis=1),
                                                      axis=0)

        # CDF variance — survival-weighted, same decomposition as hazard
        p_bar = np.mean(c_arr, axis=1)  # (base_sample_size, max_horizon)
        cdf_branch_var = np.sum(w * np.var(c_arr, axis=1), axis=0)
        grand_mean = np.sum(w * p_bar, axis=0)  # (max_horizon,)
        cdf_between_var = np.sum(w * (p_bar - grand_mean)**2, axis=0)
        cdf_total_var = cdf_branch_var + cdf_between_var

        # Bernoulli floor — derivable from CDF variance
        bernoulli_floor = grand_mean * (1.0 - grand_mean) - cdf_between_var

        return hazard_branch_var, hazard_total_var, cdf_branch_var, cdf_total_var, bernoulli_floor

    def branch_cif_at_horizon(
        self,
        base_signals: dict[str, np.ndarray],
        landmark_frame: int,
        max_horizon: int,
        hazard_calibration_func: callable,
        target_feature: str = 'Hazard',
        branch_sample_size: int = 1000,
    ) -> np.ndarray:
        """Aleatory distribution of CIF at the horizon for a fixed base history.

        Conditions on the realised trajectory up to ``landmark_frame`` (the
        cell's known present) and samples ``branch_sample_size`` stochastic
        futures, returning the cumulative incidence evaluated ``max_horizon``
        steps past the landmark for each branch. The spread of the returned
        values is the irreducible (aleatory) uncertainty in P(event within the
        horizon) given the cell's present — the noise floor a model conditioned
        on the same history cannot beat.

        Returns:
            cif_at_horizon: (branch_sample_size,) CIF(landmark + max_horizon).
        """
        base_hist = {
            f.name: np.asarray(base_signals[f.name])[:landmark_frame]
            for f in self.features
        }
        cif_at_horizon = np.empty(branch_sample_size)
        for b in range(branch_sample_size):
            signals = {
                f.name:
                np.append(base_hist[f.name], f.generate_signal(max_horizon))
                for f in self.features
            }
            for t in range(landmark_frame, landmark_frame + max_horizon):
                for rule in self.rules:
                    signals = rule.apply_step(signals, t)
            branch_traj = signals[target_feature][
                landmark_frame:landmark_frame + max_horizon]
            hazard = np.clip(hazard_calibration_func(branch_traj), 0.0, 1.0)
            cif_at_horizon[b] = (1.0 - np.cumprod(1.0 - hazard))[-1]
        return cif_at_horizon

    def _build_nx_graph(self) -> nx.DiGraph:
        """Construct a NetworkX DiGraph from the features and rules."""
        G = nx.DiGraph()
        for feature in self.features:
            for time in range(self.time_steps):
                G.add_node(feature.name, info=Node(feature, time))

        for rule in self.rules:
            for input in rule.get_inputs():
                G.add_edge(input, rule.target, rule=rule)

        return G

    def simulate(
        self,
        n_cells: int,
        window: int,
        hazard_func: callable | None = None,
        keep_context: int = 0,
        post_noise: bool = True,
        return_noise: bool = False,
        max_batch_cells: int = 1000,
        max_rejection_factor: int = 100,
    ) -> dict[str, np.ndarray] | tuple[dict, dict]:
        """Simulate ``n_cells`` trajectories at once, as (T, n_cells) arrays.

        ``self.burn_in`` frames are simulated before frame 0. If
        ``hazard_func`` is given, cells dying during the last
        ``self.death_burn_in`` of those frames are rejected (and replaced by
        further simulation), so only cells alive at frame 0 are returned.

        Returns frames [-keep_context, window) (context first), plus the
        matching pre-noise if ``return_noise``. ``post_noise`` adds each
        feature's observation noise to the returned signals only.
        """
        burn_in, death_burn_in = self.burn_in, self.death_burn_in
        if keep_context > burn_in:
            raise ValueError(
                f'keep_context ({keep_context}) > burn_in ({burn_in})')
        T = burn_in + window
        start = burn_in - keep_context
        condition = hazard_func is not None and death_burn_in > 0

        kept_sig, kept_noise = [], []
        n_kept, n_simulated, accept_rate = 0, 0, 1.0
        while n_kept < n_cells:
            if n_simulated > max_rejection_factor * n_cells:
                raise RuntimeError(
                    f'Only {n_kept}/{n_cells} cells survived the burn-in out '
                    f'of {n_simulated} simulated; the hazard is too high '
                    f'during the last {death_burn_in} burn-in frames.')
            n = min(int(np.ceil((n_cells - n_kept) / accept_rate * 1.1)) + 1,
                    max_batch_cells)
            n_simulated += n
            noise = {
                f.name: np.stack([f.generate_signal(T) for _ in range(n)],
                                 axis=1)
                for f in self.features
            }
            sig = {name: arr.copy() for name, arr in noise.items()}
            for t in range(T):
                for rule in self.rules:
                    rule.apply_step(sig, t)

            keep = np.ones(n, dtype=bool)
            if condition:
                h = np.clip(
                    hazard_func(sig['Hazard'][burn_in - death_burn_in:burn_in]),
                    0.0, 1.0)
                keep = np.random.rand(n) < np.prod(1.0 - h, axis=0)
                accept_rate = max(keep.mean(), 0.01)

            kept_sig.append({k: v[start:, keep] for k, v in sig.items()})
            kept_noise.append({k: v[start:, keep] for k, v in noise.items()})
            n_kept += int(keep.sum())

        signals = {
            f.name: np.concatenate([b[f.name] for b in kept_sig],
                                   axis=1)[:, :n_cells]
            for f in self.features
        }
        if post_noise:
            for f in self.features:
                signals[f.name] += np.stack([
                    f.post_noise.sample(keep_context + window)
                    for _ in range(n_cells)
                ],
                                            axis=1)
        if not return_noise:
            return signals
        noise = {
            f.name: np.concatenate([b[f.name] for b in kept_noise],
                                   axis=1)[:, :n_cells]
            for f in self.features
        }
        return signals, noise

    def sample_graph(self,
                     time_steps: int = None,
                     signals: dict | None = None) -> dict[str, np.ndarray]:
        """Simulate one trajectory according to the graph's rules.

        Without ``signals``, runs ``simulate`` for a single cell (so the
        burn-in applies, but not survivor conditioning, which needs a hazard
        function). With caller-supplied ``signals`` (pre-noise), the rules are
        applied to them directly from frame 0, with no burn-in.
        """
        if time_steps is None:
            time_steps = self.time_steps
        if signals is None:
            sim = self.simulate(1, time_steps)
            return {name: arr[:, 0] for name, arr in sim.items()}

        for t in range(time_steps):
            for rule in self.rules:
                signals = rule.apply_step(signals, t)

        for feature in self.features:
            signal = signals[feature.name]
            signal += feature.post_noise.sample(time_steps).astype(float)
            signals[feature.name] = signal

        return signals

    def plot_signals(self, save_path: Path) -> None:
        fig, axs = plt.subplots(nrows=len(self.features),
                                ncols=1,
                                figsize=(10, 10))
        signals = self.sample_graph()
        for i, feat in enumerate(self.features):
            axs[i].plot(np.arange(self.time_steps), signals[feat.name])
            axs[i].set_ylabel(feat.name)
        plt.savefig(save_path)

    def plot_feature_graph(self, save_path: Path) -> None:
        """Feature-level summary graph.

        Saves a static graphviz render at save_path and an interactive
        pyvis HTML file alongside it (same stem, .html extension).
        """
        save_path = Path(save_path)

        # collect (src, tgt) -> lag labels from all rules
        edges: dict[tuple[str, str], list[str]] = {}
        for rule in self.rules:
            for src, lags in rule.get_inputs().items():
                key = (src, rule.target)
                label = 'lag: ' + ', '.join(str(l) for l in sorted(set(lags)))
                edges.setdefault(key, []).append(label)

        # --- graphviz (static) ---
        g = graphviz.Digraph(engine='dot')
        g.attr(rankdir='LR', fontname='Helvetica', fontsize='12')
        g.attr('node', **node_format_atributes)
        g.attr('edge', **edge_format_attributes)

        for feature in self.features:
            g.node(feature.name)

        for (src, tgt), labels in edges.items():
            g.edge(src,
                   tgt,
                   label=' | '.join(labels),
                   style='dashed' if src == tgt else 'solid')

        fmt = save_path.suffix.lstrip('.') or 'pdf'
        g.render(str(save_path.with_suffix('')), format=fmt, cleanup=True)

        # --- pyvis (interactive HTML) ---
        net = Network(directed=True,
                      height='500px',
                      width='100%',
                      notebook=False)
        net.barnes_hut(spring_length=200)

        for feature in self.features:
            net.add_node(feature.name,
                         label=feature.name,
                         size=25,
                         color='#AED6F1',
                         font={'size': 16})

        for (src, tgt), labels in edges.items():
            net.add_edge(src,
                         tgt,
                         title=' | '.join(labels),
                         arrows='to',
                         dashes=(src == tgt))

        net.save_graph(str(save_path.with_suffix('.html')))

    def plot_temporal_graph(self, save_path: Path) -> None:
        """Plot the causal graph unrolled over time (DBN-style) using graphviz."""
        save_path = Path(save_path)

        max_lag = max(rule.max_lag for rule in self.rules)
        n_cols = max_lag * 2 + 1

        g = graphviz.Digraph(engine='neato')
        g.attr(fontname='Helvetica', fontsize='11')
        g.attr('node', **node_format_atributes)
        g.attr('edge', **edge_format_attributes)

        x_gap, y_gap = 2.0, 1.5
        n_feats = len(self.features)

        for t in range(n_cols):
            x = t * x_gap
            label = 't' if t == max_lag else f't{t - max_lag:+d}'
            g.node(f'_t{t}',
                   label=label,
                   shape='none',
                   width='0',
                   height='0',
                   fontsize='10',
                   pos=f'{x},{n_feats * y_gap}!')
            for i, feature in enumerate(self.features):
                y = (n_feats - 1 - i) * y_gap
                g.node(f'{feature.name}_{t}',
                       label=feature.name,
                       pos=f'{x},{y}!')

        # draw one edge per (src, lag) combination per rule
        for rule in self.rules:
            for src, lags in rule.get_inputs().items():
                for lag in sorted(set(lags)):
                    for t in range(n_cols):
                        t_tgt = t + lag
                        if t_tgt >= n_cols:
                            continue
                        g.edge(f'{src}_{t}',
                               f'{rule.target}_{t_tgt}',
                               label=f'lag={lag}')

        fmt = save_path.suffix.lstrip('.') or 'pdf'
        g.render(str(save_path.with_suffix('')), format=fmt, cleanup=True)

    def _get_map(self):
        mapping = defaultdict(list)
        for rule in self.rules:
            for parent, edges in rule.get_map().items():
                mapping[parent].extend(edges)
        return mapping

    def map(self):
        if self.map is None:
            self.map = self._get_map()
        return self.map


def test():
    features = [
        Feature(
            name='A',
            base_func=base_funcs.Constant(0),
            pre_noise=noise_funcs.GaussianNoise(0.5),
            # post_noise=noise_funcs.GaussianNoise(0.5)
        ),
        Feature(
            name='B',
            base_func=base_funcs.Constant(0),
            pre_noise=noise_funcs.GaussianNoise(0.5),
            # post_noise=noise_funcs.GaussianNoise(0.5)
        ),
        Feature(
            name='C',
            base_func=base_funcs.Constant(0),
            pre_noise=noise_funcs.GaussianNoise(0.5),
            # post_noise=noise_funcs.GaussianNoise(0.5),
        ),
        Feature(
            name='D',
            base_func=base_funcs.Constant(0),
            pre_noise=noise_funcs.GaussianNoise(0.5),
            # post_noise=noise_funcs.GaussianNoise(0.5)
        ),
    ]

    rules = [
        Rule(target='A', expr=0.8 * Var('A')),
        Rule(target='B', expr=(0.8 * Var('B')) - Var('A')),
        Rule(target='C', expr=Abs(Var('B', lag=5)) + (0.9 * Abs(Var('C')))),
        Rule(target='D', expr=Threshold(Var('B'), thresh=2.0) * 10)
    ]

    graph = CausalGraph(features, rules, time_steps=500)

    graph.plot_feature_graph(save_path=temp_save_path / 'features_graph.png')
    graph.plot_signals(save_path=temp_save_path / 'signals_plot.png')
    graph.plot_temporal_graph(save_path=temp_save_path / 'temporal_graph.png')
    return graph.sample_graph()


if __name__ == '__main__':
    test()
