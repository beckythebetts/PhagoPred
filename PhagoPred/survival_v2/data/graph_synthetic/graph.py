from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from pyvis.network import Network

from . import noise_funcs, base_funcs
from .rules import (
    Var,
    Const,
    Min,
    Max,
    Apply,
    Rule,
    ReLU,
    threshold,
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
                 time_steps: int = None):
        self.features = features
        self.rules = rules
        if time_steps is None:
            time_steps = max(rule.get_delay() for rule in rules) + 1
        self.time_steps = time_steps
        self.nx_graph = self._build_nx_graph()

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

    def sample_graph(self) -> dict[str, np.ndarray]:
        """Simulate feature trajectories according to the graph's rules."""
        signals = {
            feature.name: feature.generate_signal(self.time_steps)
            for feature in self.features
        }

        for t in range(self.time_steps):
            for rule in self.rules:
                signals = rule.apply_step(signals, t)

        for feature in self.features:
            signal = signals[feature.name]
            signal += feature.post_noise.sample(self.time_steps).astype(float)
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
        Rule(target='C',
             expr=Apply(abs, Var('B', lag=5)) + (0.9 * Apply(abs, Var('C')))),
        Rule(target='D', expr=Apply(threshold, Var('B'), thresh=2.0) * 10)
    ]

    graph = CausalGraph(features, rules, time_steps=500)

    graph.plot_feature_graph(save_path=temp_save_path / 'features_graph.png')
    graph.plot_signals(save_path=temp_save_path / 'signals_plot.png')
    graph.plot_temporal_graph(save_path=temp_save_path / 'temporal_graph.png')
    return graph.sample_graph()


if __name__ == '__main__':
    test()
