from __future__ import annotations
from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from pathlib import Path
import itertools as it

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from . import noise_funcs, base_funcs
from .rule.rule import Rule, Input, AutoRegressiveRule
from .rule import (
    transforms,
    combinations,
    magnitude,
    shape,
    thresholds,
    timing,
)
from .rule.timing import FixedDelay, VariableDelay, Instantaneous


@dataclass
class Node:
    """Node of graph, with feature name and time step."""
    name: str

    base_func: base_funcs.BaseFunc = field(default_factory=base_funcs.Constant)
    pre_noise: noise_funcs.Noise = field(default=noise_funcs.GaussianNoise(1))
    post_noise: noise_funcs.Noise = field(default=noise_funcs.NoNoise())

    parents: list[Node] = field(default_factory=list)

    def generate_signal(self, time_steps: int) -> np.ndarray:
        """Generate base signal for node, with base_func and pre_noise"""
        signal = self.base_func.generate(time_steps).astype(float)
        signal += self.pre_noise.sample(time_steps).astype(float)
        return signal


# class CausalGraph:
#     """Class to store base graph used for generating cell trajectores."""

#     def __init__(self, time_steps: int, nodes: list[Node]):
#         self.time_steps = time_steps
#         self.nodes = nodes
#         self.nx_graph = nx.DiGraph()

#         self._add_nodes()

#         self.signals = {}
#         self.rules: list[Rule] = []
#         self.auto_regressive_rules: list[Rule] = []


#     def _add_nodes(self) -> None:
#         for node in self.nodes:
#             self.nx_graph.add_node(node.name, node=node)
class CausalGraph:
    """Class to store base graph used for generating cell trajectores."""

    def __init__(self, time_steps: int, nodes: list[Node]):
        self.time_steps = time_steps
        self.nodes = nodes
        self.nx_graph = nx.DiGraph()

        self._add_nodes()

        self.signals = {}
        self.rules: list[Rule] = []
        self.auto_regressive_rules: list[Rule] = []

    def _add_nodes(self) -> None:
        for node in self.nodes:
            self.nx_graph.add_node(node.name, node=node)

    def add_rule(self, rule: Rule):
        if isinstance(rule, AutoRegressiveRule):
            self.auto_regressive_rules.append(rule)
            self.nx_graph.add_edge(rule.target, rule.target, rule=rule)
            return

        self.rules.append(rule)
        target_node = rule.target
        for i in rule.inputs:
            src_node = i.feature
            self.nx_graph.add_edge(src_node, target_node, rule=rule)

    def _get_node(self, name: str) -> Node:
        return self.nx_graph.nodes[name]['node']

    def simulate_graph(self) -> dict[str, np.ndarray]:

        signals = {
            node_name:
            self._get_node(node_name).generate_signal(self.time_steps)
            for node_name in self.nx_graph.nodes
        }

        for t in range(1, self.time_steps):
            for r in self.auto_regressive_rules:
                signals[r.target][t] += r.apply_step(signals, t)

        dag = nx.subgraph_view(self.nx_graph, filter_edge=lambda u, v: u != v)
        for node_name in nx.topological_sort(dag):
            incoming = self.nx_graph.in_edges(node_name, data=True)
            for _, _, edge_data in incoming:
                if edge_data['rule'] not in self.auto_regressive_rules:
                    signals[node_name] += edge_data['rule'].apply(signals)

        for node_name in self.nx_graph.nodes:
            signals[node_name] += self._get_node(node_name).post_noise.sample(
                self.time_steps).astype(float)
        self.signals = signals
        return self.signals

    def plot_signals(self, save_path: Path) -> None:
        fig, axs = plt.subplots(nrows=len(self.nodes),
                                ncols=1,
                                figsize=(10, 10))
        if self.signals is None:
            self.simulate_graph()
        for i, feat in enumerate(self.nodes):
            axs[i].plot(np.arange(self.time_steps), self.signals[feat.name])
            axs[i].set_ylabel(feat.name)
        plt.savefig(save_path)

    def plot_graph(self, save_path: Path) -> None:
        pos = nx.spring_layout(self.nx_graph)
        options = {
            "font_size": 12,
            "node_size": 1000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 3,
            "width": 3,
        }

        edge_labels = {}
        for edge, edge_data in self.nx_graph.edges.items():
            t = edge_data['rule'].timing
            if isinstance(t, Instantaneous):
                edge_labels[tuple(edge)] = "0"
            elif isinstance(t, FixedDelay):
                edge_labels[tuple(edge)] = f"{t.delay}"
            elif isinstance(t, VariableDelay):
                edge_labels[tuple(edge)] = f"{t.mean}±{t.sigma:.0f}"
        # connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

        fig, ax = plt.subplots(figsize=(8, 6))

        pos = nx.shell_layout(self.nx_graph, scale=10)
        nx.draw_networkx_nodes(self.nx_graph,
                               pos,
                               ax=ax,
                               node_size=1000,
                               node_color='white',
                               edgecolors='black',
                               linewidths=3)
        nx.draw_networkx_labels(self.nx_graph, pos, ax=plt.gca(), font_size=12)
        nx.draw_networkx_edges(self.nx_graph,
                               pos,
                               ax=ax,
                               width=3,
                               node_size=1000,
                               arrowstyle='->')
        g = nx.draw_networkx_edge_labels(
            self.nx_graph,
            pos,
            edge_labels=edge_labels,
            # verticalalignment='top',
            #  connectionstyle="arc3,rad=0.1",
            ax=ax,
            label_pos=0.3,
            font_color='red',
        )

        node_size = 1000
        loop_shift = 0.5 * 0.005 * node_size  # mirrors networkx self-loop v_shift formula

        for (u, v), obj in g.items():
            if u == v:
                x, y = obj.get_position()
                obj.set_position((x, y + loop_shift))

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_temporal_graph(self, save_path: Path) -> None:
        """Plot the causal graph unrolled over time (DBN-style)"""

        def delay_info(t_obj):
            """Return (delay_in_real_steps, is_variable, edge_label)."""
            if isinstance(t_obj, Instantaneous):
                return 0, False, "Delay = 0"
            if isinstance(t_obj, FixedDelay):
                return t_obj.delay, False, f"Delay = {t_obj.delay}"
            if isinstance(t_obj, VariableDelay):
                return t_obj.mean, True, f"delay = {t_obj.mean}±{t_obj.sigma:.0f}"
            else:
                raise ValueError(f"Unknown timing type: {type(t_obj)}")

        delays = [
            delay_info(r.timing)[0]
            for r in self.rules + self.auto_regressive_rules
        ]
        max_delay = max(delays)
        nonzero = [d for d in delays if d != 0]
        time_step = reduce(gcd, nonzero) if nonzero else 1
        n_cols = max_delay // time_step + 1

        feat_rows = {f.name: i for i, f in enumerate(self.nodes)}
        pos = {
            (f, t): (t // time_step, -feat_row)
            for t in range(0, max_delay + 1, time_step)
            for f, feat_row in feat_rows.items()
        }

        fig, ax = plt.subplots(figsize=(2 * n_cols, 1.5 * len(self.nodes)))
        node_radius = 0.3
        for (feat, t), (x, y) in pos.items():
            circle = plt.Circle((x, y),
                                node_radius,
                                color='white',
                                ec='black',
                                lw=2)
            ax.add_patch(circle)
            ax.text(x,
                    y,
                    feat,
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold')

        for j, t in enumerate(range(0, max_delay + 1, time_step)):
            ax.text(j,
                    0.8,
                    f"t=t_0 + {t}",
                    ha='center',
                    va='top',
                    fontsize=8,
                    color='black')
        all_rules = self.rules + self.auto_regressive_rules
        cmap = plt.get_cmap('Set1')
        legend_handles = []
        for i, rule in enumerate(all_rules):
            delay, is_var, dlabel = delay_info(rule.timing)
            ls = '-' if not is_var else '--'
            color = cmap(i)
            src_names = ', '.join(inp.feature for inp in rule.inputs)
            legend_handles.append(
                mlines.Line2D(
                    [], [],
                    color=color,
                    linestyle=ls,
                    linewidth=1.5,
                    marker='>',
                    markersize=5,
                    label=f"{src_names} → {rule.target}  [{dlabel}]"))
            for inp in rule.inputs:
                for t in range(0, max_delay + 1, time_step):
                    t_target = t + delay
                    if t_target > max_delay:
                        continue
                    x0, y0 = pos[(inp.feature, t)]
                    x1, y1 = pos[(rule.target, t_target)]
                    dx, dy = x1 - x0, y1 - y0
                    dist = np.hypot(dx, dy)
                    if dist > 0:
                        ux, uy = dx / dist, dy / dist
                    else:
                        ux, uy = 0.0, 0.0
                    xs = x0 + ux * node_radius
                    ys = y0 + uy * node_radius
                    xe = x1 - ux * node_radius
                    ye = y1 - uy * node_radius
                    ax.annotate("",
                                xy=(xe, ye),
                                xytext=(xs, ys),
                                arrowprops=dict(
                                    arrowstyle='->',
                                    color=color,
                                    lw=2,
                                    linestyle=ls,
                                ),
                                zorder=1)
        ax.legend(handles=legend_handles,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.05),
                  ncol=max(1, len(legend_handles)),
                  fontsize=7,
                  framealpha=0.8,
                  title="Rules",
                  title_fontsize=8)
        ax.set_xlim(-0.8, n_cols - 0.5)
        ax.set_ylim(-len(self.nodes) + 0.3, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Temporal Causal Graph", fontsize=12, pad=10)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # graph = CausalGraph(time_steps=200, features=['A', 'B', 'C'])

    features = [
        Node(name='A',
             base_func=base_funcs.Ramp(slope=0.1),
             pre_noise=noise_funcs.GaussianNoise(0.5),
             post_noise=noise_funcs.GaussianNoise(0.5)),
        Node(name='B',
             base_func=base_funcs.Constant(0),
             pre_noise=noise_funcs.GaussianNoise(0.5),
             post_noise=noise_funcs.GaussianNoise(0.5)),
        Node(name='C',
             base_func=base_funcs.Oscillation(amplitude=1, frequency=0.05),
             pre_noise=noise_funcs.GaussianNoise(0.5),
             post_noise=noise_funcs.GaussianNoise(0.5)),
        Node(name='D',
             base_func=base_funcs.RandomWalk(step_scale=0.5),
             pre_noise=noise_funcs.GaussianNoise(0.5),
             post_noise=noise_funcs.GaussianNoise(0.5)),
    ]
    graph = CausalGraph(time_steps=200, nodes=features)

    rules = [
        Rule([Input('C', transforms.Gradient(10), thresholds.Cumulative(0))],
             target='B',
             combination=combinations.Add(),
             timing=timing.FixedDelay(20),
             shape=shape.Delta(),
             magnitude=magnitude.Fixed(1)),
        Rule([
            Input('A', transforms.Value(1), thresholds.NoThreshold()),
            Input('C', transforms.Value(1), thresholds.NoThreshold()),
        ],
             target='D',
             combination=combinations.Add(),
             timing=timing.VariableDelay(10, 30),
             shape=shape.Gaussian(sigma=5),
             magnitude=magnitude.Fixed(0.5)),
        AutoRegressiveRule(
            inputs=[Input('B', transforms.Value(1), thresholds.NoThreshold())],
            target='B',
            timing=timing.FixedDelay(5),
            combination=combinations.Add(),
            magnitude=magnitude.Fixed(1.0),
        ),
    ]
    for rule in rules:
        graph.add_rule(rule)
    graph.plot_temporal_graph(
        save_path="/home/ubuntu/PhagoPred/temp/test_temporal_graph.png")
    graph.plot_graph(save_path=f"/home/ubuntu/PhagoPred/temp/test_graph.png")
    for n in range(5):
        graph.simulate_graph()
        graph.plot_signals(
            save_path=f"/home/ubuntu/PhagoPred/temp/test_data_{n}.png")
