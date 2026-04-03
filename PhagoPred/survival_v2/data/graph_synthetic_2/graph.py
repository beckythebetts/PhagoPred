from __future__ import annotations
from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from pathlib import Path
import itertools as it
# import os

# os.environ["PATH"] += os.pathsep + r"C:\\Program Files\\Graphviz\\bin"

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import graphviz

from . import noise_funcs, base_funcs
from .rules.rules import Rule, Input, AccumulatorRule, DecayingAccumulatorRule, SEMRule
from .rules import (
    collapse,
    combinations,
    magnitude,
    shape,
    timing,
    transforms,
    structural_forms,
    component_funcs,
)


@dataclass
class Feature:
    """Time varying feature"""
    name: str

    base_func: base_funcs.BaseFunc = field(
        default_factory=base_funcs.Constant
    )  # Base function MUST be constant for staionarity condition!!!
    # inital_value: noise_funcs.Noise = field(
    #     default=noise_funcs.GaussianNoise(20))
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
                G.add_node(str(feature), info=Node(feature, time))

        for rule in self.rules:
            if isinstance(rule, SEMRule):
                for feat in rule.inputs:
                    G.add_edge(feat, rule.target, rule=rule)
            else:
                for inp in rule.inputs:
                    G.add_edge(inp.feature, rule.target, rule=rule)
                    if isinstance(rule, AccumulatorRule):
                        G.add_edge(rule.target, rule.target, rule=rule)

        return G

    def sample_graph(self) -> dict[str, np.ndarray]:
        """Simulate feature trajectories according to the graph's rules."""
        signals = {
            feature.name: feature.generate_signal(self.time_steps)
            for feature in self.features
        }

        for t in tqdm(range(self.time_steps), desc="Simulating graph"):
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

    def _get_rule_delay(self, rule) -> int:
        if isinstance(rule, SEMRule):
            return rule.lag
        if isinstance(rule, AccumulatorRule):
            return 1
        t = rule.timing
        if isinstance(t, timing.Fixed):
            return t.delay
        if isinstance(t, timing.Variable):
            return t.mean
        return 0

    def _rule_edge_label(self, rule) -> str:
        if isinstance(rule, SEMRule):
            return f'Δ={rule.lag}'
        if isinstance(rule, DecayingAccumulatorRule):
            return f'AR(1) decay={rule.decay_rate}'
        if isinstance(rule, AccumulatorRule):
            return 'AR(1)'
        t = rule.timing
        if isinstance(t, timing.Variable):
            return f'Δ~N({t.mean + 1},{t.sigma:.0f})'
        return f'Δ={t.delay + 1}'

    def plot_feature_graph(self, save_path: Path) -> None:
        """Feature-level summary graph.

        Saves a static graphviz render at save_path and an interactive
        pyvis HTML file alongside it (same stem, .html extension).
        """
        import graphviz
        from pyvis.network import Network

        save_path = Path(save_path)

        # Collect edges: (src, tgt) → labels from all rules
        edges: dict[tuple[str, str], list[str]] = {}
        for rule in self.rules:
            label = self._rule_edge_label(rule)
            if isinstance(rule, SEMRule):
                for feat in rule.inputs:
                    edges.setdefault((feat, rule.target), []).append(label)
            else:
                if isinstance(rule, AccumulatorRule):
                    edges.setdefault((rule.target, rule.target),
                                     []).append(label)
                for inp in rule.inputs:
                    edges.setdefault((inp.feature, rule.target),
                                     []).append(label)

        # --- graphviz (static) ---
        g = graphviz.Digraph(engine='dot')
        g.attr(rankdir='LR', fontname='Helvetica', fontsize='12')
        g.attr('node',
               shape='circle',
               style='filled',
               fillcolor='lightblue',
               fontname='Helvetica',
               width='0.8')
        g.attr('edge', fontname='Helvetica', fontsize='10')

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
        """Plot the causal graph unrolled over time (DBN-style)"""

        def delay_info(t_obj):
            """Return (delay_in_real_steps, is_variable, edge_label)."""
            # if isinstance(t_obj, Instantaneous):
            #     return 0, False, "Delay = 0"
            if isinstance(t_obj, timing.Fixed):
                return t_obj.delay, False, f"Delay = {t_obj.delay + 1}"
            if isinstance(t_obj, timing.Variable):
                return t_obj.mean, True, f"delay = {t_obj.mean + 1}±{t_obj.sigma:.0f}"
            else:
                raise ValueError(f"Unknown timing type: {type(t_obj)}")

        delays = [
            r.lag if isinstance(r, SEMRule) else delay_info(r.timing)[0]
            for r in self.rules
        ]
        true_delays = [
            d if isinstance(r, SEMRule) else d + 1
            for r, d in zip(self.rules, delays)
        ]
        max_delay = max(true_delays) * 2
        nonzero = [d for d in true_delays if d != 0]
        time_step = reduce(gcd, nonzero) if nonzero else 1
        n_cols = max_delay // time_step + 1

        feat_rows = {f.name: i for i, f in enumerate(self.features)}
        pos = {
            (f, t): (t // time_step, -feat_row)
            for t in range(0, max_delay + 1, time_step)
            for f, feat_row in feat_rows.items()
        }

        fig, ax = plt.subplots(figsize=(2 * n_cols, 1.5 * len(self.features)))
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
        all_rules = self.rules
        cmap = plt.get_cmap('Set1')
        legend_handles = []
        for i, rule in enumerate(all_rules):
            if isinstance(rule, SEMRule):
                delay, is_var, dlabel = rule.lag, False, f"Delay = {rule.lag}"
                src_names = ', '.join(rule.inputs)
                input_features = rule.inputs
            else:
                delay, is_var, dlabel = delay_info(rule.timing)
                src_names = ', '.join(inp.feature for inp in rule.inputs)
                input_features = [inp.feature for inp in rule.inputs]
                if isinstance(rule, AccumulatorRule):
                    input_features = input_features + [rule.target]
            ls = '-' if not is_var else '--'
            color = cmap(i)
            legend_handles.append(
                mlines.Line2D(
                    [], [],
                    color=color,
                    linestyle=ls,
                    linewidth=1.5,
                    marker='>',
                    markersize=5,
                    label=f"{src_names} → {rule.target}  [{dlabel}]"))
            for feat in input_features:
                for t in range(0, max_delay + 1, time_step):
                    t_target = t + delay + 1 if not isinstance(
                        rule, SEMRule) else t + delay
                    if (feat, t) not in pos or (rule.target,
                                                t_target) not in pos:
                        continue
                    x0, y0 = pos[(feat, t)]
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
        ax.set_ylim(-len(self.features) + 0.3, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Temporal Causal Graph", fontsize=12, pad=10)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def generate_signals():
    features = [
        Feature(name='A',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(0.5),
                post_noise=noise_funcs.GaussianNoise(0.5)),
        Feature(name='B',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(0.5),
                post_noise=noise_funcs.GaussianNoise(0.5)),
        Feature(name='C',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(0.5),
                post_noise=noise_funcs.GaussianNoise(0.5)),
        Feature(name='D',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(0.5),
                post_noise=noise_funcs.GaussianNoise(0.5)),
    ]

    rules = [
        SEMRule(inputs=['A', 'B'],
                target='C',
                structural_form=structural_forms.ModulatedInteraction(
                    'A',
                    'B',
                    component_funcs.Sigmoid(k=1.0, x0=5.0),
                    component_funcs.Linear(),
                ),
                lag=3),
        SEMRule(inputs=['C', 'D'],
                target='D',
                structural_form=structural_forms.AdditiveWithInteraction({
                    'C':
                    component_funcs.Threshold(theta=0.5, scale=1.0),
                    'D':
                    component_funcs.Linear(slope=0.9, intercept=0.0)
                }),
                lag=1),
        SEMRule(inputs=['A', 'B'],
                target='A',
                structural_form=structural_forms.SingleIndex({
                    'A': 1,
                    'B': 1
                }, component_funcs.Linear(slope=0.8, intercept=0.0)),
                lag=2),
        # Rule(inputs=[Input('C', transforms.Gradient(10))],
        #      target='D',
        #      collapse=collapse.Max(),
        #      magnitude=magnitude.Fixed(1),
        #      timing=timing.Fixed(10),
        #      shape=shape.Gaussian(std=5)),
        # Rule(inputs=[Input('C', transforms.Value(1))],
        #      target='C',
        #      collapse=collapse.Mean(),
        #      magnitude=magnitude.Fixed(1),
        #      timing=timing.Fixed(0),
        #      shape=shape.Delta()),
        # AccumulatorRule(
        #     inputs=[Input('B', transforms.Threshold(5, 11, binary=False))],
        #     target='A',
        #     magnitude=magnitude.Fixed(0.1),
        #     collapse=collapse.Mean(),
        # )
    ]

    graph = CausalGraph(features, rules, time_steps=10000)

    return graph.sample_graph()
    # graph.plot_temporal_graph(save_path=Path(
    #     "C:\\Users\\php23rjb\\Documents\\PhagoPred\\temp\\test_temporal_graph.png"
    # ))
    # graph.plot_feature_graph(save_path=Path(
    #     "C:\\Users\\php23rjb\\Documents\\PhagoPred\\temp\\test_feature_graph.png"
    # ))
    # # graph.pl
    # for n in range(5):
    #     graph.plot_signals(save_path=Path(
    #         f"C:\\Users\\php23rjb\\Documents\\PhagoPred\\temp\\test_data_{n}.png"
    #     ))


if __name__ == '__main__':
    generate_signals()
