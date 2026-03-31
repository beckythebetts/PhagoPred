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
# from .rule.timing import FixedDelay, VariableDelay, Instantaneous


@dataclass
class Feature:
    """Time varying feature"""
    name: str

    base_func: base_funcs.BaseFunc = field(default_factory=base_funcs.Constant)
    pre_noise: noise_funcs.Noise = field(default=noise_funcs.GaussianNoise(1))
    post_noise: noise_funcs.Noise = field(default=noise_funcs.NoNoise())

    def generate_signal(self, time_steps: int) -> np.ndarray:
        """Generate base signal for node, with base_func and pre_noise"""
        signal = self.base_func.generate(time_steps).astype(float)
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
            for inp in rule.inputs:
                G.add_edge(inp.feature, rule.target, rule=rule)

        return G

    def sample_graph(self) -> dict[str, np.ndarray]:
        """Simulate feature trajectories according to the graph's rules."""
        signals = {
            str(feature): feature.generate_signal(self.time_steps)
            for feature in self.features
        }

        for t in range(self.time_steps):
            for rule in self.rules:
                contribution = rule.apply_step(signals, t)
                signals[rule.target][t] += contribution

        for feature in self.features:
            signal = signals[str(feature)]
            signal += feature.post_noise.sample(self.time_steps).astype(float)
            signals[str(feature)] = signal

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


if __name__ == '__main__':
    # graph = CausalGraph(time_steps=200, features=['A', 'B', 'C'])

    features = [
        Feature(name='A',
                base_func=base_funcs.Ramp(slope=0.1),
                pre_noise=noise_funcs.GaussianNoise(0.5),
                post_noise=noise_funcs.GaussianNoise(0.5)),
        Feature(name='B',
                base_func=base_funcs.Constant(0),
                pre_noise=noise_funcs.GaussianNoise(0.5),
                post_noise=noise_funcs.GaussianNoise(0.5)),
        Feature(name='C',
                base_func=base_funcs.Oscillation(amplitude=1, frequency=0.05),
                pre_noise=noise_funcs.GaussianNoise(0.5),
                post_noise=noise_funcs.GaussianNoise(0.5)),
        Feature(name='D',
                base_func=base_funcs.RandomWalk(step_scale=0.5),
                pre_noise=noise_funcs.GaussianNoise(0.5),
                post_noise=noise_funcs.GaussianNoise(0.5)),
    ]

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
        Rule(
            inputs=[Input('B', transforms.Value(1), thresholds.NoThreshold())],
            target='B',
            timing=timing.FixedDelay(5),
            combination=combinations.Add(),
            shape=shape.Delta(),
            magnitude=magnitude.Fixed(1.0),
        ),
    ]

    graph = CausalGraph(features, rules, time_steps=200)
    for n in range(5):
        graph.plot_signals(
            save_path=Path(f"/home/ubuntu/PhagoPred/temp/test_data_{n}.png"))
