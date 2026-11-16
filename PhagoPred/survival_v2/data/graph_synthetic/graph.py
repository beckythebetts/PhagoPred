from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import graphviz
from pyvis.network import Network

from .noise_funcs import Noise
from .rules import Rule

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
    pre_noise: Noise = field(default_factory=lambda: Noise.gaussian(1.0))
    post_noise: Noise = field(default_factory=Noise.none)

    def generate_signal(self,
                        time_steps: int,
                        n: int = 1,
                        rng=None) -> np.ndarray:
        """Generate base signal (noise)."""
        signal = self.pre_noise.sample(time_steps * n, rng).astype(float)
        if n > 1:
            signal = signal = signal.reshape(time_steps, n)
        return signal


class CausalGraph:
    """Directed graph of features with rules governing their interactions."""

    def __init__(self,
                 features: list[Feature],
                 rules: list[Rule],
                 burn_in: int = 0,
                 death_burn_in: int = 0,
                 hazard_from_frame_0: bool = False):
        """
        Args:
            burn_in: frames simulated (and discarded) before frame 0, so every
                trajectory starts from the process's long-run behaviour rather
                than from zero.
            death_burn_in: the last ``death_burn_in`` burn-in frames run the
                death process; cells dying there are rejected, so frame 0 only
                holds cells that were alive when observation started.
            hazard_from_frame_0: rules targeting Hazard only act from frame 0,
                and their inputs from before frame 0 count as zero. Hazard is
                then a function of observed frames only (features still get
                the full burn-in), so nothing about it is hidden in the
                pre-observation past. Incompatible with ``death_burn_in``.
        """
        self.features = features
        self.rules = rules
        if death_burn_in > burn_in:
            raise ValueError(
                f'death_burn_in ({death_burn_in}) > burn_in ({burn_in})')
        if hazard_from_frame_0 and death_burn_in > 0:
            raise ValueError(
                'hazard_from_frame_0 leaves no hazard before frame 0, so '
                'death_burn_in must be 0')
        self.burn_in = burn_in
        self.death_burn_in = death_burn_in
        self.hazard_from_frame_0 = hazard_from_frame_0

    @property
    def max_lag(self) -> int:
        return max(rule.max_lag for rule in self.rules)

    @property
    def hazard_rules(self) -> list[Rule]:
        return [rule for rule in self.rules if rule.target == 'Hazard']

    def apply_rules(self,
                    signals: dict[str, np.ndarray],
                    t: int,
                    frame_0: int = 0,
                    rules: list[Rule] | None = None) -> dict:
        """Apply ``rules`` (default: all) at step ``t``.

        ``frame_0`` is the index of observation frame 0 in ``signals``; with
        ``hazard_from_frame_0``, Hazard rules start there (see __init__). Every
        caller that steps the rules should go through this method so the
        cut-off is applied consistently."""
        for rule in (self.rules if rules is None else rules):
            start = (frame_0 if self.hazard_from_frame_0
                     and rule.target == 'Hazard' else None)
            rule.apply_step(signals, t, start)
        return signals

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
            n = min(
                int(np.ceil((n_cells - n_kept) / accept_rate * 1.1)) + 1,
                max_batch_cells)
            n_simulated += n
            noise = {
                f.name: np.stack([f.generate_signal(T) for _ in range(n)],
                                 axis=1)
                for f in self.features
            }
            sig = {name: arr.copy() for name, arr in noise.items()}
            for t in range(T):
                self.apply_rules(sig, t, frame_0=burn_in)

            keep = np.ones(n, dtype=bool)
            if condition:
                h = np.clip(
                    hazard_func(sig['Hazard'][burn_in -
                                              death_burn_in:burn_in]), 0.0,
                    1.0)
                keep = np.random.rand(n) < np.prod(1.0 - h, axis=0)
                accept_rate = max(keep.mean(), 0.01)

            kept_sig.append({k: v[start:, keep] for k, v in sig.items()})
            kept_noise.append({k: v[start:, keep] for k, v in noise.items()})
            n_kept += int(keep.sum())

        signals = {
            f.name:
            np.concatenate([b[f.name] for b in kept_sig], axis=1)[:, :n_cells]
            for f in self.features
        }
        if post_noise:
            signals = self.observe(signals)
        if not return_noise:
            return signals
        noise = {
            f.name:
            np.concatenate([b[f.name]
                            for b in kept_noise], axis=1)[:, :n_cells]
            for f in self.features
        }
        return signals, noise

    def observe(self, signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return noisy copy of clean signals"""
        return {
            f.name:
            signals[f.name] + np.stack([
                f.post_noise.sample(len(signals[f.name]))
                for _ in range(signals[f.name].shape[1])
            ],
                                       axis=1)
            for f in self.features
        }

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
