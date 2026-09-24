from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from PhagoPred.survival_v2.data.graph_synthetic.scenarios import ALL_CFGS
from PhagoPred.survival_v2.data.graph_synthetic.analytic_estimates import (
    generate_sample_with_feature_importance,
    generate_sample_with_feature_importance_batched,
    sampleWithImportances,
    importances,
)

# def plot_importances_on_ax(results: sampleWithImportances, ax: plt.Axes, type: Li) -> None:


def main():
    # np.random.seed(1)
    cfg = ALL_CFGS[
        0]  # base_linear: A(lag25)+B(lag50)+C(lag100) -> Hazard; D inert
    graph = cfg.graph
    max_len = cfg.num_frames
    horizon = 100
    bins = np.array([0, 20, 40, 60, 80, 100])

    # Calibrate the hazard the same way datasets are generated: iteratively fit
    # the OffsetSigmoid delta to the scenario's target death fraction, rather
    # than hand-tuning a percentile of the raw hazard signal.
    cfg.calibrate_hazard(target_death_fraction=cfg.target_death_fraction)
    hz = cfg.hazard_calibration_func
    print(f"using delta={hz.delta:.3f}")

    n_samples = 10
    results: list[sampleWithImportances] = []
    while len(results) < n_samples:
        print(f'Attempting to genertae sample {len(results)+1}...')
        r = generate_sample_with_feature_importance_batched(
            graph,
            hz,
            horizon=horizon,
            max_sequence_length=max_len,
            min_sequence_length=120,
            num_permutations=500,
            hazard_bins=bins,
        )
        if r is not None:
            results.append(r)

    feature_names = list(results[0].base_signals.keys())
    n_feats = len(feature_names)

    fig, axes = plt.subplots(n_samples,
                             1,
                             figsize=(11, 2.4 * n_samples),
                             squeeze=False)
    for ax, r in zip(axes[:, 0], results):
        # cdf_imp = r.sample_importances.cdf[
        #     :,
        #     -1:,
        # ].squeeze()  #[features, horizon time steps, lf times teps]
        imp = r.sample_importances
        lf = r.landmark_frame
        # prim = r['primary_quantity']
        # imp = r['importance_scalar'][prim]  # (n_feats, lf)
        # lf = r['landmark_frame']
        # vmax = np.quantile(imp, 0.99) or imp.max() or 1.0
        im = ax.imshow(
            imp,
            aspect='auto',
            origin='lower',
            cmap='magma',
            extent=[0, lf, -0.5, n_feats - 0.5],
            #    vmin=0.0,
            #    vmax=vmax
        )
        ax.set_yticks(range(n_feats))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('time step (frame)')
        df = r.death_frame
        ax.set_title(
            f"lf={lf}  death={'censored' if df is None else int(df)}  ",
            # f"(importance on {prim}, mean over bins)",
            fontsize=9)
        ax.axvline(lf, color='cyan', ls='--', lw=1)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    fig.suptitle(
        f"Ground-truth temporal-feature importance — scenario '{cfg.filename}'",
        fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = Path(__file__).parent / 'temp'
    out.mkdir(exist_ok=True)
    path = out / 'ground_truth_importance_heatmap.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"saved {path}")


if __name__ == '__main__':
    main()
