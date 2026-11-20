"""Graph-synthetic pipeline: generate datasets -> train models -> GT and
model SHAP -> plots.

Run from anywhere (it moves to the repo root, which all dataset paths are
relative to):

    python -m PhagoPred.survival_v2.run_graph_pipeline                # everything
    python -m PhagoPred.survival_v2.run_graph_pipeline --skip-generate
    python -m PhagoPred.survival_v2.run_graph_pipeline --shap-only <suite dir>

SHAP in parallel (``--workers N``): experiments are grouped by the GT file
they share (the CNN and LSTM of one scenario), the groups are spread over N
single-threaded worker processes, each logging to temp/shap_worker<i>.log,
and the suite plots are made once all workers succeed:

    python -m PhagoPred.survival_v2.run_graph_pipeline --shap-only <suite dir> --workers 4

Single-threaded because the SHAP fills are serial: with OpenBLAS's default
thread pool the spare threads only spin.

Each stage can be rerun safely: existing dataset files are kept (unless
--regenerate), and run_shap reuses samples, GT SHAP and model SHAP already
computed with the same settings, so an interrupted run resumes.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)

# pylint: disable=wrong-import-position
from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.configs.io import load_experiment_cfg
from PhagoPred.survival_v2.data.graph_synthetic.generate_datasets import (
    _SAVE_DIRECTORY, generate_dataset)
from PhagoPred.survival_v2.data.graph_synthetic.scenarios import (
    ALL_CFGS, load_scenario)
from PhagoPred.survival_v2.experiments.run_experiments import run_experiment_suite
from PhagoPred.survival_v2.experiments.plots.plot_experiments import plot_experiment_results
from PhagoPred.survival_v2.experiments.plots.shap_agreement import write_agreement
from PhagoPred.survival_v2.interpret import run_shap
from PhagoPred.survival_v2.interpret.config import INTERPRETS
from PhagoPred.survival_v2.interpret.plots import (plot_dataset_average,
                                                   plot_samples)
import matplotlib.pyplot as plt
from PhagoPred.survival_v2.interpret.run_shap import _shared_path

log = get_logger()

SUITE = 'Graph All Binary'
RESULTS_DIR = Path('PhagoPred/survival_v2/experiments/results')
DATA_SEED = 0


def generate(regenerate: bool = False) -> None:
    """All scenarios in ALL_CFGS (linear, linear_chain, nonlinear_chain,
    multiplicative_hazard, threshold_hazard), num_files shards each."""
    for cfg in ALL_CFGS:
        last = _SAVE_DIRECTORY / cfg.filename / (
            f'{cfg.filename}_{cfg.num_files - 1:02d}.h5')
        if _is_current(last, cfg) and not regenerate:
            log.info(f'{cfg.filename}: already generated, skipping')
            continue
        # Cached samples + GT SHAP are keyed by scenario name only, so they
        # would silently outlive the data they were built from.
        stale = _SAVE_DIRECTORY / cfg.filename / 'shap_samples'
        if stale.exists():
            log.info(f'Removing {stale} (built from the old data)')
            shutil.rmtree(stale)
        log.info(f'Generating {cfg.filename}')
        generate_dataset(cfg, _SAVE_DIRECTORY, seed=DATA_SEED)


def _is_current(path: Path, cfg) -> bool:
    """Exists, has the pre-frame-0 context the GT needs (older files don't),
    and was generated from the same scenario settings and rules as ``cfg``
    (the fitted hazard function isn't compared)."""
    if not path.exists():
        return False
    with h5py.File(path, 'r') as f:
        if 'Context' not in f['Cells']:
            return False
    return load_scenario(path) == cfg


def train(device: str, suite: str = SUITE) -> Path:
    return run_experiment_suite(suite_name=suite,
                                output_dir=RESULTS_DIR,
                                device=device,
                                repeats=1,
                                shap_interpret=False)


def explain(suite_dir: Path,
            interpret: str,
            device: str,
            experiments: list[str] | None = None) -> None:
    """GT SHAP (once per scenario, shared) and model SHAP for every
    experiment, then per-experiment sample figures and the suite plots
    (which include the cross-model SHAP comparisons).

    ``experiments`` (indices or directory names) restricts this to a subset
    and skips the suite plots, so several processes can share a suite: give
    each process whole scenarios (CNN and LSTM of the same scenario share one
    GT file), then run once more without ``experiments`` for the suite plots
    (everything is cached by then)."""
    cfg = INTERPRETS[interpret]
    exp_dirs = sorted(d for d in suite_dir.iterdir()
                      if (d / 'model.pkl').exists())
    if experiments is not None:
        exp_dirs = _select(exp_dirs, experiments)
    for i, exp_dir in enumerate(exp_dirs):
        log.info(f'SHAP {i + 1}/{len(exp_dirs)}: {exp_dir.name}')
        run_shap(exp_dir, cfg, device)
        plot_experiment(exp_dir)
    if experiments is not None:
        log.info('Subset of experiments: skipping suite plots')
        return
    plot_suite(suite_dir)


def plot_suite(suite_dir: Path) -> None:
    """Suite plots, plus the GT vs model SHAP agreement table."""
    plot_experiment_results(suite_dir, ignore_params=['feature_combo'])
    write_agreement(suite_dir)


def plot_experiment(exp_dir: Path) -> None:
    """``<exp_dir>/shap_plots``: the first 5 samples, and the average over
    all samples in shap.h5 (relative, and raw mean |SHAP|)."""
    out = exp_dir / 'shap_plots'
    plot_samples(exp_dir / 'shap.h5', out, num_plot_samples=5,
                 title_prefix=exp_dir.name)
    for normalise, name in ((True, 'average.png'), (False, 'average_raw.png')):
        fig = plot_dataset_average(exp_dir / 'shap.h5', exp_dir.name,
                                   normalise=normalise)
        fig.savefig(out / name, dpi=150)
        plt.close(fig)


def plot_all(suite_dir: Path) -> None:
    """Every plot, from the shap.h5 files already written (no SHAP runs)."""
    for exp_dir in sorted(d for d in suite_dir.iterdir()
                          if (d / 'shap.h5').exists()):
        log.info(f'Plotting {exp_dir.name}')
        plot_experiment(exp_dir)
    plot_suite(suite_dir)


def explain_parallel(suite_dir: Path, interpret: str, device: str,
                     workers: int) -> None:
    """``explain`` split over ``workers`` single-threaded subprocesses (see
    the module docstring), then the suite plots."""
    exp_dirs = sorted(d for d in suite_dir.iterdir()
                      if (d / 'model.pkl').exists())
    plan = _plan_workers(exp_dirs, workers)
    env = dict(os.environ,
               OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    procs = []
    try:
        for i, names in enumerate(plan):
            log_path = Path('temp') / f'shap_worker{i}.log'
            out_path = log_path.with_suffix('.out')  # stdout/stderr
            cmd = [
                sys.executable, '-m', 'PhagoPred.survival_v2.run_graph_pipeline',
                '--shap-only', str(suite_dir), '--interpret', interpret,
                '--device', device, '--experiments', *names
            ]
            with open(out_path, 'w') as out:
                p = subprocess.Popen(cmd,
                                     env=dict(env, PHAGOPRED_LOG=str(log_path)),
                                     stdout=out,
                                     stderr=subprocess.STDOUT)
            procs.append((i, names, log_path, p))
            log.info(f'worker {i} (pid {p.pid}): {" ".join(names)} '
                     f'-> {log_path}')
        failed = []
        pending = list(procs)
        while pending:
            time.sleep(10)
            for item in [x for x in pending if x[3].poll() is not None]:
                pending.remove(item)
                i, names, log_path, p = item
                if p.returncode:
                    failed.append(item)
                    log.error(f'worker {i} ({" ".join(names)}) failed with '
                              f'exit code {p.returncode}; see {log_path} '
                              f'and {log_path.with_suffix(".out")}')
                else:
                    log.info(f'worker {i} ({" ".join(names)}) done')
    except BaseException:
        for *_, p in procs:
            if p.poll() is None:
                p.terminate()
        raise
    if failed:
        raise RuntimeError(
            f'{len(failed)} SHAP worker(s) failed (logs: '
            f'{", ".join(str(x[2]) for x in failed)}); rerun to resume')
    plot_suite(suite_dir)


def _plan_workers(exp_dirs: list[Path], workers: int) -> list[list[str]]:
    """Experiment names per worker. Experiments sharing a GT file stay in one
    worker (one writer per file). Groups go biggest-first to the least
    loaded worker; a group whose GT file doesn't exist yet counts extra."""
    groups = defaultdict(list)
    for d in exp_dirs:
        dcfg = load_experiment_cfg(d / 'config.json').dataset
        groups[_shared_path(Path(dcfg.shap_paths[0]),
                            dcfg.prediction_target)].append(d.name)
    cost = {gt: len(names) + (0 if gt.exists() else 2)
            for gt, names in groups.items()}
    plan = [[] for _ in range(min(workers, len(groups)))]
    load = [0] * len(plan)
    for gt in sorted(groups, key=lambda g: -cost[g]):
        i = load.index(min(load))
        plan[i] += groups[gt]
        load[i] += cost[gt]
    return plan


def _select(exp_dirs: list[Path], wanted: list[str]) -> list[Path]:
    """``wanted`` as indices (``2``) or names (``experiment_02``)."""
    by_name = {d.name: d for d in exp_dirs}
    out = []
    for w in wanted:
        name = f'experiment_{int(w):02d}' if w.isdigit() else w
        if name not in by_name:
            raise ValueError(f'no trained experiment {name!r}; have '
                             f'{sorted(by_name)}')
        out.append(by_name[name])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--skip-generate', action='store_true')
    parser.add_argument('--suite',
                        default=SUITE,
                        help="experiment suite to train, e.g. 'Graph All "
                        "Hazard' for the hazard regression target")
    parser.add_argument('--regenerate',
                        action='store_true',
                        help='overwrite existing dataset files')
    parser.add_argument('--shap-only',
                        type=Path,
                        metavar='SUITE_DIR',
                        help='skip generation and training; explain the '
                        'models in an existing suite directory')
    parser.add_argument('--experiments',
                        nargs='+',
                        metavar='EXP',
                        help='with --shap-only: only these experiments '
                        '(indices or names, e.g. 2 7); skips suite plots')
    parser.add_argument('--plots-only',
                        type=Path,
                        metavar='SUITE_DIR',
                        help='only redo the per-experiment and suite plots '
                        'from the shap.h5 files already in SUITE_DIR')
    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help='run SHAP in this many single-threaded worker '
                        'processes (grouped by scenario)')
    parser.add_argument('--interpret',
                        default='Standard',
                        choices=sorted(INTERPRETS),
                        help='interpret.config.INTERPRETS entry')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.experiments and args.shap_only is None:
        parser.error('--experiments needs --shap-only')
    if args.plots_only is not None:
        plot_all(args.plots_only)
        return
    if args.experiments and args.workers > 1:
        parser.error('--experiments and --workers are exclusive')

    def _explain(suite_dir: Path) -> None:
        if args.workers > 1:
            explain_parallel(suite_dir, args.interpret, args.device,
                             args.workers)
        else:
            explain(suite_dir, args.interpret, args.device, args.experiments)

    if args.shap_only is not None:
        _explain(args.shap_only)
        return
    if not args.skip_generate:
        generate(args.regenerate)
    suite_dir = train(args.device, args.suite)
    log.info(f'Trained suite in {suite_dir}')
    _explain(suite_dir)


if __name__ == '__main__':
    main()
