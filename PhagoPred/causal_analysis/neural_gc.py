from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, str(Path(__file__).parent / 'Neural-GC'))
from models.cmlp import cMLP, train_model_gista

from .main import load_signals, normalise_signals

SAVE_DIR = Path('temp')


def prepare_tensor(signals: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (samples, frames, features) numpy array to float tensor."""
    return torch.tensor(signals, dtype=torch.float32, device=device)


def train_neural_gc(
    signals: np.ndarray,
    lag: int = 5,
    hidden: list[int] = [64, 64],
    lam: float = 0.01,
    lam_ridge: float = 1e-2,
    lr: float = 1e-3,
    max_iter: int = 3000,
    penalty: str = 'GL',
    device: torch.device = None,
) -> tuple[cMLP, list]:
    """
    Train a cMLP neural Granger causality model.

    Args:
        signals: array of shape (samples, frames, features)
        lag: number of past frames to condition on
        hidden: hidden layer sizes
        lam: group lasso regularisation strength (controls sparsity)
        lam_ridge: ridge regularisation on output layers
        lr: initial learning rate for GISTA
        max_iter: maximum GISTA iterations
        penalty: 'GL' (group lasso), 'GSGL', or 'H' (hierarchical)
        device: torch device

    Returns:
        trained cMLP model and training loss list
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    num_features = signals.shape[2]
    X = prepare_tensor(signals, device)  # (samples, frames, features)

    model = cMLP(num_series=num_features, lag=lag, hidden=hidden).to(device)

    train_loss, train_mse = train_model_gista(
        cmlp=model,
        X=X,
        lam=lam,
        lam_ridge=lam_ridge,
        lr=lr,
        penalty=penalty,
        max_iter=max_iter,
        verbose=1,
    )

    return model, train_loss


def plot_gc_matrix(
    gc_matrix: np.ndarray,
    feature_names: list[str],
    save_path: Path = SAVE_DIR / 'neural_gc_matrix.png',
) -> None:
    """
    Plot the Granger causality matrix as a heatmap.
    Entry (i, j) = variable j is Granger causal of variable i.
    """
    fig, ax = plt.subplots(figsize=(len(feature_names) + 1,
                                    len(feature_names) + 1))
    im = ax.imshow(gc_matrix, cmap='Reds', vmin=0)
    plt.colorbar(im, ax=ax, label='GC strength')
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel('Cause')
    ax.set_ylabel('Effect')
    ax.set_title('Neural Granger Causality\n(row i ← column j)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'GC matrix saved to {save_path}')


def plot_gc_by_lag(
    gc_by_lag: np.ndarray,
    feature_names: list[str],
    save_path: Path = SAVE_DIR / 'neural_gc_by_lag.png',
) -> None:
    """
    Plot GC strength vs lag for every (cause, effect) pair.

    Args:
        gc_by_lag: array of shape (features, features, lag) where
                   gc_by_lag[i, j, k] = strength of j -> i at lag k+1.
    """
    n = len(feature_names)
    lags = np.arange(1, gc_by_lag.shape[2] + 1)
    vmax = gc_by_lag.max()

    fig, axes = plt.subplots(n,
                             n,
                             figsize=(2.5 * n, 2 * n),
                             sharex=True,
                             sharey=True)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.bar(lags, gc_by_lag[i, j], color='crimson', alpha=0.8)
            ax.set_ylim(0, vmax * 1.1 if vmax > 0 else 1)
            if i == n - 1:
                ax.set_xlabel('Lag', fontsize=6)
            if j == 0:
                ax.set_ylabel(feature_names[i],
                              fontsize=6,
                              rotation=45,
                              ha='right')
            if i == 0:
                ax.set_title(feature_names[j],
                             fontsize=6,
                             rotation=45,
                             ha='left')
            ax.tick_params(labelsize=5)

    fig.suptitle('GC strength by lag\n(column j → row i)', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'GC by lag plot saved to {save_path}')


def plot_training_loss(
    loss_list: list,
    save_path: Path = SAVE_DIR / 'neural_gc_loss.png',
) -> None:
    fig, ax = plt.subplots()
    ax.plot([l.item() if hasattr(l, 'item') else l for l in loss_list])
    ax.set_xlabel('Check interval')
    ax.set_ylabel('Loss')
    ax.set_title('Neural GC training loss')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    hdf5_paths = [
        h for h in (Path('~/thor_server').expanduser() / '24_02').glob('*.h5')
        if h.stem in ('C', 'D', 'E', 'F')
    ]
    features = [
        'Total Fluorescence',
        'Alive Phagocytes within 250 pixels',
        'Fluorescence Distance Mean',
    ]

    signals, feature_names = load_signals(hdf5_paths)
    print(
        f'{signals.shape[0]} samples, {signals.shape[1]} frames, {signals.shape[2]} features'
    )

    # Normalise across all samples (same as PCMCI pipeline)
    signals = normalise_signals(signals)

    model, loss_list = train_neural_gc(
        signals=signals,
        lag=20,
        hidden=[64, 64],
        lam=0.5,
        lam_ridge=1e-2,
        lr=1e-3,
        max_iter=3000,
    )

    SAVE_DIR.mkdir(exist_ok=True)
    plot_training_loss(loss_list)

    # GC matrix: continuous strengths (collapsed across lags)
    gc_weights = model.GC(threshold=False,
                          ignore_lag=True).detach().cpu().numpy()
    plot_gc_matrix(gc_weights, feature_names,
                   SAVE_DIR / 'neural_gc_matrix_weights.png')

    # GC strength broken down by lag
    gc_by_lag = model.GC(threshold=False,
                         ignore_lag=False).detach().cpu().numpy()
    plot_gc_by_lag(gc_by_lag, feature_names)

    # GC matrix: thresholded (binary)
    gc_binary = model.GC(threshold=True).detach().cpu().numpy()
    plot_gc_matrix(gc_binary, feature_names,
                   SAVE_DIR / 'neural_gc_matrix_binary.png')

    print('\nGranger Causality matrix (thresholded):')
    print('Causes →', feature_names)
    for i, name in enumerate(feature_names):
        causes = [
            feature_names[j] for j in range(len(feature_names))
            if gc_binary[i, j]
        ]
        print(f'  {name} ← {causes if causes else "nothing"}')


if __name__ == '__main__':
    main()
