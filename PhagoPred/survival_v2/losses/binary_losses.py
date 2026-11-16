from __future__ import annotations

import torch


def _reduce(loss: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over all elements, or over masked-in elements if mask is given."""
    if mask is None:
        return loss.mean()
    mask = mask.float()
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def binary_cross_entropy_loss(predictions: torch.Tensor,
                              targets: torch.Tensor,
                              mask: torch.Tensor = None) -> torch.Tensor:
    """Binary cross-entropy loss for binary survival prediction."""
    assert predictions.shape == targets.shape, f'{predictions.shape}, {targets.shape}'

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, targets.float(), reduction='none')
    return _reduce(loss, mask)


def weighted_binary_cross_entropy_loss(predictions: torch.Tensor,
                                       targets: torch.Tensor,
                                       pos_weight: float,
                                       mask: torch.Tensor = None) -> torch.Tensor:
    """Weighted binary cross-entropy loss to handle class imbalance."""
    assert predictions.shape == targets.shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predictions,
        targets.float(),
        pos_weight=torch.tensor(pos_weight, device=predictions.device),
        reduction='none')
    return _reduce(loss, mask)


def binary_focal_loss(predictions: torch.Tensor,
                      targets: torch.Tensor,
                      alpha: float = 0.25,
                      gamma: float = 2.0,
                      mask: torch.Tensor = None) -> torch.Tensor:
    """Focal loss for binary classification."""
    assert predictions.shape == targets.shape
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, targets.float(), reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt)**gamma * bce_loss
    return _reduce(focal_loss, mask)
