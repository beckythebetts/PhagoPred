import torch


def binary_cross_entropy_loss(predictions: torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss for binary survival prediction."""
    assert predictions.shape == targets.shape
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, targets.float())


def weighted_binary_cross_entropy_loss(predictions: torch.Tensor,
                                       targets: torch.Tensor,
                                       pos_weight: float) -> torch.Tensor:
    """Weighted binary cross-entropy loss to handle class imbalance."""
    assert predictions.shape == targets.shape
    return torch.nn.functional.binary_cross_entropy_with_logits(
        predictions,
        targets.float(),
        pos_weight=torch.tensor(pos_weight, device=predictions.device))


def binary_focal_loss(predictions: torch.Tensor,
                      targets: torch.Tensor,
                      alpha: float = 0.25,
                      gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for binary classification."""
    assert predictions.shape == targets.shape
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        predictions, targets.float(), reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt)**gamma * bce_loss
    return focal_loss.mean()
