from __future__ import annotations

import torch

from PhagoPred.survival_v2.data import (
    BinaryCell,
    SurvivalCell,
    BinaryCellBatch,
    SurvivalCellBatch,
)
from PhagoPred.survival_v2.configs.losses import LossCfg, SurvivalLossCfg, BinaryLossCfg
from PhagoPred.utils.logger import get_logger
from .survival_losses import (
    soft_target_nll,
    negative_log_likelihood,
    hazard_nll,
    prediction_loss,
    ranking_loss_concordance,
    ranking_loss_cif,
)

from .binary_losses import (
    binary_cross_entropy_loss,
    weighted_binary_cross_entropy_loss,
    binary_focal_loss,
)

log = get_logger()


def compute_binary_loss(
    predicted_event_prob: torch.Tensor,
    true_event: torch.Tensor,
    loss_config: BinaryLossCfg,
) -> dict:
    """Compute binary classification loss for event prediction."""
    if loss_config is None:
        raise ValueError("loss_config must be provided")
    if loss_config.loss_type == 'bce':  # default: 'bce'
        # log.info(
        #     f'Caculating BCE\n{predicted_event_prob}\n{torch.unique(true_event)}'
        # )
        loss = binary_cross_entropy_loss(predicted_event_prob, true_event)
        # log.info(f'Calculated loss \n{loss}')
        return {'total': loss, 'BCE': loss}
    if loss_config.loss_type == 'weighted_bce':
        pos_weight = loss_config.pos_weight
        loss = weighted_binary_cross_entropy_loss(predicted_event_prob,
                                                  true_event, pos_weight)
        return {'total': loss, 'Weighted BCE': loss}
    if loss_config.loss_type == 'binary_focal':
        alpha = loss_config.focal_alpha
        gamma = loss_config.focal_gamma
        loss = binary_focal_loss(predicted_event_prob, true_event, alpha,
                                 gamma)
        return {'total': loss, 'Focal': loss}
    else:
        raise ValueError(f'Unknown binary loss type {loss_config.loss_type}')


def compute_survival_loss(
    outputs: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    loss_config: SurvivalLossCfg,
    y_pred: torch.Tensor = None,
    y_true: torch.Tensor = None,
    mask: torch.Tensor = None,
) -> dict:
    """
    Compute combined loss based on configuration.

    Args:
        pmf: (batch_size, num_bins) - predicted PMF
        t: (batch_size,) - event/censoring times
        e: (batch_size,) - event indicators
        y_pred: predicted features (for prediction loss)
        y_true: true features (for prediction loss)
        mask: validity mask
        loss_config: dict with weights for each loss component
                     - nll_type: 'standard' or 'soft_target' (default: 'standard')
                     - soft_target_sigma: spread parameter for soft targets (default: 0.8)

    Returns:
        dict with 'total', 'nll', 'ranking', 'prediction', 'censored', 'uncensored'
    """
    bin_weights = loss_config.bin_weights
    if bin_weights is not None:
        bin_weights = torch.tensor(loss_config.bin_weights,
                                   dtype=torch.float32,
                                   device=outputs.device)
    # NLL (standard or soft target)
    nll_type = loss_config.nll_type
    if nll_type == 'soft_target':
        sigma = loss_config.soft_target_sigma
        nll, censored, uncensored = soft_target_nll(outputs, t, e, sigma=sigma)
    else:
        nll, censored, uncensored = hazard_nll(outputs, t, e, bin_weights)

    # Ranking
    if loss_config.ranking > 0.0:
        ranking_type = loss_config.ranking_type
        if ranking_type == 'concordance':
            ranking = ranking_loss_concordance(outputs, t, e)
        elif ranking_type == 'cif':
            ranking = ranking_loss_cif(outputs, t, e)
        else:
            ranking = torch.tensor(0.0, device=outputs.device)
    else:
        ranking = torch.tensor(0.0, device=outputs.device)

    # Prediction
    if loss_config.prediction > 0.0 and y_pred is not None and y_true is not None:
        pred = prediction_loss(y_pred, y_true, mask)
    else:
        pred = torch.tensor(0.0, device=outputs.device)

    # Total
    total = (loss_config.nll * nll + loss_config.ranking * ranking +
             loss_config.prediction * pred)

    return {
        'total': total,
        'nll': nll,
        'ranking': ranking,
        'prediction': pred,
        'censored': censored,
        'uncensored': uncensored
    }


def compute_loss(outputs: torch.Tensor,
                 batch: BinaryCellBatch | SurvivalCellBatch,
                 loss_cfg: LossCfg,
                 y_pred: torch.Tensor = None) -> dict:
    """Return either binary or survival loss depeniding on batch type."""
    if isinstance(batch, BinaryCell):
        loss_dict = compute_binary_loss(
            outputs[:, 0],
            batch.event,
            loss_cfg,
        )

    elif isinstance(batch, SurvivalCell):
        # pmf = torch.nn.functional.softmax(outputs, dim=-1)
        loss_dict = compute_survival_loss(
            outputs,
            batch.time_to_event_bin,
            batch.event_indicator,
            loss_cfg,
            y_pred,
            batch.mask,
        )

    else:
        raise TypeError(
            'Batches must be either BinaryCellBatch or SurvivalCellBatch')

    return loss_dict


# def compute_loss(
#     pmf: torch.Tensor = None,
#     t: torch.Tensor = None,
#     e: torch.Tensor = None,
#     y_pred: torch.Tensor = None,
#     y_true: torch.Tensor = None,
#     mask: torch.Tensor = None,
#     loss_config: dict = None,
#     binary_label: torch.Tensor = None,
#     outputs: torch.Tensor = None,
# ) -> dict:
#     """Unified loss dispatcher: routes to binary or survival loss based on binary_label."""
#     if binary_label is not None:
#         predictions = outputs.squeeze(
#             -1) if outputs is not None else pmf.squeeze(-1)
#         return compute_binary_loss(predictions, binary_label, loss_config)
#     return compute_survival_loss(pmf, t, e, y_pred, y_true, mask, loss_config)
