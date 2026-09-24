from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

# NOTE: hazard_nll is imported lazily inside the fit functions to avoid a
# circular import (losses -> configs -> calibration -> probability_calib).


@dataclass
class CalibrationResult:
    """Base class for all fitted calibration results."""
    pass


@dataclass
class TemperatureScalingResult(CalibrationResult):
    T: float


@dataclass
class VectorScalingResult(CalibrationResult):
    w: np.ndarray  # (K,)
    b: np.ndarray  # (K,)


@dataclass
class PlattScalingResult(CalibrationResult):
    a: float
    b: float


@dataclass
class IsotonicScalingResult(CalibrationResult):
    """Fitted monotone (non-parametric) probability map for binary output.

    Stored as the interpolation knots of an sklearn IsotonicRegression so the
    map can be applied (and reconstructed from config) without re-fitting or
    importing sklearn at inference time.
    """
    x_thresholds: np.ndarray  # (K,) ascending raw probabilities (interp knots)
    y_thresholds: np.ndarray  # (K,) calibrated probabilities at each knot

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated ones by piecewise-linear interp.

        Reproduces ``IsotonicRegression.predict`` with ``out_of_bounds='clip'``:
        np.interp clamps to the endpoint values outside the fitted range.
        """
        return np.interp(probs, self.x_thresholds, self.y_thresholds)


#===SURVIVAL===


def fit_temperature_scaling(
    logits: np.ndarray,
    true_bins: np.ndarray,
    event_indicators: np.ndarray,
    from_logits: bool = True,
) -> TemperatureScalingResult:
    """Fit a single temperature T on a calibration set.

    Deep hazard model (from_logits=True): calibrated hazards = sigmoid(logits / T),
    PMF rebuilt via the model's hazard formula. Fit with hazard_nll so calibration
    matches predict_pmf and handles survived-all-bins censoring (t == num_bins).
    Classical model (from_logits=False): p_calibrated = softmax(log_pmf / T).

    Args:
        logits: (N, K) raw pre-softmax model outputs (deep models), or
                PMF probabilities (classical models, set from_logits=False)
        true_bins: (N,) bin index of the event / censoring time
        event_indicators: (N,) 1=uncensored event, 0=censored
        from_logits: if False, converts PMF to log-space first (for RF/classical)
    """
    from PhagoPred.survival_v2.losses.survival_losses import hazard_nll

    if not from_logits:
        logits = np.log(np.clip(logits, 1e-8, 1.0))
    logits_t = torch.tensor(logits, dtype=torch.float32)
    true_bins_t = torch.tensor(true_bins, dtype=torch.long)
    event_indicators_t = torch.tensor(event_indicators, dtype=torch.bool)

    T = torch.ones(1, requires_grad=True)
    optimizer = optim.LBFGS([T], max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / T.clamp(min=1e-4)
        if from_logits:
            # Deep model: `scaled` are hazard logits. Use the hazard NLL so
            # calibration matches the model's sigmoid -> h*S_prev parametrisation
            # and correctly handles survived-all-bins censoring (t == num_bins).
            loss, _, _ = hazard_nll(scaled, true_bins_t, event_indicators_t)
        else:
            # Classical model: inputs are a proper PMF -> softmax recalibration.
            log_pmf = torch.log_softmax(scaled, dim=1)
            loss = _survival_nll(log_pmf, true_bins_t, event_indicators_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return TemperatureScalingResult(T=float(T.detach()))


def apply_temperature_scaling(
    logits: np.ndarray,
    result: TemperatureScalingResult,
) -> np.ndarray:
    """Return calibrated PMF from logits using fitted temperature."""
    scaled = logits / result.T
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def fit_vector_scaling(
    logits: np.ndarray,
    true_bins: np.ndarray,
    event_indicators: np.ndarray,
    from_logits: bool = True,
) -> VectorScalingResult:
    """Fit per-bin weights w and biases b on a calibration set.

    Deep hazard model (from_logits=True): calibrated hazards = sigmoid(logits * w + b),
    PMF rebuilt via the model's hazard formula. Fit with hazard_nll so calibration
    matches predict_pmf and handles survived-all-bins censoring (t == num_bins).
    Classical model (from_logits=False): p_calibrated = softmax(log_pmf * w + b).

    Unlike temperature scaling, the bias term b can shift which bin is
    predicted as most likely, so this can improve confusion matrices.

    Args:
        logits: (N, K) raw pre-softmax model outputs (deep models), or
                PMF probabilities (classical models, set from_logits=False)
        true_bins: (N,) bin index of the event / censoring time
        event_indicators: (N,) 1=uncensored event, 0=censored
        from_logits: if False, converts PMF to log-space first (for RF/classical)
    """
    from PhagoPred.survival_v2.losses.survival_losses import hazard_nll

    if not from_logits:
        logits = np.log(np.clip(logits, 1e-8, 1.0))
    num_bins = logits.shape[1]
    logits_t = torch.tensor(logits, dtype=torch.float32)
    true_bins_t = torch.tensor(true_bins, dtype=torch.long)
    event_indicators_t = torch.tensor(event_indicators, dtype=torch.bool)

    w = torch.ones(num_bins, requires_grad=True)
    b = torch.zeros(num_bins, requires_grad=True)
    optimizer = optim.LBFGS([w, b],
                            max_iter=200,
                            line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        scaled = logits_t * w + b
        if from_logits:
            # Deep model: `scaled` are hazard logits (see fit_temperature_scaling).
            loss, _, _ = hazard_nll(scaled, true_bins_t, event_indicators_t)
        else:
            # Classical model: inputs are a proper PMF -> softmax recalibration.
            log_pmf = torch.log_softmax(scaled, dim=1)
            loss = _survival_nll(log_pmf, true_bins_t, event_indicators_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return VectorScalingResult(w=w.detach().numpy(), b=b.detach().numpy())


def apply_vector_scaling(
    logits: np.ndarray,
    result: VectorScalingResult,
) -> np.ndarray:
    """Return calibrated PMF from logits using fitted w and b."""
    scaled = logits * result.w + result.b
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _survival_nll(
    log_pmf: torch.Tensor,
    true_bins: torch.Tensor,
    event_indicators: torch.Tensor,
) -> torch.Tensor:
    """Survival NLL handling both uncensored and censored samples."""
    uncensored = event_indicators
    censored = ~event_indicators

    # log S(t) = log P(T > t) = log sum_{s > t} p_s
    # computed via reverse cumsum in log space
    log_sf = torch.logcumsumexp(log_pmf.flip(1), dim=1).flip(1)

    nll_unc = -log_pmf[uncensored][
        torch.arange(uncensored.sum()),
        true_bins[uncensored],
    ].mean()

    if censored.any():
        nll_cen = -log_sf[censored][
            torch.arange(censored.sum()),
            true_bins[censored],
        ].mean()
        return nll_unc + nll_cen

    return nll_unc


#===BINARY===


def fit_platt_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    from_logits: bool = True,
) -> PlattScalingResult:
    """Fit Platt scaling (affine transform on logits) for binary output.

    p_calibrated = sigmoid(a * logit + b)

    Trained without pos_weight so the calibrated output approximates the
    true conditional probability p*, undoing the shift from weighted BCE.

    Args:
        logits: (N,) raw pre-sigmoid model outputs (deep models), or
                predicted probabilities (classical models, set from_logits=False)
        labels: (N,) binary labels (0 or 1)
        from_logits: if False, converts probabilities to log-odds first
    """
    if not from_logits:
        logits = np.log(
            np.clip(logits, 1e-8, 1.0) / np.clip(1 - logits, 1e-8, 1.0))
    logits_t = torch.tensor(logits, dtype=torch.float32).squeeze()
    labels_t = torch.tensor(labels, dtype=torch.float32).squeeze()

    a = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = optim.LBFGS([a, b],
                            max_iter=200,
                            line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        log_p = torch.nn.functional.logsigmoid(a * logits_t + b)
        log_1mp = torch.nn.functional.logsigmoid(-(a * logits_t + b))
        # unweighted BCE — rewards the true probability p*, not the weighted optimum
        loss = -(labels_t * log_p + (1 - labels_t) * log_1mp).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return PlattScalingResult(a=float(a.detach()), b=float(b.detach()))


def apply_platt_scaling(
    logits: np.ndarray,
    result: PlattScalingResult,
) -> np.ndarray:
    """Return calibrated probability from logits using fitted a and b."""
    return 1.0 / (1.0 + np.exp(-(result.a * logits + result.b)))


def fit_isotonic_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    from_logits: bool = True,
) -> IsotonicScalingResult:
    """Fit isotonic (monotone, non-parametric) calibration for binary output.

    Unlike Platt scaling (a 2-parameter affine on the logit), isotonic
    regression fits a free monotone step function from the model's score to the
    empirical event rate. It does not compress the tails the way a global affine
    does, so it can correct systematic under-/over-confidence at the extremes —
    at the cost of needing more calibration data to stay stable.

    Args:
        logits: (N,) raw pre-sigmoid model outputs (deep models), or
                predicted probabilities (classical models, set from_logits=False)
        labels: (N,) binary labels (0 or 1)
        from_logits: if True, ``logits`` are squashed with a sigmoid to [0, 1]
                     before fitting; if False they are already probabilities.
                     (Isotonic is monotone-invariant, so this only sets the
                     domain the knots are stored in.)
    """
    from sklearn.isotonic import IsotonicRegression

    scores = np.asarray(logits, dtype=np.float64).squeeze()
    labels = np.asarray(labels, dtype=np.float64).squeeze()
    probs = 1.0 / (1.0 + np.exp(-scores)) if from_logits else scores

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(probs, labels)
    return IsotonicScalingResult(
        x_thresholds=np.asarray(iso.X_thresholds_, dtype=np.float64),
        y_thresholds=np.asarray(iso.y_thresholds_, dtype=np.float64),
    )


def apply_isotonic_scaling(
    logits: np.ndarray,
    result: IsotonicScalingResult,
    from_logits: bool = True,
) -> np.ndarray:
    """Return calibrated probability from logits/probs using fitted isotonic map."""
    scores = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-scores)) if from_logits else scores
    return result.apply(probs)
