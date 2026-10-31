#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical convergence analysis of a validation-loss series.

Used by the training report (recommendations) and by early stopping in every
trainer, so both apply the same definition of a meaningful improvement.

Method
------
1. Trend. Two candidate trends are fitted to validation loss vs. epoch:
     - decaying curve:  loss = plateau + drop * exp(-rate * epoch)
     - straight line:   loss = intercept + slope * epoch
   The candidate with the lower AICc (Akaike Information Criterion, corrected
   for small samples) is selected. AICc counts the noise variance as a fitted
   term, so the minimum number of validation epochs for each candidate is
   derived from its term count (see min_epochs_for); nothing is hard-coded.

2. Noise. The residual standard error of the selected trend (scatter of the
   validation loss around the fitted trend, all epochs).

3. Meaningful improvement. A decrease larger than k * noise, where k is the
   site_config setting "improvement_threshold_sigmas".

4. Projection. The number of further epochs until the projected decrease in
   validation loss exceeds k * noise, or a statement that the curve has
   levelled off (remaining possible decrease below k * noise). For a decaying
   curve, also the number of further epochs until it levels off by that same
   test; a straight line has no level-off point.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit


# Number of fitted trend terms for each candidate (the noise variance is added
# separately when counting AICc parameters).
LINE_TERMS = 2      # intercept, slope
DECAY_TERMS = 3     # plateau, drop, rate

TREND_LINE = "straight line"
TREND_DECAY = "decaying curve"


def min_epochs_for(trend_terms: int) -> int:
    """
    Minimum validation epochs for a trend with `trend_terms` terms.

    AICc uses k = trend_terms + 1 (the noise variance is also estimated) and
    its correction term divides by (n - k - 1), which must be at least 1.
    """
    k = trend_terms + 1
    return k + 2


MIN_EPOCHS_LINE = min_epochs_for(LINE_TERMS)     # 5
MIN_EPOCHS_DECAY = min_epochs_for(DECAY_TERMS)   # 6


# ======================================================================================================================
# ======================================================================================================================
@dataclass
class TrendFit:
    kind: str                     # TREND_LINE or TREND_DECAY
    params: tuple                 # (intercept, slope) or (plateau, drop, rate)
    rss: float                    # residual sum of squares
    n: int                        # number of epochs fitted
    trend_terms: int
    aicc: float

    def predict(self, epoch: float) -> float:
        if self.kind == TREND_LINE:
            intercept, slope = self.params
            return intercept + slope * epoch
        plateau, drop, rate = self.params
        return plateau + drop * math.exp(-rate * epoch)

    @property
    def noise(self) -> float:
        """Residual standard error of the trend (scatter around the fitted trend)."""
        return math.sqrt(self.rss / (self.n - self.trend_terms))


@dataclass
class ConvergenceResult:
    enough_epochs: bool                   # False when fewer than MIN_EPOCHS_LINE epochs
    trend: Optional[TrendFit] = None
    noise: Optional[float] = None         # residual standard error of the selected trend
    threshold: Optional[float] = None     # k * noise
    k_sigmas: Optional[float] = None
    levelled_off: bool = False            # remaining possible decrease is below threshold
    not_decreasing: bool = False          # selected trend is flat or rising
    epochs_to_meaningful_gain: Optional[int] = None
    projected_gain: Optional[float] = None  # projected decrease in val loss over that many epochs
    remaining_gain: Optional[float] = None  # decaying curve: largest decrease still possible
    epochs_to_level_off: Optional[int] = None  # decaying curve: epochs until remaining_gain <= threshold


# ======================================================================================================================
# ======================================================================================================================
def _aicc(rss: float, n: int, trend_terms: int) -> float:
    k = trend_terms + 1
    # A perfect fit has no residual; floor at the smallest positive float so the log is defined.
    rss = max(rss, np.finfo(float).tiny)
    aic = n * math.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def _fit_line(epochs: np.ndarray, losses: np.ndarray) -> Optional[TrendFit]:
    n = len(losses)
    if n < MIN_EPOCHS_LINE:
        return None
    slope, intercept = np.polyfit(epochs, losses, 1)
    resid = losses - (intercept + slope * epochs)
    rss = float(np.sum(resid ** 2))
    return TrendFit(TREND_LINE, (float(intercept), float(slope)), rss, n, LINE_TERMS,
                    _aicc(rss, n, LINE_TERMS))


def _decay(epoch, plateau, drop, rate):
    return plateau + drop * np.exp(-rate * epoch)


def _fit_decay(epochs: np.ndarray, losses: np.ndarray) -> Optional[TrendFit]:
    n = len(losses)
    if n < MIN_EPOCHS_DECAY:
        return None
    # Starting values come from the data: plateau at the lowest loss, drop sized
    # to the first epoch's loss, rate spanning the observed epochs.
    span = float(epochs[-1] - epochs[0]) or 1.0
    rate0 = 1.0 / span
    plateau0 = float(np.min(losses))
    drop0 = max(float(losses[0] - plateau0), np.finfo(float).eps) * math.exp(rate0 * float(epochs[0]))
    try:
        params, _ = curve_fit(
            _decay, epochs, losses, p0=(plateau0, drop0, rate0),
            bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        )
    except (RuntimeError, ValueError):
        return None
    plateau, drop, rate = (float(p) for p in params)
    if not all(math.isfinite(p) for p in (plateau, drop, rate)) or drop <= 0 or rate <= 0:
        return None     # no decay in the data; the straight line covers flat or rising loss
    resid = losses - _decay(epochs, plateau, drop, rate)
    rss = float(np.sum(resid ** 2))
    return TrendFit(TREND_DECAY, (plateau, drop, rate), rss, n, DECAY_TERMS,
                    _aicc(rss, n, DECAY_TERMS))


def fit_trend(epochs: Sequence[float], losses: Sequence[float]) -> Optional[TrendFit]:
    """Fit both candidates and return the one with the lower AICc (None if too few epochs)."""
    e = np.asarray(epochs, dtype=float)
    y = np.asarray(losses, dtype=float)
    candidates = [f for f in (_fit_line(e, y), _fit_decay(e, y)) if f is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda f: f.aicc)


# ======================================================================================================================
# ======================================================================================================================
def analyze_convergence(epochs: Sequence[float], losses: Sequence[float],
                        k_sigmas: float) -> ConvergenceResult:
    """
    Fit the trend, estimate noise, and project the epochs needed for the next
    meaningful decrease in validation loss.
    """
    trend = fit_trend(epochs, losses)
    if trend is None:
        return ConvergenceResult(enough_epochs=False, k_sigmas=k_sigmas)

    noise = trend.noise
    threshold = k_sigmas * noise
    result = ConvergenceResult(enough_epochs=True, trend=trend, noise=noise,
                               threshold=threshold, k_sigmas=k_sigmas)
    last_epoch = float(epochs[-1])

    if trend.kind == TREND_LINE:
        _, slope = trend.params
        if slope >= 0:
            result.not_decreasing = True
            return result
        # Smallest whole number of epochs whose projected decrease exceeds the threshold.
        steps = math.floor(threshold / -slope) + 1
        result.epochs_to_meaningful_gain = steps
        result.projected_gain = -slope * steps
        return result

    plateau, drop, rate = trend.params
    remaining = trend.predict(last_epoch) - plateau     # largest decrease still possible
    result.remaining_gain = remaining
    if remaining <= threshold:
        result.levelled_off = True
        return result
    # remaining * (1 - exp(-rate * steps)) > threshold
    steps = math.floor(-math.log(1.0 - threshold / remaining) / rate) + 1
    result.epochs_to_meaningful_gain = steps
    result.projected_gain = trend.predict(last_epoch) - trend.predict(last_epoch + steps)
    # remaining * exp(-rate * steps) <= threshold
    result.epochs_to_level_off = math.ceil(math.log(remaining / threshold) / rate)
    return result


# ======================================================================================================================
# ======================================================================================================================
class ImprovementTracker:
    """
    Early-stopping bookkeeping shared by all trainers.

    A validation epoch counts as an improvement when its loss is below the
    reference loss (the loss at the last improvement) by more than
    k * noise, with noise taken from the trend fitted to all validation
    epochs so far. Until there are enough epochs to fit a trend, any decrease
    counts, because the noise cannot yet be estimated.

    Checkpoint saving is separate: trainers still save every new lowest
    validation loss.
    """

    def __init__(self, k_sigmas: float):
        self.k_sigmas = float(k_sigmas)
        self.epochs: List[float] = []
        self.losses: List[float] = []
        self.reference_loss: Optional[float] = None
        self.epochs_without_improvement = 0
        self.last_threshold: Optional[float] = None

    def update(self, epoch: float, val_loss: float) -> bool:
        """Record one validation epoch. Returns True if it is a meaningful improvement."""
        self.epochs.append(float(epoch))
        self.losses.append(float(val_loss))

        trend = fit_trend(self.epochs, self.losses)
        self.last_threshold = self.k_sigmas * trend.noise if trend is not None else None

        if self.reference_loss is None:
            improved = True
        elif self.last_threshold is None:
            improved = val_loss < self.reference_loss
        else:
            improved = (self.reference_loss - val_loss) > self.last_threshold

        if improved:
            self.reference_loss = float(val_loss)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved
