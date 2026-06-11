"""Momentum indicators: RSI, Stochastic, CCI."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from quantlab.indicators._core import (
    Float,
    as_float_array,
    rolling_mean,
    rolling_windows,
    validate_period,
    wilder_smooth,
)


def rsi(close: ArrayLike, period: int = 14) -> Float:
    """Relative Strength Index (Wilder).

    .. math:: RSI = 100 - \\frac{100}{1 + RS},\\quad
              RS = \\frac{smooth(gains)}{smooth(losses)}

    with Wilder smoothing (``alpha = 1/period``). Bounded in [0, 100].
    """
    x = as_float_array(close, "close")
    validate_period(period + 1, len(x))

    delta = np.diff(x, prepend=np.nan)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    gains[0] = losses[0] = np.nan

    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)

    out = np.full(x.shape, np.nan)
    valid = ~np.isnan(avg_gain)
    # avoid 0/0: pure-gain windows -> RSI 100, pure-loss -> 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain[valid] / avg_loss[valid]
        out[valid] = 100.0 - 100.0 / (1.0 + rs)
    out[valid & np.isclose(avg_loss, 0.0) & np.isclose(avg_gain, 0.0)] = 50.0
    out[valid & np.isclose(avg_loss, 0.0) & (avg_gain > 0)] = 100.0
    return out


class StochasticResult(NamedTuple):
    percent_k: Float
    percent_d: Float


def stochastic(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    k_period: int = 14,
    d_period: int = 3,
) -> StochasticResult:
    """Stochastic oscillator.

    ``%K = 100 * (close - LL_n) / (HH_n - LL_n)`` where LL/HH are the
    rolling lowest low / highest high; ``%D = SMA(%K, d_period)``.
    """
    h = as_float_array(high, "high")
    l = as_float_array(low, "low")
    c = as_float_array(close, "close")
    if not (len(h) == len(l) == len(c)):
        raise ValueError("high, low, close must have equal length")
    validate_period(k_period + d_period, len(c))

    hh = np.full(c.shape, np.nan)
    ll = np.full(c.shape, np.nan)
    hh[k_period - 1:] = rolling_windows(h, k_period).max(axis=1)
    ll[k_period - 1:] = rolling_windows(l, k_period).min(axis=1)

    rng = hh - ll
    percent_k = np.full(c.shape, np.nan)
    valid = ~np.isnan(rng)
    with np.errstate(divide="ignore", invalid="ignore"):
        percent_k[valid] = 100.0 * (c[valid] - ll[valid]) / rng[valid]
    percent_k[valid & np.isclose(rng, 0.0)] = 50.0  # flat window

    percent_d = np.full(c.shape, np.nan)
    pk_valid_from = k_period - 1
    percent_d[pk_valid_from:] = rolling_mean(percent_k[pk_valid_from:], d_period)
    return StochasticResult(percent_k, percent_d)


def cci(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 20,
) -> Float:
    """Commodity Channel Index.

    .. math:: CCI = \\frac{TP - SMA(TP, n)}{0.015 \\cdot MD},\\quad
              TP = (H + L + C) / 3

    where MD is the mean absolute deviation of TP from its window mean.
    """
    h = as_float_array(high, "high")
    l = as_float_array(low, "low")
    c = as_float_array(close, "close")
    if not (len(h) == len(l) == len(c)):
        raise ValueError("high, low, close must have equal length")
    validate_period(period, len(c))

    tp = (h + l + c) / 3.0
    windows = rolling_windows(tp, period)
    window_mean = windows.mean(axis=1)
    mean_dev = np.abs(windows - window_mean[:, None]).mean(axis=1)

    out = np.full(c.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[period - 1:] = (tp[period - 1:] - window_mean) / (0.015 * mean_dev)
    return out
