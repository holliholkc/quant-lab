"""Trend indicators: SMA, EMA, MACD, ADX."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from quantlab.indicators._core import (
    Float,
    as_float_array,
    ema_recursive,
    rolling_mean,
    validate_period,
    wilder_smooth,
)


def sma(close: ArrayLike, period: int = 20) -> Float:
    """Simple Moving Average.

    .. math:: SMA_t = \\frac{1}{n} \\sum_{i=t-n+1}^{t} P_i

    First ``period - 1`` values are NaN.
    """
    x = as_float_array(close, "close")
    validate_period(period, len(x))
    return rolling_mean(x, period)


def ema(close: ArrayLike, span: int = 20) -> Float:
    """Exponential Moving Average with ``alpha = 2 / (span + 1)``,
    seeded with the SMA of the first ``span`` values.
    """
    x = as_float_array(close, "close")
    validate_period(span, len(x))
    return ema_recursive(x, span)


class MACDResult(NamedTuple):
    macd: Float
    signal: Float
    histogram: Float


def macd(
    close: ArrayLike,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """Moving Average Convergence/Divergence.

    ``macd = EMA(fast) - EMA(slow)``; ``signal = EMA(macd, signal)``;
    ``histogram = macd - signal``.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    x = as_float_array(close, "close")
    validate_period(slow + signal, len(x))

    macd_line = ema_recursive(x, fast) - ema_recursive(x, slow)
    # signal EMA starts after the macd warm-up
    valid_from = slow - 1
    signal_line = np.full(x.shape, np.nan)
    signal_line[valid_from:] = ema_recursive(macd_line[valid_from:], signal)
    return MACDResult(macd_line, signal_line, macd_line - signal_line)


class ADXResult(NamedTuple):
    adx: Float
    plus_di: Float
    minus_di: Float


def adx(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
) -> ADXResult:
    """Average Directional Index (Wilder).

    Directional movements::

        +DM = high_t - high_{t-1}   if it exceeds  low_{t-1} - low_t  and > 0
        -DM = low_{t-1} - low_t     if it exceeds  high_t - high_{t-1} and > 0

    ``+DI = 100 * smooth(+DM) / smooth(TR)`` (same for −DI),
    ``DX = 100 * |+DI − −DI| / (+DI + −DI)``, ``ADX = smooth(DX)``,
    all smoothing per Wilder (``alpha = 1/period``).
    """
    h = as_float_array(high, "high")
    l = as_float_array(low, "low")
    c = as_float_array(close, "close")
    if not (len(h) == len(l) == len(c)):
        raise ValueError("high, low, close must have equal length")
    validate_period(2 * period, len(c))

    up = np.diff(h, prepend=np.nan)
    down = -np.diff(l, prepend=np.nan)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan

    prev_close = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum.reduce([
        h - l, np.abs(h - prev_close), np.abs(l - prev_close)
    ])
    tr[0] = np.nan

    atr_s = wilder_smooth(tr, period)
    plus_di = 100.0 * wilder_smooth(plus_dm, period) / atr_s
    minus_di = 100.0 * wilder_smooth(minus_dm, period) / atr_s

    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return ADXResult(wilder_smooth(dx, period), plus_di, minus_di)
