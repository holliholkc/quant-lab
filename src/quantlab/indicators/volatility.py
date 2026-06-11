"""Volatility indicators: Bollinger Bands, ATR."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from quantlab.indicators._core import (
    Float,
    as_float_array,
    rolling_mean,
    rolling_std,
    validate_period,
    wilder_smooth,
)


class BollingerResult(NamedTuple):
    upper: Float
    middle: Float
    lower: Float


def bollinger(
    close: ArrayLike,
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    """Bollinger Bands: ``SMA(n) ± k * rolling_std(n)`` (population std,
    ``ddof=0``, per the original definition).
    """
    x = as_float_array(close, "close")
    validate_period(period, len(x))
    middle = rolling_mean(x, period)
    dev = num_std * rolling_std(x, period, ddof=0)
    return BollingerResult(middle + dev, middle, middle - dev)


def atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
) -> Float:
    """Average True Range (Wilder).

    ``TR = max(H−L, |H−C_prev|, |L−C_prev|)``, smoothed with
    ``alpha = 1/period``. The first TR value (no previous close)
    is NaN, so ATR becomes valid at index ``period``.
    """
    h = as_float_array(high, "high")
    l = as_float_array(low, "low")
    c = as_float_array(close, "close")
    if not (len(h) == len(l) == len(c)):
        raise ValueError("high, low, close must have equal length")
    validate_period(period + 1, len(c))

    prev_close = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum.reduce([
        h - l, np.abs(h - prev_close), np.abs(l - prev_close)
    ])
    tr[0] = np.nan
    return wilder_smooth(tr, period)
