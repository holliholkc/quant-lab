"""Volume indicators: OBV, VWAP."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quantlab.indicators._core import Float, as_float_array


def obv(close: ArrayLike, volume: ArrayLike) -> Float:
    """On-Balance Volume: cumulative volume signed by the direction
    of the close-to-close move. Starts at 0.
    """
    c = as_float_array(close, "close")
    v = as_float_array(volume, "volume")
    if len(c) != len(v):
        raise ValueError("close and volume must have equal length")
    direction = np.sign(np.diff(c, prepend=c[:1]))
    return np.cumsum(direction * v)


def vwap(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
) -> Float:
    """Volume-Weighted Average Price (cumulative form).

    ``VWAP_t = Σ(TP_i * V_i) / Σ(V_i)`` with ``TP = (H+L+C)/3``,
    accumulated from the start of the series. For session-anchored
    VWAP, slice the inputs per session before calling.
    """
    h = as_float_array(high, "high")
    l = as_float_array(low, "low")
    c = as_float_array(close, "close")
    v = as_float_array(volume, "volume")
    if not (len(h) == len(l) == len(c) == len(v)):
        raise ValueError("inputs must have equal length")
    tp = (h + l + c) / 3.0
    cum_v = np.cumsum(v)
    out = np.full(c.shape, np.nan)
    nonzero = cum_v > 0
    out[nonzero] = np.cumsum(tp * v)[nonzero] / cum_v[nonzero]
    return out
