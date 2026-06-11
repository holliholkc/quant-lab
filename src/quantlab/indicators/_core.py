"""Shared low-level helpers for indicator implementations.

Conventions used across the package
-----------------------------------
* Inputs are 1-D ``numpy`` arrays (anything array-like is converted).
* Outputs are ``float64`` arrays of the same length as the input.
* The warm-up region — indices where the indicator is not yet defined —
  is filled with ``NaN`` rather than zeros or partial values, so the
  caller can never silently consume a half-warmed value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Float = NDArray[np.float64]


def as_float_array(x: ArrayLike, name: str = "input") -> Float:
    """Convert to a 1-D float64 array, validating dimensionality."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def validate_period(period: int, n: int) -> None:
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if period > n:
        raise ValueError(f"period {period} exceeds series length {n}")


def rolling_windows(x: Float, period: int) -> NDArray[np.float64]:
    """Return a (n-period+1, period) view of sliding windows (no copy)."""
    return np.lib.stride_tricks.sliding_window_view(x, period)


def rolling_mean(x: Float, period: int) -> Float:
    """Simple rolling mean via cumulative sums, NaN warm-up."""
    out = np.full(x.shape, np.nan)
    c = np.concatenate([[0.0], np.cumsum(x)])
    out[period - 1:] = (c[period:] - c[:-period]) / period
    return out


def rolling_std(x: Float, period: int, ddof: int = 0) -> Float:
    """Rolling standard deviation over sliding windows, NaN warm-up.

    Uses explicit windows rather than the cumsum-of-squares trick:
    marginally slower but numerically robust for long price series.
    """
    out = np.full(x.shape, np.nan)
    out[period - 1:] = rolling_windows(x, period).std(axis=1, ddof=ddof)
    return out


def wilder_smooth(x: Float, period: int) -> Float:
    """Wilder's smoothing (RMA): an EMA with ``alpha = 1/period``,
    seeded with the simple mean of the first ``period`` values.

    Defined for RSI/ATR/ADX in Wilder (1978). The recursion
    ``y[i] = y[i-1] + (x[i] - y[i-1]) / period`` is inherently
    sequential, so this is an explicit loop — see the benchmarks
    for why we keep it honest instead of pretending it vectorizes.

    ``x`` may contain a NaN warm-up prefix; smoothing starts after it.
    """
    out = np.full(x.shape, np.nan)
    start = int(np.argmax(~np.isnan(x)))  # first valid index
    if np.isnan(x[start]):  # all-NaN input
        return out
    seed_end = start + period
    if seed_end > len(x):
        return out
    out[seed_end - 1] = np.nanmean(x[start:seed_end])
    alpha = 1.0 / period
    for i in range(seed_end, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def ema_recursive(x: Float, span: int) -> Float:
    """EMA with ``alpha = 2/(span+1)``, seeded with the SMA of the
    first ``span`` values (the common TA charting convention).
    """
    out = np.full(x.shape, np.nan)
    if span > len(x):
        return out
    out[span - 1] = x[:span].mean()
    alpha = 2.0 / (span + 1.0)
    for i in range(span, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out
