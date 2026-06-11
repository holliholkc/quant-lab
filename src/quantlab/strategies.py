"""Textbook example strategies producing target-position signals.

These are deliberately classic, deliberately untuned strategies with
book-default parameters. They exist to demonstrate the engine and to
serve as honest baselines — not to make money (see the README example:
all three lose to buy-and-hold on a 13x stock).

Every function returns a target-position array in {0.0, 1.0} of the
same length as ``close``; feed it to
:func:`quantlab.backtest.run_backtest`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quantlab import indicators as ind
from quantlab.indicators._core import Float, as_float_array


def sma_crossover(close: ArrayLike, fast: int = 50, slow: int = 200) -> Float:
    """Golden cross: long while SMA(fast) > SMA(slow), flat otherwise."""
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    c = as_float_array(close, "close")
    fast_line = ind.sma(c, fast)
    slow_line = ind.sma(c, slow)
    return np.where(fast_line > slow_line, 1.0, 0.0)


def rsi_mean_reversion(
    close: ArrayLike,
    period: int = 14,
    buy_below: float = 30.0,
    sell_above: float = 70.0,
) -> Float:
    """Long after RSI dips below ``buy_below``, exit above ``sell_above``.

    Stateful entry/exit thresholds require a sequential pass.
    """
    if not 0 <= buy_below < sell_above <= 100:
        raise ValueError("need 0 <= buy_below < sell_above <= 100")
    c = as_float_array(close, "close")
    values = ind.rsi(c, period)
    signals = np.zeros(len(c))
    in_position = False
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if not in_position and v < buy_below:
            in_position = True
        elif in_position and v > sell_above:
            in_position = False
        signals[i] = 1.0 if in_position else 0.0
    return signals


def bollinger_breakout(
    close: ArrayLike,
    period: int = 20,
    num_std: float = 2.0,
) -> Float:
    """Long on a close above the upper band, exit below the middle band."""
    c = as_float_array(close, "close")
    upper, middle, _ = ind.bollinger(c, period, num_std)
    signals = np.zeros(len(c))
    in_position = False
    for i in range(len(c)):
        if np.isnan(middle[i]):
            continue
        if not in_position and c[i] > upper[i]:
            in_position = True
        elif in_position and c[i] < middle[i]:
            in_position = False
        signals[i] = 1.0 if in_position else 0.0
    return signals
