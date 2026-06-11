"""Strategy signal generators: shape, range and basic behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab import strategies as strat

RNG = np.random.default_rng(3)
CLOSE = 100.0 * np.cumprod(1.0 + RNG.normal(0.0003, 0.02, 400))


@pytest.mark.parametrize("fn", [
    lambda c: strat.sma_crossover(c, 10, 30),
    lambda c: strat.rsi_mean_reversion(c),
    lambda c: strat.bollinger_breakout(c),
])
def test_signals_shape_and_range(fn):
    signals = fn(CLOSE)
    assert signals.shape == CLOSE.shape
    assert set(np.unique(signals)) <= {0.0, 1.0}


def test_sma_crossover_long_in_strong_uptrend():
    rising = np.linspace(100, 400, 300)
    signals = strat.sma_crossover(rising, 10, 50)
    # once both SMAs are warm, fast > slow on a monotone rise
    assert signals[100:].all()


def test_sma_crossover_validates_periods():
    with pytest.raises(ValueError):
        strat.sma_crossover(CLOSE, 50, 50)


def test_rsi_mean_reversion_validates_thresholds():
    with pytest.raises(ValueError):
        strat.rsi_mean_reversion(CLOSE, buy_below=80, sell_above=70)
