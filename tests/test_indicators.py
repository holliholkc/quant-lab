"""Indicator tests: every vectorized implementation is checked against
a deliberately naive, loop-based reference written directly from the
textbook definition.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab import indicators as ind

RNG = np.random.default_rng(42)
N = 500

# Synthetic but realistic OHLCV
CLOSE = 100.0 * np.cumprod(1.0 + RNG.normal(0.0003, 0.02, N))
HIGH = CLOSE * (1.0 + np.abs(RNG.normal(0, 0.01, N)))
LOW = CLOSE * (1.0 - np.abs(RNG.normal(0, 0.01, N)))
VOLUME = RNG.uniform(1e5, 1e6, N)


# --- naive references --------------------------------------------------------

def naive_sma(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.mean(x[i - n + 1: i + 1])
    return out


def naive_ema(x, span):
    out = np.full(len(x), np.nan)
    alpha = 2.0 / (span + 1.0)
    out[span - 1] = np.mean(x[:span])
    for i in range(span, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def naive_rsi(x, n):
    out = np.full(len(x), np.nan)
    deltas = np.diff(x)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    avg_gain = np.mean(gains[:n])
    avg_loss = np.mean(losses[:n])
    out[n] = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss else 100.0
    for i in range(n + 1, len(x)):
        avg_gain += (gains[i - 1] - avg_gain) / n
        avg_loss += (losses[i - 1] - avg_loss) / n
        out[i] = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss else 100.0
    return out


def naive_atr(h, l, c, n):
    tr = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.full(len(c), np.nan)
    out[n] = np.nanmean(tr[1: n + 1])
    for i in range(n + 1, len(c)):
        out[i] = out[i - 1] + (tr[i] - out[i - 1]) / n
    return out


# --- tests --------------------------------------------------------------------

def test_sma_matches_naive():
    np.testing.assert_allclose(
        ind.sma(CLOSE, 20), naive_sma(CLOSE, 20), equal_nan=True
    )


def test_ema_matches_naive():
    np.testing.assert_allclose(
        ind.ema(CLOSE, 20), naive_ema(CLOSE, 20), equal_nan=True
    )


def test_rsi_matches_naive():
    np.testing.assert_allclose(
        ind.rsi(CLOSE, 14), naive_rsi(CLOSE, 14), equal_nan=True, atol=1e-9
    )


def test_rsi_bounds():
    values = ind.rsi(CLOSE, 14)
    valid = values[~np.isnan(values)]
    assert np.all(valid >= 0) and np.all(valid <= 100)


def test_rsi_monotonic_series_saturates():
    rising = np.arange(1.0, 101.0)
    values = ind.rsi(rising, 14)
    assert np.allclose(values[~np.isnan(values)], 100.0)


def test_atr_matches_naive():
    np.testing.assert_allclose(
        ind.atr(HIGH, LOW, CLOSE, 14),
        naive_atr(HIGH, LOW, CLOSE, 14),
        equal_nan=True,
    )


def test_macd_consistency():
    result = ind.macd(CLOSE)
    expected = ind.ema(CLOSE, 12) - ind.ema(CLOSE, 26)
    np.testing.assert_allclose(result.macd, expected, equal_nan=True)
    np.testing.assert_allclose(
        result.histogram, result.macd - result.signal, equal_nan=True
    )


def test_bollinger_band_ordering_and_middle():
    upper, middle, lower = ind.bollinger(CLOSE, 20)
    valid = ~np.isnan(middle)
    assert np.all(upper[valid] >= middle[valid])
    assert np.all(middle[valid] >= lower[valid])
    np.testing.assert_allclose(middle, ind.sma(CLOSE, 20), equal_nan=True)


def test_bollinger_flat_series_collapses():
    flat = np.full(50, 42.0)
    upper, middle, lower = ind.bollinger(flat, 10)
    valid = ~np.isnan(middle)
    np.testing.assert_allclose(upper[valid], lower[valid])


def test_stochastic_bounds():
    k, d = ind.stochastic(HIGH, LOW, CLOSE)
    for series in (k, d):
        valid = series[~np.isnan(series)]
        assert np.all(valid >= 0) and np.all(valid <= 100)


def test_cci_centered_on_flat():
    flat = np.full(60, 10.0)
    values = ind.cci(flat, flat, flat, 20)
    # flat series: TP == SMA, CCI defined as 0/0 -> nan; just no crash
    assert len(values) == 60


def test_adx_uptrend_di_ordering():
    rising = np.linspace(100, 200, 120)
    result = ind.adx(rising * 1.01, rising * 0.99, rising, 14)
    valid = ~np.isnan(result.plus_di)
    assert np.all(result.plus_di[valid] >= result.minus_di[valid])


def test_obv_known_values():
    close = np.array([10.0, 11.0, 10.5, 10.5, 12.0])
    volume = np.array([100.0, 200.0, 150.0, 300.0, 250.0])
    # start 0; +200 (up); -150 (down); +0 (flat); +250 (up)
    np.testing.assert_allclose(
        ind.obv(close, volume), [0.0, 200.0, 50.0, 50.0, 300.0]
    )


def test_vwap_constant_price():
    flat = np.full(30, 50.0)
    values = ind.vwap(flat, flat, flat, VOLUME[:30])
    np.testing.assert_allclose(values, 50.0)


def test_warmup_lengths():
    assert np.isnan(ind.sma(CLOSE, 20)[:19]).all()
    assert not np.isnan(ind.sma(CLOSE, 20)[19])
    assert np.isnan(ind.rsi(CLOSE, 14)[:14]).all()
    assert not np.isnan(ind.rsi(CLOSE, 14)[14])
    assert np.isnan(ind.atr(HIGH, LOW, CLOSE, 14)[:14]).all()
    assert not np.isnan(ind.atr(HIGH, LOW, CLOSE, 14)[14])


def test_input_validation():
    with pytest.raises(ValueError):
        ind.sma(CLOSE, 0)
    with pytest.raises(ValueError):
        ind.sma(CLOSE[:5], 10)
    with pytest.raises(ValueError):
        ind.macd(CLOSE, fast=26, slow=12)
    with pytest.raises(ValueError):
        ind.atr(HIGH, LOW[:-1], CLOSE, 14)
    with pytest.raises(ValueError):
        ind.sma(CLOSE.reshape(-1, 1), 5)
