"""Benchmark: naive loop implementations vs quantlab's vectorized ones.

Run from the repo root::

    python benchmarks/bench_indicators.py
"""

from __future__ import annotations

import time

import numpy as np

from quantlab import indicators as ind

N = 1_000_000
RNG = np.random.default_rng(7)
CLOSE = 100.0 * np.cumprod(1.0 + RNG.normal(0.0001, 0.01, N))


def naive_sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1: i + 1].mean()
    return out


def naive_bollinger(x: np.ndarray, n: int) -> tuple:
    mid = np.full(len(x), np.nan)
    up = np.full(len(x), np.nan)
    lo = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1: i + 1]
        m, s = w.mean(), w.std()
        mid[i], up[i], lo[i] = m, m + 2 * s, m - 2 * s
    return up, mid, lo


def naive_stochastic_k(h, l, c, n):
    out = np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        hh = h[i - n + 1: i + 1].max()
        ll = l[i - n + 1: i + 1].min()
        out[i] = 100 * (c[i] - ll) / (hh - ll) if hh != ll else 50.0
    return out


def timed(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def main() -> None:
    high = CLOSE * 1.005
    low = CLOSE * 0.995

    print(f"Series length: {N:,} bars\n")
    print(f"{'Indicator':<22} {'Naive loop':>12} {'quantlab':>12} {'Speedup':>9}")
    print("-" * 58)

    cases = [
        ("SMA(20)", lambda: naive_sma(CLOSE, 20), lambda: ind.sma(CLOSE, 20)),
        ("Bollinger(20, 2)", lambda: naive_bollinger(CLOSE, 20),
         lambda: ind.bollinger(CLOSE, 20)),
        ("Stochastic %K(14)", lambda: naive_stochastic_k(high, low, CLOSE, 14),
         lambda: ind.stochastic(high, low, CLOSE)),
    ]
    for name, naive_fn, fast_fn in cases:
        t_naive = timed(naive_fn)
        t_fast = timed(fast_fn)
        print(f"{name:<22} {t_naive:>11.2f}s {t_fast:>11.3f}s "
              f"{t_naive / t_fast:>8.0f}x")

    # honest note: recursive indicators don't vectorize
    t_ema = timed(lambda: ind.ema(CLOSE, 20))
    t_rsi = timed(lambda: ind.rsi(CLOSE, 14))
    print(f"\nRecursive (inherently sequential, explicit loop):")
    print(f"{'EMA(20)':<22} {'—':>12} {t_ema:>11.2f}s")
    print(f"{'RSI(14)':<22} {'—':>12} {t_rsi:>11.2f}s")


if __name__ == "__main__":
    main()
