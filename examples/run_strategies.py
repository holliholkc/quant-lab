"""Run three textbook strategies on AMZN daily data (2015–2024) and
compare them honestly against buy-and-hold.

The point of this example is NOT that these strategies make money —
with textbook parameters they generally don't. The point is that the
engine measures them honestly: next-bar execution, 10 bps round-trip
costs, and a baseline that is brutally hard to beat on a stock that
went up 10x over the period.

Run from the repo root::

    python examples/run_strategies.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantlab import indicators as ind
from quantlab.backtest import buy_and_hold, drawdown_series, run_backtest

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "AMZN.csv"
OUTPUT = HERE / "output"


def load_close(path: Path = DATA) -> tuple[np.ndarray, list[str]]:
    """Load Close prices from the yfinance-format CSV (junk row 2)."""
    closes: list[float] = []
    dates: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                closes.append(float(row["Close"]))
                dates.append(row["Price"])  # quirk: date column is named Price
            except (ValueError, TypeError):
                continue  # the "Ticker,AMZN,..." junk row
    return np.asarray(closes), dates


# --- textbook strategies (book parameters, deliberately untuned) -------------

def sma_crossover_signals(close: np.ndarray, fast: int = 50, slow: int = 200):
    """Golden cross: long while SMA(fast) > SMA(slow), flat otherwise."""
    fast_line = ind.sma(close, fast)
    slow_line = ind.sma(close, slow)
    return np.where(fast_line > slow_line, 1.0, 0.0)


def rsi_mean_reversion_signals(close: np.ndarray, period: int = 14,
                               buy_below: float = 30, sell_above: float = 70):
    """Long after RSI dips below 30, exit when it rises above 70."""
    values = ind.rsi(close, period)
    signals = np.zeros(len(close))
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


def bollinger_breakout_signals(close: np.ndarray, period: int = 20,
                               num_std: float = 2.0):
    """Long while price is above the upper band's last breakout, exit
    at the middle band (classic breakout-with-trailing-exit)."""
    upper, middle, _ = ind.bollinger(close, period, num_std)
    signals = np.zeros(len(close))
    in_position = False
    for i in range(len(close)):
        if np.isnan(middle[i]):
            continue
        if not in_position and close[i] > upper[i]:
            in_position = True
        elif in_position and close[i] < middle[i]:
            in_position = False
        signals[i] = 1.0 if in_position else 0.0
    return signals


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    close, dates = load_close()
    print(f"AMZN daily: {len(close)} bars ({dates[0]} .. {dates[-1]})\n")

    runs = {
        "Buy & Hold": buy_and_hold(close),
        "SMA 50/200 crossover": run_backtest(close, sma_crossover_signals(close)),
        "RSI-14 mean reversion": run_backtest(close, rsi_mean_reversion_signals(close)),
        "Bollinger 20/2 breakout": run_backtest(close, bollinger_breakout_signals(close)),
    }

    header = (f"{'Strategy':<26} {'TotRet':>9} {'CAGR':>8} {'Sharpe':>7} "
              f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>7}")
    print(header)
    print("-" * len(header))
    for name, result in runs.items():
        mt = result.metrics()
        print(f"{name:<26} {mt['total_return']:>+9.1%} {mt['cagr']:>+8.1%} "
              f"{mt['sharpe']:>7.2f} {mt['max_drawdown']:>8.1%} "
              f"{mt['win_rate']:>8.1%} {int(mt['n_trades']):>7}")

    # --- chart -----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    x = np.arange(len(close))
    for name, result in runs.items():
        axes[0].plot(x, result.equity, label=name,
                     lw=2.2 if name == "Buy & Hold" else 1.2)
        axes[1].plot(x, drawdown_series(result.returns), lw=1.0,
                     label=None if name != "Buy & Hold" else name)
    axes[0].set_title("AMZN 2015–2024: textbook strategies vs Buy & Hold "
                      "(10 bps costs, next-bar execution)")
    axes[0].set_ylabel("Equity (start = 1.0)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Bar")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = OUTPUT / "strategies_vs_buyhold.png"
    plt.savefig(out, dpi=120)
    print(f"\nChart -> {out}")


if __name__ == "__main__":
    main()
