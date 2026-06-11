"""Render an indicators gallery on real AMZN data — the picture that
shows what the library actually computes.

Run from the repo root::

    python examples/plot_indicators.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantlab import indicators as ind

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "AMZN.csv"
OUTPUT = HERE / "output"


def load_ohlcv(path: Path = DATA):
    cols = {"High": [], "Low": [], "Close": [], "Volume": []}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                values = {k: float(row[k]) for k in cols}
            except (ValueError, TypeError):
                continue  # junk "Ticker" row
            for k, v in values.items():
                cols[k].append(v)
    return (np.asarray(cols["High"]), np.asarray(cols["Low"]),
            np.asarray(cols["Close"]), np.asarray(cols["Volume"]))


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    high, low, close, volume = load_ohlcv()
    # last ~2 years of daily bars keeps the chart readable
    n = 500
    high, low, close, volume = high[-n:], low[-n:], close[-n:], volume[-n:]
    x = np.arange(n)

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 11), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1, 1, 1]},
    )

    # 1. price + overlays
    upper, middle, lower = ind.bollinger(close, 20, 2.0)
    axes[0].plot(x, close, color="black", lw=1.2, label="Close")
    axes[0].plot(x, ind.sma(close, 50), lw=1.0, label="SMA(50)")
    axes[0].plot(x, ind.ema(close, 21), lw=1.0, label="EMA(21)")
    axes[0].fill_between(x, lower, upper, alpha=0.15, color="tab:blue",
                         label="Bollinger 20/2")
    axes[0].set_title("AMZN, последние 500 торговых дней — quantlab.indicators")
    axes[0].set_ylabel("Цена, $")
    axes[0].legend(loc="upper left", ncols=4)
    axes[0].grid(alpha=0.3)

    # 2. RSI
    axes[1].plot(x, ind.rsi(close, 14), lw=1.0, color="tab:purple")
    axes[1].axhline(70, color="grey", ls="--", lw=0.7)
    axes[1].axhline(30, color="grey", ls="--", lw=0.7)
    axes[1].set_ylabel("RSI(14)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)

    # 3. MACD
    macd_line, signal_line, hist = ind.macd(close)
    axes[2].bar(x, hist, width=1.0, alpha=0.4, color="tab:grey",
                label="histogram")
    axes[2].plot(x, macd_line, lw=1.0, label="MACD")
    axes[2].plot(x, signal_line, lw=1.0, label="signal")
    axes[2].set_ylabel("MACD(12,26,9)")
    axes[2].legend(loc="upper left", ncols=3)
    axes[2].grid(alpha=0.3)

    # 4. ATR
    axes[3].plot(x, ind.atr(high, low, close, 14), lw=1.0, color="tab:red")
    axes[3].set_ylabel("ATR(14), $")
    axes[3].set_xlabel("Бар")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT / "indicators_gallery.png"
    plt.savefig(out, dpi=120)
    print(f"Gallery -> {out}")


if __name__ == "__main__":
    main()
