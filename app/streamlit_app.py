"""Streamlit demo: an interactive lab for quantlab.

Two tabs:

* **Индикаторы** — price chart with overlays and an oscillator panel;
  every parameter is a live slider.
* **Бэктест** — pick a textbook strategy, tune its parameters and the
  costs, and watch the equity curve fight buy-and-hold in real time.

Run from the repo root::

    pip install -e .[dev] streamlit
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quantlab import indicators as ind  # noqa: E402
from quantlab import strategies as strat  # noqa: E402
from quantlab.backtest import buy_and_hold, drawdown_series, run_backtest  # noqa: E402

DATA = ROOT / "examples" / "data" / "AMZN.csv"

METRIC_LABELS = {
    "total_return": ("Доходность", "{:+.1%}"),
    "cagr": ("CAGR", "{:+.1%}"),
    "sharpe": ("Sharpe", "{:.2f}"),
    "sortino": ("Sortino", "{:.2f}"),
    "max_drawdown": ("Max drawdown", "{:.1%}"),
    "win_rate": ("Win rate", "{:.1%}"),
    "profit_factor": ("Profit factor", "{:.2f}"),
    "n_trades": ("Сделок", "{:.0f}"),
}


@st.cache_data
def load_ohlcv() -> pd.DataFrame:
    rows = []
    with open(DATA, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "date": row["Price"],  # quirk: date column named Price
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                })
            except (ValueError, TypeError):
                continue
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def indicators_tab(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    volume = df["volume"].to_numpy()
    dates = df["date"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Оверлеи на цене**")
        show_sma = st.checkbox("SMA", value=True)
        sma_period = st.slider("Период SMA", 5, 250, 50, disabled=not show_sma)
        show_ema = st.checkbox("EMA", value=False)
        ema_span = st.slider("Период EMA", 5, 250, 21, disabled=not show_ema)
        show_bb = st.checkbox("Bollinger Bands", value=True)
        bb_period = st.slider("Период Bollinger", 5, 100, 20, disabled=not show_bb)
        bb_std = st.slider("σ Bollinger", 1.0, 3.0, 2.0, 0.5, disabled=not show_bb)
    with right:
        st.markdown("**Нижняя панель**")
        osc = st.selectbox(
            "Осциллятор",
            ["RSI", "MACD", "Stochastic", "ATR", "ADX", "CCI", "OBV"],
        )
        osc_period = st.slider("Период осциллятора", 5, 50, 14)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    axes[0].plot(dates, close, color="black", lw=1.1, label="Close")
    if show_sma:
        axes[0].plot(dates, ind.sma(close, sma_period), lw=1.0,
                     label=f"SMA({sma_period})")
    if show_ema:
        axes[0].plot(dates, ind.ema(close, ema_span), lw=1.0,
                     label=f"EMA({ema_span})")
    if show_bb:
        upper, _, lower = ind.bollinger(close, bb_period, bb_std)
        axes[0].fill_between(dates, lower, upper, alpha=0.15,
                             label=f"BB({bb_period},{bb_std:g})")
    axes[0].legend(loc="upper left", ncols=4, fontsize=9)
    axes[0].set_ylabel("Цена, $")
    axes[0].grid(alpha=0.3)

    ax = axes[1]
    if osc == "RSI":
        ax.plot(dates, ind.rsi(close, osc_period), lw=1.0)
        ax.axhline(70, color="grey", ls="--", lw=0.7)
        ax.axhline(30, color="grey", ls="--", lw=0.7)
        ax.set_ylim(0, 100)
    elif osc == "MACD":
        macd_line, signal_line, hist = ind.macd(close)
        ax.bar(dates, hist, width=1.5, alpha=0.4, color="grey")
        ax.plot(dates, macd_line, lw=1.0, label="MACD")
        ax.plot(dates, signal_line, lw=1.0, label="signal")
        ax.legend(fontsize=8)
    elif osc == "Stochastic":
        k, d = ind.stochastic(high, low, close, osc_period)
        ax.plot(dates, k, lw=0.9, label="%K")
        ax.plot(dates, d, lw=0.9, label="%D")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
    elif osc == "ATR":
        ax.plot(dates, ind.atr(high, low, close, osc_period), lw=1.0)
    elif osc == "ADX":
        result = ind.adx(high, low, close, osc_period)
        ax.plot(dates, result.adx, lw=1.0, label="ADX")
        ax.plot(dates, result.plus_di, lw=0.8, label="+DI")
        ax.plot(dates, result.minus_di, lw=0.8, label="−DI")
        ax.legend(fontsize=8)
    elif osc == "CCI":
        ax.plot(dates, ind.cci(high, low, close, osc_period), lw=1.0)
        ax.axhline(100, color="grey", ls="--", lw=0.7)
        ax.axhline(-100, color="grey", ls="--", lw=0.7)
    elif osc == "OBV":
        ax.plot(dates, ind.obv(close, volume), lw=1.0)
    ax.set_ylabel(osc)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def backtest_tab(df: pd.DataFrame) -> None:
    close = df["close"].to_numpy()
    dates = df["date"]

    left, right = st.columns(2)
    with left:
        strategy = st.selectbox(
            "Стратегия",
            ["SMA crossover", "RSI mean reversion", "Bollinger breakout"],
        )
        if strategy == "SMA crossover":
            fast = st.slider("Быстрая SMA", 5, 100, 50)
            slow = st.slider("Медленная SMA", 50, 300, 200)
            signals = (strat.sma_crossover(close, fast, slow)
                       if fast < slow else None)
            if signals is None:
                st.error("Быстрая SMA должна быть короче медленной.")
                return
        elif strategy == "RSI mean reversion":
            period = st.slider("Период RSI", 5, 30, 14)
            buy_below = st.slider("Вход: RSI ниже", 10, 45, 30)
            sell_above = st.slider("Выход: RSI выше", 55, 90, 70)
            signals = strat.rsi_mean_reversion(close, period, buy_below, sell_above)
        else:
            period = st.slider("Период Bollinger", 10, 60, 20)
            num_std = st.slider("σ", 1.0, 3.0, 2.0, 0.5)
            signals = strat.bollinger_breakout(close, period, num_std)
    with right:
        commission = st.slider("Комиссия, б.п.", 0, 30, 5)
        slippage = st.slider("Проскальзывание, б.п.", 0, 30, 5)
        st.caption(
            "Сигнал бара t исполняется на баре t+1; каждая единица "
            "изменения позиции платит комиссию + проскальзывание."
        )

    result = run_backtest(close, signals, commission, slippage)
    baseline = buy_and_hold(close, commission, slippage)

    # --- equity + drawdown -------------------------------------------------
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    axes[0].plot(dates, baseline.equity, lw=2.0, label="Buy & Hold")
    axes[0].plot(dates, result.equity, lw=1.2, label=strategy)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Эквити (старт = 1.0)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(dates, drawdown_series(baseline.returns), lw=0.9,
                 label="Buy & Hold")
    axes[1].plot(dates, drawdown_series(result.returns), lw=0.9,
                 label=strategy)
    axes[1].set_ylabel("Просадка")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- metrics table -----------------------------------------------------
    strat_m, base_m = result.metrics(), baseline.metrics()
    rows = []
    for key, (label, fmt) in METRIC_LABELS.items():
        rows.append({
            "Метрика": label,
            strategy: fmt.format(strat_m[key]),
            "Buy & Hold": fmt.format(base_m[key]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if strat_m["sharpe"] > base_m["sharpe"]:
        st.success(
            f"По Sharpe стратегия обходит Buy & Hold "
            f"({strat_m['sharpe']:.2f} > {base_m['sharpe']:.2f}). "
            f"Прежде чем радоваться: это один актив, один период и "
            f"подбор параметров задним числом."
        )
    else:
        st.info(
            f"Buy & Hold впереди по Sharpe "
            f"({base_m['sharpe']:.2f} ≥ {strat_m['sharpe']:.2f}) — "
            f"ожидаемо для книжных параметров. Попробуй покрутить "
            f"ползунки и заметить, как легко переобучиться на историю."
        )


def main() -> None:
    st.set_page_config(page_title="quant-lab", page_icon="📐", layout="wide")
    st.title("📐 quant-lab: интерактивная лаборатория")
    st.caption(
        "Индикаторы и бэктест-движок, написанные с нуля на NumPy. "
        "Данные: AMZN, дневки 2015–2024."
    )

    df = load_ohlcv()
    tab_ind, tab_bt = st.tabs(["Индикаторы", "Бэктест"])
    with tab_ind:
        indicators_tab(df)
    with tab_bt:
        backtest_tab(df)


if __name__ == "__main__":
    main()
