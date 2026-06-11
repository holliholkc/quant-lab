"""Vectorized backtesting: engine, baseline and risk metrics."""

from quantlab.backtest.engine import BacktestResult, buy_and_hold, run_backtest
from quantlab.backtest.metrics import (
    cagr,
    drawdown_series,
    equity_curve,
    exposure,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
    total_return,
    win_rate,
)

__all__ = [
    "run_backtest", "buy_and_hold", "BacktestResult",
    "equity_curve", "total_return", "cagr", "sharpe", "sortino",
    "max_drawdown", "drawdown_series", "win_rate", "profit_factor",
    "exposure",
]
