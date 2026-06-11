"""Risk and performance metrics for backtest results.

All metrics operate on a per-bar *simple returns* array. Annualization
uses ``periods_per_year`` (252 for daily bars, 252*6.5*60 for minute
bars, etc.).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from quantlab.indicators._core import Float, as_float_array

TRADING_DAYS = 252


def equity_curve(returns: ArrayLike, initial: float = 1.0) -> Float:
    """Compound an equity curve from simple per-bar returns."""
    r = as_float_array(returns, "returns")
    return initial * np.cumprod(1.0 + r)


def total_return(returns: ArrayLike) -> float:
    r = as_float_array(returns, "returns")
    return float(np.prod(1.0 + r) - 1.0)


def cagr(returns: ArrayLike, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound Annual Growth Rate."""
    r = as_float_array(returns, "returns")
    if len(r) == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:
        return -1.0
    years = len(r) / periods_per_year
    return growth ** (1.0 / years) - 1.0


def sharpe(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sharpe ratio (population std, 0 if flat)."""
    r = as_float_array(returns, "returns") - risk_free_rate / periods_per_year
    std = r.std()
    if std == 0.0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def sortino(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sortino ratio — only downside deviation penalizes."""
    r = as_float_array(returns, "returns") - risk_free_rate / periods_per_year
    downside = r[r < 0]
    if len(downside) == 0:
        return float("inf") if r.mean() > 0 else 0.0
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0.0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: ArrayLike) -> float:
    """Maximum peak-to-trough drawdown, returned as a negative number."""
    eq = equity_curve(returns)
    peaks = np.maximum.accumulate(eq)
    drawdowns = eq / peaks - 1.0
    return float(drawdowns.min())


def drawdown_series(returns: ArrayLike) -> Float:
    """Per-bar drawdown from the running peak (for plotting)."""
    eq = equity_curve(returns)
    return eq / np.maximum.accumulate(eq) - 1.0


def win_rate(trade_pnls: ArrayLike) -> float:
    """Fraction of closed trades with positive P&L."""
    pnl = as_float_array(trade_pnls, "trade_pnls")
    if len(pnl) == 0:
        return 0.0
    return float((pnl > 0).mean())


def profit_factor(trade_pnls: ArrayLike) -> float:
    """Gross profit / gross loss over closed trades."""
    pnl = as_float_array(trade_pnls, "trade_pnls")
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def exposure(positions: ArrayLike) -> float:
    """Fraction of bars with a non-zero position."""
    pos = as_float_array(positions, "positions")
    if len(pos) == 0:
        return 0.0
    return float((pos != 0).mean())
