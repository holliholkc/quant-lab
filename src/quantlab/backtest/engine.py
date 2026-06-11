"""Vectorized signal-based backtesting engine.

Design choices, stated explicitly because they are where most
backtests silently lie:

* **Next-bar execution.** A signal computed on bar ``t`` is executed
  at bar ``t+1``: the position array is the signal array shifted by
  one. Executing on the same bar the signal was computed from is
  lookahead bias — the engine makes it structurally impossible.
* **Costs on every position change.** Each unit of position change
  pays ``commission_bps + slippage_bps`` (in basis points of traded
  notional). Flipping long→short therefore costs twice as much as
  opening from flat — as it does at a real broker.
* **Signals are target positions**, not orders: −1 (short), 0 (flat),
  +1 (long), or any float in between for partial sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from quantlab.backtest import metrics as m
from quantlab.indicators._core import Float, as_float_array


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a single backtest run."""

    returns: Float          # per-bar strategy returns, net of costs
    positions: Float        # position actually held on each bar
    trade_pnls: Float       # compounded P&L of each closed trade
    n_trades: int
    periods_per_year: int = m.TRADING_DAYS
    _metrics: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def equity(self) -> Float:
        return m.equity_curve(self.returns)

    def metrics(self) -> dict[str, float]:
        if not self._metrics:
            ppy = self.periods_per_year
            self._metrics.update({
                "total_return": m.total_return(self.returns),
                "cagr": m.cagr(self.returns, ppy),
                "sharpe": m.sharpe(self.returns, periods_per_year=ppy),
                "sortino": m.sortino(self.returns, periods_per_year=ppy),
                "max_drawdown": m.max_drawdown(self.returns),
                "win_rate": m.win_rate(self.trade_pnls),
                "profit_factor": m.profit_factor(self.trade_pnls),
                "exposure": m.exposure(self.positions),
                "n_trades": float(self.n_trades),
            })
        return dict(self._metrics)

    def summary(self) -> str:
        mt = self.metrics()
        lines = [
            f"Total return : {mt['total_return']:+.2%}",
            f"CAGR         : {mt['cagr']:+.2%}",
            f"Sharpe       : {mt['sharpe']:.2f}",
            f"Sortino      : {mt['sortino']:.2f}",
            f"Max drawdown : {mt['max_drawdown']:.2%}",
            f"Win rate     : {mt['win_rate']:.1%} over {self.n_trades} trades",
            f"Profit factor: {mt['profit_factor']:.2f}",
            f"Exposure     : {mt['exposure']:.1%}",
        ]
        return "\n".join(lines)


def _trade_pnls(positions: Float, bar_returns: Float) -> Float:
    """Compound per-bar returns into per-trade P&L.

    A trade is a maximal run of constant non-zero position.
    """
    pnls: list[float] = []
    current: float | None = None
    for pos, r in zip(positions, bar_returns):
        if pos != 0:
            if current is None:
                current = 1.0
            current *= 1.0 + r
        elif current is not None:
            pnls.append(current - 1.0)
            current = None
    if current is not None:
        pnls.append(current - 1.0)
    return np.asarray(pnls, dtype=np.float64)


def run_backtest(
    close: ArrayLike,
    signals: ArrayLike,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = m.TRADING_DAYS,
) -> BacktestResult:
    """Run a vectorized backtest of target-position ``signals`` on
    ``close`` prices.

    Parameters
    ----------
    close:
        Close prices, 1-D.
    signals:
        Target position per bar in [−1, 1]; NaN is treated as 0.
        The signal from bar ``t`` is held during bar ``t+1``.
    commission_bps, slippage_bps:
        Cost per unit of position change, in basis points.
    """
    c = as_float_array(close, "close")
    s = as_float_array(signals, "signals")
    if len(c) != len(s):
        raise ValueError("close and signals must have equal length")
    if len(c) < 2:
        raise ValueError("need at least 2 bars")

    s = np.nan_to_num(s, nan=0.0)
    if np.any(np.abs(s) > 1.0):
        raise ValueError("signals must be within [-1, 1]")

    # Next-bar execution: position on bar t is the signal from bar t-1.
    positions = np.concatenate([[0.0], s[:-1]])

    price_returns = np.concatenate([[0.0], np.diff(c) / c[:-1]])
    gross = positions * price_returns

    cost_rate = (commission_bps + slippage_bps) / 10_000.0
    position_changes = np.abs(np.diff(positions, prepend=0.0))
    costs = position_changes * cost_rate

    net = gross - costs

    entries = (positions != 0) & (np.concatenate([[0.0], positions[:-1]]) != positions)
    n_trades = int(entries.sum())

    return BacktestResult(
        returns=net,
        positions=positions,
        trade_pnls=_trade_pnls(positions, net),
        n_trades=n_trades,
        periods_per_year=periods_per_year,
    )


def buy_and_hold(
    close: ArrayLike,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    periods_per_year: int = m.TRADING_DAYS,
) -> BacktestResult:
    """The baseline every strategy must beat: buy on the first bar,
    hold to the end, pay entry costs once.
    """
    c = as_float_array(close, "close")
    signals = np.ones(len(c))
    return run_backtest(
        c, signals, commission_bps, slippage_bps, periods_per_year
    )
