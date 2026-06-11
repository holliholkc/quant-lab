"""Backtest engine tests, including the two failure modes that
invalidate most homemade backtests: lookahead bias and free trading.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.backtest import (
    buy_and_hold,
    cagr,
    max_drawdown,
    profit_factor,
    run_backtest,
    sharpe,
    total_return,
    win_rate,
)


def test_hand_computed_example():
    """One long trade with exact arithmetic, zero costs.

    Prices 100 -> 110 -> 121: signal turns on at bar 0, so position
    is held on bars 1 and 2, capturing (+10%) and (+10%).
    """
    close = np.array([100.0, 110.0, 121.0])
    signals = np.array([1.0, 1.0, 0.0])
    result = run_backtest(close, signals, commission_bps=0, slippage_bps=0)
    assert result.equity[-1] == pytest.approx(1.21)
    assert result.n_trades == 1


def test_costs_reduce_equity_exactly():
    close = np.array([100.0, 110.0, 121.0])
    signals = np.array([1.0, 1.0, 0.0])
    # 10 bps commission + 0 slippage, paid on entry (bar 1) once:
    result = run_backtest(close, signals, commission_bps=10, slippage_bps=0)
    expected = (1.0 + 0.10 - 0.001) * 1.10  # entry bar return minus fee
    assert result.equity[-1] == pytest.approx(expected)


def test_no_lookahead():
    """A signal on the last bar must not affect any equity value:
    it would only execute on the bar after the data ends.
    """
    close = np.array([100.0, 101.0, 102.0, 200.0])
    no_signal = run_backtest(close, np.zeros(4), 0, 0)
    last_bar_signal = np.array([0.0, 0.0, 0.0, 1.0])
    with_signal = run_backtest(close, last_bar_signal, 0, 0)
    np.testing.assert_allclose(no_signal.equity, with_signal.equity)


def test_flat_signals_flat_equity():
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 50))
    result = run_backtest(close, np.zeros(50), 5, 5)
    np.testing.assert_allclose(result.equity, 1.0)
    assert result.n_trades == 0


def test_short_profits_in_decline():
    close = np.array([100.0, 90.0, 81.0])
    signals = np.array([-1.0, -1.0, 0.0])
    result = run_backtest(close, signals, 0, 0)
    assert result.equity[-1] == pytest.approx(1.21)  # +10% twice


def test_flip_costs_more_than_round_trip():
    """Entering and exiting moves position by 2 units total (0->1->0);
    entering and flipping moves it by 3 (0->1->-1). Costs must scale
    with traded notional, so flip = 1.5x round trip.
    """
    close = np.array([100.0, 100.0, 100.0])
    flip = np.array([1.0, -1.0, 0.0])
    round_trip = np.array([1.0, 0.0, 0.0])
    cost_flip = 1.0 - run_backtest(close, flip, 10, 0).equity[-1]
    cost_rt = 1.0 - run_backtest(close, round_trip, 10, 0).equity[-1]
    # equity compounds multiplicatively, so the ratio is 1.5 only to
    # first order — hence the loose relative tolerance
    assert cost_flip == pytest.approx(cost_rt * 1.5, rel=1e-3)


def test_buy_and_hold_matches_price_growth():
    close = np.array([100.0, 120.0, 150.0])
    result = buy_and_hold(close, commission_bps=0, slippage_bps=0)
    assert result.equity[-1] == pytest.approx(1.5)


def test_signal_validation():
    close = np.array([100.0, 101.0])
    with pytest.raises(ValueError):
        run_backtest(close, np.array([2.0, 0.0]))
    with pytest.raises(ValueError):
        run_backtest(close, np.array([1.0]))
    with pytest.raises(ValueError):
        run_backtest(np.array([100.0]), np.array([1.0]))


# --- metrics ------------------------------------------------------------------

def test_total_return_and_cagr():
    returns = np.full(252, 0.001)
    assert total_return(returns) == pytest.approx(1.001**252 - 1)
    assert cagr(returns) == pytest.approx(1.001**252 - 1)


def test_sharpe_zero_for_flat():
    assert sharpe(np.zeros(100)) == 0.0


def test_max_drawdown_known_case():
    # equity 1.0 -> 2.0 -> 1.0: drawdown is -50%
    returns = np.array([1.0, -0.5])
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_win_rate_and_profit_factor():
    pnls = np.array([0.10, -0.05, 0.20, -0.05])
    assert win_rate(pnls) == pytest.approx(0.5)
    assert profit_factor(pnls) == pytest.approx(0.30 / 0.10)
