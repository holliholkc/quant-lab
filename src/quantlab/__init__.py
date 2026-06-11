"""quantlab — technical indicators and a vectorized backtesting engine.

Two layers:

* :mod:`quantlab.indicators` — classic TA indicators implemented from
  scratch on NumPy, with documented warm-up (NaN) conventions.
* :mod:`quantlab.backtest` — a vectorized signal-based backtester with
  next-bar execution, transaction costs and risk metrics.
"""

from quantlab import backtest, indicators

__all__ = ["indicators", "backtest"]
__version__ = "0.1.0"
