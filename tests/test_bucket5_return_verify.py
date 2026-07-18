"""Always-runnable economic checks for the B5 return-verification suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bucket5_insurance_bt import production_config
from scripts.bucket5_return_verify import performance_metrics, run_verification, sensitivity_grid


def _panel(n: int = 105) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.bdate_range("2024-01-02", periods=n)
    x = np.arange(n, dtype=float)
    # Includes a short, sharp volatility event so the put/carry records exercise
    # real daily P&L paths rather than a schema-only flat market.
    shock = np.maximum(0.0, 1.0 - np.abs(x - 55.0) / 8.0)
    panel = pd.DataFrame(
        {
            "uvix": 20.0 * np.exp(-0.002 * x) * (1.0 + 1.2 * shock),
            "svix": 25.0 * np.exp(0.001 * x) * (1.0 - 0.25 * shock),
            "vix": 16.0 + 20.0 * shock,
            "vix3m": 20.0 + 10.0 * shock,
            "synthetic": False,
        },
        index=idx,
    )
    panel["ratio"] = panel["vix"] / panel["vix3m"]
    spx = pd.Series(4800.0 * (1.0 + 0.0005 * x - 0.16 * shock), index=idx, name="spx")
    return panel, spx


def test_split_books_reconcile_and_preserve_same_initial_capital():
    panel, spx = _panel()
    result = run_verification(panel, spx, production_config(initial_capital=100_000.0))
    assert result.gates["nav_identity"]["pass"]
    assert result.gates["harvested_cash_trace"]["pass"]
    assert result.books["carry"].attrs["initial_capital"] == 100_000.0
    assert result.books["puts"].attrs["initial_capital"] == 100_000.0
    assert result.books["puts"]["nav"].iloc[0] > 0
    assert set(("uvix_pnl", "svix_pnl", "tbill_pnl", "put_mtm_change", "put_cash_flow")).issubset(result.attribution.columns)
    assert result.reconciliation["identity_residual"].abs().max() < 0.05


def test_metrics_and_predeclared_cost_grid_smoke():
    panel, spx = _panel(90)
    result = run_verification(panel, spx, production_config(initial_capital=100_000.0))
    metrics = performance_metrics(result.books["combined"]["combined_equity"])
    assert set(metrics) == {"CAGR", "Vol", "Sharpe", "MaxDD", "Calmar"}
    grid = sensitivity_grid(panel, spx, production_config(initial_capital=100_000.0))
    assert len(grid) == 12
    assert set(grid["slippage_bps"]) == {5.0, 15.0, 30.0, 50.0}
    assert set(grid["borrow_multiple"]) == {1.0, 1.5, 2.0}
