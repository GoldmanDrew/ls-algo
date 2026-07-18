"""Smoke tests for B4 backtest path charts (notebook gallery helper)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.b4_backtest_pair_charts import (
    build_pair_path_bundle,
    plot_b4_pair_path,
    select_extreme_b4_pairs,
)
from scripts.b4_historical_audit import load_b4_plan_history, summarize_b4_plan_history


def test_select_extreme_b4_pairs() -> None:
    ps = pd.DataFrame(
        [
            {"ETF": "A", "Underlying": "X", "sleeve": "inverse_decay_bucket4", "pnl_usd": 10.0},
            {"ETF": "B", "Underlying": "Y", "sleeve": "inverse_decay_bucket4", "pnl_usd": -5.0},
            {"ETF": "C", "Underlying": "Z", "sleeve": "core_leveraged", "pnl_usd": 99.0},
            {"ETF": "D", "Underlying": "W", "sleeve": "inverse_decay_bucket4", "pnl_usd": 3.0},
        ]
    )
    out = select_extreme_b4_pairs(ps, n_top=1, n_bottom=1)
    assert list(out["ETF"]) == ["A", "B"]


def test_build_and_plot_bundle_fail_soft_without_panel() -> None:
    idx = pd.bdate_range("2026-04-01", periods=30)
    daily = pd.DataFrame(
        {
            "date": idx,
            "ETF": "CLSZ",
            "Underlying": "CLSK",
            "sleeve": "inverse_decay_bucket4",
            "etf_usd": -1000.0,
            "underlying_usd": 500.0,
            "hedge_ratio": 0.5,
            "daily_pnl": np.linspace(-10, 20, len(idx)),
            "cum_pnl": np.linspace(-10, 20, len(idx)).cumsum(),
            "is_rebalance": [1 if i % 5 == 0 else 0 for i in range(len(idx))],
        }
    )
    bundle = build_pair_path_bundle(
        etf="CLSZ",
        underlying="CLSK",
        pair_daily=daily,
        prices=None,
        start="2026-04-01",
        fill_yahoo=False,
    )
    assert bundle["ok"] is True
    assert bundle["reason"] == "no_price_panel"
    fig = plot_b4_pair_path(bundle)
    assert fig is not None


def test_full_calendar_warmup_yields_many_model_rebals() -> None:
    """Cadence on long panel calendar should not collapse to ~2 rebals."""
    cal = pd.bdate_range("2025-10-01", periods=200)
    rng = np.random.default_rng(0)
    und = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, size=len(cal))))
    etf = und * 0.5 * np.exp(np.cumsum(rng.normal(0, 0.01, size=len(cal))))
    px = pd.DataFrame({"a_px": etf, "b_px": und}, index=cal)
    book = pd.bdate_range("2026-04-01", periods=60)
    daily = pd.DataFrame(
        {
            "date": book,
            "ETF": "QBTZ",
            "Underlying": "QBTS",
            "etf_usd": -1000.0,
            "underlying_usd": 450.0,
            "hedge_ratio": 0.45,
            "cum_pnl": np.linspace(0, 100, len(book)),
            "daily_pnl": 1.0,
            "is_rebalance": [1 if i % 5 == 0 else 0 for i in range(len(book))],
        }
    )
    bundle = build_pair_path_bundle(
        etf="QBTZ",
        underlying="QBTS",
        pair_daily=daily,
        prices=px,
        start="2026-04-01",
        fill_yahoo=False,
    )
    assert bundle["ok"]
    # With ~60 book days and max_interval ~10–14, expect clearly > 2 model rebals.
    assert len(bundle["model_rebals"]) >= 3


def test_plan_history_loader(tmp_path) -> None:
    p = tmp_path / "2026-05-01.csv"
    p.write_text(
        "ETF,Underlying,sleeve,Delta,gross_target_usd,long_usd,short_usd\n"
        "QBTZ,QBTS,inverse_decay_bucket4,-2.0,10000,-4000,-6000\n"
        "NVDL,NVDA,core_leveraged,2.0,5000,3000,-2000\n",
        encoding="utf-8",
    )
    hist = load_b4_plan_history(tmp_path)
    assert len(hist) == 1
    assert hist.iloc[0]["ETF"] == "QBTZ"
    summ = summarize_b4_plan_history(hist)
    assert float(summ.iloc[0]["gross_b4"]) == 10000.0
