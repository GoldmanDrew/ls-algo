"""Tests for the EOD per-pair B4 PnL + hedge chart helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.bucket4_eod_pair_charts import (
    load_b4_pair_leg_history,
    load_pair_gross_and_realized_h,
    make_b4_pair_pnl_hedge_chart,
)


def _write_run(root, ds, *, pair_pnl, etf_pnl, etf_gross=10_000.0, und_gross=4_500.0, delta=-1.0):
    acct = root / ds / "accounting"
    acct.mkdir(parents=True)
    pd.DataFrame([
        {"etf": "QBTZ", "underlying": "QBTS", "delta": delta, "total_pnl": pair_pnl},
    ]).to_csv(acct / "pnl_bucket_4_by_pair.csv", index=False)
    pd.DataFrame([
        {"symbol": "QBTZ", "total_pnl": etf_pnl},
        {"symbol": "QBTS", "total_pnl": pair_pnl - etf_pnl},
    ]).to_csv(acct / "pnl_bucket_4_by_symbol.csv", index=False)
    pd.DataFrame([
        {"underlying": "QBTS", "symbol": "QBTZ", "leg_type": "etf", "gross_notional_usd": etf_gross},
        {"underlying": "QBTS", "symbol": "QBTS", "leg_type": "underlying", "gross_notional_usd": und_gross},
    ]).to_csv(acct / "net_exposure_bucket_4_detail.csv", index=False)


def test_leg_history_splits_pair_into_etf_and_underlying(tmp_path):
    _write_run(tmp_path, "2026-06-01", pair_pnl=1000.0, etf_pnl=700.0)
    _write_run(tmp_path, "2026-06-02", pair_pnl=1500.0, etf_pnl=900.0)
    hist = load_b4_pair_leg_history(tmp_path)
    assert len(hist) == 2
    last = hist.sort_values("date").iloc[-1]
    assert last["pair"] == "QBTZ|QBTS"
    assert last["pair_pnl_cum"] == pytest.approx(1500.0)
    assert last["etf_leg_pnl_cum"] == pytest.approx(900.0)
    assert last["und_leg_pnl_cum"] == pytest.approx(600.0)


def test_realized_h_is_und_gross_over_beta_times_etf_gross(tmp_path):
    _write_run(tmp_path, "2026-06-01", pair_pnl=0.0, etf_pnl=0.0,
               etf_gross=10_000.0, und_gross=4_500.0, delta=-1.0)
    g = load_pair_gross_and_realized_h(tmp_path, "2026-06-01")
    row = g.iloc[0]
    # h = 4500 / (|-1| * 10000) = 0.45
    assert row["realized_h"] == pytest.approx(0.45)
    assert row["etf_gross_usd"] == pytest.approx(10_000.0)


def test_chart_builder_fails_soft_on_empty_root(tmp_path):
    png, csv = make_b4_pair_pnl_hedge_chart(
        "2026-06-01", runs_root=tmp_path, out_dir=tmp_path / "out",
    )
    assert png is None and csv is None
