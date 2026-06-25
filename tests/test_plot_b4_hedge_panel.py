"""Bucket 4 proposed-trades plot: hedge-ratio sparkline panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from plot_proposed_trades import (
    current_hedge_ratio_by_pair,
    plot_b4_hedge_ratio,
    render_stock_bucket,
)


def test_current_hedge_ratio_by_pair_reads_ratchet_targets(tmp_path: Path, monkeypatch) -> None:
    run_date = "2099-01-01"
    cad_dir = tmp_path / "data" / "runs" / run_date / "b4_hedge_cadence"
    cad_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"ETF": "MSTZ", "Underlying": "MSTR", "hedge_ratio": 0.37}]
    ).to_csv(cad_dir / "b4_ratchet_targets.csv", index=False)

    import plot_proposed_trades as ppt

    monkeypatch.setattr(ppt, "RUNS_DIR", tmp_path / "data" / "runs")
    df = pd.DataFrame([{"ETF": "MSTZ", "Underlying": "MSTR"}])
    got = current_hedge_ratio_by_pair(run_date, df)
    assert got[("MSTZ", "MSTR")] == pytest.approx(0.37)


def test_plot_b4_hedge_ratio_sparkline_runs() -> None:
    df = pd.DataFrame(
        [
            {"ETF": "MSTZ", "Underlying": "MSTR", "long_usd": -100, "short_usd": 100},
            {"ETF": "CONI", "Underlying": "COIN", "long_usd": -50, "short_usd": 50},
        ]
    )
    idx = pd.bdate_range("2026-01-01", periods=80)
    ser = pd.Series(np.linspace(0.32, 0.55, len(idx)), index=idx)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_b4_hedge_ratio(
        ax,
        df,
        hedge_series={"MSTR": ser, "COIN": ser * 0.9 + 0.05},
        current_h={("MSTZ", "MSTR"): 0.54, ("CONI", "COIN"): 0.48},
    )
    plt.close(fig)


def test_render_bucket4_writes_dual_panel(tmp_path: Path, monkeypatch) -> None:
    import plot_proposed_trades as ppt

    run_date = "2099-06-01"
    run_dir = tmp_path / "data" / "runs" / run_date
    run_dir.mkdir(parents=True)
    cad_dir = run_dir / "b4_hedge_cadence"
    cad_dir.mkdir()
    pd.DataFrame(
        [{"ETF": "MSTZ", "Underlying": "MSTR", "hedge_ratio": 0.42}]
    ).to_csv(cad_dir / "b4_ratchet_targets.csv", index=False)

    monkeypatch.setattr(ppt, "RUNS_DIR", tmp_path / "data" / "runs")
    monkeypatch.setattr(ppt, "build_b4_hedge_series_by_underlying", lambda *_a, **_k: {})

    b4 = pd.DataFrame(
        [
            {
                "ETF": "MSTZ",
                "Underlying": "MSTR",
                "Delta": -2.0,
                "long_usd": -5000,
                "short_usd": 5000,
                "optimal_long_usd": -6000,
                "optimal_short_usd": 6000,
                "liquidity_gap_usd": 2000,
                "purgatory": False,
            }
        ]
    )
    out = tmp_path / "b4.png"
    render_stock_bucket(run_date, b4, "Bucket 4 (inverse decay)", out, bucket4=True, hedge_panel=True)
    assert out.is_file() and out.stat().st_size > 1000
