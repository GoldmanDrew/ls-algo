"""Golden checks for frozen historical accounting + forward 05-21 MSTR fix."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from risk_dashboard.metrics import compute_bucket_detail

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "data" / "runs"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"missing fixture: {path.relative_to(PROJECT_ROOT)}")
    return path


def test_frozen_2026_05_19_b1_leaderboard() -> None:
    """Historical 05-19 economics stay on committed baseline (GDX #1, RCAT small)."""
    pnl = pd.read_csv(_require(RUNS / "2026-05-19" / "accounting" / "pnl_bucket_1.csv"))
    top = pnl.sort_values("total_pnl", ascending=False).iloc[0]
    assert top["underlying"] == "GDX"
    assert float(top["total_pnl"]) == pytest.approx(4247.66, rel=1e-3)
    rcat = pnl.loc[pnl["underlying"] == "RCAT", "total_pnl"]
    assert len(rcat) == 1
    assert float(rcat.iloc[0]) == pytest.approx(53.0, rel=0.05)


def test_2026_05_21_b1_leaderboard_incremental() -> None:
    """05-21 B1 stays on incremental chain (GDX #1, RCAT small) — not full-replay RCAT spike."""
    pnl = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_bucket_1.csv"))
    top = pnl.sort_values("total_pnl", ascending=False).iloc[0]
    assert top["underlying"] == "GDX"
    assert float(top["total_pnl"]) == pytest.approx(4282.25, rel=1e-3)
    rcat = float(pnl.loc[pnl["underlying"] == "RCAT", "total_pnl"].iloc[0])
    assert rcat == pytest.approx(152.11, rel=0.01)


def test_2026_05_21_mstr_spot_in_bucket2() -> None:
    """05-21: MSTR spot rolls up with yieldboost ETFs in bucket 2.

    Values reflect the net-planned B4 structural short (live from 2026-05-12),
    tracked point-in-time from the trade plan: B4 carries the plan's net short
    (B4 short target minus the B1/B2 long target), and the rest of the MSTR spot
    move stays in B2.
    """
    pnl = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_bucket_2.csv"))
    row = pnl.loc[pnl["underlying"] == "MSTR"].iloc[0]
    symbols = str(row["symbols"])
    assert "MSTR" in symbols
    assert all(s in symbols for s in ("MSTW", "MSTY", "MTYY"))
    assert float(row["total_pnl"]) == pytest.approx(6364.47, rel=1e-3)

    by_sym = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_symbol.csv"))
    spot_b2 = by_sym[
        (by_sym["symbol"] == "MSTR")
        & (by_sym["underlying"] == "MSTR")
        & (by_sym["bucket"] == "bucket_2")
    ]
    assert len(spot_b2) == 1
    assert float(spot_b2.iloc[0]["total_pnl"]) == pytest.approx(3240.75, rel=1e-2)


def test_2026_05_21_yieldboost_spot_in_bucket2() -> None:
    """SMCI / IONQ / IBIT spot PnL rolls into bucket 2 with income ETF legs.

    Names that also hold inverse ETFs carry a net-planned, plan-tracked B4
    structural short (live 2026-05-12), which carves a slice out of the B2 spot
    line when the plan is net short.
    """
    by_sym = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_symbol.csv"))
    for u, etf, spot_approx in (
        ("SMCI", "SMYY", 2503.09),
        ("IONQ", "IOYY", 4565.76),
        ("IBIT", "XBTY", 250.07),
    ):
        b2 = by_sym[(by_sym["underlying"] == u) & (by_sym["bucket"] == "bucket_2")]
        assert u in set(b2["symbol"]), f"{u} spot missing from bucket 2"
        assert etf in set(b2["symbol"]), f"{etf} missing from {u} bucket 2"
        spot = b2.loc[b2["symbol"] == u, "total_pnl"].iloc[0]
        assert float(spot) == pytest.approx(spot_approx, rel=1e-2)


def test_2026_05_21_b1_sum_after_yieldboost_fix() -> None:
    """B1 sleeve and book total after yieldboost + net-planned B4 structural short.

    The book total equals the broker account total (attribution-invariant); only
    the per-bucket split changes when the net-planned, plan-tracked B4 short is on.
    """
    b1 = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_bucket_1.csv"))
    bb = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_bucket.csv"))
    assert float(b1["total_pnl"].sum()) == pytest.approx(28747.13, rel=1e-3)
    assert float(bb["total_pnl"].sum()) == pytest.approx(55851.18, rel=1e-3)


def test_2026_05_21_b4_mstr_plan_structural_underlying_exposure() -> None:
    """MSTR B4 pair view shows plan-implied short underlying (not $0)."""
    detail_path = _require(RUNS / "2026-05-21" / "accounting" / "net_exposure_bucket_4_detail.csv")
    detail = pd.read_csv(detail_path)
    under = detail[
        (detail["underlying"] == "MSTR") & (detail["leg_type"] == "underlying")
    ]
    assert len(under) == 1
    assert float(under.iloc[0]["net_notional_usd"]) < -1000.0


def test_dashboard_bucket_detail_is_passthrough_of_csv() -> None:
    """The dashboard reader (``compute_bucket_detail``) must reproduce the
    accounting CSV verbatim -- it performs no rollup of its own.

    Frozen 05-21 fixture: every bucket-2 row from ``compute_bucket_detail``
    matches ``pnl_bucket_2.csv`` 1:1 on ``total_pnl`` and ``symbols``. The
    end-to-end snapshot<->CSV parity (against whatever run_date the published
    ``latest.json`` carries) is covered by
    ``tests/test_dashboard_accounting_parity.py`` so this test no longer depends
    on a frozen committed snapshot artifact.
    """
    run_dir = RUNS / "2026-05-21" / "accounting"
    pnl_csv = _require(run_dir / "pnl_bucket_2.csv")
    net_csv = _require(run_dir / "net_exposure_bucket_2.csv")

    detail = compute_bucket_detail("bucket_2", pnl_csv, net_csv)
    df = pd.read_csv(pnl_csv)
    csv_by_u = {str(r["underlying"]): r for _, r in df.iterrows()}

    assert len(detail["pnl_rows"]) == len(csv_by_u)
    for row in detail["pnl_rows"]:
        u = str(row["underlying"])
        assert u in csv_by_u
        assert row["total_pnl"] == pytest.approx(float(csv_by_u[u]["total_pnl"]), rel=1e-9)
        if "symbols" in df.columns and pd.notna(csv_by_u[u].get("symbols")):
            assert str(row["symbols"]) == str(csv_by_u[u]["symbols"])
