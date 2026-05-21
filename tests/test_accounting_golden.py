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
    """05-21 forward fix: MSTR spot rolls up with yieldboost ETFs in bucket 2."""
    pnl = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_bucket_2.csv"))
    row = pnl.loc[pnl["underlying"] == "MSTR"].iloc[0]
    symbols = str(row["symbols"])
    assert "MSTR" in symbols
    assert all(s in symbols for s in ("MSTW", "MSTY", "MTYY"))
    assert float(row["total_pnl"]) == pytest.approx(5763.15, rel=1e-3)

    by_sym = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_symbol.csv"))
    spot_b2 = by_sym[
        (by_sym["symbol"] == "MSTR")
        & (by_sym["underlying"] == "MSTR")
        & (by_sym["bucket"] == "bucket_2")
    ]
    assert len(spot_b2) == 1
    assert float(spot_b2.iloc[0]["total_pnl"]) == pytest.approx(2641.64, rel=1e-2)


def test_2026_05_21_yieldboost_spot_in_bucket2() -> None:
    """SMCI / IONQ / IBIT spot PnL rolls into bucket 2 with income ETF legs."""
    by_sym = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_symbol.csv"))
    for u, etf, spot_approx in (
        ("SMCI", "SMYY", 2577.80),
        ("IONQ", "IOYY", 8015.05),
        ("IBIT", "XBTY", -1240.64),
    ):
        b2 = by_sym[(by_sym["underlying"] == u) & (by_sym["bucket"] == "bucket_2")]
        assert u in set(b2["symbol"]), f"{u} spot missing from bucket 2"
        assert etf in set(b2["symbol"]), f"{etf} missing from {u} bucket 2"
        spot = b2.loc[b2["symbol"] == u, "total_pnl"].iloc[0]
        assert float(spot) == pytest.approx(spot_approx, rel=1e-2)


def test_2026_05_21_b1_sum_after_yieldboost_fix() -> None:
    """B1 sleeve drops when yieldboost spot moves to B2; book total unchanged."""
    b1 = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_bucket_1.csv"))
    bb = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_bucket.csv"))
    assert float(b1["total_pnl"].sum()) == pytest.approx(21349.46, rel=1e-3)
    assert float(bb["total_pnl"].sum()) == pytest.approx(59320.58, rel=1e-3)


def test_snapshot_bucket2_mstr_matches_csv() -> None:
    """Dashboard snapshot must read the same 05-21 bucket-2 CSV totals."""
    run_dir = RUNS / "2026-05-21" / "accounting"
    pnl_csv = _require(run_dir / "pnl_bucket_2.csv")
    net_csv = _require(run_dir / "net_exposure_bucket_2.csv")

    detail = compute_bucket_detail("bucket_2", pnl_csv, net_csv)
    mstr_rows = [r for r in detail["pnl_rows"] if r["underlying"] == "MSTR"]
    assert len(mstr_rows) == 1
    csv_total = float(
        pd.read_csv(pnl_csv).loc[lambda d: d["underlying"] == "MSTR", "total_pnl"].iloc[0]
    )
    assert mstr_rows[0]["total_pnl"] == pytest.approx(csv_total, rel=1e-9)

    snap_path = _require(PROJECT_ROOT / "risk_dashboard" / "data" / "latest.json")
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    assert snap.get("run_date") == "2026-05-21"
    snap_mstr = next(
        r for r in snap["buckets"]["bucket_2"]["pnl_rows"] if r["underlying"] == "MSTR"
    )
    assert snap_mstr["symbols"] == mstr_rows[0]["symbols"]
    assert snap_mstr["total_pnl"] == pytest.approx(csv_total, rel=1e-9)
