"""Golden checks for restated accounting + yieldboost bucket-2 attribution."""

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


def test_2026_05_19_rcat_not_inflated() -> None:
    """Restated 05-19 must not show the RCAT B1 spike from broken full replay."""
    pnl = pd.read_csv(_require(RUNS / "2026-05-19" / "accounting" / "pnl_bucket_1.csv"))
    rcat = float(pnl.loc[pnl["underlying"] == "RCAT", "total_pnl"].iloc[0])
    assert rcat == pytest.approx(53.45, rel=0.05)
    assert rcat < 500.0


def test_2026_05_21_b1_leaderboard() -> None:
    """05-21 B1 leaders after restate — RCAT stays small."""
    pnl = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_bucket_1.csv"))
    top = pnl.sort_values("total_pnl", ascending=False).iloc[0]
    assert top["underlying"] == "ETHA"
    rcat = float(pnl.loc[pnl["underlying"] == "RCAT", "total_pnl"].iloc[0])
    assert rcat == pytest.approx(152.70, rel=0.02)


def test_2026_05_21_mstr_spot_in_bucket2() -> None:
    """MSTR spot rolls up with yieldboost ETFs in bucket 2."""
    pnl = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_bucket_2.csv"))
    row = pnl.loc[pnl["underlying"] == "MSTR"].iloc[0]
    symbols = str(row["symbols"])
    assert "MSTR" in symbols
    assert all(s in symbols for s in ("MSTW", "MSTY", "MTYY"))
    assert float(row["total_pnl"]) == pytest.approx(5006.73, rel=1e-3)

    by_sym = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_symbol.csv"))
    spot_b2 = by_sym[
        (by_sym["symbol"] == "MSTR")
        & (by_sym["underlying"] == "MSTR")
        & (by_sym["bucket"] == "bucket_2")
    ]
    assert len(spot_b2) == 1
    assert float(spot_b2.iloc[0]["total_pnl"]) > 0.0


def test_2026_05_21_yieldboost_pairs_include_spot_in_bucket2() -> None:
    """Yieldboost B2 rows must include the long spot leg, not ETF-only rollups."""
    sym = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_symbol.csv"))
    b2 = sym[sym.bucket == "bucket_2"]
    for underlying, etf in (("SMCI", "SMYY"), ("IONQ", "IOYY")):
        u_row = b2[(b2.underlying == underlying) & (b2.symbol == underlying)]
        e_row = b2[(b2.underlying == underlying) & (b2.symbol == etf)]
        assert len(u_row) == 1, f"{underlying} spot missing from bucket_2"
        assert len(e_row) == 1, f"{etf} missing from bucket_2"
        assert u_row.iloc[0].total_pnl != e_row.iloc[0].total_pnl


def test_no_etf_only_yieldboost_bucket2_rows_on_2026_05_21() -> None:
    """No yieldboost pair should show ETF-only bucket-2 symbol lists."""
    b2 = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_bucket_2.csv"))
    sym = pd.read_csv(_require(RUNS / "2026-05-21" / "accounting" / "pnl_by_symbol.csv"))
    sym2 = sym[sym.bucket == "bucket_2"]
    offenders = []
    for _, row in b2.iterrows():
        u = row["underlying"]
        syms = sym2[sym2.underlying == u]
        if len(syms) >= 2 and u not in set(syms.symbol):
            offenders.append(u)
    assert offenders == []


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
