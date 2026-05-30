from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from risk_dashboard.metrics import (
    bucket_exposure_component_sums,
    evaluate_exposure_reconciliation,
    resolve_nav_usd,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "data" / "runs"
SNAPSHOT = PROJECT_ROOT / "risk_dashboard" / "data" / "latest.json"


def test_exposure_reconciliation_includes_unbucketed_net():
    totals = {
        "gross_exposure_total": 1_000_000.0,
        "net_exposure_total": 100.0,
        "gross_exposure_bucket_1": 800_000.0,
        "gross_exposure_bucket_2": 150_000.0,
        "gross_exposure_bucket_4": 50_000.0,
        "net_exposure_bucket_1": 10.0,
        "net_exposure_bucket_2": 20.0,
        "net_exposure_bucket_4": 30.0,
        "net_exposure_unbucketed": 40.0,
        "exposure_reconciliation_tol_gross_pct": 0.001,
        "exposure_reconciliation_tol_net_abs_usd": 500.0,
    }
    _, bucket_net = bucket_exposure_component_sums(totals)
    assert bucket_net == pytest.approx(100.0)
    recon = evaluate_exposure_reconciliation(totals)
    assert recon["reconciles"] is True


def test_exposure_reconciliation_uses_accounting_tolerances_not_hardcoded_one_pct():
    totals = {
        "gross_exposure_total": 1_000_000.0,
        "net_exposure_total": 0.0,
        "gross_exposure_bucket_1": 1_005_000.0,
        "gross_exposure_bucket_2": 0.0,
        "gross_exposure_bucket_4": 0.0,
        "net_exposure_bucket_1": 0.0,
        "net_exposure_bucket_2": 0.0,
        "net_exposure_bucket_4": 0.0,
        "net_exposure_unbucketed": 0.0,
        "exposure_reconciliation_tol_gross_pct": 0.01,
        "exposure_reconciliation_tol_net_abs_usd": 500.0,
    }
    recon = evaluate_exposure_reconciliation(totals)
    assert recon["reconciles"] is True
    assert recon["gross_diff_pct"] == pytest.approx(0.005)


def test_totals_json_reconciles_on_latest_run():
    """CI parity: dashboard gate must pass on-disk totals.json when present."""
    totals_path = Path("data/runs/2026-05-29/accounting/totals.json")
    if not totals_path.is_file():
        pytest.skip(f"missing {totals_path}")
    totals = json.loads(totals_path.read_text(encoding="utf-8"))
    recon = evaluate_exposure_reconciliation(totals)
    assert recon["reconciles"], (
        f"totals.json failed dashboard reconciliation: gross diff {recon['gross_diff_pct']:.4%}, "
        f"net abs diff ${recon['net_diff_abs_usd']:,.0f}"
    )


def test_resolve_nav_prefers_totals_over_cli_fallback():
    nav, source = resolve_nav_usd({"nav_usd": 950_000, "nav_source": "test"}, cli_fallback=800_000)
    assert nav == pytest.approx(950_000)
    assert source == "test"


# ── Phase 5: published dashboard snapshot <-> accounting CSV <-> email parity ──


def _load_committed_snapshot() -> tuple[dict, str, Path]:
    if not SNAPSHOT.is_file():
        pytest.skip(f"missing {SNAPSHOT}")
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    run_date = str(snap.get("run_date") or "")
    acc = RUNS / run_date / "accounting"
    if not run_date or not acc.is_dir():
        pytest.skip(f"snapshot run_date {run_date!r} has no accounting dir on disk")
    return snap, run_date, acc


def test_snapshot_pnl_rows_match_accounting_csvs():
    """The committed dashboard snapshot must equal the accounting CSVs it was
    built from, for the snapshot's own run_date.

    This is the core Phase 5 guardrail: the dashboard performs no PnL rollup of
    its own (``compute_bucket_detail`` is pass-through), so every per-underlying
    row in ``latest.json`` must match ``pnl_<bucket>.csv`` exactly -- same row
    set, same ``total_pnl``, same ``symbols`` string. A divergence means the
    snapshot is stale relative to accounting (the historical failure mode).
    """
    snap, run_date, acc = _load_committed_snapshot()

    checked_any = False
    for bucket in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"):
        csv_path = acc / f"pnl_{bucket}.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if "underlying" not in df.columns or "total_pnl" not in df.columns:
            continue
        rows = snap.get("buckets", {}).get(bucket, {}).get("pnl_rows", [])
        checked_any = True

        csv_by_u = {
            str(r["underlying"]): r for _, r in df.iterrows()
        }
        assert len(rows) == len(csv_by_u), (
            f"{bucket}: snapshot has {len(rows)} rows but CSV has {len(csv_by_u)} "
            f"underlyings (stale snapshot for {run_date}?)"
        )
        for r in rows:
            u = str(r["underlying"])
            assert u in csv_by_u, f"{bucket}: snapshot underlying {u} absent from CSV"
            assert float(r["total_pnl"]) == pytest.approx(
                float(csv_by_u[u]["total_pnl"]), rel=1e-9, abs=1e-6
            ), f"{bucket}/{u}: total_pnl mismatch snapshot vs CSV"
            if "symbols" in df.columns and pd.notna(csv_by_u[u].get("symbols")):
                assert str(r.get("symbols")) == str(csv_by_u[u]["symbols"]), (
                    f"{bucket}/{u}: symbols mismatch snapshot "
                    f"{r.get('symbols')!r} vs CSV {csv_by_u[u]['symbols']!r}"
                )

    assert checked_any, f"no pnl_<bucket>.csv files found under {acc}"


def test_snapshot_bucket_totals_match_email_headline():
    """Dashboard bucket totals must equal ``totals.json['bucket_pnl']`` -- the
    same field the EOD PnL email uses for its headline sleeve numbers. This is
    the literal "dashboard shows the same numbers as the email" contract.
    """
    snap, run_date, acc = _load_committed_snapshot()
    totals_path = acc / "totals.json"
    if not totals_path.is_file():
        pytest.skip(f"missing {totals_path}")
    bucket_pnl = json.loads(totals_path.read_text(encoding="utf-8")).get("bucket_pnl") or {}
    if not bucket_pnl:
        pytest.skip("totals.json carries no bucket_pnl")

    for bucket, headline in bucket_pnl.items():
        rows = snap.get("buckets", {}).get(bucket, {}).get("pnl_rows", [])
        snap_sum = sum(float(r.get("total_pnl", 0.0) or 0.0) for r in rows)
        assert snap_sum == pytest.approx(float(headline), rel=1e-6, abs=1.0), (
            f"{bucket}: dashboard row sum {snap_sum:,.2f} != email headline "
            f"bucket_pnl {float(headline):,.2f} for {run_date}"
        )
