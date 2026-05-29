from __future__ import annotations

import json
from pathlib import Path

import pytest

from risk_dashboard.metrics import (
    bucket_exposure_component_sums,
    evaluate_exposure_reconciliation,
    resolve_nav_usd,
)


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
