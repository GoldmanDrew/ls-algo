"""Smoke tests for risk_dashboard.metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from risk_dashboard.metrics import (
    DEFAULT_LIMITS,
    SLEEVE_TARGET_WEIGHTS,
    compute_book_summary,
    compute_bucket_detail,
)


@pytest.fixture
def fake_totals() -> dict:
    return {
        "run_date": "2026-05-15",
        "total_pnl": 48626.58,
        "net_exposure_total": -752462.23,
        "gross_exposure_total": 4_195_127.28,
        "net_exposure_bucket_1": -707601.71,
        "gross_exposure_bucket_1": 3_966_574.48,
        "net_exposure_bucket_2": -44860.52,
        "gross_exposure_bucket_2": 228_552.80,
        "net_exposure_bucket_3": 85047.29,
        "gross_exposure_bucket_3": 86236.36,
        "net_exposure_bucket_4": 437084.68,
        "gross_exposure_bucket_4": 437084.68,
        "bucket_pnl": {
            "bucket_1": 10714.78,
            "bucket_2": 25381.18,
            "bucket_3": 10610.49,
            "bucket_4": 1920.14,
        },
    }


def test_book_summary_pct_nav(fake_totals):
    book = compute_book_summary(
        totals=fake_totals,
        pnl_by_bucket=pd.DataFrame(),
        nav_usd=800_000.0,
    )
    assert book.gross_notional_usd == pytest.approx(4_195_127.28)
    assert book.gross_exposure_pct_nav == pytest.approx(4_195_127.28 / 800_000.0)
    assert book.pnl_today_pct_nav == pytest.approx(48626.58 / 800_000.0)
    assert len(book.sleeve_table) == 4
    b4 = next(r for r in book.sleeve_table if r["bucket"] == "bucket_4")
    assert b4["target_weight"] == 0.25
    assert b4["actual_weight"] == pytest.approx(437084.68 / 4_195_127.28)


def test_book_summary_breach_when_gross_exceeds(fake_totals):
    book = compute_book_summary(
        totals=fake_totals,
        pnl_by_bucket=pd.DataFrame(),
        nav_usd=800_000.0,
    )
    # 4.19M / 800k = 524% -- way above 250% hard limit
    assert any(b["metric"] == "gross_exposure_pct_nav" for b in book.breaches)
    breach = next(b for b in book.breaches if b["metric"] == "gross_exposure_pct_nav")
    assert breach["status"] == "hard"


def test_compute_bucket_detail_handles_missing_files(tmp_path: Path):
    detail = compute_bucket_detail(
        bucket="bucket_4",
        pnl_csv=tmp_path / "missing_pnl.csv",
        net_exposure_csv=tmp_path / "missing_expo.csv",
    )
    assert detail["bucket"] == "bucket_4"
    assert detail["n_pnl_rows"] == 0
    assert detail["n_exposure_rows"] == 0


def test_default_limits_are_sane():
    for k, v in DEFAULT_LIMITS.items():
        assert "warn" in v and "hard" in v, k
    # Sleeve targets must sum (within rounding) to roughly 1.0 when bucket_3
    # is excluded (b3 is layered, not a fixed slice).
    fixed = sum(
        v for k, v in SLEEVE_TARGET_WEIGHTS.items() if v is not None and k != "bucket_3"
    )
    assert 0.95 <= fixed <= 1.05
