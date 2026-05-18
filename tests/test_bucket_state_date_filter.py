"""Regression: trade replay window must compare YYYYMMDD to YYYYMMDD."""

from __future__ import annotations

import pandas as pd

from ibkr_accounting import yyyymmdd_from_run_date, yyyymmdd_normalize


def test_yyyymmdd_from_run_date():
    assert yyyymmdd_from_run_date("2026-05-18") == "20260518"


def test_trade_date_filter_uses_ymd_not_dashed_cutoff():
    """Trades in Jan 2026 must not pass a May 2026 dashed cutoff by string accident."""
    trades = pd.DataFrame(
        {
            "date": ["20260115", "20260518"],
            "amount": [1.0, 2.0],
        }
    )
    prev_cutoff_ymd = yyyymmdd_from_run_date("2026-05-17")
    filtered = trades[trades["date"] > prev_cutoff_ymd]
    assert list(filtered["date"]) == ["20260518"]


def test_yyyymmdd_normalize_accepts_both_formats():
    assert yyyymmdd_normalize("2026-05-18") == "20260518"
    assert yyyymmdd_normalize("20260518") == "20260518"


def test_ratio_at_date_compare_normalized():
    """State series YMD and cash YYYY-MM-DD must compare consistently."""
    state_date = "20260517"
    cash_date = "2026-05-18"
    assert yyyymmdd_normalize(state_date) <= yyyymmdd_normalize(cash_date)
