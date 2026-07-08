"""Tests for held-leg reconstruction from accounting detail."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.b4_reconstruct_state import held_inverse_short_by_pair, resolve_held_detail_run_date


def _write_detail(tmp_path: Path, run_date: str, rows: list[dict]) -> None:
    acct = tmp_path / run_date / "accounting"
    acct.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(acct / "net_exposure_bucket_4_detail.csv", index=False)


def test_held_legs_from_detail(tmp_path):
    _write_detail(
        tmp_path,
        "2026-07-07",
        [
            {"underlying": "MSTR", "symbol": "MSTR", "leg_type": "underlying", "gross_notional_usd": 3895.0},
            {"underlying": "MSTR", "symbol": "MSTZ", "leg_type": "etf", "gross_notional_usd": 21317.0},
            {"underlying": "SPCX", "symbol": "SPCG", "leg_type": "etf", "gross_notional_usd": 1500.0},
            {"underlying": "SPCX", "symbol": "SPCX", "leg_type": "underlying", "gross_notional_usd": 0.0},
        ],
    )
    legs = held_inverse_short_by_pair("2026-07-08", runs_root=tmp_path)
    assert ("MSTZ", "MSTR") in legs
    assert legs[("MSTZ", "MSTR")]["inverse_etf_short_usd"] == 21317.0
    assert legs[("MSTZ", "MSTR")]["underlying_short_usd"] == 3895.0
    assert legs[("SPCG", "SPCX")]["inverse_etf_short_usd"] == 1500.0


def test_resolve_uses_latest_on_or_before_run_date(tmp_path):
    _write_detail(
        tmp_path,
        "2026-07-06",
        [{"underlying": "QBTS", "symbol": "QBTZ", "leg_type": "etf", "gross_notional_usd": 1000.0}],
    )
    _write_detail(
        tmp_path,
        "2026-07-07",
        [{"underlying": "QBTS", "symbol": "QBTZ", "leg_type": "etf", "gross_notional_usd": 2000.0}],
    )
    d, p = resolve_held_detail_run_date("2026-07-08", runs_root=tmp_path)
    assert d == "2026-07-07"
    assert p is not None
