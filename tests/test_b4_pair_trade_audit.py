"""Unit tests for B4 per-pair trade audit."""
from __future__ import annotations

import pandas as pd

from scripts.b4_pair_trade_audit import build_b4_trade_ledger, realism_checklist


def test_ledger_classifies_enter_exit_resize() -> None:
    ps = pd.DataFrame(
        [
            {
                "ETF": "ASTN",
                "Underlying": "ASTS",
                "sleeve": "inverse_decay_bucket4",
                "pnl_usd": -100.0,
                "txn_cost_usd": 50.0,
                "rebalance_dates": "2026-04-23;2026-04-24;2026-05-27",
            }
        ]
    )
    daily = pd.DataFrame(
        [
            {"date": "2026-04-23", "ETF": "ASTN", "Underlying": "ASTS", "sleeve": "inverse_decay_bucket4",
             "etf_usd": -1000.0, "underlying_usd": -800.0, "txn_cost": 20.0},
            {"date": "2026-04-24", "ETF": "ASTN", "Underlying": "ASTS", "sleeve": "inverse_decay_bucket4",
             "etf_usd": 0.0, "underlying_usd": 0.0, "txn_cost": 20.0},
            {"date": "2026-05-26", "ETF": "ASTN", "Underlying": "ASTS", "sleeve": "inverse_decay_bucket4",
             "etf_usd": -900.0, "underlying_usd": -700.0, "txn_cost": 0.0},
            {"date": "2026-05-27", "ETF": "ASTN", "Underlying": "ASTS", "sleeve": "inverse_decay_bucket4",
             "etf_usd": -1100.0, "underlying_usd": -900.0, "txn_cost": 10.0},
        ]
    )
    ledger, summary = build_b4_trade_ledger(pair_stats=ps, pair_daily=daily, price_panel=None)
    assert list(ledger["reason"]) == ["enter", "exit", "resize"]
    assert int(summary.iloc[0]["n_enter"]) == 1
    assert int(summary.iloc[0]["n_exit"]) == 1
    assert int(summary.iloc[0]["rapid_churn_events"]) == 1


def test_realism_checklist_nonempty() -> None:
    df = realism_checklist()
    assert "dimension" in df.columns and len(df) >= 5
