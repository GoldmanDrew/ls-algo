from __future__ import annotations

import pandas as pd
import pytest

from rebalance_intents import SameRunIntentLedger
from rebalance_strategy import (
    build_hedge_trades,
    compute_hedge_delta,
    guard_phase3_against_same_run_churn,
)


def _hedge_rule_kwargs() -> dict[str, float]:
    return {
        "long_trigger_net_pct": 0.04,
        "long_target_net_pct": 0.01,
        "short_trigger_net_pct": 0.01,
        "short_target_net_pct": 0.00,
    }


class TestAsymmetricHedgeDelta:
    def test_net_long_at_trigger_does_not_trade(self) -> None:
        triggered, correction = compute_hedge_delta(
            net_notional=4_000.0,
            target_gross=100_000.0,
            **_hedge_rule_kwargs(),
        )

        assert triggered is False
        assert correction == 0.0

    def test_net_long_above_trigger_hedges_back_to_one_percent(self) -> None:
        triggered, correction = compute_hedge_delta(
            net_notional=4_500.0,
            target_gross=100_000.0,
            **_hedge_rule_kwargs(),
        )

        assert triggered is True
        assert correction == pytest.approx(-3_500.0)

    def test_net_short_at_trigger_does_not_trade(self) -> None:
        triggered, correction = compute_hedge_delta(
            net_notional=-1_000.0,
            target_gross=100_000.0,
            **_hedge_rule_kwargs(),
        )

        assert triggered is False
        assert correction == 0.0

    def test_net_short_beyond_trigger_hedges_back_to_flat(self) -> None:
        triggered, correction = compute_hedge_delta(
            net_notional=-1_500.0,
            target_gross=100_000.0,
            **_hedge_rule_kwargs(),
        )

        assert triggered is True
        assert correction == pytest.approx(1_500.0)


def test_min_trade_usd_is_post_trigger_execution_floor() -> None:
    plan = pd.DataFrame(
        [
            {
                "Underlying": "AAPL",
                "ETF": "AAPU",
                "long_usd": 10_000.0,
                "short_usd": -5_000.0,
            }
        ]
    )

    common = dict(
        hedgeable_plan=plan,
        strat_pos={"AAPL": 585.0, "AAPU": -250.0},
        prices={"AAPL": 100.0, "AAPU": 100.0},
        account_equity=50_000.0,
        gross_leverage=4.0,
        etf_to_under={"AAPU": "AAPL"},
        etf_to_delta={"AAPU": 2.0},
        short_map={},
        blocked_short_etfs=set(),
        flow_etfs=set(),
        blacklist=set(),
        **_hedge_rule_kwargs(),
    )

    assert build_hedge_trades(min_trade_usd=7_000.0, **common) == []

    trades = build_hedge_trades(min_trade_usd=200.0, **common)
    assert len(trades) == 1
    assert trades[0]["correction_usd"] == pytest.approx(-6_500.0)


def _opposing_crwv_trade(**overrides):
    trade = {
        "underlying": "CRWV",
        "symbol": "CRWV",
        "action": "SELL",
        "qty": 5,
        "ref_price": 100.0,
        "net_notional_before": 1_000.0,
        "correction_usd": -1_000.0,
    }
    trade.update(overrides)
    return trade


def _guard_crwv(ledger, trade, actual):
    return guard_phase3_against_same_run_churn(
        trades=[trade],
        ledger=ledger,
        phase="phase3_reconciliation",
        actual_positions={"CRWV": actual},
        prices={"CRWV": 100.0},
        etf_to_under={},
        etf_to_delta={},
        blocked_short_etfs=set(),
        drift_usd_tolerance=500.0,
        drift_share_tolerance=2.0,
    )


def test_reconciliation_blocks_non_risk_same_run_reversal() -> None:
    ledger = SameRunIntentLedger(projected_positions={"CRWV": 110})
    ledger.record(
        {"symbol": "CRWV", "action": "BUY", "qty": 10},
        phase="phase2b",
        status="FILLED",
    )
    approved, audit = _guard_crwv(ledger, _opposing_crwv_trade(), 110)
    assert approved == []
    assert audit[0]["event"] == "SAME_RUN_CHURN"


def test_partial_fill_drift_allows_strict_risk_override() -> None:
    ledger = SameRunIntentLedger(projected_positions={"CRWV": 110})
    ledger.record(
        {"symbol": "CRWV", "action": "BUY", "qty": 10},
        phase="phase2b",
        status="PARTIAL",
    )
    approved, audit = _guard_crwv(ledger, _opposing_crwv_trade(), 100)
    assert len(approved) == 1
    assert approved[0]["churn_guard"] == "RISK_OVERRIDE"
    assert audit[0]["reason"] == "partial_fill"


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        ({"orphan_close": True}, "orphan"),
        ({"execution_policy": "hard_exit"}, "hard_exit"),
    ],
)
def test_orphan_and_hard_exit_reversals_are_explicit_overrides(flags, reason) -> None:
    ledger = SameRunIntentLedger(projected_positions={"CRWV": 110})
    ledger.record(
        {"symbol": "CRWV", "action": "BUY", "qty": 10},
        phase="phase2b",
        status="FILLED",
    )
    approved, audit = _guard_crwv(ledger, _opposing_crwv_trade(**flags), 110)
    assert len(approved) == 1
    assert audit[0]["reason"] == reason
