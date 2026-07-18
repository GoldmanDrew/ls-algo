from rebalance_intents import (
    SameRunIntentLedger,
    clip_against_opposing_intents,
    coalesce_intents,
    project_positions,
    signed_qty,
)
from rebalance_strategy import gate_resize_against_projected_phase3


def test_signed_intent_and_whole_share_projection():
    assert signed_qty({"action": "BUY", "qty": 3}) == 3
    assert signed_qty({"action": "SELL", "qty": 2}) == -2
    assert project_positions(
        {"CRWV": 100},
        [{"symbol": "CRWV", "action": "BUY", "qty": 7}],
    )["CRWV"] == 107


def test_coalesce_exact_cancellation_drops_symbol():
    out = coalesce_intents(
        [
            {"symbol": "CRWV", "action": "BUY", "qty": 259, "phase": "2b"},
            {"symbol": "CRWV", "action": "SELL", "qty": 259, "phase": "3"},
        ]
    )
    assert out == []


def test_coalesce_partial_netting_and_shared_underlying():
    out = coalesce_intents(
        [
            {"symbol": "MU", "action": "BUY", "qty": 20, "phase": "b1"},
            {"symbol": "MU", "action": "SELL", "qty": 7, "phase": "b4"},
        ]
    )
    assert len(out) == 1
    assert out[0]["action"] == "BUY"
    assert out[0]["qty"] == 13
    assert out[0]["netted_qty"] == 14


def test_clip_never_expands_purgatory_or_borrow_capped_intent():
    primary = [
        {
            "symbol": "ETF",
            "action": "SELL",
            "qty": 10,
            "execution_policy": "reduce_only",
            "borrow_cap_qty": 10,
        }
    ]
    adjusted, audit = clip_against_opposing_intents(
        primary,
        [{"symbol": "ETF", "action": "BUY", "qty": 4}],
    )
    assert adjusted[0]["qty"] == 6
    assert adjusted[0]["execution_policy"] == "reduce_only"
    assert adjusted[0]["borrow_cap_qty"] == 10
    assert audit[0]["netted_qty"] == 4


def test_crwv_resize_buy_is_removed_when_projected_reconciliation_sells_it():
    def project_phase3(post_resize):
        if post_resize["CRWV"] == 100:
            return [], {"CRWV": 100}
        assert post_resize["CRWV"] == 359
        return (
            [
                {
                    "symbol": "CRWV",
                    "action": "SELL",
                    "qty": 259,
                    "source_phase": "phase3_projected_reconciliation",
                }
            ],
            {"CRWV": 100},
        )

    adjusted, terminal, audit = gate_resize_against_projected_phase3(
        resize_trades=[
            {
                "symbol": "CRWV",
                "action": "BUY",
                "qty": 259,
                "leg_side": "long_under",
            }
        ],
        strat_pos={"CRWV": 100},
        max_rounds=1,
        project_phase3=project_phase3,
    )
    assert adjusted == []
    assert terminal["CRWV"] == 100
    assert any(row["event"] == "PREDICTED_CHURN_NETTED" for row in audit)
    # July 13 observed path: buy 259 then sell 259. The guard reaches the
    # identical terminal position with zero submitted turnover.
    sequential_turnover_shares = 259 + 259
    guarded_turnover_shares = sum(int(t["qty"]) for t in adjusted)
    assert sequential_turnover_shares == 518
    assert guarded_turnover_shares == 0
    assert terminal["CRWV"] == 100


def test_ledger_blocks_non_risk_reversal_and_allows_explicit_override():
    ledger = SameRunIntentLedger()
    buy = {"symbol": "CRWV", "action": "BUY", "qty": 10}
    ledger.record(buy, phase="phase2b", status="FILLED")

    allowed, _, row = ledger.guard(
        {"symbol": "CRWV", "action": "SELL", "qty": 5},
        phase="phase3",
        allow_risk_override=False,
    )
    assert not allowed
    assert row["event"] == "SAME_RUN_CHURN"

    allowed, trade, row = ledger.guard(
        {"symbol": "CRWV", "action": "SELL", "qty": 5},
        phase="phase3_reconciliation",
        allow_risk_override=True,
        override_reason="partial_fill",
        evidence={"group_drift_usd": 5000},
    )
    assert allowed
    assert trade["churn_guard"] == "RISK_OVERRIDE"
    assert row["reason"] == "partial_fill"


def test_one_submitted_direction_per_symbol_without_override():
    ledger = SameRunIntentLedger()
    ledger.record(
        {"symbol": "MU", "action": "BUY", "qty": 4},
        phase="phase2b",
        status="FILLED",
    )
    for action in ("SELL", "SELL"):
        allowed, _, _ = ledger.guard(
            {"symbol": "MU", "action": action, "qty": 2},
            phase="phase3",
            allow_risk_override=False,
        )
        assert not allowed
    submitted = [
        row for row in ledger.records if row.get("event") == "SUBMITTED"
    ]
    assert submitted == []
