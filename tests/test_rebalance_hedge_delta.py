from __future__ import annotations

import pandas as pd
import pytest

from rebalance_strategy import build_hedge_trades, compute_hedge_delta


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
