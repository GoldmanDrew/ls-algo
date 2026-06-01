"""Execution-time guard: Bucket 4 inverse-ETF short leg is never covered (grow-only).

A short leg covers when action == BUY. For the inverse_decay_bucket4 sleeve that
must be converted to a skip; SELL (growing the short) and the underlying leg stay
fully bidirectional.
"""
from __future__ import annotations

import pandas as pd

from phase2b_resize import ResizeBandConfig, build_resize_trades


def _cfg() -> ResizeBandConfig:
    return ResizeBandConfig(enter_band_pct=0.10, exit_band_pct=0.03,
                            min_trim_usd=100.0, min_grow_usd=100.0)


def _plan(sleeve: str) -> pd.DataFrame:
    # short target -50k (inverse ETF), underlying long target -30k (B4 structural short)
    return pd.DataFrame([{
        "Underlying": "TSTU", "ETF": "TSTZ", "sleeve": sleeve,
        "short_usd": -50_000.0, "long_usd": -30_000.0,
    }])


def test_b4_inverse_cover_is_blocked():
    # we are MORE short than target (-80k vs -50k) -> band wants to BUY (cover) 30k
    plan = _plan("inverse_decay_bucket4")
    trades, decisions = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"TSTZ": -8000.0, "TSTU": -3000.0},  # -8000 sh * $10 = -80k
        prices={"TSTZ": 10.0, "TSTU": 10.0},
        purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        target_basis="executable",
    )
    short_legs = [d for d in decisions if d.leg_side == "short_etf"]
    assert short_legs, "expected a short_etf decision"
    sd = short_legs[0]
    assert sd.decision == "skip"
    assert sd.action is None
    assert "b4_ratchet_no_cover" in (sd.reason or "")
    # no BUY trade on the inverse ETF leg
    assert not any(t["leg_side"] == "short_etf" and t["action"] == "BUY" for t in trades)


def test_b4_inverse_grow_is_allowed():
    # we are LESS short than target (-20k vs -50k) -> band wants to SELL (grow) the short
    plan = _plan("inverse_decay_bucket4")
    trades, decisions = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"TSTZ": -2000.0, "TSTU": -3000.0},  # -20k
        prices={"TSTZ": 10.0, "TSTU": 10.0},
        purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        target_basis="executable",
    )
    sell_short = [t for t in trades if t["leg_side"] == "short_etf" and t["action"] == "SELL"]
    assert sell_short, "growing the inverse short (SELL) must be allowed"


def test_non_b4_cover_is_not_blocked():
    # same cover scenario but a B1 LETF sleeve -> BUY (cover) should be allowed
    plan = _plan("core_leveraged")
    trades, decisions = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"TSTZ": -8000.0, "TSTU": -3000.0},
        prices={"TSTZ": 10.0, "TSTU": 10.0},
        purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        target_basis="executable",
    )
    short_legs = [d for d in decisions if d.leg_side == "short_etf"]
    assert short_legs and short_legs[0].decision != "skip"
    assert any(t["leg_side"] == "short_etf" and t["action"] == "BUY" for t in trades)
