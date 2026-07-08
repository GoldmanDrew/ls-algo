"""Execution-time guard: Bucket 4 inverse-ETF short leg grow-only with optional trim."""
from __future__ import annotations

import pandas as pd

from phase2b_resize import ResizeBandConfig, build_resize_trades


def _cfg() -> ResizeBandConfig:
    return ResizeBandConfig(enter_band_pct=0.10, exit_band_pct=0.03,
                            min_trim_usd=100.0, min_grow_usd=100.0)


def _plan(sleeve: str, **extra) -> pd.DataFrame:
    row = {
        "Underlying": "TSTU", "ETF": "TSTZ", "sleeve": sleeve,
        "short_usd": -50_000.0, "long_usd": -30_000.0,
        "ratchet_released": False,
        "ratchet_trim_usd": 0.0,
    }
    row.update(extra)
    return pd.DataFrame([row])


def test_b4_inverse_cover_is_blocked():
    plan = _plan("inverse_decay_bucket4")
    trades, decisions = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"TSTZ": -8000.0, "TSTU": -3000.0},
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
    assert not any(t["leg_side"] == "short_etf" and t["action"] == "BUY" for t in trades)


def test_b4_inverse_cover_allowed_when_released():
    plan = _plan(
        "inverse_decay_bucket4",
        ratchet_released=True,
        ratchet_trim_usd=5_000.0,
    )
    trades, decisions = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"TSTZ": -8000.0, "TSTU": -3000.0},
        prices={"TSTZ": 10.0, "TSTU": 10.0},
        purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        target_basis="executable",
        b4_allow_inverse_cover=True,
    )
    short_legs = [d for d in decisions if d.leg_side == "short_etf"]
    assert short_legs and short_legs[0].decision != "skip"
    assert "b4_ratchet_trim_allowed" in (short_legs[0].reason or "") or "b4_ratchet_trim_capped" in (short_legs[0].reason or "")
    buys = [t for t in trades if t["leg_side"] == "short_etf" and t["action"] == "BUY"]
    assert buys
    assert buys[0]["trade_usd_target"] <= 5_000.0 + 1.0


def test_bucket5_vol_etp_cover_is_blocked_without_release():
    plan = _plan("volatility_etp_bucket5")
    trades, decisions = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"TSTZ": -8000.0, "TSTU": -3000.0},
        prices={"TSTZ": 10.0, "TSTU": 10.0},
        purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        target_basis="executable",
        b4_allow_inverse_cover=True,
    )
    short_legs = [d for d in decisions if d.leg_side == "short_etf"]
    assert short_legs and short_legs[0].decision == "skip"
    assert "b4_ratchet_no_cover" in (short_legs[0].reason or "")


def test_b4_inverse_grow_is_allowed():
    plan = _plan("inverse_decay_bucket4")
    trades, decisions = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"TSTZ": -2000.0, "TSTU": -3000.0},
        prices={"TSTZ": 10.0, "TSTU": 10.0},
        purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        target_basis="executable",
    )
    sell_short = [t for t in trades if t["leg_side"] == "short_etf" and t["action"] == "SELL"]
    assert sell_short, "growing the inverse short (SELL) must be allowed"


def test_non_b4_cover_is_not_blocked():
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
