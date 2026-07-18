"""Tests for per-pair Bucket 4 cadence gate."""
from __future__ import annotations

import pandas as pd

from scripts.bucket4_cadence_gate import (
    evaluate_cadence_gate,
    evaluate_pair_due,
    filter_resize_plan_for_b4_cadence,
    mark_pairs_rebalanced,
)
from scripts.bucket4_hedge_cadence import HedgeCadenceKnobs, PairPolicy, compute_pair_policy


def _policy(interval_days: int = 10, etf: str = "CLSZ", und: str = "CLSK") -> PairPolicy:
    return compute_pair_policy(1.0, 0.05, 0.04, knobs=HedgeCadenceKnobs(base_days=10.0, max_interval=21), etf=etf, underlying=und)


def test_defer_when_interval_not_elapsed():
    pol = _policy(interval_days=10)
    pol = PairPolicy(**{**pol.__dict__, "interval_days": 10})
    dec = evaluate_pair_due(pol, run_date="2026-06-03", last_rebalance="2026-05-28", knobs=HedgeCadenceKnobs(max_interval=21))
    assert dec.due is False
    assert "defer" in dec.reason


def test_due_when_interval_elapsed():
    pol = _policy()
    pol = PairPolicy(**{**pol.__dict__, "interval_days": 5})
    dec = evaluate_pair_due(pol, run_date="2026-06-10", last_rebalance="2026-06-01", knobs=HedgeCadenceKnobs(max_interval=21))
    assert dec.due is True


def test_due_on_first_rebalance():
    dec = evaluate_pair_due(_policy(), run_date="2026-06-03", last_rebalance=None, knobs=HedgeCadenceKnobs(max_interval=21))
    assert dec.due is True
    assert dec.reason == "no_prior_rebalance"


def test_filter_resize_plan_marks_deferred_b4_without_dropping_netting_row():
    df = pd.DataFrame([
        {"sleeve": "core_leveraged", "ETF": "TQQQ", "Underlying": "QQQ", "long_usd": 1000, "short_usd": -500},
        {"sleeve": "inverse_decay_bucket4", "ETF": "CLSZ", "Underlying": "CLSK", "long_usd": -200, "short_usd": -800},
        {"sleeve": "inverse_decay_bucket4", "ETF": "MSTZ", "Underlying": "MSTR", "long_usd": -100, "short_usd": -400},
    ])
    out = filter_resize_plan_for_b4_cadence(df, {("CLSZ", "CLSK")})
    assert len(out) == 3
    assert bool(out.loc[out["ETF"] == "CLSZ", "b4_cadence_due"].iloc[0]) is True
    assert bool(out.loc[out["ETF"] == "MSTZ", "b4_cadence_due"].iloc[0]) is False


def test_purgatory_reduce_only_does_not_bypass_cadence():
    df = pd.DataFrame([
        {
            "sleeve": "inverse_decay_bucket4",
            "ETF": "ASTN",
            "Underlying": "ASTS",
            "purgatory": True,
            "execution_policy": "reduce_only",
        },
        {
            "sleeve": "inverse_decay_bucket4",
            "ETF": "IREZ",
            "Underlying": "IREN",
            "purgatory": True,
            "execution_policy": "reduce_only",
        },
    ])
    out = filter_resize_plan_for_b4_cadence(df, {("IREZ", "IREN")})
    assert bool(out.loc[out["ETF"] == "ASTN", "b4_cadence_due"].iloc[0]) is False
    assert bool(out.loc[out["ETF"] == "IREZ", "b4_cadence_due"].iloc[0]) is True


def test_mark_pairs_rebalanced_updates_state():
    st = mark_pairs_rebalanced({}, [("CLSZ", "CLSK")], "2026-06-03")
    assert st["CLSZ|CLSK"] == "2026-06-03"
