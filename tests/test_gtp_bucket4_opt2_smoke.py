"""Smoke: bucket4_weekly_opt2 config must construct V6PfParams without error."""
from __future__ import annotations

from strategy_config import load_config
from scripts.v6_b4_pf_weights import V6PfParams


def test_v6_pf_params_from_opt2_dict_accepts_strategy_keys():
    cfg = load_config()
    opt2 = (
        (cfg.get("portfolio") or {})
        .get("sleeves", {})
        .get("inverse_decay_bucket4", {})
        .get("rules", {})
        .get("bucket4_weekly_opt2", {})
        or {}
    )
    p = V6PfParams.from_opt2_dict(opt2, min_pairs=5)
    assert p.min_pairs == 5
    assert p.borrow_aversion_source == "posterior"
    assert p.borrow_uncertainty_penalty == 3.0
    assert p.decay_borrow_quad == 0.0
