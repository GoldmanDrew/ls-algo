"""Tests for the iteration toggles added to ``_decay_score_weights`` and the cap stack:

- Score concavity (Candidate A): signed-power exponent < 1 flattens the top tail.
- Sigma-aware sizing (Candidate B): names with higher pair-spread sigma get smaller weight.
- EMA score blending (Stability #1): repeated calls with identical inputs and ``rho > 0``
  converge weights to the steady state and the state dict is persisted in place.
- Per-pair weight hysteresis (Stability #2): small deltas snap to the prior gross book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from generate_trade_plan import _decay_score_weights, apply_gross_sizing_book_caps


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ETF": ["AAA", "BBB", "CCC", "DDD"],
            "Underlying": ["U1", "U2", "U3", "U4"],
            "Beta": [2.0, 2.0, 2.0, 2.0],
            "beta_abs": [2.0, 2.0, 2.0, 2.0],
            "blended_gross_decay": [0.10, 0.20, 0.40, 0.80],
            "borrow_current": [0.0, 0.0, 0.0, 0.0],
            "net_edge_p50_annual": [0.10, 0.20, 0.40, 0.80],
        }
    )


def test_concavity_flattens_top_tail():
    """Candidate A: p < 1 reduces the gap between best and worst names."""
    df = _base_df()
    cfg = {"sizing_signal": "blended_decay", "borrow_aversion": 0.0, "max_name_weight": 1.0}
    w_lin = _decay_score_weights(df, cfg, sleeve_name="core_leveraged")
    w_con = _decay_score_weights(
        df,
        {**cfg, "score_concavity_p": 0.85},
        sleeve_name="core_leveraged",
    )
    # Concavity must shrink the top weight and grow the bottom weight relative to linear.
    assert float(w_con[-1]) < float(w_lin[-1])
    assert float(w_con[0]) > float(w_lin[0])
    assert abs(float(w_con.sum()) - 1.0) < 1e-9


def test_sigma_aware_downweights_high_vol_pairs():
    """Candidate B: high pair-spread sigma -> smaller weight when score is identical."""
    df = pd.DataFrame(
        {
            "ETF": ["AAA", "BBB"],
            "Underlying": ["U1", "U2"],
            "Beta": [2.0, 2.0],
            "beta_abs": [2.0, 2.0],
            "blended_gross_decay": [0.20, 0.20],
            "borrow_current": [0.0, 0.0],
        }
    )
    sigma_map = {("AAA", "U1"): 0.30, ("BBB", "U2"): 0.90}  # B is 3x as volatile
    cfg = {
        "sizing_signal": "blended_decay",
        "borrow_aversion": 0.0,
        "max_name_weight": 1.0,
        "sigma_aware_sizing": True,
        "sigma_aware_floor": 0.10,
        "sigma_aware_min_mult": 0.25,
        "sigma_aware_max_mult": 4.0,
    }
    w = _decay_score_weights(df, cfg, sleeve_name="core_leveraged", pair_sigma_map=sigma_map)
    assert float(w[0]) > float(w[1])  # AAA has lower sigma -> heavier
    assert abs(float(w.sum()) - 1.0) < 1e-9


def test_score_ema_state_persists_and_blends():
    """Stability #1: rho > 0 stores blended scores in the supplied dict (in-place)."""
    df = _base_df()
    cfg = {
        "sizing_signal": "blended_decay",
        "borrow_aversion": 0.0,
        "max_name_weight": 1.0,
        "score_ema_rho": 0.5,
    }
    state: dict[tuple[str, str], float] = {}
    _decay_score_weights(df, cfg, sleeve_name="core_leveraged", score_ema_state=state)
    assert state, "EMA state was not populated"
    keys = {("AAA", "U1"), ("BBB", "U2"), ("CCC", "U3"), ("DDD", "U4")}
    assert keys.issubset(set(state.keys()))
    # Re-call with all-zero signals; EMA should pull blended scores halfway toward zero.
    df2 = df.copy()
    df2["blended_gross_decay"] = 0.0
    df2["net_edge_p50_annual"] = 0.0
    prev_state = dict(state)
    _decay_score_weights(df2, cfg, sleeve_name="core_leveraged", score_ema_state=state)
    for k, prev in prev_state.items():
        new = state[k]
        assert abs(new - 0.5 * prev) < 1e-9


def test_weight_hysteresis_snaps_small_changes():
    """Stability #2: small deltas vs ``prev_gross_by_pair`` snap to the prior book."""
    sized = pd.DataFrame(
        {
            "ETF": ["AAA", "BBB"],
            "Underlying": ["U1", "U2"],
            "Beta": [2.0, 2.0],
            "beta_abs": [2.0, 2.0],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "gross_target_usd": [510.0, 490.0],  # tiny tilt vs prior 500/500
            "borrow_price_ref": [50.0, 50.0],
            "shares_available": [1e9, 1e9],
        }
    )
    strategy = {
        "gross_sizing_caps": {
            "enabled": True,
            "max_pair_weight_cap": 0.99,
            "max_underlying_weight_cap": 0.99,
            "missing_shares_cap": 0.99,
            "aum_use_pct": 0.0,
            "short_avail_use_pct": 0.0,
            "shares_outstanding_use_frac": 0.0,
            "median_daily_volume_use_pct": 0.0,
            "weight_hysteresis_abs": 0.05,  # 5% of book
            "weight_hysteresis_rel": 0.10,  # 10% rel
        }
    }
    prev = {("AAA", "U1"): 500.0, ("BBB", "U2"): 500.0}
    out, diag = apply_gross_sizing_book_caps(
        sized,
        target_gross_usd=2_000_000.0,
        beta_floor=0.1,
        strategy=strategy,
        shares_out_map={},
        prev_gross_by_pair=prev,
    )
    assert diag.get("hysteresis_applied") is True
    g = out["gross_target_usd"].to_numpy(float)
    # Both pairs should be very close to 500 / 500 (snap-back).
    assert abs(g[0] - 500.0) < 1e-3 and abs(g[1] - 500.0) < 1e-3
