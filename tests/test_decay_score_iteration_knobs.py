"""Tests for the iteration toggles added to ``_decay_score_weights`` and the cap stack:

- Score concavity (Candidate A): signed-power exponent < 1 flattens the top tail.
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
            "Delta": [2.0, 2.0, 2.0, 2.0],
            "delta_abs": [2.0, 2.0, 2.0, 2.0],
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


def test_weight_hysteresis_snaps_small_changes():
    """Stability #2: small deltas vs ``prev_gross_by_pair`` snap to the prior book."""
    sized = pd.DataFrame(
        {
            "ETF": ["AAA", "BBB"],
            "Underlying": ["U1", "U2"],
            "Delta": [2.0, 2.0],
            "delta_abs": [2.0, 2.0],
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
        delta_floor=0.1,
        strategy=strategy,
        shares_out_map={},
        prev_gross_by_pair=prev,
    )
    assert diag.get("hysteresis_applied") is True
    g = out["gross_target_usd"].to_numpy(float)
    # Both pairs should be very close to 500 / 500 (snap-back).
    assert abs(g[0] - 500.0) < 1e-3 and abs(g[1] - 500.0) < 1e-3
