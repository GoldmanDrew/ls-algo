import numpy as np
import pandas as pd

from generate_trade_plan import (
    _enforce_max_pair_weight_within_deployed_sleeve_gross,
    apply_gross_sizing_book_caps,
)


def _base_sized() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ETF": ["AAA", "BBB"],
            "Underlying": ["UUU", "UUU"],
            "Beta": [2.0, 2.0],
            "beta_abs": [2.0, 2.0],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "gross_target_usd": [500.0, 500.0],
            "borrow_price_ref": [50.0, 50.0],
            "shares_available": [1e9, 1e9],
        }
    )


def test_gross_caps_disabled_noop():
    df = _base_sized()
    out, diag = apply_gross_sizing_book_caps(
        df,
        target_gross_usd=1_000_000.0,
        beta_floor=0.1,
        strategy={},
        shares_out_map={},
    )
    assert not diag.get("applied")
    assert float(out["gross_target_usd"].sum()) == 1000.0


def test_max_underlying_redistributes_mass():
    df = _base_sized()
    strategy = {
        "gross_sizing_caps": {
            "enabled": True,
            "max_pair_weight_cap": 0.99,
            "max_underlying_weight_cap": 0.35,
            "aum_use_pct": 0.0,
            "short_avail_use_pct": 0.0,
            "missing_shares_cap": 0.99,
            "shares_outstanding_use_frac": 0.0,
            "median_daily_volume_use_pct": 0.0,
        }
    }
    out, diag = apply_gross_sizing_book_caps(
        df,
        target_gross_usd=1000.0,
        beta_floor=0.1,
        strategy=strategy,
        shares_out_map={},
    )
    assert diag.get("applied")
    # Single underlying with cap 35% of book ? only 35% of reference gross can be placed here.
    assert abs(float(out["gross_target_usd"].sum()) - 350.0) < 1e-4
    w = out["gross_target_usd"].to_numpy(float) / 1000.0
    assert float(w.sum()) <= 0.35 + 1e-8


def test_liquidity_cap_tightens_pair_weight():
    # Two different underlyings; tiny shares_available forces low per-pair cap vs book gross.
    df = pd.DataFrame(
        {
            "ETF": ["X", "Y"],
            "Underlying": ["U1", "U2"],
            "Beta": [2.0, 2.0],
            "beta_abs": [2.0, 2.0],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "gross_target_usd": [400.0, 600.0],
            "borrow_price_ref": [100.0, 100.0],
            "shares_available": [1000.0, 1000.0],
        }
    )
    strategy = {
        "gross_sizing_caps": {
            "enabled": True,
            "max_pair_weight_cap": 0.50,
            "max_underlying_weight_cap": 0.60,
            "aum_use_pct": 0.0,
            "short_avail_use_pct": 0.25,
            "missing_shares_cap": 0.02,
            "shares_outstanding_use_frac": 0.0,
            "median_daily_volume_use_pct": 0.0,
        }
    }
    out, diag = apply_gross_sizing_book_caps(
        df,
        target_gross_usd=10_000.0,
        beta_floor=0.1,
        strategy=strategy,
        shares_out_map={},
    )
    assert diag.get("applied")
    assert abs(float(out["gross_target_usd"].sum()) - 1000.0) < 1e-5
    w = out["gross_target_usd"].to_numpy(float) / 1000.0
    assert np.all(w <= 0.50 + 1e-8)


def test_per_sleeve_concentration_vs_book_level():
    """Per-sleeve caps apply to weights within each sleeve's gross (DCQ-style)."""
    df = pd.DataFrame(
        {
            "ETF": ["A", "B", "C"],
            "Underlying": ["U1", "U2", "U3"],
            "Beta": [2.0, 2.0, 2.0],
            "beta_abs": [2.0, 2.0, 2.0],
            "sleeve": ["core_leveraged", "core_leveraged", "whitelist_stock"],
            "gross_target_usd": [400.0, 400.0, 200.0],
            "borrow_price_ref": [50.0, 50.0, 50.0],
            "shares_available": [1e9, 1e9, 1e9],
        }
    )
    strategy = {
        "gross_sizing_caps": {
            "enabled": True,
            "max_pair_weight_cap": 0.05,
            "max_underlying_weight_cap": 0.05,
            "aum_use_pct": 0.0,
            "short_avail_use_pct": 0.0,
            "missing_shares_cap": 0.99,
            "shares_outstanding_use_frac": 0.0,
            "median_daily_volume_use_pct": 0.0,
            "per_sleeve": {
                "core_leveraged": {"max_pair_weight": 0.60, "max_underlying_weight": 0.99},
                "whitelist_stock": {"max_pair_weight": 0.99, "max_underlying_weight": 0.99},
            },
        }
    }
    out, diag = apply_gross_sizing_book_caps(
        df,
        target_gross_usd=1000.0,
        beta_floor=0.1,
        strategy=strategy,
        shares_out_map={},
    )
    assert diag.get("per_sleeve_enforced") is True
    assert abs(float(out["gross_target_usd"].sum()) - 1000.0) < 1e-4
    core = out.loc[out["sleeve"] == "core_leveraged", "gross_target_usd"].to_numpy(float)
    g_core = float(core.sum())
    assert g_core > 1e-6
    # Each core pair <= 60% of **core** gross (not 60% of whole book).
    assert float(core.max()) <= 0.60 * g_core + 1e-6


def test_max_pair_weight_within_deployed_sleeve_redistributes():
    """No pair may exceed ``max_pair_weight`` of that sleeve's deployed gross (sum of row gross)."""
    sized = pd.DataFrame(
        {
            "sleeve": ["wl"] * 4,
            "ETF": ["a", "b", "c", "d"],
            "Underlying": ["u1", "u2", "u3", "u4"],
        }
    )
    gross = np.array([500.0, 200.0, 200.0, 100.0], dtype=float)
    caps = {"wl": {"max_pair_weight": 0.25}}
    out = _enforce_max_pair_weight_within_deployed_sleeve_gross(gross, sized, caps)
    s_dep = float(out.sum())
    assert abs(s_dep - 1000.0) < 1e-6
    assert float(np.max(out)) <= 0.25 * s_dep + 1e-4


def test_max_pair_within_deployed_relaxes_cap_when_mathematically_tight():
    """With k active names and ``k * max_pair_weight < 1``, equal split needs weight 1/k > cap."""
    sized = pd.DataFrame(
        {
            "sleeve": ["wl"] * 3,
            "ETF": ["a", "b", "c"],
            "Underlying": ["u1", "u2", "u3"],
        }
    )
    gross = np.array([700.0, 200.0, 100.0], dtype=float)
    caps = {"wl": {"max_pair_weight": 0.25}}
    out = _enforce_max_pair_weight_within_deployed_sleeve_gross(gross, sized, caps)
    s_dep = float(out.sum())
    assert abs(s_dep - 1000.0) < 1e-6
    assert float(np.max(out)) <= s_dep / 3.0 + 1e-4


def test_max_pair_within_deployed_clips_lone_active_row():
    """Multiple sleeve rows but only one positive gross: clip to cap * sleeve deployed sum."""
    sized = pd.DataFrame(
        {
            "sleeve": ["wl"] * 3,
            "ETF": ["a", "b", "c"],
            "Underlying": ["u1", "u2", "u3"],
        }
    )
    gross = np.array([800.0, 0.0, 0.0], dtype=float)
    caps = {"wl": {"max_pair_weight": 0.25}}
    out = _enforce_max_pair_weight_within_deployed_sleeve_gross(gross, sized, caps)
    assert abs(float(out[0]) - 200.0) < 1e-6
    assert float(out[1]) == 0.0 and float(out[2]) == 0.0
