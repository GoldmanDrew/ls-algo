import numpy as np
import pandas as pd

from generate_trade_plan import (
    _enforce_max_pair_weight_within_deployed_sleeve_gross,
    apply_gross_sizing_book_caps,
    rescale_gross_targets_to_sleeve_budget_weights,
)


def _base_sized() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ETF": ["AAA", "BBB"],
            "Underlying": ["UUU", "UUU"],
            "Delta": [2.0, 2.0],
            "delta_abs": [2.0, 2.0],
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
        delta_floor=0.1,
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
        delta_floor=0.1,
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
            "Delta": [2.0, 2.0],
            "delta_abs": [2.0, 2.0],
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
        delta_floor=0.1,
        strategy=strategy,
        shares_out_map={},
    )
    assert diag.get("applied")
    assert abs(float(out["gross_target_usd"].sum()) - 1000.0) < 1e-5
    w = out["gross_target_usd"].to_numpy(float) / 1000.0
    assert np.all(w <= 0.50 + 1e-8)


def test_deployed_liquidity_anchor_uses_sum_gross_not_yaml_target():
    """With ``liquidity_book_reference: deployed_book``, liquidity ladders use current book sum."""
    df = pd.DataFrame(
        {
            "ETF": ["X", "Y"],
            "Underlying": ["U1", "U2"],
            "Delta": [2.0, 2.0],
            "delta_abs": [2.0, 2.0],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "gross_target_usd": [400.0, 600.0],
            "borrow_price_ref": [100.0, 100.0],
            "shares_available": [1000.0, 1000.0],
        }
    )
    strategy = {
        "gross_sizing_caps": {
            "enabled": True,
            "liquidity_book_reference": "deployed_book",
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
        delta_floor=0.1,
        strategy=strategy,
        shares_out_map={},
    )
    assert diag.get("applied")
    assert abs(float(diag.get("liquidity_book_anchor_usd", 0)) - 1000.0) < 1e-5
    assert abs(float(out["gross_target_usd"].sum()) - 1000.0) < 1e-5


def _three_sleeve_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ETF": ["A", "B", "C"],
            "Underlying": ["U1", "U2", "U3"],
            "Delta": [2.0, 2.0, -1.0],
            "delta_abs": [2.0, 2.0, 1.0],
            "sleeve": ["core_leveraged", "yieldboost", "inverse_decay_bucket4"],
            "gross_target_usd": [500_000.0, 200_000.0, 200_000.0],
            "borrow_price_ref": [50.0, 50.0, 50.0],
            "shares_available": [1e9, 1e9, 1e9],
        }
    )


def _three_sleeve_bud() -> dict[str, float]:
    return {"core_leveraged": 600_000.0, "yieldboost": 280_000.0, "inverse_decay_bucket4": 120_000.0}


def test_rescale_sleeves_absolute_budget_default():
    """Default mode rescales each sleeve's deployed sum to its YAML absolute dollar budget."""
    tg = 1_000_000.0
    out, diag = rescale_gross_targets_to_sleeve_budget_weights(
        _three_sleeve_df(), target_gross_usd=tg, sleeve_budget_usd=_three_sleeve_bud(),
    )
    assert diag.get("applied")
    assert diag.get("rescale_to") == "absolute_budget"
    core = float(out.loc[out["sleeve"] == "core_leveraged", "gross_target_usd"].sum())
    yb = float(out.loc[out["sleeve"] == "yieldboost", "gross_target_usd"].sum())
    b4 = float(out.loc[out["sleeve"] == "inverse_decay_bucket4", "gross_target_usd"].sum())
    np.testing.assert_allclose(core, 600_000.0, rtol=1e-8)
    np.testing.assert_allclose(yb, 280_000.0, rtol=1e-8)
    np.testing.assert_allclose(b4, 120_000.0, rtol=1e-8)
    np.testing.assert_allclose(float(out["gross_target_usd"].sum()), 1_000_000.0, rtol=1e-8)


def test_rescale_sleeves_fraction_of_deployed_legacy():
    """``fraction_of_deployed`` preserves the current total and aligns sleeve fractions only."""
    tg = 1_000_000.0
    out, diag = rescale_gross_targets_to_sleeve_budget_weights(
        _three_sleeve_df(),
        target_gross_usd=tg,
        sleeve_budget_usd=_three_sleeve_bud(),
        rescale_to="fraction_of_deployed",
    )
    assert diag.get("applied")
    assert diag.get("rescale_to") == "fraction_of_deployed"
    S = 900_000.0
    core = float(out.loc[out["sleeve"] == "core_leveraged", "gross_target_usd"].sum())
    yb = float(out.loc[out["sleeve"] == "yieldboost", "gross_target_usd"].sum())
    b4 = float(out.loc[out["sleeve"] == "inverse_decay_bucket4", "gross_target_usd"].sum())
    assert abs(core / S - 0.60) < 1e-5
    assert abs(yb / S - 0.28) < 1e-5
    assert abs(b4 / S - 0.12) < 1e-5
    np.testing.assert_allclose(float(out["gross_target_usd"].sum()), S, rtol=1e-8)


def test_vol_etp_bucket5_uses_inverse_short_leg_fraction():
    from generate_trade_plan import VOL_ETP_BUCKET5_SLEEVE, _short_leg_frac_array

    out = _short_leg_frac_array(
        np.array([2.0, 2.0]),
        0.1,
        np.array(["inverse_decay_bucket4", VOL_ETP_BUCKET5_SLEEVE]),
    )
    np.testing.assert_allclose(out, np.array([1.0, 1.0]))


def test_pre_cap_score_haircut_blocks_low_score_uplift_to_pair_cap():
    """A low-score row (e.g. high-borrow YB name) should not be lifted to ``max_pair_weight`` by
    the cap projector's headroom redistribution when ``pre_cap_score_haircut_multiplier`` is set."""
    df = pd.DataFrame(
        {
            "ETF": ["A", "B", "C", "D", "E"],
            "Underlying": ["U1", "U2", "U3", "U4", "U5"],
            "Delta": [2.0] * 5,
            "delta_abs": [2.0] * 5,
            "sleeve": ["yb"] * 5,
            "gross_target_usd": [400.0, 400.0, 400.0, 30.0, 30.0],
            "borrow_price_ref": [50.0] * 5,
            "shares_available": [1e9] * 5,
        }
    )
    base = {
        "enabled": True,
        "max_pair_weight_cap": 0.99,
        "max_underlying_weight_cap": 0.99,
        "aum_use_pct": 0.0,
        "short_avail_use_pct": 0.0,
        "missing_shares_cap": 0.99,
        "shares_outstanding_use_frac": 0.0,
        "median_daily_volume_use_pct": 0.0,
        "per_sleeve": {"yb": {"max_pair_weight": 0.20, "max_underlying_weight": 0.99}},
    }
    out_no, _ = apply_gross_sizing_book_caps(
        df,
        target_gross_usd=10_000.0,
        delta_floor=0.1,
        strategy={"gross_sizing_caps": base},
        shares_out_map={},
    )
    cfg_with = {**base, "per_sleeve": {"yb": {**base["per_sleeve"]["yb"], "pre_cap_score_haircut_multiplier": 1.5}}}
    out_yes, diag_yes = apply_gross_sizing_book_caps(
        df,
        target_gross_usd=10_000.0,
        delta_floor=0.1,
        strategy={"gross_sizing_caps": cfg_with},
        shares_out_map={},
    )

    g_no = out_no["gross_target_usd"].to_numpy(float)
    g_yes = out_yes["gross_target_usd"].to_numpy(float)
    w_low_no = g_no[4] / float(g_no.sum())
    w_low_yes = g_yes[4] / float(g_yes.sum())
    sleeve_cap = 0.20
    assert w_low_no >= 0.15, (
        f"setup sanity: without haircut the low-score row should be lifted via headroom redistribution; got {w_low_no:.4f}"
    )
    assert w_low_yes < 0.5 * sleeve_cap + 1e-6, (
        f"haircut should hold low-score row well below sleeve max_pair_weight={sleeve_cap}, got {w_low_yes:.4f}"
    )
    assert w_low_yes < w_low_no * 0.5, (
        f"haircut should materially reduce the low-score row vs no-haircut: {w_low_yes:.4f} vs {w_low_no:.4f}"
    )
    hc = diag_yes.get("pre_cap_score_haircut") or {}
    assert "yb" in hc and abs(hc["yb"]["pre_cap_score_haircut_multiplier"] - 1.5) < 1e-9
    assert int(hc["yb"]["n_rows_capped_by_haircut"]) >= 1


def test_per_sleeve_concentration_vs_book_level():
    """Per-sleeve caps apply to weights within each sleeve's gross (DCQ-style)."""
    df = pd.DataFrame(
        {
            "ETF": ["A", "B", "C"],
            "Underlying": ["U1", "U2", "U3"],
            "Delta": [2.0, 2.0, 2.0],
            "delta_abs": [2.0, 2.0, 2.0],
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
        delta_floor=0.1,
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
