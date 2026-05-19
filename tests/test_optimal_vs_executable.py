"""Dual optimal-vs-executable target pipeline.

Covers:
- ``cap_mode`` partition: ``structural_only`` ignores ``shares_available`` / ADV;
  ``structural_plus_day_liquidity`` (default) honors them. Optimal target should be
  >= executable target whenever day-liquidity binds.
- ``binding_per_row`` labels surface the binding cap (``shares_available`` /
  ``shares_outstanding`` / ``aum`` / ``adv`` / ``pair_cap`` / ``haircut``).
- Harvest ``_resolve_target_basis_columns`` prefers ``optimal_*`` when present.
- Phase 2b ``_resize_target_columns`` returns hybrid pair when optimal columns are present.
"""

import numpy as np
import pandas as pd

from generate_trade_plan import apply_gross_sizing_book_caps, _liquidity_tight_book_weights
from harvest_underexposed_shorts import _resolve_target_basis_columns
from phase2b_resize import _resize_target_columns, build_resize_trades, ResizeBandConfig


def _liq_squeezed_df() -> pd.DataFrame:
    """Two pairs whose pair-cap headroom is large but ``shares_available`` is tight on both."""
    return pd.DataFrame(
        {
            "ETF": ["A", "B"],
            "Underlying": ["U1", "U2"],
            "Delta": [2.0, 2.0],
            "delta_abs": [2.0, 2.0],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "gross_target_usd": [500.0, 500.0],
            "borrow_price_ref": [100.0, 100.0],
            # 2 shares ? $100 = $200 short cap per pair before short_avail_use_pct.
            "shares_available": [2.0, 2.0],
        }
    )


def _strategy_with_per_sleeve(extra: dict | None = None) -> dict:
    s = {
        "gross_sizing_caps": {
            "enabled": True,
            "max_pair_weight_cap": 0.5,
            "max_underlying_weight_cap": 0.6,
            "aum_use_pct": 0.0,
            "short_avail_use_pct": 0.5,
            "missing_shares_cap": 0.5,
            "shares_outstanding_use_frac": 0.0,
            "median_daily_volume_use_pct": 0.0,
            "per_sleeve": {"core_leveraged": {"max_pair_weight": 0.5, "max_underlying_weight": 0.6}},
        }
    }
    if extra:
        s["gross_sizing_caps"].update(extra)
    return s


def test_cap_mode_structural_only_ignores_shares_available():
    """In ``structural_only`` mode, tight ``shares_available`` shouldn't bind — total deployed
    should remain at the original sleeve gross sum (no liquidity squeeze)."""
    df = _liq_squeezed_df()
    s = _strategy_with_per_sleeve()
    # Executable: with 2 shares ? $100 ? 0.5 = $100 short cap each, gross drops well below 1000.
    out_exe, diag_exe = apply_gross_sizing_book_caps(
        df, target_gross_usd=10_000.0, delta_floor=0.1,
        strategy=s, shares_out_map={}, cap_mode=None,
    )
    out_opt, diag_opt = apply_gross_sizing_book_caps(
        df, target_gross_usd=10_000.0, delta_floor=0.1,
        strategy=s, shares_out_map={}, cap_mode="structural_only",
    )
    g_exe = float(out_exe["gross_target_usd"].sum())
    g_opt = float(out_opt["gross_target_usd"].sum())
    assert diag_exe.get("applied") and diag_opt.get("applied")
    assert g_opt > g_exe + 1.0, f"optimal should exceed executable when shares_available is tight: opt={g_opt:.2f} exe={g_exe:.2f}"
    assert abs(g_opt - 1000.0) < 1e-3, f"with no liquidity binders left, optimal should preserve full gross; got {g_opt:.2f}"


def test_cap_mode_default_is_legacy_structural_plus_day_liquidity():
    """``cap_mode=None`` (default) keeps legacy behavior identical to passing the explicit string."""
    df = _liq_squeezed_df()
    s = _strategy_with_per_sleeve()
    out_default, _ = apply_gross_sizing_book_caps(
        df, target_gross_usd=10_000.0, delta_floor=0.1,
        strategy=s, shares_out_map={}, cap_mode=None,
    )
    out_explicit, _ = apply_gross_sizing_book_caps(
        df, target_gross_usd=10_000.0, delta_floor=0.1,
        strategy=s, shares_out_map={}, cap_mode="structural_plus_day_liquidity",
    )
    np.testing.assert_allclose(
        out_default["gross_target_usd"].to_numpy(), out_explicit["gross_target_usd"].to_numpy(), rtol=1e-12,
    )


def test_binding_label_marks_shares_available_in_executable_pass():
    """When the binding cap on a row is short_avail, ``binding_per_row`` should say so."""
    df = _liq_squeezed_df()
    s = _strategy_with_per_sleeve()
    _out, diag = apply_gross_sizing_book_caps(
        df, target_gross_usd=10_000.0, delta_floor=0.1,
        strategy=s, shares_out_map={},
    )
    binding = diag.get("binding_per_row")
    assert binding is not None
    assert "shares_available" in set(map(str, binding.tolist()))


def test_binding_label_optimal_pass_drops_shares_available():
    """Structural-only pass shouldn't list ``shares_available`` as a binder."""
    df = _liq_squeezed_df()
    s = _strategy_with_per_sleeve()
    _out, diag = apply_gross_sizing_book_caps(
        df, target_gross_usd=10_000.0, delta_floor=0.1,
        strategy=s, shares_out_map={}, cap_mode="structural_only",
    )
    binding = diag.get("binding_per_row")
    assert binding is not None
    assert "shares_available" not in set(map(str, binding.tolist()))


def test_liquidity_tight_returns_binding_label_when_requested():
    df = _liq_squeezed_df()
    caps = {
        "missing_shares_cap": 0.5,
        "aum_use_pct": 0.0,
        "short_avail_use_pct": 0.5,
        "shares_outstanding_use_frac": 0.0,
        "median_daily_volume_use_pct": 0.0,
    }
    out, lab = _liquidity_tight_book_weights(
        df, target_gross_usd=1_000.0, delta_floor=0.1, caps=caps,
        shares_out_map={}, return_binding_label=True,
    )
    assert out.shape == (2,)
    assert lab.shape == (2,)
    assert all(str(x) == "shares_available" for x in lab.tolist())


# ---------- harvest helper ----------


def test_harvest_resolve_target_basis_columns_prefers_optimal_when_present():
    plan = pd.DataFrame({
        "ETF": ["X", "Y"],
        "Underlying": ["UX", "UY"],
        "long_usd": [100.0, 200.0],
        "short_usd": [-100.0, -200.0],
        "optimal_long_usd": [300.0, 400.0],
        "optimal_short_usd": [-300.0, -400.0],
    })
    long_col, short_col = _resolve_target_basis_columns(plan, "optimal")
    assert long_col == "optimal_long_usd" and short_col == "optimal_short_usd"


def test_harvest_resolve_target_basis_falls_back_to_executable_when_no_optimal():
    plan = pd.DataFrame({
        "ETF": ["X"], "Underlying": ["UX"],
        "long_usd": [100.0], "short_usd": [-100.0],
    })
    long_col, short_col = _resolve_target_basis_columns(plan, "optimal")
    assert long_col == "long_usd" and short_col == "short_usd"


def test_harvest_executable_basis_explicit_uses_legacy_columns():
    plan = pd.DataFrame({
        "ETF": ["X"], "Underlying": ["UX"],
        "long_usd": [100.0], "short_usd": [-100.0],
        "optimal_long_usd": [200.0], "optimal_short_usd": [-200.0],
    })
    long_col, short_col = _resolve_target_basis_columns(plan, "executable")
    assert long_col == "long_usd" and short_col == "short_usd"


# ---------- phase 2b helper ----------


def test_resize_target_columns_hybrid_default_uses_optimal_for_band_executable_for_clip():
    plan = pd.DataFrame({
        "ETF": ["X"], "Underlying": ["UX"],
        "long_usd": [100.0], "short_usd": [-100.0],
        "optimal_long_usd": [200.0], "optimal_short_usd": [-200.0],
    })
    bl, bs, el, es = _resize_target_columns(plan, "hybrid")
    assert (bl, bs) == ("optimal_long_usd", "optimal_short_usd")
    assert (el, es) == ("long_usd", "short_usd")


def test_resize_target_columns_executable_legacy_all_executable():
    plan = pd.DataFrame({
        "ETF": ["X"], "Underlying": ["UX"],
        "long_usd": [100.0], "short_usd": [-100.0],
        "optimal_long_usd": [200.0], "optimal_short_usd": [-200.0],
    })
    bl, bs, el, es = _resize_target_columns(plan, "executable")
    assert (bl, bs, el, es) == ("long_usd", "short_usd", "long_usd", "short_usd")


def test_resize_hybrid_clips_grow_to_executable():
    """In hybrid mode, a grow toward optimal should be clipped to today's executable target.

    Setup: structural target = $1000 long, executable target = $400 long, current = $0.
    Without clipping, the grow trade USD would be ~$1000 (or band-trimmed). With hybrid clip,
    it should be at most $400 — today's executable allows no more.
    """
    plan = pd.DataFrame({
        "ETF": ["A"], "Underlying": ["U1"],
        "long_usd": [400.0], "short_usd": [-100.0],
        "optimal_long_usd": [1000.0], "optimal_short_usd": [-100.0],
    })
    cfg = ResizeBandConfig.from_dict({
        "enabled": True,
        "enter_band_pct": 0.05,
        "exit_band_pct": 0.01,
        "min_trim_usd": 1.0,
        "min_grow_usd": 1.0,
    })
    trades, _ = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"U1": 0, "A": 0},
        prices={"U1": 1.0, "A": 1.0},
        purgatory_etfs=set(),
        flow_etfs=set(),
        blacklist=set(),
        cfg=cfg,
        target_basis="hybrid",
    )
    long_trade = next((t for t in trades if t["leg_side"] == "long_under"), None)
    assert long_trade is not None, f"expected a long_under grow trade; got {trades}"
    # Action must be BUY (grow), trade_usd_target must be capped at executable ($400).
    assert long_trade["action"] == "BUY"
    assert long_trade["trade_usd_target"] <= 400.0 + 1e-6
    assert "clipped_to_executable" in str(long_trade["reason"])


def test_resize_executable_basis_uses_executable_target_for_band():
    """In executable mode, a grow stops at the executable target — mirrors legacy behavior."""
    plan = pd.DataFrame({
        "ETF": ["A"], "Underlying": ["U1"],
        "long_usd": [400.0], "short_usd": [-100.0],
        "optimal_long_usd": [1000.0], "optimal_short_usd": [-100.0],
    })
    cfg = ResizeBandConfig.from_dict({
        "enabled": True,
        "enter_band_pct": 0.05,
        "exit_band_pct": 0.01,
        "min_trim_usd": 1.0,
        "min_grow_usd": 1.0,
    })
    trades, _ = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={"U1": 0, "A": 0},
        prices={"U1": 1.0, "A": 1.0},
        purgatory_etfs=set(),
        flow_etfs=set(),
        blacklist=set(),
        cfg=cfg,
        target_basis="executable",
    )
    long_trade = next((t for t in trades if t["leg_side"] == "long_under"), None)
    assert long_trade is not None
    assert long_trade["action"] == "BUY"
    assert long_trade["trade_usd_target"] <= 400.0 + 1e-6
    assert "clipped_to_executable" not in str(long_trade["reason"])
