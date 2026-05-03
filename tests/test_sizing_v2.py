# -*- coding: utf-8 -*-
"""Unit tests for ``scripts/sizing_v2.py``.

Covers:

* ``net_edge_signal_table`` reads the bootstrap percentile columns and
  computes the sigma proxies + bootstrap availability flag.
* ``kelly_pre_weights_from_net_edge`` zeros names with ``mu_hat <= 0`` and
  applies the Sinclair-style uncertainty haircut.
* ``_project_simplex_nonneg`` and ``_apply_caps_with_redistribution`` behave.
* ``kelly_qp_overlay`` returns weights that sum to the input total, respect
  non-negativity, and respect the per-name cap.
* ``dd_brake_multiplier`` returns 1 above threshold and the floor for deep DDs.
* ``pair_weights_with_net_edge_kelly`` produces normalized PAIR_WEIGHTS and
  PAIR_FRAC_BY_KEY shapes consistent with the existing notebook contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.sizing_v2 import (
    _apply_caps_with_redistribution,
    _project_simplex_nonneg,
    _shrink_cov_constant_corr,
    dd_brake_multiplier,
    kelly_pre_weights_from_net_edge,
    kelly_qp_overlay,
    net_edge_signal_table,
    pair_weights_with_net_edge_kelly,
)


def _toy_mirror_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ETF": ["AMDL", "AMUU", "BITX", "AAPU", "DUMD"],
            "Underlying": ["AMD", "AMD", "IBIT", "AAPL", "AMD"],
            "Beta": [2.0, 2.0, 2.0, 2.0, -2.0],
            "blended_gross_decay": [0.50, 0.55, 0.30, 0.10, 0.20],
            "borrow_current": [0.04, 0.05, 0.03, 0.06, 0.08],
            "net_edge_p05_annual": [0.18, 0.22, 0.10, -0.05, -0.30],
            "net_edge_p25_annual": [0.28, 0.30, 0.14, 0.00, -0.15],
            "net_edge_p50_annual": [0.36, 0.37, 0.20, 0.01, -0.05],
            "net_edge_p75_annual": [0.44, 0.45, 0.26, 0.04, 0.05],
            "net_edge_p95_annual": [0.55, 0.55, 0.34, 0.07, 0.20],
        }
    )


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


def test_project_simplex_nonneg_sums_to_total_and_is_nonneg():
    v = np.array([0.4, -0.1, 0.6, 0.2])
    p = _project_simplex_nonneg(v, total=1.0)
    assert p.shape == v.shape
    assert (p >= -1e-12).all()
    assert abs(float(p.sum()) - 1.0) < 1e-9


def test_project_simplex_nonneg_zero_total_returns_zeros():
    v = np.array([1.0, 2.0, 3.0])
    out = _project_simplex_nonneg(v, total=0.0)
    assert (out == 0).all()


def test_apply_caps_with_redistribution_preserves_total_and_caps():
    d = np.array([0.5, 0.2, 0.2, 0.1])
    c = np.array([0.3, 0.3, 0.3, 0.3])
    out = _apply_caps_with_redistribution(d, c)
    assert abs(float(out.sum()) - float(d.sum())) < 1e-9
    assert (out <= c + 1e-9).all()


def test_apply_caps_with_redistribution_fully_constrained_caps_clip_total():
    d = np.array([0.5, 0.5])
    c = np.array([0.2, 0.2])
    out = _apply_caps_with_redistribution(d, c)
    assert (out <= c + 1e-9).all()
    assert float(out.sum()) <= 0.4 + 1e-9


def test_shrink_cov_constant_corr_preserves_diag():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    cov = pd.DataFrame(np.cov(X.T), index=list("abcd"), columns=list("abcd"))
    sh = _shrink_cov_constant_corr(cov, alpha=0.5)
    assert sh.shape == cov.shape
    np.testing.assert_allclose(np.diag(sh.values), np.diag(cov.values), atol=1e-12)


# ---------------------------------------------------------------------------
# Net-edge signal extraction
# ---------------------------------------------------------------------------


def test_net_edge_signal_table_reads_p50_and_sigma_proxy():
    df = _toy_mirror_df()
    sig = net_edge_signal_table(df, sigma_floor=0.04)
    assert {"mu_hat", "sigma_hat", "has_bootstrap"}.issubset(sig.columns)
    np.testing.assert_allclose(sig["mu_hat"].values, df["net_edge_p50_annual"].values, atol=1e-9)
    expected_sigma = (df["net_edge_p95_annual"] - df["net_edge_p05_annual"]) / (2 * 1.6448536269514722)
    np.testing.assert_allclose(
        sig["sigma_hat"].values,
        expected_sigma.clip(lower=0.04).values,
        atol=1e-9,
    )
    assert bool(sig["has_bootstrap"].all())


def test_net_edge_signal_table_falls_back_to_blended_when_bootstrap_missing():
    df = _toy_mirror_df()
    df.loc[3, ["net_edge_p05_annual", "net_edge_p50_annual", "net_edge_p95_annual"]] = np.nan
    sig = net_edge_signal_table(df, sigma_floor=0.04)
    assert not bool(sig.loc[3, "has_bootstrap"])
    expected_fb = float(df.loc[3, "blended_gross_decay"] - df.loc[3, "borrow_current"])
    assert abs(float(sig.loc[3, "mu_hat"]) - expected_fb) < 1e-9


# ---------------------------------------------------------------------------
# Kelly pre-weights
# ---------------------------------------------------------------------------


def test_kelly_pre_weights_zero_for_negative_p50():
    df = _toy_mirror_df()
    pw = kelly_pre_weights_from_net_edge(df, kelly_fraction=0.35, edge_floor=0.0)
    assert float(pw.loc[pw["ETF"] == "DUMD", "w_pre"].iloc[0]) == 0.0


def test_kelly_pre_weights_haircut_reduces_weight_when_t_small():
    df = _toy_mirror_df()
    df.loc[3, "net_edge_p50_annual"] = 0.05
    df.loc[3, "net_edge_p05_annual"] = -0.30
    df.loc[3, "net_edge_p95_annual"] = 0.40
    pw_no = kelly_pre_weights_from_net_edge(df, kelly_fraction=0.35, uncertainty_haircut=False)
    pw_hc = kelly_pre_weights_from_net_edge(df, kelly_fraction=0.35, uncertainty_haircut=True, haircut_floor=0.0)
    a = float(pw_no.loc[3, "w_pre"])
    b = float(pw_hc.loc[3, "w_pre"])
    assert b < a, f"haircut should shrink low-conviction names, got {b} >= {a}"


def test_kelly_pre_weights_high_t_unhairred():
    df = _toy_mirror_df()
    df.loc[0, "net_edge_p50_annual"] = 1.0
    df.loc[0, "net_edge_p05_annual"] = 0.95
    df.loc[0, "net_edge_p95_annual"] = 1.05
    pw_no = kelly_pre_weights_from_net_edge(df, kelly_fraction=0.35, uncertainty_haircut=False)
    pw_hc = kelly_pre_weights_from_net_edge(df, kelly_fraction=0.35, uncertainty_haircut=True)
    a = float(pw_no.loc[0, "w_pre"])
    b = float(pw_hc.loc[0, "w_pre"])
    assert abs(b - a) / max(a, 1e-9) < 0.05


# ---------------------------------------------------------------------------
# Kelly QP overlay
# ---------------------------------------------------------------------------


def test_kelly_qp_overlay_respects_simplex_and_cap():
    rng = np.random.default_rng(42)
    syms = list("abcde")
    R = pd.DataFrame(rng.normal(size=(300, 5)), columns=syms)
    pre = np.array([0.20, 0.05, 0.15, 0.30, 0.30])
    cap = 0.40
    w_post, _, _ = kelly_qp_overlay(
        pre, syms, R, kelly_fraction=0.5, shrink_alpha=0.35, max_w=cap
    )
    assert abs(float(w_post.sum()) - float(pre.sum())) < 1e-6
    assert (w_post >= -1e-9).all()
    assert (w_post <= cap * float(pre.sum()) + 1e-9).all()


def test_kelly_qp_overlay_zero_pre_returns_pre():
    rng = np.random.default_rng(0)
    syms = list("abc")
    R = pd.DataFrame(rng.normal(size=(60, 3)), columns=syms)
    pre = np.zeros(3)
    w_post, _, _ = kelly_qp_overlay(pre, syms, R, kelly_fraction=0.35)
    assert (w_post == 0).all()


def test_kelly_qp_overlay_reduces_concentration_in_correlated_block():
    rng = np.random.default_rng(7)
    n = 600
    base = rng.normal(size=(n, 1))
    block_a = base + 0.15 * rng.normal(size=(n, 3))
    block_b = rng.normal(size=(n, 2))
    R = pd.DataFrame(np.hstack([block_a, block_b]), columns=list("a1 a2 a3 b1 b2".split()))
    pre = np.array([0.25, 0.25, 0.25, 0.125, 0.125])
    w_post, _, _ = kelly_qp_overlay(
        pre, list(R.columns), R, kelly_fraction=0.5, shrink_alpha=0.0, max_w=None
    )
    # Block-a weights should sink, block-b weights should swell.
    assert w_post[:3].sum() < pre[:3].sum() - 1e-3
    assert w_post[3:].sum() > pre[3:].sum() + 1e-3


# ---------------------------------------------------------------------------
# Drawdown brake
# ---------------------------------------------------------------------------


def test_dd_brake_returns_one_when_above_threshold():
    nav = pd.Series([100, 100.5, 101.0, 101.4])
    assert dd_brake_multiplier(nav, dd_threshold=-0.05, gamma=5.0) == pytest.approx(1.0)


def test_dd_brake_floor_for_deep_drawdown():
    nav = pd.Series([100.0] * 5 + [50.0])
    m = dd_brake_multiplier(nav, dd_threshold=-0.05, gamma=10.0, lev_floor_ratio=0.30)
    assert m == pytest.approx(0.30)


def test_dd_brake_linear_in_excess():
    nav = pd.Series([100, 100, 100, 90])  # 10% drawdown
    m = dd_brake_multiplier(nav, dd_threshold=-0.05, gamma=5.0, lev_floor_ratio=0.0)
    # excess = -0.05 - (-0.10) = 0.05 ; m = 1 - 5*0.05 = 0.75
    assert m == pytest.approx(0.75, abs=1e-9)


# ---------------------------------------------------------------------------
# End-to-end builder
# ---------------------------------------------------------------------------


def _toy_underlying_prices(symbols: list[str], n: int = 250, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    ret = rng.normal(scale=0.02, size=(n, len(symbols)))
    px = pd.DataFrame(np.cumprod(1 + ret, axis=0) * 100.0, index=dates, columns=symbols)
    return px


def test_pair_weights_with_net_edge_kelly_normalizes_and_caps():
    df = _toy_mirror_df()
    universe = [
        ("AMDL", "AMD", 2.0),
        ("AMUU", "AMD", 2.0),
        ("BITX", "IBIT", 2.0),
        ("AAPU", "AAPL", 2.0),
        ("DUMD", "AMD", -2.0),
    ]
    syms = sorted({u for _, u, _ in universe})
    px = _toy_underlying_prices(syms, n=300)
    pair_w, pair_frac, diag = pair_weights_with_net_edge_kelly(
        universe,
        df,
        px,
        kelly_fraction=0.35,
        cov_shrink_alpha=0.35,
        max_pair_weight=0.40,
        max_underlying_weight=0.50,
    )
    assert abs(sum(pair_w.values()) - 1.0) < 1e-6
    assert abs(sum(pair_frac.values()) - 1.0) < 1e-6
    # DUMD has p50 < 0 so it must be zero
    assert pair_w.get("DUMD", 0.0) == 0.0
    # Per-pair cap respected (within a small tolerance because the projected
    # gradient scheme is iterative).
    assert max(pair_frac.values()) <= 0.40 + 5e-3


def test_pair_weights_with_net_edge_kelly_keeps_screened_out_at_zero_under_cap():
    """If a name has p50 <= 0 it must stay at zero even when caps redistribute."""
    df = _toy_mirror_df()
    df.loc[df["ETF"] == "AMDL", "net_edge_p50_annual"] = 1.0
    df.loc[df["ETF"] == "AMUU", "net_edge_p50_annual"] = 0.95
    universe = [
        ("AMDL", "AMD", 2.0),
        ("AMUU", "AMD", 2.0),
        ("BITX", "IBIT", 2.0),
        ("AAPU", "AAPL", 2.0),
        ("DUMD", "AMD", -2.0),
    ]
    syms = sorted({u for _, u, _ in universe})
    px = _toy_underlying_prices(syms, n=300)
    pair_w, _pair_frac, _diag = pair_weights_with_net_edge_kelly(
        universe,
        df,
        px,
        kelly_fraction=0.35,
        cov_shrink_alpha=0.35,
        max_pair_weight=0.30,
        max_underlying_weight=0.50,
    )
    # DUMD must stay at zero regardless of cap-driven redistribution.
    assert pair_w.get("DUMD", 0.0) == 0.0


def test_pair_weights_with_net_edge_kelly_falls_back_to_uniform_when_no_signal():
    df = _toy_mirror_df()
    df["net_edge_p50_annual"] = -1.0  # everyone negative
    df["blended_gross_decay"] = 0.0
    df["borrow_current"] = 0.0
    universe = [(e, u, 2.0) for e, u in zip(df["ETF"], df["Underlying"])]
    syms = sorted({u for _, u, _ in universe})
    px = _toy_underlying_prices(syms, n=120)
    pair_w, pair_frac, _diag = pair_weights_with_net_edge_kelly(
        universe, df, px, kelly_fraction=0.35
    )
    assert abs(sum(pair_w.values()) - 1.0) < 1e-6
    assert abs(sum(pair_frac.values()) - 1.0) < 1e-6
    # All weights equal under uniform fallback.
    vals = sorted(pair_w.values())
    assert abs(vals[0] - vals[-1]) < 1e-9
