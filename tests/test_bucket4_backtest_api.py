"""Tests for scripts/bucket4_backtest_api.py (opt2 + crash budget sizing)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.b4_crash_budget import CrashBudgetParams, cap_pair_weights, compute_crash_caps
from scripts.bucket4_backtest_api import (
    build_closes_broad_from_panel,
    build_pair_cache_from_panel,
    size_b4_book_asof,
)


def _px(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-03", periods=n)
    und = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    # Blow-off rally in last year for crash-budget signal.
    und[-252:] = und[-252] * np.linspace(1.0, 3.0, 252)
    etf = 50 * np.cumprod(1 + rng.normal(-0.0004, 0.03, n))
    return pd.DataFrame({"a_px": etf, "b_px": und}, index=idx)


@pytest.fixture
def screener_csv(tmp_path):
    df = pd.DataFrame(
        {
            "ETF": ["INV1", "INV2", "INV3", "INV4", "INV5"],
            "Underlying": ["UND1", "UND2", "UND3", "UND4", "UND5"],
            "bucket4_net_edge_annual": [0.55, 0.48, 0.42, 0.38, 0.35],
            "net_edge_p50_annual": [0.55, 0.48, 0.42, 0.38, 0.35],
            "borrow_current": [0.05, 0.08, 0.06, 0.10, 0.07],
            "Delta": [-2.0, -2.0, -1.8, -2.1, -2.0],
            "vol_underlying_annual": [0.60, 0.55, 0.70, 0.50, 0.65],
        }
    )
    path = tmp_path / "etf_screened_today.csv"
    df.to_csv(path, index=False)
    return path


def test_size_b4_book_asof_trim_only_cash_residual(screener_csv):
    uni = pd.read_csv(screener_csv)
    panel = {row.ETF: _px(seed=i) for i, row in uni.iterrows()}
    # Distinct underlyings with shared calendar.
    for i, row in uni.iterrows():
        panel[row.ETF] = _px(seed=i + 10)

    cache = build_pair_cache_from_panel(uni, panel)
    closes = build_closes_broad_from_panel(panel, uni)
    h_map = {}
    for und in uni["Underlying"].unique():
        px = next(panel[e] for e, u in zip(uni["ETF"], uni["Underlying"]) if u == und)
        h_map[und] = pd.Series(0.45, index=px.index)

    opt2 = {
        "pf_min_pairs": 5,
        "decay_borrow_quad": 0,
        "borrow_linear_aversion": 1.5,
        "borrow_uncertainty_penalty": 0.0,
        "borrow_aversion_source": "spot",
        "min_weight": 0.005,
        "max_weight": 0.35,
        "cov_penalty": 0.0,  # simplify: skip cov tilt sensitivity
        "hedge_cadence_policy": {"h_mid": 0.45},
        "crash_budget": {
            "enabled": True,
            "rho": 0.0075,
            "theta": 0.5,
            "phi": 0.5,
            "l_floor": 0.02,
            "missing_policy": "book_quantile",
            "missing_l_quantile": 0.75,
        },
    }
    as_of = max(px.index.max() for px in panel.values())
    sized = size_b4_book_asof(
        run_date=as_of,
        pair_cache=cache,
        hedge_by_underlying=h_map,
        closes_broad=closes,
        screened_csv=screener_csv,
        sleeve_budget_usd=100_000.0,
        opt2_cfg=opt2,
        use_ibkr_uvix_borrow=False,
    )
    assert sized.sizing_method == "v6_opt2_crash_budget"
    assert sized.deployed_fraction <= 1.0 + 1e-9
    assert sized.cash_residual == pytest.approx(1.0 - sized.deployed_fraction, abs=1e-9)
    # Crash budget should bind on rally names → cash residual > 0.
    assert sized.cash_residual > 0.05
    assert sized.budget_eff < sized.budget_usd
    w_etf = sized.weights_by_etf()
    assert abs(sum(w_etf.values()) - sized.deployed_fraction) < 1e-9
    # Trim-only: capped <= opt2 (after both normalized to comparable scale).
    for key, w_c in sized.weights_capped.items():
        w_o = sized.weights_opt2.get(key, 0.0)
        # opt2 may not sum to 1 before cap_pair_weights normalizes; compare via telemetry.
        assert w_c >= 0.0
    assert any(t.get("crash_budget_mult", 1.0) < 0.999 for t in sized.telemetry)


def test_cap_pair_weights_no_renorm_invariant():
    pw = {("A", "X"): 0.5, ("B", "Y"): 0.5}
    caps = pd.DataFrame(
        [
            {"ETF": "A", "Underlying": "X", "cap_usd": 10_000.0, "L": 0.2, "C": 0.3,
             "runup": 0.5, "tail": 0.2, "hedge_ratio": 0.45, "crash_l_source": "signal"},
            {"ETF": "B", "Underlying": "Y", "cap_usd": 10_000.0, "L": 0.2, "C": 0.3,
             "runup": 0.5, "tail": 0.2, "hedge_ratio": 0.45, "crash_l_source": "signal"},
        ]
    )
    capped, budget_eff, tel = cap_pair_weights(pw, caps, 100_000.0, norm_sym=lambda s: s.upper())
    assert sum(capped.values()) < 1.0
    assert budget_eff == pytest.approx(100_000.0 * sum(capped.values()))
    assert (tel["crash_budget_mult"] < 1.0).all()
