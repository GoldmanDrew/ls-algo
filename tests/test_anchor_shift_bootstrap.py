"""Tests for the anchor-shift bootstrap in ``screener_v2_fields``.

The schema-v2 net-edge bootstrap should anchor its block-resampled realized
gross draws onto the model-based ``expected_gross_decay_p50_annual`` when
that column is present and the row's product class supports an expected
forecast. The shift moves the mean of the gross draws to match the
expected p50 without changing their dispersion / autocorrelation shape;
the resulting net-edge p50 then reflects (expected gross decay − expected
borrow) instead of (realized gross decay − realized borrow).

Tests we run:
    1. With expected p50 set, the gross-side mean of the bootstrap draws
       lands within MC noise of the anchor target.
    2. Without an expected p50 (NaN), no shift is applied; the bootstrap
       reproduces the legacy realized-only behavior.
    3. ``passive_low_delta`` rows skip the shift even when expected p50 is
       present, because ``_expected_decay_available("passive_low_delta")``
       is False (Itô identity at β≈1 is unreliable).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from screener_v2_fields import (
    TRADING_DAYS,
    enrich_screener_v2_fields,
)


def _gbm_tr(n_days: int, sigma_annual: float, mu_annual: float = 0.0, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    shocks = rng.normal(loc=drift, scale=sigma_annual * math.sqrt(dt), size=n_days)
    log_levels = np.cumsum(shocks)
    idx = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n_days)
    return pd.Series(np.exp(log_levels), index=idx)


def _build_letf_tr(
    und_tr: pd.Series,
    beta: float,
    daily_drag: float = 0.0005,
    noise_sigma_daily: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Generate an LETF TR series implied by a daily drag (optionally noisy).

    daily_drag = β·r_und − r_etf  ⇒  r_etf = β·r_und − daily_drag.
    With ``noise_sigma_daily > 0`` the drag fluctuates day-by-day; the
    block-bootstrap then produces a finite ``sigma_realized`` (non-zero
    standard error of the mean) instead of a degenerate point.
    """
    r_und = np.log(und_tr / und_tr.shift(1)).fillna(0.0)
    if noise_sigma_daily and noise_sigma_daily > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=float(noise_sigma_daily), size=len(r_und))
        drag_series = float(daily_drag) + noise
    else:
        drag_series = float(daily_drag)
    r_etf = beta * r_und - drag_series
    levels = np.exp(np.cumsum(r_etf))
    return pd.Series(levels.values, index=und_tr.index)


def _minimal_letf_row(
    *,
    etf: str,
    underlying: str,
    beta: float,
    expected_p50: float | None,
    is_yieldboost: bool = False,
    n_obs: int = 252,
) -> dict[str, Any]:
    return {
        "ETF": etf,
        "Underlying": underlying,
        "Delta": beta,
        "Delta_n_obs": float(n_obs),
        "Leverage": beta if not is_yieldboost else 2.0,
        "is_yieldboost": is_yieldboost,
        "borrow_current": 0.0,
        "gross_decay_annual": 0.10,
        "expected_gross_decay_annual": 0.12,
        "blended_gross_decay": 0.11,
        "expected_gross_decay_p50_annual": (
            np.nan if expected_p50 is None else float(expected_p50)
        ),
        "expected_gross_decay_dist_model": (
            "harq_log_anchored" if expected_p50 is not None else None
        ),
    }


def test_anchor_shift_fallback_when_only_p50_present():
    """Pre-blend behaviour: with only ``expected_p50`` (no band), the bootstrap
    falls back to the legacy anchor-shift (E2 fallback). ``gross_blend_method``
    surfaces ``anchor_shift_fallback`` to distinguish from inverse-variance."""
    sigma = 0.30
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=42)
    etf = _build_letf_tr(und, beta=2.0, daily_drag=0.0004)  # ~10% annual realized drag
    tr_map = {"ABC": etf, "ABCUND": und}

    anchor = 0.30
    df = pd.DataFrame(
        [
            _minimal_letf_row(
                etf="ABC",
                underlying="ABCUND",
                beta=2.0,
                expected_p50=anchor,
                n_obs=n - 1,
            )
        ]
    )
    out = enrich_screener_v2_fields(df, tr_map, bootstrap_seed=7)
    row = out.iloc[0]
    # No band -> deterministic anchor-shift -> net-edge p50 lands on anchor.
    assert abs(float(row["net_edge_p50_annual"]) - anchor) < 0.05
    shift_recorded = float(row["gross_anchor_shift_annual"])
    target_recorded = float(row["gross_anchor_target_annual"])
    assert math.isfinite(shift_recorded)
    assert abs(target_recorded - anchor) < 1e-9
    assert abs(shift_recorded) > 0.05
    assert row["gross_anchor_source"] == "harq_log_anchored"
    assert row["gross_blend_method"] == "anchor_shift_fallback"
    assert pd.isna(row["gross_blend_weight_forward"])


def test_inverse_variance_blend_with_band():
    """When a p10/p90 band is present, the blend weight depends on how tight
    the forward forecast is vs the realized bootstrap dispersion. Posterior
    sits between the realized mean and the forward p50, and matches the
    closed-form Normal-Normal conjugate identity.
    """
    sigma = 0.30
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=42)
    etf = _build_letf_tr(und, beta=2.0, daily_drag=0.0004, noise_sigma_daily=0.005, seed=1)
    tr_map = {"BCD": etf, "BCDUND": und}

    # Wide forward band (sigma_F large) → forward should get LESS weight; the
    # posterior should be pulled toward the realized ~10% drag.
    anchor_p50 = 0.30
    anchor_p10 = 0.05
    anchor_p90 = 0.55  # very wide band (sigma_F ≈ 0.195)
    row_d = _minimal_letf_row(
        etf="BCD",
        underlying="BCDUND",
        beta=2.0,
        expected_p50=anchor_p50,
        n_obs=n - 1,
    )
    row_d["expected_gross_decay_p10_annual"] = anchor_p10
    row_d["expected_gross_decay_p90_annual"] = anchor_p90
    df = pd.DataFrame([row_d])
    out = enrich_screener_v2_fields(df, tr_map, bootstrap_seed=7)
    row = out.iloc[0]
    assert row["gross_blend_method"] == "inverse_variance"
    w_F = float(row["gross_blend_weight_forward"])
    assert 0.0 < w_F < 1.0  # genuine blend (neither extreme)
    sigF = float(row["gross_sigma_forward_annual"])
    sigR = float(row["gross_sigma_realized_annual"])
    assert sigF > 0 and sigR > 0
    # Mathematical identity: w_F = sigR^2 / (sigF^2 + sigR^2).
    expected_w = sigR ** 2 / (sigF ** 2 + sigR ** 2)
    assert abs(w_F - expected_w) < 1e-9
    # Posterior must lie between realized mean and forward p50.
    mu_R = float(row["gross_realized_mean_annual"])
    posterior = float(row["gross_anchor_target_annual"])
    assert min(mu_R, anchor_p50) <= posterior <= max(mu_R, anchor_p50)


def test_inverse_variance_blend_tight_forward_dominates():
    """Tight forward band (sigma_F small) relative to noisy realized history
    → forward dominates → posterior sits close to forward p50, recovering
    legacy anchor-shift behaviour.
    """
    sigma = 0.30
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=42)
    # Big realized noise so sigma_R is comfortably larger than sigma_F.
    etf = _build_letf_tr(und, beta=2.0, daily_drag=0.0004, noise_sigma_daily=0.03, seed=2)
    tr_map = {"CDE": etf, "CDEUND": und}

    anchor_p50 = 0.30
    # Tight band — sigma_F ≈ 0.012.
    anchor_p10 = 0.285
    anchor_p90 = 0.315
    row_d = _minimal_letf_row(
        etf="CDE",
        underlying="CDEUND",
        beta=2.0,
        expected_p50=anchor_p50,
        n_obs=n - 1,
    )
    row_d["expected_gross_decay_p10_annual"] = anchor_p10
    row_d["expected_gross_decay_p90_annual"] = anchor_p90
    df = pd.DataFrame([row_d])
    out = enrich_screener_v2_fields(df, tr_map, bootstrap_seed=7)
    row = out.iloc[0]
    assert row["gross_blend_method"] == "inverse_variance"
    w_F = float(row["gross_blend_weight_forward"])
    assert w_F > 0.9
    posterior = float(row["gross_anchor_target_annual"])
    assert abs(posterior - anchor_p50) < 0.05


def test_inverse_variance_blend_matches_closed_form():
    """Direct unit test of the blend helper to lock the math identity."""
    from screener_v2_fields import _inverse_variance_blend, _band_to_sigma

    sigma_F = _band_to_sigma(p10=0.20, p90=0.40)
    assert sigma_F is not None
    expected_sigma_F = (0.40 - 0.20) / (2.0 * 1.2815515655446004)
    assert abs(sigma_F - expected_sigma_F) < 1e-9
    result = _inverse_variance_blend(
        mu_forward=0.30, sigma_forward=sigma_F,
        mu_realized=0.10, sigma_realized=0.10,
    )
    assert result is not None
    posterior, w_F = result
    # Closed form: w_F = sigR² / (sigF² + sigR²).
    vF = sigma_F ** 2
    vR = 0.10 ** 2
    assert abs(w_F - vR / (vF + vR)) < 1e-12
    assert abs(posterior - (w_F * 0.30 + (1 - w_F) * 0.10)) < 1e-12

    # Limit: sigma_F → 0 ⇒ pure forward.
    res2 = _inverse_variance_blend(0.30, 1e-9, 0.10, 0.10)
    assert res2 is not None
    assert abs(res2[0] - 0.30) < 1e-6
    assert res2[1] > 0.999

    # Limit: sigma_R → 0 ⇒ pure realized.
    res3 = _inverse_variance_blend(0.30, 0.10, 0.10, 1e-9)
    assert res3 is not None
    assert abs(res3[0] - 0.10) < 1e-6
    assert res3[1] < 1e-3


def test_no_anchor_shift_when_expected_p50_missing():
    sigma = 0.30
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=23)
    etf = _build_letf_tr(und, beta=2.0, daily_drag=0.0004)
    tr_map = {"ABC": etf, "ABCUND": und}

    df = pd.DataFrame(
        [
            _minimal_letf_row(
                etf="ABC",
                underlying="ABCUND",
                beta=2.0,
                expected_p50=None,
                n_obs=n - 1,
            )
        ]
    )
    out = enrich_screener_v2_fields(df, tr_map, bootstrap_seed=7)
    row = out.iloc[0]
    # No anchor available → bootstrap mean should track the realized drag
    # (~10% annualized), well below the contrived 30%.
    assert float(row["net_edge_p50_annual"]) < 0.20
    # No anchor recorded.
    shift_recorded = row["gross_anchor_shift_annual"]
    assert pd.isna(shift_recorded) or shift_recorded == 0.0
    assert row["gross_anchor_source"] == ""


def test_passive_low_delta_skips_anchor_shift():
    """expected_decay_available is False for passive_low_delta → no shift."""
    sigma = 0.20
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=99)
    # β=1.0 ⇒ classified as passive_low_delta
    etf = _build_letf_tr(und, beta=1.0, daily_drag=0.0002)
    tr_map = {"DEF": etf, "DEFUND": und}

    # Even if some upstream stage filled an expected p50 by accident, the
    # shift should be skipped for passive_low_delta.
    df = pd.DataFrame(
        [
            _minimal_letf_row(
                etf="DEF",
                underlying="DEFUND",
                beta=1.0,
                expected_p50=0.50,    # large bogus anchor
                n_obs=n - 1,
            )
        ]
    )
    out = enrich_screener_v2_fields(df, tr_map, bootstrap_seed=11)
    row = out.iloc[0]
    assert row["product_class"] == "passive_low_delta"
    # Bootstrap mean stays anchored on realized (~5% annualized at this drag),
    # not on the bogus 50%.
    assert float(row["net_edge_p50_annual"]) < 0.20
    shift_recorded = row["gross_anchor_shift_annual"]
    assert pd.isna(shift_recorded) or shift_recorded == 0.0


def test_yieldboost_routes_to_blended_realized_expected():
    """``is_yieldboost`` rows now use the blended-realized-expected route."""
    sigma = 0.40
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=51)
    etf = _build_letf_tr(und, beta=0.5, daily_drag=0.001)
    tr_map = {"YBX": etf, "YBXUND": und}

    df = pd.DataFrame(
        [
            _minimal_letf_row(
                etf="YBX",
                underlying="YBXUND",
                beta=0.5,
                expected_p50=0.40,
                is_yieldboost=True,
                n_obs=n - 1,
            )
        ]
    )
    out = enrich_screener_v2_fields(df, tr_map, bootstrap_seed=3)
    row = out.iloc[0]
    assert row["product_class"] == "income_yieldboost"
    assert row["gross_edge_definition"] == "blended_realized_expected"
    # YB rows should also receive the anchor shift (expected_decay_available
    # is True for income_yieldboost in the v2 taxonomy).
    assert math.isfinite(float(row["gross_anchor_shift_annual"]))
    assert abs(float(row["gross_anchor_target_annual"]) - 0.40) < 1e-9


def test_net_edge_borrow_subtracted_with_act360_factor():
    """Point-in-time borrow uses 365/360 effective annual charge (etf-dashboard)."""
    sigma = 0.30
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=42)
    etf = _build_letf_tr(und, beta=2.0, daily_drag=0.0004)
    tr_map = {"ABC": etf, "ABCUND": und}

    anchor = 0.30
    quoted_borrow = 0.036  # 3.6% annual quoted
    df = pd.DataFrame(
        [
            {
                **_minimal_letf_row(
                    etf="ABC",
                    underlying="ABCUND",
                    beta=2.0,
                    expected_p50=anchor,
                    n_obs=n - 1,
                ),
                "borrow_current": quoted_borrow,
            }
        ]
    )
    out = enrich_screener_v2_fields(df, tr_map, bootstrap_seed=7)
    row = out.iloc[0]
    expected_net = anchor - quoted_borrow * (365.0 / 360.0)
    assert abs(float(row["net_edge_p50_annual"]) - expected_net) < 0.05
    assert float(row["borrow_for_net_annual"]) == pytest.approx(
        quoted_borrow * (365.0 / 360.0), rel=1e-9
    )
