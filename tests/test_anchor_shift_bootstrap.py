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
    3. ``passive_low_beta`` rows skip the shift even when expected p50 is
       present, because ``_expected_decay_available("passive_low_beta")``
       is False (Itô identity at β≈1 is unreliable).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

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


def _build_letf_tr(und_tr: pd.Series, beta: float, daily_drag: float = 0.0005) -> pd.Series:
    """Generate an LETF TR series implied by a constant daily drag.

    daily_drag = β·r_und − r_etf  ⇒  r_etf = β·r_und − daily_drag.
    """
    r_und = np.log(und_tr / und_tr.shift(1)).fillna(0.0)
    r_etf = beta * r_und - daily_drag
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
        "Beta": beta,
        "Beta_n_obs": float(n_obs),
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


def test_anchor_shift_centers_bootstrap_on_expected_p50():
    sigma = 0.30
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=42)
    etf = _build_letf_tr(und, beta=2.0, daily_drag=0.0004)  # ~10% annual realized drag
    tr_map = {"ABC": etf, "ABCUND": und}

    # Anchor at 30% expected (way above the 10% realized) — bootstrap mean
    # should land near 30% after shift.
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
    # Net-edge p50 ≈ shifted gross p50 (borrow=0) ≈ anchor (block-bootstrap mean
    # ≈ anchor, p50 close to mean for symmetric draws).
    assert abs(float(row["net_edge_p50_annual"]) - anchor) < 0.05
    # Anchor diagnostics surface the shift amount.
    shift_recorded = float(row["gross_anchor_shift_annual"])
    target_recorded = float(row["gross_anchor_target_annual"])
    assert math.isfinite(shift_recorded)
    assert abs(target_recorded - anchor) < 1e-9
    assert abs(shift_recorded) > 0.05  # shift was meaningful (~+0.20)
    assert row["gross_anchor_source"] == "harq_log_anchored"


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


def test_passive_low_beta_skips_anchor_shift():
    """expected_decay_available is False for passive_low_beta → no shift."""
    sigma = 0.20
    n = 600
    und = _gbm_tr(n, sigma_annual=sigma, seed=99)
    # β=1.0 ⇒ classified as passive_low_beta
    etf = _build_letf_tr(und, beta=1.0, daily_drag=0.0002)
    tr_map = {"DEF": etf, "DEFUND": und}

    # Even if some upstream stage filled an expected p50 by accident, the
    # shift should be skipped for passive_low_beta.
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
    assert row["product_class"] == "passive_low_beta"
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
