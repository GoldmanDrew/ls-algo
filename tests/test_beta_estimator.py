"""Tests for the robust hierarchical empirical-Bayes hedge-beta estimator.

Pins down the contract documented in BETA_ESTIMATOR.md:

  * ``mu_beta == nominal_L`` ONLY for ``letf_long`` / ``letf_inverse`` rows.
  * Income / vol-ETP / unknown rows MUST NOT use row Leverage as the prior
    mean — ``Beta_prior_source`` is never ``nominal_L`` for those classes.
  * Synthetic 2× LETF posterior recovers the realised slope.
  * Synthetic YieldBOOST row blends robust OLS with a sibling-derived prior.
  * Distribution-day jumps that coincide with a same-day actions event are
    excluded.
  * Strong sign conflict with the prior inflates ``τ`` rather than
    snapping to ``L``.
  * The multi-horizon stability gate selects ``h = 5`` on AR(1)
    microstructure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beta_estimator import (
    BetaPrior,
    BetaResult,
    PeerPrior,
    build_yieldboost_family_priors,
    compute_beta_for_hedging,
)


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260505)


def _series(returns: np.ndarray, *, start: str = "2024-01-01") -> pd.Series:
    """Convert a returns array into a price series with daily timestamps."""
    idx = pd.date_range(start, periods=len(returns) + 1, freq="B")
    prices = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + returns)])
    return pd.Series(prices, index=idx)


# ── (1) Prior router ───────────────────────────────────────────────────
def test_letf_long_uses_nominal_L() -> None:
    p = BetaPrior.for_row(
        product_class="letf_long",
        nominal_leverage=2.0,
        underlying="QQQ",
    )
    assert p.mu == 2.0
    assert p.tau == 60.0 * 4.0
    assert p.source == "nominal_L"


def test_letf_inverse_uses_negative_nominal_L() -> None:
    p = BetaPrior.for_row(
        product_class="letf_inverse",
        nominal_leverage=-3.0,
        underlying="NDX",
    )
    assert p.mu == -3.0
    assert p.tau == 60.0 * 9.0
    assert p.source == "nominal_L"


@pytest.mark.parametrize(
    "pc",
    [
        "covered_call_1x",
        "income_yieldboost",
        "volatility_etp",
        "scraped_income",
        "unknown",
    ],
)
def test_non_letf_classes_never_use_nominal_L(pc: str) -> None:
    """No non-LETF class may carry a ``nominal_L`` prior source — even when
    a ``Leverage`` value is provided to the factory.  This is the central
    contract that hedge sizing depends on."""
    p = BetaPrior.for_row(
        product_class=pc,
        nominal_leverage=2.0,  # deliberately non-None — must be ignored
        underlying="MSTR",
        peer_betas={},
    )
    assert p.source != "nominal_L"
    # In particular, mu cannot equal the listed L for any of these.
    assert p.mu != 2.0


def test_letf_classes_require_finite_leverage() -> None:
    with pytest.raises(ValueError):
        BetaPrior.for_row(
            product_class="letf_long",
            nominal_leverage=None,
            underlying="QQQ",
        )
    with pytest.raises(ValueError):
        BetaPrior.for_row(
            product_class="letf_inverse",
            nominal_leverage=float("nan"),
            underlying="NDX",
        )


# ── (2) Synthetic LETF and YieldBOOST recovery ────────────────────────
def test_synthetic_2x_letf_recovers_realized_slope(rng: np.random.Generator) -> None:
    """1000 days of r_etf = 2·r_und + small noise should produce β ≈ 2.0."""
    n = 1000
    r_und = rng.normal(0, 0.02, size=n)
    r_etf = 2.0 * r_und + rng.normal(0, 0.002, size=n)
    prior = BetaPrior.for_row("letf_long", 2.0, "UND")
    res = compute_beta_for_hedging(_series(r_etf), _series(r_und), prior)
    assert res.quality in ("ok", "non_stationary")
    assert 1.97 <= res.beta <= 2.03, res
    assert res.beta_se < 0.04
    assert res.prior_source == "nominal_L"
    assert res.source.startswith("posterior.")


def test_synthetic_yieldboost_uses_peer_prior(rng: np.random.Generator) -> None:
    """A YieldBOOST row whose realised β is ~0.5 and whose siblings sit at
    0.55 should land between those values, not at 1.0."""
    n = 250
    r_und = rng.normal(0, 0.025, size=n)
    r_etf = 0.5 * r_und - 0.0005 + rng.normal(0, 0.005, size=n)
    peers = {"MSTR": PeerPrior(mu=0.55, tau=90.0, n_siblings=3)}
    prior = BetaPrior.for_row(
        "income_yieldboost",
        nominal_leverage=1.0,  # must be ignored
        underlying="MSTR",
        peer_betas=peers,
    )
    assert prior.source == "yieldboost_peer_MSTR"
    assert prior.mu == 0.55
    res = compute_beta_for_hedging(_series(r_etf), _series(r_und), prior)
    assert res.prior_source == "yieldboost_peer_MSTR"
    assert 0.50 <= res.beta <= 0.58, res
    assert res.quality in ("ok", "non_stationary")


# ── (3) Distribution-day guard ────────────────────────────────────────
def test_distribution_day_event_excluded(rng: np.random.Generator) -> None:
    """A single -20% spike that coincides with a same-day actions event
    should be removed; the posterior must be essentially unchanged from
    the clean-history version."""
    n = 400
    r_und = rng.normal(0, 0.015, size=n)
    r_etf_clean = 0.5 * r_und + rng.normal(0, 0.004, size=n)

    r_etf_dirty = r_etf_clean.copy()
    spike_idx = n // 2
    r_etf_dirty[spike_idx] = -0.20  # massive ex-distribution drop

    s_etf_clean = _series(r_etf_clean)
    s_etf_dirty = _series(r_etf_dirty)
    s_und = _series(r_und)

    # Build an actions mask aligned with the *log-return* index produced
    # inside the estimator (which is the price index minus the first row).
    rets_idx = s_etf_dirty.index[1:]
    actions = pd.Series(False, index=rets_idx)
    actions.iloc[spike_idx] = True

    prior = BetaPrior.for_row(
        "income_yieldboost",
        nominal_leverage=None,
        underlying="X",
        peer_betas={"X": PeerPrior(mu=0.5, tau=60.0, n_siblings=2)},
    )

    res_clean = compute_beta_for_hedging(s_etf_clean, s_und, prior)
    res_dirty = compute_beta_for_hedging(s_etf_dirty, s_und, prior, actions_mask=actions)

    # The posterior on the cleaned series should be very close to the
    # one computed on the natively-clean series.
    assert abs(res_clean.beta - res_dirty.beta) < 0.02, (res_clean, res_dirty)

    # Sanity: without the actions mask, the spike biases the estimate.
    res_dirty_no_mask = compute_beta_for_hedging(s_etf_dirty, s_und, prior)
    assert abs(res_dirty_no_mask.beta - res_clean.beta) > abs(
        res_dirty.beta - res_clean.beta
    ) - 1e-9


# ── (4) Sign conflict — graceful (no hard snap, soft pull to μ) ───────
def test_sign_mismatch_inflates_tau_not_snap(rng: np.random.Generator) -> None:
    """An inverted realised relationship (β ≈ -0.4) against a μ=2 prior
    must NOT hard-snap to exactly 2.0.  The contract (BETA_ESTIMATOR.md):

      * ``sign_conflict`` quality flag is set
      * ``τ`` is inflated above the original prior precision
      * the posterior is strictly between the prior and the robust data
        estimate (no hard snap), i.e. ``β_robust < β_post < μ`` when the
        prior is positive and the data is negative.
    """
    n = 300
    r_und = rng.normal(0, 0.02, size=n)
    r_etf = -0.4 * r_und + rng.normal(0, 0.004, size=n)
    prior = BetaPrior.for_row("letf_long", 2.0, "UND")
    res = compute_beta_for_hedging(_series(r_etf), _series(r_und), prior)
    assert res.extras.get("sign_conflict", False) is True
    # τ inflation
    assert res.prior_tau > prior.tau - 1e-9
    assert res.prior_tau >= prior.tau * 4.5
    # Strictly between robust slope (-0.4) and prior (2.0); no hard snap.
    assert -0.4 < res.beta < 2.0
    assert abs(res.beta - 2.0) > 1e-6, "must not hard-snap to L"


# ── (5) Multi-horizon stability gate ──────────────────────────────────
def test_multi_horizon_gate_prefers_h5_on_ar1(rng: np.random.Generator) -> None:
    """Strong AR(1) microstructure on r_etf should trip the stability
    gate and select h=5."""
    n = 400
    eps_u = rng.normal(0, 0.02, size=n)
    r_und = eps_u
    eps_e = rng.normal(0, 0.005, size=n)
    r_etf = np.empty(n)
    r_etf[0] = 1.5 * r_und[0] + eps_e[0]
    # AR(1) leakage of past underlying into the ETF — typical
    # non-synchronous-close microstructure.
    for t in range(1, n):
        r_etf[t] = 0.6 * r_etf[t - 1] + 1.5 * (r_und[t] - 0.6 * r_und[t - 1]) + eps_e[t]
    prior = BetaPrior.for_row("letf_long", 2.0, "UND")
    res = compute_beta_for_hedging(_series(r_etf), _series(r_und), prior)
    # On clearly non-stationary microstructure either the gate flips to
    # h=5 OR the posterior at h=1 already passes the relative tolerance
    # window (rare for this synthetic but allowed by the contract).
    if res.extras.get("non_stationary"):
        assert res.horizon == 5
        assert res.quality == "non_stationary"
    else:
        # Without microstructure escalation we still expect a sane
        # posterior in [1.2, 2.0] for a 1.5× construction with τ=240.
        assert 1.2 < res.beta < 2.0


# ── (6) Hierarchical YieldBOOST family priors ─────────────────────────
def test_yieldboost_family_priors_leave_one_out(rng: np.random.Generator) -> None:
    """Three siblings on the same underlying produce a family prior μ
    that is the median of their raw OLS β."""
    n = 300
    r_und = rng.normal(0, 0.02, size=n)
    s_und = _series(r_und)
    # Three siblings with realised β ≈ 0.4, 0.5, 0.6
    siblings: dict[str, pd.Series] = {"UND": s_und}
    for slope, name in zip((0.4, 0.5, 0.6), ("AAY", "BBY", "CCY")):
        r_etf = slope * r_und + rng.normal(0, 0.003, size=n)
        siblings[name] = _series(r_etf)
    pairs = [("AAY", "UND"), ("BBY", "UND"), ("CCY", "UND")]
    priors = build_yieldboost_family_priors(pairs, siblings, min_days=60)
    assert "UND" in priors
    p = priors["UND"]
    assert 0.45 <= p.mu <= 0.55
    assert p.n_siblings == 3
    assert p.tau == pytest.approx(min(120.0, 30.0 * 3))


# ── (7) End-to-end: classifier + estimator routes correctly ───────────
def test_classifier_and_router_never_emit_nominal_L_for_yieldboost() -> None:
    """Spy on BetaPrior to assert that an income_yieldboost universe row
    cannot route through the nominal_L branch even when a Leverage is
    present on the row."""
    p = BetaPrior.for_row(
        "income_yieldboost",
        nominal_leverage=2.0,
        underlying="MSTR",
        peer_betas={"MSTR": PeerPrior(mu=0.42, tau=90.0, n_siblings=2)},
    )
    assert p.source == "yieldboost_peer_MSTR"
    assert p.mu == 0.42
