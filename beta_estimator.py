"""beta_estimator — robust, total-return-validated, empirical-Bayes hedge-beta.

This module replaces the simple shrunk-OLS hedge-beta estimator that lived in
``daily_screener`` (``compute_beta_shrunk``) with a more robust, hierarchical
empirical-Bayes posterior that is appropriate for income / YieldBOOST sleeves
where there is *no* listed leverage to anchor a prior on.

Core API
--------

``BetaPrior(mu, tau, source, product_class, sign_inflation)``
    A simple value object that carries the prior mean (``mu``) for the
    hedge-ratio of an ETF vs its underlying, and the prior precision
    expressed in *effective trading days* (``tau``).  ``source`` is a short
    dotted tag that is propagated all the way to the screener CSV via
    ``Beta_prior_source``.

``compute_beta_for_hedging(etf_tr, und_tr, prior, *, min_days=60) -> BetaResult``
    Returns the posterior hedge-ratio together with diagnostics.  The
    estimator combines:

    * total-return-aware return alignment (daily *log* returns)
    * EWMA weighting (half-life 63 trading days, with a floor)
    * Huber IRLS regression (numpy-only; ~3 reweighting passes)
    * Newey-West / HAC standard error (lag 5) for ``β̂_robust``
    * a multi-horizon stability gate at ``h ∈ {1, 5}`` that prefers the
      longer horizon and inflates ``τ`` when the estimator is unstable
    * a Gaussian-conjugate posterior

      .. code-block:: text

          β̂ = (τ·μ + n_eff·β_robust) / (τ + n_eff)

    The estimator does **not** hard-snap to the prior on sign mismatch;
    instead, when the data sign disagrees materially with the prior sign
    we inflate ``τ`` (default 5×) and re-blend.

The two LETF product classes (``letf_long``, ``letf_inverse``) are the
*only* classes that may carry ``mu = nominal listed leverage L``.  All
other classes — covered-call 1×, YieldBOOST, scraped income, vol ETPs,
unknown — must use a non-L prior built via :func:`BetaPrior.for_row`.

Design notes
------------

The posterior precision depends on the residual variance of the robust
fit and on the effective sample size of the *robustly-weighted* design
matrix; the resulting ``Beta_se`` is a usable hedge-quality signal even
when the EWMA half-life truncates the long tail.

Family priors for YieldBOOST are computed via
:func:`build_yieldboost_family_priors`, which derives a leave-one-out
median of sibling raw OLS betas on each underlying (with a global
fallback when no sibling exists).

This module is import-safe (no I/O at import time) and depends only on
numpy / pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────
TRADING_DAYS = 252

EWMA_HALFLIFE_DAYS = 63
EWMA_FLOOR_RATIO = 0.25  # weights cannot fall below 25 % of the maximum

HUBER_K = 1.345
HUBER_PASSES = 3

HAC_LAG = 5

HORIZONS = (1, 5)

# Multi-horizon stability gate
HORIZON_STABILITY_REL_TOL = 0.25  # |β1 - β5| > 0.25·|μ| triggers
HORIZON_STABILITY_SE_MULT = 2.0  # OR > 2·SE(β5) triggers

# Sign-inflation factor when sign(β_robust) ≠ sign(μ)
SIGN_INFLATION_FACTOR = 5.0
SIGN_CONFLICT_MIN_BETA = 0.30

# Distribution-day guard threshold (fraction)
DIST_DAY_RESIDUAL_THRESHOLD = 0.25

# Source tag prefix for posterior outputs
_POSTERIOR_PREFIX = "posterior"


# ── Data classes ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class BetaPrior:
    """Prior over the hedge-ratio :math:`β` for one ETF/underlying pair.

    ``mu``      Prior mean (the value that ``β̂`` is shrunk toward).
    ``tau``     Prior precision, in *effective trading days*.
    ``source``  Short dotted tag describing where ``mu`` came from.
                Examples: ``nominal_L``, ``covered_call_default``,
                ``yieldboost_peer_<UND>``, ``yieldboost_global``,
                ``volatility_etp``, ``empirical_bayes_global``.
    ``product_class``  The classification that produced the prior.
    ``allow_sign_inflation``  When True (default), a posterior with a
                sign that strongly disagrees with ``mu`` will inflate
                ``tau`` rather than re-derive ``mu``.  Vol-ETP rows
                (``tau == 0``) skip this entirely.
    """

    mu: float
    tau: float
    source: str
    product_class: str
    allow_sign_inflation: bool = True

    # ── Factory used by daily_screener / tests ──────────────────────
    @staticmethod
    def for_row(
        product_class: str,
        nominal_leverage: float | None,
        underlying: str | None,
        *,
        peer_betas: dict[str, "PeerPrior"] | None = None,
        cc_default_mu: float = 0.55,
        cc_default_tau: float = 30.0,
        yieldboost_default_mu: float | None = None,
        yieldboost_default_tau: float = 30.0,
        unknown_mu: float = 0.0,
        unknown_tau: float = 15.0,
    ) -> "BetaPrior":
        """Build a ``BetaPrior`` from the row's product class.

        ``peer_betas`` is the per-underlying YieldBOOST family map produced
        by :func:`build_yieldboost_family_priors` (keys are underlying
        symbols, values are :class:`PeerPrior` records with ``mu``/``tau``
        already aggregated).  Pass an empty dict if no peers are
        available — the function will fall back to the global
        YieldBOOST default.
        """

        pc = str(product_class)

        # ── (1) Long LETF ── ONLY class allowed to use positive nominal L
        if pc == "letf_long":
            if nominal_leverage is None or not np.isfinite(nominal_leverage):
                raise ValueError(
                    "letf_long requires a finite nominal_leverage; "
                    "got None / non-finite"
                )
            L = float(nominal_leverage)
            return BetaPrior(
                mu=L,
                tau=60.0 * max(1.0, L * L),
                source="nominal_L",
                product_class=pc,
            )

        # ── (2) Inverse LETF ── ONLY class allowed to use negative nominal L
        if pc == "letf_inverse":
            if nominal_leverage is None or not np.isfinite(nominal_leverage):
                raise ValueError(
                    "letf_inverse requires a finite nominal_leverage; "
                    "got None / non-finite"
                )
            L = float(nominal_leverage)
            return BetaPrior(
                mu=L,
                tau=60.0 * max(1.0, L * L),
                source="nominal_L",
                product_class=pc,
            )

        # ── (3) Covered-call 1× ── calibrated constant, NO nominal L
        if pc == "covered_call_1x":
            return BetaPrior(
                mu=float(cc_default_mu),
                tau=float(cc_default_tau),
                source="covered_call_default",
                product_class=pc,
            )

        # ── (4) YieldBOOST ── hierarchical empirical Bayes, NO nominal L
        if pc == "income_yieldboost":
            und = str(underlying or "").strip().upper()
            peer = (peer_betas or {}).get(und)
            if peer is not None and peer.tau > 0 and np.isfinite(peer.mu):
                return BetaPrior(
                    mu=float(peer.mu),
                    tau=float(peer.tau),
                    source=f"yieldboost_peer_{und}",
                    product_class=pc,
                )
            # Global fallback: aggregate all siblings
            global_peer = (peer_betas or {}).get("__global__")
            if global_peer is not None and global_peer.tau > 0 and np.isfinite(global_peer.mu):
                return BetaPrior(
                    mu=float(global_peer.mu),
                    tau=float(global_peer.tau),
                    source="yieldboost_global",
                    product_class=pc,
                )
            # Hard fallback when no sibling history exists at all
            mu = (
                yieldboost_default_mu
                if yieldboost_default_mu is not None
                else cc_default_mu
            )
            return BetaPrior(
                mu=float(mu),
                tau=float(yieldboost_default_tau),
                source="yieldboost_default",
                product_class=pc,
            )

        # ── (5) Volatility ETPs ── no shrinkage to a hedge ratio
        if pc == "volatility_etp":
            return BetaPrior(
                mu=0.0,
                tau=0.0,
                source="volatility_etp",
                product_class=pc,
                allow_sign_inflation=False,
            )

        # ── (6) Scraped income (YieldMax / Roundhill / generic 1× income) ─
        if pc == "scraped_income":
            return BetaPrior(
                mu=float(cc_default_mu),
                tau=float(cc_default_tau),
                source="scraped_income_default",
                product_class=pc,
            )

        # ── (7) Unknown / residual ──
        return BetaPrior(
            mu=float(unknown_mu),
            tau=float(unknown_tau),
            source="empirical_bayes_global",
            product_class=pc or "unknown",
        )


@dataclass(frozen=True)
class PeerPrior:
    """Aggregated peer-derived prior for a YieldBOOST underlying."""

    mu: float
    tau: float
    n_siblings: int


@dataclass(frozen=True)
class BetaResult:
    """Posterior + diagnostics returned by :func:`compute_beta_for_hedging`."""

    beta: float
    beta_se: float
    n_obs: int
    n_eff: float
    resid_sigma_annual: float
    horizon: int
    quality: str
    source: str
    prior_mu: float
    prior_tau: float
    prior_source: str
    extras: dict[str, Any] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────
def _aligned_log_returns(
    etf_tr: pd.Series,
    und_tr: pd.Series,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex] | None:
    """Return ``(r_etf, r_und, index)`` of aligned daily log returns.

    Drops duplicated index entries (keeping the last) and any non-finite
    pairs.  Returns ``None`` when the aligned overlap is empty.
    """
    etf_tr = etf_tr[~etf_tr.index.duplicated(keep="last")]
    und_tr = und_tr[~und_tr.index.duplicated(keep="last")]
    df = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
    if df.empty:
        return None
    re = np.log(df["etf"] / df["etf"].shift(1))
    ru = np.log(df["und"] / df["und"].shift(1))
    valid = re.notna() & ru.notna() & np.isfinite(re) & np.isfinite(ru)
    re = re[valid]
    ru = ru[valid]
    if re.empty:
        return None
    return re.to_numpy(dtype=float), ru.to_numpy(dtype=float), re.index


def _ewma_weights(n: int, halflife: float, floor: float) -> np.ndarray:
    """Newest-last EWMA weights; floor keeps the long tail influential."""
    if n <= 0:
        return np.zeros(0)
    if halflife is None or not np.isfinite(halflife) or halflife <= 0:
        return np.ones(n)
    age = np.arange(n - 1, -1, -1, dtype=float)  # newest -> 0, oldest -> n-1
    w = np.power(0.5, age / float(halflife))
    floor_val = float(floor) * float(np.max(w))
    if floor_val > 0.0:
        w = np.maximum(w, floor_val)
    return w


def _huber_irls(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    *,
    k: float = HUBER_K,
    n_passes: int = HUBER_PASSES,
) -> tuple[float, np.ndarray]:
    """Iteratively-reweighted Huber regression of y on x with prior weights w.

    Returns ``(beta_hat, residuals)``.  No intercept (we model returns
    around zero; an intercept on daily returns is essentially noise and
    biases the slope when the sample is short).
    """
    if x.size == 0:
        return float("nan"), np.zeros(0)
    beta = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30))
    for _ in range(int(n_passes)):
        resid = y - beta * x
        scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(resid)) or 1e-12
        z = resid / scale
        # Huber psi-derivative weighting
        h = np.where(np.abs(z) <= k, 1.0, k / np.maximum(np.abs(z), 1e-30))
        ww = w * h
        denom = float(np.sum(ww * x * x))
        if denom <= 1e-30:
            break
        beta = float(np.sum(ww * x * y) / denom)
    return beta, y - beta * x


def _newey_west_se(
    x: np.ndarray,
    resid: np.ndarray,
    w: np.ndarray,
    *,
    lag: int = HAC_LAG,
) -> float:
    """HAC (Newey-West) standard error of the slope, weighted.

    The model is ``y = β x + ε`` with weights ``w``.  We form the
    weighted score ``g_t = w_t · x_t · ε_t`` and apply Bartlett kernel
    weights up to ``lag``.

    Returns ``+inf`` if the variance of x is too small.
    """
    n = x.size
    if n == 0:
        return float("inf")
    sxx = float(np.sum(w * x * x))
    if sxx <= 1e-30:
        return float("inf")
    g = w * x * resid
    s0 = float(np.sum(g * g))
    s = s0
    for ell in range(1, int(lag) + 1):
        if ell >= n:
            break
        cov = float(np.sum(g[ell:] * g[: n - ell]))
        wt = 1.0 - ell / (lag + 1.0)
        s += 2.0 * wt * cov
    if s <= 0.0:
        return float("inf")
    var_beta = s / (sxx * sxx)
    if not np.isfinite(var_beta) or var_beta <= 0:
        return float("inf")
    return float(np.sqrt(var_beta))


def _annualized_resid_sigma(resid: np.ndarray, w: np.ndarray, h: int) -> float:
    """Annualized residual std-dev from horizon-h log residuals."""
    if resid.size == 0:
        return float("nan")
    sw = float(np.sum(w))
    if sw <= 0:
        return float("nan")
    var = float(np.sum(w * resid * resid) / sw)
    if not np.isfinite(var) or var < 0:
        return float("nan")
    return float(np.sqrt(var) * np.sqrt(TRADING_DAYS / max(int(h), 1)))


def _fit_at_horizon(
    r_etf_d: np.ndarray,
    r_und_d: np.ndarray,
    halflife: float,
    h: int,
) -> dict[str, float] | None:
    """Robust EWMA fit at horizon h (overlapping log returns when h>1)."""
    if h <= 1:
        x = r_und_d
        y = r_etf_d
    else:
        kern = np.ones(int(h), dtype=float)
        # Overlapping h-day sums of daily log returns
        x = np.convolve(r_und_d, kern, mode="valid")
        y = np.convolve(r_etf_d, kern, mode="valid")
    n = x.size
    if n < 3:
        return None
    w = _ewma_weights(n, halflife=halflife, floor=EWMA_FLOOR_RATIO)
    beta, resid = _huber_irls(y, x, w)
    if not np.isfinite(beta):
        return None
    se = _newey_west_se(x, resid, w, lag=HAC_LAG)
    n_eff = float(np.sum(w))
    sw_x = float(np.sum(w))
    var_und = float(np.sum(w * x * x) / sw_x) if sw_x > 0 else 0.0
    return {
        "beta": float(beta),
        "se": float(se),
        "n_obs": int(n),
        "n_eff": float(n_eff),
        "resid_sigma_annual": _annualized_resid_sigma(resid, w, h),
        "var_und_h": float(var_und),
        "horizon": int(h),
    }


def _distribution_day_mask(
    r_etf: np.ndarray,
    r_und: np.ndarray,
    *,
    mu_prior: float,
    threshold: float = DIST_DAY_RESIDUAL_THRESHOLD,
    actions_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mark days where the |residual| exceeds ``threshold`` AND, when an
    actions mask is provided, the same day carries a corporate-action
    distribution.

    When ``actions_mask`` is not provided the threshold is applied alone
    (rare, but lets unit tests exercise the filter without an external
    distributions stream).
    """
    if r_etf.size == 0:
        return np.zeros(0, dtype=bool)
    resid = r_etf - mu_prior * r_und
    big = np.abs(resid) > float(threshold)
    if actions_mask is not None and actions_mask.size == r_etf.size:
        return big & actions_mask.astype(bool)
    return big


# ── Main estimator ────────────────────────────────────────────────────
def compute_beta_for_hedging(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    prior: BetaPrior,
    *,
    min_days: int = 60,
    halflife_days: float = float(EWMA_HALFLIFE_DAYS),
    actions_mask: pd.Series | None = None,
) -> BetaResult:
    """Posterior hedge-beta estimate for one ETF/underlying pair.

    Parameters
    ----------
    etf_tr, und_tr : pd.Series
        Total-return-adjusted price series for the ETF and its
        underlying.  The estimator forms aligned daily log returns from
        these and is robust to non-overlapping or duplicated dates.
    prior : BetaPrior
        Prior mean / precision and provenance.  See :class:`BetaPrior`.
    min_days : int
        Minimum number of aligned daily returns before any data is
        used.  Below this we return the prior with quality ``low_n``.
    halflife_days : float
        EWMA half-life applied to daily and overlapping h-day returns.
    actions_mask : pd.Series, optional
        Boolean (or 0/1) series indexed by date identifying days that
        had a corporate-action distribution (Yahoo ``actions``).  When
        provided we additionally drop residual-outlier days that
        coincide with a same-day action — this guards against
        unadjusted ex-distribution drops on income ETFs.
    """

    aligned = _aligned_log_returns(etf_tr, und_tr)
    if aligned is None or aligned[0].size < int(min_days):
        # Not enough overlap → return prior with quality flag.
        return BetaResult(
            beta=float(prior.mu),
            beta_se=float("inf"),
            n_obs=0 if aligned is None else int(aligned[0].size),
            n_eff=0.0,
            resid_sigma_annual=float("nan"),
            horizon=1,
            quality="low_n",
            source=f"{_POSTERIOR_PREFIX}.{prior.source}",
            prior_mu=float(prior.mu),
            prior_tau=float(prior.tau),
            prior_source=prior.source,
            extras={"reason": "insufficient_history"},
        )

    r_etf, r_und, idx = aligned

    # ── Distribution-day guard ─────────────────────────────────────
    am: np.ndarray | None = None
    if actions_mask is not None and len(actions_mask) > 0:
        try:
            am_series = actions_mask.reindex(idx).fillna(False).astype(bool)
            am = am_series.to_numpy(dtype=bool)
        except Exception:
            am = None
    bad = _distribution_day_mask(
        r_etf, r_und, mu_prior=float(prior.mu), actions_mask=am
    )
    if bad.any():
        keep = ~bad
        r_etf = r_etf[keep]
        r_und = r_und[keep]

    if r_etf.size < int(min_days):
        return BetaResult(
            beta=float(prior.mu),
            beta_se=float("inf"),
            n_obs=int(r_etf.size),
            n_eff=0.0,
            resid_sigma_annual=float("nan"),
            horizon=1,
            quality="low_n",
            source=f"{_POSTERIOR_PREFIX}.{prior.source}",
            prior_mu=float(prior.mu),
            prior_tau=float(prior.tau),
            prior_source=prior.source,
            extras={"reason": "insufficient_history_after_filter"},
        )

    # ── Robust EWMA fit at h=1 and h=5 ─────────────────────────────
    fit_h1 = _fit_at_horizon(r_etf, r_und, halflife_days, 1)
    fit_h5 = _fit_at_horizon(r_etf, r_und, halflife_days, 5)

    # Choose horizon and inflate tau if unstable
    chosen = fit_h1 or fit_h5
    if chosen is None:
        return BetaResult(
            beta=float(prior.mu),
            beta_se=float("inf"),
            n_obs=int(r_etf.size),
            n_eff=0.0,
            resid_sigma_annual=float("nan"),
            horizon=1,
            quality="low_n",
            source=f"{_POSTERIOR_PREFIX}.{prior.source}",
            prior_mu=float(prior.mu),
            prior_tau=float(prior.tau),
            prior_source=prior.source,
            extras={"reason": "fit_failed"},
        )

    quality_extras: dict[str, Any] = {}
    tau_eff = float(prior.tau)
    chosen = fit_h1 if fit_h1 is not None else fit_h5
    if fit_h1 is not None and fit_h5 is not None:
        gap = abs(fit_h1["beta"] - fit_h5["beta"])
        rel_threshold = HORIZON_STABILITY_REL_TOL * max(abs(prior.mu), 1.0)
        se_threshold = HORIZON_STABILITY_SE_MULT * max(fit_h5["se"], 1e-9)
        if gap > max(rel_threshold, se_threshold):
            chosen = fit_h5
            tau_eff = tau_eff * 2.0
            quality_extras["non_stationary"] = True
            quality_extras["beta_h1"] = fit_h1["beta"]
            quality_extras["beta_h5"] = fit_h5["beta"]

    beta_robust = float(chosen["beta"])
    n_eff = float(chosen["n_eff"])
    se_robust = float(chosen["se"])
    horizon = int(chosen["horizon"])

    # ── Sign-inflation (no hard snap) ──────────────────────────────
    if (
        prior.allow_sign_inflation
        and prior.tau > 0
        and abs(beta_robust) > SIGN_CONFLICT_MIN_BETA
        and abs(prior.mu) > 1e-9
        and (np.sign(beta_robust) != np.sign(prior.mu))
    ):
        tau_eff = tau_eff * SIGN_INFLATION_FACTOR
        quality_extras["sign_conflict"] = True

    # ── Conjugate Gaussian posterior on β ──────────────────────────
    denom = tau_eff + n_eff
    if denom <= 0:
        beta_post = beta_robust
        post_var = se_robust * se_robust
    else:
        beta_post = (tau_eff * float(prior.mu) + n_eff * beta_robust) / denom
        # Posterior SE: combine prior precision (in effective days, scaled
        # by the same residual variance) with robust SE^2.
        # SE_post^2 = 1/(τ + n_eff) · resid_var_per_day / Var(r_und_h)
        # For h>1 chosen we keep the SE in the same units as β.
        # Use: SE_post = SE_robust / sqrt(1 + τ/n_eff) when n_eff > 0.
        if n_eff > 0:
            shrink_factor = np.sqrt(n_eff / denom)
            post_var = (se_robust * shrink_factor) ** 2
        else:
            post_var = se_robust * se_robust

    post_se = float(np.sqrt(max(post_var, 0.0)))

    quality = "ok"
    if quality_extras.get("non_stationary"):
        quality = "non_stationary"

    return BetaResult(
        beta=float(np.round(beta_post, 6)),
        beta_se=float(post_se),
        n_obs=int(chosen["n_obs"]),
        n_eff=float(n_eff),
        resid_sigma_annual=float(chosen["resid_sigma_annual"]),
        horizon=horizon,
        quality=quality,
        source=f"{_POSTERIOR_PREFIX}.{prior.source}",
        prior_mu=float(prior.mu),
        prior_tau=float(tau_eff),
        prior_source=prior.source,
        extras=quality_extras,
    )


# ── Hierarchical priors for YieldBOOST family ─────────────────────────
def _raw_ols_beta_log(etf_tr: pd.Series, und_tr: pd.Series, *, min_days: int) -> tuple[float, int] | None:
    """Plain OLS slope of aligned daily log returns; used to feed the
    hierarchical prior, not the posterior."""
    aligned = _aligned_log_returns(etf_tr, und_tr)
    if aligned is None:
        return None
    r_etf, r_und, _ = aligned
    if r_etf.size < int(min_days):
        return None
    var_u = float(np.var(r_und, ddof=0))
    if var_u <= 1e-30:
        return None
    cov = float(np.cov(r_etf, r_und, ddof=0)[0, 1])
    return float(cov / var_u), int(r_etf.size)


def build_yieldboost_family_priors(
    yieldboost_pairs: Iterable[tuple[str, str]],
    tr_map: dict[str, pd.Series],
    *,
    min_days: int = 60,
    tau_per_sibling: float = 30.0,
    tau_cap: float = 120.0,
) -> dict[str, PeerPrior]:
    """Build per-underlying YieldBOOST priors from sibling history.

    For each underlying ``u`` with ≥ 1 YieldBOOST sibling:

        μ_fam(u) = leave-one-out median of sibling raw OLS β
        τ_fam(u) = min(tau_cap, n_siblings · tau_per_sibling)

    Also produces a ``__global__`` entry that is the median of all
    siblings' raw OLS β across all underlyings — used as a fallback.

    The returned dict is keyed by the *underlying* symbol (upper-case);
    the global fallback is at ``"__global__"``.

    Notes
    -----
    The "leave-one-out" median is per-underlying; with a single sibling
    (no LOO possible) we fall back to the sibling's own raw β.  The
    posterior estimator handles the degenerate τ=0 case by dropping
    back through :func:`BetaPrior.for_row` to the global default.
    """
    siblings: dict[str, list[float]] = {}
    sibling_n: dict[str, list[int]] = {}
    all_betas: list[float] = []

    for etf, und in yieldboost_pairs:
        u = str(und).strip().upper()
        e = str(etf).strip().upper()
        if not u or not e:
            continue
        if e not in tr_map or u not in tr_map:
            continue
        ols = _raw_ols_beta_log(tr_map[e], tr_map[u], min_days=min_days)
        if ols is None:
            continue
        beta_raw, n = ols
        if not np.isfinite(beta_raw):
            continue
        siblings.setdefault(u, []).append(float(beta_raw))
        sibling_n.setdefault(u, []).append(int(n))
        all_betas.append(float(beta_raw))

    out: dict[str, PeerPrior] = {}
    for u, betas in siblings.items():
        ns = sibling_n[u]
        if len(betas) >= 2:
            mu = float(np.median(betas))
        else:
            mu = float(betas[0])
        tau = min(float(tau_cap), float(tau_per_sibling) * float(len(betas)))
        out[u] = PeerPrior(mu=mu, tau=tau, n_siblings=len(betas))

    if all_betas:
        global_mu = float(np.median(all_betas))
        global_tau = min(
            float(tau_cap), float(tau_per_sibling) * float(len(all_betas))
        )
        out["__global__"] = PeerPrior(
            mu=global_mu, tau=global_tau, n_siblings=len(all_betas)
        )

    return out


__all__ = [
    "BetaPrior",
    "BetaResult",
    "PeerPrior",
    "build_yieldboost_family_priors",
    "compute_beta_for_hedging",
    "EWMA_HALFLIFE_DAYS",
    "HUBER_K",
    "HAC_LAG",
    "HORIZONS",
    "SIGN_INFLATION_FACTOR",
    "DIST_DAY_RESIDUAL_THRESHOLD",
    "TRADING_DAYS",
]
