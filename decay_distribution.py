"""Distributional forecast of LETF gross decay from underlying volatility.

This is Phase 1 of the plan in
``Dropbox/Levered ETFs/Research Papers/PLAN_decay_distribution_from_underlying_vol.md``.

Goal
----
Replace a single ``expected_gross_decay_annual`` point estimate with a
forecast **distribution** of integrated variance for the underlying, then map
through the Avellaneda–Zhang identity to get decay quantiles per ETF.

Approach
--------
1. Compute daily realised variance ``RV_t = r_t^2`` (log returns) from the
   underlying total-return series, plus weekly / monthly averages and a
   daily-frequency realised-quarticity proxy ``RQ_t = r_t^4`` (Bollerslev,
   Patton & Quaedvlieg 2016, eqn. (5) reduces to this when only daily data
   is available).
2. **Anchor** the lognormal of horizon-T integrated variance on the
   *empirical* mean and std-dev of ``log( Σ_{t..t+T} RV_t )`` measured by
   rolling the historical RV series. This automatically resolves the
   Jensen-inequality gap between ``E[log RV]`` and ``log E[RV]`` that
   poisons naïve plug-in estimators of the lognormal centre, and it
   natively absorbs autocorrelation, fat tails, and regime mixing.
3. **Refine** the centre with a HARQ-Log forecast shift::

       log RV_{t+1} = α + β₁·log RV^d_t + β₂·log RV^w_t + β₃·log RV^m_t
                       + β₁^Q · √RQ_t / RV_t · log RV^d_t + ε

   The deviation between the latest one-step-ahead conditional forecast
   and the unconditional mean of ``log RV`` gets multiplied by the
   AR(1)-style geometric averaging factor ``(1 − p^T)/(T·(1 − p))`` so
   short-horizon forecasts feel HAR, long-horizon forecasts revert to
   the empirical anchor.
4. Map to decay quantiles via Avellaneda & Zhang::

       D(q) = (β² − β)/2 · IV_T(q)
       IV_T(q) ≈ exp( μ_logIV_T + σ_logIV_T · Φ⁻¹(q) )

References
----------
Avellaneda & Zhang (2010); Andersen, Bollerslev, Diebold & Labys (2003);
Corsi (2009); Bollerslev, Patton & Quaedvlieg (2016).

The module imports only ``numpy`` and ``pandas`` to match the existing
``ls-algo`` dependency surface.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ─── module-level constants ────────────────────────────────────────────────

TRADING_DAYS = 252
_HORIZON_DEFAULT_DAYS = TRADING_DAYS         # 1 year by default
_MIN_DAYS_FOR_HAR = 220                      # need ≥1y of daily RV to fit HARQ
_MIN_DAYS_FOR_EMPIRICAL_SIGMA = 60           # rolling-IV samples after window
_LOOKBACK_MAX_DAYS = 5 * TRADING_DAYS        # cap fit history at 5y
_QUANTILES_DEFAULT = (0.10, 0.50, 0.90)
_RV_FLOOR = 1e-10                            # avoid log(0) on quiet days
_LOG_IV_SIGMA_CAP = 2.0                      # ≈ 7× span at p90, sane bound
_DECAY_SIGN_POSITIVE = True                  # match expected_gross_decay sign


# ─── inverse normal (Acklam 2003) ──────────────────────────────────────────

def _acklam_inv_normal(p: float) -> float:
    """Acklam's rational approximation to ``Φ⁻¹``.

    Max abs error < 1.15e-9 across (0,1). Public-domain reference
    implementation. We avoid scipy/statsmodels here because ls-algo only
    has numpy + pandas.
    """
    if not (0.0 < p < 1.0):
        if p <= 0.0:
            return -math.inf
        return math.inf

    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00)

    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


# ─── realised variance panel ───────────────────────────────────────────────

def _realized_variance_panel(tr_series: pd.Series) -> Optional[pd.DataFrame]:
    """Build the daily RV / weekly / monthly / RQ feature panel.

    Returns ``None`` if the series is too short to be usable.
    """
    if tr_series is None or len(tr_series) < _MIN_DAYS_FOR_HAR:
        return None
    s = pd.to_numeric(tr_series, errors="coerce").dropna()
    s = s.iloc[-(_LOOKBACK_MAX_DAYS + 30):]
    if len(s) < _MIN_DAYS_FOR_HAR:
        return None

    log_ret = np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(log_ret) < _MIN_DAYS_FOR_HAR:
        return None

    rv = (log_ret ** 2).clip(lower=_RV_FLOOR)
    rq = (log_ret ** 4).clip(lower=_RV_FLOOR ** 2)

    panel = pd.DataFrame({
        "RV": rv,
        "RQ": rq,
        "RV_w": rv.rolling(5, min_periods=5).mean(),
        "RV_m": rv.rolling(22, min_periods=22).mean(),
    })
    panel = panel.dropna()
    if len(panel) < _MIN_DAYS_FOR_HAR:
        return None
    return panel


# ─── HARQ-Log regression ───────────────────────────────────────────────────

def _fit_harq_log(panel: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Fit ``log RV_{t+1} = α + β₁·log RV^d_t + β₂·log RV^w_t + β₃·log RV^m_t
    + β₁^Q · √RQ_t/RV_t · log RV^d_t + ε`` by OLS.

    Returns coefficients, residual std-dev, and stationarity flag, or None
    if there isn't enough data or the design is singular.
    """
    if panel is None or len(panel) < _MIN_DAYS_FOR_HAR:
        return None

    log_rv_d = np.log(panel["RV"].to_numpy())
    log_rv_w = np.log(panel["RV_w"].to_numpy())
    log_rv_m = np.log(panel["RV_m"].to_numpy())
    rq_ratio = np.sqrt(panel["RQ"].to_numpy()) / panel["RV"].to_numpy()

    # target = next-day log RV
    y = log_rv_d[1:]
    x_d = log_rv_d[:-1]
    x_w = log_rv_w[:-1]
    x_m = log_rv_m[:-1]
    x_q = rq_ratio[:-1] * log_rv_d[:-1]

    n = len(y)
    if n < _MIN_DAYS_FOR_HAR - 1:
        return None

    # Centre / scale x_q to keep the design well-conditioned (the raw
    # rq_ratio is roughly O(|r|), tiny for low-vol days). We undo this when
    # we use the coefficient on residual diagnostics only — the steady-state
    # forecast does not depend on β₁^Q because it nets out at the long-run
    # mean of x_q (≈ const after centring).
    xq_mean = float(np.mean(x_q))
    xq_std = float(np.std(x_q, ddof=1))
    if xq_std <= 0 or not np.isfinite(xq_std):
        x_q_scaled = np.zeros_like(x_q)
    else:
        x_q_scaled = (x_q - xq_mean) / xq_std

    X = np.column_stack([np.ones(n), x_d, x_w, x_m, x_q_scaled])

    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None

    resid = y - X @ coef
    if len(resid) <= X.shape[1]:
        return None
    rss = float(np.sum(resid ** 2))
    sigma_resid = math.sqrt(rss / (len(resid) - X.shape[1]))

    alpha = float(coef[0])
    b1, b2, b3 = float(coef[1]), float(coef[2]), float(coef[3])
    b1q = float(coef[4])
    persistence = b1 + b2 + b3

    return {
        "alpha": alpha,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b1q": b1q,
        "xq_mean": xq_mean,
        "xq_std": xq_std,
        "sigma_resid": sigma_resid,
        "n_obs": n,
        "persistence": persistence,
        "stationary": persistence < 0.999,
    }


# ─── empirical lognormal of horizon integrated variance ────────────────────

def _empirical_log_iv_moments(
    panel: pd.DataFrame,
    horizon_days: int,
) -> Optional[Tuple[float, float]]:
    """Mean and std-dev of ``log( Σ_{t..t+T} RV_t )`` from history.

    This is the natural empirical lognormal of the horizon-T integrated
    variance: both moments come from the same rolling-window sample, so
    Jensen's inequality between ``E[log RV]`` and ``log E[RV]`` is
    handled automatically. Cap the dispersion to keep extreme tails
    sane on degenerate underlyings.
    """
    if panel is None or horizon_days < 5:
        return None
    rv = panel["RV"]
    if len(rv) < horizon_days + _MIN_DAYS_FOR_EMPIRICAL_SIGMA:
        return None
    rolling = rv.rolling(horizon_days, min_periods=horizon_days).sum()
    log_iv = np.log(rolling.dropna()).replace([np.inf, -np.inf], np.nan).dropna()
    if len(log_iv) < _MIN_DAYS_FOR_EMPIRICAL_SIGMA:
        return None
    mu = float(log_iv.mean())
    sigma = float(log_iv.std(ddof=1))
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
        return None
    return mu, min(sigma, _LOG_IV_SIGMA_CAP)


# ─── HARQ-Log conditional forecast shift ───────────────────────────────────

def _harq_log_conditional_shift(
    fit: Mapping[str, float],
    panel: pd.DataFrame,
    horizon_days: int,
) -> float:
    """One-step-ahead HAR-Log forecast translated into a horizon-T shift
    on top of the unconditional log-IV centre.

    Intuition: HAR says "today's vol is X log-points above the long-run
    mean". Over a T-day forecast horizon, that excess decays back to the
    mean at rate ``persistence``. The *time-averaged* excess over the
    horizon is the geometric-series factor
    ``avg_decay(T) = (1 − p^T) / (T·(1 − p))``. We multiply the one-step
    deviation by ``avg_decay(T)`` to get the average effect on log-IV.

    For 1y horizons (T=252) ``avg_decay`` is small, so HAR-Log mostly
    refines short-horizon forecasts. The β₁^Q HARQ term cancels in
    expectation because we centred the regressor — it shows up in
    residual variance, not in the conditional mean.
    """
    if fit is None or panel is None or horizon_days <= 0:
        return 0.0
    try:
        log_rv_d = float(np.log(panel["RV"].iloc[-1]))
        log_rv_w = float(np.log(panel["RV_w"].iloc[-1]))
        log_rv_m = float(np.log(panel["RV_m"].iloc[-1]))
    except (KeyError, IndexError, ValueError):
        return 0.0

    one_step = (
        fit["alpha"]
        + fit["b1"] * log_rv_d
        + fit["b2"] * log_rv_w
        + fit["b3"] * log_rv_m
    )
    unconditional = float(np.log(panel["RV"]).mean())

    p = float(fit["persistence"])
    if not (0.0 < p < 1.0):
        avg_decay = 0.0
    else:
        try:
            avg_decay = (1.0 - p ** horizon_days) / (horizon_days * (1.0 - p))
        except (OverflowError, ValueError):
            avg_decay = 0.0
    shift = (one_step - unconditional) * avg_decay
    if not np.isfinite(shift):
        return 0.0
    return float(np.clip(shift, -1.0, 1.0))  # belt-and-braces clamp


# ─── decay quantile mapping ────────────────────────────────────────────────

def _c_beta(beta: float) -> float:
    """Avellaneda–Zhang coefficient on integrated variance: (β² − β)/2.

    Positive for |β|>1 and for β<0; zero at β∈{0,1}. We multiply by this
    and report decay as a positive drag, matching ``expected_gross_decay``.
    """
    return 0.5 * abs(beta) * abs(beta - 1.0)


def _lognormal_decay_from_logiv(
    mu_log_iv: float,
    sigma_log_iv: float,
    beta: float,
    quantiles: Sequence[float],
) -> Dict[str, float]:
    """Map ``log IV_T ~ N(μ, σ²)`` to gross-decay quantiles + mean."""
    cb = _c_beta(beta)
    out: Dict[str, float] = {}
    sigma = max(0.0, float(sigma_log_iv)) if np.isfinite(sigma_log_iv) else 0.0
    for q in quantiles:
        z = _acklam_inv_normal(float(q))
        iv_q = math.exp(mu_log_iv + sigma * z)
        out[f"p{int(round(q * 100)):02d}"] = round(cb * iv_q, 6)
    mean_iv = math.exp(mu_log_iv + 0.5 * sigma ** 2)
    out["mean"] = round(cb * mean_iv, 6)
    out["mu_log_iv"] = round(float(mu_log_iv), 6)
    out["sigma_log_iv"] = round(float(sigma), 6)
    return out


# ─── public per-underlying entry point ─────────────────────────────────────

def forecast_decay_distribution(
    tr_series: pd.Series,
    beta: float,
    *,
    horizon_days: int = _HORIZON_DEFAULT_DAYS,
    quantiles: Sequence[float] = _QUANTILES_DEFAULT,
    fallback_expected_decay: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Forecast the gross-decay distribution for one (underlying, β) pair.

    Returns ``None`` if there is neither enough history to fit HARQ-Log
    nor a usable fallback expected-decay value.
    """
    if beta is None or not np.isfinite(beta):
        return None
    if abs(beta - 1.0) < 1e-9 or abs(beta) < 1e-9:
        # No vol drag at β∈{0,1} by construction.
        zero = {f"p{int(round(q * 100)):02d}": 0.0 for q in quantiles}
        zero.update({"mean": 0.0, "mu_log_iv": float("nan"),
                     "sigma_log_iv": 0.0, "model": "no_drag_beta"})
        return zero

    panel = _realized_variance_panel(tr_series)
    moments = _empirical_log_iv_moments(panel, horizon_days) if panel is not None else None
    fit = _fit_harq_log(panel) if panel is not None else None

    if moments is not None:
        mu_emp, sigma_emp = moments
        if fit is not None:
            mu = mu_emp + _harq_log_conditional_shift(fit, panel, horizon_days)
            model = "harq_log_anchored"
            n_obs = float(fit["n_obs"])
        else:
            mu = mu_emp
            model = "empirical_lognormal"
            n_obs = float(len(panel)) if panel is not None else 0.0
        out = _lognormal_decay_from_logiv(mu, sigma_emp, beta, quantiles)
        out["model"] = model
        out["n_obs"] = n_obs
        if fit is not None:
            out["persistence"] = round(fit["persistence"], 6)
        return out

    # Fallback: not enough history for an empirical lognormal. Centre on the
    # caller-supplied simple-Itô expected decay and emit a point estimate
    # (zero dispersion).
    if fallback_expected_decay is not None and np.isfinite(fallback_expected_decay):
        cb = _c_beta(beta)
        if cb > 0:
            iv_implied = max(float(fallback_expected_decay) / cb, _RV_FLOOR)
            mu_log_iv = math.log(iv_implied)
        else:
            mu_log_iv = float("nan")
        out = _lognormal_decay_from_logiv(mu_log_iv, 0.0, beta, quantiles)
        out["model"] = "simple_ito_fallback"
        out["n_obs"] = 0.0
        return out

    return None


# ─── DataFrame enrichment ──────────────────────────────────────────────────

_OUTPUT_COLUMNS = (
    "expected_gross_decay_p10_annual",
    "expected_gross_decay_p50_annual",
    "expected_gross_decay_p90_annual",
    "expected_gross_decay_mean_annual",
    "expected_gross_decay_dist_model",
    "expected_logIV_mu_annual",
    "expected_logIV_sigma_annual",
    "expected_gross_decay_dist_n_obs",
    "expected_gross_decay_dist_horizon_days",
)


def enrich_with_decay_distribution(
    df: pd.DataFrame,
    tr_map: Mapping[str, pd.Series],
    *,
    horizon_days: int = _HORIZON_DEFAULT_DAYS,
    quantiles: Sequence[float] = _QUANTILES_DEFAULT,
    underlying_col: str = "Underlying",
    beta_col: str = "Beta",
    fallback_col: str = "expected_gross_decay_annual",
    norm_sym: Optional[callable] = None,
) -> pd.DataFrame:
    """Add ``expected_gross_decay_{p10,p50,p90,mean}_annual`` per row.

    The forecast is computed **once per unique underlying** (matching the
    existing ``enrich_with_decay_and_vol`` pattern), then applied to every
    ETF with that underlying using its own β.

    The function never mutates ``expected_gross_decay_annual``; it adds
    sibling columns so downstream consumers can opt in incrementally.
    """
    if norm_sym is None:
        def norm_sym(x):  # type: ignore[no-redef]
            try:
                return str(x).strip().upper()
            except Exception:
                return ""

    out = df.copy()
    for col in _OUTPUT_COLUMNS:
        if col not in out.columns:
            if col == "expected_gross_decay_dist_model":
                out[col] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
            else:
                out[col] = np.nan

    if underlying_col not in out.columns or beta_col not in out.columns:
        return out

    # Cache: underlying symbol → (panel, fit, sigma_log_iv, mu_log_rv_steady)
    und_cache: Dict[str, Optional[Dict[str, float]]] = {}

    print(
        f"[DECAY-DIST] Forecasting decay distribution at "
        f"horizon={horizon_days}d (quantiles={list(quantiles)})..."
    )
    n_har = n_fallback_a = n_fallback_b = n_skip = 0

    for idx, row in out.iterrows():
        und_raw = row.get(underlying_col)
        if pd.isna(und_raw) or not str(und_raw).strip():
            n_skip += 1
            continue
        und = norm_sym(und_raw)
        beta_val = row.get(beta_col)
        try:
            beta = float(beta_val)
        except (TypeError, ValueError):
            n_skip += 1
            continue
        if not np.isfinite(beta):
            n_skip += 1
            continue

        if und not in und_cache:
            tr_series = tr_map.get(und) if isinstance(tr_map, Mapping) else None
            panel = _realized_variance_panel(tr_series) if tr_series is not None else None
            moments = _empirical_log_iv_moments(panel, horizon_days) if panel is not None else None
            fit = _fit_harq_log(panel) if panel is not None else None

            if moments is not None:
                mu_emp, sigma_emp = moments
                if fit is not None:
                    mu = mu_emp + _harq_log_conditional_shift(fit, panel, horizon_days)
                    cache_model = "harq_log_anchored"
                    cache_n_obs = float(fit["n_obs"])
                else:
                    mu = mu_emp
                    cache_model = "empirical_lognormal"
                    cache_n_obs = float(len(panel))
                und_cache[und] = {
                    "model": cache_model,
                    "mu_log_iv": mu,
                    "sigma_log_iv": sigma_emp,
                    "n_obs": cache_n_obs,
                }
            else:
                und_cache[und] = None

        cache = und_cache[und]
        if cache is not None:
            mapped = _lognormal_decay_from_logiv(
                cache["mu_log_iv"],
                cache["sigma_log_iv"],
                beta,
                quantiles,
            )
            model = cache["model"]
            n_obs = cache["n_obs"]
            if model == "harq_log_anchored":
                n_har += 1
            else:
                n_fallback_a += 1
        else:
            fb = row.get(fallback_col) if fallback_col in out.columns else None
            try:
                fb_val = float(fb) if fb is not None and pd.notna(fb) else None
            except (TypeError, ValueError):
                fb_val = None
            if fb_val is None or not np.isfinite(fb_val):
                n_skip += 1
                continue
            cb = _c_beta(beta)
            if cb <= 0:
                n_skip += 1
                continue
            iv_implied = max(fb_val / cb, _RV_FLOOR)
            mapped = _lognormal_decay_from_logiv(
                math.log(iv_implied), 0.0, beta, quantiles,
            )
            model = "simple_ito_fallback"
            n_obs = 0.0
            n_fallback_b += 1

        out.at[idx, "expected_gross_decay_p10_annual"] = mapped.get("p10", np.nan)
        out.at[idx, "expected_gross_decay_p50_annual"] = mapped.get("p50", np.nan)
        out.at[idx, "expected_gross_decay_p90_annual"] = mapped.get("p90", np.nan)
        out.at[idx, "expected_gross_decay_mean_annual"] = mapped.get("mean", np.nan)
        out.at[idx, "expected_logIV_mu_annual"] = mapped.get("mu_log_iv", np.nan)
        out.at[idx, "expected_logIV_sigma_annual"] = mapped.get("sigma_log_iv", np.nan)
        out.at[idx, "expected_gross_decay_dist_model"] = model
        out.at[idx, "expected_gross_decay_dist_n_obs"] = n_obs
        out.at[idx, "expected_gross_decay_dist_horizon_days"] = float(horizon_days)

    print(
        f"[DECAY-DIST] harq_log={n_har} | "
        f"empirical_sigma={n_fallback_a} | "
        f"simple_ito_fallback={n_fallback_b} | "
        f"skipped={n_skip}"
    )
    return out


__all__ = (
    "enrich_with_decay_distribution",
    "forecast_decay_distribution",
)
