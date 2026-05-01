# -*- coding: utf-8 -*-
"""
Sizing v2: net-edge p50 + bootstrap uncertainty + Kelly-QP cov + drawdown brake.

This module is *additive* to ``scripts/gtp_sizing_mirror.py`` -- it consumes
the same ``GTP_MIRROR_DF`` produced by ``mirror_generate_trade_plan_sizing`` and
returns drop-in replacements for ``PAIR_WEIGHTS`` / ``PAIR_FRAC_BY_KEY``.

Design principles:

* The expected-edge signal is the **bootstrap p50 of the net edge** computed in
  ``screener_v2_fields.py`` (column ``net_edge_p50_annual``, sign convention
  short-favorable positive, after borrow). This replaces the legacy
  ``blended_gross_decay - borrow_aversion * borrow_current`` proxy.
* A Sinclair-style **parameter-uncertainty haircut** uses the same bootstrap
  distribution: ``sigma_edge ~= (p95 - p05) / (2 * 1.6449)`` (one-sided 5/95
  interval is approx +/-1.645 sigma for normal). Names whose p50 is small
  relative to this sigma get shrunk toward zero (Han et al. 2019 weight
  shrinkage).
* The covariance penalty is a **closed-form-ish Kelly QP** on shrunk underlying
  covariance -- strictly generalizes the heuristic ``1/(1 + lambda * MRC_tilde)``
  in ``gtp_sizing_mirror.covariance_overlay_signal_weights``.
* An optional pair-spread vol normalization replaces ``(1/|beta|)^margin_power``.
* A scalar **drawdown brake** (Hsieh & Barmish 2017 / Busseti, Ryu, Boyd 2016)
  reduces book gross when running NAV drawdown breaches a threshold.

Imports do not pull in ``generate_trade_plan`` to keep this module loadable
from a notebook without the full production runtime; it depends only on
NumPy/pandas plus the small ``_norm_sym`` helper inlined below.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Small utilities (intentionally duplicated from generate_trade_plan to keep
# this module importable without the full ls-algo runtime).
# ---------------------------------------------------------------------------


def _norm_sym(x: Any) -> str:
    return str(x).strip().upper().replace(".", "-")


def _clip01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _safe_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns from a (date x sym) price frame, NaN-safe."""
    p = price_df.replace([np.inf, -np.inf], np.nan).astype(float)
    p = p.where(p > 1e-12)
    return np.log(p).diff()


# ---------------------------------------------------------------------------
# 1) Net-edge -> per-pair pre-weights (with Sinclair uncertainty haircut)
# ---------------------------------------------------------------------------

# 1.6449 ~ Phi^{-1}(0.95). p95 - p05 spans approx +/-1.645 sigma under a
# normal, so sigma_hat ~= (p95 - p05) / (2 * 1.6449).
_NORMAL_Z90 = 1.6448536269514722


def net_edge_signal_table(
    mirror_df: pd.DataFrame,
    *,
    edge_col: str = "net_edge_p50_annual",
    p05_col: str = "net_edge_p05_annual",
    p95_col: str = "net_edge_p95_annual",
    p25_col: str = "net_edge_p25_annual",
    p75_col: str = "net_edge_p75_annual",
    sigma_floor: float = 0.04,
    fallback_blended: bool = True,
) -> pd.DataFrame:
    """
    Build a per-pair signal frame from ``mirror_df``.

    Output columns (one row per (ETF, Underlying)):

    * ``mu_hat``        bootstrap p50 of net edge (annual, short-favorable +).
    * ``sigma_hat``     bootstrap sigma proxy ``(p95 - p05) / (2 * 1.645)``
                        clipped at ``sigma_floor`` so we never divide by ~0.
    * ``sigma_iqr``     robust sigma proxy ``(p75 - p25) / 1.349`` (Gaussian IQR).
    * ``has_bootstrap`` bool, whether all bootstrap percentiles were finite.
    * ``mu_fallback``   ``blended_gross_decay - borrow_current`` (today's proxy)
                        used only if ``fallback_blended`` and bootstrap is missing.

    The signal is in the **same units / sign convention** as the production
    ``net_edge_*`` columns: positive = short-favorable, after borrow, annual.
    """
    df = mirror_df.copy()
    if "ETF" not in df.columns or "Underlying" not in df.columns:
        raise KeyError("mirror_df must contain ETF and Underlying columns")
    df["ETF"] = df["ETF"].astype(str).map(_norm_sym)
    df["Underlying"] = df["Underlying"].astype(str).map(_norm_sym)
    p50 = pd.to_numeric(df.get(edge_col), errors="coerce")
    p05 = pd.to_numeric(df.get(p05_col), errors="coerce")
    p95 = pd.to_numeric(df.get(p95_col), errors="coerce")
    p25 = pd.to_numeric(df.get(p25_col), errors="coerce")
    p75 = pd.to_numeric(df.get(p75_col), errors="coerce")

    sigma_boot = (p95 - p05) / (2.0 * _NORMAL_Z90)
    sigma_iqr = (p75 - p25) / 1.349
    sigma_hat = sigma_boot.where(sigma_boot.notna(), sigma_iqr)
    sigma_hat = sigma_hat.clip(lower=float(sigma_floor))

    has_boot = p50.notna() & p05.notna() & p95.notna()

    if fallback_blended:
        bd = pd.to_numeric(df.get("blended_gross_decay"), errors="coerce")
        bc = pd.to_numeric(df.get("borrow_current"), errors="coerce").fillna(0.0)
        mu_fb = bd - bc
    else:
        mu_fb = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(
        {
            "ETF": df["ETF"].values,
            "Underlying": df["Underlying"].values,
            "mu_hat": p50.where(has_boot, mu_fb).astype(float).values,
            "sigma_hat": sigma_hat.astype(float).values,
            "sigma_iqr": sigma_iqr.astype(float).values,
            "has_bootstrap": has_boot.astype(bool).values,
            "mu_fallback": mu_fb.astype(float).values,
        },
        index=df.index,
    )
    return out


def kelly_pre_weights_from_net_edge(
    mirror_df: pd.DataFrame,
    *,
    kelly_fraction: float = 0.35,
    sigma_floor: float = 0.04,
    uncertainty_haircut: bool = True,
    haircut_floor: float = 0.10,
    haircut_cap: float = 1.0,
    edge_floor: float = 0.0,
) -> pd.DataFrame:
    """
    Per-pair non-negative pre-weights from the ``net_edge_p50_annual`` signal.

    Formula (per pair i):

        w_pre_i = max(mu_hat_i - edge_floor, 0)
                  * (kelly_fraction / sigma_hat_i^2)
                  * uncertainty_haircut(mu_hat_i / sigma_hat_i)

    where the Sinclair-style uncertainty haircut is

        h(t) = clip( |t| / sqrt(1 + t^2), haircut_floor, haircut_cap )

    so a name whose p50 is < 1 sigma_hat above zero gets at least an
    additional ~50% discount on top of the fractional-Kelly scale, while a
    name whose p50 >> sigma_hat gets ~1 (unhaircut). Setting
    ``uncertainty_haircut=False`` recovers pure ``mu_hat / sigma_hat^2``
    weighting.

    The output is non-negative -- names with mu_hat <= edge_floor get 0.
    The caller is responsible for sleeve-level normalization.
    """
    sig = net_edge_signal_table(mirror_df, sigma_floor=sigma_floor)
    mu = sig["mu_hat"].astype(float)
    s = sig["sigma_hat"].astype(float).clip(lower=float(sigma_floor))

    edge = (mu - float(edge_floor)).clip(lower=0.0)
    base = float(kelly_fraction) * edge / (s * s)

    if uncertainty_haircut:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = mu / s
            t = t.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            h = np.abs(t) / np.sqrt(1.0 + t * t)
        h = h.clip(lower=float(haircut_floor), upper=float(haircut_cap))
        w_pre = base * h
    else:
        w_pre = base

    out = sig.copy()
    out["edge_clipped"] = edge.values
    out["w_pre"] = w_pre.astype(float).values
    return out


# ---------------------------------------------------------------------------
# 2) Pair-spread sigma^2 normalization (replaces (1/|beta|)^margin_power)
# ---------------------------------------------------------------------------


def pair_spread_vol(
    mirror_df: pd.DataFrame,
    underlying_returns: pd.DataFrame,
    etf_returns: pd.DataFrame | None = None,
    *,
    lookback: int = 126,
    sigma_floor_annual: float = 0.05,
    trading_days: int = 252,
) -> pd.Series:
    """
    Annualized sigma of the pair spread ``r_und - r_etf / |beta|`` per
    (ETF, Underlying).

    If ``etf_returns`` is None we fall back to a naive proxy
    ``sigma_und / |beta|`` (the spread vol of a perfectly hedged pair under
    GBM).

    Returns a series indexed by ``(etf, und)`` tuples, in **annualized stdev**,
    floored at ``sigma_floor_annual``.

    NOTE: ``underlying_returns`` (and ``etf_returns``) are expected as **price
    levels** (columns = symbols, index = date); we convert to log returns
    internally.
    """
    df = mirror_df.copy()
    df["ETF"] = df["ETF"].astype(str).map(_norm_sym)
    df["Underlying"] = df["Underlying"].astype(str).map(_norm_sym)
    if "Beta" not in df.columns and "beta_abs" not in df.columns:
        raise KeyError("mirror_df missing Beta / beta_abs column")
    beta = pd.to_numeric(df.get("Beta"), errors="coerce").abs()
    if "beta_abs" in df.columns:
        beta = beta.where(
            beta.notna(), pd.to_numeric(df["beta_abs"], errors="coerce").abs()
        )
    beta = beta.clip(lower=0.5)

    u_ret = _safe_log_returns(underlying_returns)
    u_ret = u_ret.iloc[-int(max(lookback, 30)):]

    e_ret = None
    if etf_returns is not None and not etf_returns.empty:
        e_ret = _safe_log_returns(etf_returns).iloc[-int(max(lookback, 30)):]

    out: dict[tuple[str, str], float] = {}
    for idx, row in df.iterrows():
        e = _norm_sym(row["ETF"])
        u = _norm_sym(row["Underlying"])
        b = float(beta.loc[idx]) if pd.notna(beta.loc[idx]) else 1.0
        sig: float | None = None
        if e_ret is not None and (e in e_ret.columns) and (u in u_ret.columns):
            spread = u_ret[u] - e_ret[e] / max(abs(b), 0.5)
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
            if len(spread) >= 20:
                sig = float(spread.std(ddof=1) * math.sqrt(trading_days))
        if sig is None and (u in u_ret.columns):
            sru = u_ret[u].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sru) >= 20:
                sig = (
                    float(sru.std(ddof=1) * math.sqrt(trading_days))
                    / max(abs(b), 0.5)
                )
        out[(e, u)] = max(
            float(sigma_floor_annual),
            float(sig if sig is not None else sigma_floor_annual),
        )
    return pd.Series(out, name="pair_spread_vol_annual")


# ---------------------------------------------------------------------------
# 3) Kelly-QP overlay on shrunk underlying covariance
# ---------------------------------------------------------------------------


def _shrink_cov_constant_corr(cov: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Ledoit-Wolf-style shrinkage to a constant-correlation target.

    Target Sigma_hat has identical diagonal as ``cov`` and all off-diagonals
    equal to the average pairwise correlation times the geometric mean of
    the standard deviations. Pure Sigma_diagonal shrinkage is the special
    case alpha -> 1 with constant-corr -> diag.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    C = cov.values.astype(float)
    if C.size == 0:
        return cov.copy()
    d = np.sqrt(np.clip(np.diag(C), 0.0, None))
    if np.any(d <= 0):
        D = np.diag(np.diag(C))
        return pd.DataFrame(
            (1 - a) * C + a * D, index=cov.index, columns=cov.columns
        )
    R = C / np.outer(d, d)
    n = R.shape[0]
    if n > 1:
        offdiag = R[~np.eye(n, dtype=bool)]
        rho_bar = float(np.nanmean(offdiag)) if offdiag.size else 0.0
        rho_bar = float(np.clip(rho_bar, -0.95, 0.95))
    else:
        rho_bar = 0.0
    R_target = np.full_like(R, rho_bar)
    np.fill_diagonal(R_target, 1.0)
    C_target = R_target * np.outer(d, d)
    out = (1 - a) * C + a * C_target
    return pd.DataFrame(out, index=cov.index, columns=cov.columns)


def _project_simplex_nonneg(v: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Project ``v`` onto ``{w : w >= 0, sum w = total}`` (Duchi et al. 2008)."""
    if total <= 0:
        return np.zeros_like(v)
    n = v.size
    if n == 0:
        return v.copy()
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - total
    rng = np.arange(1, n + 1)
    cond = u - css / rng > 0
    if not np.any(cond):
        return np.full_like(v, total / n)
    rho = int(np.max(np.where(cond)))
    theta = css[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def kelly_qp_overlay(
    pre_weights: np.ndarray,
    sym_labels: Sequence[str],
    ret_df: pd.DataFrame,
    *,
    mu_per_sym: np.ndarray | None = None,
    kelly_fraction: float = 0.35,
    shrink_alpha: float = 0.35,
    min_obs: int = 30,
    max_w: float | None = None,
    n_iters: int = 60,
    step_decay: float = 0.85,
    project_to_total: float | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Closed-form-ish Kelly QP on shrunk covariance.

    Solves (heuristically, projected gradient):

        max_w   w . mu_hat  -  (1 / (2 * kappa)) * w . Sigma_shrunk . w
        s.t.    w >= 0, sum(w) <= sum(pre_weights) [or = project_to_total]
                w_i <= max_w (per-name cap, optional)

    where mu_hat defaults to ``pre_weights`` (i.e. start from the signal
    weights as expected-return surrogates), and kappa = ``kelly_fraction``.

    This **strictly generalizes** the heuristic in
    ``gtp_sizing_mirror.covariance_overlay_signal_weights``: the heuristic is
    one Newton-like step at the symmetric point ``w_n = pre_weights / sum``;
    here we run a few projected-gradient iterations on the shrunk QP, which
    gives a true (penalized) variance-aware allocation.

    Returns ``(w_out, shrunk_df, meta)`` matching the existing helper's
    tuple so the notebook A/B harness can drop this in.
    """
    pre = np.asarray(pre_weights, dtype=float)
    n = pre.size
    if n == 0:
        return pre.copy(), pd.DataFrame(), pd.DataFrame()

    target_total = (
        float(project_to_total) if project_to_total is not None else float(max(pre.sum(), 0.0))
    )
    if target_total <= 0:
        return pre.copy(), pd.DataFrame(), pd.DataFrame()

    r = ret_df.reindex(columns=list(sym_labels)).astype(float)
    cov = r.cov(min_periods=int(min_obs))
    if cov.isna().values.any():
        diag = pd.Series(
            {
                c: float(r[c].dropna().var(ddof=1))
                if r[c].dropna().size >= int(min_obs)
                else 1e-4
                for c in sym_labels
            }
        )
        cov = pd.DataFrame(
            np.diag(diag.values), index=list(sym_labels), columns=list(sym_labels)
        )
    shrunk_df = _shrink_cov_constant_corr(cov, alpha=float(shrink_alpha))
    S = shrunk_df.values

    if mu_per_sym is None:
        mu = pre.copy()
    else:
        mu = np.asarray(mu_per_sym, dtype=float)

    cap = (
        float(max_w) * target_total
        if (max_w is not None and float(max_w) > 0)
        else None
    )

    try:
        eig = float(np.max(np.abs(np.linalg.eigvalsh(S))))
        if not np.isfinite(eig) or eig <= 0:
            eig = float(np.max(np.diag(S)) + 1e-8)
    except np.linalg.LinAlgError:
        eig = float(np.max(np.diag(S)) + 1e-8)
    kappa = max(float(kelly_fraction), 1e-6)
    step = kappa / max(eig, 1e-12)

    w = pre.copy()
    for _ in range(int(max(n_iters, 1))):
        grad = mu - (S @ w) / kappa
        w = w + step * grad
        if cap is not None:
            w = np.minimum(w, cap)
        w = _project_simplex_nonneg(w, total=target_total)
        step *= float(step_decay)

    meta = pd.DataFrame(
        {
            "sym": list(sym_labels),
            "w_pre": pre,
            "mu_qp": mu,
            "w_post": w,
            "var_contrib": (w * (S @ w)),
        }
    )
    return w, shrunk_df, meta


# ---------------------------------------------------------------------------
# 4) Drawdown brake (Hsieh & Barmish 2017 / Busseti et al. 2016 inspired)
# ---------------------------------------------------------------------------


def dd_brake_multiplier(
    nav_series: pd.Series | np.ndarray,
    *,
    dd_threshold: float = -0.05,
    gamma: float = 5.0,
    lev_floor_ratio: float = 0.40,
) -> float:
    """
    Scalar in [lev_floor_ratio, 1.0] that multiplies the target gross.

        DD_t = NAV_t / runmax(NAV) - 1   (<= 0)
        m_t  = clip(1 - gamma * max(0, dd_threshold - DD_t),
                    lev_floor_ratio, 1.0)

    With defaults: 5 % drawdown gives m = 1; 10 % gives m = 1 - 5*0.05 = 0.75;
    20 % gives m = floor (0.40). Pure scalar -- apply at the engine level.
    """
    s = pd.Series(nav_series).astype(float).dropna()
    if s.size < 2:
        return 1.0
    cm = s.cummax()
    if cm.iloc[-1] <= 0:
        return float(lev_floor_ratio)
    dd = float(s.iloc[-1] / cm.iloc[-1] - 1.0)
    excess = max(0.0, float(dd_threshold) - dd)
    m = 1.0 - float(gamma) * excess
    return float(np.clip(m, float(lev_floor_ratio), 1.0))


# ---------------------------------------------------------------------------
# 5) End-to-end builder: mirror_df -> PAIR_WEIGHTS / PAIR_FRAC_BY_KEY
# ---------------------------------------------------------------------------


def pair_weights_with_net_edge_kelly(
    universe: list[tuple[str, str, float]],
    mirror_df: pd.DataFrame,
    underlying_returns: pd.DataFrame,
    *,
    kelly_fraction: float = 0.35,
    sigma_floor: float = 0.04,
    uncertainty_haircut: bool = True,
    haircut_floor: float = 0.10,
    edge_floor: float = 0.0,
    use_pair_vol_normalize: bool = False,
    etf_returns: pd.DataFrame | None = None,
    pair_vol_lookback: int = 126,
    cov_shrink_alpha: float = 0.35,
    cov_min_obs: int = 30,
    max_pair_weight: float | None = None,
    max_underlying_weight: float | None = None,
    fallback_to_mirror: bool = True,
) -> tuple[dict[str, float], dict[tuple[str, str], float], pd.DataFrame]:
    """
    Build PAIR_WEIGHTS (ETF -> weight) and PAIR_FRAC_BY_KEY ((ETF, U) -> frac)
    using the bootstrap p50 of net edge as the expected-return signal, with
    optional pair-sigma normalization, the Kelly QP cov overlay, and per-name
    caps.

    Pipeline:

    1. Compute per-pair ``w_pre`` from ``net_edge_p50`` (with uncertainty
       haircut), keyed by (ETF, U).
    2. Optional: divide ``w_pre`` by ``sigma_pair_spread^2`` (still per pair).
    3. Aggregate to underlying-level signal exposure
       ``s_u = sum_e w_pre(e, u)``.
    4. Run the Kelly QP overlay on shrunk Sigma of underlying log returns to
       obtain ``s_u_post`` (still summing to ``sum s_u``, with simplex
       projection plus optional per-name cap).
    5. Distribute multipliers ``m_u = s_u_post / s_u`` back to each ETF and
       renormalize so PAIR_WEIGHTS sums to 1 across the universe.
    6. Build PAIR_FRAC_BY_KEY by the same multipliers applied at the (ETF, U)
       level so Bucket 4 / Bucket 1 / Bucket 2 weights are preserved.
    7. Apply per-name cap with **redistribution** (not discard) to close the
       silent-gross leak in the legacy overlay.

    Returns:
        pair_weights : dict[etf -> float] (sums to 1 across universe ETFs)
        pair_frac    : dict[(etf, und) -> float] (sums to 1 across pairs)
        meta         : DataFrame with diagnostics (mu_hat, sigma_hat,
                       w_pre, w_post, ...)
    """
    if mirror_df is None or mirror_df.empty:
        n = max(len(universe), 1)
        return (
            {_norm_sym(e): 1.0 / n for e, _, _ in universe},
            {(_norm_sym(e), _norm_sym(u)): 1.0 / n for e, u, _ in universe},
            pd.DataFrame(),
        )

    sig = kelly_pre_weights_from_net_edge(
        mirror_df,
        kelly_fraction=kelly_fraction,
        sigma_floor=sigma_floor,
        uncertainty_haircut=uncertainty_haircut,
        haircut_floor=haircut_floor,
        edge_floor=edge_floor,
    )
    sig["pair_key"] = list(zip(sig["ETF"], sig["Underlying"]))
    pair_pre: dict[tuple[str, str], float] = {}
    for _, r in sig.iterrows():
        pk = (_norm_sym(r["ETF"]), _norm_sym(r["Underlying"]))
        pair_pre[pk] = pair_pre.get(pk, 0.0) + float(max(r["w_pre"], 0.0))

    universe_keys = [(_norm_sym(e), _norm_sym(u)) for e, u, _ in universe]
    for k in universe_keys:
        pair_pre.setdefault(k, 0.0)

    if use_pair_vol_normalize and pair_pre:
        psv = pair_spread_vol(
            mirror_df,
            underlying_returns,
            etf_returns=etf_returns,
            lookback=int(pair_vol_lookback),
        )
        for k in list(pair_pre.keys()):
            sv = float(psv.get(k, np.nan)) if hasattr(psv, "get") else float("nan")
            if not np.isfinite(sv) or sv <= 0:
                continue
            pair_pre[k] = pair_pre[k] / (sv * sv)

    total_pre = float(sum(pair_pre.values()))
    if total_pre <= 0:
        if fallback_to_mirror:
            n = max(len(universe), 1)
            pair_weights = {_norm_sym(e): 1.0 / n for e, _, _ in universe}
            pair_frac = {
                (_norm_sym(e), _norm_sym(u)): 1.0 / n for e, u, _ in universe
            }
            return pair_weights, pair_frac, sig
        return ({_norm_sym(e): 0.0 for e, _, _ in universe}, {}, sig)

    u_exposure: dict[str, float] = {}
    for (_e, u), v in pair_pre.items():
        u_exposure[u] = u_exposure.get(u, 0.0) + float(v)
    syms = sorted(s for s, v in u_exposure.items() if v > 1e-18)
    if not syms:
        n = max(len(universe), 1)
        return (
            {_norm_sym(e): 1.0 / n for e, _, _ in universe},
            {(_norm_sym(e), _norm_sym(u)): 1.0 / n for e, u, _ in universe},
            sig,
        )

    s_u_pre = np.array([float(u_exposure[s]) for s in syms])

    r = underlying_returns.copy()
    r.columns = [_norm_sym(c) for c in r.columns]
    r = r[[c for c in r.columns if c in syms]]
    r_log = _safe_log_returns(r) if r.shape[1] > 0 else pd.DataFrame()
    can_qp = (
        r_log.shape[1] > 1
        and len(r_log.dropna(axis=0, how="all")) >= int(cov_min_obs)
    )
    if can_qp:
        s_u_post, shrunk_df, qp_meta = kelly_qp_overlay(
            s_u_pre,
            syms,
            r_log,
            mu_per_sym=s_u_pre,
            kelly_fraction=float(kelly_fraction),
            shrink_alpha=float(cov_shrink_alpha),
            min_obs=int(cov_min_obs),
            max_w=(
                float(max_underlying_weight)
                if max_underlying_weight is not None and float(max_underlying_weight) > 0
                else None
            ),
            project_to_total=float(s_u_pre.sum()),
        )
    else:
        s_u_post = s_u_pre.copy()
        shrunk_df = pd.DataFrame()
        qp_meta = pd.DataFrame({"sym": syms, "w_pre": s_u_pre, "w_post": s_u_post})

    mult: dict[str, float] = {}
    for i, s in enumerate(syms):
        denom = float(s_u_pre[i])
        mult[s] = float(s_u_post[i] / denom) if denom > 1e-18 else 0.0

    pair_post: dict[tuple[str, str], float] = {}
    for (e, u), v in pair_pre.items():
        m = float(mult.get(u, 1.0 if v > 0 else 0.0))
        pair_post[(e, u)] = max(0.0, float(v) * m)

    if max_pair_weight is not None and float(max_pair_weight) > 0:
        cap_total = float(sum(pair_post.values()))
        if cap_total > 0:
            cap_abs = float(max_pair_weight) * cap_total
            # Only redistribute among rows with positive pre-cap desired so that
            # screened-out pairs (mu_hat <= 0) stay at zero. The legacy helper
            # would otherwise spread the surplus across every row with headroom.
            pos_keys = [k for k, v in pair_post.items() if float(v) > 0]
            if pos_keys:
                desired = np.array([pair_post[k] for k in pos_keys], dtype=float)
                caps = np.full_like(desired, cap_abs)
                capped = _apply_caps_with_redistribution(desired, caps)
                for k, v in zip(pos_keys, capped):
                    pair_post[k] = float(v)

    tot_pair = float(sum(pair_post.values()))
    if tot_pair <= 0:
        n = max(len(universe), 1)
        pair_weights = {_norm_sym(e): 1.0 / n for e, _, _ in universe}
        pair_frac = {
            (_norm_sym(e), _norm_sym(u)): 1.0 / n for e, u, _ in universe
        }
        return pair_weights, pair_frac, sig
    pair_frac = {k: v / tot_pair for k, v in pair_post.items()}

    etf_weight: dict[str, float] = {}
    for (e, _u), v in pair_post.items():
        etf_weight[e] = etf_weight.get(e, 0.0) + float(v)
    tot_etf = float(sum(etf_weight.values()))
    pair_weights = (
        {e: v / tot_etf for e, v in etf_weight.items()}
        if tot_etf > 0
        else {e: 0.0 for e in etf_weight}
    )

    diag = sig.copy()
    diag["pair_key"] = list(zip(diag["ETF"], diag["Underlying"]))
    diag["w_pre_pair"] = diag["pair_key"].map(
        lambda k: pair_pre.get((k[0], k[1]), 0.0)
    )
    diag["w_post_pair"] = diag["pair_key"].map(
        lambda k: pair_post.get((k[0], k[1]), 0.0)
    )
    diag["mult_underlying"] = diag["Underlying"].map(
        lambda u: float(mult.get(u, 1.0))
    )
    diag = diag.drop(columns=["pair_key"])
    return pair_weights, pair_frac, diag


# ---------------------------------------------------------------------------
# 6) Cap-with-redistribution (re-implemented locally to keep this module
#    importable without generate_trade_plan; mirrors the production helper).
# ---------------------------------------------------------------------------


def _apply_caps_with_redistribution(
    desired: np.ndarray,
    caps: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Iteratively cap ``desired`` at per-row ``caps``, redistributing overflow
    to rows with headroom proportional to original ``desired`` shares.

    Mirrors ``generate_trade_plan._apply_notional_caps_with_redistribution``
    behavior on a positive numeric array.
    """
    d = np.maximum(np.asarray(desired, dtype=float), 0.0)
    c = np.asarray(caps, dtype=float)
    if d.size == 0:
        return d
    target_total = float(d.sum())
    if target_total <= 0:
        return d
    base = d / target_total
    out = d.copy()
    for _ in range(int(max_iter)):
        over = out > c
        if not np.any(over):
            break
        overflow = float(np.sum(np.maximum(out - c, 0.0)))
        out = np.minimum(out, c)
        if overflow <= tol:
            break
        head = c - out
        eligible = head > tol
        if not np.any(eligible):
            break
        share = base.copy()
        share[~eligible] = 0.0
        s = float(share.sum())
        if s <= 0:
            share = eligible.astype(float)
            s = float(share.sum())
            if s <= 0:
                break
        share = share / s
        add = overflow * share
        add = np.minimum(add, head)
        out = out + add
    return out
