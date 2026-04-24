# screener_v2_fields — schema v2 uncertainty + product_class for daily_screener export
# Spec: v2.0 (block bootstrap on mean daily log-drag; stress borrow rho grid when sigma_b>0)
from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
import pandas as pd

# Match daily_screener
TRADING_DAYS = 252
_MIN_DAYS_DECAY = 40
_STRESS_BORROW_RHOS = (0.0, 0.2, 0.4)
_BOOT_N = 400
_BLOCK_LEN_DEFAULT = 10


def _extract_daily_drag(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_days: int = _MIN_DAYS_DECAY,
) -> np.ndarray | None:
    etf_tr = etf_tr[~etf_tr.index.duplicated(keep="last")]
    und_tr = und_tr[~und_tr.index.duplicated(keep="last")]
    combined = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
    if len(combined) < min_days + 1:
        return None
    if abs(float(beta)) < 0.1:
        return None
    r_etf = np.log(combined["etf"] / combined["etf"].shift(1))
    r_und = np.log(combined["und"] / combined["und"].shift(1))
    valid = r_etf.notna() & r_und.notna() & np.isfinite(r_etf) & np.isfinite(r_und)
    r_etf, r_und = r_etf[valid], r_und[valid]
    if len(r_etf) < min_days:
        return None
    daily_drag = (float(beta) * r_und - r_etf).to_numpy(dtype=float)
    return daily_drag


def _block_bootstrap_annual_gross(
    daily_drag: np.ndarray,
    *,
    n_boot: int = _BOOT_N,
    block_len: int = _BLOCK_LEN_DEFAULT,
    seed: int = 42,
) -> tuple[float, float, float, float] | None:
    """Return (p05, p50, p95, mean) of *annualized* gross from resampled mean(drag)*252."""
    x = np.asarray(daily_drag, dtype=float)
    t = int(x.size)
    if t < _MIN_DAYS_DECAY:
        return None
    b = min(block_len, max(5, t // 5))
    n_blocks = int(np.ceil(t / b))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    mean0 = float(np.mean(x)) * TRADING_DAYS
    for i in range(n_boot):
        parts = [rng.choice(x, size=b, replace=True) for _ in range(n_blocks)]
        samp = np.concatenate(parts)[:t]
        out[i] = float(np.mean(samp)) * TRADING_DAYS
    p05, p50, p95 = np.percentile(out, [5, 50, 95]).tolist()
    return float(p05), float(p50), float(p95), float(mean0)


def _ar1(series: np.ndarray) -> float | None:
    if series.size < 20:
        return None
    y = series[1:]
    x = series[:-1]
    vx = float(np.var(x, ddof=0))
    if vx < 1e-20:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / vx)


def _regime_warning(und_tr: pd.Series, min_points: int = 60) -> tuple[float | None, str]:
    und_tr = und_tr[~und_tr.index.duplicated(keep="last")].dropna()
    r = np.log(und_tr / und_tr.shift(1)).dropna()
    r = r[np.isfinite(r)].to_numpy()[-min_points:]
    if r.size < 30:
        return None, "insufficient_history"
    phi = _ar1(r)
    if phi is None:
        return None, "insufficient_history"
    a = min(21, max(5, r.size // 3))
    r_short = r[-a:]
    trend = float(np.sum(r_short))
    if abs(phi) < 0.05 and abs(trend) < 1e-3:
        return float(phi), "none"
    if phi < -0.05:
        return float(phi), "mean_reversion"
    if phi > 0.05 and abs(trend) > 0.02:
        return float(phi), "momentum_incomparable_to_vol_drag"
    return float(phi), "none"


def _product_class(lev: Any, beta: Any) -> str:
    if lev is not None and pd.notna(lev) and not _nanf(lev):
        l = float(lev)
        if abs(l - 1.0) < 0.01:
            return "income_put_spread"
        return "standard_letf"
    if beta is not None and pd.notna(beta) and not _nanf(beta) and float(beta) < 0:
        return "standard_letf"
    return "other_structured"


def _nanf(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _gross_edge_definition(
    n_obs: int, beta: Any, realized_ok: bool
) -> str:
    if n_obs < _MIN_DAYS_DECAY or not realized_ok:
        return "expected_only"
    if beta is not None and pd.notna(beta) and not _nanf(beta):
        b = float(beta)
        if 0 < b <= 1.5:
            return "realized_daily_log_drag"
    return "blended_realized_expected"


def enrich_screener_v2_fields(
    df: pd.DataFrame,
    tr_map: dict[str, pd.Series],
    *,
    min_days: int = _MIN_DAYS_DECAY,
) -> pd.DataFrame:
    """
    Add schema v2 columns (add-only). `primary_edge_annual` matches net_decay_annual
    (short-favourable: higher = better for structural short on decay).
    """
    asof = _dt.date.today().isoformat()
    n = len(df)
    (
        pclass,
        gdef,
        primary,
        gprim,
        bfor,
        bmed,
        p05,
        p50,
        p95,
        mech,
        rtrack,
        slp,
        idist,
        blen,
        breps,
        akey,
        sigb,
        srho,
        cnote,
        ctype,
        rac,
        rwarn,
        hirk,
        dnote,
    ) = (
        [""] * n,
        [""] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [np.nan] * n,
        [""] * n,
        [np.nan] * n,
        [np.nan] * n,
        [""] * n,
        [""] * n,
        [np.nan] * n,
        [""] * n,
        [False] * n,
        [""] * n,
    )

    for j, (_, row) in enumerate(df.iterrows()):
        etf = str(row.get("ETF", "")).strip()
        und = str(row.get("Underlying", "")).strip() if pd.notna(row.get("Underlying")) else ""
        beta = row.get("Beta")
        lev = row.get("Leverage") if "Leverage" in row else np.nan
        pclass[j] = _product_class(lev, beta)
        bcur = float(row["borrow_current"]) if not _nanf(row.get("borrow_current")) else 0.0
        bfor[j] = bcur

        n_obs = int(row["Beta_n_obs"]) if not _nanf(row.get("Beta_n_obs")) else 0
        realized_ok = bool(
            etf in tr_map
            and und in tr_map
            and not _nanf(beta)
            and abs(float(beta)) >= 0.1
        )
        gdef[j] = _gross_edge_definition(
            n_obs, beta, realized_ok and n_obs >= min_days
        )

        g_real = row.get("gross_decay_annual")
        g_exp = row.get("expected_gross_decay_annual")
        g_blend = row.get("blended_gross_decay")
        if gdef[j] == "realized_daily_log_drag" and not _nanf(g_real):
            gprim[j] = float(g_real)
        elif gdef[j] == "expected_only" and not _nanf(g_exp):
            gprim[j] = float(g_exp)
        elif not _nanf(g_blend):
            gprim[j] = float(g_blend)
        elif not _nanf(g_real):
            gprim[j] = float(g_real)
        else:
            gprim[j] = np.nan

        nd_annual = row.get("net_decay_annual")
        if pd.notna(nd_annual) and not _nanf(nd_annual):
            primary[j] = float(nd_annual)
        else:
            primary[j] = (
                float(gprim[j]) - bcur
                if not _nanf(gprim[j]) else np.nan
            )
        bmed[j] = np.nan

        mech[j] = float(g_exp) if not _nanf(g_exp) else np.nan
        if not _nanf(g_real) and not _nanf(g_exp):
            rtrack[j] = float(g_real) - float(g_exp)
        else:
            rtrack[j] = np.nan

        idist[j] = np.nan
        slp[j] = np.nan

        drag = None
        if etf in tr_map and und in tr_map and not _nanf(beta) and abs(float(beta)) >= 0.1:
            drag = _extract_daily_drag(tr_map[etf], tr_map[und], float(beta), min_days=min_days)

        boot = _block_bootstrap_annual_gross(drag) if drag is not None else None
        g_real = row.get("gross_decay_annual")
        if boot is not None:
            g05, g50, g95, _ = boot
            p05[j] = g05 - bcur
            p50[j] = g50 - bcur
            p95[j] = g95 - bcur
            blen[j] = float(_BLOCK_LEN_DEFAULT)
            breps[j] = float(_BOOT_N)
            akey[j] = "trading_days_252"
            cnote[j] = "block_bootstrap_daily_drag; borrow_point_in_time"
            ctype[j] = "empirical_block_marginal"
            sigb[j] = 0.0
            srho[j] = 0.0
            p05[j] = p05_stress(
                p05[j], 0.0, _STRESS_BORROW_RHOS
            )
        else:
            p05[j] = np.nan
            p50[j] = np.nan
            p95[j] = np.nan
            blen[j] = np.nan
            breps[j] = np.nan
            akey[j] = "trading_days_252" if g_real is not None and pd.notna(g_real) else ""
            cnote[j] = "insufficient_history_for_bootstrap" if boot is None else ""
            ctype[j] = "none"
            sigb[j] = 0.0
            srho[j] = np.nan

        if pclass[j] == "income_put_spread":
            dnote[j] = "income_dist_missing"
        else:
            dnote[j] = "ok"
        if und in tr_map:
            rac[j], rwarn[j] = _regime_warning(tr_map[und])
        else:
            rac[j] = np.nan
            rwarn[j] = "insufficient_history"
        if str(rwarn[j]) != "none" and dnote[j] == "ok":
            dnote[j] = "incomplete_regime"

    out = df.copy()
    out["asof_date"] = asof
    out["product_class"] = pclass
    out["gross_edge_definition"] = gdef
    out["primary_edge_annual"] = primary
    out["gross_for_primary_annual"] = gprim
    out["borrow_for_net_annual"] = bfor
    out["borrow_median_60d"] = bmed
    out["net_edge_p05_annual"] = p05
    out["net_edge_p50_annual"] = p50
    out["net_edge_p95_annual"] = p95
    out["mechanical_decay_annual"] = mech
    out["realized_tracking_component_annual"] = rtrack
    out["slippage_proxy_annual"] = slp
    out["income_distributions_annual"] = idist
    out["block_len"] = blen
    out["B_reps"] = breps
    out["annualization_key"] = akey
    out["hac_lag"] = np.nan
    out["sigma_b_annual"] = sigb
    out["stress_borrow_rho"] = srho
    out["copula_note"] = cnote
    out["copula_type"] = ctype
    out["borrow_dispersion_type"] = "deferred"
    out["regime_autocorr_und_21d_proxy"] = rac
    out["regime_warning"] = rwarn
    out["high_intraday_risk"] = hirk
    out["decomposition_note"] = dnote
    out["schema_v"] = 2
    out["edge_sign_convention"] = "short_favorable_positive"
    return out


def p05_stress(
    base_net_p05: float,
    sigma_b: float,
    rhos: tuple[float, ...] = _STRESS_BORROW_RHOS,
) -> float:
    """Conservative 5% net: min over rho of (p05_net − rho * sigma_b) when sigma_b > 0."""
    if not sigma_b or float(sigma_b) <= 0.0 or not np.isfinite(sigma_b):
        return float(base_net_p05)
    return float(min(float(base_net_p05) - float(rho) * float(sigma_b) for rho in rhos))
