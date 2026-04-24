# screener_v2_fields — schema v2 uncertainty + product_class for daily_screener export
# Spec: v2.1 (block bootstrap on mean daily log-drag; optional weighted borrow resample
#       from full history; stress borrow rho grid when sigma_b>0)
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
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


def _block_bootstrap_annual_gross_draws(
    daily_drag: np.ndarray,
    *,
    n_boot: int = _BOOT_N,
    block_len: int = _BLOCK_LEN_DEFAULT,
    seed: int = 42,
) -> np.ndarray | None:
    """Return length-`n_boot` array of annualized gross from block-resampled mean(drag)*252."""
    x = np.asarray(daily_drag, dtype=float)
    t = int(x.size)
    if t < _MIN_DAYS_DECAY:
        return None
    b = min(block_len, max(5, t // 5))
    n_blocks = int(np.ceil(t / b))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        parts = [rng.choice(x, size=b, replace=True) for _ in range(n_blocks)]
        samp = np.concatenate(parts)[:t]
        out[i] = float(np.mean(samp)) * TRADING_DAYS
    return out


def _block_bootstrap_annual_gross(
    daily_drag: np.ndarray,
    *,
    n_boot: int = _BOOT_N,
    block_len: int = _BLOCK_LEN_DEFAULT,
    seed: int = 42,
) -> tuple[float, float, float, float] | None:
    """Return (p05, p50, p95, mean) of *annualized* gross from resampled mean(drag)*252."""
    x = np.asarray(daily_drag, dtype=float)
    draws = _block_bootstrap_annual_gross_draws(
        x, n_boot=n_boot, block_len=block_len, seed=seed
    )
    if draws is None:
        return None
    mean0 = float(np.mean(x)) * TRADING_DAYS
    p05, p50, p95 = np.percentile(draws, [5, 50, 95]).tolist()
    return float(p05), float(p50), float(p95), float(mean0)


def load_borrow_history_json(path: str | os.PathLike[str]) -> dict[str, list]:
    """Load etf-dashboard ``borrow_history.json`` (top-level ``symbols`` map).

    Keys are normalized to upper-case stripped tickers. Values are the original
    per-symbol lists of observation dicts.
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        doc = json.load(f)
    raw = doc.get("symbols", doc)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list] = {}
    for k, v in raw.items():
        sym = str(k).strip().upper()
        if sym and isinstance(v, list):
            out[sym] = v
    return out


def _net_edge_hist_json(net_draws: np.ndarray, n_bins: int = 20) -> str | None:
    """Compact JSON for dashboard mini-histogram: ``{\"e\":[edges],\"c\":[counts]}``."""
    a = np.asarray(net_draws, dtype=float)
    if a.size < 2:
        return None
    nb = int(n_bins) if int(n_bins) > 2 else 10
    lo, hi = np.percentile(a, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.min(a)), float(np.max(a))
    if lo >= hi:
        hi = lo + 1e-8
    counts, edges = np.histogram(a, bins=nb, range=(float(lo), float(hi)))
    return json.dumps(
        {"e": [round(float(x), 6) for x in edges], "c": [int(x) for x in counts]},
        separators=(",", ":"),
    )


def _borrow_history_row_usable_for_resample(row: dict, borrow_fv: float) -> bool:
    """Exclude placeholder quotes: ~zero fee when shares_available is zero (not shortable).

    Rows without ``shares_available`` are kept (legacy history); true 0%% with positive
    shares (easy to borrow) are kept.
    """
    if abs(float(borrow_fv)) > 1e-12:
        return True
    sh = row.get("shares_available")
    if sh is None:
        return True
    try:
        si = int(float(sh))
    except (TypeError, ValueError):
        return True
    return si > 0


def _weighted_borrow_values_probs(
    history_rows: list,
    asof: _dt.date,
    halflife_days: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build (values, normalized_probs) for weighted resampling of borrow_current.

    Weight per observation: ``0.5 ** (age_calendar_days / halflife_days)`` where
    ``age`` is ``asof - observation_date`` (observations strictly after ``asof``
    are skipped). If ``halflife_days`` is non-finite or <= 0, uses uniform weights.

    Observations with ~zero ``borrow_current`` and ``shares_available`` <= 0 are
    dropped (IBKR-style placeholder when nothing is available to short).
    """
    if not history_rows or halflife_days is None:
        return None
    dates: list[_dt.date] = []
    vals: list[float] = []
    for row in history_rows:
        if not isinstance(row, dict):
            continue
        ds = row.get("date")
        if ds is None:
            continue
        try:
            if isinstance(ds, _dt.date) and not isinstance(ds, _dt.datetime):
                d = ds
            else:
                d = _dt.date.fromisoformat(str(ds)[:10])
        except (TypeError, ValueError):
            continue
        if d > asof:
            continue
        br = row.get("borrow_current")
        if br is None or (isinstance(br, float) and np.isnan(br)):
            continue
        try:
            fv = float(br)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        if not _borrow_history_row_usable_for_resample(row, fv):
            continue
        age = (asof - d).days
        if age < 0:
            continue
        dates.append(d)
        vals.append(fv)
    n = len(vals)
    if n == 0:
        return None
    v_arr = np.asarray(vals, dtype=float)
    if not np.isfinite(halflife_days) or float(halflife_days) <= 0.0:
        w = np.ones(n, dtype=float)
    else:
        ages = np.asarray([(asof - d).days for d in dates], dtype=float)
        w = np.power(0.5, ages / float(halflife_days))
    s = float(w.sum())
    if s <= 0.0 or not np.isfinite(s):
        return None
    p = (w / s).astype(float)
    return v_arr, p


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
    borrow_history_map: dict[str, list] | None = None,
    borrow_weight_halflife_days: float = 90.0,
    asof_date: _dt.date | None = None,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    """
    Add schema v2 columns (add-only). `primary_edge_annual` matches net_decay_annual
    (short-favourable: higher = better for structural short on decay).

    When ``borrow_history_map`` is provided (etf-dashboard ``borrow_history.json``),
    net-edge percentiles use independent weighted resampling of historical
    ``borrow_current`` (recent-heavy half-life in calendar days); otherwise the
    legacy point-in-time ``borrow_current`` subtraction is used.
    """
    asof_d = asof_date if asof_date is not None else _dt.date.today()
    asof = asof_d.isoformat()
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
        bhalf,
        bnpts,
        bmode,
        p25,
        p75,
        nehist,
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
        [np.nan] * n,
        [np.nan] * n,
        [""] * n,
        [np.nan] * n,
        [np.nan] * n,
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

        gross_draws = (
            _block_bootstrap_annual_gross_draws(
                drag,
                n_boot=_BOOT_N,
                block_len=_BLOCK_LEN_DEFAULT,
                seed=bootstrap_seed,
            )
            if drag is not None
            else None
        )
        g_real = row.get("gross_decay_annual")
        if gross_draws is not None:
            rng_borrow = np.random.default_rng(int(bootstrap_seed) + 1_000_003)
            etf_key = etf.strip().upper()
            hist = None
            if borrow_history_map:
                hist = borrow_history_map.get(etf_key)
                if hist is None and etf != etf_key:
                    hist = borrow_history_map.get(etf.strip())
            wb = (
                _weighted_borrow_values_probs(
                    hist, asof_d, float(borrow_weight_halflife_days)
                )
                if (hist and borrow_history_map)
                else None
            )
            if wb is not None:
                vals, probs = wb
                idx = rng_borrow.choice(vals.size, size=gross_draws.size, p=probs, replace=True)
                b_draws = vals[idx]
                net_draws = gross_draws - b_draws
                cnote[j] = (
                    "block_bootstrap_daily_drag;"
                    "weighted_borrow_resample_full_history;"
                    f"halflife_cal_days={float(borrow_weight_halflife_days):g}"
                )
                ctype[j] = "empirical_block_marginal_x_weighted_borrow_marginal"
                bhalf[j] = float(borrow_weight_halflife_days)
                bnpts[j] = float(vals.size)
                bmode[j] = "weighted_empirical"
            else:
                net_draws = gross_draws - bcur
                cnote[j] = "block_bootstrap_daily_drag; borrow_point_in_time"
                ctype[j] = "empirical_block_marginal"
                bhalf[j] = np.nan
                bnpts[j] = np.nan
                bmode[j] = "point_in_time"
            blen[j] = float(_BLOCK_LEN_DEFAULT)
            breps[j] = float(_BOOT_N)
            akey[j] = "trading_days_252"
            sigb[j] = 0.0
            srho[j] = 0.0
            p05[j], p25[j], p50[j], p75[j], p95[j] = np.percentile(
                net_draws, [5, 25, 50, 75, 95]
            ).tolist()
            p05[j] = p05_stress(float(p05[j]), 0.0, _STRESS_BORROW_RHOS)
            hj = _net_edge_hist_json(net_draws)
            nehist[j] = hj if hj is not None else ""
        else:
            p05[j] = np.nan
            p25[j] = np.nan
            p50[j] = np.nan
            p75[j] = np.nan
            p95[j] = np.nan
            blen[j] = np.nan
            breps[j] = np.nan
            akey[j] = "trading_days_252" if g_real is not None and pd.notna(g_real) else ""
            cnote[j] = "insufficient_history_for_bootstrap" if gross_draws is None else ""
            ctype[j] = "none"
            sigb[j] = 0.0
            srho[j] = np.nan
            bhalf[j] = np.nan
            bnpts[j] = np.nan
            bmode[j] = ""
            nehist[j] = ""

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
    out["net_edge_p25_annual"] = p25
    out["net_edge_p50_annual"] = p50
    out["net_edge_p75_annual"] = p75
    out["net_edge_p95_annual"] = p95
    out["net_edge_hist_json"] = nehist
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
    out["borrow_weight_halflife_days"] = bhalf
    out["borrow_history_points_used"] = bnpts
    out["borrow_resample_mode"] = bmode
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
