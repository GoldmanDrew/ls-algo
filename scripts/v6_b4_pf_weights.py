"""
v6 Option-2 portfolio-style **internal weights** for inverse_decay_bucket4 (B4) sleeves.

Score stack (production, 2026-07-10+):
  expected decay / borrow aversion (quadratic + linear + optional uncertainty)
  → weight bounds → covariance-concentration tilt → optional vol-ETP haircut.

Crash risk is **not** sized here. The unconditional tail penalty
(``dd_risk_lambda`` / ``risk_denom_coeff``) was removed; per-name crash
exposure is capped by ``scripts/b4_crash_budget.py`` after the opt2 solve
(see ``bucket4_weekly_opt2.crash_budget`` in strategy_config.yml).

Returns a dict ``{(ETF, Underlying): weight}`` summing to 1 over live B4 pairs.
High borrow is handled continuously (linear score ramp ``borrow_ramp_lo`` →
``borrow_ramp_hi``, posterior-borrow based). Pass the result to
``mirror_generate_trade_plan_sizing(..., b4_weight_override_by_pair=...)``
so book-wide ``apply_gross_sizing_book_caps`` and ``apply_covariance_balance``
still run in the mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class V6PfParams:
    """Defaults match the production B4 opt2 weight engine (decay/borrow + cov)."""

    decay_borrow_quad: float = 18.0
    #: Extra downweight beyond ``decay / (1 + decay_borrow_quad * borrow^2)``: divide by ``(1 + λ * borrow)``.
    #: ``0.0`` matches legacy notebook behavior (quadratic-only borrow term).
    borrow_linear_aversion: float = 0.0
    #: Continuous high-borrow downweight: score × 1.0 below ``borrow_ramp_lo``,
    #: linear to 0.0 at ``borrow_ramp_hi``. A pair whose ramp reaches 0 is
    #: dropped from the solve (weight already approached zero); the caller's
    #: crash-budget clamp still caps any fallback path that sizes it.
    borrow_ramp_lo: float = 0.80
    borrow_ramp_hi: float = 1.20
    min_expected_decay_annual: float = 0.01
    min_pairs: int = 5
    decay_exponent: float = 1.0
    cov_lookback_days: int = 1260
    cov_min_obs: int = 30
    cov_preinception_fallback: bool = True
    cov_shrink: float = 0.35
    cov_penalty: float = 0.85
    min_weight: float = 0.005
    max_weight: float = 0.35
    #: Flat haircut on the FINAL weight of volatility-ETP pairs (VIX complex):
    #: w_vol *= (1 - penalty), then renormalize so the freed weight flows to the
    #: other pairs. 0.0 = off (vol ETPs live in their own B5 sleeve).
    vol_etp_weight_penalty: float = 0.0
    #: ``spot`` uses pair-cache ``borrow_a_annual``; ``posterior`` prefers
    #: ``borrow_posterior_annual`` from the screener when present.
    borrow_aversion_source: str = "spot"
    #: Downweight by ``1 / (1 + penalty * borrow_posterior_var_annual)`` when > 0.
    borrow_uncertainty_penalty: float = 0.0

    @classmethod
    def from_opt2_dict(cls, opt2: dict, *, min_pairs: int | None = None) -> "V6PfParams":
        """Build params from ``bucket4_weekly_opt2`` config; ignores unknown keys.

        Unknown / retired keys (e.g. ``dd_risk_lambda`` / ``risk_denom_*`` /
        ``tail_as_of`` / ``exclude_if_borrow_annual_gt``) are silently dropped
        so stale YAML does not break the engine.
        """
        fields = cls.__dataclass_fields__
        kwargs = {k: v for k, v in dict(opt2 or {}).items() if k in fields}
        if min_pairs is not None:
            kwargs["min_pairs"] = int(min_pairs)
        return cls(**kwargs)


def _nonneg_borrow_annual(x: float) -> float:
    v = float(x)
    if not np.isfinite(v) or v < 0.0:
        return 0.0
    return float(v)


def load_borrow_posterior_by_pair(
    screened_csv: str,
    *,
    norm_sym: Callable[[str], str],
) -> dict[tuple[str, str], dict[str, float]]:
    """Per-pair posterior borrow mean/var from the screener CSV."""
    p = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in p.columns}
    ec, uc = cols.get("etf"), cols.get("underlying")
    if ec is None or uc is None:
        return {}
    ann_col = cols.get("borrow_posterior_annual")
    var_col = cols.get("borrow_posterior_var_annual")
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in p.iterrows():
        key = (norm_sym(str(row[ec])), norm_sym(str(row[uc])))
        annual = (
            float(pd.to_numeric(row.get(ann_col), errors="coerce"))
            if ann_col is not None
            else float("nan")
        )
        var = (
            float(pd.to_numeric(row.get(var_col), errors="coerce"))
            if var_col is not None
            else float("nan")
        )
        if np.isfinite(annual):
            out[key] = {
                "annual": _nonneg_borrow_annual(annual),
                "var": float(var) if np.isfinite(var) and var >= 0 else 0.0,
            }
    return out


def load_net_decay_by_pair(
    screened_csv: str,
    *,
    norm_sym: Callable[[str], str],
) -> tuple[dict[tuple[str, str], float], str]:
    """Load expected-decay map; column preference matches the v6 portfolio cell."""
    p = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in p.columns}
    ec, uc = cols.get("etf"), cols.get("underlying")
    if ec is None or uc is None:
        raise ValueError("Screener needs ETF and Underlying columns for decay weighting.")
    decay_candidates = [
        "net_edge_p50_annual",
        "net_edge_annual",
        "net_decay_annual",
        "bucket4_net_edge_annual",
    ]
    decay_col = None
    for name in decay_candidates:
        if name in cols:
            decay_col = cols[name]
            break
    if decay_col is None:
        raise ValueError(f"No decay column; tried {decay_candidates}")
    out: dict[tuple[str, str], float] = {}
    for _, r in p.iterrows():
        e = norm_sym(str(r[ec]))
        u = norm_sym(str(r[uc]))
        val = float(pd.to_numeric(r.get(decay_col), errors="coerce") or 0.0)
        if e and u:
            out[(e, u)] = max(out.get((e, u), 0.0), val)
    return out, str(decay_col)


def _first_date_min_active_pairs(pair_indices: list[pd.DatetimeIndex], min_n: int) -> pd.Timestamp:
    """Earliest date with at least ``min_n`` pairs that have a price row (cov window)."""
    counts: dict[pd.Timestamp, int] = {}
    for ix in pair_indices:
        for d in ix:
            td = pd.Timestamp(d)
            counts[td] = counts.get(td, 0) + 1
    for d in sorted(counts.keys()):
        if counts[d] >= min_n:
            return d
    raise RuntimeError(f"No calendar date has at least {min_n} pairs with a price row.")


def _v6_pf_naive_dti(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ix = pd.DatetimeIndex(ix)
    if ix.tz is not None:
        ix = ix.tz_convert("UTC").tz_localize(None)
    return ix


# Fallback VIX-complex universe; the live set is daily_screener.VOLATILITY_ETP_SYMBOLS.
_VOL_ETP_SYMBOLS_FALLBACK = frozenset(
    {"UVIX", "SVIX", "UVXY", "SVXY", "VXX", "VIXY", "VIXM", "VIX", "VIX1D", "VIX3M"}
)


def _vol_etp_symbols() -> frozenset[str]:
    try:
        from daily_screener import VOLATILITY_ETP_SYMBOLS

        return frozenset(str(s).strip().upper() for s in VOLATILITY_ETP_SYMBOLS)
    except ImportError:
        return _VOL_ETP_SYMBOLS_FALLBACK


def is_vol_etp_pair(etf: str, underlying: str, *, symbols: frozenset[str] | None = None) -> bool:
    syms = symbols if symbols is not None else _vol_etp_symbols()
    e = str(etf).strip().upper().replace(".", "-")
    u = str(underlying).strip().upper().replace(".", "-")
    return e in syms or u in syms


def apply_vol_etp_weight_penalty(
    w: pd.Series,
    is_vol: pd.Series,
    *,
    penalty: float,
) -> pd.Series:
    """Haircut vol-ETP weights by ``penalty`` and renormalize to sum to 1."""
    pen = float(np.clip(penalty, 0.0, 1.0))
    if pen <= 0.0 or not bool(is_vol.any()):
        return w
    z = w.astype(float).copy()
    z.loc[is_vol] = z.loc[is_vol] * (1.0 - pen)
    s = float(z.sum())
    return z / s if s > 0 else w


def _apply_weight_bounds(w: pd.Series, *, lo: float, hi: float) -> pd.Series:
    z = w.astype(float).copy()
    for _ in range(8):
        z = z.clip(lower=lo, upper=hi)
        sz = float(z.sum())
        if sz <= 0:
            break
        z = z / sz
    return z


def compute_v6_b4_pf_weight_dict(
    *,
    pair_cache: dict[tuple[str, str], dict[str, Any]],
    v6_opt2_h_daily_map: dict[str, pd.Series],
    screened_csv: str,
    closes_broad: pd.DataFrame | None,
    norm_sym: Callable[[str], str],
    get_ibkr_borrow_map: Callable[[list[str]], dict[str, float]],
    opt2_h_base: float,
    params: V6PfParams | None = None,
    use_ibkr_uvix_borrow: bool = True,
) -> tuple[dict[tuple[str, str], float], pd.DataFrame, dict[str, Any]]:
    """
    Return ``(weights_by_pair, diagnostics_df, meta)`` where weights sum to 1 over B4 pairs used.

    Score = decay/borrow only. Crash risk is applied later by ``b4_crash_budget``.
    """
    p = params or V6PfParams()
    decay_map, decay_src = load_net_decay_by_pair(screened_csv, norm_sym=norm_sym)
    borrow_src = str(p.borrow_aversion_source or "spot").strip().lower()
    posterior_borrow = (
        load_borrow_posterior_by_pair(screened_csv, norm_sym=norm_sym)
        if borrow_src == "posterior"
        else {}
    )

    uvix_borrow_annual_base: float | None = None
    if use_ibkr_uvix_borrow:
        ibkr_uvix = get_ibkr_borrow_map(["UVIX"]).get("UVIX")
        if ibkr_uvix is not None and np.isfinite(ibkr_uvix) and float(ibkr_uvix) > 0:
            uvix_borrow_annual_base = _nonneg_borrow_annual(float(ibkr_uvix))

    def _etf_borrow_annual_actual(etf_sym: str, kw0: dict) -> float:
        if use_ibkr_uvix_borrow and norm_sym(str(etf_sym)) == "UVIX" and uvix_borrow_annual_base is not None:
            return float(uvix_borrow_annual_base)
        return _nonneg_borrow_annual(float(kw0.get("borrow_a_annual", 0.0)))

    pairs_candidate: list[tuple[str, str]] = []
    for key, c in pair_cache.items():
        if "skip_reason" in c:
            continue
        etf_sym, und_sym = key
        if und_sym not in v6_opt2_h_daily_map:
            continue
        pairs_candidate.append((etf_sym, und_sym))

    def _pair_borrow(etf_sym: str, und_sym: str) -> tuple[float, float]:
        """(borrow_annual, posterior_var) from the SAME estimate the score uses.

        One estimate per quantity: the ramp and the aversion terms must not see
        different borrow numbers for the same name (spot vs posterior routing is
        how CORD escaped sizing entirely).
        """
        kw0 = pair_cache[(etf_sym, und_sym)]["kw"]
        post = posterior_borrow.get((norm_sym(etf_sym), norm_sym(und_sym))) or {}
        if borrow_src == "posterior" and post.get("annual") is not None:
            return float(post["annual"]), float(post.get("var", 0.0) or 0.0)
        return _etf_borrow_annual_actual(etf_sym, kw0), 0.0

    def _borrow_ramp_mult(b: float) -> float:
        """Continuous high-borrow downweight: 1.0 below ``borrow_ramp_lo``,
        linear to 0.0 at ``borrow_ramp_hi``. No cliffs: a 2-point borrow move
        can never flip a name between sizing regimes."""
        lo, hi = float(p.borrow_ramp_lo), float(p.borrow_ramp_hi)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return 1.0
        return float(np.clip((hi - float(b)) / (hi - lo), 0.0, 1.0))

    excluded_borrow: list[dict[str, Any]] = []
    pairs_live: list[tuple[str, str]] = []
    ramp_by_pair: dict[tuple[str, str], float] = {}
    for etf_sym, und_sym in pairs_candidate:
        b, _ = _pair_borrow(etf_sym, und_sym)
        m = _borrow_ramp_mult(b)
        if m <= 0.0:
            # Weight already ramped to ~0 approaching the boundary, so dropping
            # the pair is continuous. Fail-closed: the caller's crash-budget
            # clamp still binds on any fallback path that sizes this name.
            excluded_borrow.append({
                "pair": f"{etf_sym}/{und_sym}",
                "borrow_etf_annual": b,
                "reason": f"borrow ramp reached 0 (>= {p.borrow_ramp_hi:.0%})",
            })
            continue
        pairs_live.append((etf_sym, und_sym))
        ramp_by_pair[(etf_sym, und_sym)] = m

    if len(pairs_live) < p.min_pairs:
        # Do not abort the whole trade plan on thin B4 days (common on older /
        # incomplete screened archives). Callers treat empty weights as "no B4".
        return (
            {},
            pd.DataFrame(),
            {
                "excluded_high_borrow": excluded_borrow,
                "n_pairs_live": int(len(pairs_live)),
                "min_pairs": int(p.min_pairs),
                "skipped_thin_book": True,
                "reason": (
                    f"Need at least {p.min_pairs} tradable pairs after borrow ramp; "
                    f"got {len(pairs_live)} (excluded {len(excluded_borrow)} with "
                    f"borrow >= {p.borrow_ramp_hi:.0%})."
                ),
            },
        )

    ix_list = [pair_cache[k]["prices"].index for k in pairs_live]
    start_sim = _first_date_min_active_pairs(ix_list, p.min_pairs)

    rows_w: list[dict[str, Any]] = []
    for etf_sym, und_sym in pairs_live:
        pair_lbl = f"{etf_sym}/{und_sym}"
        decay_u = float(decay_map.get((norm_sym(etf_sym), norm_sym(und_sym)), p.min_expected_decay_annual))
        decay_eff = max(p.min_expected_decay_annual, decay_u)
        borrow_a, borrow_var = _pair_borrow(etf_sym, und_sym)
        quad = 1.0 + float(p.decay_borrow_quad) * (borrow_a**2)
        lin = 1.0 + float(p.borrow_linear_aversion) * borrow_a
        if lin <= 0.0:
            lin = 1e-18
        base_score = (decay_eff ** float(p.decay_exponent)) / quad / lin
        unc_pen = float(p.borrow_uncertainty_penalty)
        if unc_pen > 0.0 and borrow_var > 0.0:
            base_score = base_score / (1.0 + unc_pen * borrow_var)
        ramp_m = float(ramp_by_pair.get((etf_sym, und_sym), 1.0))
        base_score = base_score * ramp_m
        rows_w.append(
            {
                "pair": pair_lbl,
                "etf": etf_sym,
                "underlying": und_sym,
                "expected_decay_annual": decay_u,
                "decay_eff": decay_eff,
                "borrow_etf_annual": borrow_a,
                "borrow_ramp_mult": ramp_m,
                "base_score": base_score,
            }
        )

    wdf = pd.DataFrame(rows_w)
    if wdf.empty:
        raise RuntimeError("No live pairs left for sizing table.")

    # Score = decay/borrow only (old unconditional tail penalty removed).
    score_tot = float(wdf["base_score"].sum())
    if score_tot <= 0:
        raise RuntimeError("Non-positive sizing score total.")
    wdf["weight_raw"] = wdf["base_score"] / score_tot
    w_signal = _apply_weight_bounds(wdf["weight_raw"].astype(float), lo=p.min_weight, hi=p.max_weight)
    wdf["weight_signal"] = w_signal

    sim_cut = pd.Timestamp(start_sim)
    if getattr(sim_cut, "tzinfo", None) is not None:
        sim_cut = sim_cut.tz_convert("UTC").tz_localize(None)

    min_px_bars = int(p.cov_min_obs) + 1
    ret_proxy_map: dict[str, pd.Series] = {}
    for etf_sym, und_sym in pairs_live:
        pair_lbl = f"{etf_sym}/{und_sym}"
        c = pair_cache[(etf_sym, und_sym)]
        px_full = c["prices"]
        if px_full is None or len(px_full) == 0:
            continue
        px_na = px_full.copy()
        px_na.index = _v6_pf_naive_dti(px_full.index)
        px_pre = px_na.loc[px_na.index < sim_cut]

        rp_etf = None
        n_etf = -1
        if len(px_pre) >= min_px_bars:
            a_r = pd.to_numeric(px_pre["a_px"], errors="coerce").pct_change()
            b_r = pd.to_numeric(px_pre["b_px"], errors="coerce").pct_change()
            if und_sym in v6_opt2_h_daily_map:
                hm = v6_opt2_h_daily_map[und_sym].copy()
                hm.index = _v6_pf_naive_dti(hm.index)
                h_d = hm.reindex(px_pre.index).ffill().fillna(float(opt2_h_base))
            else:
                h_d = pd.Series(float(opt2_h_base), index=px_pre.index)
            rp_etf = (-(a_r + h_d * b_r)).replace([np.inf, -np.inf], np.nan)
            n_etf = int(rp_etf.dropna().shape[0])

        rp_under = None
        n_under = -1
        cb = closes_broad if closes_broad is not None else pd.DataFrame()
        if bool(p.cov_preinception_fallback) and und_sym in getattr(cb, "columns", []):
            u_ser = cb[und_sym]
            u_na = u_ser.copy()
            u_na.index = _v6_pf_naive_dti(u_ser.index)
            u_hist = u_na.loc[u_na.index < sim_cut]
            if len(u_hist.dropna()) >= min_px_bars:
                u_r = pd.to_numeric(u_hist, errors="coerce").pct_change()
                rp_under = (-(1.0 + float(opt2_h_base)) * u_r).replace([np.inf, -np.inf], np.nan)
                n_under = int(rp_under.dropna().shape[0])

        rp_best = None
        if n_etf >= int(p.cov_min_obs) and n_under >= int(p.cov_min_obs):
            rp_best = rp_etf if n_etf >= n_under else rp_under
        elif n_etf >= int(p.cov_min_obs):
            rp_best = rp_etf
        elif n_under >= int(p.cov_min_obs):
            rp_best = rp_under

        if rp_best is None:
            continue
        ret_proxy_map[pair_lbl] = rp_best

    cov_pair_cols = wdf["pair"].astype(str).tolist()
    ret_proxy_df = pd.DataFrame(ret_proxy_map).sort_index()
    for c in cov_pair_cols:
        if c not in ret_proxy_df.columns:
            ret_proxy_df[c] = np.nan
    ret_proxy_df = ret_proxy_df.reindex(columns=cov_pair_cols)
    if len(ret_proxy_df) == 0:
        ret_proxy_df.index = pd.DatetimeIndex([], tz=None)
    elif not isinstance(ret_proxy_df.index, pd.DatetimeIndex):
        ret_proxy_df.index = pd.to_datetime(ret_proxy_df.index, errors="coerce")
        ret_proxy_df = ret_proxy_df.loc[ret_proxy_df.index.notna()].sort_index()
    elif ret_proxy_df.index.tz is not None:
        ret_proxy_df = ret_proxy_df.copy()
        ret_proxy_df.index = _v6_pf_naive_dti(ret_proxy_df.index)
    ret_proxy_df = ret_proxy_df.loc[ret_proxy_df.index < sim_cut]

    ret_cov_window = ret_proxy_df.tail(int(p.cov_lookback_days))
    pair_counts = ret_cov_window.count()
    n_usable = int((pair_counts >= int(p.cov_min_obs)).sum())
    if n_usable >= 2:
        ret_cov = ret_cov_window.cov(min_periods=int(p.cov_min_obs)).astype(float)
        diag = pd.Series(np.diag(ret_cov.values), index=ret_cov.index).replace([np.inf, -np.inf], np.nan)
        diag_fill = float(diag.median(skipna=True)) if diag.notna().any() else 0.0
        diag_filled = diag.fillna(diag_fill).astype(float)
        cov_vals = ret_cov.values.copy()
        np.fill_diagonal(cov_vals, diag_filled.values)
        cov_vals = np.where(np.isfinite(cov_vals), cov_vals, 0.0)
        d = np.diag(diag_filled.values)
        ret_cov_shrunk = pd.DataFrame(
            (1.0 - float(p.cov_shrink)) * cov_vals + float(p.cov_shrink) * d,
            index=ret_cov.index,
            columns=ret_cov.columns,
        )
        w_align = (
            pd.Series(w_signal.astype(float).values, index=wdf["pair"].astype(str))
            .reindex(ret_cov.index)
            .fillna(0.0)
        )
        cov_mrc = ret_cov_shrunk @ w_align
        cov_contrib = (w_align * cov_mrc).clip(lower=0.0)
        pc = cov_contrib[cov_contrib > 0]
        cov_scale = float(pc.median()) if len(pc) > 0 else 1.0
        cov_scale = max(cov_scale, 1e-12)
        wdf["cov_contrib_norm"] = wdf["pair"].map((cov_contrib / cov_scale)).fillna(0.0).astype(float)
        wdf["cov_penalty"] = 1.0 / (1.0 + float(p.cov_penalty) * wdf["cov_contrib_norm"])
        wdf["cov_obs"] = wdf["pair"].map(pair_counts).fillna(0).astype(int)
    else:
        wdf["cov_contrib_norm"] = 0.0
        wdf["cov_penalty"] = 1.0
        wdf["cov_obs"] = 0

    w_cov_raw = (w_signal * wdf["cov_penalty"].astype(float)).astype(float)
    if float(w_cov_raw.sum()) > 0:
        w_cov_raw = w_cov_raw / float(w_cov_raw.sum())
    w_cov = _apply_weight_bounds(w_cov_raw, lo=p.min_weight, hi=p.max_weight)

    vsyms = _vol_etp_symbols()
    wdf["is_vol_etp"] = [
        is_vol_etp_pair(e, u, symbols=vsyms)
        for e, u in zip(wdf["etf"].astype(str), wdf["underlying"].astype(str))
    ]
    if float(p.vol_etp_weight_penalty) > 0.0 and bool(wdf["is_vol_etp"].any()):
        w_cov = apply_vol_etp_weight_penalty(
            w_cov, wdf["is_vol_etp"], penalty=float(p.vol_etp_weight_penalty)
        )
        w_cov = _apply_weight_bounds(w_cov, lo=p.min_weight, hi=p.max_weight)
    wdf["new_weight"] = w_cov

    wdf = wdf.sort_values("new_weight", ascending=False).set_index("pair")
    weight_by_key: dict[tuple[str, str], float] = {}
    for ix in wdf.index:
        etf_s, und_s = str(ix).split("/", 1)
        weight_by_key[(norm_sym(etf_s), norm_sym(und_s))] = float(wdf.loc[ix, "new_weight"])

    meta = {
        "decay_src_col": decay_src,
        "start_sim": start_sim,
        "n_pairs_live": len(pairs_live),
        "excluded_high_borrow": excluded_borrow,
        "borrow_ramp": {"lo": float(p.borrow_ramp_lo), "hi": float(p.borrow_ramp_hi)},
        "uvix_borrow_ftp_annual": uvix_borrow_annual_base,
        "vol_etp_weight_penalty": float(p.vol_etp_weight_penalty),
        "n_vol_etp_pairs": int(wdf["is_vol_etp"].sum()),
        "crash_risk_sizing": "b4_crash_budget (post-solve)",
    }
    return weight_by_key, wdf.reset_index(), meta
