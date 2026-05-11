"""
v6 Option-2 portfolio-style **internal weights** for inverse_decay_bucket4 (B4) sleeves.

Expected-decay column order matches ``Bucket_4_Backtest.ipynb`` v6 portfolio cell:
``net_edge_p50_annual``, then ``net_edge_annual``, ``net_decay_annual``, then ``bucket4_net_edge_annual``.
This is the decay + borrow aversion (quadratic + optional linear) + tail-risk +
covariance-concentration stack from the ``Bucket_4_Backtest.ipynb`` portfolio cell, factored for
reuse from ``Buckets1-4Backtest.ipynb``.

Returns a dict ``{(ETF, Underlying): weight}`` summing to 1 over live B4 pairs (after the high-borrow
filter). Pass the result to ``mirror_generate_trade_plan_sizing(..., b4_weight_override_by_pair=...)``
so book-wide ``apply_gross_sizing_book_caps`` and ``apply_covariance_balance`` still run in the mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class V6PfParams:
    """Defaults match the v6 portfolio cell in ``Bucket_4_Backtest.ipynb``."""

    decay_borrow_quad: float = 18.0
    #: Extra downweight beyond ``decay / (1 + decay_borrow_quad * borrow^2)``: divide by ``(1 + λ * borrow)``.
    #: ``0.0`` matches legacy notebook behavior (quadratic-only borrow term).
    borrow_linear_aversion: float = 0.0
    exclude_if_borrow_annual_gt: float = 0.90
    min_expected_decay_annual: float = 0.01
    min_pairs: int = 5
    tail_horizon_days: int = 20
    tail_lookback_days: int = 1260
    tail_full_history_blend: float = 0.70
    downside_vol_lookback: int = 126
    downside_vol_blend: float = 0.45
    dd_risk_lambda: float = 2.5
    risk_penalty_floor: float = 0.25
    risk_penalty_cap: float = 1.00
    use_beta_risk_scale: bool = True
    decay_exponent: float = 1.0
    risk_denom_coeff: float = 3.0
    risk_denom_power: float = 1.5
    tail_risk_symbol_overrides: dict[str, str] | None = None
    cov_lookback_days: int = 1260
    cov_min_obs: int = 30
    cov_preinception_fallback: bool = True
    cov_shrink: float = 0.35
    cov_penalty: float = 0.85
    min_weight: float = 0.005
    max_weight: float = 0.35
    collinear_rho: float = 0.80
    collinear_damp: float = 0.60

    def __post_init__(self) -> None:
        if self.tail_risk_symbol_overrides is None:
            self.tail_risk_symbol_overrides = {"UVIX/SVIX": "UVIX"}


def _nonneg_borrow_annual(x: float) -> float:
    v = float(x)
    if not np.isfinite(v) or v < 0.0:
        return 0.0
    return float(v)


def load_net_decay_by_pair(
    screened_csv: str,
    *,
    norm_sym: Callable[[str], str],
) -> tuple[dict[tuple[str, str], float], str]:
    p = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in p.columns}
    ec, uc = cols.get("etf"), cols.get("underlying")
    if ec is None or uc is None:
        raise ValueError("Screener needs ETF and Underlying columns for decay weighting.")
    decay_candidates = ["net_edge_p50_annual", "net_edge_annual", "net_decay_annual", "bucket4_net_edge_annual"]
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


def _tail_risk_raw(
    und_sym: str,
    as_of: pd.Timestamp,
    *,
    norm_sym: Callable[[str], str],
    pair_cache: dict[tuple[str, str], dict[str, Any]],
    closes_broad: pd.DataFrame | None,
    params: V6PfParams,
) -> float:
    u = norm_sym(str(und_sym))
    px = None
    if closes_broad is not None and u in closes_broad.columns:
        px = pd.to_numeric(closes_broad[u], errors="coerce").dropna()
    if px is None or px.empty:
        for (_, und_k), c in pair_cache.items():
            if norm_sym(str(und_k)) != u or "skip_reason" in c:
                continue
            s = pd.to_numeric(c["prices"].get("b_px"), errors="coerce").dropna()
            if not s.empty:
                px = s
                break
    if px is None or px.empty:
        return 0.0

    s = px.loc[px.index <= pd.Timestamp(as_of)].dropna().astype(float)
    if len(s) < max(40, params.tail_horizon_days + 5):
        return 0.0

    hret = s.pct_change(params.tail_horizon_days).dropna()
    if hret.empty:
        return 0.0

    win_h = hret.iloc[-int(params.tail_lookback_days) :]
    tail_loss_trailing = max(0.0, -float(win_h.min())) if len(win_h) > 0 else 0.0
    tail_loss_full = max(0.0, -float(hret.min()))
    b = float(np.clip(params.tail_full_history_blend, 0.0, 1.0))
    tail_loss = (1.0 - b) * tail_loss_trailing + b * tail_loss_full

    dret = s.pct_change().dropna()
    win_d = dret.iloc[-int(params.downside_vol_lookback) :]
    down = win_d[win_d < 0.0]
    down_vol_ann = float(down.std(ddof=1) * np.sqrt(252.0)) if len(down) >= 5 else 0.0

    return float(tail_loss + params.downside_vol_blend * down_vol_ann)


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
    """
    p = params or V6PfParams()
    decay_map, decay_src = load_net_decay_by_pair(screened_csv, norm_sym=norm_sym)

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

    excluded_borrow: list[dict[str, Any]] = []
    pairs_live: list[tuple[str, str]] = []
    for etf_sym, und_sym in pairs_candidate:
        kw0 = pair_cache[(etf_sym, und_sym)]["kw"]
        b = _etf_borrow_annual_actual(etf_sym, kw0)
        if b > p.exclude_if_borrow_annual_gt:
            excluded_borrow.append({"pair": f"{etf_sym}/{und_sym}", "borrow_etf_annual": b})
            continue
        pairs_live.append((etf_sym, und_sym))

    if len(pairs_live) < p.min_pairs:
        raise RuntimeError(
            f"Need at least {p.min_pairs} tradable pairs after borrow filter; got {len(pairs_live)} "
            f"(excluded {len(excluded_borrow)} for borrow > {p.exclude_if_borrow_annual_gt:.0%})."
        )

    ix_list = [pair_cache[k]["prices"].index for k in pairs_live]
    start_sim = _first_date_min_active_pairs(ix_list, p.min_pairs)

    rows_w: list[dict[str, Any]] = []
    for etf_sym, und_sym in pairs_live:
        kw0 = pair_cache[(etf_sym, und_sym)]["kw"]
        pair_lbl = f"{etf_sym}/{und_sym}"
        decay_u = float(decay_map.get((norm_sym(etf_sym), norm_sym(und_sym)), p.min_expected_decay_annual))
        decay_eff = max(p.min_expected_decay_annual, decay_u)
        borrow_a = _etf_borrow_annual_actual(etf_sym, kw0)
        risk_symbol = str(p.tail_risk_symbol_overrides.get(pair_lbl, und_sym))
        risk_raw = _tail_risk_raw(
            risk_symbol, start_sim, norm_sym=norm_sym, pair_cache=pair_cache, closes_broad=closes_broad, params=p
        )
        beta_scale = abs(float(kw0.get("beta_a", -2.0))) if p.use_beta_risk_scale else 1.0
        risk_adj = float(max(0.0, risk_raw) * max(1.0, beta_scale))
        quad = 1.0 + float(p.decay_borrow_quad) * (borrow_a**2)
        lin = 1.0 + float(p.borrow_linear_aversion) * borrow_a
        if lin <= 0.0:
            lin = 1e-18
        base_score = (decay_eff ** float(p.decay_exponent)) / quad / lin
        rows_w.append(
            {
                "pair": pair_lbl,
                "etf": etf_sym,
                "underlying": und_sym,
                "risk_symbol": risk_symbol,
                "expected_decay_annual": decay_u,
                "decay_eff": decay_eff,
                "borrow_etf_annual": borrow_a,
                "base_score": base_score,
                "risk_raw": risk_raw,
                "beta_scale": beta_scale,
                "risk_adj": risk_adj,
            }
        )

    wdf = pd.DataFrame(rows_w)
    if wdf.empty:
        raise RuntimeError("No live pairs left for sizing table.")

    corr_risk_borrow = np.nan
    if wdf["borrow_etf_annual"].nunique() > 1 and wdf["risk_adj"].nunique() > 1:
        corr_risk_borrow = float(wdf[["borrow_etf_annual", "risk_adj"]].corr().iloc[0, 1])

    lambda_eff = float(p.dd_risk_lambda)
    if np.isfinite(corr_risk_borrow) and abs(corr_risk_borrow) >= p.collinear_rho:
        lambda_eff = float(lambda_eff * p.collinear_damp)

    _pos = wdf.loc[wdf["risk_adj"] > 0, "risk_adj"].astype(float)
    risk_scale = float(_pos.median()) if len(_pos) > 0 else 1.0
    risk_scale = max(risk_scale, 1e-9)
    wdf["risk_adj_norm"] = (wdf["risk_adj"].astype(float) / risk_scale).clip(lower=0.0)

    wdf["risk_penalty"] = np.clip(
        np.exp(-lambda_eff * wdf["risk_adj_norm"].astype(float)),
        p.risk_penalty_floor,
        p.risk_penalty_cap,
    )
    wdf["risk_denom"] = 1.0 + float(p.risk_denom_coeff) * (wdf["risk_adj_norm"].astype(float) ** float(p.risk_denom_power))
    wdf["old_score"] = wdf["base_score"].astype(float)
    wdf["new_score"] = (wdf["base_score"] * wdf["risk_penalty"] / wdf["risk_denom"]).astype(float)

    old_tot = float(wdf["old_score"].sum())
    new_tot = float(wdf["new_score"].sum())
    if old_tot <= 0 or new_tot <= 0:
        raise RuntimeError("Non-positive sizing score total for old/new weights.")

    wdf["old_weight"] = wdf["old_score"] / old_tot
    wdf["new_weight_raw"] = wdf["new_score"] / new_tot

    w_signal = _apply_weight_bounds(wdf["new_weight_raw"].astype(float), lo=p.min_weight, hi=p.max_weight)
    wdf["new_weight_signal"] = w_signal

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
        "corr_risk_borrow": corr_risk_borrow,
        "lambda_eff": lambda_eff,
        "uvix_borrow_ftp_annual": uvix_borrow_annual_base,
    }
    return weight_by_key, wdf.reset_index(), meta
