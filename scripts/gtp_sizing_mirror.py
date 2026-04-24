"""
Notebook + tests: mirror ``generate_trade_plan.py`` sizing and optional covariance overlay.

Imported from ``notebooks/Buckets1-4Backtest.ipynb``.  Keep logic aligned with
``generate_trade_plan.main()`` sleeve block (budgets, masks, long/short mapping).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Reuse production implementations (single source of truth for weight math).
from generate_trade_plan import (  # noqa: E402
    _apply_notional_caps_with_redistribution,
    _clamp01,
    _core_net_decay_gate_for_core,
    _decay_score_weights,
    _norm_sym,
    compute_borrow_annual_series,
    get_borrow_col,
    load_blacklist,
    load_shares_outstanding_map,
)


def _parse_feerate_decimal(x) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return float("nan")
    s = s.replace("%", "").strip()
    try:
        return float(s) / 100.0
    except Exception:
        return float("nan")


def ibkr_borrow_map_from_cache(
    cache_path: Path | str,
    tickers: Iterable[str],
) -> dict[str, float]:
    """
    Build ETF -> annual borrow (decimal) from IBKR short-stock cache CSV
    (same schema as ``data/borrow_cache.csv``: sym, feerate).
    """
    p = Path(cache_path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    sc = cols.get("sym") or cols.get("symbol")
    fc = cols.get("feerate") or cols.get("fee_rate")
    if sc is None or fc is None:
        return {}
    d = df[[sc, fc]].dropna(subset=[sc]).copy()
    d[sc] = d[sc].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    d["_b"] = d[fc].map(_parse_feerate_decimal)
    d = d[np.isfinite(d["_b"])].copy()
    # max feerate per symbol if duplicates
    agg = d.groupby(sc, as_index=False)["_b"].max()
    want = {_norm_sym(t) for t in tickers}
    out: dict[str, float] = {}
    for _, r in agg.iterrows():
        sym = _norm_sym(r[sc])
        if sym in want:
            out[sym] = float(r["_b"])
    return out


def overlay_ibkr_borrow_on_map(
    borrow_map: dict[str, float],
    etf_syms: Iterable[str],
    cache_path: Path | str,
) -> dict[str, float]:
    """Fill / override ``borrow_map`` entries using IBKR cache where available."""
    ib = ibkr_borrow_map_from_cache(cache_path, etf_syms)
    out = dict(borrow_map)
    hit = 0
    for e in etf_syms:
        k = _norm_sym(e)
        if k in ib:
            out[k] = float(ib[k])
            hit += 1
    print(f"[IBKR] borrow_cache override: {hit}/{len(list(etf_syms))} ETF legs from {Path(cache_path).name}")
    return out


def _isolated_decay_state_path(real_path: Path | None) -> tuple[Path, bool]:
    """
    Return a path for hysteresis reads/writes without touching production.

    If *real_path* exists, copy to a temp file and return that path + ``True`` (caller
    should delete). Otherwise return an empty-object temp JSON.
    """
    fd, name = tempfile.mkstemp(prefix="gtp_mirror_decay_", suffix=".json")
    os.close(fd)
    tmp = Path(name)
    if real_path and real_path.exists():
        shutil.copy2(real_path, tmp)
    else:
        tmp.write_text("{}", encoding="utf-8")
    return tmp, True


def mirror_generate_trade_plan_sizing(
    screened: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    run_date: str,
    paths: dict[str, Any] | None = None,
    hysteresis_touch_disk: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reproduce ``generate_trade_plan.main()`` stock-sleeve sizing (no flow ledger append).

    Parameters
    ----------
    screened :
        Same columns as ``etf_screened_today.csv`` after coercion (ETF, Underlying,
        purgatory, Beta, blended_gross_decay, borrow columns, net_decay_annual, …).
    cfg :
        Full YAML dict (``load_config()``).
    run_date :
        YYYY-MM-DD for hysteresis JSON only.
    hysteresis_touch_disk :
        If False (default), hysteresis state is copied to a **temp** JSON path so the
        real ``core_leveraged_decay_state_json`` file is never read for writes in a
        way that persists — actually reads still use copy. Production file is never
        overwritten; temp file is deleted after the call.
    """
    paths = paths or (cfg.get("paths") or {})
    strategy = cfg.get("strategy", {}) or {}
    sleeves = (cfg.get("portfolio", {}) or {}).get("sleeves", {}) or {}

    core = sleeves.get("core_leveraged", {}) or {}
    wl = sleeves.get("whitelist_stock", {}) or {}
    b4 = sleeves.get("inverse_decay_bucket4", {}) or {}

    capital_usd = float(strategy.get("capital_usd", 0.0))
    gross_leverage = float(strategy.get("gross_leverage", 0.0))
    target_gross_usd = capital_usd * gross_leverage
    beta_floor = float(strategy.get("beta_floor", 0.1))

    core_w = float(core.get("target_weight", 0.85))
    wl_w = float(wl.get("target_weight", 0.15))
    b4_w = float(b4.get("target_weight", 0.0))
    b4_enabled = bool(b4.get("enabled", True))

    core_rules = core.get("rules", {}) or {}
    core_beta_min = float(core_rules.get("min_beta_used", 1.5))
    b4_rules = b4.get("rules", {}) or {}
    b4_min_edge = float(b4_rules.get("min_net_edge_annual", 0.0))
    b4_partial_hedge_ratio = _clamp01(b4_rules.get("partial_hedge_ratio", 1.0))
    b4_max_shares_outstanding_frac = _clamp01(b4_rules.get("max_shares_outstanding_frac", 0.20))
    b4_min_underlying_vol = float(b4_rules.get("min_underlying_vol", 0.50))

    soft_borrow_cap = float((cfg.get("screener") or {}).get("borrow_low", 1.0))
    b4_hard_borrow_cap = float(
        b4_rules.get("bucket4_borrow_cap", b4_rules.get("hard_borrow_cap", float("inf")))
    )

    wl_list = [_norm_sym(x) for x in ((wl.get("universe") or {}).get("etfs") or [])]
    wl_set = set(wl_list)
    zipf_exp = float((wl.get("weighting") or {}).get("zipf_exponent", 1.0))

    core_weighting_cfg = core.get("weighting", {}) or {}
    wl_weighting_cfg = wl.get("weighting", {}) or {}
    b4_weighting_cfg = b4.get("weighting", {}) or {}
    core_weight_method = str(core_weighting_cfg.get("method", "equal")).lower()
    wl_weight_method = str(wl_weighting_cfg.get("method", "decay_score")).lower()
    b4_weight_method = str(b4_weighting_cfg.get("method", "decay_score")).lower()

    shares_out_map, _ = load_shares_outstanding_map(paths)

    real_decay_path = Path(
        str(paths.get("core_leveraged_decay_state_json", "data/core_leveraged_decay_state.json"))
    )
    tmp_decay_path: Path | None = None
    cleanup_tmp = False
    if not hysteresis_touch_disk:
        tmp_decay_path, cleanup_tmp = _isolated_decay_state_path(real_decay_path if real_decay_path.exists() else None)
        core_decay_state_path = tmp_decay_path
    else:
        core_decay_state_path = real_decay_path

    keep = screened.copy()
    blist = load_blacklist(cfg)
    keep = keep[(~keep["Underlying"].isin(blist)) & (~keep["ETF"].isin(blist))].copy()
    keep["ETF"] = keep["ETF"].astype(str).map(_norm_sym)
    keep["Underlying"] = keep["Underlying"].astype(str).map(_norm_sym)
    keep["Beta"] = pd.to_numeric(keep["Beta"], errors="coerce")
    keep["beta_abs"] = keep["Beta"].abs()
    for _col in ("blended_gross_decay", "borrow_current", "net_decay_annual"):
        if _col not in keep.columns:
            keep[_col] = np.nan
        else:
            keep[_col] = pd.to_numeric(keep[_col], errors="coerce")
    borrow_col = get_borrow_col(keep)
    keep["borrow_annual"] = compute_borrow_annual_series(keep, borrow_col)

    keep["long_usd"] = 0.0
    keep["short_usd"] = 0.0
    keep["underlying_target_usd"] = 0.0
    keep["etf_target_usd"] = 0.0
    keep["gross_target_usd"] = 0.0
    keep["sleeve"] = ""

    eligible = keep.loc[keep["purgatory"] != True].copy()  # noqa: E712
    diag: dict[str, Any] = {
        "target_gross_usd": target_gross_usd,
        "n_eligible": int(len(eligible)),
    }
    if eligible.empty:
        if cleanup_tmp and tmp_decay_path and tmp_decay_path.exists():
            tmp_decay_path.unlink(missing_ok=True)
        return keep, diag

    eligible["in_whitelist"] = eligible["ETF"].isin(wl_set)
    net_decay_non_negative = ~(eligible["net_decay_annual"] < 0)
    b = eligible["borrow_annual"]
    core_borrow_ok = (~np.isfinite(b)) | (b <= soft_borrow_cap) | (eligible["in_whitelist"])
    wl_borrow_ok = True
    b4_borrow_ok = (~np.isfinite(b)) | (b <= b4_hard_borrow_cap)
    positive_beta = eligible["Beta"].gt(0)
    negative_beta = eligible["Beta"].lt(0)
    if "inverse_shortable" in eligible.columns:
        inverse_shortable = eligible["inverse_shortable"].fillna(False).astype(bool)
    else:
        inverse_shortable = negative_beta
    edge_col = "bucket4_net_edge_annual" if "bucket4_net_edge_annual" in eligible.columns else "net_decay_annual"
    b4_edge = pd.to_numeric(eligible.get(edge_col), errors="coerce")
    b4_edge_ok = (~np.isfinite(b4_edge)) | (b4_edge >= b4_min_edge)
    b4_und_vol = pd.to_numeric(eligible.get("vol_underlying_annual"), errors="coerce")
    b4_vol_ok = np.isfinite(b4_und_vol) & (b4_und_vol >= b4_min_underlying_vol)

    core_pre_decay = (
        positive_beta & eligible["beta_abs"].ge(core_beta_min) & core_borrow_ok & net_decay_non_negative
    )
    core_neg_decay_reset = (
        positive_beta & eligible["beta_abs"].ge(core_beta_min) & core_borrow_ok & ~net_decay_non_negative
    )
    try:
        core_decay_gate = _core_net_decay_gate_for_core(
            eligible,
            core_pre_decay=core_pre_decay,
            core_neg_decay_reset=core_neg_decay_reset,
            net_decay_non_negative=net_decay_non_negative,
            core_rules=core_rules,
            state_path=core_decay_state_path,
            run_date=run_date,
        )
    finally:
        if cleanup_tmp and tmp_decay_path and tmp_decay_path.exists():
            tmp_decay_path.unlink(missing_ok=True)

    eligible["in_core"] = core_pre_decay & core_decay_gate
    eligible["in_wl"] = positive_beta & eligible["in_whitelist"] & wl_borrow_ok & net_decay_non_negative
    eligible["in_b4"] = (
        negative_beta & inverse_shortable & b4_borrow_ok & b4_edge_ok & b4_vol_ok
        if b4_enabled
        else pd.Series(False, index=eligible.index)
    )

    core_names = eligible.loc[eligible["in_core"]].copy()
    wl_names = eligible.loc[eligible["in_wl"]].copy()
    b4_names = eligible.loc[eligible["in_b4"]].copy()

    b4_budget = target_gross_usd * b4_w if (not b4_names.empty and b4_w > 0 and b4_enabled) else 0.0
    b4_budget = min(b4_budget, target_gross_usd)
    remainder_budget = max(0.0, target_gross_usd - b4_budget)
    stock_weight_sum = 0.0
    if not core_names.empty:
        stock_weight_sum += core_w
    if not wl_names.empty:
        stock_weight_sum += wl_w
    if stock_weight_sum > 0:
        core_budget = remainder_budget * (core_w / stock_weight_sum) if not core_names.empty else 0.0
        wl_budget = remainder_budget * (wl_w / stock_weight_sum) if not wl_names.empty else 0.0
    else:
        core_budget = 0.0
        wl_budget = 0.0

    from generate_trade_plan import _zipf_weights  # local import

    if not core_names.empty and core_budget > 0:
        if core_weight_method == "decay_score":
            w = _decay_score_weights(core_names, core_weighting_cfg)
        else:
            w = np.ones(len(core_names)) / len(core_names)
        core_names = core_names.copy()
        core_names["gross_target_usd"] = core_budget * w
        core_names["sleeve"] = "core_leveraged"

    if not wl_names.empty and wl_budget > 0:
        if wl_weight_method == "decay_score":
            w = _decay_score_weights(wl_names, wl_weighting_cfg)
        else:
            wl_names = wl_names.copy()
            wl_names["wl_rank"] = wl_names["ETF"].map(lambda x: wl_list.index(x) if x in wl_set else 10**9)
            wl_names = wl_names.sort_values("wl_rank").copy()
            if wl_weight_method == "equal":
                w = np.ones(len(wl_names)) / len(wl_names)
            else:
                w = _zipf_weights(len(wl_names), exponent=zipf_exp)
        wl_names = wl_names.copy()
        wl_names["gross_target_usd"] = wl_budget * w
        wl_names["sleeve"] = "whitelist_stock"

    if not b4_names.empty and b4_budget > 0 and b4_enabled:
        if b4_weight_method == "equal":
            w = np.ones(len(b4_names)) / len(b4_names)
        else:
            w = _decay_score_weights(b4_names, b4_weighting_cfg)
        b4_names = b4_names.copy()
        b4_names["gross_target_usd"] = b4_budget * w
        b4_names["shares_outstanding_total"] = b4_names["ETF"].map(shares_out_map)
        b4_names["price_ref"] = pd.to_numeric(b4_names.get("borrow_price_ref", np.nan), errors="coerce")
        b4_names["gross_target_cap_usd"] = (
            b4_max_shares_outstanding_frac
            * pd.to_numeric(b4_names["shares_outstanding_total"], errors="coerce")
            * pd.to_numeric(b4_names["price_ref"], errors="coerce")
        )
        if shares_out_map:
            b4_names["gross_target_usd"] = _apply_notional_caps_with_redistribution(
                b4_names["gross_target_usd"],
                b4_names["gross_target_cap_usd"],
            )
        b4_names["sleeve"] = "inverse_decay_bucket4"

    sized = pd.concat([core_names, wl_names, b4_names], axis=0, ignore_index=False)
    sized = sized[~sized.index.duplicated(keep="first")].copy()
    sized["beta_used_abs"] = sized["beta_abs"].clip(lower=beta_floor).fillna(1.0)
    sized["hedge_ratio"] = 1.0 / sized["beta_used_abs"]
    b4_mask = sized["sleeve"].eq("inverse_decay_bucket4")
    stock_mask = ~b4_mask

    sized.loc[stock_mask, "long_usd"] = sized.loc[stock_mask, "gross_target_usd"] / (
        1.0 + sized.loc[stock_mask, "hedge_ratio"]
    )
    sized.loc[stock_mask, "short_usd"] = -(sized.loc[stock_mask, "hedge_ratio"] * sized.loc[stock_mask, "long_usd"])

    sized.loc[b4_mask, "short_usd"] = -sized.loc[b4_mask, "gross_target_usd"]
    sized.loc[b4_mask, "long_usd"] = -(
        b4_partial_hedge_ratio * sized.loc[b4_mask, "beta_used_abs"] * sized.loc[b4_mask, "gross_target_usd"]
    )
    sized["underlying_target_usd"] = sized["long_usd"]
    sized["etf_target_usd"] = sized["short_usd"]

    keep.loc[sized.index, "long_usd"] = sized["long_usd"]
    keep.loc[sized.index, "short_usd"] = sized["short_usd"]
    keep.loc[sized.index, "underlying_target_usd"] = sized["underlying_target_usd"]
    keep.loc[sized.index, "etf_target_usd"] = sized["etf_target_usd"]
    keep.loc[sized.index, "gross_target_usd"] = sized["gross_target_usd"]
    keep.loc[sized.index, "sleeve"] = sized["sleeve"]

    diag.update(
        {
            "core_budget": core_budget,
            "wl_budget": wl_budget,
            "b4_budget": b4_budget,
            "n_core": int(len(core_names)),
            "n_wl": int(len(wl_names)),
            "n_b4": int(len(b4_names)),
        }
    )
    return keep, diag


def covariance_overlay_signal_weights(
    signal_w: np.ndarray,
    sym_labels: list[str],
    ret_df: pd.DataFrame,
    *,
    shrink: float = 0.35,
    penalty_strength: float = 0.85,
    min_obs: int = 30,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Shrink covariance, compute marginal contribution, apply ``cov_penalty``, renormalize.

    *sym_labels* align rows of *signal_w* with columns of *ret_df* (same strings).
    """
    n = len(signal_w)
    if n == 0:
        return signal_w, pd.DataFrame(), pd.DataFrame()
    r = ret_df.reindex(columns=sym_labels).astype(float)
    cov = r.cov(min_periods=min_obs)
    d = np.diag(np.diag(cov.values))
    shrunk = (1.0 - shrink) * cov.values + shrink * d
    shrunk_df = pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)
    w_probe = np.asarray(signal_w, dtype=float)
    if w_probe.sum() <= 0:
        return w_probe, shrunk_df, pd.DataFrame()
    w_n = w_probe / w_probe.sum()
    mrc = shrunk_df.values @ w_n
    contrib = np.clip(w_n * mrc, 0.0, None)
    med = float(np.median(contrib[contrib > 0])) if np.any(contrib > 0) else 1.0
    med = max(med, 1e-12)
    cov_contrib_norm = contrib / med
    cov_penalty = 1.0 / (1.0 + penalty_strength * cov_contrib_norm)
    adj = w_probe * cov_penalty
    ssum = float(adj.sum())
    if ssum <= 0:
        out_w = np.ones(n) / n
    else:
        out_w = adj / ssum
    meta = pd.DataFrame(
        {
            "sym": sym_labels,
            "signal_w_in": w_probe,
            "cov_contrib_norm": cov_contrib_norm,
            "cov_penalty": cov_penalty,
            "signal_w_out": out_w,
        }
    )
    return out_w, shrunk_df, meta


def pair_weights_from_gtp_mirror(
    mirror_df: pd.DataFrame,
    universe: list[tuple[str, str, float]],
) -> dict[str, float]:
    """
    Map ``mirror_generate_trade_plan_sizing`` output to **PAIR_WEIGHTS** keyed by ETF,
    proportional to ``gross_target_usd`` (else ``max(long_usd - short_usd, 0)``).
    """
    df = mirror_df.copy()
    if "ETF" not in df.columns or "Underlying" not in df.columns:
        raise KeyError("mirror_df must contain ETF and Underlying columns")
    df["ETF"] = df["ETF"].astype(str).map(_norm_sym)
    df["Underlying"] = df["Underlying"].astype(str).map(_norm_sym)
    if "gross_target_usd" in df.columns:
        df["_gross"] = pd.to_numeric(df["gross_target_usd"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        lu = pd.to_numeric(df["long_usd"], errors="coerce").fillna(0.0)
        su = pd.to_numeric(df["short_usd"], errors="coerce").fillna(0.0)
        df["_gross"] = (lu - su).clip(lower=0.0)
    grp = df.groupby(["ETF", "Underlying"], as_index=True)["_gross"].max()
    gross_by_etf: dict[str, float] = {}
    for e, u, _ in universe:
        k = (_norm_sym(e), _norm_sym(u))
        try:
            gv = float(grp.loc[k])
        except KeyError:
            gv = 0.0
        gross_by_etf[_norm_sym(e)] = gross_by_etf.get(_norm_sym(e), 0.0) + gv
    tot = float(sum(gross_by_etf.values()))
    if tot <= 0:
        n = max(len(universe), 1)
        return {e: 1.0 / n for e, _, _ in universe}
    return {e: gross_by_etf.get(_norm_sym(e), 0.0) / tot for e, _, _ in universe}


def pair_weights_with_underlying_cov_penalty(
    universe: list[tuple[str, str, float]],
    mirror_df: pd.DataFrame,
    underlying_returns: pd.DataFrame,
    *,
    base_weights: dict[str, float] | None = None,
    lookback: int = 252,
    shrink: float = 0.35,
    penalty_strength: float = 0.85,
    min_obs: int = 30,
    max_pair_weight: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Post-process **ETF-level** ``PAIR_WEIGHTS`` using shrunk covariance of **underlying** log returns.

    1. Start from ``base_weights`` or ``pair_weights_from_gtp_mirror(mirror_df, universe)``.
    2. Aggregate ETF weights onto underlyings with exposure ∝ ``w_etf × |beta|``.
    3. Run :func:`covariance_overlay_signal_weights` on that underlying exposure vector.
    4. Push multipliers back to each ETF (same multiplier for all names sharing an underlying), renormalize.
    5. Optionally clip each ETF weight at ``max_pair_weight`` then renormalize (surplus discarded — v1).
    """
    w0 = dict(base_weights) if base_weights is not None else pair_weights_from_gtp_mirror(mirror_df, universe)
    s0 = float(sum(max(0.0, float(v)) for v in w0.values()))
    if s0 <= 0:
        n = max(len(universe), 1)
        w0 = {e: 1.0 / n for e, _, _ in universe}
    else:
        w0 = {e: max(0.0, float(v)) / s0 for e, v in w0.items()}

    u_exposure: dict[str, float] = {}
    for e, u, b in universe:
        ue = _norm_sym(u)
        beta_abs = abs(float(b)) if np.isfinite(b) else 1.0
        u_exposure[ue] = u_exposure.get(ue, 0.0) + float(w0.get(e, 0.0)) * float(beta_abs)

    syms = sorted(u for u, v in u_exposure.items() if v > 1e-18)
    if not syms or underlying_returns is None or underlying_returns.empty:
        return w0, pd.DataFrame()

    r = underlying_returns.copy()
    r.columns = [_norm_sym(str(c)) for c in r.columns]
    r = r[[c for c in r.columns if c in syms]]
    if r.empty or len(r) < int(min_obs):
        return w0, pd.DataFrame()

    r = r.iloc[-int(max(lookback, min_obs)) :]
    r = r.astype(float).replace([np.inf, -np.inf], np.nan)
    r = np.log(r.clip(lower=1e-12)).diff().iloc[1:]
    r = r.dropna(axis=0, how="all")
    if len(r) < int(min_obs):
        return w0, pd.DataFrame()

    w_probe = np.array([float(u_exposure[s]) for s in syms], dtype=float)
    if float(w_probe.sum()) <= 0:
        return w0, pd.DataFrame()

    w_out, shrunk_df, meta = covariance_overlay_signal_weights(
        w_probe,
        syms,
        r,
        shrink=float(shrink),
        penalty_strength=float(penalty_strength),
        min_obs=int(min_obs),
    )

    S = float(sum(u_exposure[s] for s in syms))
    if S <= 1e-18:
        return w0, meta
    p_u = {s: float(u_exposure[s]) / S for s in syms}
    q_u = {syms[i]: float(w_out[i]) for i in range(len(syms))}
    mult: dict[str, float] = {}
    for s in syms:
        pu = p_u.get(s, 0.0)
        qu = q_u.get(s, 0.0)
        if pu <= 1e-18:
            mult[s] = 1.0
        else:
            mult[s] = float(qu / pu) if qu > 0 else 0.0

    new_w: dict[str, float] = {}
    for e, u, _b in universe:
        m = mult.get(_norm_sym(u), 1.0)
        new_w[e] = float(w0.get(e, 0.0) * m)
    tot = float(sum(max(0.0, v) for v in new_w.values()))
    if tot <= 1e-18:
        return w0, meta
    new_w = {e: max(0.0, v) / tot for e, v in new_w.items()}

    if max_pair_weight is not None and float(max_pair_weight) > 0:
        cap = float(max_pair_weight)
        new_w = {e: min(v, cap) for e, v in new_w.items()}
        tot2 = float(sum(new_w.values()))
        if tot2 > 1e-18:
            new_w = {e: v / tot2 for e, v in new_w.items()}

    return new_w, meta
