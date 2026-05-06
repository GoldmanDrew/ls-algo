#!/usr/bin/env python3
"""
generate_trade_plan.py

UPDATE: Portfolio construction matches YAML sleeves (85/15 stock sleeves + flow overlay)

Implements:
- Stock sleeves (rebalanced as part of each run):
    * core_leveraged:   target_weight=0.85 of target_gross_usd
        - requires |Beta| >= min_beta_used
        - apply soft borrow ban (screener.borrow_low) UNLESS whitelisted sleeve overrides
        - equal weight within sleeve (optional max_name_weight cap not enforced here; can be added)

    * whitelist_stock:  target_weight=0.15 of target_gross_usd
        - ETF must be in whitelist list
        - zipf weighting by list order (rank_order=as_listed)
        - OPTIONAL hard borrow cap for whitelist sleeve (e.g. 20%) if provided in cfg

- Purgatory handling (from screened CSV; see daily_screener ``recompute_purgatory_by_bucket``):
    * OUTPUT purgatory rows with 0 targets so execution won’t auto-close.
    * Does NOT allocate new exposure to purgatory (borrow soft band OR net_edge_p50 in 0–5%).
    * Sizing set also requires ``net_edge_p50_annual >= 0`` when that column exists.

- inverse_decay_bucket4: optional ``enabled: false`` on the sleeve — disables all B4 targets; core/wl
  split the full gross budget.

- core_leveraged (bucket-1 style core): optional ``min_net_decay_annual`` and ``net_decay_hysteresis`` in
  YAML rules — tighter net-decay selectivity for core only; whitelist unchanged. If ``min_net_decay_annual``
  > 0 it is a **hard floor** (always enforced, including over hysteresis and missing-data paths).
  Hysteresis uses ``paths.core_leveraged_decay_state_json`` to reduce pairs bouncing in/out when decay oscillates.

- Flow overlay (tracked, not "sized into" the 85/15 stock budgets):
    * weekly_add_usd = (annual_deployment_rate / 52) * deployment_base
      where deployment_base is "current_equity" or "initial_equity"
    * distributed across flow weights (fixed)
    * CUMULATIVE notional tracked in a ledger CSV under data/flow_ledger.csv (or cfg path)
    * optional hard borrow cap for flow sleeve (e.g. 20%) if provided in cfg
    * optional constraints:
        - max_cumulative_weight (cap as % of current equity)
        - pause_if_drawdown_exceeds (requires equity curve; here we support manual "pause" flag)

Outputs:
- proposed_trades.csv (stock sleeves sized + purgatory keep-open rows)
- flow_ledger.csv (append-only record of cumulative flow)

Notes:
- This script assumes screened_csv includes: ETF, Underlying, purgatory, Beta
- Borrow caps require screened_csv to include a borrow column OR you pass a borrow_map externally.
  If you don’t have borrow in screened_csv, set caps high or skip borrow filtering here.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path("config/strategy_config.yml")
TRADING_DAYS = 252


# -----------------------------
# Basic helpers
# -----------------------------
def today_str() -> str:
    return date.today().isoformat()


def run_dir(run_date: str) -> Path:
    return Path("data") / "runs" / run_date


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _load_core_decay_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "by_etf": {}}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("by_etf"), dict):
            out = {"version": int(d.get("version", 1)), "by_etf": dict(d["by_etf"])}
            return out
    except Exception:
        pass
    return {"version": 1, "by_etf": {}}


def _save_core_decay_state(path: Path, by_etf: dict[str, Any], run_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "updated_run_date": run_date, "by_etf": by_etf}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _core_net_decay_gate_for_core(
    eligible: pd.DataFrame,
    *,
    core_pre_decay: pd.Series,
    core_neg_decay_reset: pd.Series,
    net_decay_non_negative: pd.Series,
    core_rules: dict,
    state_path: Path,
    run_date: str,
) -> pd.Series:
    """Extra core-only gate from net_decay_annual (bucket-1 selectivity).

    ``core_neg_decay_reset``: structural core shape (beta, borrow) but **negative** net decay —
    clears that ETF's sticky flag so a later recovery requires the enter threshold again.

    Only rows satisfying ``core_pre_decay`` (structural core candidate + non-negative decay) can
    be gated off; all other rows return pass-through True so whitelist-only names do not corrupt
    sticky state.

    - If hysteresis disabled and min_net_decay_annual <= 0: all-True.
    - If hysteresis disabled and min_net_decay_annual > 0: require finite net_decay >= min on
      ``core_pre_decay`` rows.
    - If hysteresis enabled: sticky per ETF in ``state_path``; ``enter_above`` > ``exit_below``.
      Missing decay: if sticky was on, keep on (data-gap churn guard); otherwise off.
    - Whenever ``min_net_decay_annual`` > 0, that floor is **always** applied last: no NaN bypass,
      no hysteresis carry can admit core below the minimum (``sticky_in`` in state matches final).
    """
    nd = pd.to_numeric(eligible["net_decay_annual"], errors="coerce")
    min_nd = float(core_rules.get("min_net_decay_annual", 0.0) or 0.0)
    hyst_cfg = core_rules.get("net_decay_hysteresis") or {}
    hyst_on = bool(hyst_cfg.get("enabled", False))

    gate = pd.Series(True, index=eligible.index)
    idx_core = eligible.loc[core_pre_decay.fillna(False)].index

    if not hyst_on:
        if min_nd <= 0:
            return gate
        gate.loc[idx_core] = np.isfinite(nd.loc[idx_core]) & (nd.loc[idx_core] >= min_nd)
        return gate

    enter = float(hyst_cfg.get("enter_above", 0.0))
    exit_b = float(hyst_cfg.get("exit_below", -1.0))
    if enter <= exit_b:
        raise ValueError(
            "portfolio.sleeves.core_leveraged.rules.net_decay_hysteresis: "
            "enter_above must be strictly greater than exit_below"
        )

    doc = _load_core_decay_state(state_path)
    by_etf: dict[str, Any] = deepcopy(doc.get("by_etf", {}))

    for idx in eligible.loc[core_neg_decay_reset.fillna(False)].index:
        etf = _norm_sym(str(eligible.loc[idx, "ETF"]))
        by_etf[etf] = {"sticky_in": False}

    for idx in idx_core:
        etf = _norm_sym(str(eligible.loc[idx, "ETF"]))
        v_raw = nd.loc[idx]
        v = float(v_raw) if pd.notna(v_raw) else float("nan")
        prev_entry = by_etf.get(etf, {})
        prev = prev_entry.get("sticky_in") if isinstance(prev_entry, dict) else None
        prev_on = prev is True

        if not bool(net_decay_non_negative.loc[idx]):
            raw = False
        elif not np.isfinite(v):
            raw = prev_on
        elif prev_on:
            raw = bool(v >= exit_b)
        else:
            raw = bool(v >= enter)

        if min_nd > 0.0:
            meets_min = bool(np.isfinite(v) and v >= min_nd)
        else:
            meets_min = True
        final_in = bool(raw) and meets_min
        gate.loc[idx] = final_in
        by_etf[etf] = {"sticky_in": final_in}

    _save_core_decay_state(state_path, by_etf, run_date)
    return gate


def _zipf_weights(n: int, exponent: float = 1.0) -> np.ndarray:
    r = np.arange(1, n + 1, dtype=float)
    w = 1.0 / np.power(r, exponent)
    return w / w.sum() if w.sum() > 0 else w


def _sizing_signal_column(weighting_cfg: dict, *, sleeve_name: str | None = None) -> tuple[str, str]:
    """
    Return (mode, column_name) for decay_score sizing.

    *mode* is ``blended_decay`` or ``net_edge``.  Column is the annual edge/decay field to read.
    """
    mode = str(weighting_cfg.get("sizing_signal", "blended_decay") or "blended_decay").lower().strip()
    if mode in ("blended", "blended_decay", "decay", "gross_decay"):
        mode = "blended_decay"
    elif mode in ("net_edge", "edge", "net"):
        mode = "net_edge"
    else:
        mode = "blended_decay"

    if mode == "blended_decay":
        return mode, "blended_gross_decay"

    col = weighting_cfg.get("sizing_edge_column")
    if isinstance(col, str) and col.strip():
        return mode, col.strip()

    sn = (sleeve_name or "").strip().lower()
    if sn == "inverse_decay_bucket4":
        return mode, "bucket4_net_edge_annual"
    return mode, "net_edge_p50_annual"


def _decay_score_weights(
    df: pd.DataFrame,
    weighting_cfg: dict,
    beta_col: str = "beta_abs",
    sleeve_name: str | None = None,
) -> np.ndarray:
    """Compute portfolio weights from decay-score signal blended with equal weight.

    Parameters
    ----------
    df : DataFrame of eligible names. Must contain *beta_col* and borrow ``borrow_current``.
         Primary signal column depends on ``weighting_cfg['sizing_signal']`` (default
         ``blended_decay`` uses ``blended_gross_decay``; ``net_edge`` uses ``net_edge_p50_annual``
         or ``sizing_edge_column``, with the same borrow discount as blended decay).
    weighting_cfg : sleeve ``weighting`` dict from strategy_config.yml.
    beta_col : column name for absolute-value beta (default ``beta_abs``).
    sleeve_name : optional sleeve key (e.g. ``inverse_decay_bucket4``) for net-edge column defaults.

    Returns
    -------
    np.ndarray of weights summing to 1.0, aligned with *df* row order.
    """
    n = len(df)
    if n == 0:
        return np.array([])

    # --- config ----------------------------------------------------------
    borrow_aversion = float(weighting_cfg.get("borrow_aversion", 1.0))
    margin_power    = float(weighting_cfg.get("margin_efficiency_power", 0.0))
    eq_blend        = _clamp01(weighting_cfg.get("eq_blend", 0.0))
    max_w           = float(weighting_cfg.get("max_name_weight", 1.0))

    # --- raw sizing score (same borrow discount as blended decay path) ---
    mode, sig_col = _sizing_signal_column(weighting_cfg, sleeve_name=sleeve_name)
    if mode == "blended_decay":
        signal = pd.to_numeric(df["blended_gross_decay"], errors="coerce")
    elif sig_col in df.columns:
        signal = pd.to_numeric(df[sig_col], errors="coerce")
    else:
        signal = pd.Series(np.nan, index=df.index)
    if mode == "net_edge" and (sig_col not in df.columns or bool(signal.isna().all())):
        signal = pd.to_numeric(df["blended_gross_decay"], errors="coerce")

    borrow = pd.to_numeric(df["borrow_current"], errors="coerce").fillna(0.0)
    raw_score = signal - borrow_aversion * borrow  # higher = better (annual units)

    # --- margin efficiency adjustment ------------------------------------
    beta_abs = pd.to_numeric(df[beta_col], errors="coerce").clip(lower=0.1)
    margin_adj = np.power(1.0 / beta_abs, margin_power)  # favours 2x over 3x
    adjusted = (raw_score * margin_adj).fillna(0.0).clip(lower=0.0)

    # --- normalise signal weights ----------------------------------------
    sig_total = adjusted.sum()
    signal_w = adjusted.values / sig_total if sig_total > 0 else np.zeros(n)

    # --- blend with equal weight -----------------------------------------
    eq_w = np.ones(n) / n
    final_w = eq_blend * eq_w + (1.0 - eq_blend) * signal_w

    # --- max-name-weight cap with redistribution -------------------------
    for _ in range(10):
        excess = np.maximum(final_w - max_w, 0.0)
        total_excess = excess.sum()
        if total_excess < 1e-12:
            break
        final_w = np.minimum(final_w, max_w)
        uncapped = final_w < max_w - 1e-12
        if uncapped.any():
            uc_total = final_w[uncapped].sum()
            if uc_total > 0:
                final_w[uncapped] += total_excess * (final_w[uncapped] / uc_total)
        else:
            break

    # --- safety normalisation --------------------------------------------
    s = final_w.sum()
    return final_w / s if s > 0 else eq_w


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_shares_outstanding_map(paths_cfg: dict) -> tuple[dict[str, float], Path | None]:
    """
    Load ETF shares outstanding map from etf-dashboard CSV.
    Returns (symbol->shares_outstanding, resolved_path).
    """
    configured = paths_cfg.get("etf_shares_outstanding_csv")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(str(configured)))
    candidates.extend(
        [
            Path("../etf-dashboard/data/etf_shares_outstanding.csv"),
            Path("../etf-dashboard/etf_shares_outstanding.csv"),
            Path("data/etf_shares_outstanding.csv"),
        ]
    )
    src = _first_existing_path(candidates)
    if src is None:
        return {}, None

    try:
        df = pd.read_csv(src)
    except Exception:
        return {}, src
    if df.empty:
        return {}, src

    symbol_col = None
    for c in ["ETF", "ticker", "symbol", "Ticker", "Symbol"]:
        if c in df.columns:
            symbol_col = c
            break
    shares_col = None
    for c in [
        "shares_outstanding",
        "sharesOutstanding",
        "SharesOutstanding",
        "total_shares_outstanding",
        "totalSharesOutstanding",
        "shares",
    ]:
        if c in df.columns:
            shares_col = c
            break
    if symbol_col is None or shares_col is None:
        return {}, src

    out: dict[str, float] = {}
    for _, r in df[[symbol_col, shares_col]].dropna().iterrows():
        sym = _norm_sym(r[symbol_col])
        sh = pd.to_numeric(r[shares_col], errors="coerce")
        if pd.notna(sh) and float(sh) > 0:
            out[sym] = float(sh)
    return out, src


def _apply_notional_caps_with_redistribution(
    desired: pd.Series,
    caps: pd.Series,
) -> pd.Series:
    """
    Cap desired notionals at per-row limits and redistribute excess to uncapped rows.
    """
    desired = pd.to_numeric(desired, errors="coerce").fillna(0.0).clip(lower=0.0)
    caps = pd.to_numeric(caps, errors="coerce")
    caps = np.where((~np.isfinite(caps)) | (caps <= 0), np.inf, caps)
    caps_s = pd.Series(caps, index=desired.index, dtype=float)

    alloc = desired.copy()
    for _ in range(20):
        over = (alloc - caps_s).clip(lower=0.0)
        excess = float(over.sum())
        if excess <= 1e-9:
            break
        alloc = np.minimum(alloc, caps_s)
        headroom_mask = alloc < (caps_s - 1e-9)
        if not headroom_mask.any():
            break
        w = desired[headroom_mask].copy()
        wsum = float(w.sum())
        if wsum <= 1e-12:
            w = pd.Series(1.0, index=w.index)
            wsum = float(w.sum())
        alloc.loc[headroom_mask] = alloc.loc[headroom_mask] + excess * (w / wsum)
    alloc = np.minimum(alloc, caps_s)
    return alloc


def _pick_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in cols:
            return cols[key]
    return None


def _short_leg_frac_array(
    beta_abs: np.ndarray,
    beta_floor: float,
    sleeve: np.ndarray,
) -> np.ndarray:
    """
    Fraction of pair gross that is ETF short USD (stock sleeves: hedge_ratio / (1+hr)).
    Bucket 4: gross is the inverse ETF short leg → fraction 1.0.
    """
    b = np.maximum(np.asarray(beta_abs, dtype=float), float(beta_floor))
    hr = 1.0 / b
    sf = hr / (1.0 + hr)
    s = np.asarray(sleeve).astype(str)
    is_b4 = s == "inverse_decay_bucket4"
    sf = np.where(is_b4, 1.0, sf)
    return np.clip(sf, 1e-12, 1.0)


def _project_to_capped_simplex_numpy(desired: np.ndarray, caps: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Match backtest notebook: project nonnegative desired onto caps, sum to 1."""
    d = np.asarray(desired, dtype=float).copy()
    c = np.asarray(caps, dtype=float).copy()
    n = len(d)
    if n == 0:
        return d
    d = np.clip(d, 0.0, None)
    s = float(d.sum())
    if s <= 0:
        d[:] = 1.0 / n
    else:
        d /= s
    c = np.clip(c, 0.0, None)
    if float(c.sum()) < 1.0 - tol:
        c = np.maximum(c, d)
    w = np.zeros(n, dtype=float)
    free = np.ones(n, dtype=bool)
    rem = 1.0
    while bool(free.any()) and rem > tol:
        idx = np.where(free)[0]
        base = d[idx]
        bsum = float(base.sum())
        trial = (rem * base / bsum) if bsum > 0 else np.full(len(idx), rem / max(len(idx), 1))
        hit = trial > (c[idx] + tol)
        if not bool(hit.any()):
            w[idx] = trial
            rem = 0.0
            break
        hit_idx = idx[hit]
        w[hit_idx] = c[hit_idx]
        rem = 1.0 - float(w.sum())
        free[hit_idx] = False
    if rem > tol:
        idx = np.where(free)[0]
        if len(idx):
            room = np.clip(c[idx] - w[idx], 0.0, None)
            rs = float(room.sum())
            if rs > tol:
                w[idx] += rem * (room / rs)
    w = np.clip(w, 0.0, None)
    s2 = float(w.sum())
    if s2 > 0:
        w /= s2
    m = np.minimum(w, c)
    ms = float(m.sum())
    return m / max(ms, 1e-12)


def _project_pair_and_underlying_numpy(
    w: np.ndarray,
    pair_caps: np.ndarray,
    und_code: np.ndarray,
    und_cap: float,
    *,
    max_iter: int = 400,
    tol: float = 1e-11,
) -> np.ndarray:
    """Iterative projection (pair caps + per-underlying mass cap). Weights sum to ≤ 1."""
    w = np.asarray(w, dtype=float).copy()
    pc = np.asarray(pair_caps, dtype=float).copy()
    uc = np.asarray(und_code, dtype=int).copy()
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w[:] = 1.0 / max(len(w), 1)
    else:
        w /= w.sum()
    pc = np.clip(pc, 0.0, None)
    n = len(w)
    if n == 0:
        return w
    max_u = int(uc.max()) + 1 if n else 0
    for _ in range(int(max_iter)):
        prev = w.copy()
        w = np.minimum(w, pc)
        und_tot = np.bincount(uc, weights=w, minlength=max_u)
        for u in range(max_u):
            tot = float(und_tot[u])
            if tot > float(und_cap) + tol:
                mask = uc == u
                if tot > 0:
                    w[mask] *= float(und_cap) / tot
        rem = 1.0 - float(w.sum())
        if rem > tol:
            und_tot = np.bincount(uc, weights=w, minlength=max_u)
            und_head = np.clip(float(und_cap) - und_tot, 0.0, None)
            name_und_head = und_head[uc]
            pair_head = np.clip(pc - w, 0.0, None)
            head = np.minimum(pair_head, name_und_head)
            hs = float(head.sum())
            if hs > tol:
                w += rem * (head / hs)
        # Do not force renorm to 1 here: if constraints leave slack unplaceable, total weight < 1
        # (partial gross deployment vs target book).
        if float(np.max(np.abs(w - prev))) < 1e-10:
            break
    st = float(w.sum())
    if st > 1.0 + 1e-9:
        w /= st
    return w


def _compute_pair_weight_caps_array(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    beta_floor: float,
    caps: dict[str, Any],
    shares_out_map: dict[str, float],
) -> np.ndarray:
    """
    Per-row max weight (fraction of total book gross) from hard pair cap + liquidity.
    Liquidity caps mirror backtest notebooks (AUM, shares_available, float, median volume).
    """
    n = len(sized)
    if n == 0:
        return np.zeros(0, dtype=float)
    T = max(float(target_gross_usd), 1.0)
    max_pair = float(caps.get("max_pair_weight_cap", 0.05) or 0.05)
    missing = float(caps.get("missing_shares_cap", 0.02) or 0.02)
    aum_pct = float(caps.get("aum_use_pct", 0.0) or 0.0)
    sh_av_pct = float(caps.get("short_avail_use_pct", 0.25) or 0.25)
    sout_frac = float(caps.get("shares_outstanding_use_frac", 0.0) or 0.0)
    mv_pct = float(caps.get("median_daily_volume_use_pct", 0.0) or 0.0)

    if "beta_abs" in sized.columns:
        ba = pd.to_numeric(sized["beta_abs"], errors="coerce").to_numpy(dtype=float)
    else:
        ba = pd.to_numeric(sized["Beta"], errors="coerce").abs().to_numpy(dtype=float)
    if "sleeve" in sized.columns:
        slv = sized["sleeve"].astype(str).to_numpy()
    else:
        slv = np.array(["core_leveraged"] * n, dtype=object)
    sf = _short_leg_frac_array(ba, beta_floor, slv)
    ref_short = T * sf

    if "borrow_price_ref" in sized.columns:
        px = pd.to_numeric(sized["borrow_price_ref"], errors="coerce").to_numpy(dtype=float)
    else:
        px = np.full(n, np.nan, dtype=float)

    if "shares_available" in sized.columns:
        sh_av = pd.to_numeric(sized["shares_available"], errors="coerce").to_numpy(dtype=float)
    else:
        sh_av = np.full(n, np.nan, dtype=float)
    aum_col = caps.get("aum_column")
    if isinstance(aum_col, str) and aum_col in sized.columns:
        aum = pd.to_numeric(sized[aum_col], errors="coerce").to_numpy(dtype=float)
    else:
        ac = _pick_first_column(
            sized,
            ["aum_usd", "etf_aum_usd", "AUM_USD", "aum", "etf_aum", "nav_aum_usd"],
        )
        aum = pd.to_numeric(sized[ac], errors="coerce").to_numpy(dtype=float) if ac else np.full(n, np.nan)

    mv_col = caps.get("median_volume_column")
    if isinstance(mv_col, str) and mv_col in sized.columns:
        med_vol = pd.to_numeric(sized[mv_col], errors="coerce").to_numpy(dtype=float)
    else:
        vc = _pick_first_column(
            sized,
            [
                "median_daily_volume_shares",
                "median_volume_shares_60d",
                "adv_median_shares",
                "adv_shares_median",
                "median_volume_shares",
            ],
        )
        med_vol = pd.to_numeric(sized[vc], errors="coerce").to_numpy(dtype=float) if vc else np.full(n, np.nan)

    etf_syms = sized["ETF"].astype(str).map(_norm_sym).to_numpy()
    sh_out = np.array([float(shares_out_map.get(str(e), float("nan"))) for e in etf_syms], dtype=float)

    def _w_cap(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den
        out = np.where(np.isfinite(out) & (out > 0), out, np.nan)
        return out

    pieces = []
    if aum_pct > 0:
        num = aum_pct * np.where(np.isfinite(aum) & (aum > 0), aum, np.nan)
        pieces.append(_w_cap(num, ref_short))
    if sh_av_pct > 0:
        num = sh_av_pct * np.where(np.isfinite(sh_av) & (sh_av > 0), sh_av, np.nan) * px
        pieces.append(_w_cap(num, ref_short))
    if sout_frac > 0 and shares_out_map:
        num = sout_frac * np.where(np.isfinite(sh_out) & (sh_out > 0), sh_out, np.nan) * px
        pieces.append(_w_cap(num, ref_short))
    if mv_pct > 0:
        num = mv_pct * np.where(np.isfinite(med_vol) & (med_vol > 0), med_vol, np.nan) * px
        pieces.append(_w_cap(num, ref_short))

    if pieces:
        mat = np.column_stack(pieces)
        row_min = np.nanmin(mat, axis=1)
        tight = np.where(np.isfinite(row_min), row_min, missing)
    else:
        tight = np.full(n, missing, dtype=float)
    tight = np.clip(tight, 0.0, None)
    return np.minimum(max_pair, tight).astype(float)


def _shrunk_covariance(returns: pd.DataFrame, *, shrink: float) -> pd.DataFrame:
    """Sample covariance shrunk towards its diagonal: (1-s)*Σ + s*diag(Σ)."""
    cov = returns.cov(min_periods=1)
    diag = np.diag(np.diag(cov.values))
    s = float(np.clip(shrink, 0.0, 1.0))
    out = (1.0 - s) * cov.values + s * diag
    return pd.DataFrame(out, index=cov.index, columns=cov.columns)


def _normalize_underlying_returns(
    returns_df: pd.DataFrame | None,
    *,
    lookback: int,
    min_obs: int,
    needed_underlyings: list[str],
) -> pd.DataFrame | None:
    """
    Coerce a wide returns frame (dates × underlyings) of EITHER prices OR returns into
    log returns covering the *needed_underlyings* list. Returns ``None`` when coverage is thin.
    """
    if returns_df is None or returns_df.empty:
        return None
    df = returns_df.copy()
    df.columns = [_norm_sym(str(c)) for c in df.columns]
    cols = [c for c in df.columns if c in set(needed_underlyings)]
    if len(cols) < 2:
        return None
    df = df[cols]
    df = df.replace([np.inf, -np.inf], np.nan)
    looks_like_prices = bool((df.dropna(how="all") > 1.0).any().mean() > 0.5) if len(df) else False
    if looks_like_prices:
        df = np.log(df.clip(lower=1e-12)).diff()
    df = df.dropna(axis=0, how="all")
    if len(df) > int(lookback):
        df = df.iloc[-int(lookback) :]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2 or len(df) < int(min_obs):
        return None
    return df


def apply_covariance_balance(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    beta_floor: float,
    strategy: dict[str, Any],
    paths: dict[str, Any] | None = None,
    shares_out_map: dict[str, float] | None = None,
    returns_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    One-shot covariance penalty on per-pair ``gross_target_usd`` to attenuate exposure to
    high-correlation underlyings, then re-apply :func:`apply_gross_sizing_book_caps` so
    pair / underlying / liquidity caps still bind.

    Aggregates exposure per underlying as ``e_u = Σ_p w_p · |β_p|``, computes diagonally-shrunk
    covariance Σ̃ on log-returns, marginal contribution ``mrc_u = (Σ̃ ê)_u`` and risk contribution
    ``c_u = max(ê_u·mrc_u, 0)``. Multiplier ``m_u = 1 / (1 + λ · c_u / median(c>0))`` is applied to
    every pair sharing that underlying. Gross is preserved by renormalizing to the cap-respecting
    sum, then re-projected through the cap stack.

    Config: ``strategy.covariance_balance`` (enabled, lookback_days, min_obs, shrink,
    penalty_strength, max_relative_shift). Disabled → no-op.
    """
    diag: dict[str, Any] = {"applied": False}
    if sized is None or sized.empty:
        return sized, diag
    raw = strategy.get("covariance_balance")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return sized, diag
    cfg = dict(raw)
    lookback = int(cfg.get("lookback_days", 252) or 252)
    min_obs = int(cfg.get("min_obs", 60) or 60)
    shrink = float(cfg.get("shrink", 0.35) or 0.35)
    lam = float(cfg.get("penalty_strength", 0.85) or 0.85)
    max_shift = float(cfg.get("max_relative_shift", 0.50) or 0.50)

    df = sized.copy()
    df["ETF"] = df["ETF"].astype(str).map(_norm_sym)
    df["Underlying"] = df["Underlying"].astype(str).map(_norm_sym)
    if "beta_abs" not in df.columns:
        df["beta_abs"] = pd.to_numeric(df["Beta"], errors="coerce").abs()

    g = pd.to_numeric(df["gross_target_usd"], errors="coerce").fillna(0.0).clip(lower=0.0)
    gsum = float(g.sum())
    if gsum <= 1e-18:
        return sized, diag

    needed = sorted({str(u) for u in df["Underlying"].astype(str).tolist()})
    R = _normalize_underlying_returns(
        returns_df, lookback=lookback, min_obs=min_obs, needed_underlyings=needed,
    )
    if R is None:
        diag.update({"applied": False, "reason": "insufficient_returns"})
        return sized, diag

    syms = [c for c in R.columns]
    sym_idx = {s: i for i, s in enumerate(syms)}

    beta_abs = pd.to_numeric(df["beta_abs"], errors="coerce").fillna(1.0).clip(lower=float(beta_floor))
    exposure = np.zeros(len(syms), dtype=float)
    for u, w_pair, b in zip(df["Underlying"].tolist(), g.tolist(), beta_abs.tolist()):
        i = sym_idx.get(str(u))
        if i is None:
            continue
        exposure[i] += float(w_pair) * float(b)
    e_sum = float(exposure.sum())
    if e_sum <= 1e-18:
        diag.update({"applied": False, "reason": "no_exposure_overlap"})
        return sized, diag
    e_hat = exposure / e_sum

    Sigma = _shrunk_covariance(R, shrink=shrink).values
    mrc = Sigma @ e_hat
    contrib = np.clip(e_hat * mrc, 0.0, None)
    pos = contrib[contrib > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    med = max(med, 1e-18)
    contrib_norm = contrib / med
    mult = 1.0 / (1.0 + lam * contrib_norm)
    if max_shift > 0:
        mult = np.clip(mult, 1.0 - max_shift, 1.0 + max_shift)
    mult_by_und = {syms[i]: float(mult[i]) for i in range(len(syms))}

    new_g = np.zeros(len(df), dtype=float)
    for k, (u, w_pair) in enumerate(zip(df["Underlying"].tolist(), g.tolist())):
        m = float(mult_by_und.get(str(u), 1.0))
        new_g[k] = float(w_pair) * m
    s2 = float(new_g.sum())
    if s2 <= 1e-18:
        diag.update({"applied": False, "reason": "all_zero_after_penalty"})
        return sized, diag
    new_g = new_g * (gsum / s2)

    out = sized.copy()
    out["gross_target_usd"] = new_g

    out, _cap_diag = apply_gross_sizing_book_caps(
        out,
        target_gross_usd=float(target_gross_usd),
        beta_floor=float(beta_floor),
        strategy=strategy,
        shares_out_map=(shares_out_map or {}),
    )

    attenuated = sorted(
        ((u, float(m)) for u, m in mult_by_und.items() if m < 1.0 - 1e-9),
        key=lambda kv: kv[1],
    )[:5]
    diag.update(
        {
            "applied": True,
            "n_underlyings": int(len(syms)),
            "obs_used": int(len(R)),
            "lookback_used": int(min(len(R), lookback)),
            "shrink": float(shrink),
            "penalty_strength": float(lam),
            "max_relative_shift": float(max_shift),
            "gross_sum_before": gsum,
            "gross_sum_after": float(pd.to_numeric(out["gross_target_usd"], errors="coerce").fillna(0).sum()),
            "median_risk_contrib": float(med),
            "top_attenuated_underlyings": attenuated,
            "post_cov_cap_diag": _cap_diag,
        }
    )
    return out, diag


def apply_gross_sizing_book_caps(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    beta_floor: float,
    strategy: dict[str, Any],
    shares_out_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Enforce book-level max pair / max underlying weights and liquidity-style per-pair caps
    (AUM, shares available, shares outstanding, median daily volume). Normally preserves total
    allocated gross; if constraints cannot place all mass (e.g. a tight underlying cap on a
    single-name book), total gross is scaled down.

    Config: ``strategy.gross_sizing_caps`` in YAML. Omitted or ``enabled: false`` → no-op.
    """
    diag: dict[str, Any] = {"applied": False}
    if sized is None or sized.empty:
        return sized, diag
    raw = strategy.get("gross_sizing_caps")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return sized, diag
    caps = dict(raw)
    und_cap = float(caps.get("max_underlying_weight_cap", 0.11) or 0.11)
    som = shares_out_map if shares_out_map is not None else {}

    gross = pd.to_numeric(sized["gross_target_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gsum = float(gross.sum())
    if gsum <= 1e-18:
        return sized, diag

    pair_caps = _compute_pair_weight_caps_array(
        sized,
        target_gross_usd=float(target_gross_usd),
        beta_floor=float(beta_floor),
        caps=caps,
        shares_out_map=som,
    )
    desired = gross / gsum
    w1 = _project_to_capped_simplex_numpy(desired, pair_caps)
    und_code = pd.factorize(sized["Underlying"].astype(str).map(_norm_sym).to_numpy())[0]
    w2 = _project_pair_and_underlying_numpy(w1, pair_caps, und_code, float(und_cap))
    new_gross = w2 * gsum
    out = sized.copy()
    out["gross_target_usd"] = new_gross
    diag.update(
        {
            "applied": True,
            "gross_sum_before": gsum,
            "gross_sum_after": float(new_gross.sum()),
            "max_pair_weight_cap": float(caps.get("max_pair_weight_cap", 0.05) or 0.05),
            "max_underlying_weight_cap": float(und_cap),
        }
    )
    return out, diag


def load_blacklist(cfg: dict) -> Set[str]:
    raw = cfg.get("strategy", {}).get("blacklist", []) or []
    return {_norm_sym(sym) for sym in raw if str(sym).strip()}


def get_borrow_col(df: pd.DataFrame) -> str | None:
    """
    Try to infer an annual borrow column from screened_csv.
    Accepts common names; returns column name or None.
    """
    candidates = [
        "borrow_current", "Borrow_Current", "borrow_fee_annual", "Borrow_Fee_Annual",
        "borrow_annual", "Borrow_annual", "borrow", "Borrow",
        "borrow_rate", "borrowRate", "feerate", "fee_annual"
    ]
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def compute_borrow_annual_series(df: pd.DataFrame, borrow_col: str | None) -> pd.Series:
    """
    Return annual borrow as decimal. If missing, returns NaN series.
    If the borrow column looks like percent (e.g. 3.63), you should
    normalize upstream; here we assume decimal already.
    """
    if borrow_col is None:
        return pd.Series(index=df.index, data=np.nan, dtype=float)
    s = pd.to_numeric(df[borrow_col], errors="coerce")
    return s.astype(float)


# -----------------------------
# Flow ledger helpers
# -----------------------------
def load_flow_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker", "delta_usd", "cum_usd"])
    df = pd.read_csv(path)
    for c in ["date", "ticker"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in ["delta_usd", "cum_usd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_current_flow_cum(ledger: pd.DataFrame) -> Dict[str, float]:
    """
    Returns latest cumulative per ticker (cum_usd). If ledger empty, 0.
    """
    if ledger.empty:
        return {}
    # take last cum_usd per ticker by date order
    led = ledger.copy()
    led["date"] = pd.to_datetime(led["date"], errors="coerce")
    led = led.sort_values(["ticker", "date"])
    out = {}
    for t, g in led.groupby("ticker", sort=False):
        v = g["cum_usd"].dropna()
        out[str(t)] = float(v.iloc[-1]) if len(v) else 0.0
    return out


def append_flow_ledger(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows.to_csv(path, mode="a", header=False, index=False)
    else:
        rows.to_csv(path, index=False)


# -----------------------------
# Portfolio sizing logic
# -----------------------------
def hedge_ratio_from_beta(beta: float, beta_floor: float) -> float:
    b = float(beta) if np.isfinite(beta) else 1.0
    b_abs = max(abs(b), float(beta_floor))
    return 1.0 / b_abs


def size_pair_long_short(gross_usd: float, beta: float, beta_floor: float) -> Tuple[float, float]:
    """
    gross = long + |short| where short = -hedge_ratio * long
    """
    hr = hedge_ratio_from_beta(beta, beta_floor)
    long_usd = float(gross_usd) / (1.0 + hr)
    short_usd = -(hr * long_usd)
    return long_usd, short_usd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=today_str(), help="YYYY-MM-DD for data/runs/<date>/ outputs")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    core_decay_state_path = Path(
        paths.get("core_leveraged_decay_state_json", "data/core_leveraged_decay_state.json")
    )
    strategy = cfg["strategy"]
    sleeves = cfg.get("portfolio", {}).get("sleeves", {})

    screened_csv = Path(paths["screened_csv"])
    proposed_latest_csv = Path(paths["proposed_trades_csv"])

    tag = str(strategy.get("tag", "")).strip() or "strategy"
    blacklist = load_blacklist(cfg)

    # Key sizing params
    capital_usd = float(strategy["capital_usd"])
    gross_leverage = float(strategy["gross_leverage"])
    target_gross_usd = capital_usd * gross_leverage
    beta_floor = float(strategy.get("beta_floor", 0.1))

    # Sleeves
    core = sleeves.get("core_leveraged", {})
    wl   = sleeves.get("whitelist_stock", {})
    b4   = sleeves.get("inverse_decay_bucket4", {})
    flow = sleeves.get("flow_program", {})

    core_w = float(core.get("target_weight", 0.85))
    wl_w   = float(wl.get("target_weight", 0.15))
    b4_w   = float(b4.get("target_weight", 0.0))
    b4_enabled = bool(b4.get("enabled", True))

    core_rules = core.get("rules", {}) or {}
    core_beta_min = float(core_rules.get("min_beta_used", 1.5))
    b4_rules = b4.get("rules", {}) or {}
    b4_min_edge = float(b4_rules.get("min_net_edge_annual", 0.0))
    b4_partial_hedge_ratio = _clamp01(b4_rules.get("partial_hedge_ratio", 1.0))
    b4_max_shares_outstanding_frac = _clamp01(b4_rules.get("max_shares_outstanding_frac", 0.20))
    # Universe-entry floor on underlying realized volatility (annualized).
    b4_min_underlying_vol = float(b4_rules.get("min_underlying_vol", 0.50))
    b4_excluded_etfs = {_norm_sym(x) for x in (b4_rules.get("excluded_etfs") or [])}

    # Borrow caps (soft vs hard)
    soft_borrow_cap = float(cfg.get("screener", {}).get("borrow_low", 1.0))  # e.g. 0.08
    wl_hard_borrow_cap = float(wl.get("rules", {}).get("hard_borrow_cap", np.inf))
    # Dedicated Bucket 4 borrow cap (supports legacy hard_borrow_cap key).
    b4_hard_borrow_cap = float(
        b4_rules.get("bucket4_borrow_cap", b4_rules.get("hard_borrow_cap", np.inf))
    )
    flow_hard_borrow_cap = float(flow.get("rules", {}).get("hard_borrow_cap", np.inf))

    # Whitelist list order
    wl_list = [_norm_sym(x) for x in (wl.get("universe", {}).get("etfs", []) or [])]
    wl_set = set(wl_list)
    zipf_exp = float(wl.get("weighting", {}).get("zipf_exponent", 1.0))

    # Weighting configs (full dicts, consumed by _decay_score_weights)
    core_weighting_cfg = core.get("weighting", {})
    wl_weighting_cfg   = wl.get("weighting", {})
    b4_weighting_cfg   = b4.get("weighting", {})
    core_weight_method = str(core_weighting_cfg.get("method", "equal")).lower()
    wl_weight_method   = str(wl_weighting_cfg.get("method", "decay_score")).lower()
    b4_weight_method   = str(b4_weighting_cfg.get("method", "decay_score")).lower()

    # Flow config
    flow_shorts = [_norm_sym(x) for x in (flow.get("universe", {}).get("shorts", []) or [])]
    flow_weights_raw = (flow.get("weighting", {}) or {}).get("weights", {}) or {}
    flow_weights = { _norm_sym(k): float(v) for k, v in flow_weights_raw.items() if np.isfinite(v) and float(v) > 0 }
    if flow_shorts:
        missing = [s for s in flow_shorts if s not in flow_weights]
        if missing:
            raise ValueError(f"flow_program shorts missing weights: {missing}")
        # normalize
        ssum = sum(flow_weights[s] for s in flow_shorts)
        flow_weights = {s: flow_weights[s] / ssum for s in flow_shorts}

    annual_deploy = float(flow.get("annual_deployment_rate", 0.0))
    deploy_base = str(flow.get("deployment_base", "current_equity")).lower()

    # Flow ledger paths (new)
    flow_ledger_path = Path(paths.get("flow_ledger_csv", "data/flow_ledger.csv"))

    shares_out_map, shares_src = load_shares_outstanding_map(paths)
    print(
        f"[INFO] target_gross_usd=${target_gross_usd:,.0f} | "
        f"core={core_w:.0%} wl={wl_w:.0%} b4={b4_w:.0%} (enabled={b4_enabled}) | beta_floor={beta_floor}"
    )
    print(
        f"[INFO] core soft borrow cap={soft_borrow_cap:.1%} | "
        f"wl hard cap={wl_hard_borrow_cap if np.isfinite(wl_hard_borrow_cap) else 'inf'} | "
        f"b4 hard cap={b4_hard_borrow_cap if np.isfinite(b4_hard_borrow_cap) else 'inf'} | "
        f"flow hard cap={flow_hard_borrow_cap if np.isfinite(flow_hard_borrow_cap) else 'inf'}"
    )
    print(
        f"[INFO] b4 universe filters: min_underlying_vol={b4_min_underlying_vol:.0%} | "
        f"min_net_edge_annual={b4_min_edge:.0%}"
    )
    if b4_excluded_etfs:
        print(f"[INFO] b4 excluded_etfs ({len(b4_excluded_etfs)}): {', '.join(sorted(b4_excluded_etfs))}")
    _core_nd_min = float(core_rules.get("min_net_decay_annual", 0.0) or 0.0)
    _core_hyst = core_rules.get("net_decay_hysteresis") or {}
    if _core_nd_min > 0 or bool(_core_hyst.get("enabled", False)):
        print(
            f"[INFO] core (bucket-1) net-decay selectivity: min_net_decay_annual={_core_nd_min:.2%} | "
            f"hysteresis_enabled={bool(_core_hyst.get('enabled', False))} | "
            f"state_file={core_decay_state_path}"
        )
        if bool(_core_hyst.get("enabled", False)):
            print(
                f"[INFO]   hysteresis enter_above={float(_core_hyst.get('enter_above', 0)):.2%} "
                f"exit_below={float(_core_hyst.get('exit_below', 0)):.2%}"
            )
    if not b4_enabled:
        print("[INFO] inverse_decay_bucket4 disabled — no B4 allocation or targets in proposed trades")
    print(f"[INFO] whitelist size={len(wl_list)} | flow shorts={len(flow_shorts)}")
    print(f"[INFO] weighting: core={core_weight_method} wl={wl_weight_method} b4={b4_weight_method}")
    if shares_out_map:
        print(
            f"[INFO] bucket4 shares-outstanding cap enabled: "
            f"{b4_max_shares_outstanding_frac:.0%} of float "
            f"(source={shares_src})"
        )
    else:
        print("[WARN] bucket4 shares-outstanding source not found/usable; cap enforcement skipped.")

    if not screened_csv.exists():
        raise FileNotFoundError(f"Screened CSV not found: {screened_csv}")

    screened = pd.read_csv(screened_csv)
    if screened.empty:
        print("[WARN] Screened universe is empty.")
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_latest_csv, index=False)
        return

    required_cols = {"ETF", "Underlying", "purgatory", "Beta"}
    missing = required_cols - set(screened.columns)
    if missing:
        raise ValueError(f"Screened CSV missing required columns: {sorted(missing)}. Found: {list(screened.columns)}")

    # Normalize tickers
    screened["ETF"] = screened["ETF"].astype(str).map(_norm_sym)
    screened["Underlying"] = screened["Underlying"].astype(str).map(_norm_sym)

    # Blacklist filter
    screened = screened[(~screened["Underlying"].isin(blacklist)) & (~screened["ETF"].isin(blacklist))].copy()
    if screened.empty:
        print("[WARN] No eligible pairs after blacklist filtering.")
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_latest_csv, index=False)
        return

    screened["Beta"] = pd.to_numeric(screened["Beta"], errors="coerce")
    screened["beta_abs"] = screened["Beta"].abs()

    # Coerce decay columns for weighting (may be absent in older CSVs)
    for _col in ("blended_gross_decay", "borrow_current", "net_decay_annual"):
        if _col not in screened.columns:
            screened[_col] = np.nan
        else:
            screened[_col] = pd.to_numeric(screened[_col], errors="coerce")

    # Borrow if available
    borrow_col = get_borrow_col(screened)
    screened["borrow_annual"] = compute_borrow_annual_series(screened, borrow_col)
    # if borrow missing, borrow_annual is NaN, and caps won't filter (we treat NaN as "unknown -> allow")
    # if you prefer "unknown -> exclude", change the predicate below accordingly.

    # KEEP set: full screened rows. Final output is filtered to
    # non-zero targets plus purgatory rows.
    keep = screened.copy()

    # Initialize outputs on KEEP
    keep["strategy_tag"] = tag
    keep["long_usd"] = 0.0
    keep["short_usd"] = 0.0
    keep["underlying_target_usd"] = 0.0
    keep["etf_target_usd"] = 0.0
    keep["underlying_target_from_b12_usd"] = 0.0
    keep["underlying_target_from_b4_usd"] = 0.0
    keep["underlying_internalized_usd"] = 0.0
    keep["underlying_external_trade_usd"] = 0.0
    keep["sleeve"] = ""

    # SIZING set: not purgatory (borrow + net-edge soft band in daily_screener) and
    # net_edge_p50_annual >= 0 when present (NaN treated as ineligible).
    if "net_edge_p50_annual" in screened.columns:
        screened["net_edge_p50_annual"] = pd.to_numeric(
            screened["net_edge_p50_annual"], errors="coerce"
        )
        _ne_ok = screened["net_edge_p50_annual"].fillna(-1.0).ge(0.0)
    else:
        _ne_ok = pd.Series(True, index=screened.index)
    eligible = screened.loc[(screened["purgatory"] != True) & _ne_ok].copy()  # noqa: E712
    if eligible.empty:
        print("[WARN] No eligible rows to size (all rows are purgatory).")
        proposed = keep.copy()
        nonzero_mask = (proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)
        proposed = proposed[nonzero_mask | (proposed["purgatory"] == True)]  # noqa: E712

        cols_to_drop = ["Leverage", "ExpectedLeverage", "cagr_positive", "beta_abs"]
        proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns], errors="ignore")

        dated_path = run_dir(args.run_date) / "proposed_trades.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(dated_path, index=False)
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(proposed_latest_csv, index=False)
        print(f"[OK] Wrote proposed trades -> {dated_path}  (n={len(proposed)})")
        print(f"[OK] Updated latest proposed trades -> {proposed_latest_csv}  (n={len(proposed)})")
    else:
        # -----------------------------
        # Build sleeve membership
        # -----------------------------
        # core: |beta| >= min_beta_used AND passes soft borrow cap (unless also whitelist)
        eligible["in_whitelist"] = eligible["ETF"].isin(wl_set)
        # Hard rule: negative net decay names are excluded from stock sleeves (core + whitelist).
        net_decay_non_negative = ~(eligible["net_decay_annual"] < 0)

        # Borrow cap predicates
        # - soft cap applies to core ONLY when NOT whitelist
        # - whitelist has hard cap (20%) if provided
        # - for NaN borrow, allow (conservative in the sense of "don't accidentally drop new ETFs");
        #   if you want the opposite, flip the np.isfinite checks.
        b = eligible["borrow_annual"]

        # CORE: enforce soft borrow cap ONLY for non-whitelist names
        core_borrow_ok = (~np.isfinite(b)) | (b <= soft_borrow_cap) | (eligible["in_whitelist"])

        # WHITELIST: ALWAYS allow, independent of borrow
        wl_borrow_ok = True
        # (or: pd.Series(True, index=eligible.index))
        b4_borrow_ok = (~np.isfinite(b)) | (b <= b4_hard_borrow_cap)

        # Exclude inverse (β < 0) ETFs — they belong to the flow program, not core/whitelist
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
        # Require realized underlying vol above the configured floor. Missing vol is treated as
        # a rejection: we will not admit a pair whose underlying-vol we cannot measure.
        b4_vol_ok = np.isfinite(b4_und_vol) & (b4_und_vol >= b4_min_underlying_vol)
        core_pre_decay = (
            positive_beta
            & eligible["beta_abs"].ge(core_beta_min)
            & core_borrow_ok
            & net_decay_non_negative
        )
        core_neg_decay_reset = (
            positive_beta
            & eligible["beta_abs"].ge(core_beta_min)
            & core_borrow_ok
            & ~net_decay_non_negative
        )
        try:
            core_decay_gate = _core_net_decay_gate_for_core(
                eligible,
                core_pre_decay=core_pre_decay,
                core_neg_decay_reset=core_neg_decay_reset,
                net_decay_non_negative=net_decay_non_negative,
                core_rules=core_rules,
                state_path=core_decay_state_path,
                run_date=args.run_date,
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
        eligible["in_core"] = core_pre_decay & core_decay_gate
        eligible["in_wl"] = positive_beta & eligible["in_whitelist"] & wl_borrow_ok & net_decay_non_negative
        b4_not_excluded = ~eligible["ETF"].isin(b4_excluded_etfs)
        eligible["in_b4"] = (
            negative_beta & inverse_shortable & b4_borrow_ok & b4_edge_ok & b4_vol_ok & b4_not_excluded
        )

        core_names = eligible.loc[eligible["in_core"]].copy()
        wl_names   = eligible.loc[eligible["in_wl"]].copy()
        b4_names   = eligible.loc[eligible["in_b4"]].copy()
        if not b4_enabled:
            b4_names = eligible.loc[[]].copy()
        n_neg_decay_excluded = int((~net_decay_non_negative).sum())
        if n_neg_decay_excluded:
            print(f"[INFO] Excluded {n_neg_decay_excluded} names with negative net_decay_annual from stock sleeves.")
        n_core_decay_blocked = int((core_pre_decay & ~core_decay_gate).sum())
        if n_core_decay_blocked:
            print(
                f"[INFO] Excluded {n_core_decay_blocked} core (bucket-1) candidate row(s) "
                f"via min_net_decay_annual / hysteresis gate."
            )

        # Budgeting:
        # - Bucket 4 gets an explicit % of total plan when active.
        # - Remaining budget is split across core/wl by relative sleeve weights.
        b4_budget = target_gross_usd * b4_w if (not b4_names.empty and b4_w > 0) else 0.0
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

        # -----------------------------
        # Allocate CORE
        # -----------------------------
        if not core_names.empty and core_budget > 0:
            if core_weight_method == "decay_score":
                w = _decay_score_weights(core_names, core_weighting_cfg, sleeve_name="core_leveraged")
            else:   # "equal" or unrecognised → equal weight
                w = np.ones(len(core_names)) / len(core_names)
            core_names["gross_target_usd"] = core_budget * w
            core_names["sleeve"] = "core_leveraged"

        # -----------------------------
        # Allocate WHITELIST
        # -----------------------------
        if not wl_names.empty and wl_budget > 0:
            if wl_weight_method == "decay_score":
                w = _decay_score_weights(wl_names, wl_weighting_cfg, sleeve_name="whitelist_stock")
            else:   # "zipf" (default) or "equal"
                wl_names["wl_rank"] = wl_names["ETF"].map(
                    lambda x: wl_list.index(x) if x in wl_set else 10**9)
                wl_names = wl_names.sort_values("wl_rank").copy()
                if wl_weight_method == "equal":
                    w = np.ones(len(wl_names)) / len(wl_names)
                else:
                    w = _zipf_weights(len(wl_names), exponent=zipf_exp)
            wl_names["gross_target_usd"] = wl_budget * w
            wl_names["sleeve"] = "whitelist_stock"

        # -----------------------------
        # Allocate BUCKET 4
        # -----------------------------
        if not b4_names.empty and b4_budget > 0:
            if b4_weight_method == "equal":
                w = np.ones(len(b4_names)) / len(b4_names)
            else:
                w = _decay_score_weights(b4_names, b4_weighting_cfg, sleeve_name="inverse_decay_bucket4")
            b4_names["gross_target_usd"] = b4_budget * w
            # Cap bucket-4 ETF notionals by shares outstanding and reference price.
            b4_names["shares_outstanding_total"] = b4_names["ETF"].map(shares_out_map)
            b4_names["price_ref"] = pd.to_numeric(
                b4_names.get("borrow_price_ref", np.nan), errors="coerce"
            )
            b4_names["gross_target_cap_usd"] = (
                b4_max_shares_outstanding_frac
                * pd.to_numeric(b4_names["shares_outstanding_total"], errors="coerce")
                * pd.to_numeric(b4_names["price_ref"], errors="coerce")
            )
            if shares_out_map:
                before_sum = float(pd.to_numeric(b4_names["gross_target_usd"], errors="coerce").fillna(0.0).sum())
                b4_names["gross_target_usd"] = _apply_notional_caps_with_redistribution(
                    b4_names["gross_target_usd"],
                    b4_names["gross_target_cap_usd"],
                )
                after_sum = float(pd.to_numeric(b4_names["gross_target_usd"], errors="coerce").fillna(0.0).sum())
                if after_sum + 1e-6 < before_sum:
                    print(
                        "[WARN] bucket4 shares-outstanding caps constrained allocated notional: "
                        f"requested=${before_sum:,.0f}, capped=${after_sum:,.0f}"
                    )
            b4_names["sleeve"] = "inverse_decay_bucket4"

        sized = pd.concat([core_names, wl_names, b4_names], axis=0, ignore_index=False)
        sized = sized[~sized.index.duplicated(keep="first")].copy()

        sized, _cap_diag = apply_gross_sizing_book_caps(
            sized,
            target_gross_usd=float(target_gross_usd),
            beta_floor=float(beta_floor),
            strategy=strategy,
            shares_out_map=shares_out_map,
        )
        if _cap_diag.get("applied"):
            print(
                "[INFO] gross_sizing_caps: "
                f"pair_cap={_cap_diag.get('max_pair_weight_cap'):.1%} "
                f"under_cap={_cap_diag.get('max_underlying_weight_cap'):.1%} "
                f"gross_before=${_cap_diag.get('gross_sum_before', 0):,.0f} "
                f"gross_after=${_cap_diag.get('gross_sum_after', 0):,.0f}"
            )

        cov_cfg = strategy.get("covariance_balance") or {}
        if isinstance(cov_cfg, dict) and bool(cov_cfg.get("enabled", False)):
            ur_csv = paths.get("underlying_returns_csv") if isinstance(paths, dict) else None
            ur_df: pd.DataFrame | None = None
            if isinstance(ur_csv, str) and ur_csv.strip():
                ur_path = Path(ur_csv)
                if ur_path.exists():
                    try:
                        ur_df = pd.read_csv(ur_path, index_col=0)
                    except Exception:
                        ur_df = None
            sized, _cov_diag = apply_covariance_balance(
                sized,
                target_gross_usd=float(target_gross_usd),
                beta_floor=float(beta_floor),
                strategy=strategy,
                paths=paths,
                shares_out_map=shares_out_map,
                returns_df=ur_df,
            )
            if _cov_diag.get("applied"):
                top = ", ".join(f"{u}(x{m:.2f})" for u, m in (_cov_diag.get("top_attenuated_underlyings") or []))
                print(
                    f"[INFO] covariance_balance: shrink={_cov_diag.get('shrink'):.2f} "
                    f"lambda={_cov_diag.get('penalty_strength'):.2f} "
                    f"n_und={_cov_diag.get('n_underlyings')} obs={_cov_diag.get('obs_used')} "
                    f"attenuated=[{top}] gross_after=${_cov_diag.get('gross_sum_after', 0):,.0f}"
                )
            else:
                print(f"[INFO] covariance_balance: skipped ({_cov_diag.get('reason', 'disabled')})")

        # Now compute long/short for each sized row
        sized["beta_used_abs"] = sized["beta_abs"].clip(lower=beta_floor).fillna(1.0)
        sized["hedge_ratio"] = 1.0 / sized["beta_used_abs"]
        b4_mask = sized["sleeve"].eq("inverse_decay_bucket4")
        stock_mask = ~b4_mask

        sized.loc[stock_mask, "long_usd"] = sized.loc[stock_mask, "gross_target_usd"] / (
            1.0 + sized.loc[stock_mask, "hedge_ratio"]
        )
        sized.loc[stock_mask, "short_usd"] = -(
            sized.loc[stock_mask, "hedge_ratio"] * sized.loc[stock_mask, "long_usd"]
        )

        # Bucket 4: short inverse ETF and short underlying hedge.
        sized.loc[b4_mask, "short_usd"] = -sized.loc[b4_mask, "gross_target_usd"]
        sized.loc[b4_mask, "long_usd"] = -(
            b4_partial_hedge_ratio
            * sized.loc[b4_mask, "beta_used_abs"]
            * sized.loc[b4_mask, "gross_target_usd"]
        )
        sized["underlying_target_usd"] = sized["long_usd"]
        sized["etf_target_usd"] = sized["short_usd"]

        # Write sized notionals back into KEEP (purgatory remains 0)
        keep.loc[sized.index, "long_usd"] = sized["long_usd"]
        keep.loc[sized.index, "short_usd"] = sized["short_usd"]
        keep.loc[sized.index, "underlying_target_usd"] = sized["underlying_target_usd"]
        keep.loc[sized.index, "etf_target_usd"] = sized["etf_target_usd"]
        keep.loc[sized.index, "sleeve"] = sized["sleeve"]

        print(
            f"[INFO] sized core={len(core_names)} wl={len(wl_names)} b4={len(b4_names)} | "
            f"budgets: core=${core_budget:,.0f} wl=${wl_budget:,.0f} b4=${b4_budget:,.0f}"
        )

        # Weight diagnostics
        if core_weight_method == "decay_score" and not core_names.empty and core_budget > 0:
            cw = core_names["gross_target_usd"] / core_budget
            print(f"[INFO] core weights: max={cw.max():.3f} min={cw.min():.3f} "
                  f"nonzero={int((cw > 1e-9).sum())}/{len(core_names)}")
        if wl_weight_method == "decay_score" and not wl_names.empty and wl_budget > 0:
            ww = wl_names["gross_target_usd"] / wl_budget
            print(f"[INFO] wl weights: max={ww.max():.3f} min={ww.min():.3f} "
                  f"nonzero={int((ww > 1e-9).sum())}/{len(wl_names)}")
        if b4_weight_method == "decay_score" and not b4_names.empty and b4_budget > 0:
            bw = b4_names["gross_target_usd"] / b4_budget
            print(
                f"[INFO] b4 weights: max={bw.max():.3f} min={bw.min():.3f} "
                f"nonzero={int((bw > 1e-9).sum())}/{len(b4_names)}"
            )

        # Internalization diagnostics by underlying.
        if not keep.empty:
            b12_under = (
                keep[keep["sleeve"].isin(["core_leveraged", "whitelist_stock"])]
                .groupby("Underlying", as_index=False)["long_usd"]
                .sum()
                .rename(columns={"long_usd": "underlying_target_from_b12_usd"})
            )
            b4_under = (
                keep[keep["sleeve"].eq("inverse_decay_bucket4")]
                .groupby("Underlying", as_index=False)["long_usd"]
                .sum()
                .rename(columns={"long_usd": "underlying_target_from_b4_usd"})
            )
            by_under = (
                keep[["Underlying"]]
                .drop_duplicates()
                .merge(b12_under, on="Underlying", how="left")
                .merge(b4_under, on="Underlying", how="left")
            )
            by_under["underlying_target_from_b12_usd"] = pd.to_numeric(
                by_under["underlying_target_from_b12_usd"], errors="coerce"
            ).fillna(0.0)
            by_under["underlying_target_from_b4_usd"] = pd.to_numeric(
                by_under["underlying_target_from_b4_usd"], errors="coerce"
            ).fillna(0.0)
            by_under["underlying_internalized_usd"] = np.minimum(
                by_under["underlying_target_from_b12_usd"].clip(lower=0.0),
                (-by_under["underlying_target_from_b4_usd"]).clip(lower=0.0),
            )
            by_under["underlying_external_trade_usd"] = (
                by_under["underlying_target_from_b12_usd"] + by_under["underlying_target_from_b4_usd"]
            )
            keep = keep.merge(by_under, on="Underlying", how="left", suffixes=("", "_calc"))
            for col in (
                "underlying_target_from_b12_usd",
                "underlying_target_from_b4_usd",
                "underlying_internalized_usd",
                "underlying_external_trade_usd",
            ):
                calc_col = f"{col}_calc"
                if calc_col in keep.columns:
                    keep[col] = pd.to_numeric(keep[calc_col], errors="coerce").fillna(0.0)
                    keep = keep.drop(columns=[calc_col])

        # Output proposed trades:
        proposed = keep.copy()
        nonzero_mask = (proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)
        proposed = proposed[nonzero_mask | (proposed["purgatory"] == True)]  # noqa: E712

        cols_to_drop = ["Leverage", "ExpectedLeverage", "cagr_positive", "beta_abs"]
        proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns], errors="ignore")

        dated_path = run_dir(args.run_date) / "proposed_trades.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(dated_path, index=False)
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(proposed_latest_csv, index=False)

        print(f"[OK] Wrote proposed trades -> {dated_path}  (n={len(proposed)})")
        print(f"[OK] Updated latest proposed trades -> {proposed_latest_csv}  (n={len(proposed)})")

    # =========================================================
    # FLOW PROGRAM TRACKING (cumulative weekly adds)
    # =========================================================
    # We *track* desired cumulative notional shorts; execution layer can implement deltas.
    # weekly_add_usd is based on deployment_base.
    equity_base = capital_usd if deploy_base != "current_equity" else capital_usd  # (no live equity here)
    weekly_add_usd = (annual_deploy / 52.0) * float(equity_base)

    # Load ledger and compute current cum
    ledger = load_flow_ledger(flow_ledger_path)
    cur_cum = compute_current_flow_cum(ledger)

    # Build this week's deltas (respect hard cap if we have borrow_annual info for those tickers in screened)
    # If you want borrow-based filtering for flow, you need a borrow source for those tickers.
    # Here: if the ticker exists in screened borrow_annual column, use it; else treat as NaN and allow.
    screened_borrow_map = {}
    if "borrow_annual" in screened.columns:
        tmp = screened[["ETF", "borrow_annual"]].dropna()
        screened_borrow_map = { _norm_sym(r["ETF"]): float(r["borrow_annual"]) for _, r in tmp.iterrows() }

    edge_p50_map: dict[str, float] = {}
    if "net_edge_p50_annual" in screened.columns:
        _tmp = screened[["ETF", "net_edge_p50_annual"]].copy()
        _tmp["ETF"] = _tmp["ETF"].astype(str).map(_norm_sym)
        _tmp["net_edge_p50_annual"] = pd.to_numeric(_tmp["net_edge_p50_annual"], errors="coerce")
        for sym, grp in _tmp.groupby("ETF"):
            v = float(grp["net_edge_p50_annual"].iloc[0])
            if np.isfinite(v):
                edge_p50_map[sym] = v

    flow_rows = []
    for s in flow_shorts:
        w = float(flow_weights.get(s, 0.0))
        delta = weekly_add_usd * w

        b_ann = screened_borrow_map.get(s, np.nan)
        if np.isfinite(flow_hard_borrow_cap) and np.isfinite(b_ann) and (b_ann > flow_hard_borrow_cap):
            # skip adds if we know it's above the cap
            delta = 0.0
        p50 = edge_p50_map.get(s, np.nan)
        if np.isfinite(p50) and float(p50) < 0.0:
            delta = 0.0

        prev = float(cur_cum.get(s, 0.0))
        new  = prev + float(delta)

        flow_rows.append({"ticker": s, "delta_usd": float(delta), "cum_usd": float(new)})

    flow_df = pd.DataFrame(flow_rows)
    flow_df.insert(0, "date", args.run_date)

    # Append ledger (only nonzero deltas to keep it clean)
    to_append = flow_df.loc[flow_df["delta_usd"].abs() > 1e-9].copy()
    if not to_append.empty:
        append_flow_ledger(flow_ledger_path, to_append)

    print(f"[OK] Flow tracking: weekly_add=${weekly_add_usd:,.2f} | flow targets snapshot deprecated (ledger-only)")
    print(f"[OK] Flow ledger: {flow_ledger_path} (appended {len(to_append)} rows)")


if __name__ == "__main__":
    main()
