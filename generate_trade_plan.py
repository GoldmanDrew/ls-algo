#!/usr/bin/env python3
"""
generate_trade_plan.py

Production portfolio construction for YAML sleeves.

Implements:
- ``core_leveraged`` sleeve: high-beta bucket-1 positives only (**excludes** ``is_yieldboost``
  rows; those size in ``yieldboost``).
    * Post-B4 gross is split vs ``yieldboost`` using normalized ``target_weight`` on each sleeve.
- ``yieldboost`` sleeve: bucket‑2 / YieldBoost only (`is_yieldboost`), gated by
  ``portfolio.sleeves.yieldboost.rules.min_net_edge_annual`` on ``net_edge_p50_annual``.

- Bucket 4: ``inverse_decay_bucket4`` unchanged.

- Purgatory handling (from screened CSV; see daily_screener ``recompute_purgatory_by_bucket``):
    * OUTPUT purgatory rows with 0 targets so execution won’t auto-close.
    * Does NOT allocate new exposure to purgatory (borrow soft band OR net_edge_p50 in 0–5%).
    * Sizing set also requires ``net_edge_p50_annual >= 0`` when that column exists.

- inverse_decay_bucket4: optional ``enabled: false`` on the sleeve — disables all B4 targets; stock
  sleeve absorbs the full gross budget.

- ``core_leveraged`` bucket-1 path: optional ``min_net_decay_annual`` and ``net_decay_hysteresis`` in
  YAML — tighter net-decay selectivity for **high-beta core rows only**. If ``min_net_decay_annual``
  > 0 it is a **hard floor** (always enforced, including over hysteresis and missing-data paths).
  Hysteresis uses ``paths.core_leveraged_decay_state_json`` to reduce pairs bouncing in/out when decay oscillates.

Outputs:
- proposed_trades.csv (stock sleeves sized + purgatory keep-open rows)

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
from typing import Any, Set, Tuple

import numpy as np
import pandas as pd

from strategy_config import load_config

TRADING_DAYS = 252


# -----------------------------
# Basic helpers
# -----------------------------
def today_str() -> str:
    return date.today().isoformat()


def run_dir(run_date: str) -> Path:
    return Path("data") / "runs" / run_date


def _norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize_two_nonnegative_weights(a: float, b: float) -> tuple[float, float]:
    """Return (frac_a, frac_b) that sum to 1; if both zero, (1, 0)."""
    aa = max(0.0, float(a))
    bb = max(0.0, float(b))
    s = aa + bb
    if s <= 1e-18:
        return 1.0, 0.0
    return aa / s, bb / s


def _b2_b4_universe_masks(
    eligible: pd.DataFrame,
    *,
    flow_program_etfs: set[str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the B2/B4 universe gating masks shared by ``main()`` and the sizing mirror.

    Returns
    -------
    is_yieldboost : pd.Series[bool]
        Per-row screener flag (``False`` when the column is absent).
    in_b2_universe : pd.Series[bool]
        ``is_yieldboost`` only (bucket‑2 stock candidates for ``generate_trade_plan``).
    in_flow_program : pd.Series[bool]
        Rows whose ``ETF`` ticker lives in ``flow_program.universe.shorts`` —
        excluded from both B2 and B4 because the weekly flow sleeve sizes them.
    """
    if "is_yieldboost" in eligible.columns:
        is_yieldboost = eligible["is_yieldboost"].fillna(False).astype(bool)
    else:
        is_yieldboost = pd.Series(False, index=eligible.index)
    in_b2_universe = is_yieldboost
    in_flow_program = (
        eligible["ETF"].isin(flow_program_etfs)
        if "ETF" in eligible.columns
        else pd.Series(False, index=eligible.index)
    )
    return is_yieldboost, in_b2_universe, in_flow_program


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
    be gated off; all other rows return pass-through True so non-core rows do not corrupt
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



def _sizing_signal_column(weighting_cfg: dict, *, sleeve_name: str | None = None) -> tuple[str, str]:
    """
    Return (mode, column_name) for decay_score sizing.

    *mode* is ``blended_decay`` or ``net_edge`` (affects fallbacks only). The default primary
    signal for **all** sleeves is ``net_edge_p50_annual``. Override via ``sizing_edge_column``.
    """
    mode = str(weighting_cfg.get("sizing_signal", "blended_decay") or "blended_decay").lower().strip()
    if mode in ("blended", "blended_decay", "decay", "gross_decay"):
        mode = "blended_decay"
    elif mode in ("net_edge", "edge", "net"):
        mode = "net_edge"
    else:
        mode = "blended_decay"

    col = weighting_cfg.get("sizing_edge_column")
    if isinstance(col, str) and col.strip():
        return mode, col.strip()

    return mode, "net_edge_p50_annual"


def _b4_eligibility_edge_column(df: pd.DataFrame) -> str:
    """Screener column for B4 ``min_net_edge_annual`` gate (inverse_decay_bucket4)."""
    if "net_edge_p50_annual" in df.columns:
        return "net_edge_p50_annual"
    if "bucket4_net_edge_annual" in df.columns:
        return "bucket4_net_edge_annual"
    return "net_decay_annual"


def _decay_score_weights(
    df: pd.DataFrame,
    weighting_cfg: dict,
    beta_col: str = "beta_abs",
    sleeve_name: str | None = None,
    *,
    pair_sigma_map: dict[tuple[str, str], float] | None = None,
    score_ema_state: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Compute portfolio weights from decay-score signal blended with equal weight.

    Parameters
    ----------
    df : DataFrame of eligible names. Must contain *beta_col* and borrow ``borrow_current``.
         Primary signal column defaults to ``net_edge_p50_annual`` for every sleeve (override
         with ``sizing_edge_column``). If that column is missing or all-NaN, falls back to
         ``blended_gross_decay``. Same borrow discount on the resolved signal in all modes.
    weighting_cfg : sleeve ``weighting`` dict from strategy_config.yml.
    beta_col : column name for absolute-value beta (default ``beta_abs``).
    sleeve_name : optional sleeve key (e.g. ``inverse_decay_bucket4``) for net-edge column defaults.
    pair_sigma_map : optional dict ``{(etf, und): annualized pair-spread sigma}`` enabling sigma-aware
         sizing (Candidate B): score is multiplied by ``clip(1 / max(sigma, sigma_floor), [m_min, m_max])``.
    score_ema_state : optional in-place dict ``{(etf, und): prev_score}``. When supplied with
         ``score_ema_rho`` > 0, current scores are blended ``rho * prev + (1 - rho) * cur`` and the
         dict is updated. Stabilizes weights against week-to-week net_edge wiggle (production use).

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
    # New iteration knobs (default: identity / off so baseline behavior is unchanged).
    score_p         = float(weighting_cfg.get("score_concavity_p", 1.0) or 1.0)
    sigma_aware_on  = bool(weighting_cfg.get("sigma_aware_sizing", False))
    sigma_floor     = float(weighting_cfg.get("sigma_aware_floor", 0.25) or 0.25)
    sigma_clip_min  = float(weighting_cfg.get("sigma_aware_min_mult", 0.5) or 0.5)
    sigma_clip_max  = float(weighting_cfg.get("sigma_aware_max_mult", 2.0) or 2.0)
    ema_rho         = _clamp01(weighting_cfg.get("score_ema_rho", 0.0))

    # --- raw sizing score (same borrow discount as blended decay path) ---
    _mode, sig_col = _sizing_signal_column(weighting_cfg, sleeve_name=sleeve_name)
    if sig_col in df.columns:
        signal = pd.to_numeric(df[sig_col], errors="coerce")
    else:
        signal = pd.Series(np.nan, index=df.index)
    if bool(signal.isna().all()) and "blended_gross_decay" in df.columns:
        signal = pd.to_numeric(df["blended_gross_decay"], errors="coerce")

    borrow = pd.to_numeric(df["borrow_current"], errors="coerce").fillna(0.0)
    raw_score = signal - borrow_aversion * borrow  # higher = better (annual units)

    # --- Candidate A: signed-power concavity on the raw score -------------
    # score = sign(raw) * |raw|^p; p in (0, 1] de-peaks the top tail without flipping signs.
    if abs(score_p - 1.0) > 1e-9:
        rs = pd.to_numeric(raw_score, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        raw_score = pd.Series(np.sign(rs) * np.power(np.abs(rs), float(score_p)), index=df.index)

    # --- Candidate B: sigma-aware sizing (1 / pair-spread sigma) ----------
    if sigma_aware_on and pair_sigma_map:
        try:
            etfs = df["ETF"].astype(str).map(_norm_sym).tolist()
            unds = df["Underlying"].astype(str).map(_norm_sym).tolist()
            inv = np.array(
                [
                    1.0
                    / max(
                        float(pair_sigma_map.get((e, u), float("nan"))),
                        float(sigma_floor),
                    )
                    if np.isfinite(float(pair_sigma_map.get((e, u), float("nan"))))
                    else 1.0
                    for e, u in zip(etfs, unds)
                ],
                dtype=float,
            )
            # Center around 1 (median(inv)) before clipping so the multiplier averages to 1.
            med_inv = float(np.nanmedian(inv)) if np.isfinite(np.nanmedian(inv)) else 1.0
            mult = inv / max(med_inv, 1e-9)
            mult = np.clip(mult, float(sigma_clip_min), float(sigma_clip_max))
            raw_score = raw_score.fillna(0.0) * pd.Series(mult, index=df.index)
        except Exception:
            pass

    # --- Stability #1: EMA blend with prior per-pair scores (production) --
    if ema_rho > 0 and score_ema_state is not None:
        try:
            etfs = df["ETF"].astype(str).map(_norm_sym).tolist()
            unds = df["Underlying"].astype(str).map(_norm_sym).tolist()
            cur = raw_score.fillna(0.0).to_numpy(dtype=float)
            prev = np.array(
                [float(score_ema_state.get((e, u), cur[i])) for i, (e, u) in enumerate(zip(etfs, unds))],
                dtype=float,
            )
            blended = ema_rho * prev + (1.0 - ema_rho) * cur
            # Update state in place (only for rows we just saw).
            for i, (e, u) in enumerate(zip(etfs, unds)):
                score_ema_state[(e, u)] = float(blended[i])
            raw_score = pd.Series(blended, index=df.index)
        except Exception:
            pass

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


def _liquidity_tight_book_weights(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    beta_floor: float,
    caps: dict[str, Any],
    shares_out_map: dict[str, float],
) -> np.ndarray:
    """
    Per-row upper bound on **book** weight ``gross_i / sum(gross)`` implied by liquidity
    inputs only (AUM, shares available, shares outstanding, median volume vs short leg).
    """
    n = len(sized)
    if n == 0:
        return np.zeros(0, dtype=float)
    T = max(float(target_gross_usd), 1.0)
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
    return np.clip(tight, 0.0, None).astype(float)


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
    max_pair = float(caps.get("max_pair_weight_cap", 0.05) or 0.05)
    tight = _liquidity_tight_book_weights(
        sized,
        target_gross_usd=target_gross_usd,
        beta_floor=beta_floor,
        caps=caps,
        shares_out_map=shares_out_map,
    )
    return np.minimum(max_pair, tight).astype(float)


def _enforce_max_pair_weight_within_deployed_sleeve_gross(
    gross: np.ndarray,
    sized: pd.DataFrame,
    sleeve_caps: dict[str, dict[str, float]],
) -> np.ndarray:
    """
    Re-scale **within each sleeve** so no pair holds more than ``max_pair_weight`` times that
    sleeve's **deployed** gross (sum of ``gross_target_usd`` over rows in the sleeve).

    Liquidity can leave most sleeve rows at ~0 while a few absorb the budget; normalizing by
    *deployed* mass avoids a single name showing 60%+ of placed gross while still being under
    the target-book pair cap. Mass is redistributed across rows that already have positive
    gross after liquidity (we do not revive names hard-zeroed by liquidity).

    Diversification **within a sleeve** requires at least two sized rows; single-row sleeves are
    unchanged (that pair is the sleeve). For multi-row sleeves with only one positive-gross row,
    gross is clipped to ``max_pair_weight * sleeve_deployed_sum`` when needed.
    """
    g = np.asarray(gross, dtype=float).copy()
    if "sleeve" not in sized.columns or not sleeve_caps:
        return g
    slv = sized["sleeve"].astype(str).to_numpy()
    for sleeve, meta_cap in sleeve_caps.items():
        cap = float((meta_cap or {}).get("max_pair_weight", 1.0))
        if cap >= 1.0 - 1e-12:
            continue
        idx = np.where(slv == str(sleeve))[0]
        if len(idx) <= 1:
            # One pair defines the whole sleeve; ``max_pair_weight`` is for diversity across rows.
            continue
        s_sleeve = float(np.sum(g[idx]))
        if s_sleeve <= 1e-18:
            continue
        pos = idx[g[idx] > 1e-18]
        if len(pos) == 0:
            continue
        if len(pos) == 1:
            i = int(pos[0])
            max_g = cap * s_sleeve
            if g[i] > max_g + 1e-6:
                g[i] = max_g
            continue
        s_dep = float(g[pos].sum())
        if s_dep <= 1e-18:
            continue
        w = g[pos] / s_dep
        if float(np.max(w)) <= cap + 1e-8:
            continue
        n_p = len(pos)
        cap_eff = float(cap)
        # Capped simplex {w: sum w = 1, w_i <= cap} is non-empty iff n * cap >= 1.
        if n_p * cap_eff < 1.0 - 1e-10:
            cap_eff = 1.0 / float(n_p)
        w_new = _project_positive_weights_to_sum_one_hard_cap(w, cap_eff)
        g[pos] = w_new * s_dep
    return g


def _project_positive_weights_to_sum_one_hard_cap(w: np.ndarray, cap: float, *, tol: float = 1e-10) -> np.ndarray:
    """
    Find ``w_out`` with ``w_out.sum() == 1`` (or maximal feasible ``< 1`` if ``n * cap < 1``),
    ``0 <= w_out[i] <= cap``, preferring to stay close to proportional redistribution
    from an initial ``w``.

    Unlike :func:`_project_to_capped_simplex_numpy`, caps here are **hard** — they must not be
    relaxed toward an infeasibly concentrated desired vector.
    """
    w0 = np.asarray(w, dtype=float).copy()
    if len(w0) == 0:
        return w0
    s0 = float(np.sum(w0))
    if s0 <= 1e-18:
        n = len(w0)
        return np.full(n, 1.0 / max(n, 1), dtype=float)
    w0 /= s0
    cap_f = float(cap)
    w0 = np.clip(w0, 0.0, cap_f)
    n = len(w0)
    if n * cap_f < 1.0 - tol:
        return np.full(n, cap_f, dtype=float)
    for _ in range(max(4, n * 8)):
        rem = 1.0 - float(np.sum(w0))
        if abs(rem) <= tol:
            break
        if rem < 0:
            tot = float(np.sum(w0))
            if tot > 1e-18:
                w0 /= tot
            w0 = np.clip(w0, 0.0, cap_f)
            continue
        head = np.clip(cap_f - w0, 0.0, None)
        hs = float(np.sum(head))
        if hs <= 1e-18:
            break
        w0 += rem * (head / hs)
        w0 = np.clip(w0, 0.0, cap_f)
    s1 = float(np.sum(w0))
    if s1 >= 1.0 - tol:
        w0 /= max(s1, 1e-18)
    # else: infeasible to reach sum 1 under cap — keep sub‑unit mass (caller scales gross down)
    return np.clip(w0, 0.0, cap_f)


def _apply_gross_sizing_per_sleeve_book_caps(
    sized: pd.DataFrame,
    gross: np.ndarray,
    *,
    target_gross_usd: float,
    beta_floor: float,
    caps: dict[str, Any],
    shares_out_map: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Diamond-Creek-style **per-sleeve** concentration: ``max_pair_weight`` and
    ``max_underlying_weight`` apply to weights **within each sleeve's allocated gross**,
    while liquidity rows still cap **book** weights (same construction as
    :func:`_liquidity_tight_book_weights`). Preserves each sleeve's share of total gross.
    """
    meta: dict[str, Any] = {}
    n = len(sized)
    if n == 0:
        return gross, meta
    gsum = float(np.sum(gross))
    if gsum <= 1e-18:
        return gross, meta
    per_sleeve = caps.get("per_sleeve") or {}
    if not isinstance(per_sleeve, dict) or not per_sleeve:
        return gross, meta
    if "sleeve" not in sized.columns:
        return gross, meta

    liq_book = _liquidity_tight_book_weights(
        sized,
        target_gross_usd=target_gross_usd,
        beta_floor=beta_floor,
        caps=caps,
        shares_out_map=shares_out_map,
    )
    default_pair = float(caps.get("max_pair_weight_cap", 0.05) or 0.05)
    default_und = float(caps.get("max_underlying_weight_cap", 0.11) or 0.11)

    slv = sized["sleeve"].astype(str).to_numpy()
    w = gross / gsum
    w_out = np.zeros_like(w, dtype=float)

    sleeve_caps_out: dict[str, dict[str, float]] = {}
    for sleeve in sorted({str(x) for x in slv.tolist()}):
        idx = np.where(slv == sleeve)[0]
        s_b = float(w[idx].sum())
        if s_b <= 1e-18:
            continue
        rules = per_sleeve.get(sleeve)
        if isinstance(rules, dict):
            mp_raw = rules.get("max_pair_weight", default_pair)
            mu_raw = rules.get("max_underlying_weight", default_und)
            mp_f = float(mp_raw) if mp_raw is not None and float(mp_raw) > 0 else default_pair
            mu_f = float(mu_raw) if mu_raw is not None and float(mu_raw) > 0 else default_und
        else:
            mp_f, mu_f = default_pair, default_und
        sleeve_caps_out[str(sleeve)] = {"max_pair_weight": mp_f, "max_underlying_weight": mu_f}

        v = w[idx] / s_b
        liq_slice = liq_book[idx]
        if len(idx) == 1:
            # One active row holds the entire sleeve; ``max_pair_weight`` is a diversification
            # knob across pairs and does not apply below 100% of sleeve gross. Liquidity still can.
            cap0 = min(1.0, float(liq_slice[0]) / max(s_b, 1e-18))
            cap_v = np.array([max(cap0, 1e-18)], dtype=float)
        else:
            cap_v = np.minimum(mp_f, liq_slice / max(s_b, 1e-18))
            cap_v = np.clip(cap_v, 1e-18, None)

        und_sub = sized.iloc[idx]["Underlying"].astype(str).map(_norm_sym).to_numpy()
        und_code = pd.factorize(und_sub)[0]
        v1 = _project_to_capped_simplex_numpy(v, cap_v)
        # Same reasoning as ``cap_v`` for a lone pair: one underlying may carry 100% of sleeve gross.
        mu_eff = 1.0 if len(idx) == 1 else float(mu_f)
        v2 = _project_pair_and_underlying_numpy(v1, cap_v, und_code, mu_eff)
        w_out[idx] = v2 * s_b

    meta["per_sleeve_caps"] = sleeve_caps_out
    new_gross = w_out * gsum
    new_gross = _enforce_max_pair_weight_within_deployed_sleeve_gross(
        new_gross,
        sized,
        sleeve_caps_out,
    )
    return new_gross, meta


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
    need_set = {_norm_sym(str(u)) for u in needed_underlyings}
    # Subset columns **before** copy — avoids O(rows × full_universe) allocation when callers
    # pass the full Yahoo matrix (notebook grid / diagnostics).
    pick: list[Any] = []
    seen_nc: set[str] = set()
    for col in returns_df.columns:
        nc = _norm_sym(str(col))
        if nc not in need_set or nc in seen_nc:
            continue
        seen_nc.add(nc)
        pick.append(col)
    if len(pick) < 2:
        return None
    df = returns_df.loc[:, pick].copy()
    df.columns = [_norm_sym(str(c)) for c in pick]
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


def _apply_weight_hysteresis(
    sized: pd.DataFrame,
    new_gross: np.ndarray,
    *,
    prev_gross_by_pair: dict[tuple[str, str], float] | None,
    abs_threshold: float,
    rel_threshold: float,
) -> np.ndarray:
    """Stability #2: snap each pair's new gross to its prior value when the change is
    smaller than ``max(abs_threshold * sum_gross, rel_threshold * prev_gross)``.

    Operates in dollar space (gross_target_usd) so it composes with cap projections.
    Total book gross is rescaled back to the original ``new_gross.sum()`` after snapping
    so the projection stack downstream is unchanged.
    """
    if prev_gross_by_pair is None or not prev_gross_by_pair:
        return new_gross
    g = np.asarray(new_gross, dtype=float).copy()
    s_total = float(np.sum(g))
    if s_total <= 1e-18:
        return new_gross
    abs_t = float(abs_threshold) * s_total if abs_threshold and abs_threshold > 0 else 0.0
    rel_t = float(rel_threshold) if rel_threshold and rel_threshold > 0 else 0.0
    etfs = sized["ETF"].astype(str).map(_norm_sym).to_numpy()
    unds = sized["Underlying"].astype(str).map(_norm_sym).to_numpy()
    snapped = g.copy()
    for i, (e, u) in enumerate(zip(etfs, unds)):
        prev = float(prev_gross_by_pair.get((str(e), str(u)), float("nan")))
        if not np.isfinite(prev) or prev <= 0:
            continue
        delta = abs(g[i] - prev)
        if delta <= abs_t:
            snapped[i] = prev
            continue
        if rel_t > 0 and delta <= rel_t * prev:
            snapped[i] = prev
    s_new = float(np.sum(snapped))
    if s_new > 1e-18:
        snapped *= s_total / s_new
    return snapped


def apply_gross_sizing_book_caps(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    beta_floor: float,
    strategy: dict[str, Any],
    shares_out_map: dict[str, float] | None = None,
    prev_gross_by_pair: dict[tuple[str, str], float] | None = None,
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

    per_sleeve_cfg = caps.get("per_sleeve") or caps.get("per_bucket")
    use_per_sleeve = (
        isinstance(per_sleeve_cfg, dict)
        and bool(per_sleeve_cfg)
        and ("sleeve" in sized.columns)
    )

    abs_t = float(caps.get("weight_hysteresis_abs", 0.0) or 0.0)
    rel_t = float(caps.get("weight_hysteresis_rel", 0.0) or 0.0)
    apply_hyst = bool(prev_gross_by_pair) and (abs_t > 0 or rel_t > 0)

    if use_per_sleeve:
        new_gross, ps_meta = _apply_gross_sizing_per_sleeve_book_caps(
            sized,
            gross,
            target_gross_usd=float(target_gross_usd),
            beta_floor=float(beta_floor),
            caps=caps,
            shares_out_map=som,
        )
        if apply_hyst:
            new_gross = _apply_weight_hysteresis(
                sized,
                new_gross,
                prev_gross_by_pair=prev_gross_by_pair,
                abs_threshold=abs_t,
                rel_threshold=rel_t,
            )
        out = sized.copy()
        out["gross_target_usd"] = new_gross
        diag.update(
            {
                "applied": True,
                "gross_sum_before": gsum,
                "gross_sum_after": float(new_gross.sum()),
                "max_pair_weight_cap": float(caps.get("max_pair_weight_cap", 0.05) or 0.05),
                "max_underlying_weight_cap": float(und_cap),
                "per_sleeve_enforced": True,
                "per_sleeve_caps": ps_meta.get("per_sleeve_caps", {}),
                "hysteresis_applied": bool(apply_hyst),
            }
        )
        return out, diag

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
    if apply_hyst:
        new_gross = _apply_weight_hysteresis(
            sized,
            new_gross,
            prev_gross_by_pair=prev_gross_by_pair,
            abs_threshold=abs_t,
            rel_threshold=rel_t,
        )
    out = sized.copy()
    out["gross_target_usd"] = new_gross
    diag.update(
        {
            "applied": True,
            "gross_sum_before": gsum,
            "gross_sum_after": float(new_gross.sum()),
            "max_pair_weight_cap": float(caps.get("max_pair_weight_cap", 0.05) or 0.05),
            "max_underlying_weight_cap": float(und_cap),
            "per_sleeve_enforced": False,
            "hysteresis_applied": bool(apply_hyst),
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
    b4 = sleeves.get("inverse_decay_bucket4", {})
    flow = sleeves.get("flow_program", {})
    yb_sleeve = sleeves.get("yieldboost", {}) or {}

    b4_w = float(b4.get("target_weight", 0.0))
    b4_enabled = bool(b4.get("enabled", True))
    yb_enabled = bool(yb_sleeve.get("enabled", True))
    core_stock_tw = float(core.get("target_weight", 1.0))
    yb_stock_tw = float(yb_sleeve.get("target_weight", 0.0))
    core_stock_frac, yb_stock_frac = _normalize_two_nonnegative_weights(
        core_stock_tw,
        yb_stock_tw if yb_enabled else 0.0,
    )
    stock_nominal_w = (
        max(0.0, min(1.0, 1.0 - b4_w)) if b4_enabled else 1.0
    )

    core_rules = core.get("rules", {}) or {}
    core_beta_min = float(core_rules.get("min_beta_used", 1.5))
    yb_rules = yb_sleeve.get("rules", {}) or {}
    _yb_edge_yaml = yb_rules.get("min_net_edge_annual", None)
    if _yb_edge_yaml is None or (isinstance(_yb_edge_yaml, float) and not np.isfinite(_yb_edge_yaml)):
        yb_min_edge = float(core_rules.get("yieldboost_min_net_edge_annual", 0.0) or 0.0)
    else:
        yb_min_edge = float(_yb_edge_yaml or 0.0)
    b4_rules = b4.get("rules", {}) or {}
    b4_min_edge = float(b4_rules.get("min_net_edge_annual", 0.0))
    b4_partial_hedge_ratio = _clamp01(b4_rules.get("partial_hedge_ratio", 1.0))
    b4_max_shares_outstanding_frac = _clamp01(b4_rules.get("max_shares_outstanding_frac", 0.20))
    # Universe-entry floor on underlying realized volatility (annualized).
    b4_min_underlying_vol = float(b4_rules.get("min_underlying_vol", 0.50))
    b4_excluded_etfs = {_norm_sym(x) for x in (b4_rules.get("excluded_etfs") or [])}

    # Borrow caps (soft vs hard)
    soft_borrow_cap = float(cfg.get("screener", {}).get("borrow_low", 1.0))  # e.g. 0.08
    # Dedicated Bucket 4 borrow cap. Use ``bucket4_borrow_cap`` in YAML; old aliases are deprecated.
    b4_hard_borrow_cap = float(b4_rules.get("bucket4_borrow_cap", np.inf))
    flow_borrow_cap = float(flow.get("rules", {}).get("hard_borrow_cap", np.inf))

    # Weighting configs (full dicts, consumed by _decay_score_weights)
    core_weighting_cfg = core.get("weighting", {})
    b4_weighting_cfg = b4.get("weighting", {})
    yb_weighting_cfg = yb_sleeve.get("weighting", {}) or {}
    core_weight_method = str(core_weighting_cfg.get("method", "equal")).lower()
    b4_weight_method = str(b4_weighting_cfg.get("method", "decay_score")).lower()
    yb_weight_method = str(yb_weighting_cfg.get("method", "decay_score")).lower()

    # Flow config
    flow_shorts = [_norm_sym(x) for x in (flow.get("universe", {}).get("shorts", []) or [])]

    shares_out_map, shares_src = load_shares_outstanding_map(paths)
    print(
        f"[INFO] target_gross_usd=${target_gross_usd:,.0f} | "
        f"post_b4_stock(nominal)={stock_nominal_w:.0%} "
        f"(core_frac={core_stock_frac:.1%} yb_frac={yb_stock_frac:.1%} of that) "
        f"b4={b4_w:.0%} (enabled={b4_enabled}) "
        f"| beta_floor={beta_floor}"
    )
    print(
        f"[INFO] stock soft borrow cap={soft_borrow_cap:.1%} | "
        f"b4 hard cap={b4_hard_borrow_cap if np.isfinite(b4_hard_borrow_cap) else 'inf'} | "
        f"flow hard cap={flow_borrow_cap if np.isfinite(flow_borrow_cap) else 'inf'}"
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
    print(f"[INFO] flow shorts={len(flow_shorts)}")
    if yb_min_edge > 0:
        print(f"[INFO] yieldboost sleeve min net_edge_p50_annual floor={yb_min_edge:.0%}")
    print(f"[INFO] weighting: core={core_weight_method} yieldboost={yb_weight_method} b4={b4_weight_method}")
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
    for _col in ("blended_gross_decay", "borrow_current", "net_decay_annual", "net_edge_p50_annual"):
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
        # core_leveraged: high-beta bucket-1 path only (~not ``is_yieldboost``).
        # yieldboost: ``is_yieldboost`` rows that pass borrow + min_net_edge on net_edge_p50.
        flow_program_etfs = {_norm_sym(x) for x in (flow_shorts or [])}
        is_yieldboost, in_b2_universe, in_flow_program = _b2_b4_universe_masks(
            eligible, flow_program_etfs=flow_program_etfs
        )
        # Hard rule: negative net decay names are excluded from stock sleeve.
        net_decay_non_negative = ~(eligible["net_decay_annual"] < 0)

        b = eligible["borrow_annual"]
        core_borrow_ok = (~np.isfinite(b)) | (b <= soft_borrow_cap)
        b4_borrow_ok = (~np.isfinite(b)) | (b <= b4_hard_borrow_cap)

        # Exclude inverse (β < 0) ETFs — they belong to Bucket 4 / flow, not the stock sleeve
        positive_beta = eligible["Beta"].gt(0)
        negative_beta = eligible["Beta"].lt(0)
        if "inverse_shortable" in eligible.columns:
            inverse_shortable = eligible["inverse_shortable"].fillna(False).astype(bool)
        else:
            inverse_shortable = negative_beta
        edge_col = _b4_eligibility_edge_column(eligible)
        b4_edge = pd.to_numeric(eligible.get(edge_col), errors="coerce")
        b4_edge_ok = (~np.isfinite(b4_edge)) | (b4_edge >= b4_min_edge)
        ne_p50 = pd.to_numeric(eligible.get("net_edge_p50_annual"), errors="coerce")
        yieldboost_edge_ok = (
            np.isfinite(ne_p50) & (ne_p50 >= yb_min_edge)
            if yb_min_edge > 0
            else pd.Series(True, index=eligible.index)
        )
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
        eligible["in_core"] = core_pre_decay & core_decay_gate & ~in_b2_universe

        in_yieldboost_stock = (
            positive_beta
            & in_b2_universe
            & ~in_flow_program
            & core_borrow_ok
            & yieldboost_edge_ok
            & net_decay_non_negative
        )

        n_b2_yb_rows = int(in_b2_universe.sum())
        n_b2_flow_excluded = int((positive_beta & in_b2_universe & in_flow_program).sum())
        b4_not_excluded = ~eligible["ETF"].isin(b4_excluded_etfs)
        n_b4_flow_excluded = int(
            (
                negative_beta
                & inverse_shortable
                & b4_borrow_ok
                & b4_edge_ok
                & b4_vol_ok
                & b4_not_excluded
                & in_flow_program
            ).sum()
        )
        if n_b2_flow_excluded or n_b4_flow_excluded:
            print(
                f"[INFO] B2 YieldBoost ticker rows≈{n_b2_yb_rows} | "
                f"B2↔flow overlap (excluded from stock sleeve)={n_b2_flow_excluded}, "
                f"B4↔flow overlap={n_b4_flow_excluded}"
            )
        eligible["in_b4"] = (
            negative_beta
            & inverse_shortable
            & b4_borrow_ok
            & b4_edge_ok
            & b4_vol_ok
            & b4_not_excluded
            & ~in_flow_program
        )

        core_names = eligible.loc[eligible["in_core"]].copy()
        yb_names = eligible.loc[in_yieldboost_stock].copy()
        if not yb_enabled:
            yb_names = eligible.loc[[]].copy()
        b4_names = eligible.loc[eligible["in_b4"]].copy()
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
        n_yb_edge_blocked = int((positive_beta & in_b2_universe & ~in_flow_program & ~yieldboost_edge_ok).sum())
        if n_yb_edge_blocked:
            print(
                f"[INFO] Excluded {n_yb_edge_blocked} yieldboost sleeve candidate row(s) "
                f"below min_net_edge_annual={yb_min_edge:.2%}."
            )

        # Budgeting:
        # - Bucket 4 gets ``target_weight`` of total gross when active.
        # - Remaining gross is split core vs yieldboost using normalized YAML ``target_weights``.
        b4_budget = target_gross_usd * b4_w if (not b4_names.empty and b4_w > 0) else 0.0
        b4_budget = min(b4_budget, target_gross_usd)
        remainder_budget = max(0.0, target_gross_usd - b4_budget)
        core_budget = remainder_budget * core_stock_frac
        yb_budget = remainder_budget * yb_stock_frac
        if core_budget > 1e-9 and core_names.empty:
            yb_budget += core_budget
            core_budget = 0.0
        if yb_budget > 1e-9 and yb_names.empty:
            core_budget += yb_budget
            yb_budget = 0.0

        core_names_fit = core_names.copy()
        yb_names_fit = yb_names.copy()

        # -----------------------------
        # Allocate CORE stock sleeve (`core_leveraged`)
        # -----------------------------
        if not core_names_fit.empty and core_budget > 0:
            if core_weight_method == "decay_score":
                w = _decay_score_weights(core_names_fit, core_weighting_cfg, sleeve_name="core_leveraged")
            else:   # "equal" or unrecognised → equal weight
                w = np.ones(len(core_names_fit)) / len(core_names_fit)
            core_names_fit["gross_target_usd"] = core_budget * w
            core_names_fit["sleeve"] = "core_leveraged"
        elif not core_names_fit.empty:
            core_names_fit["gross_target_usd"] = 0.0
            core_names_fit["sleeve"] = "core_leveraged"

        # -----------------------------
        # Allocate YIELDBOOST sleeve (bucket‑2 candidates only)
        # -----------------------------
        if not yb_names_fit.empty and yb_budget > 0:
            if yb_weight_method == "decay_score":
                w = _decay_score_weights(yb_names_fit, yb_weighting_cfg, sleeve_name="yieldboost")
            else:
                w = np.ones(len(yb_names_fit)) / len(yb_names_fit)
            yb_names_fit["gross_target_usd"] = yb_budget * w
            yb_names_fit["sleeve"] = "yieldboost"
        elif not yb_names_fit.empty:
            yb_names_fit["gross_target_usd"] = 0.0
            yb_names_fit["sleeve"] = "yieldboost"

        stock_frames = []
        if not core_names_fit.empty:
            stock_frames.append(core_names_fit)
        if not yb_names_fit.empty:
            stock_frames.append(yb_names_fit)
        stock_names = pd.concat(stock_frames, axis=0, ignore_index=False) if stock_frames else eligible.loc[[]].copy()

        # -----------------------------
        # Allocate BUCKET 4
        # -----------------------------
        if not b4_names.empty and b4_budget > 0:
            if b4_weight_method == "equal":
                w = np.ones(len(b4_names)) / len(b4_names)
            else:
                w = _decay_score_weights(b4_names, b4_weighting_cfg, sleeve_name="inverse_decay_bucket4")
            b4_names["gross_target_usd"] = b4_budget * w
            b4_opt2 = b4_rules.get("bucket4_weekly_opt2") or {}
            if bool(b4_opt2.get("enabled")):
                try:
                    from scripts.bucket4_weekly_opt2 import (
                        Bucket4WeeklyConfig,
                        build_bucket4_state,
                        compute_bucket4_targets,
                        compute_bucket4_weights,
                    )
                    from scripts.v6_b4_pf_weights import V6PfParams

                    pairs_subset = [
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"])))
                        for _, r in b4_names[["ETF", "Underlying"]].iterrows()
                    ]
                    excl_inv = frozenset({"SCO"} | {_norm_sym(x) for x in (b4_opt2.get("excluded_inverse_etfs") or [])})
                    mp = int(b4_opt2.get("pf_min_pairs", 5))
                    mp = min(mp, max(1, len(pairs_subset)))
                    cfg_b4 = Bucket4WeeklyConfig(
                        screened_csv=str(screened_csv),
                        start=str(b4_opt2.get("history_start", "2018-01-01")),
                        end=str(args.run_date),
                        weekly_rebalance_freq=str(b4_opt2.get("weekly_rebalance_freq", "W-FRI")),
                        warmup_bdays=int(b4_opt2.get("warmup_bdays", 65)),
                        fee_bps=float(b4_opt2.get("fee_bps", 1.0)),
                        slippage_bps=float(b4_opt2.get("slippage_bps", 20.0)),
                        borrow_multiplier=float(b4_opt2.get("borrow_multiplier", 1.0)),
                        excluded_inverse_etfs=excl_inv,
                        min_underlying_vol=float(b4_opt2.get("min_underlying_vol", b4_min_underlying_vol)),
                        min_net_decay=float(b4_opt2.get("min_net_decay", b4_min_edge)),
                        use_ibkr_uvix_borrow=bool(b4_opt2.get("use_ibkr_uvix_borrow", False)),
                        pf_params=V6PfParams(min_pairs=mp),
                    )
                    st_b4 = build_bucket4_state(cfg_b4, bucket4_pairs=pairs_subset)
                    pw, _, _ = compute_bucket4_weights(st_b4)
                    tgt_df, _ = compute_bucket4_targets(
                        st_b4,
                        pw,
                        args.run_date,
                        float(b4_budget),
                        fee_bps=float(b4_opt2.get("fee_bps", 1.0)),
                        slippage_bps=float(b4_opt2.get("slippage_bps", 20.0)),
                        partial_hedge_ratio=b4_partial_hedge_ratio,
                        beta_floor=beta_floor,
                    )
                    gross_by_key = {
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"]))): float(r["gross_target_usd"])
                        for _, r in tgt_df.iterrows()
                        if pd.notna(r.get("gross_target_usd"))
                    }
                    inv_short_by_key = {
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"]))): float(r["inverse_etf_short_usd"])
                        for _, r in tgt_df.iterrows()
                        if pd.notna(r.get("inverse_etf_short_usd"))
                    }
                    und_short_by_key = {
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"]))): float(r["underlying_short_usd"])
                        for _, r in tgt_df.iterrows()
                        if pd.notna(r.get("underlying_short_usd"))
                    }
                    hedge_by_key = {
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"]))): float(r["hedge_ratio"])
                        for _, r in tgt_df.iterrows()
                        if pd.notna(r.get("hedge_ratio"))
                    }
                    for idx in b4_names.index:
                        k = (
                            _norm_sym(str(b4_names.at[idx, "ETF"])),
                            _norm_sym(str(b4_names.at[idx, "Underlying"])),
                        )
                        if k in gross_by_key:
                            b4_names.at[idx, "gross_target_usd"] = gross_by_key[k]
                        if k in inv_short_by_key:
                            b4_names.at[idx, "b4_opt2_inverse_etf_short_usd"] = inv_short_by_key[k]
                        if k in und_short_by_key:
                            b4_names.at[idx, "b4_opt2_underlying_short_usd"] = und_short_by_key[k]
                        if k in hedge_by_key:
                            b4_names.at[idx, "b4_opt2_hedge_ratio"] = hedge_by_key[k]
                    print(f"[INFO] bucket4_weekly_opt2: tail-risk weights + dynamic hedge targets (n={len(tgt_df)})")
                except Exception as e:
                    print(f"[WARN] bucket4_weekly_opt2 disabled for this run ({e}); using decay_score sizing")
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

        sized = pd.concat([stock_names, b4_names], axis=0, ignore_index=False)
        sized = sized[~sized.index.duplicated(keep="first")].copy()

        sized, _cap_diag = apply_gross_sizing_book_caps(
            sized,
            target_gross_usd=float(target_gross_usd),
            beta_floor=float(beta_floor),
            strategy=strategy,
            shares_out_map=shares_out_map,
        )
        if _cap_diag.get("applied"):
            if _cap_diag.get("per_sleeve_enforced"):
                _ps = _cap_diag.get("per_sleeve_caps") or {}
                print(
                    "[INFO] gross_sizing_caps (per-sleeve, DCQ-style): "
                    f"sleeves={list(_ps.keys())} "
                    f"gross_before=${_cap_diag.get('gross_sum_before', 0):,.0f} "
                    f"gross_after=${_cap_diag.get('gross_sum_after', 0):,.0f}"
                )
            else:
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

        # Bucket 4: short inverse ETF (`short_usd` / `etf_target_usd`) and short underlying hedge
        # (`long_usd` / `underlying_target_usd` — column names are GTP-wide; both USD amounts are negative).
        opt2_leg_cols = {"b4_opt2_inverse_etf_short_usd", "b4_opt2_underlying_short_usd"}
        b4_opt2_mask = b4_mask & opt2_leg_cols.issubset(set(sized.columns))
        b4_default_mask = b4_mask & ~b4_opt2_mask
        sized.loc[b4_default_mask, "short_usd"] = -sized.loc[b4_default_mask, "gross_target_usd"]
        sized.loc[b4_default_mask, "long_usd"] = -(
            b4_partial_hedge_ratio
            * sized.loc[b4_default_mask, "beta_used_abs"]
            * sized.loc[b4_default_mask, "gross_target_usd"]
        )
        if bool(b4_opt2_mask.any()):
            opt2_gross = (
                pd.to_numeric(sized.loc[b4_opt2_mask, "b4_opt2_inverse_etf_short_usd"], errors="coerce").fillna(0.0)
                + pd.to_numeric(sized.loc[b4_opt2_mask, "b4_opt2_underlying_short_usd"], errors="coerce").fillna(0.0)
            ).replace(0.0, np.nan)
            cap_scale = (
                pd.to_numeric(sized.loc[b4_opt2_mask, "gross_target_usd"], errors="coerce").fillna(0.0) / opt2_gross
            ).fillna(0.0).clip(lower=0.0)
            sized.loc[b4_opt2_mask, "short_usd"] = -(
                pd.to_numeric(sized.loc[b4_opt2_mask, "b4_opt2_inverse_etf_short_usd"], errors="coerce").fillna(0.0)
                * cap_scale
            )
            sized.loc[b4_opt2_mask, "long_usd"] = -(
                pd.to_numeric(sized.loc[b4_opt2_mask, "b4_opt2_underlying_short_usd"], errors="coerce").fillna(0.0)
                * cap_scale
            )
            if "b4_opt2_hedge_ratio" in sized.columns:
                sized.loc[b4_opt2_mask, "hedge_ratio"] = pd.to_numeric(
                    sized.loc[b4_opt2_mask, "b4_opt2_hedge_ratio"], errors="coerce"
                ).fillna(sized.loc[b4_opt2_mask, "hedge_ratio"])
        sized["underlying_target_usd"] = sized["long_usd"]
        sized["etf_target_usd"] = sized["short_usd"]

        # Write sized notionals back into KEEP (purgatory remains 0)
        keep.loc[sized.index, "long_usd"] = sized["long_usd"]
        keep.loc[sized.index, "short_usd"] = sized["short_usd"]
        keep.loc[sized.index, "underlying_target_usd"] = sized["underlying_target_usd"]
        keep.loc[sized.index, "etf_target_usd"] = sized["etf_target_usd"]
        keep.loc[sized.index, "sleeve"] = sized["sleeve"]

        print(
            f"[INFO] sized core={len(core_names_fit)} yb={len(yb_names_fit)} b4={len(b4_names)} | "
            f"budgets: core=${core_budget:,.0f} yb=${yb_budget:,.0f} (post-b4 ${remainder_budget:,.0f}) "
            f"b4=${b4_budget:,.0f}"
        )

        # Weight diagnostics
        if core_weight_method == "decay_score" and not core_names_fit.empty and core_budget > 0:
            cw = core_names_fit["gross_target_usd"] / core_budget
            print(
                f"[INFO] core_leveraged weights: max={cw.max():.3f} min={cw.min():.3f} "
                f"nonzero={int((cw > 1e-9).sum())}/{len(core_names_fit)}"
            )
        if yb_weight_method == "decay_score" and not yb_names_fit.empty and yb_budget > 0:
            yw = yb_names_fit["gross_target_usd"] / yb_budget
            print(
                f"[INFO] yieldboost weights: max={yw.max():.3f} min={yw.min():.3f} "
                f"nonzero={int((yw > 1e-9).sum())}/{len(yb_names_fit)}"
            )
        if b4_weight_method == "decay_score" and not b4_names.empty and b4_budget > 0:
            bw = b4_names["gross_target_usd"] / b4_budget
            print(
                f"[INFO] b4 weights: max={bw.max():.3f} min={bw.min():.3f} "
                f"nonzero={int((bw > 1e-9).sum())}/{len(b4_names)}"
            )

        # Internalization diagnostics by underlying.
        if not keep.empty:
            stock_mask_plan = keep["sleeve"].astype(str).isin(["core_leveraged", "yieldboost"])
            b12_under = (
                keep.loc[stock_mask_plan]
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

    print("[OK] Bucket 1/2/4 position generation complete. Flow sleeve execution remains separate.")


if __name__ == "__main__":
    main()
