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

- Purgatory handling:
    * OUTPUT purgatory rows with 0 targets so execution won’t auto-close.
    * Does NOT allocate new exposure to purgatory.

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


def _decay_score_weights(
    df: pd.DataFrame,
    weighting_cfg: dict,
    beta_col: str = "beta_abs",
) -> np.ndarray:
    """Compute portfolio weights from decay-score signal blended with equal weight.

    Parameters
    ----------
    df : DataFrame of eligible names. Must contain columns ``blended_gross_decay``,
         ``borrow_current``, and *beta_col*.
    weighting_cfg : sleeve ``weighting`` dict from strategy_config.yml.
    beta_col : column name for absolute-value beta (default ``beta_abs``).

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

    # --- raw sizing score ------------------------------------------------
    blended = pd.to_numeric(df["blended_gross_decay"], errors="coerce")
    borrow  = pd.to_numeric(df["borrow_current"], errors="coerce").fillna(0.0)
    raw_score = blended - borrow_aversion * borrow       # higher = better

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

    # SIZING set: all non-purgatory names. Sleeve membership rules
    # determine tradability; no include_for_algo dependency.
    eligible = screened.loc[screened["purgatory"] != True].copy()  # noqa: E712
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
                w = _decay_score_weights(core_names, core_weighting_cfg)
            else:   # "equal" or unrecognised → equal weight
                w = np.ones(len(core_names)) / len(core_names)
            core_names["gross_target_usd"] = core_budget * w
            core_names["sleeve"] = "core_leveraged"

        # -----------------------------
        # Allocate WHITELIST
        # -----------------------------
        if not wl_names.empty and wl_budget > 0:
            if wl_weight_method == "decay_score":
                w = _decay_score_weights(wl_names, wl_weighting_cfg)
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
                w = _decay_score_weights(b4_names, b4_weighting_cfg)
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

    flow_rows = []
    for s in flow_shorts:
        w = float(flow_weights.get(s, 0.0))
        delta = weekly_add_usd * w

        b_ann = screened_borrow_map.get(s, np.nan)
        if np.isfinite(flow_hard_borrow_cap) and np.isfinite(b_ann) and (b_ann > flow_hard_borrow_cap):
            # skip adds if we know it's above the cap
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
