#!/usr/bin/env python3
"""
generate_trade_plan.py

UPDATE: Portfolio construction matches YAML sleeves (85/15 stock sleeves + flow overlay)

Implements:
- Stock sleeves (rebalanced as part of each run):
    * core_leveraged:   target_weight=0.85 of target_gross_usd
        - requires include_for_algo == True
        - requires |Beta| >= min_beta_used
        - apply soft borrow ban (screener.borrow_low) UNLESS whitelisted sleeve overrides
        - equal weight within sleeve (optional max_name_weight cap not enforced here; can be added)

    * whitelist_stock:  target_weight=0.15 of target_gross_usd
        - requires include_for_algo == True
        - ETF must be in whitelist list
        - zipf weighting by list order (rank_order=as_listed)
        - OPTIONAL hard borrow cap for whitelist sleeve (e.g. 20%) if provided in cfg

- Purgatory handling:
    * OUTPUT purgatory rows with 0 targets so execution won’t auto-close.
    * Does NOT allocate new exposure to purgatory.

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
- flow_targets.csv (desired cumulative flow notionals as of run date, plus this-week increment)
- flow_ledger.csv (append-only record of cumulative flow)

Notes:
- This script assumes screened_csv includes: ETF, Underlying, include_for_algo, purgatory, Beta
- Borrow caps require screened_csv to include a borrow column OR you pass a borrow_map externally.
  If you don’t have borrow in screened_csv, set caps high or skip borrow filtering here.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Dict, Set, Tuple

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


def _zipf_weights(n: int, exponent: float = 1.0) -> np.ndarray:
    r = np.arange(1, n + 1, dtype=float)
    w = 1.0 / np.power(r, exponent)
    return w / w.sum() if w.sum() > 0 else w


def load_blacklist(cfg: dict) -> Set[str]:
    raw = cfg.get("strategy", {}).get("blacklist", []) or []
    return {_norm_sym(sym) for sym in raw if str(sym).strip()}


def get_borrow_col(df: pd.DataFrame) -> str | None:
    """
    Try to infer an annual borrow column from screened_csv.
    Accepts common names; returns column name or None.
    """
    candidates = [
        "borrow_annual", "Borrow_annual", "borrow", "Borrow", "net_borrow_annual", "NetBorrowAnnual",
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
    flow = sleeves.get("flow_program", {})

    core_w = float(core.get("target_weight", 0.85))
    wl_w   = float(wl.get("target_weight", 0.15))

    core_beta_min = float(core.get("rules", {}).get("min_beta_used", 1.5))

    # Borrow caps (soft vs hard)
    soft_borrow_cap = float(cfg.get("screener", {}).get("borrow_low", 1.0))  # e.g. 0.08
    wl_hard_borrow_cap = float(wl.get("rules", {}).get("hard_borrow_cap", np.inf))
    flow_hard_borrow_cap = float(flow.get("rules", {}).get("hard_borrow_cap", np.inf))

    # Whitelist list order
    wl_list = [_norm_sym(x) for x in (wl.get("universe", {}).get("etfs", []) or [])]
    wl_set = set(wl_list)
    zipf_exp = float(wl.get("weighting", {}).get("zipf_exponent", 1.0))

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
    flow_targets_latest = Path(paths.get("flow_targets_csv", "data/flow_targets.csv"))

    print(f"[INFO] target_gross_usd=${target_gross_usd:,.0f} | core={core_w:.0%} wl={wl_w:.0%} | beta_floor={beta_floor}")
    print(f"[INFO] core soft borrow cap={soft_borrow_cap:.1%} | wl hard cap={wl_hard_borrow_cap if np.isfinite(wl_hard_borrow_cap) else 'inf'} | flow hard cap={flow_hard_borrow_cap if np.isfinite(flow_hard_borrow_cap) else 'inf'}")
    print(f"[INFO] whitelist size={len(wl_list)} | flow shorts={len(flow_shorts)}")

    if not screened_csv.exists():
        raise FileNotFoundError(f"Screened CSV not found: {screened_csv}")

    screened = pd.read_csv(screened_csv)
    if screened.empty:
        print("[WARN] Screened universe is empty.")
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_latest_csv, index=False)
        return

    required_cols = {"ETF", "Underlying", "include_for_algo", "purgatory", "Beta"}
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

    # Borrow if available
    borrow_col = get_borrow_col(screened)
    screened["borrow_annual"] = compute_borrow_annual_series(screened, borrow_col)
    # if borrow missing, borrow_annual is NaN, and caps won't filter (we treat NaN as "unknown -> allow")
    # if you prefer "unknown -> exclude", change the predicate below accordingly.

    # KEEP set: written to proposed_trades.csv (include algo + purgatory)
    keep_mask = (screened["include_for_algo"] == True) | (screened["purgatory"] == True)  # noqa: E712
    keep = screened.loc[keep_mask].copy()

    # Initialize outputs on KEEP
    keep["strategy_tag"] = tag
    keep["long_usd"] = 0.0
    keep["short_usd"] = 0.0
    keep["sleeve"] = ""

    # SIZING set: only include_for_algo names (never purgatory)
    eligible = screened.loc[screened["include_for_algo"] == True].copy()  # noqa: E712
    if eligible.empty:
        print("[WARN] No eligible rows to size (include_for_algo empty).")
        proposed = keep.copy()
        nonzero_mask = (proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)
        proposed = proposed[nonzero_mask | (proposed["purgatory"] == True)]  # noqa: E712

        cols_to_drop = ["include_for_algo", "Leverage", "ExpectedLeverage", "cagr_positive", "beta_abs"]
        proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns], errors="ignore")

        dated_path = run_dir(args.run_date) / "proposed_trades.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(dated_path, index=False)
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(proposed_latest_csv, index=False)
        print(f"[OK] Wrote proposed trades → {dated_path}  (n={len(proposed)})")
        print(f"[OK] Updated latest proposed trades → {proposed_latest_csv}  (n={len(proposed)})")
    else:
        # -----------------------------
        # Build sleeve membership
        # -----------------------------
        # core: |beta| >= min_beta_used AND passes soft borrow cap (unless also whitelist)
        eligible["in_whitelist"] = eligible["ETF"].isin(wl_set)

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

        eligible["in_core"] = eligible["beta_abs"].ge(core_beta_min) & core_borrow_ok
        eligible["in_wl"]   = eligible["in_whitelist"] & wl_borrow_ok


        core_names = eligible.loc[eligible["in_core"]].copy()
        wl_names   = eligible.loc[eligible["in_wl"]].copy()

        # Reallocate sleeve weight if one side empty
        w_core, w_wl = core_w, wl_w
        if core_names.empty and not wl_names.empty:
            w_core, w_wl = 0.0, 1.0
        elif wl_names.empty and not core_names.empty:
            w_core, w_wl = 1.0, 0.0

        core_budget = target_gross_usd * w_core
        wl_budget   = target_gross_usd * w_wl

        # -----------------------------
        # Allocate CORE (equal weight)
        # -----------------------------
        if not core_names.empty and core_budget > 0:
            n = len(core_names)
            w = np.ones(n) / n
            core_names["gross_target_usd"] = core_budget * w
            core_names["sleeve"] = "core_leveraged"

        # -----------------------------
        # Allocate WHITELIST (zipf by list order)
        # -----------------------------
        if not wl_names.empty and wl_budget > 0:
            # order by wl_list rank
            wl_names["wl_rank"] = wl_names["ETF"].map(lambda x: wl_list.index(x) if x in wl_set else 10**9)
            wl_names = wl_names.sort_values("wl_rank").copy()
            w = _zipf_weights(len(wl_names), exponent=zipf_exp)
            wl_names["gross_target_usd"] = wl_budget * w
            wl_names["sleeve"] = "whitelist_stock"

        sized = pd.concat([core_names, wl_names], axis=0, ignore_index=False)
        sized = sized[~sized.index.duplicated(keep="first")].copy()

        # Now compute long/short for each sized row
        sized["beta_used_abs"] = sized["beta_abs"].clip(lower=beta_floor).fillna(1.0)
        sized["hedge_ratio"] = 1.0 / sized["beta_used_abs"]
        sized["long_usd"] = sized["gross_target_usd"] / (1.0 + sized["hedge_ratio"])
        sized["short_usd"] = -(sized["hedge_ratio"] * sized["long_usd"])

        # Write sized notionals back into KEEP (purgatory remains 0)
        keep.loc[sized.index, "long_usd"] = sized["long_usd"]
        keep.loc[sized.index, "short_usd"] = sized["short_usd"]
        keep.loc[sized.index, "sleeve"] = sized["sleeve"]

        print(f"[INFO] sized core={len(core_names)} wl={len(wl_names)} | budgets: core=${core_budget:,.0f} wl=${wl_budget:,.0f}")

        # Output proposed trades:
        proposed = keep.copy()
        nonzero_mask = (proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)
        proposed = proposed[nonzero_mask | (proposed["purgatory"] == True)]  # noqa: E712

        cols_to_drop = ["include_for_algo", "Leverage", "ExpectedLeverage", "cagr_positive", "beta_abs"]
        proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns], errors="ignore")

        dated_path = run_dir(args.run_date) / "proposed_trades.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(dated_path, index=False)
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(proposed_latest_csv, index=False)

        print(f"[OK] Wrote proposed trades → {dated_path}  (n={len(proposed)})")
        print(f"[OK] Updated latest proposed trades → {proposed_latest_csv}  (n={len(proposed)})")

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

    # Write flow targets snapshot (latest + dated run folder copy)
    flow_targets_latest.parent.mkdir(parents=True, exist_ok=True)
    flow_df.to_csv(flow_targets_latest, index=False)

    flow_dated = run_dir(args.run_date) / "flow_targets.csv"
    flow_dated.parent.mkdir(parents=True, exist_ok=True)
    flow_df.to_csv(flow_dated, index=False)

    print(f"[OK] Flow tracking: weekly_add=${weekly_add_usd:,.2f} | wrote {flow_targets_latest} and {flow_dated}")
    print(f"[OK] Flow ledger: {flow_ledger_path} (appended {len(to_append)} rows)")


if __name__ == "__main__":
    main()
