#!/usr/bin/env python3
"""
generate_trade_plan.py

Quarterly rebalance-friendly trade plan generator.

Key behavior:
- SIZE only rows where include_for_algo == True.
- Still OUTPUT rows where purgatory == True (with 0 targets) so execute_trade_plan.py can
  avoid auto-closing existing purgatory positions.
- Does NOT allocate new exposure to purgatory names.
- Low-beta handling:
    * Low-beta names are ONLY allowed in the output (including purgatory keep-open rows)
      if they are in strategy.low_beta_whitelist.
    * High-beta names can appear in output if include_for_algo or purgatory is True.

Config:
- strategy.low_beta_whitelist: list of ETF tickers allowed when |Beta| <= beta_cutoff
- strategy.beta_cutoff: cutoff defining "low beta" vs "high beta"

Outputs:
- data/runs/YYYY-MM-DD/proposed_trades.csv
- cfg["paths"]["proposed_trades_csv"] (latest convenience copy)
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path("config/strategy_config.yml")


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


def load_blacklist(cfg: dict) -> Set[str]:
    raw = cfg.get("strategy", {}).get("blacklist", []) or []
    return {_norm_sym(sym) for sym in raw if str(sym).strip()}


def load_low_beta_whitelist(cfg: dict) -> Set[str]:
    raw = cfg.get("strategy", {}).get("low_beta_whitelist", []) or []
    return {_norm_sym(sym) for sym in raw if str(sym).strip()}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=today_str(), help="YYYY-MM-DD for data/runs/<date>/ outputs")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    strategy = cfg["strategy"]

    screened_csv = Path(paths["screened_csv"])
    proposed_latest_csv = Path(paths["proposed_trades_csv"])

    tag = str(strategy.get("tag", "")).strip() or "strategy"
    blacklist = load_blacklist(cfg)
    low_beta_whitelist = load_low_beta_whitelist(cfg)

    beta_cutoff = float(strategy.get("beta_cutoff", 1.5))
    beta_split_high = _clamp01(strategy.get("beta_split_high", 0.8))
    beta_floor = float(strategy.get("beta_floor", 0.05))

    print(
        f"[INFO] Loaded {len(blacklist)} blacklisted symbols: "
        f"{sorted(blacklist)[:25]}{'...' if len(blacklist) > 25 else ''}"
    )
    print(f"[INFO] Low-beta whitelist size: {len(low_beta_whitelist)}")
    print(f"[INFO] beta_cutoff={beta_cutoff:.3f} (low beta = |Beta| <= cutoff)")

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
        raise ValueError(
            f"Screened CSV missing required columns: {sorted(missing)}. Found: {list(screened.columns)}"
        )

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

    # Beta classification
    screened["Beta"] = pd.to_numeric(screened["Beta"], errors="coerce")
    screened["beta_abs"] = screened["Beta"].abs()
    screened["is_low_beta"] = screened["beta_abs"].notna() & (screened["beta_abs"] <= beta_cutoff)

    # Low-beta allowed only if whitelisted
    screened["low_beta_allowed"] = ~screened["is_low_beta"] | screened["ETF"].isin(low_beta_whitelist)

    # Drop low-beta rows (including purgatory) unless whitelisted
    screened = screened.loc[screened["low_beta_allowed"]].copy()
    if screened.empty:
        print("[WARN] All rows removed by low-beta whitelist filter.")
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_latest_csv, index=False)
        return

    # KEEP set: written to proposed_trades.csv
    # - includes purgatory so we don't auto-close
    # - BUT low-beta purgatory only survives if whitelisted (enforced above)
    keep_mask = (screened["include_for_algo"] == True) | (screened["purgatory"] == True)  # noqa: E712
    keep = screened.loc[keep_mask].copy()

    # SIZING set: only rows allowed to receive NEW exposure
    eligible = screened.loc[screened["include_for_algo"] == True].copy()  # noqa: E712

    # Initialize outputs on KEEP
    keep["strategy_tag"] = tag
    keep["long_usd"] = 0.0
    keep["short_usd"] = 0.0

    if eligible.empty:
        print("[WARN] No eligible rows to size (include_for_algo empty).")
        proposed = keep.copy()

        # Keep purgatory rows even with 0 targets
        nonzero_mask = (proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)
        proposed = proposed[nonzero_mask | (proposed["purgatory"] == True)]  # noqa: E712

        cols_to_drop = ["include_for_algo", "Leverage", "ExpectedLeverage", "cagr_positive"]
        proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns], errors="ignore")

        dated_path = run_dir(args.run_date) / "proposed_trades.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(dated_path, index=False)

        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(proposed_latest_csv, index=False)

        print(f"[OK] Wrote proposed trades → {dated_path}  (n={len(proposed)})")
        print(f"[OK] Updated latest proposed trades → {proposed_latest_csv}  (n={len(proposed)})")
        return

    # Strategy params
    capital_usd = float(strategy["capital_usd"])
    gross_leverage = float(strategy["gross_leverage"])
    target_gross_usd = capital_usd * gross_leverage

    # Bucketing (cutoff is bucket split)
    eligible["beta_abs"] = eligible["beta_abs"].astype(float)
    high_mask = eligible["beta_abs"].gt(beta_cutoff)
    low_mask = ~high_mask  # (this "low" already whitelist-allowed)

    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())

    # Allocate gross across buckets (if one empty, allocate 100% to the other)
    w_high = beta_split_high
    w_low = 1.0 - w_high
    if n_high == 0 and n_low > 0:
        w_high, w_low = 0.0, 1.0
    elif n_low == 0 and n_high > 0:
        w_high, w_low = 1.0, 0.0
    elif n_high == 0 and n_low == 0:
        raise RuntimeError("No eligible rows found after filtering (unexpected).")

    gross_high_total = target_gross_usd * w_high
    gross_low_total = target_gross_usd * w_low

    gross_per_high = (gross_high_total / n_high) if n_high > 0 else 0.0
    gross_per_low = (gross_low_total / n_low) if n_low > 0 else 0.0

    eligible["gross_target_usd"] = 0.0
    eligible.loc[high_mask, "gross_target_usd"] = gross_per_high
    eligible.loc[low_mask, "gross_target_usd"] = gross_per_low

    # Hedge ratio from Beta: hedge_ratio = 1/max(|beta|, beta_floor)
    eligible["beta_used_abs"] = eligible["beta_abs"].clip(lower=beta_floor).fillna(1.0)
    eligible["hedge_ratio"] = 1.0 / eligible["beta_used_abs"]

    # gross = long + |short| = long*(1+hedge_ratio)
    eligible["long_usd"] = eligible["gross_target_usd"] / (1.0 + eligible["hedge_ratio"])
    eligible["short_usd"] = -(eligible["hedge_ratio"] * eligible["long_usd"])

    # Write sized notionals back into KEEP (purgatory rows remain at 0)
    keep.loc[eligible.index, "long_usd"] = eligible["long_usd"]
    keep.loc[eligible.index, "short_usd"] = eligible["short_usd"]

    print(
        f"[INFO] Target gross=${target_gross_usd:,.0f} | "
        f"beta_cutoff={beta_cutoff:.3f} | beta_split_high={beta_split_high:.1%} | beta_floor={beta_floor:.3f} | "
        f"n_high={n_high}, n_low={n_low} (low-beta requires whitelist for BOTH include and purgatory)"
    )

    # Formatting
    borrow_cols = [c for c in keep.columns if "borrow" in str(c).lower()]
    for c in borrow_cols:
        keep[c] = pd.to_numeric(keep[c], errors="coerce").round(4)

    usd_cols = [c for c in keep.columns if str(c).lower().endswith("_usd")]
    for c in usd_cols:
        keep[c] = pd.to_numeric(keep[c], errors="coerce").round(2)

    # Output proposed trades:
    # - keep purgatory rows even if targets are 0
    # - include_for_algo rows must be nonzero targets to appear
    proposed = keep.copy()
    nonzero_mask = (proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)
    proposed = proposed[nonzero_mask | (proposed["purgatory"] == True)]  # noqa: E712

    # Drop internal columns you don’t want in the plan file (keep purgatory!)
    cols_to_drop = ["include_for_algo", "Leverage", "ExpectedLeverage", "cagr_positive", "beta_abs", "is_low_beta", "low_beta_allowed"]
    proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns], errors="ignore")

    dated_path = run_dir(args.run_date) / "proposed_trades.csv"
    dated_path.parent.mkdir(parents=True, exist_ok=True)
    proposed.to_csv(dated_path, index=False)

    proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
    proposed.to_csv(proposed_latest_csv, index=False)

    print(f"[OK] Wrote proposed trades → {dated_path}  (n={len(proposed)})")
    print(f"[OK] Updated latest proposed trades → {proposed_latest_csv}  (n={len(proposed)})")


if __name__ == "__main__":
    main()
