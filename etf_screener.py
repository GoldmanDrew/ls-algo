#!/usr/bin/env python3
"""
etf_screener.py (FTP version) — uses notebooks/all_pairs_with_betas.csv

- Preserves all columns from all_pairs_with_betas.csv
- Pulls IBKR shortstock (usa.txt) via FTP and maps:
    borrow_current (decimal, e.g. 0.12 = 12%)
    shares_available (int)
    borrow_spiking (bool placeholder)
    borrow_missing_from_ftp (bool)
- Screening:
    include_for_algo = (borrow_current <= borrow_low) OR whitelisted
    purgatory = borrow_low < borrow_current <= borrow_low + purgatory_margin  (and not whitelisted)
- Optional:
    exclude_negative_cagr: if true and cagr_port_hist exists, requires cagr_port_hist > 0 for include_for_algo.
- Diagnostics columns are kept; shares/spike do NOT gate inclusion.

Outputs:
- data/runs/YYYY-MM-DD/etf_screened_today.csv
- data/etf_screened_today.csv (unless --no-write-latest)
"""

from __future__ import annotations

import argparse
import ftplib
import io
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

# -----------------------------
# Paths / Config
# -----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.getenv("GITHUB_WORKSPACE", str(SCRIPT_DIR))).resolve()

STRATEGY_CONFIG = Path(os.getenv("STRATEGY_CONFIG", str(REPO_ROOT / "strategy_config.yml")))
if not STRATEGY_CONFIG.exists():
    alt = REPO_ROOT / "config" / "strategy_config.yml"
    if alt.exists():
        STRATEGY_CONFIG = alt

PAIRS_CSV = Path(os.getenv("PAIRS_CSV", str(REPO_ROOT / "notebooks" / "all_pairs_with_betas.csv")))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(REPO_ROOT / "data")))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", str(OUTPUT_DIR / "etf_screened_today.csv")))

FTP_HOST = os.getenv("IBKR_FTP_HOST") or "ftp2.interactivebrokers.com"
FTP_USER = os.getenv("IBKR_FTP_USER") or "shortstock"
FTP_PASS = os.getenv("IBKR_FTP_PASS") or ""
FTP_FILE = os.getenv("IBKR_FTP_FILE") or "usa.txt"

# Defaults (used if config missing)
BORROW_LOW_DEFAULT = 0.10
PURGATORY_MARGIN_DEFAULT = 0.01
MIN_SHARES_AVAILABLE_DEFAULT = 1000
EXCLUDE_NEGATIVE_CAGR_DEFAULT = False


def load_strategy_config(path: Path = STRATEGY_CONFIG) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def today_str() -> str:
    return date.today().isoformat()


def run_dir(run_date: str) -> Path:
    return REPO_ROOT / "data" / "runs" / run_date


def _norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


@dataclass
class ScreeningParams:
    borrow_low: float = BORROW_LOW_DEFAULT
    purgatory_margin: float = PURGATORY_MARGIN_DEFAULT
    min_shares_available: int = MIN_SHARES_AVAILABLE_DEFAULT
    whitelist_etfs: set | None = None
    exclude_negative_cagr: bool = EXCLUDE_NEGATIVE_CAGR_DEFAULT


# -----------------------------
# FTP helpers
# -----------------------------

def fetch_ibkr_shortstock_file(filename: str = FTP_FILE) -> pd.DataFrame:
    print(f"Connecting to IBKR FTP: {FTP_HOST}, file: {filename}")

    ftp = ftplib.FTP(timeout=30)
    try:
        ftp.connect(FTP_HOST, 21)
        ftp.login(user=FTP_USER, passwd=FTP_PASS)
        ftp.set_pasv(True)

        buf = io.BytesIO()
        ftp.retrbinary(f"RETR {filename}", buf.write)
        print("FTP download complete, parsing...")
    finally:
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass

    text = buf.getvalue().decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("#SYM|")), None)
    if header_idx is None:
        raise ValueError("Could not find header line starting with '#SYM|'")

    header_cols = [c.strip().lstrip("#").lower() for c in lines[header_idx].split("|")]
    data_lines = lines[header_idx + 1 :]

    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="|", header=None, engine="python")
    df = df.iloc[:, : min(len(header_cols), df.shape[1])]
    df.columns = header_cols[: df.shape[1]]
    df = df.drop(columns=[c for c in df.columns if not c or str(c).startswith("unnamed")], errors="ignore")

    print(f"Parsed {df.shape[0]} rows and {df.shape[1]} columns from FTP file.")
    return df


def _parse_rate_to_decimal(x) -> float:
    """Normalize IBKR rates like '12.34' or '12.34%' to decimal (0.1234)."""
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


def get_ibkr_borrow_snapshot_from_ftp(etf_list: Iterable[str]) -> pd.DataFrame:
    etf_list = list(dict.fromkeys([_norm_sym(x) for x in etf_list if str(x).strip()]))

    short_df = fetch_ibkr_shortstock_file(FTP_FILE)
    for req in ("sym", "rebaterate", "feerate"):
        if req not in short_df.columns:
            raise ValueError(f"Expected '{req}' column in FTP file; got: {list(short_df.columns)}")

    df = short_df.copy()
    df["sym"] = df["sym"].astype(str).str.upper().str.strip()

    df["borrow_fee_annual"] = df["feerate"].map(_parse_rate_to_decimal)
    df["borrow_rebate_annual"] = df["rebaterate"].map(_parse_rate_to_decimal)
    df["available_int"] = pd.to_numeric(df.get("available", 0), errors="coerce").fillna(0)

    df["borrow_net_annual"] = df["borrow_fee_annual"] - df["borrow_rebate_annual"]
    m = df["borrow_net_annual"].notna()
    df.loc[m, "borrow_net_annual"] = df.loc[m, "borrow_net_annual"].clip(lower=0)

    agg = (
        df.groupby("sym", as_index=False)
          .agg(
              borrow_fee_annual=("borrow_fee_annual", "max"),
              borrow_rebate_annual=("borrow_rebate_annual", "max"),
              borrow_net_annual=("borrow_net_annual", "max"),
              shares_available=("available_int", "max"),
          )
    )

    agg = agg.rename(columns={"sym": "ETF"})
    agg["borrow_current"] = agg["borrow_net_annual"]
    agg["borrow_spiking"] = False
    agg["borrow_missing_from_ftp"] = False

    present = set(agg["ETF"].values)
    missing = [sym for sym in etf_list if sym not in present]
    if missing:
        missing_df = pd.DataFrame(
            {
                "ETF": missing,
                "borrow_fee_annual": np.nan,
                "borrow_rebate_annual": np.nan,
                "borrow_net_annual": np.nan,
                "shares_available": 0,
                "borrow_current": np.nan,
                "borrow_spiking": False,
                "borrow_missing_from_ftp": True,
            }
        )
        agg = pd.concat([agg, missing_df], ignore_index=True)

    return agg.drop_duplicates(subset=["ETF"], keep="first").reset_index(drop=True)


# -----------------------------
# Screening logic
# -----------------------------

def screen_universe_for_algo(df: pd.DataFrame, params: ScreeningParams) -> pd.DataFrame:
    df = df.copy()

    df["ETF"] = df["ETF"].astype(str).map(_norm_sym)

    # Borrow / shares coercion
    df["borrow_current"] = pd.to_numeric(df.get("borrow_current"), errors="coerce")
    df["shares_available"] = pd.to_numeric(df.get("shares_available"), errors="coerce").fillna(0).astype(int)

    if "borrow_spiking" not in df.columns:
        df["borrow_spiking"] = False
    if "borrow_missing_from_ftp" not in df.columns:
        df["borrow_missing_from_ftp"] = False

    wl = params.whitelist_etfs or set()
    df["whitelisted"] = df["ETF"].isin(wl)

    # Optional CAGR filter (only if column exists)
    if "cagr_port_hist" in df.columns:
        df["cagr_port_hist"] = pd.to_numeric(df["cagr_port_hist"], errors="coerce")
        df["cagr_positive"] = df["cagr_port_hist"] > 0
    else:
        df["cagr_positive"] = pd.NA

    # Borrow thresholding
    df["borrow_leq_low"] = df["borrow_current"].notna() & (df["borrow_current"] <= params.borrow_low)
    df["borrow_gt_low"] = ~df["borrow_leq_low"]

    # Inclusion rule
    include_base = df["borrow_leq_low"] | df["whitelisted"]
    if params.exclude_negative_cagr and "cagr_port_hist" in df.columns:
        include_base = include_base & (df["cagr_positive"] == True)  # noqa: E712

    df["include_for_algo"] = include_base

    # Purgatory: just above borrow_low, within margin; never whitelisted
    df["purgatory"] = (
        df["borrow_current"].notna()
        & (~df["borrow_missing_from_ftp"])
        & (df["borrow_current"] > params.borrow_low)
        & (df["borrow_current"] <= (params.borrow_low + params.purgatory_margin))
        & (~df["whitelisted"])
    )

    # Diagnostics (not gating inclusion)
    df["exclude_borrow_gt_low"] = df["borrow_gt_low"] & ~df["whitelisted"]
    df["exclude_no_shares"] = df["shares_available"] < params.min_shares_available
    df["exclude_borrow_spike"] = df["borrow_spiking"].fillna(False)

    return df


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=os.getenv("RUN_DATE") or today_str(), help="YYYY-MM-DD (default: today).")
    ap.add_argument("--no-write-latest", action="store_true", help="Skip writing data/etf_screened_today.csv.")
    args = ap.parse_args()

    run_date = args.run_date
    dated_dir = run_dir(run_date)
    dated_dir.mkdir(parents=True, exist_ok=True)
    dated_output = dated_dir / "etf_screened_today.csv"

    print(f"Repo root: {REPO_ROOT}")
    print(f"Pairs CSV: {PAIRS_CSV}")
    print(f"Run date: {run_date}")
    print(f"Dated output: {dated_output}")
    print(f"Latest output: {OUTPUT_FILE} (write_latest={not args.no_write_latest})")

    cfg = load_strategy_config()
    screener_cfg = (cfg or {}).get("screener", {}) or {}

    # Map your new config keys
    borrow_low = float(screener_cfg.get("borrow_low", BORROW_LOW_DEFAULT))
    purg_margin = float(screener_cfg.get("purgatory_margin", PURGATORY_MARGIN_DEFAULT))
    min_shares_available = int(screener_cfg.get("min_shares_available", MIN_SHARES_AVAILABLE_DEFAULT))
    exclude_negative_cagr = bool(screener_cfg.get("exclude_negative_cagr", EXCLUDE_NEGATIVE_CAGR_DEFAULT))

    wl_raw = screener_cfg.get("whitelist_etfs", []) or []
    whitelist = {_norm_sym(x) for x in wl_raw if str(x).strip()}

    params = ScreeningParams(
        borrow_low=borrow_low,
        purgatory_margin=purg_margin,
        min_shares_available=min_shares_available,
        whitelist_etfs=whitelist,
        exclude_negative_cagr=exclude_negative_cagr,
    )

    print(f"Borrow low: {params.borrow_low:.2%}")
    print(f"Purgatory margin: {params.purgatory_margin:.2%}")
    print(f"Min shares available: {params.min_shares_available}")
    print(f"Exclude negative CAGR: {params.exclude_negative_cagr}")
    print(f"Whitelist size: {len(whitelist)}")

    if not PAIRS_CSV.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {PAIRS_CSV}")

    pairs_df = pd.read_csv(PAIRS_CSV)
    pairs_df.columns = [c.strip() for c in pairs_df.columns]
    if "ETF" not in pairs_df.columns:
        raise ValueError(f"{PAIRS_CSV} must contain column: 'ETF'")

    # Normalize ETF / Underlying
    pairs_df["ETF"] = pairs_df["ETF"].astype(str).map(_norm_sym)
    if "Underlying" in pairs_df.columns:
        pairs_df["Underlying"] = pairs_df["Underlying"].astype(str).map(_norm_sym)

    pairs_df = pairs_df.drop_duplicates(subset=["ETF"]).reset_index(drop=True)

    borrow_df = get_ibkr_borrow_snapshot_from_ftp(pairs_df["ETF"].unique())
    metrics = pairs_df.merge(borrow_df, on="ETF", how="left")

    screened = screen_universe_for_algo(metrics, params=params)

    # Write outputs
    screened.to_csv(dated_output, index=False)
    print(f"[SCREENER] Saved dated screened universe: {dated_output}")

    if not args.no_write_latest:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        screened.to_csv(OUTPUT_FILE, index=False)
        print(f"[SCREENER] Updated latest screened universe: {OUTPUT_FILE}")
    else:
        print("[SCREENER] Skipped writing latest output (per --no-write-latest).")

    included = int(screened["include_for_algo"].sum())
    purg = int(screened["purgatory"].sum())
    print(f"[SCREENER] Included for algo: {included} | Purgatory: {purg} | Total: {len(screened)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
