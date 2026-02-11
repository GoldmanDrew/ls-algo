#!/usr/bin/env python3
"""
ibkr_accounting_rebuilt.py

Rebuild an accounting-grade PnL report from IBKR Flex XML exports, with explicit
assumptions and a transparent PnL decomposition.

WHAT THIS PRODUCES
- pnl_by_symbol_rebuilt.csv
- pnl_by_underlying_rebuilt.csv
- pnl_by_pair_rebuilt.csv
- totals_rebuilt.json

PNL DECOMPOSITION (Base Currency)
Total PnL per symbol =

  (1) Trading PnL
      - realized_pnl   : from Trades Flex -> FIFOPerformanceSummaryInBase.totalRealizedPnl
      - unrealized_pnl : from Trades Flex -> FIFOPerformanceSummaryInBase.totalUnrealizedPnl

  (2) Dividends & related cash flows (from CashTransactions)
      - dividends       : CashTransactions where type == "Dividends"
      - pil_dividends   : CashTransactions where type == "Payment In Lieu Of Dividends"
      - withholding_tax : CashTransactions where type == "Withholding Tax"

  (3) Borrow economics (from CashTransactions; NOT modeled)
      - borrow_fees          : CashTransactions where type == "Broker Interest Paid"
                               AND description contains "BORROW FEES"
      - short_credit_interest: CashTransactions where type == "Broker Interest Received"
                               AND description contains "SHORT CREDIT INTEREST"

  (4) Other fees (optional)
      - other_fees      : CashTransactions where type == "Other Fees"

EXPLICIT ASSUMPTIONS
A1) We do NOT include interest paid/earned on cash balances.
    - We exclude CashTransactions rows of type "Broker Interest Paid/Received"
      unless they are explicitly "BORROW FEES" or "SHORT CREDIT INTEREST".
    - We ignore InterestAccruals entirely (currency-level cash interest accruals).

A2) Borrow fees are often reported by IBKR as account-level monthly totals without symbol.
    - When borrow rows have no symbol, we allocate them pro-rata across symbols that are
      short at period end using end-of-period short market value (OpenPositions.positionValue).
    - This is an allocation assumption for attribution; it does NOT change the account total.

A3) Underlying mapping:
    - underlying = underlyingSymbol if present; else underlying = symbol.
    - This ensures single-name equities (e.g., Berkshire) are INCLUDED.

A4) Pair definition (for attribution only):
    - If symbol == underlying -> pair = "{underlying} (spot)"
    - Else                  -> pair = "{underlying} | {symbol}"
      (e.g., "BRK.B | BRKU" for a levered ETF mapped to BRK.B if IBKR provides underlyingSymbol)

NOTES / LIMITATIONS
- FIFOPerformanceSummaryInBase is used as the authoritative Trading PnL source because it
  is already "accounting grade" for the report window (realized + unrealized, FIFO).
- If you need average-cost realized PnL computed from raw Trades (instead of IBKR FIFO),
  we can add a second mode; this script focuses on reconcilable IBKR-native PnL.

Usage:
  python ibkr_accounting_rebuilt.py \
      --flex-trades flex_trades.xml \
      --flex-positions flex_positions.xml \
      --flex-cash flex_cash.xml \
      --outdir .

"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
import sys
import yaml

# ------------------------------------------------------------
# Resolve project root (directory that contains /data)
# ------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(6):  # walk up a few levels safely
        if (cur / "data").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not locate project root containing /data")

PROJECT_ROOT = find_project_root(Path(__file__).resolve())

RUN_DATE = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
RUN_DIR = PROJECT_ROOT / "data" / "runs" / RUN_DATE / "ibkr_flex"

FLEX_CASH_PATH = RUN_DIR / "flex_cash.xml"
FLEX_POSITIONS_PATH = RUN_DIR / "flex_positions.xml"
FLEX_TRADES_PATH = RUN_DIR / "flex_trades.xml"

missing = [p for p in [FLEX_CASH_PATH, FLEX_POSITIONS_PATH, FLEX_TRADES_PATH] if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing required IBKR Flex files:\n"
        + "\n".join(str(p) for p in missing)
    )

# ------------------------------------------------------------
# Universe file (ETF -> Underlying) that defines which pairs are included
# config/ is on the same level as this script
# ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ETF_CAGR_PATH = SCRIPT_DIR / "config" / "etf_cagr.csv"

if not ETF_CAGR_PATH.exists():
    raise FileNotFoundError(f"Missing etf_cagr.csv at: {ETF_CAGR_PATH}")

def load_etf_to_under_map(etf_cagr_path: Path) -> dict[str, str]:
    u = pd.read_csv(etf_cagr_path)
    cols = {c.lower(): c for c in u.columns}

    etf_col = None
    for cand in ["etf", "symbol", "ticker", "etf_symbol"]:
        if cand in cols:
            etf_col = cols[cand]
            break
    under_col = None
    for cand in ["underlying", "underlyingsymbol", "underlying_symbol", "underlyingticker", "root"]:
        if cand in cols:
            under_col = cols[cand]
            break

    if etf_col is None or under_col is None:
        raise ValueError(f"etf_cagr.csv columns not recognized: {list(u.columns)}")

    u = u[[etf_col, under_col]].dropna()
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)

    return dict(zip(u[etf_col], u[under_col]))


def load_pair_universe(etf_cagr_path: Path) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    """
    Read config/etf_cagr.csv and return:
      - allowed_pairs: set of (underlying, etf_symbol)
      - allowed_underlyings: set of underlyings present in universe
      - allowed_etfs: set of ETF symbols present in universe

    The CSV column names vary by file versions, so we search for likely columns.
    """
    u = pd.read_csv(etf_cagr_path)
    cols = {c.lower(): c for c in u.columns}

    # Try to locate ETF and underlying columns with flexible naming
    etf_col = None
    for cand in ["etf", "symbol", "ticker", "etf_symbol"]:
        if cand in cols:
            etf_col = cols[cand]
            break
    if etf_col is None:
        raise ValueError(f"etf_cagr.csv missing an ETF ticker column. Found columns: {list(u.columns)}")

    under_col = None
    for cand in ["underlying", "underlyingsymbol", "underlying_symbol", "underlyingticker", "root"]:
        if cand in cols:
            under_col = cols[cand]
            break
    if under_col is None:
        raise ValueError(f"etf_cagr.csv missing an underlying column. Found columns: {list(u.columns)}")

    u = u[[etf_col, under_col]].copy()
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)

    u = u[(u[etf_col].astype(bool)) & (u[under_col].astype(bool))].drop_duplicates()

    allowed_pairs = set(zip(u[under_col].tolist(), u[etf_col].tolist()))
    allowed_underlyings = set(u[under_col].tolist())
    allowed_etfs = set(u[etf_col].tolist())

    return allowed_pairs, allowed_underlyings, allowed_etfs

CONFIG_YML_PATH = SCRIPT_DIR / "config" / "strategy_config.yml"
if not CONFIG_YML_PATH.exists():
    raise FileNotFoundError(f"Missing strategy_config.yml at: {CONFIG_YML_PATH}")



def canonical_symbol(sym: str) -> str:
    """
    Normalize IBKR class-share formats to dot notation:
      BRK B / BRK-B / BRK.B  -> BRK.B
    """
    if sym is None:
        return ""
    s = str(sym).strip().upper()

    # Normalize common class share forms: space or hyphen -> dot
    m = re.match(r"^([A-Z]{1,5})[ \-\.]([A-Z])$", s)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    return s


# ------------------------------------------------------------
# Hard exclusion: remove Berkshire from analysis entirely
# ------------------------------------------------------------
EXCLUDE_SYMBOLS = {canonical_symbol("BRK.B"), canonical_symbol("BRKB")}

def is_excluded_symbol(sym: str) -> bool:
    return canonical_symbol(sym) in EXCLUDE_SYMBOLS

def load_blacklist(config_yml: Path) -> set[str]:
    cfg = yaml.safe_load(config_yml.read_text(encoding="utf-8")) or {}

    # 1) YAML list
    bl = set()
    for sym in (cfg.get("strategy", {}) or {}).get("blacklist", []) or []:
        s = canonical_symbol(str(sym).upper().strip())
        if s:
            bl.add(s)

    # 2) Optional txt file referenced by config
    bl_txt = ((cfg.get("paths", {}) or {}).get("blacklist_txt", "") or "").strip()
    if bl_txt:
        # path is relative to project root or script dir; safest: resolve relative to SCRIPT_DIR
        p = (SCRIPT_DIR / bl_txt).resolve()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                bl.add(canonical_symbol(line.upper()))

    return bl

def to_base(amount: float | str, fx: float | str) -> float:
    try:
        return float(amount) * float(fx)
    except Exception:
        return float(amount)


def parse_fifo_perf(trades_xml: Path) -> pd.DataFrame:
    r = ET.parse(trades_xml).getroot()
    fps = r.find(".//FIFOPerformanceSummaryInBase")
    if fps is None:
        raise ValueError("Could not find FIFOPerformanceSummaryInBase in trades XML.")
    rows = []
    for node in fps:
        a = node.attrib
        rows.append(
            {
                "symbol": canonical_symbol(a.get("symbol", "")),
                "underlyingSymbol": canonical_symbol(a.get("underlyingSymbol", "") or ""),
                "description": a.get("description", ""),
                "realized_pnl": float(a.get("totalRealizedPnl", "0") or 0),
                "unrealized_pnl": float(a.get("totalUnrealizedPnl", "0") or 0),
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["symbol"].astype(bool)]
    return df


def parse_open_positions(pos_xml: Path) -> pd.DataFrame:
    r = ET.parse(pos_xml).getroot()
    op = r.find(".//OpenPositions")
    if op is None:
        raise ValueError("Could not find OpenPositions in positions XML.")
    rows = []
    for node in op:
        a = node.attrib
        rows.append(
            {
                "symbol": canonical_symbol(a.get("symbol", "")),
                "underlyingSymbol": canonical_symbol(a.get("underlyingSymbol", "") or ""),
                "position": float(a.get("position", "0") or 0),
                "positionValue": float(a.get("positionValue", "0") or 0),
                "fxRateToBase": float(a.get("fxRateToBase", "1") or 1),
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["symbol"].astype(bool)]
    df["positionValue_base"] = df.apply(lambda r: to_base(r["positionValue"], r["fxRateToBase"]), axis=1)

    # Some Flex exports report 'position' as absolute. If all positions are >=0,
    # infer direction from sign of positionValue_base (shorts usually negative).
    if (df["position"] < 0).sum() == 0 and (df["positionValue_base"] < 0).sum() > 0:
        df["is_short"] = df["positionValue_base"] < 0
    else:
        df["is_short"] = df["position"] < 0

    return df



def parse_cash_transactions(cash_xml: Path) -> pd.DataFrame:
    r = ET.parse(cash_xml).getroot()
    ct = r.find(".//CashTransactions")
    if ct is None:
        raise ValueError("Could not find CashTransactions in cash XML.")
    rows = []
    for node in ct:
        a = node.attrib
        rows.append(
            {
                "date": (a.get("dateTime", "") or "")[:8],
                "type": a.get("type", ""),
                "currency": a.get("currency", ""),
                "symbol": canonical_symbol(a.get("symbol", "") or ""),
                "underlyingSymbol": canonical_symbol(a.get("underlyingSymbol", "") or ""),
                "description": a.get("description", ""),
                "amount": float(a.get("amount", "0") or 0),
                "fxRateToBase": float(a.get("fxRateToBase", "1") or 1),
            }
        )
    df = pd.DataFrame(rows)
    df["amount_base"] = df.apply(lambda r: to_base(r["amount"], r["fxRateToBase"]), axis=1)
    return df

def split_symbol_vs_account_level(rows: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    Returns:
      - direct_by_symbol: Series indexed by symbol for rows that have a symbol
      - remainder_total: float for rows that do NOT have a symbol (account-level)
    """
    has_sym = rows["symbol"].astype(str).str.len() > 0
    direct = rows[has_sym].groupby("symbol")["amount_base"].sum()
    remainder = float(rows[~has_sym]["amount_base"].sum())
    return direct, remainder


def categorize_cash_row(row: pd.Series) -> str:
    t = row["type"]
    desc = (row["description"] or "").upper()

    if t == "Dividends":
        return "dividends"
    if t == "Withholding Tax":
        return "withholding_tax"
    if t == "Payment In Lieu Of Dividends":
        return "pil_dividends"
    if t == "Other Fees":
        return "other_fees"

    # Interest types: include only explicit borrow economics; exclude cash interest.
    if t in ("Broker Interest Paid", "Broker Interest Received"):
        if "BORROW FEES" in desc:
            return "borrow_fees"
        if "SHORT CREDIT INTEREST" in desc:
            return "short_credit_interest"
        return "exclude_interest"

    return "other"


def allocate_to_shorts(unallocated_amount: float, pos: pd.DataFrame) -> pd.Series:
    """Allocate an account-level amount pro-rata to end-of-period short market value."""
    if pos.empty or unallocated_amount == 0:
        return pd.Series(dtype=float)

    if "is_short" in pos.columns:
        shorts = pos[pos["is_short"]].copy()
    else:
        shorts = pos[pos["position"] < 0].copy()

    if shorts.empty:
        return pd.Series(dtype=float)

    weights = shorts.groupby("symbol")["positionValue_base"].apply(lambda s: float(np.abs(s).sum()))
    wsum = float(weights.sum())
    if wsum == 0:
        return pd.Series(dtype=float)
    return (weights / wsum) * float(unallocated_amount)

def main() -> int:
    # -----------------------------
    # No CLI args: use run folder defaults
    # -----------------------------
    trades_path = FLEX_TRADES_PATH
    positions_path = FLEX_POSITIONS_PATH
    cash_path = FLEX_CASH_PATH

    # accounting folder OUTSIDE ibkr_flex, same parent as ibkr_flex
    outdir: Path = RUN_DIR.parent / "accounting"
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Parse inputs
    # -----------------------------
    fifo = parse_fifo_perf(trades_path)
    pos = parse_open_positions(positions_path)
    cash = parse_cash_transactions(cash_path)

    # ---------------------------------------------------------
    # Exclude Berkshire (BRK.B / BRKB) everywhere: trades, positions, cash
    # ---------------------------------------------------------
    if not fifo.empty:
        fifo = fifo[~fifo["symbol"].isin(EXCLUDE_SYMBOLS) & ~fifo["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if not pos.empty:
        pos = pos[~pos["symbol"].isin(EXCLUDE_SYMBOLS) & ~pos["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if not cash.empty:
        cash = cash[~cash["symbol"].isin(EXCLUDE_SYMBOLS) & ~cash["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()

    cash["category"] = cash.apply(categorize_cash_row, axis=1)

    # -----------------------------
    # Build MASTER symbol universe so dividends/borrow never disappear
    # -----------------------------
    all_syms = set()
    if not fifo.empty:
        all_syms |= set(fifo["symbol"].dropna().astype(str))
    if not pos.empty:
        all_syms |= set(pos["symbol"].dropna().astype(str))
    if not cash.empty:
        all_syms |= set(cash["symbol"].dropna().astype(str))

    master = pd.DataFrame({"symbol": sorted(s for s in all_syms if s)})

    # FIFO trading PnL (0 if missing)
    fifo_agg = fifo.groupby("symbol", as_index=False)[["realized_pnl", "unrealized_pnl"]].sum()
    df = master.merge(fifo_agg, on="symbol", how="left")
    df["realized_pnl"] = df["realized_pnl"].fillna(0.0)
    df["unrealized_pnl"] = df["unrealized_pnl"].fillna(0.0)

    # Metadata: underlyingSymbol + description (best effort)
    fifo_meta = fifo.groupby("symbol", as_index=False).agg(
        underlyingSymbol=("underlyingSymbol", "first"),
        description=("description", "first"),
    )
    pos_meta = pos.groupby("symbol", as_index=False).agg(
        underlyingSymbol_pos=("underlyingSymbol", "first"),
    )
    cash_meta = cash.groupby("symbol", as_index=False).agg(
        underlyingSymbol_cash=("underlyingSymbol", "first"),
    )

    df = df.merge(fifo_meta, on="symbol", how="left")
    df = df.merge(pos_meta, on="symbol", how="left")
    df = df.merge(cash_meta, on="symbol", how="left")

    df["underlyingSymbol"] = (
        df["underlyingSymbol"]
        .fillna(df["underlyingSymbol_pos"])
        .fillna(df["underlyingSymbol_cash"])
        .fillna("")
    )
    df["description"] = df["description"].fillna("")

    # Cash flows per symbol (divs / PIL / withhold / other fees)
    cash_sym = cash[cash["category"].isin(["dividends", "withholding_tax", "pil_dividends", "other_fees"])].copy()
    cash_pivot = (
        cash_sym.pivot_table(index="symbol", columns="category", values="amount_base", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    cash_pivot.columns.name = None

    df = df.merge(cash_pivot, on="symbol", how="left")
    for c in ["dividends", "withholding_tax", "pil_dividends", "other_fees"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    # -----------------------------
    # Force underlying from etf_cagr.csv (source of truth)
    # This is the key fix that makes universe filtering match your real book.
    # -----------------------------
    etf_to_under = load_etf_to_under_map(ETF_CAGR_PATH)

    # If symbol is an ETF in universe, force underlying from map.
    # Otherwise fall back to IBKR underlyingSymbol; else to symbol.
    df["underlying"] = df["symbol"].map(etf_to_under)
    df["underlying"] = df["underlying"].fillna(df["underlyingSymbol"]).fillna(df["symbol"])

    # Drop anything that maps to Berkshire as the underlying (e.g., BRK.B-linked products)
    df = df[(~df["symbol"].isin(EXCLUDE_SYMBOLS)) & (~df["underlying"].isin(EXCLUDE_SYMBOLS))].copy()

    # Pair label (for attribution)
    df["pair"] = np.where(
        df["symbol"] == df["underlying"],
        df["underlying"] + " (spot)",
        df["underlying"] + " | " + df["symbol"],
    )

    # -----------------------------
    # Apply blacklist BEFORE universe filter
    # -----------------------------
    blacklist = load_blacklist(CONFIG_YML_PATH)
    df = df[~df["symbol"].isin(blacklist)].copy()
    df = df[~df["underlying"].isin(blacklist)].copy()

    # -----------------------------
    # Universe filter: only keep pairs explicitly in etf_cagr.csv
    # - Keep ETF legs if symbol is in allowed_etfs
    # - Keep spot/underlying legs only if underlying is in allowed_underlyings
    # -----------------------------
    allowed_pairs, allowed_underlyings, allowed_etfs = load_pair_universe(ETF_CAGR_PATH)

    is_spot = (df["symbol"] == df["underlying"])
    is_allowed_spot = is_spot & df["underlying"].isin(allowed_underlyings)
    is_allowed_etf = (~is_spot) & df["symbol"].isin(allowed_etfs)

    df = df[is_allowed_spot | is_allowed_etf].copy()

    # Rebuild pair labels post-filter (clean)
    df["pair"] = np.where(
        df["symbol"] == df["underlying"],
        df["underlying"] + " (spot)",
        df["underlying"] + " | " + df["symbol"],
    )

    # -----------------------------
    # Borrow economics: attribute direct-by-symbol, allocate remainder ONLY across kept shorts
    # -----------------------------
    borrow_rows = cash[cash["category"] == "borrow_fees"].copy()
    scr_rows = cash[cash["category"] == "short_credit_interest"].copy()

    borrow_direct, borrow_remainder = split_symbol_vs_account_level(borrow_rows)
    scr_direct, scr_remainder = split_symbol_vs_account_level(scr_rows)

    kept_syms = set(df["symbol"].unique())
    pos_kept = pos[(pos["symbol"].isin(kept_syms)) & (pos["is_short"])].copy()

    # If there are no kept shorts, borrow allocation can't work and PIL will likely be excluded by universe too.
    # This usually indicates etf_cagr.csv universe doesn't line up with what's actually short in positions.
    if pos_kept.empty and (borrow_remainder != 0 or scr_remainder != 0):
        print("WARNING: No short positions in kept universe; cannot allocate account-level borrow/short-credit interest.")

    borrow_alloc = allocate_to_shorts(borrow_remainder, pos_kept)
    scr_alloc = allocate_to_shorts(scr_remainder, pos_kept)

    df["borrow_fees"] = df["symbol"].map(borrow_direct).fillna(0.0) + df["symbol"].map(borrow_alloc).fillna(0.0)
    df["short_credit_interest"] = df["symbol"].map(scr_direct).fillna(0.0) + df["symbol"].map(scr_alloc).fillna(0.0)

    # -----------------------------
    # Total PnL
    # -----------------------------
    df["total_pnl"] = (
        df["realized_pnl"]
        + df["unrealized_pnl"]
        + df["dividends"]
        + df["withholding_tax"]
        + df["pil_dividends"]
        + df["borrow_fees"]
        + df["short_credit_interest"]
        + df["other_fees"]
    )

    # -----------------------------
    # Like-for-like reconciliation prints (optional; remove once stable)
    # -----------------------------
    kept_syms = set(df["symbol"].unique())

    raw_pil_kept = cash[(cash["category"] == "pil_dividends") & (cash["symbol"].isin(kept_syms))]["amount_base"].sum()
    raw_div_kept = cash[(cash["category"] == "dividends") & (cash["symbol"].isin(kept_syms))]["amount_base"].sum()
    raw_wht_kept = cash[(cash["category"] == "withholding_tax") & (cash["symbol"].isin(kept_syms))]["amount_base"].sum()

    raw_borrow_total = float(cash[cash["category"] == "borrow_fees"]["amount_base"].sum())
    raw_scr_total = float(cash[cash["category"] == "short_credit_interest"]["amount_base"].sum())

    print("Kept symbols:", len(kept_syms))
    print("Kept short positions:", int(pos_kept.shape[0]))
    print("Kept short MV base:", float(pos_kept["positionValue_base"].abs().sum()) if not pos_kept.empty else 0.0)

    print("DF PIL total:", float(df["pil_dividends"].sum()), "| Raw PIL on kept symbols:", float(raw_pil_kept))
    print("DF Div total:", float(df["dividends"].sum()), "| Raw Div on kept symbols:", float(raw_div_kept))
    print("DF WHT total:", float(df["withholding_tax"].sum()), "| Raw WHT on kept symbols:", float(raw_wht_kept))

    print("DF Borrow total:", float(df["borrow_fees"].sum()), "| Raw borrow total (account):", raw_borrow_total)
    print("DF Short credit interest:", float(df["short_credit_interest"].sum()), "| Raw SCR total (account):", raw_scr_total)

    # -----------------------------
    # Output tables
    # -----------------------------
    base_cols = [
        "realized_pnl",
        "unrealized_pnl",
        "dividends",
        "withholding_tax",
        "pil_dividends",
        "borrow_fees",
        "short_credit_interest",
        "other_fees",
        "total_pnl",
    ]

    pnl_by_symbol = df[["symbol", "underlying", "pair", "description"] + base_cols].sort_values("total_pnl", ascending=False)
    pnl_by_underlying = df.groupby("underlying", as_index=False)[base_cols].sum().sort_values("total_pnl", ascending=False)
    pnl_by_pair = df.groupby("pair", as_index=False)[base_cols].sum().sort_values("total_pnl", ascending=False)

    pnl_by_symbol.to_csv(outdir / "pnl_by_symbol.csv", index=False)
    pnl_by_underlying.to_csv(outdir / "pnl_by_underlying.csv", index=False)
    pnl_by_pair.to_csv(outdir / "pnl_by_pair.csv", index=False)

    totals = {
        "total_realized_pnl": float(df["realized_pnl"].sum()),
        "total_unrealized_pnl": float(df["unrealized_pnl"].sum()),
        "total_dividends": float(df["dividends"].sum()),
        "total_withholding_tax": float(df["withholding_tax"].sum()),
        "total_pil_dividends": float(df["pil_dividends"].sum()),
        "total_borrow_fees": float(df["borrow_fees"].sum()),
        "total_short_credit_interest": float(df["short_credit_interest"].sum()),
        "total_other_fees": float(df["other_fees"].sum()),
        "total_pnl": float(df["total_pnl"].sum()),
        "excluded_cash_interest_base": float(cash.loc[cash["category"] == "exclude_interest", "amount_base"].sum()),
        "borrow_allocation_method": "account-level borrow allocated pro-rata to end-of-period short market value across kept short symbols",
    }
    with open(outdir / "totals.json", "w") as f:
        json.dump(totals, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
