#!/usr/bin/env python3
"""
ibkr_accounting.py

Rebuild an accounting-grade PnL report from IBKR Flex XML exports.

UPDATED:
- Uses flex_borrow_fee_details.xml (Borrow Fee Details) as authoritative borrow fees BY SYMBOL (daily),
  filtered to RUN_DATE, and grouped by symbol.
- Robust parsing: FIFO performance summary node varies by Flex query configuration.
  We try:
    1) FIFOPerformanceSummaryInBase
    2) FIFOPerformanceSummary (convert via fxRateToBase when available)
  If neither is present, trading PnL is set to 0 and the script still runs.
- Universe + ETF->Underlying mapping comes from: data/etf_screened_today.csv
  (NOT config/etf_cagr.csv)

Outputs (written to: data/runs/<RUN_DATE>/accounting/):
- pnl_by_symbol.csv
- pnl_by_underlying.csv
- pnl_by_pair.csv
- totals.json

Run:
  python ibkr_accounting.py YYYY-MM-DD
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# -----------------------------
# Helpers / normalization
# -----------------------------
def canonical_symbol(sym: str) -> str:
    if sym is None:
        return ""
    s = str(sym).strip().upper()
    m = re.match(r"^([A-Z]{1,5})[ \-\.]([A-Z])$", s)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return s


def to_base(amount: float | str, fx: float | str) -> float:
    try:
        return float(amount) * float(fx)
    except Exception:
        try:
            return float(amount)
        except Exception:
            return 0.0


def yyyymmdd_from_run_date(run_date: str) -> str:
    return run_date.replace("-", "")


# -----------------------------
# Resolve project root
# -----------------------------
def find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
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
FLEX_BORROW_DETAILS_PATH = RUN_DIR / "flex_borrow_fee_details.xml"  # optional but preferred

missing = [p for p in [FLEX_CASH_PATH, FLEX_POSITIONS_PATH, FLEX_TRADES_PATH] if not p.exists()]
if missing:
    raise FileNotFoundError("Missing required IBKR Flex files:\n" + "\n".join(str(p) for p in missing))

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_YML_PATH = SCRIPT_DIR / "config" / "strategy_config.yml"
if not CONFIG_YML_PATH.exists():
    raise FileNotFoundError(f"Missing strategy_config.yml at: {CONFIG_YML_PATH}")

# NEW: universe lives in /data (not config)
ETF_SCREENED_PATH = PROJECT_ROOT / "data" / "etf_screened_today.csv"
if not ETF_SCREENED_PATH.exists():
    raise FileNotFoundError(f"Missing etf_screened_today.csv at: {ETF_SCREENED_PATH}")


# -----------------------------
# Hard exclusion
# -----------------------------
EXCLUDE_SYMBOLS = {canonical_symbol("BRK.B"), canonical_symbol("BRKB")}


def load_blacklist(config_yml: Path) -> set[str]:
    cfg = yaml.safe_load(config_yml.read_text(encoding="utf-8")) or {}
    bl = set()
    for sym in (cfg.get("strategy", {}) or {}).get("blacklist", []) or []:
        s = canonical_symbol(str(sym).upper().strip())
        if s:
            bl.add(s)

    bl_txt = ((cfg.get("paths", {}) or {}).get("blacklist_txt", "") or "").strip()
    if bl_txt:
        p = (SCRIPT_DIR / bl_txt).resolve()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                bl.add(canonical_symbol(line.upper()))
    return bl


def _find_col(cols_lc: dict[str, str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in cols_lc:
            return cols_lc[cand]
    return None


def load_etf_to_under_map(screened_csv: Path) -> dict[str, str]:
    """
    Read data/etf_screened_today.csv and return ETF->Underlying mapping.
    Column names vary; we search common variants.
    """
    u = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in u.columns}

    etf_col = _find_col(cols, ["etf", "symbol", "ticker", "etf_symbol", "etf_ticker"])
    under_col = _find_col(cols, ["underlying", "underlyingsymbol", "underlying_symbol", "root", "underlyingticker"])

    if etf_col is None or under_col is None:
        raise ValueError(
            f"etf_screened_today.csv must contain ETF + underlying columns. Found: {list(u.columns)}"
        )

    u = u[[etf_col, under_col]].dropna()
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)

    u = u.drop_duplicates(subset=[etf_col], keep="last")
    return dict(zip(u[etf_col], u[under_col]))


def load_universe_from_screened(screened_csv: Path) -> tuple[set[str], set[str]]:
    """
    Returns:
      - allowed_etfs: screened ETFs
      - allowed_underlyings: their mapped underlyings
    """
    u = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in u.columns}

    etf_col = _find_col(cols, ["etf", "symbol", "ticker", "etf_symbol", "etf_ticker"])
    under_col = _find_col(cols, ["underlying", "underlyingsymbol", "underlying_symbol", "root", "underlyingticker"])

    if etf_col is None or under_col is None:
        raise ValueError(
            f"etf_screened_today.csv must contain ETF + underlying columns. Found: {list(u.columns)}"
        )

    u = u[[etf_col, under_col]].dropna()
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)
    u = u[(u[etf_col].astype(bool)) & (u[under_col].astype(bool))].drop_duplicates()

    allowed_etfs = set(u[etf_col].tolist())
    allowed_underlyings = set(u[under_col].tolist())
    return allowed_etfs, allowed_underlyings


# -----------------------------
# Parsers
# -----------------------------
def parse_fifo_perf(trades_xml: Path) -> pd.DataFrame:
    """
    Robust: tries multiple nodes.
    Returns columns: symbol, underlyingSymbol, description, realized_pnl, unrealized_pnl
    """
    r = ET.parse(trades_xml).getroot()

    def _rows_from_node(node, in_base: bool) -> list[dict]:
        rows = []
        for child in node:
            a = child.attrib
            sym = canonical_symbol(a.get("symbol", "") or "")
            if not sym:
                continue
            und = canonical_symbol(a.get("underlyingSymbol", "") or "")
            desc = a.get("description", "") or ""

            if in_base:
                realized = float(a.get("totalRealizedPnl", "0") or 0)
                unreal = float(a.get("totalUnrealizedPnl", "0") or 0)
            else:
                fx = float(a.get("fxRateToBase", "1") or 1)
                realized_local = float(a.get("totalRealizedPnl", a.get("realizedPnl", "0")) or 0)
                unreal_local = float(a.get("totalUnrealizedPnl", a.get("unrealizedPnl", "0")) or 0)
                realized = to_base(realized_local, fx)
                unreal = to_base(unreal_local, fx)

            rows.append(
                {
                    "symbol": sym,
                    "underlyingSymbol": und,
                    "description": desc,
                    "realized_pnl": float(realized),
                    "unrealized_pnl": float(unreal),
                }
            )
        return rows

    n1 = r.find(".//FIFOPerformanceSummaryInBase")
    if n1 is not None:
        df = pd.DataFrame(_rows_from_node(n1, in_base=True))
        return df[df["symbol"].astype(bool)] if not df.empty else df

    n2 = r.find(".//FIFOPerformanceSummary")
    if n2 is not None:
        df = pd.DataFrame(_rows_from_node(n2, in_base=False))
        return df[df["symbol"].astype(bool)] if not df.empty else df

    # No FIFO summary present — return empty but with correct schema (do not crash)
    return pd.DataFrame(columns=["symbol", "underlyingSymbol", "description", "realized_pnl", "unrealized_pnl"])


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


def parse_borrow_fee_details(borrow_xml: Path, run_date: str) -> pd.DataFrame:
    """
    Parse Borrow Fee Details for RUN_DATE.
    Works with nodes commonly seen:
      - HardToBorrowDetail
    Returns guaranteed columns even if empty.
    """
    cols = ["date", "symbol", "underlyingSymbol", "borrowFeeRate", "borrowFee_base"]
    if not borrow_xml.exists():
        return pd.DataFrame(columns=cols)

    r = ET.parse(borrow_xml).getroot()
    target = yyyymmdd_from_run_date(run_date)
    rows: list[dict] = []

    nodes = r.findall(".//HardToBorrowDetail")
    if not nodes:
        nodes = r.findall(".//*[@valueDate]")  # fallback: any node with a valueDate attr

    for node in nodes:
        a = node.attrib
        vd = (a.get("valueDate", "") or "").strip()
        if vd != target:
            continue

        sym = canonical_symbol(a.get("symbol", "") or "")
        if not sym:
            continue

        fx = float(a.get("fxRateToBase", "1") or 1)
        borrow_fee = float(a.get("borrowFee", "0") or 0)
        borrow_fee_base = to_base(borrow_fee, fx)

        br = a.get("borrowFeeRate", "")
        try:
            borrow_rate = float(br) if br not in ("", None) else np.nan
        except Exception:
            borrow_rate = np.nan

        rows.append(
            {
                "date": vd,
                "symbol": sym,
                "underlyingSymbol": canonical_symbol(a.get("underlyingSymbol", "") or ""),
                "borrowFeeRate": borrow_rate,
                "borrowFee_base": float(borrow_fee_base),
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out

    out = (
        out.groupby(["date", "symbol"], as_index=False)
        .agg(
            underlyingSymbol=("underlyingSymbol", "first"),
            borrowFeeRate=("borrowFeeRate", "max"),
            borrowFee_base=("borrowFee_base", "sum"),
        )
    )
    return out


# -----------------------------
# Cash categorization + allocation fallback
# -----------------------------
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

    if t in ("Broker Interest Paid", "Broker Interest Received"):
        if "BORROW FEES" in desc:
            return "borrow_fees"
        if "SHORT CREDIT INTEREST" in desc:
            return "short_credit_interest"
        return "exclude_interest"

    return "other"


def split_symbol_vs_account_level(rows: pd.DataFrame) -> tuple[pd.Series, float]:
    if rows.empty:
        return pd.Series(dtype=float), 0.0
    has_sym = rows["symbol"].astype(str).str.len() > 0
    direct = rows[has_sym].groupby("symbol")["amount_base"].sum()
    remainder = float(rows[~has_sym]["amount_base"].sum())
    return direct, remainder


def allocate_to_shorts(unallocated_amount: float, pos: pd.DataFrame) -> pd.Series:
    if pos.empty or unallocated_amount == 0:
        return pd.Series(dtype=float)

    shorts = pos[pos["is_short"]].copy() if "is_short" in pos.columns else pos[pos["position"] < 0].copy()
    if shorts.empty:
        return pd.Series(dtype=float)

    weights = shorts.groupby("symbol")["positionValue_base"].apply(lambda s: float(np.abs(s).sum()))
    wsum = float(weights.sum())
    if wsum == 0:
        return pd.Series(dtype=float)

    return (weights / wsum) * float(unallocated_amount)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    outdir: Path = RUN_DIR.parent / "accounting"
    outdir.mkdir(parents=True, exist_ok=True)

    fifo = parse_fifo_perf(FLEX_TRADES_PATH)
    pos = parse_open_positions(FLEX_POSITIONS_PATH)
    cash = parse_cash_transactions(FLEX_CASH_PATH)
    borrow_details = parse_borrow_fee_details(FLEX_BORROW_DETAILS_PATH, RUN_DATE)

    # Exclusions
    if not fifo.empty:
        fifo = fifo[~fifo["symbol"].isin(EXCLUDE_SYMBOLS) & ~fifo["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if not pos.empty:
        pos = pos[~pos["symbol"].isin(EXCLUDE_SYMBOLS) & ~pos["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if not cash.empty:
        cash = cash[~cash["symbol"].isin(EXCLUDE_SYMBOLS) & ~cash["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if not borrow_details.empty:
        borrow_details = borrow_details[~borrow_details["symbol"].isin(EXCLUDE_SYMBOLS)].copy()

    cash["category"] = cash.apply(categorize_cash_row, axis=1)

    # Master universe (so cash / borrow rows don't disappear)
    all_syms = set()
    if not fifo.empty:
        all_syms |= set(fifo["symbol"].dropna().astype(str))
    if not pos.empty:
        all_syms |= set(pos["symbol"].dropna().astype(str))
    if not cash.empty:
        all_syms |= set(cash["symbol"].dropna().astype(str))
    if not borrow_details.empty:
        all_syms |= set(borrow_details["symbol"].dropna().astype(str))

    master = pd.DataFrame({"symbol": sorted(s for s in all_syms if s)})

    # Trading PnL by symbol (IBKR FIFO)
    if fifo.empty:
        fifo_agg = pd.DataFrame(columns=["symbol", "realized_pnl", "unrealized_pnl"])
    else:
        fifo_agg = fifo.groupby("symbol", as_index=False)[["realized_pnl", "unrealized_pnl"]].sum()

    df = master.merge(fifo_agg, on="symbol", how="left")
    df["realized_pnl"] = df.get("realized_pnl", 0.0).fillna(0.0)
    df["unrealized_pnl"] = df.get("unrealized_pnl", 0.0).fillna(0.0)

    # Metadata
    fifo_meta = (
        fifo.groupby("symbol", as_index=False).agg(
            underlyingSymbol=("underlyingSymbol", "first"),
            description=("description", "first"),
        )
        if not fifo.empty
        else pd.DataFrame(columns=["symbol", "underlyingSymbol", "description"])
    )
    pos_meta = (
        pos.groupby("symbol", as_index=False).agg(underlyingSymbol_pos=("underlyingSymbol", "first"))
        if not pos.empty
        else pd.DataFrame(columns=["symbol", "underlyingSymbol_pos"])
    )
    cash_meta = (
        cash.groupby("symbol", as_index=False).agg(underlyingSymbol_cash=("underlyingSymbol", "first"))
        if not cash.empty
        else pd.DataFrame(columns=["symbol", "underlyingSymbol_cash"])
    )
    bfd_meta = (
        borrow_details.groupby("symbol", as_index=False).agg(underlyingSymbol_bfd=("underlyingSymbol", "first"))
        if ("symbol" in borrow_details.columns and not borrow_details.empty)
        else pd.DataFrame(columns=["symbol", "underlyingSymbol_bfd"])
    )

    df = df.merge(fifo_meta, on="symbol", how="left")
    df = df.merge(pos_meta, on="symbol", how="left")
    df = df.merge(cash_meta, on="symbol", how="left")
    df = df.merge(bfd_meta, on="symbol", how="left")

    df["underlyingSymbol"] = (
        df.get("underlyingSymbol")
        .fillna(df.get("underlyingSymbol_pos"))
        .fillna(df.get("underlyingSymbol_cash"))
        .fillna(df.get("underlyingSymbol_bfd"))
        .fillna("")
    )
    df["description"] = df.get("description", "").fillna("")

    # Cash flows per symbol
    cash_sym = cash[cash["category"].isin(["dividends", "withholding_tax", "pil_dividends", "other_fees"])].copy()
    cash_pivot = (
        cash_sym.pivot_table(index="symbol", columns="category", values="amount_base", aggfunc="sum", fill_value=0).reset_index()
        if not cash_sym.empty
        else pd.DataFrame({"symbol": df["symbol"].unique()})
    )
    cash_pivot.columns.name = None

    df = df.merge(cash_pivot, on="symbol", how="left")
    for c in ["dividends", "withholding_tax", "pil_dividends", "other_fees"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    # Underlying mapping (source of truth: data/etf_screened_today.csv)
    etf_to_under = load_etf_to_under_map(ETF_SCREENED_PATH)
    df["underlying"] = df["symbol"].map(etf_to_under)
    df["underlying"] = df["underlying"].fillna(df["underlyingSymbol"]).fillna(df["symbol"])

    # Drop Berkshire mapping
    df = df[(~df["symbol"].isin(EXCLUDE_SYMBOLS)) & (~df["underlying"].isin(EXCLUDE_SYMBOLS))].copy()

    # Pair label pre-filter
    df["pair"] = np.where(
        df["symbol"] == df["underlying"],
        df["underlying"] + " (spot)",
        df["underlying"] + " | " + df["symbol"],
    )

    # Blacklist
    blacklist = load_blacklist(CONFIG_YML_PATH)
    df = df[~df["symbol"].isin(blacklist)].copy()
    df = df[~df["underlying"].isin(blacklist)].copy()

    # Universe filter (source: data/etf_screened_today.csv)
    allowed_etfs, allowed_underlyings = load_universe_from_screened(ETF_SCREENED_PATH)

    is_spot = (df["symbol"] == df["underlying"])
    is_allowed_spot = is_spot & df["underlying"].isin(allowed_underlyings)
    is_allowed_etf = (~is_spot) & df["symbol"].isin(allowed_etfs)
    df = df[is_allowed_spot | is_allowed_etf].copy()

    # Rebuild pair labels post-filter
    df["pair"] = np.where(
        df["symbol"] == df["underlying"],
        df["underlying"] + " (spot)",
        df["underlying"] + " | " + df["symbol"],
    )

    # Borrow + SCR
    kept_syms = set(df["symbol"].unique())
    pos_kept_shorts = pos[(pos["symbol"].isin(kept_syms)) & (pos["is_short"])].copy()

    # SCR from cash
    scr_rows = cash[cash["category"] == "short_credit_interest"].copy()
    scr_direct, scr_remainder = split_symbol_vs_account_level(scr_rows)
    scr_alloc = allocate_to_shorts(scr_remainder, pos_kept_shorts)
    df["short_credit_interest"] = df["symbol"].map(scr_direct).fillna(0.0) + df["symbol"].map(scr_alloc).fillna(0.0)

    # Borrow fees: prefer borrow details
    borrow_details_kept = (
        borrow_details[borrow_details["symbol"].isin(kept_syms)].copy()
        if ("symbol" in borrow_details.columns)
        else pd.DataFrame()
    )
    if not borrow_details_kept.empty:
        bfd_map = borrow_details_kept.set_index("symbol")["borrowFee_base"].to_dict()
        df["borrow_fees"] = df["symbol"].map(bfd_map).fillna(0.0)
        borrow_mode = "borrow_fee_details_by_symbol"
        borrow_total_account_source = float(borrow_details_kept["borrowFee_base"].sum())
    else:
        borrow_rows = cash[cash["category"] == "borrow_fees"].copy()
        borrow_direct, borrow_remainder = split_symbol_vs_account_level(borrow_rows)
        borrow_alloc = allocate_to_shorts(borrow_remainder, pos_kept_shorts)
        df["borrow_fees"] = df["symbol"].map(borrow_direct).fillna(0.0) + df["symbol"].map(borrow_alloc).fillna(0.0)
        borrow_mode = "cash_borrow_allocated_pro_rata_to_kept_shorts"
        borrow_total_account_source = float(borrow_rows["amount_base"].sum()) if not borrow_rows.empty else 0.0

    # Total PnL
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
        "run_date": RUN_DATE,
        "fifo_summary_present": bool(not fifo.empty),
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
        "borrow_mode": borrow_mode,
        "borrow_total_account_source": float(borrow_total_account_source),
        "borrow_details_file_present": bool(FLEX_BORROW_DETAILS_PATH.exists()),
        "borrow_details_rows_for_day": int(borrow_details_kept.shape[0]) if isinstance(borrow_details_kept, pd.DataFrame) else 0,
        "universe_source": str(ETF_SCREENED_PATH),
        "universe_allowed_etfs": int(len(load_universe_from_screened(ETF_SCREENED_PATH)[0])),
        "universe_allowed_underlyings": int(len(load_universe_from_screened(ETF_SCREENED_PATH)[1])),
        "kept_symbols": int(df["symbol"].nunique()) if not df.empty else 0,
        "kept_underlyings": int(df["underlying"].nunique()) if not df.empty else 0,
    }
    with open(outdir / "totals.json", "w", encoding="utf-8") as f:
        json.dump(totals, f, indent=2)

    if fifo.empty:
        print(
            "WARNING: No FIFO performance summary found in flex_trades.xml; trading PnL set to 0. "
            "Fix your Flex query to include FIFOPerformanceSummary(InBase) if you want realized/unrealized PnL."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
