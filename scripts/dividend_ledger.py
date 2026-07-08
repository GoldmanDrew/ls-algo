#!/usr/bin/env python3
"""Maintain per-day dividend / PIL / withholding cash ledger from IBKR Flex."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ibkr_accounting import (
    DIVIDEND_CASH_TYPES,
    _yyyymmdd_to_iso,
    categorize_cash_row,
    parse_cash_transactions,
)

LEDGER_DIR = PROJECT_ROOT / "data" / "ledger"
RUNS_ROOT = PROJECT_ROOT / "data" / "runs"
DIVIDEND_CASH_HISTORY_CSV = LEDGER_DIR / "dividend_cash_history.csv"
START_DATE = "2026-02-27"

DIVIDEND_CASH_HISTORY_COLS: tuple[str, ...] = (
    "date",
    "symbol",
    "underlying",
    "bucket",
    "pair",
    "type",
    "category",
    "amount_usd",
    "ex_date",
    "description",
)


def _cash_booking_date(row: pd.Series) -> str:
    rd = _yyyymmdd_to_iso(str(row.get("reportDate") or ""))
    if rd:
        return rd
    return _yyyymmdd_to_iso(str(row.get("date") or ""))


def _symbol_meta(pnl_symbol: pd.DataFrame) -> dict[str, dict[str, str]]:
    if pnl_symbol.empty or "symbol" not in pnl_symbol.columns:
        return {}
    out: dict[str, dict[str, str]] = {}
    for _, r in pnl_symbol.iterrows():
        sym = str(r.get("symbol") or "").strip()
        if not sym:
            continue
        out[sym] = {
            "underlying": str(r.get("underlying") or sym),
            "bucket": str(r.get("bucket") or ""),
            "pair": str(r.get("pair") or ""),
        }
    return out


def _run_dates_with_accounting(runs_root: Path) -> list[str]:
    dates: list[str] = []
    if not runs_root.is_dir():
        return dates
    for child in runs_root.iterdir():
        if child.is_dir() and (child / "accounting" / "pnl_by_symbol.csv").is_file():
            dates.append(child.name)
    return sorted(dates)


def _latest_flex_cash_xml(runs_root: Path) -> Path | None:
    best: tuple[str, Path] | None = None
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        cash_xml = child / "ibkr_flex" / "flex_cash.xml"
        if cash_xml.is_file():
            if best is None or child.name > best[0]:
                best = (child.name, cash_xml)
    return best[1] if best else None


def _meta_for_booking_date(runs_root: Path, booking_date: str) -> dict[str, dict[str, str]]:
    run_dates = _run_dates_with_accounting(runs_root)
    candidates = [d for d in run_dates if d <= booking_date]
    pick = candidates[-1] if candidates else (run_dates[-1] if run_dates else None)
    if not pick:
        return {}
    pnl_sym = runs_root / pick / "accounting" / "pnl_by_symbol.csv"
    try:
        return _symbol_meta(pd.read_csv(pnl_sym))
    except Exception:
        return {}


def rebuild_dividend_cash_history(
    *,
    runs_root: Path = RUNS_ROOT,
    cash_xml: Path | None = None,
) -> pd.DataFrame:
    """Rebuild ledger from the newest cumulative Flex cash file."""
    cash_path = cash_xml or _latest_flex_cash_xml(runs_root)
    if cash_path is None or not cash_path.is_file():
        return pd.DataFrame(columns=list(DIVIDEND_CASH_HISTORY_COLS))

    cash = parse_cash_transactions(cash_path)
    if cash.empty:
        return pd.DataFrame(columns=list(DIVIDEND_CASH_HISTORY_COLS))
    cash = cash[cash["type"].isin(DIVIDEND_CASH_TYPES)].copy()
    if cash.empty:
        return pd.DataFrame(columns=list(DIVIDEND_CASH_HISTORY_COLS))

    cash["booking_date"] = cash.apply(_cash_booking_date, axis=1)
    cash = cash[cash["booking_date"].astype(str).str.len() > 0]
    cash = cash[cash["booking_date"] >= START_DATE]

    meta_cache: dict[str, dict[str, dict[str, str]]] = {}
    rows: list[dict[str, str | float]] = []
    for _, r in cash.iterrows():
        booking = str(r.get("booking_date") or "")
        if booking not in meta_cache:
            meta_cache[booking] = _meta_for_booking_date(runs_root, booking)
        sym = str(r.get("symbol") or "").strip()
        info = meta_cache[booking].get(sym, {})
        und = info.get("underlying") or str(r.get("underlyingSymbol") or sym)
        rows.append(
            {
                "date": booking,
                "symbol": sym,
                "underlying": und,
                "bucket": info.get("bucket", ""),
                "pair": info.get("pair", ""),
                "type": str(r.get("type") or ""),
                "category": categorize_cash_row(r),
                "amount_usd": float(r.get("amount_base") or 0.0),
                "ex_date": _yyyymmdd_to_iso(str(r.get("exDate") or "")),
                "description": str(r.get("description") or ""),
            }
        )

    hist = pd.DataFrame(rows)
    if hist.empty:
        return pd.DataFrame(columns=list(DIVIDEND_CASH_HISTORY_COLS))
    hist = hist.drop_duplicates(
        subset=["date", "symbol", "type", "category", "amount_usd", "description"],
        keep="last",
    )
    hist = hist.sort_values(["date", "symbol", "type"]).reset_index(drop=True)
    return hist


def extract_dividend_cash_rows(
    run_date: str,
    *,
    cash_xml: Path,
    pnl_symbol_csv: Path,
) -> list[dict[str, str | float]]:
    """Return dividend-related cash rows booked on ``run_date`` (report date)."""
    hist = rebuild_dividend_cash_history(cash_xml=cash_xml)
    if hist.empty:
        return []
    day = hist[hist["date"].astype(str) == str(run_date)]
    return day.to_dict(orient="records")


def enrich_dividend_cash_history_from_runs(hist: pd.DataFrame) -> pd.DataFrame:
    """Replace ledger contents from the newest cumulative Flex cash export."""
    _ = hist
    return rebuild_dividend_cash_history()


def update_dividend_cash_history(
    run_date: str,
    *,
    cash_xml: Path,
    pnl_symbol_csv: Path,
) -> pd.DataFrame:
    """Refresh dividend_cash_history.csv from the latest cumulative Flex cash."""
    _ = (run_date, pnl_symbol_csv)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    hist = rebuild_dividend_cash_history(cash_xml=cash_xml if cash_xml.is_file() else None)
    for c in DIVIDEND_CASH_HISTORY_COLS:
        if c not in hist.columns:
            hist[c] = ""
    hist = hist[list(DIVIDEND_CASH_HISTORY_COLS)]
    hist.to_csv(DIVIDEND_CASH_HISTORY_CSV, index=False)
    return hist


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Backfill dividend_cash_history.csv from runs.")
    p.add_argument("--run-date", help="Ignored; full rebuild uses latest Flex cash")
    args = p.parse_args()
    _ = args.run_date
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    hist = rebuild_dividend_cash_history()
    hist.to_csv(DIVIDEND_CASH_HISTORY_CSV, index=False)
    print(f"Wrote {len(hist)} rows to {DIVIDEND_CASH_HISTORY_CSV}")


if __name__ == "__main__":
    main()
