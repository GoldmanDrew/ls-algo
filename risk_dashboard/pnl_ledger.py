"""Ledger writers for P&L attribution history (bucket × underlying)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BUCKET_KEYS: tuple[str, ...] = (
    "bucket_1",
    "bucket_2",
    "bucket_3",
    "bucket_4",
    "bucket_5",
)

PNL_BUCKET_UNDERLYING_HISTORY_COLS: tuple[str, ...] = (
    "date",
    "bucket",
    "underlying",
    "symbols",
    "cum_pnl_usd",
)


def _clean_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _first_nonblank(*vals) -> str:
    for v in vals:
        s = _clean_str(v)
        if s:
            return s
    return ""


def rows_from_bucket_pnl_csv(run_date: str, bucket: str, pnl_csv: Path) -> list[dict]:
    if not pnl_csv.is_file():
        return []
    df = pd.read_csv(pnl_csv)
    if df.empty or "total_pnl" not in df.columns:
        return []
    rows: list[dict] = []
    for _, r in df.iterrows():
        underlying = _first_nonblank(r.get("underlying"), r.get("symbol"))
        if not underlying:
            continue
        rows.append(
            {
                "date": run_date,
                "bucket": bucket,
                "underlying": underlying,
                "symbols": _first_nonblank(r.get("symbols"), r.get("symbol"), underlying),
                "cum_pnl_usd": float(r.get("total_pnl") or 0.0),
            }
        )
    return rows


def upsert_bucket_underlying_history(
    run_date: str,
    accounting_dir: Path,
    history_csv: Path,
) -> pd.DataFrame:
    """Append/replace rows for ``run_date`` from ``pnl_<bucket>.csv`` files."""
    new_rows: list[dict] = []
    for bucket in BUCKET_KEYS:
        new_rows.extend(
            rows_from_bucket_pnl_csv(run_date, bucket, accounting_dir / f"pnl_{bucket}.csv")
        )
    if not new_rows:
        return pd.DataFrame(columns=list(PNL_BUCKET_UNDERLYING_HISTORY_COLS))

    frame = pd.DataFrame(new_rows)
    if history_csv.is_file():
        hist = pd.read_csv(history_csv)
    else:
        hist = pd.DataFrame(columns=list(PNL_BUCKET_UNDERLYING_HISTORY_COLS))

    for col in PNL_BUCKET_UNDERLYING_HISTORY_COLS:
        if col not in hist.columns:
            hist[col] = pd.NA

    hist["date"] = hist["date"].astype(str)
    hist = hist[hist["date"] != run_date]
    hist = pd.concat([hist, frame], ignore_index=True)
    hist["date"] = hist["date"].astype(str)
    hist["cum_pnl_usd"] = pd.to_numeric(hist["cum_pnl_usd"], errors="coerce")
    hist = hist.sort_values(["date", "bucket", "underlying"]).drop_duplicates(
        subset=["date", "bucket", "underlying"], keep="last"
    )
    hist = hist[list(PNL_BUCKET_UNDERLYING_HISTORY_COLS)]
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(history_csv, index=False)
    return hist


def rebuild_bucket_underlying_history_from_runs(
    runs_root: Path,
    history_csv: Path,
    *,
    start_date: str = "2026-02-27",
) -> pd.DataFrame:
    """Rebuild full bucket-underlying ledger from archived accounting runs."""
    rows: list[dict] = []
    for totals_path in sorted(runs_root.glob("*/accounting/totals.json")):
        run_date = totals_path.parent.parent.name
        if run_date < start_date:
            continue
        acct = totals_path.parent
        for bucket in BUCKET_KEYS:
            rows.extend(rows_from_bucket_pnl_csv(run_date, bucket, acct / f"pnl_{bucket}.csv"))
    frame = pd.DataFrame(rows, columns=list(PNL_BUCKET_UNDERLYING_HISTORY_COLS))
    if frame.empty:
        history_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(history_csv, index=False)
        return frame
    frame["cum_pnl_usd"] = pd.to_numeric(frame["cum_pnl_usd"], errors="coerce")
    frame = frame.sort_values(["date", "bucket", "underlying"]).drop_duplicates(
        subset=["date", "bucket", "underlying"], keep="last"
    )
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(history_csv, index=False)
    return frame
