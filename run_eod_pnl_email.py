#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from datetime import date, datetime
import smtplib
from email.message import EmailMessage
from email.utils import getaddresses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ibkr_accounting import (
    EXCLUDE_SYMBOLS,
    SUPPLEMENTAL_ETF_MAP,
    canonical_symbol,
    complete_etf_maps_from_positions,
    compute_net_exposure,
    expand_blacklist,
    format_exposure_table,
    load_blacklist,
    load_etf_delta_map,
    load_etf_to_under_map,
    load_universe_from_screened,
    normalize_plan_etf_ticker,
    parse_open_positions,
    _filter_exposure_df,
    _filter_positions_blacklist,
)
from strategy_config import load_config
from trade_plan_targets import (
    maybe_merge_optimal_targets as _maybe_merge_optimal_targets,
    resolve_target_basis_columns as _resolve_target_basis_columns,
)
from reporting_scope import (
    load_blocked_exposure_sets as _scope_blocked_exposure_sets,
    load_screened_for_run as _scope_load_screened_for_run,
    screened_etf_and_underlying_sets as _screened_etf_and_underlying_sets,
    screened_universe_symbols as _screened_universe_symbols,
)


def _load_screened_for_run(run_date: str) -> pd.DataFrame:
    return _scope_load_screened_for_run(
        run_date, runs_root=RUNS_ROOT, project_root=PROJECT_ROOT
    )


def _blocked_exposure_sets(run_date: str) -> tuple[set[str], set[str]]:
    return _scope_blocked_exposure_sets(
        run_date, runs_root=RUNS_ROOT, project_root=PROJECT_ROOT
    )


def _spot_capital_bucket_ratios(
    position_qty: float,
    ledger_qty: dict[str, float],
) -> tuple[float, float, float]:
    """
    Split spot MV across buckets from ledger qty only.

    Unlike exposure attribution, unattributed shares (ledger total < IBKR line)
    are excluded from bucket capital rather than pushed into bucket 1.
    """
    if abs(position_qty) <= 1e-12:
        return 0.0, 0.0, 0.0
    b1 = float(ledger_qty.get("bucket_1", 0.0))
    b2 = float(ledger_qty.get("bucket_2", 0.0))
    b4 = float(ledger_qty.get("bucket_4", 0.0))
    ledger_total = b1 + b2 + b4
    if abs(ledger_total) > abs(position_qty) and abs(ledger_total) > 1e-12:
        scale = abs(position_qty) / abs(ledger_total)
        b1, b2, b4 = b1 * scale, b2 * scale, b4 * scale
    return b1 / position_qty, b2 / position_qty, b4 / position_qty


PROJECT_ROOT = Path(__file__).resolve().parent  # adjust if needed

IBKR_FLEX_SCRIPT = PROJECT_ROOT / "ibkr_flex.py"
# EOD accounting: PnL attribution + exposure totals share this script; exposure
# uses ratio-split bucket fields in totals.json (pair view in net_exposure_bucket_4.csv).
IBKR_ACCT_SCRIPT = PROJECT_ROOT / "scripts" / "ibkr_accounting_pnl.py"

# History / plot outputs
LEDGER_DIR = PROJECT_ROOT / "data" / "ledger"
RUNS_ROOT = PROJECT_ROOT / "data" / "runs"
PNL_HISTORY_CSV = LEDGER_DIR / "pnl_history.csv"
PLOT_PNG = LEDGER_DIR / "pnl_since_2026-02-27.png"
ATTRIBUTION_HISTORY_CSV = LEDGER_DIR / "pnl_attribution_history.csv"
PLOT_ATTRIBUTION_PNG = LEDGER_DIR / "pnl_attribution_timeseries.png"
START_DATE = "2026-02-27"
TOP_NET_EXPOSURE_MIN_ABS_USD = 500.0
TOP_NET_EXPOSURE_MAX_ROWS = 25
BUCKET_KEYS: tuple[str, ...] = ("bucket_1", "bucket_2", "bucket_3", "bucket_4")
STOCK_SLEEVE_BUCKETS: tuple[str, ...] = ("bucket_1", "bucket_2", "bucket_4")
BUCKET_LABELS: dict[str, str] = {
    "bucket_1": "Bucket 1",
    "bucket_2": "Bucket 2",
    "bucket_3": "Bucket 3",
    "bucket_4": "Bucket 4",
}
BUCKET_EMAIL_TITLES: dict[str, str] = {
    "bucket_1": "Bucket 1 — Levered (β > 1.5)",
    "bucket_2": "Bucket 2 — Standard (0 < β ≤ 1.5)",
    "bucket_3": "Bucket 3 — Inverse / Hedge (β < 0)",
    "bucket_4": "Bucket 4 — Inverse Decay Internalized (β < 0)",
}
SUBJECT_BUCKET_SHORT: dict[str, str] = {
    "bucket_1": "B1",
    "bucket_2": "B2",
    "bucket_3": "B3",
    "bucket_4": "B4",
}
SUBJECT_MAX_LEN = 120
# EOD email / history / plots: stock sleeves (B1+B2+B4) vs flow-inverse hedge (B3).
DISPLAY_PNL_KEYS: tuple[str, ...] = ("stock_sleeves", "bucket_3")
DISPLAY_LABELS: dict[str, str] = {
    "stock_sleeves": "Stock sleeves (B1+B2+B4)",
    "bucket_3": "Bucket 3 — Inverse / Hedge",
}
SUBJECT_B124_LABEL = "Buckets 1, 2, and 4"
SUBJECT_B3_LABEL = "B3"
# ROC = PnL / net capital when net exposure is positive (stock sleeves only).
BUCKETS_WITHOUT_ROC: frozenset[str] = frozenset({"bucket_3"})
DEFAULT_MAINT_MARGIN_LONG = 0.25
DEFAULT_MAINT_MARGIN_SHORT = 0.30

# Columns persisted in pnl_attribution_history.csv (YTD snapshot per run_date, same convention as pnl_history).
ATTRIBUTION_HISTORY_COLS: tuple[str, ...] = (
    "date",
    "long_realized_pnl",
    "long_unrealized_pnl",
    "short_realized_pnl",
    "short_unrealized_pnl",
    "gross_realized_pnl",
    "gross_unrealized_pnl",
    "other_fees",
    "borrow_fees",
    "short_credit_interest",
    "excluded_cash_interest_base",
    "dividends",
    "withholding_tax",
    "pil_dividends",
    "bond_interest",
    "strategy_total_pnl",
)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )


def load_outputs(run_date: str) -> tuple[Path, Path, Path, Path, dict]:
    outdir = PROJECT_ROOT / "data" / "runs" / run_date / "accounting"
    pnl_under = outdir / "pnl_by_underlying.csv"
    pnl_symbol = outdir / "pnl_by_symbol.csv"
    pnl_bucket = outdir / "pnl_by_bucket.csv"
    totals = outdir / "totals.json"
    if not pnl_under.exists():
        raise FileNotFoundError(f"Missing: {pnl_under}")
    if not totals.exists():
        raise FileNotFoundError(f"Missing: {totals}")
    totals_obj = json.loads(totals.read_text(encoding="utf-8"))
    return pnl_under, pnl_symbol, pnl_bucket, totals, totals_obj


def _format_underlying_section(
    df: pd.DataFrame,
    sym_df: pd.DataFrame,
    section_label: str | None = None,
    *,
    max_rows: int | None = None,
) -> tuple[str, float]:
    """
    Format a single section of the underlying table.
    Returns (formatted_text, section_total).
    """
    # Drop rows with missing underlying (delisted ETFs, mapping gaps)
    df = df.dropna(subset=["underlying"]).copy()
    df["underlying"] = df["underlying"].astype(str)
    if not sym_df.empty and "underlying" in sym_df.columns:
        sym_df = sym_df.dropna(subset=["underlying"]).copy()
        sym_df["underlying"] = sym_df["underlying"].astype(str)
    if not sym_df.empty and "symbol" in sym_df.columns:
        sym_df = sym_df.dropna(subset=["symbol"]).copy()
        sym_df["symbol"] = sym_df["symbol"].astype(str)

    if df.empty:
        return "(no data)", 0.0

    full_df = df.sort_values("total_pnl", ascending=False)
    section_total = float(full_df["total_pnl"].sum())
    truncated = 0
    display_df = full_df
    if max_rows is not None and len(full_df) > max_rows:
        display_df = full_df.head(max_rows)
        truncated = len(full_df) - max_rows
    df = display_df

    # Column widths
    all_labels = list(df["underlying"].astype(str))
    if not sym_df.empty and "symbol" in sym_df.columns:
        all_labels += ["  " + s for s in sym_df["symbol"].astype(str)]
    all_labels += ["TOTAL"]
    label_width = max(12, max(len(s) for s in all_labels))

    all_pnl_strs = [f"{v:,.2f}" for v in list(df["total_pnl"]) + [section_total]]
    if not sym_df.empty and "total_pnl" in sym_df.columns:
        all_pnl_strs += [f"{v:,.2f}" for v in sym_df["total_pnl"]]
    pnl_width = max(12, max(len(s) for s in all_pnl_strs))

    lines: list[str] = []
    if section_label:
        lines.append(f"--- {section_label} ---")
    header = f"{'UNDERLYING / SYMBOL'.ljust(label_width)}  {'TOTAL_PNL'.rjust(pnl_width)}"
    lines.append(header)
    lines.append("-" * (label_width + 2 + pnl_width))

    for _, r in df.iterrows():
        underlying = str(r["underlying"])
        pnl_str = f"{float(r['total_pnl']):,.2f}"
        lines.append(f"{underlying.ljust(label_width)}  {pnl_str.rjust(pnl_width)}")

        # Indented symbol rows for this underlying
        if not sym_df.empty and "underlying" in sym_df.columns and "symbol" in sym_df.columns:
            syms = sym_df[sym_df["underlying"] == underlying].sort_values("total_pnl", ascending=False)
            for _, sr in syms.iterrows():
                sym_label = "  " + str(sr["symbol"])
                sym_pnl = f"{float(sr['total_pnl']):,.2f}"
                detail = ""
                if "realized_pnl" in sr and "unrealized_pnl" in sr:
                    r_pnl = float(sr["realized_pnl"])
                    u_pnl = float(sr["unrealized_pnl"])
                    detail = f"  (r: {r_pnl:,.2f}  u: {u_pnl:,.2f})"
                lines.append(f"{sym_label.ljust(label_width)}  {sym_pnl.rjust(pnl_width)}{detail}")

    lines.append("-" * (label_width + 2 + pnl_width))
    total_str = f"{section_total:,.2f}"
    lines.append(f"{'TOTAL'.ljust(label_width)}  {total_str.rjust(pnl_width)}")
    if truncated:
        lines.append(f"  ... and {truncated} more underlyings (table truncated)")

    return "\n".join(lines), section_total


def format_underlying_table(pnl_under_csv: Path, pnl_symbol_csv: Path) -> tuple[str, float]:
    """
    Returns:
      - formatted table (plain text) of total_pnl by underlying (buckets 1, 2 & 4),
        with per-symbol PnL rows indented underneath each underlying
      - total_pnl sum
    """
    df = pd.read_csv(pnl_under_csv)
    if "underlying" not in df.columns or "total_pnl" not in df.columns:
        raise ValueError("pnl_by_underlying.csv missing required columns")

    df["total_pnl"] = pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0)

    # Load per-symbol detail if available; filter to bucket_12 if bucket column exists
    sym_df: pd.DataFrame = pd.DataFrame()
    if pnl_symbol_csv.exists():
        try:
            sym_df = pd.read_csv(pnl_symbol_csv)
            sym_df["total_pnl"] = pd.to_numeric(sym_df["total_pnl"], errors="coerce").fillna(0.0)
            if "bucket" in sym_df.columns:
                sym_df = sym_df[sym_df["bucket"].isin(["bucket_1", "bucket_2", "bucket_4"])]
        except Exception:
            sym_df = pd.DataFrame()

    table_text, total = _format_underlying_section(df, sym_df)
    return table_text, total


def format_bucket_3_pnl(pnl_b3_csv: Path) -> tuple[str, float]:
    """
    Format Bucket 3 (inverse/hedge) PnL as a plain-text table by symbol.
    Returns (table_str, total).
    """
    if not pnl_b3_csv.exists():
        return "(no bucket 3 data)", 0.0

    df = pd.read_csv(pnl_b3_csv)
    if df.empty or "total_pnl" not in df.columns:
        return "(no bucket 3 positions)", 0.0

    df["total_pnl"] = pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values("total_pnl", ascending=False)
    total = float(df["total_pnl"].sum())

    all_labels = list(df["symbol"]) + ["TOTAL"]
    label_width = max(12, max(len(s) for s in all_labels))
    all_pnl = [f"{v:,.2f}" for v in list(df["total_pnl"]) + [total]]
    pnl_width = max(12, max(len(s) for s in all_pnl))

    lines: list[str] = []
    header = f"{'SYMBOL'.ljust(label_width)}  {'TOTAL_PNL'.rjust(pnl_width)}"
    lines.append(header)
    lines.append("-" * (label_width + 2 + pnl_width))

    for _, r in df.iterrows():
        sym = str(r["symbol"])
        pnl_str = f"{float(r['total_pnl']):,.2f}"
        detail = ""
        if "realized_pnl" in r and "unrealized_pnl" in r:
            r_pnl = float(r["realized_pnl"])
            u_pnl = float(r["unrealized_pnl"])
            detail = f"  (r: {r_pnl:,.2f}  u: {u_pnl:,.2f})"
        lines.append(f"{sym.ljust(label_width)}  {pnl_str.rjust(pnl_width)}{detail}")

    lines.append("-" * (label_width + 2 + pnl_width))
    lines.append(f"{'TOTAL'.ljust(label_width)}  {f'{total:,.2f}'.rjust(pnl_width)}")

    return "\n".join(lines), total


def format_bucket_pnl_section(
    bucket: str,
    pnl_csv: Path,
    pnl_symbol_csv: Path,
    *,
    top_n: int | None = None,
) -> tuple[str, float]:
    """Format per-bucket PnL from accounting ``pnl_<bucket>.csv``."""
    if bucket == "bucket_3":
        return format_bucket_3_pnl(pnl_csv)

    if not pnl_csv.exists():
        return f"(no {BUCKET_LABELS[bucket]} data)", 0.0

    df = pd.read_csv(pnl_csv)
    if df.empty or "total_pnl" not in df.columns:
        return f"(no {BUCKET_LABELS[bucket]} positions)", 0.0

    df["total_pnl"] = pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0)
    sym_df = pd.DataFrame()
    if pnl_symbol_csv.exists():
        try:
            sym_df = pd.read_csv(pnl_symbol_csv)
            sym_df["total_pnl"] = pd.to_numeric(sym_df["total_pnl"], errors="coerce").fillna(0.0)
            if "bucket" in sym_df.columns:
                sym_df = sym_df[sym_df["bucket"] == bucket]
        except Exception:
            sym_df = pd.DataFrame()

    if "underlying" in df.columns:
        return _format_underlying_section(df, sym_df, max_rows=top_n)

    if "symbol" not in df.columns:
        return f"({bucket} data missing underlying/symbol columns)", 0.0

    df = df.dropna(subset=["symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str)
    full_total = float(df["total_pnl"].sum())
    display_df = df.sort_values("total_pnl", ascending=False)
    truncated = 0
    if top_n is not None and len(display_df) > top_n:
        display_df = display_df.head(top_n)
        truncated = len(df) - top_n
    all_labels = list(display_df["symbol"]) + ["TOTAL"]
    label_width = max(12, max(len(s) for s in all_labels))
    all_pnl = [f"{v:,.2f}" for v in list(display_df["total_pnl"]) + [full_total]]
    pnl_width = max(12, max(len(s) for s in all_pnl))
    lines: list[str] = []
    header = f"{'SYMBOL'.ljust(label_width)}  {'TOTAL_PNL'.rjust(pnl_width)}"
    lines.append(header)
    lines.append("-" * (label_width + 2 + pnl_width))
    for _, r in display_df.iterrows():
        sym = str(r["symbol"])
        pnl_str = f"{float(r['total_pnl']):,.2f}"
        lines.append(f"{sym.ljust(label_width)}  {pnl_str.rjust(pnl_width)}")
    lines.append("-" * (label_width + 2 + pnl_width))
    lines.append(f"{'TOTAL'.ljust(label_width)}  {f'{full_total:,.2f}'.rjust(pnl_width)}")
    if truncated:
        lines.append(f"  ... and {truncated} more symbols (table truncated)")
    return "\n".join(lines), full_total


def format_bucket_exposure_section(
    bucket: str,
    exposure_csv: Path,
    *,
    exposure_detail_csv: Path | None = None,
    blocked_keys: set[str] | None = None,
    top_n: int | None = None,
) -> tuple[str, float, float]:
    """Format per-bucket net/gross exposure from accounting CSVs."""
    if not exposure_csv.exists():
        return f"(no {BUCKET_LABELS[bucket]} exposure data)", 0.0, 0.0

    rollup_df = pd.read_csv(exposure_csv)
    if rollup_df.empty or "net_notional_usd" not in rollup_df.columns:
        return f"(no {BUCKET_LABELS[bucket]} exposure positions)", 0.0, 0.0

    if "underlying" not in rollup_df.columns and "symbol" in rollup_df.columns:
        rollup_df = rollup_df.rename(columns={"symbol": "underlying"})

    rollup_df["net_notional_usd"] = pd.to_numeric(
        rollup_df["net_notional_usd"], errors="coerce"
    ).fillna(0.0)
    rollup_df["gross_notional_usd"] = pd.to_numeric(
        rollup_df.get("gross_notional_usd", rollup_df["net_notional_usd"]),
        errors="coerce",
    ).fillna(0.0)

    if blocked_keys:
        rollup_df = _filter_exposure_df(rollup_df, blocked_keys)

    net_total = float(rollup_df["net_notional_usd"].sum())
    gross_total = float(rollup_df["gross_notional_usd"].sum())

    display_df = rollup_df.sort_values(
        "net_notional_usd", key=lambda s: s.abs(), ascending=False
    )
    truncated = 0
    if top_n is not None and len(display_df) > top_n:
        truncated = len(display_df) - top_n
        display_df = display_df.head(top_n)

    table_text, _, _ = format_exposure_table(
        display_df.sort_values("net_notional_usd", ascending=False)
    )
    lines = [table_text]
    if truncated:
        lines.append(f"  ... and {truncated} more underlyings (table truncated)")

    if bucket == "bucket_4" and exposure_detail_csv and exposure_detail_csv.exists():
        try:
            leg_df = pd.read_csv(exposure_detail_csv)
            leg_df["net_notional_usd"] = pd.to_numeric(
                leg_df["net_notional_usd"], errors="coerce"
            ).fillna(0.0)
            if blocked_keys and "underlying" in leg_df.columns:
                leg_df = leg_df[
                    ~leg_df["underlying"].astype(str).isin(blocked_keys)
                ].copy()
            if not leg_df.empty and "leg_type" in leg_df.columns:
                lines.extend(
                    [
                        "",
                        "--- B4 exposure legs (inverse ETF short + structural underlying short) ---",
                    ]
                )
                for underlying in sorted(leg_df["underlying"].astype(str).unique()):
                    sub = leg_df[leg_df["underlying"].astype(str) == underlying].sort_values(
                        ["leg_type", "symbol"], ascending=[True, True]
                    )
                    for _, r in sub.iterrows():
                        sym = str(r.get("symbol", ""))
                        leg = str(r.get("leg_type", ""))
                        net_v = float(r.get("net_notional_usd", 0.0))
                        tag = ""
                        if leg == "underlying":
                            tag = " short" if net_v < 0 else " long"
                        lines.append(f"  {underlying} / {sym} ({leg}{tag})  {net_v:,.2f}")
        except Exception:
            pass

    return "\n".join(lines), net_total, gross_total


def format_bucket_4_pnl(
    pnl_b4_csv: Path,
    pnl_symbol_csv: Path,
) -> tuple[str, float]:
    """Format Bucket 4 PnL (wrapper for tests / legacy callers)."""
    return format_bucket_pnl_section("bucket_4", pnl_b4_csv, pnl_symbol_csv)


def format_bucket_4_exposure(
    exposure_b4_csv: Path,
    exposure_b4_detail_csv: Path,
) -> tuple[str, float, float]:
    """Format Bucket 4 exposure (wrapper for tests / legacy callers)."""
    return format_bucket_exposure_section(
        "bucket_4",
        exposure_b4_csv,
        exposure_detail_csv=exposure_b4_detail_csv,
    )


def format_bucket_email_block(
    bucket: str,
    *,
    headline_pnl: float,
    pnl_csv: Path,
    pnl_symbol_csv: Path,
    exposure_csv: Path,
    exposure_detail_csv: Path | None = None,
    blocked_keys: set[str],
    pnl_tol: float = 0.01,
) -> str:
    """Full email section for one bucket: PnL detail + net/gross exposure."""
    title = BUCKET_EMAIL_TITLES.get(bucket, BUCKET_LABELS[bucket])
    pnl_table, csv_sum = format_bucket_pnl_section(bucket, pnl_csv, pnl_symbol_csv)
    exp_table, net, gross = format_bucket_exposure_section(
        bucket,
        exposure_csv,
        exposure_detail_csv=exposure_detail_csv,
        blocked_keys=blocked_keys,
    )

    lines = [
        "════════════════════════════════════════",
        title,
        f"  YTD PnL (totals.json): {headline_pnl:,.2f}",
    ]
    if abs(csv_sum - headline_pnl) > pnl_tol:
        lines.append(
            f"  Note: CSV detail sum {csv_sum:,.2f} != headline {headline_pnl:,.2f}"
        )
    lines.extend(
        [
            "",
            "PnL detail:",
            "----------------------------------------",
            pnl_table,
            "----------------------------------------",
            "",
            f"Net exposure — net: {net:,.2f}  |  gross: {gross:,.2f}",
            "----------------------------------------",
            exp_table,
            "----------------------------------------",
            "",
        ]
    )
    return "\n".join(lines)


def format_bucket_table(pnl_bucket_csv: Path) -> str:
    """
    Returns a formatted plain-text table of PnL by bucket, with symbol lists.
    """
    if not pnl_bucket_csv.exists():
        return "(pnl_by_bucket.csv not found)"

    df = pd.read_csv(pnl_bucket_csv)
    if df.empty or "bucket" not in df.columns or "total_pnl" not in df.columns:
        return "(bucket data unavailable)"

    df["total_pnl"] = pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0)
    df = df.sort_values("total_pnl", ascending=False)

    # Friendly label mapping
    _LABELS = {
        "bucket_1":  "Bucket 1 — Levered (β > 1.5)",
        "bucket_2":  "Bucket 2 — Standard (0 < β ≤ 1.5)",
        "bucket_3":  "Bucket 3 — Inverse / Hedge (β < 0)",
        "bucket_4":  "Bucket 4 — Inverse Decay Internalized (β < 0)",
        # Legacy keys
        "bucket_12": "Bucket 1&2 — Long/Short (β ≥ 0)",
    }

    lines: list[str] = []
    for _, row in df.iterrows():
        bucket_key = str(row["bucket"])
        label = _LABELS.get(bucket_key, bucket_key)
        pnl = float(row["total_pnl"])
        lines.append(f"  {label}")
        lines.append(f"    PnL: {pnl:>12,.2f}")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_bucket_ytd_headline(bucket_pnl: dict[str, float]) -> str:
    """Compact YTD PnL by accounting bucket for the email headline block."""
    lines = ["YTD PnL BY BUCKET (accounting):"]
    for bucket in BUCKET_KEYS:
        pnl = float(bucket_pnl.get(bucket, 0.0) or 0.0)
        short = SUBJECT_BUCKET_SHORT[bucket]
        lines.append(f"  {short} ({BUCKET_LABELS[bucket]}): {pnl:,.2f}")
    stock = _stock_sleeves_pnl_from_buckets(bucket_pnl)
    lines.append(f"  Stock sleeves (B1+B2+B4): {stock:,.2f}")
    return "\n".join(lines)


def compute_bucket_daily_deltas(
    run_date: str,
    current_buckets: dict[str, float],
) -> dict[str, float]:
    """Per-bucket YTD change vs the prior accounting run (account daily PnL)."""
    prior_date = _prior_accounting_run_date(run_date)
    if not prior_date:
        return {b: float(current_buckets.get(b, 0.0) or 0.0) for b in BUCKET_KEYS}
    prior_path = RUNS_ROOT / prior_date / "accounting" / "totals.json"
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception:
        return {b: float(current_buckets.get(b, 0.0) or 0.0) for b in BUCKET_KEYS}
    prior_buckets = _bucket_pnl_from_totals(prior)
    return {
        b: float(current_buckets.get(b, 0.0) or 0.0) - float(prior_buckets.get(b, 0.0) or 0.0)
        for b in BUCKET_KEYS
    }


def format_bucket_daily_headline(daily_buckets: dict[str, float]) -> str:
    """One-line daily PnL change per bucket."""
    parts = [
        f"{SUBJECT_BUCKET_SHORT[b]}: {float(daily_buckets.get(b, 0.0) or 0.0):+,.0f}"
        for b in BUCKET_KEYS
    ]
    total_daily = sum(float(daily_buckets.get(b, 0.0) or 0.0) for b in BUCKET_KEYS)
    parts.append(f"Total: {total_daily:+,.0f}")
    return "Daily PnL change: " + " | ".join(parts)


def format_headline_pnl_block(
    bucket_pnl: dict[str, float],
    *,
    total_pnl: float,
    daily_buckets: dict[str, float] | None = None,
) -> str:
    """Email headline: YTD per bucket + optional daily deltas."""
    lines = [
        "HEADLINE YTD PnL",
        f"  Total: {total_pnl:,.2f}",
    ]
    for bucket in BUCKET_KEYS:
        pnl = float(bucket_pnl.get(bucket, 0.0) or 0.0)
        short = SUBJECT_BUCKET_SHORT[bucket]
        lines.append(f"  {short}: {pnl:,.2f}")
    stock = _stock_sleeves_pnl_from_buckets(bucket_pnl)
    lines.append(f"  Stock sleeves (B1+B2+B4): {stock:,.2f}")
    if daily_buckets is not None:
        lines.append(format_bucket_daily_headline(daily_buckets))
    return "\n".join(lines)


def format_accounting_method_line(totals: dict) -> str:
    """One-line note of the accounting attribution settings used for this run."""
    keys = (
        "b12_spot_pnl_method",
        "b12_pnl_mode",
        "plan_b4_pnl_mode",
        "b12_spot_exposure_method",
    )
    parts = [f"{k}={totals[k]}" for k in keys if totals.get(k)]
    if not parts:
        return ""
    return "Accounting: " + ", ".join(parts)


def ensure_ledger_dir() -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)


PNL_HISTORY_BUCKET_COLS = tuple(f"pnl_{b}" for b in BUCKET_KEYS)
PNL_HISTORY_DISPLAY_PNL_COLS = ("pnl_stock_sleeves", "pnl_bucket_3")
PNL_HISTORY_DISPLAY_CAPITAL_COLS = tuple(
    f"{metric}_{group}"
    for metric in ("net_capital", "gross_capital", "margin_req")
    for group in ("stock_sleeves", "bucket_3")
)
PNL_HISTORY_CAPITAL_COLS = tuple(
    f"{metric}_{bucket}"
    for metric in ("net_capital", "gross_capital", "margin_req")
    for bucket in BUCKET_KEYS
)
PNL_HISTORY_RETURN_BASE_COLS = tuple(
    dict.fromkeys(
        (
            *PNL_HISTORY_BUCKET_COLS,
            *PNL_HISTORY_DISPLAY_PNL_COLS,
            *PNL_HISTORY_CAPITAL_COLS,
            *PNL_HISTORY_DISPLAY_CAPITAL_COLS,
        )
    )
)


def _ensure_pnl_history_derived_cols(hist: pd.DataFrame) -> pd.DataFrame:
    """Ensure consolidated sleeve columns exist alongside per-bucket history columns."""
    if hist is None or hist.empty:
        return hist
    out = hist.copy()

    def _col_series(col: str, default: float = 0.0) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype=float)

    for bucket in BUCKET_KEYS:
        col = f"pnl_{bucket}"
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "pnl_stock_sleeves" not in out.columns:
        if any(c in out.columns for c in ("pnl_bucket_1", "pnl_bucket_2", "pnl_bucket_4")):
            out["pnl_stock_sleeves"] = (
                _col_series("pnl_bucket_1") + _col_series("pnl_bucket_2") + _col_series("pnl_bucket_4")
            )
        else:
            out["pnl_stock_sleeves"] = np.nan
    if "pnl_bucket_3" in out.columns:
        out["pnl_bucket_3"] = pd.to_numeric(out["pnl_bucket_3"], errors="coerce")
    for metric in ("net_capital", "gross_capital", "margin_req"):
        ss_col = f"{metric}_stock_sleeves"
        if ss_col not in out.columns:
            if any(f"{metric}_{b}" in out.columns for b in STOCK_SLEEVE_BUCKETS):
                out[ss_col] = sum(_col_series(f"{metric}_{b}") for b in STOCK_SLEEVE_BUCKETS)
            else:
                out[ss_col] = np.nan
        b3_col = f"{metric}_bucket_3"
        if b3_col not in out.columns:
            out[b3_col] = _col_series(f"{metric}_bucket_3", default=np.nan) if f"{metric}_bucket_3" in out.columns else np.nan
    return out


def _migrate_pnl_history_to_display_cols(hist: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for callers expecting the old migrate name."""
    return _ensure_pnl_history_derived_cols(hist)


def _canonical_pnl_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Persist per-bucket and consolidated sleeve columns in pnl_history.csv."""
    if hist is None or hist.empty:
        return hist
    out = _ensure_pnl_history_derived_cols(hist)
    keep = [
        "date",
        "total_pnl",
        *PNL_HISTORY_BUCKET_COLS,
        *PNL_HISTORY_DISPLAY_PNL_COLS,
        *PNL_HISTORY_CAPITAL_COLS,
        *PNL_HISTORY_DISPLAY_CAPITAL_COLS,
    ]
    cols: list[str] = []
    seen: set[str] = set()
    for c in keep:
        if c in out.columns and c not in seen:
            cols.append(c)
            seen.add(c)
    return out[cols].copy()


def _stock_sleeves_pnl_from_buckets(buckets: dict[str, float]) -> float:
    return sum(float(buckets.get(b, 0.0) or 0.0) for b in STOCK_SLEEVE_BUCKETS)


def _consolidate_capital_snapshot(snap: dict[str, float]) -> dict[str, float]:
    """Roll B1+B2+B4 capital into stock_sleeves; keep bucket_3 separate."""
    out = {c: 0.0 for c in PNL_HISTORY_DISPLAY_CAPITAL_COLS}
    for metric in ("net_capital", "gross_capital", "margin_req"):
        out[f"{metric}_stock_sleeves"] = sum(
            float(snap.get(f"{metric}_{b}", 0.0) or 0.0) for b in STOCK_SLEEVE_BUCKETS
        )
        out[f"{metric}_bucket_3"] = float(snap.get(f"{metric}_bucket_3", 0.0) or 0.0)
    return out


def _empty_bucket_capital_snapshot() -> dict[str, float]:
    return {c: 0.0 for c in PNL_HISTORY_CAPITAL_COLS}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if pd.isna(out):
            return default
        return float(out)
    except Exception:
        return default


def _load_flow_universe_sets(run_date: str) -> tuple[set[str], set[str]]:
    """Flow-program shorts and low-delta symbols (bucket 3 overlay), matching accounting."""
    flow_short: set[str] = set()
    flow_low: set[str] = set()
    cfg_path = PROJECT_ROOT / "config" / "strategy_config.yml"
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        portfolio_cfg = cfg.get("portfolio", {}) or {}
        sleeves_cfg = portfolio_cfg.get("sleeves", {}) or {}
        flow_cfg = sleeves_cfg.get("flow_program", {}) or {}
        flow_universe_cfg = flow_cfg.get("universe", {}) or {}
        flow_short = {
            canonical_symbol(x)
            for x in (flow_universe_cfg.get("shorts", []) or [])
            if str(x).strip()
        }
    totals_path = RUNS_ROOT / run_date / "accounting" / "totals.json"
    if totals_path.exists():
        try:
            obj = json.loads(totals_path.read_text(encoding="utf-8"))
            for key in (
                "bucket3_flow_low_delta_symbols",
                "bucket2_flow_low_delta_symbols",
            ):
                flow_low |= {
                    canonical_symbol(x)
                    for x in (obj.get(key) or [])
                    if str(x).strip()
                }
        except Exception:
            pass
    return flow_short, flow_low


def _eod_bucket_pnl_continuity_enabled() -> bool:
    """When false, EOD uses raw accounting bucket_pnl (post-restate ledger_fifo default)."""
    cfg_path = PROJECT_ROOT / "config" / "strategy_config.yml"
    if not cfg_path.exists():
        return False
    cfg = load_config(cfg_path)
    accounting_cfg = cfg.get("accounting", {}) or {}
    return bool(accounting_cfg.get("eod_bucket_pnl_continuity", False))


def _bucket_pnl_from_totals(totals: dict) -> dict[str, float]:
    """Canonical cumulative strategy PnL by bucket from accounting totals.json."""
    bp = totals.get("bucket_pnl") or {}
    return {b: float(bp.get(b, 0.0) or 0.0) for b in BUCKET_KEYS}


def _prior_accounting_run_date(run_date: str) -> str | None:
    """Latest run folder before ``run_date`` that has accounting totals."""
    try:
        target = pd.to_datetime(run_date).normalize()
    except (ValueError, TypeError):
        return None
    best: pd.Timestamp | None = None
    best_name: str | None = None
    if not RUNS_ROOT.is_dir():
        return None
    for child in RUNS_ROOT.iterdir():
        if not child.is_dir():
            continue
        try:
            dt = pd.to_datetime(child.name).normalize()
        except (ValueError, TypeError):
            continue
        if dt >= target:
            continue
        if not (child / "accounting" / "totals.json").exists():
            continue
        if best is None or dt > best:
            best = dt
            best_name = child.name
    return best_name


def _bucket_daily_jump_threshold(account_daily: float) -> float:
    return max(50_000.0, 5.0 * abs(account_daily) + 1.0)


def _allocate_daily_by_abs_prior(
    amount: float, prior_buckets: dict[str, float], buckets: list[str]
) -> dict[str, float]:
    weights = {b: abs(float(prior_buckets.get(b, 0.0) or 0.0)) for b in buckets}
    wsum = sum(weights.values())
    if wsum <= 1e-12:
        share = 1.0 / len(buckets)
        return {b: amount * share for b in buckets}
    return {b: amount * (weights[b] / wsum) for b in buckets}


def _bucket_pnl_continuity_adjusted(
    *,
    prior_buckets: dict[str, float],
    current_buckets: dict[str, float],
    account_total: float,
    prior_account_total: float,
) -> dict[str, float]:
    """
    When a single-day accounting run re-attributes YTD history across buckets,
    keep plausible per-bucket daily moves from the raw run and re-spread only
    the outlier sleeves (large YTD jumps with a small account daily change).
    """
    account_daily = float(account_total) - float(prior_account_total)
    jump_tol = _bucket_daily_jump_threshold(account_daily)
    raw_daily = {
        b: float(current_buckets.get(b, 0.0) or 0.0) - float(prior_buckets.get(b, 0.0) or 0.0)
        for b in BUCKET_KEYS
    }
    trusted: list[str] = []
    outlier: list[str] = []
    for b in BUCKET_KEYS:
        ytd_jump = abs(float(current_buckets.get(b, 0.0) or 0.0) - float(prior_buckets.get(b, 0.0) or 0.0))
        if ytd_jump <= jump_tol:
            trusted.append(b)
        else:
            outlier.append(b)

    daily: dict[str, float] = {b: raw_daily[b] for b in trusted}
    trusted_sum = sum(daily.values())
    remainder = account_daily - trusted_sum

    if outlier:
        alloc = _allocate_daily_by_abs_prior(remainder, prior_buckets, outlier)
        daily.update(alloc)

    out = {b: float(prior_buckets.get(b, 0.0) or 0.0) + daily[b] for b in BUCKET_KEYS}
    residual = float(account_total) - sum(out.values())
    if abs(residual) > 0.01:
        # Rounding / empty-outlier edge case: land residual on bucket_4.
        out["bucket_4"] = out.get("bucket_4", 0.0) + residual
    return out


def _scale_pnl_bucket_csv(path: Path, new_total: float) -> None:
    """Scale per-bucket PnL CSV rows so they sum to continuity-adjusted total."""
    if not path.is_file():
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    if df.empty or "total_pnl" not in df.columns:
        return
    old_sum = float(pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0).sum())
    if abs(old_sum) <= 1e-9:
        return
    ratio = float(new_total) / old_sum
    for col in (
        "total_pnl",
        "realized_pnl",
        "unrealized_pnl",
        "borrow_fees",
        "short_credit_interest",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) * ratio
    df.to_csv(path, index=False)


def persist_totals_dashboard_fields(
    run_date: str,
    totals: dict,
    capital_snap: dict[str, float],
) -> dict:
    """Persist capital snapshot and NAV on totals.json for the risk dashboard."""
    totals = dict(totals)
    consolidated = _consolidate_capital_snapshot(capital_snap)
    totals["capital_snapshot"] = {
        **consolidated,
        **{k: float(v) for k, v in capital_snap.items()},
    }
    nav_raw = os.getenv("MAGIS_NAV_USD", "").strip()
    if nav_raw:
        try:
            nav_f = float(nav_raw)
            if nav_f > 0:
                totals["nav_usd"] = nav_f
                totals["nav_source"] = "MAGIS_NAV_USD"
        except ValueError:
            pass
    outdir = RUNS_ROOT / run_date / "accounting"
    (outdir / "totals.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    return totals


def apply_bucket_pnl_continuity(run_date: str, totals: dict) -> dict:
    """
    If bucket YTD jumps are implausible vs the prior run, rewrite bucket_pnl in
    totals.json and pnl_by_bucket.csv using prior-day bucket weights.

    Disabled by default (``accounting.eod_bucket_pnl_continuity: false``) so EOD
    reports match restated ledger_fifo accounting outputs.
    """
    if not _eod_bucket_pnl_continuity_enabled():
        return totals

    prior_date = _prior_accounting_run_date(run_date)
    if not prior_date:
        return totals

    prior_path = RUNS_ROOT / prior_date / "accounting" / "totals.json"
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except Exception:
        return totals

    prior_buckets = _bucket_pnl_from_totals(prior)
    current_buckets = _bucket_pnl_from_totals(totals)
    prior_total = float(prior.get("total_pnl", 0.0) or 0.0)
    account_total = float(totals.get("total_pnl", 0.0) or 0.0)
    account_daily = float(account_total) - float(prior_total)
    jump_tol = _bucket_daily_jump_threshold(account_daily)
    max_bucket_jump = max(
        abs(current_buckets[b] - prior_buckets[b]) for b in BUCKET_KEYS
    )
    # e.g. May 19: |B1| jumped ~200k while account moved ~7k.
    if max_bucket_jump <= jump_tol:
        return totals

    adjusted = _bucket_pnl_continuity_adjusted(
        prior_buckets=prior_buckets,
        current_buckets=current_buckets,
        account_total=account_total,
        prior_account_total=prior_total,
    )
    totals = dict(totals)
    totals["bucket_pnl"] = adjusted
    outdir = RUNS_ROOT / run_date / "accounting"
    (outdir / "totals.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")

    bucket_csv = outdir / "pnl_by_bucket.csv"
    if bucket_csv.exists():
        try:
            df = pd.read_csv(bucket_csv)
            for b in BUCKET_KEYS:
                m = df["bucket"] == b if "bucket" in df.columns else pd.Series(dtype=bool)
                if m.any():
                    df.loc[m, "total_pnl"] = adjusted[b]
            df.to_csv(bucket_csv, index=False)
        except Exception:
            pass

    for b in BUCKET_KEYS:
        _scale_pnl_bucket_csv(outdir / f"pnl_{b}.csv", adjusted[b])

    print(
        f"[EOD] Bucket PnL continuity: adjusted {run_date} buckets using {prior_date} "
        f"weights (max raw jump {max_bucket_jump:,.0f} vs account daily {account_daily:,.0f})."
    )
    return totals


def _etf_capital_bucket(
    sym: str,
    etf_to_delta: dict[str, float],
    *,
    flow_short_set: set[str],
    flow_low_delta_syms: set[str],
    b4_etf_syms: set[str],
) -> str | None:
    """
    Map an ETF line to a capital bucket using the same beta rules as ibkr_accounting
    exposure. Returns None for bucket-3-only symbols (excluded from b124 capital).
    """
    beta = float(etf_to_delta.get(sym, 1.0))
    if beta < 0 and sym in flow_short_set:
        return None
    if sym in flow_low_delta_syms and 0.0 < beta <= 1.5:
        return None
    if sym in b4_etf_syms:
        return "bucket_4"
    if beta < 0:
        return "bucket_4"
    if beta > 1.5:
        return "bucket_1"
    if beta > 0:
        return "bucket_2"
    return None


def _normalise_bucket_key(value: object) -> str | None:
    s = str(value or "").strip()
    return s if s in BUCKET_KEYS else None


def _pnl_symbol_bucket_map(pnl_symbol: pd.DataFrame) -> dict[str, set[str]]:
    if pnl_symbol is None or pnl_symbol.empty or "symbol" not in pnl_symbol.columns or "bucket" not in pnl_symbol.columns:
        return {}
    out: dict[str, set[str]] = {}
    for _, row in pnl_symbol.iterrows():
        sym = canonical_symbol(str(row.get("symbol", "") or ""))
        bucket = _normalise_bucket_key(row.get("bucket"))
        if sym and bucket:
            out.setdefault(sym, set()).add(bucket)
    return out


def _etf_maps_from_screened(screened: pd.DataFrame) -> tuple[dict[str, str], dict[str, float]]:
    """Build ETF→underlying and ETF→delta maps from a screened DataFrame."""
    if screened is None or screened.empty:
        return {}, {}
    cols = {c.lower(): c for c in screened.columns}
    etf_col = next((cols[k] for k in ("etf", "symbol", "ticker", "etf_symbol") if k in cols), None)
    under_col = next(
        (cols[k] for k in ("underlying", "underlyingsymbol", "underlying_symbol", "root") if k in cols),
        None,
    )
    delta_col = next((cols[k] for k in ("delta", "beta", "leverage", "lev") if k in cols), None)
    if etf_col is None or under_col is None:
        return {}, {}
    u = screened[[etf_col, under_col] + ([delta_col] if delta_col else [])].dropna(subset=[etf_col, under_col])
    u = u.copy()
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)
    etf_to_under = dict(zip(u[etf_col], u[under_col]))
    if delta_col:
        u[delta_col] = pd.to_numeric(u[delta_col], errors="coerce").fillna(1.0)
        etf_to_delta = dict(zip(u[etf_col], u[delta_col]))
    else:
        etf_to_delta = {k: 1.0 for k in etf_to_under}
    return etf_to_under, etf_to_delta


def _screened_margin_and_bucket_maps(screened: pd.DataFrame) -> tuple[set[str], dict[str, str], dict[str, tuple[float, float]]]:
    etf_symbols: set[str] = set()
    bucket_by_symbol: dict[str, str] = {}
    margin_by_symbol: dict[str, tuple[float, float]] = {}
    if screened is None or screened.empty or "ETF" not in screened.columns:
        return etf_symbols, bucket_by_symbol, margin_by_symbol

    for _, row in screened.iterrows():
        sym = canonical_symbol(str(row.get("ETF", "") or ""))
        if not sym:
            continue
        etf_symbols.add(sym)
        bucket = _normalise_bucket_key(row.get("bucket"))
        if bucket:
            bucket_by_symbol[sym] = bucket
        margin_by_symbol[sym] = (
            _safe_float(row.get("maint_pct_long"), DEFAULT_MAINT_MARGIN_LONG),
            _safe_float(row.get("maint_pct_short"), DEFAULT_MAINT_MARGIN_SHORT),
        )
    return etf_symbols, bucket_by_symbol, margin_by_symbol


def _lot_qty_map(lot_state: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if lot_state is None or lot_state.empty or "underlying" not in lot_state.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for _, row in lot_state.iterrows():
        under = canonical_symbol(str(row.get("underlying", "") or ""))
        if not under:
            continue
        out[under] = {
            "bucket_1": _safe_float(row.get("qty_b1"), 0.0),
            "bucket_2": _safe_float(row.get("qty_b2"), 0.0),
            "bucket_4": _safe_float(row.get("qty_b4"), 0.0),
        }
    return out


def compute_bucket_capital_snapshot(
    positions: pd.DataFrame,
    pnl_symbol: pd.DataFrame,
    screened: pd.DataFrame,
    lot_state: pd.DataFrame | None = None,
    *,
    blocked_symbols: set[str] | None = None,
    blocked_underlyings: set[str] | None = None,
    run_date: str | None = None,
    flow_short_set: set[str] | None = None,
    flow_low_delta_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
) -> dict[str, float]:
    """
    Compute current per-bucket capital bases from open positions (buckets 1, 2, 4).

    Uses the same universe filters and ETF beta→bucket rules as ibkr_accounting
    exposure (excluding bucket-3 flow symbols). Only symbols listed in
    etf_screened_today.csv (ETF or Underlying column) are included. Net capital
    is signed MV; gross is |MV|; margin uses screened maintenance rates for ETFs.
    """
    snap = _empty_bucket_capital_snapshot()
    if positions is None or positions.empty or "symbol" not in positions.columns:
        return snap

    _etf_syms, _, margin_by_symbol = _screened_margin_and_bucket_maps(screened)
    screened_etf_syms, screened_under_syms = _screened_etf_and_underlying_sets(screened)
    lot_qty_by_under = _lot_qty_map(lot_state)

    if screened is not None:
        if not screened.empty:
            etf_to_under, etf_to_delta = _etf_maps_from_screened(screened)
        else:
            etf_to_under, etf_to_delta = {}, {}
    else:
        screened_path = PROJECT_ROOT / "data" / "etf_screened_today.csv"
        if run_date:
            dated = RUNS_ROOT / run_date / "etf_screened_today.csv"
            if dated.exists():
                screened_path = dated
        etf_to_under, etf_to_delta = load_etf_delta_map(screened_path)
    for e_sym, u_sym in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(e_sym, u_sym)
        etf_to_delta.setdefault(e_sym, etf_to_delta.get(e_sym, -1.0))

    pos = positions.copy()
    pos["symbol"] = pos["symbol"].map(lambda s: canonical_symbol(str(s)))
    pos = pos[(pos["symbol"].astype(bool)) & (~pos["symbol"].isin(EXCLUDE_SYMBOLS))].copy()
    etf_to_under, etf_to_delta = complete_etf_maps_from_positions(
        pos, etf_to_under, etf_to_delta
    )

    if blocked_symbols or blocked_underlyings:
        pos = _filter_positions_blacklist(
            pos,
            blocked_symbols or set(),
            blocked_underlyings or set(),
            etf_to_under,
        )

    if run_date and (flow_short_set is None or flow_low_delta_syms is None):
        _fs, _fl = _load_flow_universe_sets(run_date)
        flow_short_set = flow_short_set if flow_short_set is not None else _fs
        flow_low_delta_syms = flow_low_delta_syms if flow_low_delta_syms is not None else _fl
    flow_short_set = flow_short_set or set()
    flow_low_delta_syms = flow_low_delta_syms or set()
    bucket3_only = flow_short_set | flow_low_delta_syms
    pos = pos[~pos["symbol"].isin(bucket3_only)].copy()

    screened_universe = _screened_universe_symbols(screened)
    if screened_universe:
        pos = pos[pos["symbol"].isin(screened_universe)].copy()

    if b4_etf_syms is None and run_date:
        b4_path = RUNS_ROOT / run_date / "accounting" / "bucket4_pairs.csv"
        if b4_path.exists():
            try:
                b4_etf_syms = set(pd.read_csv(b4_path)["etf"].astype(str).map(canonical_symbol))
            except Exception:
                b4_etf_syms = set()
        else:
            b4_etf_syms = set()
    b4_etf_syms = b4_etf_syms or set()

    if pos.empty:
        return snap

    def _add(bucket: str | None, signed_mv: float, margin_key: str | None) -> None:
        if bucket not in BUCKET_KEYS or bucket == "bucket_3" or abs(signed_mv) <= 1e-12:
            return
        long_pct, short_pct = margin_by_symbol.get(
            margin_key or "",
            (DEFAULT_MAINT_MARGIN_LONG, DEFAULT_MAINT_MARGIN_SHORT),
        )
        margin_pct = long_pct if signed_mv >= 0 else short_pct
        snap[f"net_capital_{bucket}"] += signed_mv
        snap[f"gross_capital_{bucket}"] += abs(signed_mv)
        snap[f"margin_req_{bucket}"] += abs(signed_mv) * margin_pct

    for _, row in pos.iterrows():
        sym = str(row.get("symbol", "") or "")
        position = _safe_float(row.get("position"), 0.0)
        mark = _safe_float(row.get("markPrice"), 0.0)
        fx = _safe_float(row.get("fxRateToBase"), 1.0)
        signed_mv = _safe_float(row.get("positionValue_base"), position * mark * fx)
        px_base = mark * fx
        if abs(signed_mv) <= 1e-12 and abs(position) > 1e-12 and px_base > 0:
            signed_mv = position * px_base

        is_etf = sym in screened_etf_syms
        if is_etf:
            bucket = _etf_capital_bucket(
                sym,
                etf_to_delta,
                flow_short_set=flow_short_set,
                flow_low_delta_syms=flow_low_delta_syms,
                b4_etf_syms=b4_etf_syms,
            )
            _add(bucket, signed_mv, sym)
            continue

        if sym not in screened_under_syms:
            continue

        lot_qty = lot_qty_by_under.get(sym)
        if lot_qty and abs(position) > 1e-12:
            r1, r2, r4 = _spot_capital_bucket_ratios(position, lot_qty)
            for bucket, ratio in (
                ("bucket_1", r1),
                ("bucket_2", r2),
                ("bucket_4", r4),
            ):
                if abs(ratio) > 1e-12:
                    _add(bucket, signed_mv * ratio, None)
            continue

        # Spot in screened underlyings but no ledger attribution: skip (not strategy capital).
        continue

    return snap


def build_bucket_capital_snapshot_from_run(run_date: str) -> dict[str, float]:
    outdir = RUNS_ROOT / run_date / "accounting"
    flex_positions_xml = RUNS_ROOT / run_date / "ibkr_flex" / "flex_positions.xml"
    pnl_symbol_csv = outdir / "pnl_by_symbol.csv"
    lot_state_csv = outdir / "bucket_lot_cost_state.csv"
    if not flex_positions_xml.exists() or not pnl_symbol_csv.exists():
        return _empty_bucket_capital_snapshot()

    try:
        positions = parse_open_positions(flex_positions_xml)
        pnl_symbol = pd.read_csv(pnl_symbol_csv)
        screened = _load_screened_for_run(run_date)
        lot_state = pd.read_csv(lot_state_csv) if lot_state_csv.exists() else pd.DataFrame()
        blocked_symbols, blocked_underlyings = _blocked_exposure_sets(run_date)
    except Exception:
        return _empty_bucket_capital_snapshot()
    return compute_bucket_capital_snapshot(
        positions,
        pnl_symbol,
        screened,
        lot_state,
        blocked_symbols=blocked_symbols,
        blocked_underlyings=blocked_underlyings,
        run_date=run_date,
    )


def _safe_return(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{100.0 * float(value):,.2f}%"


def compute_average_bucket_capital(history: pd.DataFrame) -> dict[str, float]:
    """
    Time-average each per-bucket capital column across persisted history rows.
    """
    out = {c: 0.0 for c in PNL_HISTORY_CAPITAL_COLS}
    if history is None or history.empty:
        return out
    hist = _ensure_pnl_history_derived_cols(history)
    for c in PNL_HISTORY_CAPITAL_COLS:
        if c not in hist.columns:
            continue
        vals = pd.to_numeric(hist[c], errors="coerce").dropna()
        if not vals.empty:
            out[c] = float(vals.mean())
    return out


def format_bucket_return_table(
    bucket_pnl: dict[str, float],
    capital_snapshot: dict[str, float],
) -> str:
    headers = [
        "BUCKET",
        "PNL_YTD",
        "AVG_NET_CAP",
        "AVG_GROSS_CAP",
        "AVG_MAINT_MARGIN",
        "ROC",
        "ROG",
        "ROM",
    ]
    rows: list[list[str]] = []
    for bucket in BUCKET_KEYS:
        pnl = float(bucket_pnl.get(bucket, 0.0) or 0.0)
        net_cap = float(capital_snapshot.get(f"net_capital_{bucket}", 0.0) or 0.0)
        gross_cap = float(capital_snapshot.get(f"gross_capital_{bucket}", 0.0) or 0.0)
        margin_req = float(capital_snapshot.get(f"margin_req_{bucket}", 0.0) or 0.0)
        roc = (
            "n/a"
            if bucket in BUCKETS_WITHOUT_ROC or net_cap <= 0
            else _fmt_pct(_safe_return(pnl, net_cap))
        )
        rows.append(
            [
                BUCKET_LABELS[bucket],
                f"{pnl:,.2f}",
                f"{net_cap:,.2f}",
                f"{gross_cap:,.2f}",
                f"{margin_req:,.2f}",
                roc,
                _fmt_pct(_safe_return(pnl, gross_cap)),
                _fmt_pct(_safe_return(pnl, margin_req)),
            ]
        )

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]
    lines = [
        "  ".join(headers[i].ljust(widths[i]) if i == 0 else headers[i].rjust(widths[i]) for i in range(len(headers))),
        "-" * (sum(widths) + 2 * (len(headers) - 1)),
    ]
    for r in rows:
        lines.append(
            "  ".join(r[i].ljust(widths[i]) if i == 0 else r[i].rjust(widths[i]) for i in range(len(r)))
        )
    return "\n".join(lines)


def format_top_underlying_net_exposure(
    exposure_df: pd.DataFrame,
    *,
    net_col: str = "net_notional_usd",
    gross_col: str = "gross_notional_usd",
    label_col: str = "underlying",
    min_abs_net_usd: float = TOP_NET_EXPOSURE_MIN_ABS_USD,
    max_rows: int = TOP_NET_EXPOSURE_MAX_ROWS,
) -> str:
    """
    Compact headline block: per-underlying net/gross from the B1+B2+B4 book
    rollup (``net_exposure_by_underlying.csv``), not B4 pair-view rows alone.
    """
    if exposure_df is None or exposure_df.empty:
        return "Net exposure (stock sleeves): (no underlying exposure table)."
    if label_col not in exposure_df.columns or net_col not in exposure_df.columns:
        return "Net exposure (stock sleeves): (missing columns for net/gross)."

    df = exposure_df.copy()
    df[net_col] = pd.to_numeric(df[net_col], errors="coerce").fillna(0.0)
    has_gross = gross_col in df.columns
    if has_gross:
        df[gross_col] = pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0)

    book_net = float(df[net_col].sum())
    book_gross = float(df[gross_col].sum()) if has_gross else 0.0

    rows_df = df[df[net_col].abs() >= min_abs_net_usd].copy()
    rows_df = rows_df.sort_values(net_col, key=lambda s: s.abs(), ascending=False)
    n_above = int(rows_df.shape[0])
    if max_rows > 0:
        rows_df = rows_df.head(max_rows)

    lines = [
        "Net exposure by underlying (stock sleeves — B1+B2+B4 combined, delta-normalized):",
        f"  Book net: ${book_net:+,.0f}  |  Book gross: ${book_gross:,.0f}",
        "",
    ]

    if rows_df.empty:
        lines.append(f"  (no underlying with |net| >= ${min_abs_net_usd:,.0f})")
        return "\n".join(lines)

    label_w = max(8, max(len(str(u)) for u in rows_df[label_col]))
    for _, r in rows_df.iterrows():
        lab = str(r[label_col])
        net = float(r[net_col])
        if has_gross:
            gross = float(r[gross_col])
            lines.append(
                f"  • {lab:<{label_w}}  net ${net:+,.0f}  gross ${gross:>11,.0f}"
            )
        else:
            lines.append(f"  • {lab:<{label_w}}  net ${net:+,.0f}")

    if n_above > len(rows_df):
        lines.append(f"  ... and {n_above - len(rows_df)} more (|net| >= ${min_abs_net_usd:,.0f})")
    return "\n".join(lines)


def format_b124_underlying_exposure_table(
    exposure_csv: Path,
    *,
    blocked_keys: set[str] | None = None,
) -> str:
    """Full B1+B2+B4 net/gross exposure by underlying, sorted by net (high to low)."""
    if not exposure_csv.exists():
        return "(no net_exposure_by_underlying.csv)"

    df = pd.read_csv(exposure_csv)
    if df.empty or "underlying" not in df.columns or "net_notional_usd" not in df.columns:
        return "(missing underlying/net columns in net_exposure_by_underlying.csv)"

    df = df.copy()
    df["underlying"] = df["underlying"].astype(str)
    if blocked_keys:
        df = df[~df["underlying"].isin(blocked_keys)].copy()
    df["net_notional_usd"] = pd.to_numeric(df["net_notional_usd"], errors="coerce").fillna(0.0)
    has_gross = "gross_notional_usd" in df.columns
    if has_gross:
        df["gross_notional_usd"] = pd.to_numeric(df["gross_notional_usd"], errors="coerce").fillna(0.0)

    df = df.sort_values("net_notional_usd", ascending=False)
    book_net = float(df["net_notional_usd"].sum())
    book_gross = float(df["gross_notional_usd"].sum()) if has_gross else 0.0

    label_w = max(10, max(len(str(u)) for u in df["underlying"]))
    net_w = max(12, max(len(f"{v:,.0f}") for v in df["net_notional_usd"]))
    gross_w = max(12, max(len(f"{v:,.0f}") for v in df["gross_notional_usd"])) if has_gross else 12

    lines = [
        f"Book net: ${book_net:+,.0f}  |  Book gross: ${book_gross:,.0f}",
        "",
        f"{'UNDERLYING'.ljust(label_w)}  {'NET_USD'.rjust(net_w)}"
        + (f"  {'GROSS_USD'.rjust(gross_w)}" if has_gross else ""),
        "-" * (label_w + net_w + (gross_w + 2 if has_gross else 0) + 2),
    ]
    for _, r in df.iterrows():
        lab = str(r["underlying"])
        net = f"{float(r['net_notional_usd']):,.0f}"
        if has_gross:
            gross = f"{float(r['gross_notional_usd']):,.0f}"
            lines.append(f"{lab.ljust(label_w)}  {net.rjust(net_w)}  {gross.rjust(gross_w)}")
        else:
            lines.append(f"{lab.ljust(label_w)}  {net.rjust(net_w)}")
    return "\n".join(lines)


def read_bucket_pnl_from_run(run_date_str: str) -> tuple[float, float, float, float] | None:
    """
    Bucket YTD-style totals from a prior accounting run.

    Uses ``pnl_by_bucket.csv`` (same source as the EOD email headline). Falls back
    to ``totals.json`` ``bucket_pnl`` when the CSV is missing.
    """
    outdir = RUNS_ROOT / run_date_str / "accounting"
    if not outdir.is_dir():
        return None

    bucket_csv = outdir / "pnl_by_bucket.csv"
    if bucket_csv.exists():
        try:
            df = pd.read_csv(bucket_csv)
            if not df.empty and "bucket" in df.columns and "total_pnl" in df.columns:
                df["total_pnl"] = pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0)

                def _sum(bucket: str) -> float:
                    rows = df[df["bucket"] == bucket]
                    return float(rows["total_pnl"].sum()) if not rows.empty else 0.0

                return (
                    _sum("bucket_1"),
                    _sum("bucket_2"),
                    _sum("bucket_3"),
                    _sum("bucket_4"),
                )
        except Exception:
            pass

    totals_path = outdir / "totals.json"
    if not totals_path.exists():
        return None

    obj = json.loads(totals_path.read_text(encoding="utf-8"))
    bp = obj.get("bucket_pnl")
    if not isinstance(bp, dict):
        raise RuntimeError(
            f"Missing or invalid bucket_pnl in {totals_path}; cannot build bucket history."
        )
    required = ("bucket_1", "bucket_2", "bucket_3")
    missing = [k for k in required if k not in bp]
    if missing:
        raise RuntimeError(
            f"Missing bucket keys {missing} in {totals_path}; cannot build bucket history."
        )

    split_method = str(obj.get("bucket_split_method", "")).strip().lower()
    if split_method == "pnl_weighted":
        raise RuntimeError(
            f"Run {run_date_str} still reports bucket_split_method='pnl_weighted'. "
            "Re-run accounting with held_exposure before generating email history."
        )

    return (
        float(bp.get("bucket_1", 0.0)),
        float(bp.get("bucket_2", 0.0)),
        float(bp.get("bucket_3", 0.0)),
        float(bp.get("bucket_4", 0.0)),
    )


def split_long_short_realized_unrealized(
    pnl_symbol: pd.DataFrame,
    pos: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Split per-symbol FIFO realized / unrealized into long-book vs short-book using
    end-of-day net position sign (position < 0 => short). Symbols with PnL rows
    but no open position are treated as long-book (flat / closed lots).

    Realized on names that flipped intraday is only approximated by this rule.
    """
    long_r = long_u = short_r = short_u = 0.0
    if pnl_symbol.empty:
        return long_r, long_u, short_r, short_u

    df = pnl_symbol.copy()
    if "symbol" not in df.columns:
        return long_r, long_u, short_r, short_u

    side_short: dict[str, bool] = {}
    if not pos.empty and "symbol" in pos.columns:
        p = pos.copy()
        p["symbol"] = p["symbol"].astype(str).map(canonical_symbol)
        if "position" in p.columns:
            p["_pv"] = pd.to_numeric(p["position"], errors="coerce").fillna(0.0)
            agg = p.groupby("symbol", as_index=False)["_pv"].sum()
            for _, row in agg.iterrows():
                side_short[str(row["symbol"])] = float(row["_pv"]) < 0
        elif "is_short" in p.columns:
            agg = p.groupby("symbol", as_index=False)["is_short"].any()
            for _, row in agg.iterrows():
                side_short[str(row["symbol"])] = bool(row["is_short"])

    for _, r in df.iterrows():
        sym = canonical_symbol(str(r.get("symbol", "") or ""))
        if not sym:
            continue
        rv = float(pd.to_numeric(r.get("realized_pnl"), errors="coerce") or 0.0)
        uv = float(pd.to_numeric(r.get("unrealized_pnl"), errors="coerce") or 0.0)
        is_short = side_short.get(sym, False)
        if is_short:
            short_r += rv
            short_u += uv
        else:
            long_r += rv
            long_u += uv

    return long_r, long_u, short_r, short_u


def build_attribution_row(
    run_date: str,
    totals: dict,
    pnl_symbol: pd.DataFrame,
    pos: pd.DataFrame,
) -> dict[str, float | str]:
    lr, lu, sr, su = split_long_short_realized_unrealized(pnl_symbol, pos)
    return {
        "date": run_date,
        "long_realized_pnl": lr,
        "long_unrealized_pnl": lu,
        "short_realized_pnl": sr,
        "short_unrealized_pnl": su,
        "gross_realized_pnl": float(totals.get("total_realized_pnl", 0.0) or 0.0),
        "gross_unrealized_pnl": float(totals.get("total_unrealized_pnl", 0.0) or 0.0),
        "other_fees": float(totals.get("total_other_fees", 0.0) or 0.0),
        "borrow_fees": float(totals.get("total_borrow_fees", 0.0) or 0.0),
        "short_credit_interest": float(totals.get("total_short_credit_interest", 0.0) or 0.0),
        "excluded_cash_interest_base": float(totals.get("excluded_cash_interest_base", 0.0) or 0.0),
        "dividends": float(totals.get("total_dividends", 0.0) or 0.0),
        "withholding_tax": float(totals.get("total_withholding_tax", 0.0) or 0.0),
        "pil_dividends": float(totals.get("total_pil_dividends", 0.0) or 0.0),
        "bond_interest": float(totals.get("total_bond_interest", 0.0) or 0.0),
        "strategy_total_pnl": float(totals.get("total_pnl", 0.0) or 0.0),
    }


def compute_attribution_snapshot_from_run(run_date_str: str) -> dict[str, float | str] | None:
    """Load accounting outputs for a run date and build one attribution row."""
    outdir = RUNS_ROOT / run_date_str / "accounting"
    totals_path = outdir / "totals.json"
    pnl_sym_path = outdir / "pnl_by_symbol.csv"
    flex_pos = RUNS_ROOT / run_date_str / "ibkr_flex" / "flex_positions.xml"
    if not totals_path.exists() or not pnl_sym_path.exists():
        return None
    totals = json.loads(totals_path.read_text(encoding="utf-8"))
    sym_df = pd.read_csv(pnl_sym_path)
    pos = parse_open_positions(flex_pos) if flex_pos.exists() else pd.DataFrame()
    return build_attribution_row(run_date_str, totals, sym_df, pos)


def enrich_attribution_history_from_runs(hist: pd.DataFrame) -> pd.DataFrame:
    """Fill missing dates from data/runs/<date>/accounting (same pattern as bucket history)."""
    if not RUNS_ROOT.is_dir():
        return hist

    start_dt = pd.to_datetime(START_DATE)
    updates: dict[str, dict[str, float | str]] = {}
    for child in RUNS_ROOT.iterdir():
        if not child.is_dir():
            continue
        ds = child.name
        try:
            dt = pd.to_datetime(ds)
        except (ValueError, TypeError):
            continue
        if dt.normalize() < start_dt.normalize():
            continue
        row = compute_attribution_snapshot_from_run(ds)
        if row is None:
            continue
        updates[ds] = row

    if not updates:
        return hist

    hist = hist.copy()
    for c in ATTRIBUTION_HISTORY_COLS:
        if c not in hist.columns:
            hist[c] = np.nan
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"])
    date_key = hist["date"].dt.strftime("%Y-%m-%d")
    new_rows: list[dict[str, float | str]] = []
    numeric_keys = [c for c in ATTRIBUTION_HISTORY_COLS if c != "date"]
    for ds, row in updates.items():
        m = date_key == ds
        if m.any():
            idx = int(hist.index[m][0])
            for k in numeric_keys:
                if k in row:
                    hist.loc[idx, k] = row[k]
        else:
            new_rows.append(row)

    if new_rows:
        hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    hist = hist.reset_index(drop=True)
    return hist


def update_attribution_history(
    run_date: str,
    totals: dict,
    pnl_symbol_csv: Path,
    flex_positions_xml: Path,
) -> pd.DataFrame:
    """
    Upsert YTD attribution snapshot for run_date into pnl_attribution_history.csv.
    Long/short split uses EOD position side; enrich merges other run folders.
    """
    ensure_ledger_dir()
    sym_df = pd.read_csv(pnl_symbol_csv) if pnl_symbol_csv.exists() else pd.DataFrame()
    pos = parse_open_positions(flex_positions_xml) if flex_positions_xml.exists() else pd.DataFrame()
    row = build_attribution_row(run_date, totals, sym_df, pos)

    if ATTRIBUTION_HISTORY_CSV.exists():
        hist = pd.read_csv(ATTRIBUTION_HISTORY_CSV)
        if "date" not in hist.columns:
            hist = pd.DataFrame([row])
        else:
            hist["date"] = hist["date"].astype(str)
            hist = hist[hist["date"] != run_date]
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    for c in ATTRIBUTION_HISTORY_COLS:
        if c == "date":
            continue
        if c not in hist.columns:
            hist[c] = np.nan
        hist[c] = pd.to_numeric(hist[c], errors="coerce")

    hist = enrich_attribution_history_from_runs(hist)
    hist = hist[hist["date"] >= pd.to_datetime(START_DATE)].copy()

    hist_out = hist.copy()
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    for c in ATTRIBUTION_HISTORY_COLS:
        if c not in hist_out.columns:
            hist_out[c] = np.nan
    hist_out = hist_out[list(ATTRIBUTION_HISTORY_COLS)]
    hist_out.to_csv(ATTRIBUTION_HISTORY_CSV, index=False)

    return hist.reset_index(drop=True)


def make_attribution_plot(history: pd.DataFrame) -> Path:
    """
    Multi-panel YTD time series: long/short trading, gross trading, costs & interest, dividends.
    """
    ensure_ledger_dir()
    if history.empty:
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.set_title(f"PnL attribution (no data yet) since {START_DATE}")
        fig.tight_layout()
        fig.savefig(PLOT_ATTRIBUTION_PNG, dpi=150)
        plt.close(fig)
        return PLOT_ATTRIBUTION_PNG

    h = history.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"]).sort_values("date")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_tl, ax_tr, ax_bl, ax_br = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def _line(ax, cols: list[tuple[str, str]], title: str) -> None:
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        for col, lab in cols:
            if col not in h.columns:
                continue
            y = pd.to_numeric(h[col], errors="coerce")
            if y.notna().any():
                ax.plot(h["date"], y, marker="o", ms=3, lw=1.3, label=lab)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.legend(loc="best", fontsize=6)
        ax.set_ylabel("USD (YTD)")

    _line(
        ax_tl,
        [
            ("long_realized_pnl", "Long realized"),
            ("long_unrealized_pnl", "Long unrealized"),
            ("short_realized_pnl", "Short realized"),
            ("short_unrealized_pnl", "Short unrealized"),
        ],
        "Long vs short trading (FIFO by EOD side)",
    )
    _line(
        ax_tr,
        [
            ("gross_realized_pnl", "Gross realized"),
            ("gross_unrealized_pnl", "Gross unrealized"),
        ],
        "Gross trading PnL (totals.json)",
    )
    _line(
        ax_bl,
        [
            ("borrow_fees", "Borrow fees"),
            ("short_credit_interest", "Short credit interest"),
            ("other_fees", "Other fees (t-costs)"),
            ("excluded_cash_interest_base", "Excluded cash / margin interest"),
        ],
        "Financing & fees (excluded cash not in strategy total)",
    )
    _line(
        ax_br,
        [
            ("dividends", "Dividends"),
            ("withholding_tax", "Withholding tax"),
            ("pil_dividends", "PIL dividends"),
            ("bond_interest", "Bond interest"),
        ],
        "Dividends & income items",
    )

    fig.suptitle(f"YTD PnL attribution since {START_DATE} (snapshot per run date)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_ATTRIBUTION_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return PLOT_ATTRIBUTION_PNG


def _omit_from_pnl_history_calendar_date(dt: pd.Timestamp) -> bool:
    """Saturday run folders are excluded from pnl_history rows."""
    return int(dt.weekday()) == 5


def enrich_history_bucket_cols_from_runs(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Fill pnl_bucket_*, total_pnl, and bucket capital bases from
    data/runs/<date>/accounting for every run date on or after START_DATE that
    has bucket outputs.
    """
    if not RUNS_ROOT.is_dir():
        return hist

    start_dt = pd.to_datetime(START_DATE)
    updates: dict[str, tuple[float, float, float, float]] = {}
    capital_updates: dict[str, dict[str, float]] = {}
    for child in RUNS_ROOT.iterdir():
        if not child.is_dir():
            continue
        ds = child.name
        try:
            dt = pd.to_datetime(ds)
        except (ValueError, TypeError):
            continue
        if dt.normalize() < start_dt.normalize():
            continue
        if _omit_from_pnl_history_calendar_date(dt.normalize()):
            continue
        triple = read_bucket_pnl_from_run(ds)
        if triple is None:
            continue
        updates[ds] = triple
        capital_updates[ds] = build_bucket_capital_snapshot_from_run(ds)

    if not updates:
        return hist

    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"])
    for c in PNL_HISTORY_BUCKET_COLS:
        if c not in hist.columns:
            hist[c] = np.nan
    for c in PNL_HISTORY_DISPLAY_PNL_COLS:
        if c not in hist.columns:
            hist[c] = np.nan
    if "total_pnl" not in hist.columns:
        hist["total_pnl"] = np.nan
    for c in PNL_HISTORY_CAPITAL_COLS:
        if c not in hist.columns:
            hist[c] = np.nan
    for c in PNL_HISTORY_DISPLAY_CAPITAL_COLS:
        if c not in hist.columns:
            hist[c] = np.nan

    date_key = hist["date"].dt.strftime("%Y-%m-%d")
    new_rows: list[dict] = []
    for ds, (b1, b2, b3, b4) in updates.items():
        stock_pnl = b1 + b2 + b4
        tot = stock_pnl + b3
        cap_full = capital_updates.get(ds, {})
        cap = _consolidate_capital_snapshot(cap_full)
        m = date_key == ds
        if m.any():
            hist.loc[m, "pnl_bucket_1"] = b1
            hist.loc[m, "pnl_bucket_2"] = b2
            hist.loc[m, "pnl_bucket_3"] = b3
            hist.loc[m, "pnl_bucket_4"] = b4
            hist.loc[m, "pnl_stock_sleeves"] = stock_pnl
            hist.loc[m, "total_pnl"] = tot
            for c in PNL_HISTORY_CAPITAL_COLS:
                hist.loc[m, c] = float(cap_full.get(c, np.nan))
            for c in PNL_HISTORY_DISPLAY_CAPITAL_COLS:
                hist.loc[m, c] = float(cap.get(c, np.nan))
        else:
            row = {
                "date": pd.to_datetime(ds),
                "pnl_bucket_1": b1,
                "pnl_bucket_2": b2,
                "pnl_bucket_3": b3,
                "pnl_bucket_4": b4,
                "pnl_stock_sleeves": stock_pnl,
                "total_pnl": tot,
            }
            for c in PNL_HISTORY_CAPITAL_COLS:
                row[c] = float(cap_full.get(c, np.nan))
            for c in PNL_HISTORY_DISPLAY_CAPITAL_COLS:
                row[c] = float(cap.get(c, np.nan))
            new_rows.append(row)

    if new_rows:
        hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)

    hist = hist.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    hist = hist.reset_index(drop=True)
    return _canonical_pnl_history(hist)


def update_pnl_history(
    run_date: str,
    *,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    bucket_capital: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Appends (or overwrites) a row in pnl_history.csv for the given run_date.
    Persists per-bucket and consolidated sleeve PnL/capital.
    Returns the full history DF filtered from START_DATE onward.
    """
    ensure_ledger_dir()

    stock_pnl = float(b1) + float(b2) + float(b4)
    total_pnl = stock_pnl + float(b3)
    cap_full = bucket_capital or _empty_bucket_capital_snapshot()
    display_capital = _consolidate_capital_snapshot(cap_full)
    row_obj = {
        "date": run_date,
        "pnl_bucket_1": float(b1),
        "pnl_bucket_2": float(b2),
        "pnl_bucket_3": float(b3),
        "pnl_bucket_4": float(b4),
        "pnl_stock_sleeves": stock_pnl,
        "total_pnl": total_pnl,
    }
    for c in PNL_HISTORY_CAPITAL_COLS:
        row_obj[c] = float(cap_full.get(c, 0.0) or 0.0)
    for c in PNL_HISTORY_DISPLAY_CAPITAL_COLS:
        row_obj[c] = float(display_capital.get(c, 0.0) or 0.0)
    row = pd.DataFrame([row_obj])

    run_dt = datetime.strptime(run_date, "%Y-%m-%d")
    skip_row = run_dt.weekday() == 5  # Saturday: never persist to pnl_history

    if PNL_HISTORY_CSV.exists():
        hist = pd.read_csv(PNL_HISTORY_CSV)
        if "date" not in hist.columns:
            hist = row if not skip_row else pd.DataFrame(columns=row.columns)
        else:
            hist["date"] = hist["date"].astype(str)
            hist = _ensure_pnl_history_derived_cols(hist)
            for c in row.columns:
                if c not in hist.columns:
                    hist[c] = np.nan
            if "total_pnl" not in hist.columns:
                hist["total_pnl"] = np.nan
            hist = hist[hist["date"] != run_date]
            if not skip_row:
                hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row if not skip_row else pd.DataFrame(columns=row.columns)

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    pnl_cols = tuple(dict.fromkeys((*PNL_HISTORY_BUCKET_COLS, *PNL_HISTORY_DISPLAY_PNL_COLS)))
    capital_cols = tuple(dict.fromkeys((*PNL_HISTORY_CAPITAL_COLS, *PNL_HISTORY_DISPLAY_CAPITAL_COLS)))
    for c in pnl_cols:
        hist[c] = pd.to_numeric(hist.get(c, np.nan), errors="coerce")
    for c in capital_cols:
        hist[c] = pd.to_numeric(hist.get(c, np.nan), errors="coerce")
    hist["total_pnl"] = pd.to_numeric(hist["total_pnl"], errors="coerce")

    hist = enrich_history_bucket_cols_from_runs(hist)
    hist = hist[hist["date"].dt.weekday != 5].copy()

    hist_out = _canonical_pnl_history(hist)
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    hist_out.to_csv(PNL_HISTORY_CSV, index=False)

    start_dt = pd.to_datetime(START_DATE)
    hist = hist[hist["date"] >= start_dt].copy()
    return hist


def rebuild_pnl_history_from_runs() -> pd.DataFrame:
    """
    Rebuild pnl_history.csv from every run's accounting totals.json and a fresh
    per-day capital snapshot. Use after restating historical accounting outputs.
    """
    ensure_ledger_dir()
    hist = pd.DataFrame(columns=["date"] + list(PNL_HISTORY_RETURN_BASE_COLS) + ["total_pnl"])
    hist = enrich_history_bucket_cols_from_runs(hist)
    hist = hist[hist["date"].dt.weekday != 5].copy()
    hist_out = _canonical_pnl_history(hist)
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    hist_out.to_csv(PNL_HISTORY_CSV, index=False)
    start_dt = pd.to_datetime(START_DATE)
    return hist[hist["date"] >= start_dt].copy()


def compute_period_pnl_deltas(
    history: pd.DataFrame,
    run_date: str,
    *,
    period: str = "daily",
) -> dict[str, float] | None:
    """
    Return per-bucket/total PnL changes vs a prior cumulative snapshot in history.

    Keys: bucket_1..bucket_4, total. ``period`` is daily (prior run date),
    wtd (prior calendar week), or mtd (prior calendar month).
    """
    if history is None or history.empty:
        return None

    hist = _ensure_pnl_history_derived_cols(history.copy())
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if hist.empty:
        return None

    cols = list(PNL_HISTORY_BUCKET_COLS) + ["total_pnl"]
    for c in cols:
        hist[c] = pd.to_numeric(hist.get(c, np.nan), errors="coerce")

    target = pd.to_datetime(run_date).normalize()
    current_rows = hist[hist["date"].dt.normalize() <= target]
    if current_rows.empty:
        return None
    current = current_rows.iloc[-1]

    def _prior_before(cutoff: pd.Timestamp) -> pd.Series | None:
        rows = hist[hist["date"].dt.normalize() < cutoff.normalize()]
        if rows.empty:
            return None
        return rows.iloc[-1]

    def _delta(start: pd.Series | None, col: str) -> float:
        cur = float(current.get(col, 0.0) or 0.0)
        if start is None:
            return 0.0
        if pd.isna(start.get(col, np.nan)):
            return cur
        return cur - float(start.get(col, 0.0) or 0.0)

    period = period.strip().lower()
    if period == "daily":
        base = _prior_before(current["date"])
    elif period == "wtd":
        week_start = target - pd.Timedelta(days=int(target.dayofweek))
        base = _prior_before(week_start)
    elif period == "mtd":
        month_start = target.replace(day=1)
        base = _prior_before(month_start)
    else:
        raise ValueError(f"unknown period {period!r}; use daily, wtd, or mtd")

    out: dict[str, float] = {}
    for bucket in BUCKET_KEYS:
        out[bucket] = _delta(base, f"pnl_{bucket}")
    out["total"] = _delta(base, "total_pnl")
    return out


def _compact_usd(value: float) -> str:
    """Compact dollar formatting for long email subjects."""
    v = float(value)
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1_000_000:
        return f"{sign}{av / 1e6:.1f}M"
    if av >= 10_000:
        return f"{sign}{av / 1e3:.1f}k"
    return f"{v:,.0f}"


def _format_subject_bucket_pnl_line(
    bucket_pnl: dict[str, float],
    *,
    total_pnl: float,
    compact: bool = False,
) -> str:
    parts: list[str] = []
    for bucket in BUCKET_KEYS:
        label = SUBJECT_BUCKET_SHORT[bucket]
        val = float(bucket_pnl.get(bucket, 0.0) or 0.0)
        if compact:
            parts.append(f"{label}: {_compact_usd(val)}")
        else:
            parts.append(f"{label}: {val:,.0f}")
    total_fmt = _compact_usd(total_pnl) if compact else f"{total_pnl:,.0f}"
    parts.append(f"Total: {total_fmt}")
    return " | ".join(parts)


def format_eod_subject(
    run_date: str,
    bucket_pnl: dict[str, float],
    *,
    total_pnl: float,
) -> str:
    """Email subject with four-bucket YTD PnL from accounting totals.json."""
    prefix = f"EOD PnL — {run_date} — "
    line = _format_subject_bucket_pnl_line(bucket_pnl, total_pnl=total_pnl, compact=False)
    if len(prefix + line) > SUBJECT_MAX_LEN:
        line = _format_subject_bucket_pnl_line(bucket_pnl, total_pnl=total_pnl, compact=True)
    return prefix + line


def _format_subject_pnl_line(
    *,
    b124_pnl: float,
    b3_pnl: float,
    total_pnl: float,
    daily_prefix: bool = False,
) -> str:
    """Legacy two-group subject formatter (kept for format_eod_pnl_subject)."""
    prefix = "Daily " if daily_prefix else ""
    return (
        f"{prefix}{SUBJECT_B124_LABEL}: {b124_pnl:,.2f} | "
        f"{SUBJECT_B3_LABEL} : {b3_pnl:,.2f} | "
        f"Total: {total_pnl:,.2f}"
    )


def format_eod_pnl_subject(
    run_date: str,
    *,
    daily: dict[str, float],
    ytd_total: float,
) -> str:
    """Email subject: daily sleeve changes (what moved today), YTD total in parentheses."""
    line = _format_subject_pnl_line(
        b124_pnl=daily["stock_sleeves"],
        b3_pnl=daily["bucket_3"],
        total_pnl=daily["total"],
        daily_prefix=True,
    )
    return f"EOD PnL — {run_date} — {line} (YTD Total: {ytd_total:,.2f})"


def format_period_pnl_summary(history: pd.DataFrame, run_date: str) -> str:
    """Format daily, week-to-date, and month-to-date changes from cumulative PnL history."""
    if history.empty:
        return "PERIOD PnL: unavailable (no history rows)."

    labels = {
        "bucket_1": "B1",
        "bucket_2": "B2",
        "bucket_3": "B3",
        "bucket_4": "B4",
        "total": "Total",
    }
    lines = ["PERIOD PnL changes from cumulative history:"]
    for name, period in (
        ("Daily", "daily"),
        ("Week-to-date", "wtd"),
        ("Month-to-date", "mtd"),
    ):
        deltas = compute_period_pnl_deltas(history, run_date, period=period)
        if deltas is None:
            lines.append(f"  {name}: unavailable")
            continue
        pieces = [f"{labels[k]}: {deltas[k]:,.2f}" for k in labels]
        lines.append(f"  {name}: " + " | ".join(pieces))
    return "\n".join(lines)


# Stable label placement per series: offset from marker in points (dx, dy), ha, va.
_PNL_LABEL_STYLE: dict[str, tuple[float, float, str, str]] = {
    "#1f77b4": (11, 7, "left", "bottom"),   # Bucket 1
    "#ff7f0e": (-11, 7, "right", "bottom"),  # Bucket 2
    "#2ca02c": (11, -9, "left", "top"),      # Bucket 3
    "#d62728": (-11, -9, "right", "top"),    # Bucket 4
}
_PNL_LABEL_LEGACY_STYLE = (0, -9, "center", "top")  # below marker, away from bucket labels above


def _annotate_pnl_point_labels_stable(
    ax,
    points: list[tuple[pd.Timestamp, float, str]],
    *,
    fontsize: float,
    is_legacy: bool,
) -> None:
    """
    Annotate each (date, y, color) with a small label at a fixed offset for that color.
    """
    for x, y, color in points:
        ckey = str(color).lower()
        if is_legacy:
            dx, dy, ha, va = _PNL_LABEL_LEGACY_STYLE
        elif ckey in _PNL_LABEL_STYLE:
            dx, dy, ha, va = _PNL_LABEL_STYLE[ckey]
        else:
            dx, dy, ha, va = (0, 8, "center", "bottom")

        ax.annotate(
            f"${y:,.0f}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            clip_on=False,
        )


def make_pnl_plot(history: pd.DataFrame) -> Path:
    """
    Creates a PNG plot showing YTD PnL since START_DATE for buckets 1–4.
    """
    ensure_ledger_dir()

    bucket_specs: tuple[tuple[str, str, str], ...] = (
        ("pnl_bucket_1", SUBJECT_BUCKET_SHORT["bucket_1"], "#1f77b4"),
        ("pnl_bucket_2", SUBJECT_BUCKET_SHORT["bucket_2"], "#ff7f0e"),
        ("pnl_bucket_3", SUBJECT_BUCKET_SHORT["bucket_3"], "#2ca02c"),
        ("pnl_bucket_4", SUBJECT_BUCKET_SHORT["bucket_4"], "#d62728"),
    )

    hist = _ensure_pnl_history_derived_cols(history) if not history.empty else history

    if hist.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_title(f"YTD PnL by bucket since {START_DATE} (no data yet)", fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("YTD PnL (base)")
        fig.tight_layout()
        fig.savefig(PLOT_PNG, dpi=150)
        plt.close(fig)
        return PLOT_PNG

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    bucket_label_points: list[tuple[pd.Timestamp, float, str]] = []
    label_fs = 5.0

    for col, label, color in bucket_specs:
        if col not in hist.columns:
            continue
        y = pd.to_numeric(hist[col], errors="coerce")
        mask = y.notna()
        if not mask.any():
            continue
        dates = hist.loc[mask, "date"]
        yv = y[mask]
        ax.plot(dates, yv, marker="o", linewidth=1.6, markersize=4, label=label, color=color)
        bucket_label_points.append((pd.Timestamp(dates.iloc[-1]), float(yv.iloc[-1]), color))

    if bucket_label_points:
        _annotate_pnl_point_labels_stable(
            ax, bucket_label_points, fontsize=label_fs, is_legacy=False
        )

    if "total_pnl" in hist.columns:
        tot = pd.to_numeric(hist["total_pnl"], errors="coerce")
        if all(c in hist.columns for c in PNL_HISTORY_BUCKET_COLS):
            legacy_only = hist[list(PNL_HISTORY_BUCKET_COLS)].isna().all(axis=1)
        else:
            legacy_only = pd.Series(True, index=hist.index)
        mask = tot.notna() & legacy_only
        if mask.any():
            dates = hist.loc[mask, "date"]
            yv = tot[mask]
            ax.plot(
                dates,
                yv,
                marker="o",
                linewidth=1.6,
                markersize=4,
                color="0.35",
                linestyle="--",
                label="Total (legacy, before bucket history)",
            )
            legacy_pts = [
                (pd.Timestamp(dates.iloc[-1]), float(yv.iloc[-1]), "0.35")
            ]
            _annotate_pnl_point_labels_stable(
                ax, legacy_pts, fontsize=label_fs, is_legacy=True
            )

    ax.set_title(f"YTD PnL by bucket since {START_DATE}", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("YTD PnL (base)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="best", fontsize=7)
    ax.margins(x=0.03, y=0.18)

    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    plt.close(fig)
    return PLOT_PNG


def parse_recipients(raw: str) -> list[str]:
    """
    Robust parsing for PNL_RECIPIENTS.

    Supports:
      - "a@x.com,b@y.com"
      - "a@x.com, b@y.com"
      - "Name <a@x.com>, b@y.com"
      - newline/semicolon separated lists
    Returns a list of email addresses suitable for SMTP envelope recipients.
    """
    if not raw:
        return []
    normalized = raw.replace(";", ",").replace("\n", ",")
    pairs = getaddresses([normalized])  # handles "Name <email>"
    emails = [addr.strip() for _, addr in pairs if addr and addr.strip()]
    # Light sanity filter (avoid passing "a@b.com, c@d.com" as one token)
    emails = [e for e in emails if "@" in e and " " not in e]
    return emails


def resolve_proposed_trades_path(run_date: str) -> Path:
    dated = PROJECT_ROOT / "data" / "runs" / run_date / "proposed_trades.csv"
    if dated.exists():
        return dated

    cfg = load_config(PROJECT_ROOT / "config" / "strategy_config.yml")
    proposed_cfg = (cfg.get("paths", {}) or {}).get("proposed_trades_csv", "")
    if proposed_cfg:
        p = Path(proposed_cfg)
        if p.exists():
            return p
    fallback = PROJECT_ROOT / "data" / "proposed_trades.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not locate proposed_trades.csv (dated or latest).")


def load_position_discrepancies(run_date: str) -> pd.DataFrame:
    screened_csv = PROJECT_ROOT / "data" / "etf_screened_today.csv"
    if not screened_csv.exists():
        raise FileNotFoundError(f"Missing etf_screened_today.csv at: {screened_csv}")

    flex_positions_xml = PROJECT_ROOT / "data" / "runs" / run_date / "ibkr_flex" / "flex_positions.xml"
    if not flex_positions_xml.exists():
        raise FileNotFoundError(f"Missing Flex positions XML at: {flex_positions_xml}")

    proposed_path = resolve_proposed_trades_path(run_date)
    plan = pd.read_csv(proposed_path)
    plan = _maybe_merge_optimal_targets(plan, run_date, runs_root=RUNS_ROOT)
    if plan.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "target_net_usd",
                "actual_net_usd",
                "discrepancy_usd",
                "abs_discrepancy_usd",
                "target_gross_usd",
                "actual_gross_usd",
                "gross_gap_usd",
                "under_exposed",
            ]
        )

    cfg = load_config(PROJECT_ROOT / "config" / "strategy_config.yml")
    strategy_tag = str((cfg.get("strategy", {}) or {}).get("tag", "")).strip()
    if strategy_tag and "strategy_tag" in plan.columns:
        plan = plan[plan["strategy_tag"].astype(str) == strategy_tag].copy()

    for c in ("ETF", "Underlying"):
        if c not in plan.columns:
            raise ValueError(f"proposed_trades.csv missing required column: {c}")
    plan["ETF"] = plan["ETF"].astype(str).map(canonical_symbol).map(normalize_plan_etf_ticker)
    plan["Underlying"] = plan["Underlying"].astype(str).map(canonical_symbol)
    plan["long_usd"] = pd.to_numeric(plan.get("long_usd", 0.0), errors="coerce").fillna(0.0)
    plan["short_usd"] = pd.to_numeric(plan.get("short_usd", 0.0), errors="coerce").fillna(0.0)
    if "optimal_long_usd" in plan.columns:
        plan["optimal_long_usd"] = pd.to_numeric(plan["optimal_long_usd"], errors="coerce").fillna(
            plan["long_usd"]
        )
    if "optimal_short_usd" in plan.columns:
        plan["optimal_short_usd"] = pd.to_numeric(plan["optimal_short_usd"], errors="coerce").fillna(
            plan["short_usd"]
        )
    long_col, short_col = _resolve_target_basis_columns(plan, "optimal")

    target_under = (
        plan.groupby("Underlying", as_index=False)[long_col].sum()
        .rename(columns={"Underlying": "symbol", long_col: "target_net_usd"})
    )
    target_etf = (
        plan.groupby("ETF", as_index=False)[short_col].sum()
        .rename(columns={"ETF": "symbol", short_col: "target_net_usd"})
    )
    target = pd.concat([target_under, target_etf], ignore_index=True)
    target = target[target["symbol"].astype(bool)].copy()
    target = target.groupby("symbol", as_index=False)["target_net_usd"].sum()

    pos = parse_open_positions(flex_positions_xml)
    if pos.empty:
        actual = pd.DataFrame(columns=["symbol", "actual_net_usd"])
    else:
        pos = pos.copy()
        pos["symbol"] = pos["symbol"].astype(str).map(canonical_symbol)
        pos = pos[~pos["symbol"].isin(EXCLUDE_SYMBOLS)].copy()
        pos["actual_net_usd"] = (
            pd.to_numeric(pos["position"], errors="coerce").fillna(0.0)
            * pd.to_numeric(pos["markPrice"], errors="coerce").fillna(0.0)
            * pd.to_numeric(pos["fxRateToBase"], errors="coerce").fillna(1.0)
        )
        actual = pos.groupby("symbol", as_index=False)["actual_net_usd"].sum()

    allowed_etfs, _ = load_universe_from_screened(screened_csv)
    allowed_etfs |= set(SUPPLEMENTAL_ETF_MAP.keys())

    etf_to_under = load_etf_to_under_map(screened_csv)
    for e_sym, u_sym in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(e_sym, u_sym)

    blacklist_raw = ((cfg.get("strategy", {}) or {}).get("blacklist", [])) or []
    blacklist = {canonical_symbol(str(s)) for s in blacklist_raw if str(s).strip()}
    blocked_etfs = {s for s in blacklist if s in allowed_etfs}
    blocked_etfs |= {e for e, u in etf_to_under.items() if u in blacklist}

    merged = target.merge(actual, on="symbol", how="outer")
    merged["target_net_usd"] = pd.to_numeric(merged["target_net_usd"], errors="coerce").fillna(0.0)
    merged["actual_net_usd"] = pd.to_numeric(merged["actual_net_usd"], errors="coerce").fillna(0.0)
    merged["symbol"] = merged["symbol"].astype(str).map(canonical_symbol)
    merged = merged[merged["symbol"].isin(allowed_etfs)].copy()
    if blocked_etfs:
        merged = merged[~merged["symbol"].isin(blocked_etfs)].copy()

    merged["discrepancy_usd"] = merged["actual_net_usd"] - merged["target_net_usd"]
    merged["abs_discrepancy_usd"] = merged["discrepancy_usd"].abs()
    merged["target_gross_usd"] = merged["target_net_usd"].abs()
    merged["actual_gross_usd"] = merged["actual_net_usd"].abs()
    merged["gross_gap_usd"] = merged["actual_gross_usd"] - merged["target_gross_usd"]
    merged["under_exposed"] = merged["gross_gap_usd"] < -1e-9
    merged = merged.sort_values("abs_discrepancy_usd", ascending=False).reset_index(drop=True)
    return merged


def format_largest_discrepancies(discrepancy_df: pd.DataFrame, top_n: int = 15) -> str:
    if discrepancy_df.empty:
        return "(no discrepancy rows in screened universe)"

    top = discrepancy_df.head(top_n).copy()
    headers = ["SYMBOL", "TARGET_NET", "ACTUAL_NET", "DISCREP", "|DISCREP|", "GROSS_GAP", "FLAG"]
    row_labels = top["symbol"].astype(str).tolist() + [headers[0]]
    col_vals = {
        "TARGET_NET": [f"{v:,.0f}" for v in top["target_net_usd"]] + [headers[1]],
        "ACTUAL_NET": [f"{v:,.0f}" for v in top["actual_net_usd"]] + [headers[2]],
        "DISCREP": [f"{v:,.0f}" for v in top["discrepancy_usd"]] + [headers[3]],
        "|DISCREP|": [f"{v:,.0f}" for v in top["abs_discrepancy_usd"]] + [headers[4]],
        "GROSS_GAP": [f"{v:,.0f}" for v in top["gross_gap_usd"]] + [headers[5]],
        "FLAG": [("UNDER" if b else "") for b in top["under_exposed"]] + [headers[6]],
    }

    sym_w = max(8, max(len(s) for s in row_labels))
    tgt_w = max(10, max(len(s) for s in col_vals["TARGET_NET"]))
    act_w = max(10, max(len(s) for s in col_vals["ACTUAL_NET"]))
    d_w = max(10, max(len(s) for s in col_vals["DISCREP"]))
    ad_w = max(10, max(len(s) for s in col_vals["|DISCREP|"]))
    gg_w = max(10, max(len(s) for s in col_vals["GROSS_GAP"]))
    fl_w = max(5, max(len(s) for s in col_vals["FLAG"]))

    lines = [
        f"{'SYMBOL'.ljust(sym_w)}  {'TARGET_NET'.rjust(tgt_w)}  {'ACTUAL_NET'.rjust(act_w)}  "
        f"{'DISCREP'.rjust(d_w)}  {'|DISCREP|'.rjust(ad_w)}  {'GROSS_GAP'.rjust(gg_w)}  {'FLAG'.ljust(fl_w)}",
        "-" * (sym_w + tgt_w + act_w + d_w + ad_w + gg_w + fl_w + 12),
    ]
    for _, r in top.iterrows():
        flag = "UNDER" if bool(r["under_exposed"]) else ""
        target_net = f"{float(r['target_net_usd']):,.0f}"
        actual_net = f"{float(r['actual_net_usd']):,.0f}"
        discrep = f"{float(r['discrepancy_usd']):,.0f}"
        abs_discrep = f"{float(r['abs_discrepancy_usd']):,.0f}"
        gross_gap = f"{float(r['gross_gap_usd']):,.0f}"
        lines.append(
            f"{str(r['symbol']).ljust(sym_w)}  "
            f"{target_net.rjust(tgt_w)}  "
            f"{actual_net.rjust(act_w)}  "
            f"{discrep.rjust(d_w)}  "
            f"{abs_discrep.rjust(ad_w)}  "
            f"{gross_gap.rjust(gg_w)}  "
            f"{flag.ljust(fl_w)}"
        )
    return "\n".join(lines)


def make_position_discrepancy_plot(
    discrepancy_df: pd.DataFrame,
    run_date: str,
    top_n: int = 30,
) -> Path:
    ensure_ledger_dir()
    out_path = LEDGER_DIR / f"position_discrepancies_top_{top_n}_{run_date}.png"
    top = discrepancy_df.head(top_n).copy()

    if top.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "No discrepancies to plot", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    top = top.sort_values("abs_discrepancy_usd", ascending=True)
    y = np.arange(len(top))
    colors = np.where(top["under_exposed"], "#d62728", "#1f77b4")

    fig_h = max(6.0, 0.35 * len(top))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    bars = ax.barh(y, top["discrepancy_usd"], color=colors, alpha=0.88)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["symbol"].astype(str), fontsize=8)
    ax.set_xlabel("Discrepancy USD (actual net - target net)")
    ax.set_title(f"Top {min(top_n, len(top))} Position Discrepancies ({run_date})")
    ax.grid(axis="x", alpha=0.25)

    for bar, _, gross_gap, under in zip(
        bars, top["discrepancy_usd"], top["gross_gap_usd"], top["under_exposed"]
    ):
        x = bar.get_width()
        label = f"gap {gross_gap:,.0f}"
        align = "left" if x >= 0 else "right"
        dx = 3 if x >= 0 else -3
        ax.annotate(
            label,
            xy=(x, bar.get_y() + bar.get_height() / 2),
            xytext=(dx, 0),
            textcoords="offset points",
            va="center",
            ha=align,
            fontsize=7,
            color="#d62728" if under else "#333333",
        )

    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor="#d62728", label="Under-exposed (actual gross < target gross)"),
        Patch(facecolor="#1f77b4", label="Other discrepancies"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_position_discrepancy_csvs(run_date: str, discrepancy_df: pd.DataFrame) -> tuple[Path, Path]:
    outdir = PROJECT_ROOT / "data" / "runs" / run_date / "accounting"
    outdir.mkdir(parents=True, exist_ok=True)
    dated_path = outdir / "position_discrepancies_all.csv"
    latest_path = PROJECT_ROOT / "data" / "position_discrepancies_all.csv"

    cols = [
        "symbol",
        "target_net_usd",
        "actual_net_usd",
        "discrepancy_usd",
        "abs_discrepancy_usd",
        "target_gross_usd",
        "actual_gross_usd",
        "gross_gap_usd",
        "under_exposed",
    ]
    out = discrepancy_df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    out.to_csv(dated_path, index=False)
    out.to_csv(latest_path, index=False)
    return dated_path, latest_path


def send_email(
    *,
    subject: str,
    body: str,
    attachments: list[Path],
    recipients: list[str],
) -> None:
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASS"]
    from_addr = os.environ.get("FROM_EMAIL", smtp_user)

    if not recipients:
        raise ValueError("No valid recipients. Check PNL_RECIPIENTS.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    # Header can be a single string; envelope recipients must be a list (handled below)
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    for p in attachments:
        data = p.read_bytes()
        suffix = p.suffix.lower()

        if suffix == ".csv":
            maintype, subtype = "text", "csv"
        elif suffix == ".json":
            maintype, subtype = "application", "json"
        elif suffix == ".png":
            maintype, subtype = "image", "png"
        else:
            maintype, subtype = "application", "octet-stream"

        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=p.name)

    with smtplib.SMTP(smtp_host, smtp_port) as s:
        s.starttls()
        s.login(smtp_user, smtp_pass)
        # IMPORTANT: pass explicit envelope recipients list
        s.send_message(msg, to_addrs=recipients)

from datetime import timedelta, date

def get_previous_day(run_date: str | None = None) -> str:
    """Returns run_date string; defaults to previous day if not provided."""
    if run_date:
        return run_date
    prev_day = date.today() - timedelta(days=1)
    return prev_day.isoformat()

def main() -> int:
    run_date = os.environ.get("RUN_DATE") or get_previous_day()

    env = os.environ.copy()
    env["RUN_DATE"] = run_date
    skip_pipeline = os.environ.get("EOD_SKIP_PIPELINE", "").strip().lower() in ("1", "true", "yes")

    if not skip_pipeline:
        # 1) Pull Flex files for RUN_DATE
        run_cmd(["python", str(IBKR_FLEX_SCRIPT), "--run-date", run_date], env=env)
        # 2) Build accounting PnL outputs
        run_cmd(["python", str(IBKR_ACCT_SCRIPT), run_date], env=env)

    # 3) Load outputs (fix implausible bucket re-attribution vs prior run)
    pnl_under_csv, pnl_symbol_csv, pnl_bucket_csv, totals_json_path, totals = load_outputs(run_date)
    totals = apply_bucket_pnl_continuity(run_date, totals)
    total_pnl = float(totals.get("total_pnl", 0.0))
    headline_bucket_pnl = _bucket_pnl_from_totals(totals)
    daily_bucket_pnl = compute_bucket_daily_deltas(run_date, headline_bucket_pnl)
    headline_block = format_headline_pnl_block(
        headline_bucket_pnl,
        total_pnl=total_pnl,
        daily_buckets=daily_bucket_pnl,
    )
    accounting_method_line = format_accounting_method_line(totals)

    outdir = PROJECT_ROOT / "data" / "runs" / run_date / "accounting"
    pnl_bucket_csv = outdir / "pnl_by_bucket.csv"
    blocked_symbols, blocked_underlyings = _blocked_exposure_sets(run_date)
    blocked_exposure_keys = blocked_symbols | blocked_underlyings

    bucket_email_sections: list[str] = []
    bucket_attachment_csvs: list[Path] = []
    for bucket in BUCKET_KEYS:
        pnl_csv = outdir / f"pnl_{bucket}.csv"
        exposure_csv = outdir / f"net_exposure_{bucket}.csv"
        detail_csv = outdir / "net_exposure_bucket_4_detail.csv" if bucket == "bucket_4" else None
        bucket_email_sections.append(
            format_bucket_email_block(
                bucket,
                headline_pnl=float(headline_bucket_pnl[bucket]),
                pnl_csv=pnl_csv,
                pnl_symbol_csv=pnl_symbol_csv,
                exposure_csv=exposure_csv,
                exposure_detail_csv=detail_csv,
                blocked_keys=blocked_exposure_keys,
            )
        )
        for path in (pnl_csv, exposure_csv, detail_csv):
            if path is not None and path.exists():
                bucket_attachment_csvs.append(path)

    b3_pnl_total = float(headline_bucket_pnl["bucket_3"])

    # 5) Update history + plot since START_DATE
    grand_total = float(total_pnl)
    bucket_capital_snapshot = build_bucket_capital_snapshot_from_run(run_date)
    totals = persist_totals_dashboard_fields(run_date, totals, bucket_capital_snapshot)
    hist = update_pnl_history(
        run_date,
        b1=headline_bucket_pnl["bucket_1"],
        b2=headline_bucket_pnl["bucket_2"],
        b3=b3_pnl_total,
        b4=headline_bucket_pnl["bucket_4"],
        bucket_capital=bucket_capital_snapshot,
    )
    bucket_capital_avg = compute_average_bucket_capital(hist)
    roc_bucket_pnl = dict(headline_bucket_pnl)
    if not hist.empty:
        _hist_row = hist[hist["date"].dt.strftime("%Y-%m-%d") == run_date]
        if not _hist_row.empty:
            _r = _hist_row.iloc[-1]
            for bucket in BUCKET_KEYS:
                col = f"pnl_{bucket}"
                if col in _r.index and pd.notna(_r.get(col)):
                    roc_bucket_pnl[bucket] = float(_r.get(col) or 0.0)
    bucket_return_table = format_bucket_return_table(roc_bucket_pnl, bucket_capital_avg)
    period_pnl_summary = format_period_pnl_summary(hist, run_date)
    plot_path = make_pnl_plot(hist)

    flex_positions_xml = PROJECT_ROOT / "data" / "runs" / run_date / "ibkr_flex" / "flex_positions.xml"
    att_hist = update_attribution_history(run_date, totals, pnl_symbol_csv, flex_positions_xml)
    att_plot_path = make_attribution_plot(att_hist)

    discrepancy_df = load_position_discrepancies(run_date)
    discrepancy_plot_path = make_position_discrepancy_plot(discrepancy_df, run_date, top_n=30)
    discrepancy_table = format_largest_discrepancies(discrepancy_df, top_n=30)
    under_exposed_count = int(discrepancy_df["under_exposed"].sum()) if not discrepancy_df.empty else 0
    discrepancy_csv_path, _discrepancy_latest_csv_path = write_position_discrepancy_csvs(run_date, discrepancy_df)

    # 6) Compose email
    skip_email = os.environ.get("EOD_SKIP_EMAIL", "").strip().lower() in ("1", "true", "yes")
    recipients_raw = os.environ.get("PNL_RECIPIENTS", "")
    recipients = parse_recipients(recipients_raw)
    if not recipients and not skip_email:
        raise ValueError(f"PNL_RECIPIENTS parsed to empty list. Raw={recipients_raw!r}")

    # Use NY time in the email "As of"
    try:
        import pytz  # optional; already in your requirements
        asof = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        asof = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = format_eod_subject(run_date, headline_bucket_pnl, total_pnl=grand_total)

    n_days = int(hist.shape[0])
    if not hist.empty and all(c in hist.columns for c in PNL_HISTORY_DISPLAY_PNL_COLS):
        last = hist.iloc[-1]
        if all(pd.notna(last[c]) for c in PNL_HISTORY_DISPLAY_PNL_COLS):
            hist_summary = (
                f"Since {START_DATE}: {n_days} day(s) — latest logged YTD "
                f"B1: {float(headline_bucket_pnl['bucket_1']):,.2f} | "
                f"B2: {float(headline_bucket_pnl['bucket_2']):,.2f} | "
                f"B3: {float(headline_bucket_pnl['bucket_3']):,.2f} | "
                f"B4: {float(headline_bucket_pnl['bucket_4']):,.2f} | "
                f"Total: {float(last['total_pnl']):,.2f}\n"
            )
        else:
            cum_total = float(last["total_pnl"]) if pd.notna(last.get("total_pnl")) else grand_total
            hist_summary = (
                f"Since {START_DATE}: {n_days} day(s) logged | "
                f"Cumulative PnL (total): {cum_total:,.2f}\n"
            )
    elif not hist.empty:
        cum_total = float(hist["total_pnl"].iloc[-1])
        hist_summary = (
            f"Since {START_DATE}: {n_days} day(s) logged | "
            f"Cumulative PnL (total): {cum_total:,.2f}\n"
        )
    else:
        hist_summary = f"Since {START_DATE}: no history rows yet.\n"

    method_block = f"{accounting_method_line}\n\n" if accounting_method_line else ""
    per_bucket_body = "\n".join(bucket_email_sections)

    b124_pnl_table, _b124_pnl_total = format_underlying_table(pnl_under_csv, pnl_symbol_csv)
    net_exp_under_csv = outdir / "net_exposure_by_underlying.csv"
    b124_exp_table = format_b124_underlying_exposure_table(
        net_exp_under_csv,
        blocked_keys=blocked_exposure_keys,
    )

    body = (
        f"As of: {asof}\n"
        f"Run date: {run_date} — Previous day mark-to-market\n\n"
        f"{method_block}"
        f"Position discrepancy rows: {len(discrepancy_df)} "
        f"(under-exposed: {under_exposed_count})\n\n"
        f"{headline_block}\n\n"
        f"BUCKET RETURNS ON CAPITAL (denominators = average per-day capital since {START_DATE}):\n"
        "----------------------------------------\n"
        f"{bucket_return_table}\n"
        "----------------------------------------\n"
        "AVG_NET_CAP = average daily net capital deployed on etf_screened_today.csv tickers "
        "(signed MV: long +, short −); "
        "AVG_GROSS_CAP = average daily gross MV on the same universe; "
        "AVG_MAINT_MARGIN = average daily maintenance margin requirement. "
        "ROC = YTD PnL / avg net capital when net capital is positive. "
        "ROG / ROM = YTD PnL divided by the matching averaged denominator.\n\n"
        f"{period_pnl_summary}\n\n"
        f"B1+B2+B4 PnL BY UNDERLYING:\n"
        "----------------------------------------\n"
        f"{b124_pnl_table}\n"
        "----------------------------------------\n\n"
        f"B1+B2+B4 NET / GROSS EXPOSURE BY UNDERLYING (sorted by net, high to low):\n"
        "----------------------------------------\n"
        f"{b124_exp_table}\n"
        "----------------------------------------\n\n"
        f"{per_bucket_body}"
        "Largest Position Discrepancies (actual net vs proposed target net):\n"
        "----------------------------------------\n"
        f"{discrepancy_table}\n"
        "----------------------------------------\n"
        "UNDER flag = actual gross exposure is below target gross exposure.\n"
        "TARGET_NET uses optimal_long_usd / optimal_short_usd (structural, pre-shares_available) "
        "from proposed_trades.csv or optimal_targets.csv when merged.\n\n"
        f"{hist_summary}\n"
        "Attribution plot: long/short split uses EOD position sign (short if net shares < 0); "
        "realized on flat symbols is booked to long. Excluded cash interest is not in strategy total_pnl.\n\n"
        "Attachments:\n"
        "- pnl_by_underlying.csv  (stock sleeves rollup: B1+B2+B4)\n"
        "- pnl_by_symbol.csv\n"
        "- pnl_by_bucket.csv\n"
        "- pnl_bucket_1.csv, pnl_bucket_2.csv, pnl_bucket_3.csv, pnl_bucket_4.csv\n"
        "- net_exposure_bucket_1.csv … net_exposure_bucket_4.csv\n"
        "- net_exposure_bucket_4_detail.csv\n"
        "- totals.json\n"
        "- pnl_history.csv  (sleeve PnL and capital history)\n"
        "- pnl_attribution_history.csv\n"
        f"- {plot_path.name}\n"
        f"- {att_plot_path.name}\n"
        f"- {discrepancy_plot_path.name}\n"
        f"- {discrepancy_csv_path.name}\n"
    )

    # 7) Send (attach consolidated CSVs + totals + plots)
    attachments = [
        pnl_under_csv,
        totals_json_path,
        plot_path,
        att_plot_path,
        PNL_HISTORY_CSV,
        ATTRIBUTION_HISTORY_CSV,
        discrepancy_plot_path,
        discrepancy_csv_path,
    ]
    if pnl_symbol_csv.exists():
        attachments.insert(1, pnl_symbol_csv)
    if pnl_bucket_csv.exists():
        attachments.append(pnl_bucket_csv)
    net_exp_under_csv = outdir / "net_exposure_by_underlying.csv"
    if net_exp_under_csv.exists():
        attachments.append(net_exp_under_csv)
    seen: set[Path] = set()
    for csv_path in bucket_attachment_csvs:
        if csv_path not in seen:
            attachments.append(csv_path)
            seen.add(csv_path)

    if skip_email:
        print(f"[EOD] EOD_SKIP_EMAIL=1 — email not sent. Subject: {subject}")
    else:
        send_email(
            subject=subject,
            body=body,
            attachments=attachments,
            recipients=recipients,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())