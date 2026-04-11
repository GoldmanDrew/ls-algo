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
    compute_net_exposure,
    format_exposure_table,
    load_etf_to_under_map,
    load_universe_from_screened,
    parse_open_positions,
)
from strategy_config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent  # adjust if needed

IBKR_FLEX_SCRIPT = PROJECT_ROOT / "ibkr_flex.py"
IBKR_ACCT_SCRIPT = PROJECT_ROOT / "ibkr_accounting.py"

# History / plot outputs
LEDGER_DIR = PROJECT_ROOT / "data" / "ledger"
RUNS_ROOT = PROJECT_ROOT / "data" / "runs"
PNL_HISTORY_CSV = LEDGER_DIR / "pnl_history.csv"
PLOT_PNG = LEDGER_DIR / "pnl_since_2026-02-27.png"
START_DATE = "2026-02-27"
PAIR_EXPOSURE_MIN_ABS_NET_USD = 500.0


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

    df = df.sort_values("total_pnl", ascending=False)
    section_total = float(df["total_pnl"].sum())

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

    return "\n".join(lines), section_total


def format_underlying_table(pnl_under_csv: Path, pnl_symbol_csv: Path) -> tuple[str, float]:
    """
    Returns:
      - formatted table (plain text) of total_pnl by underlying (bucket 1&2 only),
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
                sym_df = sym_df[sym_df["bucket"].isin(["bucket_1", "bucket_2"])]
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


def ensure_ledger_dir() -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)


PNL_HISTORY_BUCKET_COLS = ("pnl_bucket_1", "pnl_bucket_2", "pnl_bucket_3")


def format_pair_exposure_flags(
    exposure_df: pd.DataFrame,
    *,
    net_col: str = "net_notional_usd",
    gross_col: str = "gross_notional_usd",
    label_col: str = "underlying",
    threshold: float = 0.05,
    min_abs_net_usd: float = PAIR_EXPOSURE_MIN_ABS_NET_USD,
) -> str:
    """
    Flag rows (e.g. per-underlying pairs) where |net| > threshold * gross and
    |net| >= min_abs_net_usd. Returns multi-line text for the email body.
    """
    if exposure_df is None or exposure_df.empty:
        return "Pair exposure: (no bucket 1&2 underlying exposure table)."
    if label_col not in exposure_df.columns or net_col not in exposure_df.columns or gross_col not in exposure_df.columns:
        return "Pair exposure: (missing columns for net/gross check)."

    df = exposure_df.copy()
    df[net_col] = pd.to_numeric(df[net_col], errors="coerce").fillna(0.0)
    df[gross_col] = pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0)

    rows: list[tuple[str, float, float, float]] = []
    for _, r in df.iterrows():
        gross = float(r[gross_col])
        net = float(r[net_col])
        anet = abs(net)
        if gross <= 0:
            continue
        if anet < min_abs_net_usd:
            continue
        if anet <= threshold * gross:
            continue
        lab = str(r[label_col])
        pct = 100.0 * anet / gross
        rows.append((lab, net, gross, pct))

    if not rows:
        return (
            "Pair exposure: no underlying exceeds 5% |net| vs gross "
            f"(bucket 1&2 pairs; |net| ≥ ${min_abs_net_usd:,.0f} only)."
        )

    rows.sort(key=lambda t: abs(t[1]), reverse=True)
    lines = [
        "⚠️ Pair exposure — |net| > 5% of gross "
        f"(listed only if |net| ≥ ${min_abs_net_usd:,.0f}):",
        "",
    ]
    for lab, net, gross, pct in rows:
        lines.append(
            f"  • {lab:8}  net ${abs(net):>10,.0f}  gross ${gross:>11,.0f}  ratio {pct:5.1f}%"
        )
    return "\n".join(lines)


def read_bucket_pnl_from_run(run_date_str: str) -> tuple[float, float, float] | None:
    """
    Bucket YTD-style totals from a prior accounting run, if present.
    Prefer totals.json bucket_pnl; else sum pnl_bucket_{1,2,3}.csv.
    """
    outdir = RUNS_ROOT / run_date_str / "accounting"
    if not outdir.is_dir():
        return None

    totals_path = outdir / "totals.json"
    if totals_path.exists():
        try:
            obj = json.loads(totals_path.read_text(encoding="utf-8"))
            bp = obj.get("bucket_pnl")
            if isinstance(bp, dict) and bp:
                return (
                    float(bp.get("bucket_1", 0.0)),
                    float(bp.get("bucket_2", 0.0)),
                    float(bp.get("bucket_3", 0.0)),
                )
        except Exception:
            pass

    paths = [outdir / f"pnl_bucket_{i}.csv" for i in (1, 2, 3)]
    if not all(p.exists() for p in paths):
        return None
    sums: list[float] = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty or "total_pnl" not in df.columns:
            sums.append(0.0)
        else:
            sums.append(
                float(pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0).sum())
            )
    return sums[0], sums[1], sums[2]


def enrich_history_bucket_cols_from_runs(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Fill pnl_bucket_* (and total_pnl) from data/runs/<date>/accounting for every
    run date on or after START_DATE that has bucket outputs.
    """
    if not RUNS_ROOT.is_dir():
        return hist

    start_dt = pd.to_datetime(START_DATE)
    updates: dict[str, tuple[float, float, float]] = {}
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
        triple = read_bucket_pnl_from_run(ds)
        if triple is None:
            continue
        updates[ds] = triple

    if not updates:
        return hist

    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"])
    for c in PNL_HISTORY_BUCKET_COLS:
        if c not in hist.columns:
            hist[c] = np.nan
    if "total_pnl" not in hist.columns:
        hist["total_pnl"] = np.nan

    date_key = hist["date"].dt.strftime("%Y-%m-%d")
    new_rows: list[dict] = []
    for ds, (b1, b2, b3) in updates.items():
        tot = b1 + b2 + b3
        m = date_key == ds
        if m.any():
            hist.loc[m, "pnl_bucket_1"] = b1
            hist.loc[m, "pnl_bucket_2"] = b2
            hist.loc[m, "pnl_bucket_3"] = b3
            hist.loc[m, "total_pnl"] = tot
        else:
            new_rows.append(
                {
                    "date": pd.to_datetime(ds),
                    "pnl_bucket_1": b1,
                    "pnl_bucket_2": b2,
                    "pnl_bucket_3": b3,
                    "total_pnl": tot,
                }
            )

    if new_rows:
        hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)

    hist = hist.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    hist = hist.reset_index(drop=True)
    return hist


def update_pnl_history(
    run_date: str,
    *,
    b1: float,
    b2: float,
    b3: float,
) -> pd.DataFrame:
    """
    Appends (or overwrites) a row in pnl_history.csv for the given run_date.
    Stores per-bucket YTD PnL columns plus total_pnl (= sum of buckets).
    Returns the full history DF filtered from START_DATE onward.
    """
    ensure_ledger_dir()

    total_pnl = float(b1) + float(b2) + float(b3)
    row = pd.DataFrame(
        [
            {
                "date": run_date,
                "pnl_bucket_1": float(b1),
                "pnl_bucket_2": float(b2),
                "pnl_bucket_3": float(b3),
                "total_pnl": total_pnl,
            }
        ]
    )

    if PNL_HISTORY_CSV.exists():
        hist = pd.read_csv(PNL_HISTORY_CSV)
        if "date" not in hist.columns:
            hist = row
        else:
            hist["date"] = hist["date"].astype(str)
            for c in PNL_HISTORY_BUCKET_COLS:
                if c not in hist.columns:
                    hist[c] = np.nan
            if "total_pnl" not in hist.columns:
                hist["total_pnl"] = np.nan
            hist = hist[hist["date"] != run_date]
            hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    for c in PNL_HISTORY_BUCKET_COLS:
        hist[c] = pd.to_numeric(hist.get(c, np.nan), errors="coerce")
    hist["total_pnl"] = pd.to_numeric(hist["total_pnl"], errors="coerce")

    hist = enrich_history_bucket_cols_from_runs(hist)

    hist_out = hist.copy()
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    hist_out.to_csv(PNL_HISTORY_CSV, index=False)

    start_dt = pd.to_datetime(START_DATE)
    hist = hist[hist["date"] >= start_dt].copy()
    return hist


# Stable label placement per series: offset from marker in points (dx, dy), ha, va.
# Spreads B1/B2/B3 horizontally on the same date so labels do not stack in one column.
_PNL_LABEL_STYLE: dict[str, tuple[float, float, str, str]] = {
    "#1f77b4": (11, 7, "left", "bottom"),   # Bucket 1 — levered
    "#ff7f0e": (0, 9, "center", "bottom"),  # Bucket 2 — standard
    "#2ca02c": (-11, 7, "right", "bottom"),  # Bucket 3 — inverse
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
    Creates a PNG plot showing YTD PnL since START_DATE, one series per bucket.
    Each row should hold YTD cumulative PnL per bucket (and total_pnl); legacy
    rows may only have total_pnl with NaN bucket columns (no lines for those dates).
    """
    ensure_ledger_dir()

    bucket_specs: tuple[tuple[str, str, str], ...] = (
        ("pnl_bucket_1", "Bucket 1 — Levered", "#1f77b4"),
        ("pnl_bucket_2", "Bucket 2 — Standard", "#ff7f0e"),
        ("pnl_bucket_3", "Bucket 3 — Inverse", "#2ca02c"),
    )

    if history.empty:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.set_title(f"YTD PnL by bucket since {START_DATE} (no data yet)", fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("YTD PnL (base)")
        fig.tight_layout()
        fig.savefig(PLOT_PNG, dpi=150)
        plt.close(fig)
        return PLOT_PNG

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    bucket_label_points: list[tuple[pd.Timestamp, float, str]] = []
    label_fs = 5.0

    for col, label, color in bucket_specs:
        if col not in history.columns:
            continue
        y = pd.to_numeric(history[col], errors="coerce")
        mask = y.notna()
        if not mask.any():
            continue
        dates = history.loc[mask, "date"]
        yv = y[mask]
        ax.plot(dates, yv, marker="o", linewidth=1.6, markersize=4, label=label, color=color)
        for xi, yi in zip(dates, yv):
            bucket_label_points.append((pd.Timestamp(xi), float(yi), color))

    if bucket_label_points:
        _annotate_pnl_point_labels_stable(
            ax, bucket_label_points, fontsize=label_fs, is_legacy=False
        )

    # Dates with only legacy total_pnl (no per-bucket columns filled) still appear as one series
    if "total_pnl" in history.columns:
        tot = pd.to_numeric(history["total_pnl"], errors="coerce")
        if all(c in history.columns for c in PNL_HISTORY_BUCKET_COLS):
            legacy_only = history[list(PNL_HISTORY_BUCKET_COLS)].isna().all(axis=1)
        else:
            legacy_only = pd.Series(True, index=history.index)
        mask = tot.notna() & legacy_only
        if mask.any():
            dates = history.loc[mask, "date"]
            yv = tot[mask]
            ax.plot(
                dates,
                yv,
                marker="o",
                linewidth=1.6,
                markersize=4,
                color="0.35",
                linestyle="--",
                label="Total (legacy, before per-bucket history)",
            )
            legacy_pts = [
                (pd.Timestamp(xi), float(yi), "0.35") for xi, yi in zip(dates, yv)
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
    plan["ETF"] = plan["ETF"].astype(str).map(canonical_symbol)
    plan["Underlying"] = plan["Underlying"].astype(str).map(canonical_symbol)
    plan["long_usd"] = pd.to_numeric(plan.get("long_usd", 0.0), errors="coerce").fillna(0.0)
    plan["short_usd"] = pd.to_numeric(plan.get("short_usd", 0.0), errors="coerce").fillna(0.0)

    target_under = (
        plan.groupby("Underlying", as_index=False)["long_usd"].sum()
        .rename(columns={"Underlying": "symbol", "long_usd": "target_net_usd"})
    )
    target_etf = (
        plan.groupby("ETF", as_index=False)["short_usd"].sum()
        .rename(columns={"ETF": "symbol", "short_usd": "target_net_usd"})
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

    # 1) Pull Flex files for RUN_DATE
    env = os.environ.copy()
    env["RUN_DATE"] = run_date
    run_cmd(["python", str(IBKR_FLEX_SCRIPT), "--run-date", run_date], env=env)

    # 2) Build accounting PnL outputs
    run_cmd(["python", str(IBKR_ACCT_SCRIPT), run_date], env=env)

    # 3) Load outputs
    pnl_under_csv, pnl_symbol_csv, pnl_bucket_csv, totals_json_path, totals = load_outputs(run_date)
    total_pnl = float(totals.get("total_pnl", 0.0))

    # 4) Create underlying breakdown table (bucket 1&2 combined) + bucket table
    underlying_table, underlying_total = format_underlying_table(pnl_under_csv, pnl_symbol_csv)
    bucket_table = format_bucket_table(pnl_bucket_csv)

    # 4a) Per-bucket PnL
    outdir = PROJECT_ROOT / "data" / "runs" / run_date / "accounting"

    pnl_b1_csv = outdir / "pnl_bucket_1.csv"
    pnl_b2_csv = outdir / "pnl_bucket_2.csv"
    pnl_b3_csv = outdir / "pnl_bucket_3.csv"

    # Bucket 1 PnL (levered ETFs + pro-rata spot)
    b1_pnl_total = 0.0
    b1_pnl_table = "(no bucket 1 data)"
    if pnl_b1_csv.exists():
        try:
            _b1_df = pd.read_csv(pnl_b1_csv)
            if not _b1_df.empty and "total_pnl" in _b1_df.columns:
                _b1_df["total_pnl"] = pd.to_numeric(_b1_df["total_pnl"], errors="coerce").fillna(0.0)
                b1_pnl_total = float(_b1_df["total_pnl"].sum())
                # Build per-symbol detail from pnl_by_symbol filtered to bucket_1
                _sym_b1 = pd.DataFrame()
                if pnl_symbol_csv.exists():
                    try:
                        _sym_all = pd.read_csv(pnl_symbol_csv)
                        _sym_b1 = _sym_all[_sym_all.get("bucket", pd.Series()) == "bucket_1"]
                    except Exception:
                        pass
                b1_pnl_table, _ = _format_underlying_section(_b1_df, _sym_b1, section_label=None)
        except Exception:
            pass

    # Bucket 2 PnL (standard ETFs + pro-rata spot)
    b2_pnl_total = 0.0
    b2_pnl_table = "(no bucket 2 data)"
    if pnl_b2_csv.exists():
        try:
            _b2_df = pd.read_csv(pnl_b2_csv)
            if not _b2_df.empty and "total_pnl" in _b2_df.columns:
                _b2_df["total_pnl"] = pd.to_numeric(_b2_df["total_pnl"], errors="coerce").fillna(0.0)
                b2_pnl_total = float(_b2_df["total_pnl"].sum())
                _sym_b2 = pd.DataFrame()
                if pnl_symbol_csv.exists():
                    try:
                        _sym_all2 = pd.read_csv(pnl_symbol_csv)
                        _sym_b2 = _sym_all2[_sym_all2.get("bucket", pd.Series()) == "bucket_2"]
                    except Exception:
                        pass
                b2_pnl_table, _ = _format_underlying_section(_b2_df, _sym_b2, section_label=None)
        except Exception:
            pass

    # Bucket 3 PnL (inverse/hedge by symbol)
    b3_pnl_table, b3_pnl_total = format_bucket_3_pnl(pnl_b3_csv)

    # 4b) Load pre-computed exposure tables from accounting outputs
    exposure_csv_path = outdir / "net_exposure_by_underlying.csv"
    exposure_b1_csv_path = outdir / "net_exposure_bucket_1.csv"
    exposure_b2_csv_path = outdir / "net_exposure_bucket_2.csv"
    exposure_b3_csv_path = outdir / "net_exposure_bucket_3.csv"

    # Combined bucket 1+2 exposure
    exposure_table_str = "(exposure data unavailable)"
    total_net = 0.0
    total_gross = 0.0
    pair_exposure_line = "Pair exposure: (exposure file missing)."
    if exposure_csv_path.exists():
        try:
            exposure_df = pd.read_csv(exposure_csv_path)
            exposure_table_str, total_net, total_gross = format_exposure_table(exposure_df)
            pair_exposure_line = format_pair_exposure_flags(exposure_df)
        except Exception as e:
            exposure_table_str = f"(exposure error: {e})"
            pair_exposure_line = f"Pair exposure: (error loading exposure: {e})"

    # Bucket 1 exposure
    b1_exposure_table_str = "(no bucket 1 exposure data)"
    b1_net = 0.0
    b1_gross = 0.0
    if exposure_b1_csv_path.exists():
        try:
            exposure_b1_df = pd.read_csv(exposure_b1_csv_path)
            b1_exposure_table_str, b1_net, b1_gross = format_exposure_table(exposure_b1_df)
        except Exception as e:
            b1_exposure_table_str = f"(bucket 1 exposure error: {e})"

    # Bucket 2 exposure
    b2_exposure_table_str = "(no bucket 2 exposure data)"
    b2_net = 0.0
    b2_gross = 0.0
    if exposure_b2_csv_path.exists():
        try:
            exposure_b2_df = pd.read_csv(exposure_b2_csv_path)
            b2_exposure_table_str, b2_net, b2_gross = format_exposure_table(exposure_b2_df)
        except Exception as e:
            b2_exposure_table_str = f"(bucket 2 exposure error: {e})"

    # Bucket 3 exposure
    b3_exposure_table_str = "(no bucket 3 exposure data)"
    b3_net = 0.0
    b3_gross = 0.0
    if exposure_b3_csv_path.exists():
        try:
            exposure_b3_df = pd.read_csv(exposure_b3_csv_path)
            b3_exposure_table_str, b3_net, b3_gross = format_exposure_table(exposure_b3_df)
        except Exception as e:
            b3_exposure_table_str = f"(bucket 3 exposure error: {e})"

    # 5) Update history + plot since START_DATE
    grand_total = b1_pnl_total + b2_pnl_total + b3_pnl_total
    hist = update_pnl_history(
        run_date, b1=b1_pnl_total, b2=b2_pnl_total, b3=b3_pnl_total
    )
    plot_path = make_pnl_plot(hist)

    discrepancy_df = load_position_discrepancies(run_date)
    discrepancy_plot_path = make_position_discrepancy_plot(discrepancy_df, run_date, top_n=30)
    discrepancy_table = format_largest_discrepancies(discrepancy_df, top_n=30)
    under_exposed_count = int(discrepancy_df["under_exposed"].sum()) if not discrepancy_df.empty else 0

    # 6) Compose email
    recipients_raw = os.environ.get("PNL_RECIPIENTS", "")
    recipients = parse_recipients(recipients_raw)
    if not recipients:
        raise ValueError(f"PNL_RECIPIENTS parsed to empty list. Raw={recipients_raw!r}")

    # Use NY time in the email "As of"
    try:
        import pytz  # optional; already in your requirements
        asof = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        asof = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = (
        f"EOD PnL — {run_date} — "
        f"B1: {b1_pnl_total:,.2f} | B2: {b2_pnl_total:,.2f} | B3: {b3_pnl_total:,.2f} | "
        f"Total: {grand_total:,.2f}"
    )

    n_days = int(hist.shape[0])
    if not hist.empty and all(c in hist.columns for c in PNL_HISTORY_BUCKET_COLS):
        last = hist.iloc[-1]
        if all(pd.notna(last[c]) for c in PNL_HISTORY_BUCKET_COLS):
            hist_summary = (
                f"Since {START_DATE}: {n_days} day(s) — latest logged YTD "
                f"B1: {float(last['pnl_bucket_1']):,.2f} | "
                f"B2: {float(last['pnl_bucket_2']):,.2f} | "
                f"B3: {float(last['pnl_bucket_3']):,.2f} | "
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

    body = (
        f"As of: {asof}\n"
        f"Run date: {run_date} — Previous day mark-to-market\n"
        f"{pair_exposure_line}\n\n"
        f"Position discrepancy rows: {len(discrepancy_df)} "
        f"(under-exposed: {under_exposed_count})\n\n"
        f"TOTAL PnL (base): {grand_total:,.2f}\n"
        f"  Bucket 1 (Levered, β > 1.5):    {b1_pnl_total:,.2f}\n"
        f"  Bucket 2 (Standard, 0 < β ≤ 1.5): {b2_pnl_total:,.2f}\n"
        f"  Bucket 3 (Inverse, β < 0):       {b3_pnl_total:,.2f}\n\n"
        "════════════════════════════════════════\n"
        "PnL Bucket 1 — Levered (β > 1.5) by underlying:\n"
        "----------------------------------------\n"
        f"{b1_pnl_table}\n"
        "----------------------------------------\n\n"
        "PnL Bucket 2 — Standard (0 < β ≤ 1.5) by underlying:\n"
        "----------------------------------------\n"
        f"{b2_pnl_table}\n"
        "----------------------------------------\n\n"
        "PnL Bucket 3 — Inverse / Hedge (by symbol):\n"
        "----------------------------------------\n"
        f"{b3_pnl_table}\n"
        "----------------------------------------\n\n"
        "PnL by underlying (Bucket 1&2 combined):\n"
        "----------------------------------------\n"
        f"{underlying_table}\n"
        "----------------------------------------\n\n"
        "PnL BY BUCKET:\n"
        "----------------------------------------\n"
        f"{bucket_table}\n"
        "----------------------------------------\n\n"
        f"NET EXPOSURE Bucket 1 — Levered (beta-normalized):\n"
        f"  Net notional:   {b1_net:,.2f}\n"
        f"  Gross notional: {b1_gross:,.2f}\n"
        "----------------------------------------\n"
        f"{b1_exposure_table_str}\n"
        "----------------------------------------\n\n"
        f"NET EXPOSURE Bucket 2 — Standard (beta-normalized):\n"
        f"  Net notional:   {b2_net:,.2f}\n"
        f"  Gross notional: {b2_gross:,.2f}\n"
        "----------------------------------------\n"
        f"{b2_exposure_table_str}\n"
        "----------------------------------------\n\n"
        f"NET EXPOSURE Bucket 3 — Inverse / Hedge:\n"
        f"  Net notional:   {b3_net:,.2f}\n"
        f"  Gross notional: {b3_gross:,.2f}\n"
        "----------------------------------------\n"
        f"{b3_exposure_table_str}\n"
        "----------------------------------------\n\n"
        f"NET EXPOSURE by underlying (Bucket 1&2 combined, beta-normalized):\n"
        f"  Net notional:   {total_net:,.2f}\n"
        f"  Gross notional: {total_gross:,.2f}\n"
        "----------------------------------------\n"
        f"{exposure_table_str}\n"
        "----------------------------------------\n\n"
        "Largest Position Discrepancies (actual net vs proposed target net):\n"
        "----------------------------------------\n"
        f"{discrepancy_table}\n"
        "----------------------------------------\n"
        "UNDER flag = actual gross exposure is below target gross exposure.\n\n"
        f"{hist_summary}\n"
        "Attachments:\n"
        "- pnl_by_underlying.csv  (bucket 1&2 combined)\n"
        "- pnl_bucket_1.csv\n"
        "- pnl_bucket_2.csv\n"
        "- pnl_bucket_3.csv\n"
        "- pnl_by_symbol.csv\n"
        "- pnl_by_bucket.csv\n"
        "- totals.json\n"
        f"- {plot_path.name}\n"
        f"- {discrepancy_plot_path.name}\n"
        "- net_exposure_by_underlying.csv\n"
        "- net_exposure_bucket_1.csv\n"
        "- net_exposure_bucket_2.csv\n"
        "- net_exposure_bucket_3.csv\n"
    )

    # 7) Send (attach all CSVs + totals + plot + exposure)
    attachments = [pnl_under_csv, totals_json_path, plot_path, discrepancy_plot_path]
    if pnl_symbol_csv.exists():
        attachments.insert(1, pnl_symbol_csv)
    for csv_path in [pnl_b1_csv, pnl_b2_csv, pnl_b3_csv, pnl_bucket_csv]:
        if csv_path.exists():
            attachments.append(csv_path)
    for csv_path in [exposure_csv_path, exposure_b1_csv_path, exposure_b2_csv_path, exposure_b3_csv_path]:
        if csv_path.exists():
            attachments.append(csv_path)

    send_email(
        subject=subject,
        body=body,
        attachments=attachments,
        recipients=recipients,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())