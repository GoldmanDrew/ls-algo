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
    canonical_symbol,
    compute_net_exposure,
    format_exposure_table,
)


PROJECT_ROOT = Path(__file__).resolve().parent  # adjust if needed

IBKR_FLEX_SCRIPT = PROJECT_ROOT / "ibkr_flex.py"
IBKR_ACCT_SCRIPT = PROJECT_ROOT / "ibkr_accounting.py"

# History / plot outputs
LEDGER_DIR = PROJECT_ROOT / "data" / "ledger"
PNL_HISTORY_CSV = LEDGER_DIR / "pnl_history.csv"
PLOT_PNG = LEDGER_DIR / "pnl_since_2026-02-27.png"
START_DATE = "2026-02-27"


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
) -> str:
    """
    Flag rows (e.g. per-underlying pairs) where |net| > threshold * gross for that row.
    Returns a single line suitable for the top of the email body.
    """
    if exposure_df is None or exposure_df.empty:
        return "Pair exposure: (no bucket 1&2 underlying exposure table)."
    if label_col not in exposure_df.columns or net_col not in exposure_df.columns or gross_col not in exposure_df.columns:
        return "Pair exposure: (missing columns for net/gross check)."

    df = exposure_df.copy()
    df[net_col] = pd.to_numeric(df[net_col], errors="coerce").fillna(0.0)
    df[gross_col] = pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0)

    flags: list[str] = []
    for _, r in df.iterrows():
        gross = float(r[gross_col])
        net = float(r[net_col])
        if gross <= 0:
            continue
        if abs(net) > threshold * gross:
            lab = str(r[label_col])
            pct = 100.0 * abs(net) / gross
            flags.append(
                f"{lab} (|net|={abs(net):,.0f} vs gross={gross:,.0f} → {pct:.1f}%)"
            )

    if not flags:
        return "Pair exposure: no underlying exceeds 5% |net| vs its gross (bucket 1&2 pairs)."

    joined = "; ".join(flags)
    return f"⚠️ Pair exposure — |net| > 5% of gross for: {joined}"


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

    hist_out = hist.copy()
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    hist_out.to_csv(PNL_HISTORY_CSV, index=False)

    start_dt = pd.to_datetime(START_DATE)
    hist = hist[hist["date"] >= start_dt].copy()
    return hist


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
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.set_title(f"YTD PnL by bucket since {START_DATE} (no data yet)")
        ax.set_xlabel("Date")
        ax.set_ylabel("YTD PnL (base)")
        fig.tight_layout()
        fig.savefig(PLOT_PNG, dpi=150)
        plt.close(fig)
        return PLOT_PNG

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    for si, (col, label, color) in enumerate(bucket_specs):
        if col not in history.columns:
            continue
        y = pd.to_numeric(history[col], errors="coerce")
        mask = y.notna()
        if not mask.any():
            continue
        dates = history.loc[mask, "date"]
        yv = y[mask]
        ax.plot(dates, yv, marker="o", linewidth=2, markersize=5, label=label, color=color)
        for xi, yi in zip(dates, yv):
            ax.annotate(
                f"${yi:,.0f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8 + si * 13),
                ha="center",
                fontsize=7,
                color=color,
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
                linewidth=2,
                markersize=6,
                color="0.35",
                linestyle="--",
                label="Total (legacy, before per-bucket history)",
            )
            for xi, yi in zip(dates, yv):
                ax.annotate(
                    f"${yi:,.0f}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    color="0.35",
                )

    ax.set_title(f"YTD PnL by bucket since {START_DATE}")
    ax.set_xlabel("Date")
    ax.set_ylabel("YTD PnL (base)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", fontsize=8)

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
        "- net_exposure_by_underlying.csv\n"
        "- net_exposure_bucket_1.csv\n"
        "- net_exposure_bucket_2.csv\n"
        "- net_exposure_bucket_3.csv\n"
    )

    # 7) Send (attach all CSVs + totals + plot + exposure)
    attachments = [pnl_under_csv, totals_json_path, plot_path]
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