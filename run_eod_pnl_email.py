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


def format_underlying_table(pnl_under_csv: Path, pnl_symbol_csv: Path) -> tuple[str, float]:
    """
    Returns:
      - formatted table (plain text) of total_pnl by underlying, with per-symbol
        PnL rows indented underneath each underlying
      - total_pnl sum
    """
    df = pd.read_csv(pnl_under_csv)
    if "underlying" not in df.columns or "total_pnl" not in df.columns:
        raise ValueError("pnl_by_underlying.csv missing required columns")

    df["total_pnl"] = pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0)
    df = df.sort_values("total_pnl", ascending=False)
    total = float(df["total_pnl"].sum())

    # Load per-symbol detail if available
    sym_df: pd.DataFrame = pd.DataFrame()
    if pnl_symbol_csv.exists():
        try:
            sym_df = pd.read_csv(pnl_symbol_csv)
            sym_df["total_pnl"] = pd.to_numeric(sym_df["total_pnl"], errors="coerce").fillna(0.0)
        except Exception:
            sym_df = pd.DataFrame()

    # Column widths — include symbol names in the width calculation
    all_labels = list(df["underlying"].astype(str))
    if not sym_df.empty and "symbol" in sym_df.columns:
        all_labels += ["  " + s for s in sym_df["symbol"].astype(str)]
    all_labels += ["TOTAL"]
    label_width = max(12, max(len(s) for s in all_labels))

    all_pnl_strs = [f"{v:,.2f}" for v in list(df["total_pnl"]) + [total]]
    if not sym_df.empty and "total_pnl" in sym_df.columns:
        all_pnl_strs += [f"{v:,.2f}" for v in sym_df["total_pnl"]]
    pnl_width = max(12, max(len(s) for s in all_pnl_strs))

    lines: list[str] = []
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
                # Show realized vs unrealized breakdown if available
                detail = ""
                if "realized_pnl" in sr and "unrealized_pnl" in sr:
                    r_pnl = float(sr["realized_pnl"])
                    u_pnl = float(sr["unrealized_pnl"])
                    detail = f"  (r: {r_pnl:,.2f}  u: {u_pnl:,.2f})"
                lines.append(f"{sym_label.ljust(label_width)}  {sym_pnl.rjust(pnl_width)}{detail}")

    lines.append("-" * (label_width + 2 + pnl_width))
    total_str = f"{total:,.2f}"
    lines.append(f"{'TOTAL'.ljust(label_width)}  {total_str.rjust(pnl_width)}")

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
        "bucket_1_high_leverage": "Bucket 1 — High Leverage (β ≥ cutoff)",
        "bucket_2_low_leverage":  "Bucket 2 — Low Leverage / Spot (0 ≤ β < cutoff)",
        "bucket_3_inverse":       "Bucket 3 — Inverse (β < 0)",
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


def update_pnl_history(run_date: str, total_pnl: float) -> pd.DataFrame:
    """
    Appends (or overwrites) a row in pnl_history.csv for the given run_date.
    Returns the full history DF filtered from START_DATE onward.
    """
    ensure_ledger_dir()

    row = pd.DataFrame([{"date": run_date, "total_pnl": float(total_pnl)}])

    if PNL_HISTORY_CSV.exists():
        hist = pd.read_csv(PNL_HISTORY_CSV)
        if "date" not in hist.columns or "total_pnl" not in hist.columns:
            hist = row
        else:
            hist["date"] = hist["date"].astype(str)
            hist = hist[hist["date"] != run_date]
            hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    hist["total_pnl"] = pd.to_numeric(hist["total_pnl"], errors="coerce").fillna(0.0)

    hist_out = hist.copy()
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    hist_out.to_csv(PNL_HISTORY_CSV, index=False)

    start_dt = pd.to_datetime(START_DATE)
    hist = hist[hist["date"] >= start_dt].copy()
    return hist


def make_pnl_plot(history: pd.DataFrame) -> Path:
    """
    Creates a PNG plot showing YTD PnL since START_DATE.
    Each row in history already contains the YTD cumulative total_pnl,
    so we plot it directly (no cumsum).
    """
    ensure_ledger_dir()

    if history.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"PnL since {START_DATE} (no data yet)")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL (base)")
        fig.tight_layout()
        fig.savefig(PLOT_PNG, dpi=150)
        plt.close(fig)
        return PLOT_PNG

    x = history["date"]
    y = history["total_pnl"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker="o", linewidth=2, markersize=6)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_title(f"YTD PnL since {START_DATE}")
    ax.set_xlabel("Date")
    ax.set_ylabel("YTD PnL (base)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # Annotate each point with its value
    for xi, yi in zip(x, y):
        label = f"${yi:,.0f}"
        ax.annotate(label, (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

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


def main() -> int:
    run_date = os.environ.get("RUN_DATE") or date.today().isoformat()

    # 1) Pull Flex files for RUN_DATE
    env = os.environ.copy()
    env["RUN_DATE"] = run_date
    run_cmd(["python", str(IBKR_FLEX_SCRIPT), "--run-date", run_date], env=env)

    # 2) Build accounting PnL outputs
    run_cmd(["python", str(IBKR_ACCT_SCRIPT), run_date], env=env)

    # 3) Load outputs
    pnl_under_csv, pnl_symbol_csv, pnl_bucket_csv, totals_json_path, totals = load_outputs(run_date)
    total_pnl = float(totals.get("total_pnl", 0.0))

    # 4) Create underlying breakdown table + bucket table
    underlying_table, underlying_total = format_underlying_table(pnl_under_csv, pnl_symbol_csv)
    bucket_table = format_bucket_table(pnl_bucket_csv)

    # 4b) Compute beta-normalized net exposure by underlying
    flex_dir = PROJECT_ROOT / "data" / "runs" / run_date / "ibkr_flex"
    pos_xml = flex_dir / "flex_positions.xml"
    screened_csv = PROJECT_ROOT / "data" / "etf_screened_today.csv"

    pnl_underlyings: set[str] | None = None
    try:
        pnl_df = pd.read_csv(pnl_under_csv)
        if "underlying" in pnl_df.columns:
            pnl_underlyings = set(pnl_df["underlying"].dropna().astype(str))
    except Exception:
        pass

    exposure_df = pd.DataFrame()
    exposure_table_str = "(exposure data unavailable)"
    total_net = 0.0
    total_gross = 0.0
    exposure_csv_path: Path | None = None

    if pos_xml.exists() and screened_csv.exists():
        try:
            exposure_df, pos_detail_df = compute_net_exposure(pos_xml, screened_csv, pnl_underlyings)
            exposure_table_str, total_net, total_gross = format_exposure_table(exposure_df, pos_detail_df)

            # Save exposure CSV alongside the other accounting outputs
            outdir = PROJECT_ROOT / "data" / "runs" / run_date / "accounting"
            exposure_csv_path = outdir / "net_exposure_by_underlying.csv"
            exposure_df.to_csv(exposure_csv_path, index=False)
        except Exception as e:
            exposure_table_str = f"(exposure calculation error: {e})"

    # 5) Update history + plot since START_DATE
    hist = update_pnl_history(run_date, total_pnl=underlying_total)
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

    subject = f"EOD PnL by Underlying — {run_date} — Total: {underlying_total:,.2f}"

    cum_total = float(hist["total_pnl"].iloc[-1]) if not hist.empty else 0.0
    n_days = int(hist.shape[0])

    beta_cutoff = totals.get("beta_cutoff", 1.5)

    body = (
        f"As of: {asof}\n"
        f"Run date: {run_date}\n\n"
        f"TOTAL PnL (base): {underlying_total:,.2f}\n\n"
        "TOTAL PnL by underlying:\n"
        "----------------------------------------\n"
        f"{underlying_table}\n"
        "----------------------------------------\n\n"
        f"PnL BY BUCKET (beta_cutoff = {beta_cutoff}):\n"
        "----------------------------------------\n"
        f"{bucket_table}\n"
        "----------------------------------------\n\n"
        f"NET EXPOSURE by underlying (beta-normalized):\n"
        f"  Net notional:   {total_net:,.2f}\n"
        f"  Gross notional: {total_gross:,.2f}\n"
        "----------------------------------------\n"
        f"{exposure_table_str}\n"
        "----------------------------------------\n\n"
        f"Since {START_DATE}: {n_days} day(s) logged | Cumulative PnL: {cum_total:,.2f}\n\n"
        "Attachments:\n"
        "- pnl_by_underlying.csv\n"
        "- pnl_by_symbol.csv\n"
        "- pnl_by_bucket.csv\n"
        "- totals.json\n"
        f"- {plot_path.name}\n"
        "- net_exposure_by_underlying.csv\n"
    )

    # 7) Send (attach CSV + symbol + bucket + totals + plot + exposure)
    attachments = [pnl_under_csv, totals_json_path, plot_path]
    if pnl_symbol_csv.exists():
        attachments.insert(1, pnl_symbol_csv)   # underlying → symbol → bucket → totals
    if pnl_bucket_csv.exists():
        attachments.insert(2, pnl_bucket_csv)
    if exposure_csv_path is not None and exposure_csv_path.exists():
        attachments.append(exposure_csv_path)

    send_email(
        subject=subject,
        body=body,
        attachments=attachments,
        recipients=recipients,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())