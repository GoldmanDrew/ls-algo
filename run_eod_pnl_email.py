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

import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent  # adjust if needed

IBKR_FLEX_SCRIPT = PROJECT_ROOT / "ibkr_flex.py"
IBKR_ACCT_SCRIPT = PROJECT_ROOT / "ibkr_accounting.py"

# History / plot outputs
LEDGER_DIR = PROJECT_ROOT / "data" / "ledger"
PNL_HISTORY_CSV = LEDGER_DIR / "pnl_history.csv"
PLOT_PNG = LEDGER_DIR / "pnl_since_2026-02-19.png"
START_DATE = "2026-02-19"


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )


def load_outputs(run_date: str) -> tuple[Path, Path, dict]:
    outdir = PROJECT_ROOT / "data" / "runs" / run_date / "accounting"
    pnl_under = outdir / "pnl_by_underlying.csv"
    totals = outdir / "totals.json"
    if not pnl_under.exists():
        raise FileNotFoundError(f"Missing: {pnl_under}")
    if not totals.exists():
        raise FileNotFoundError(f"Missing: {totals}")
    totals_obj = json.loads(totals.read_text(encoding="utf-8"))
    return pnl_under, totals, totals_obj


def format_underlying_table(pnl_under_csv: Path) -> tuple[str, float]:
    """
    Returns:
      - formatted table (plain text) of total_pnl by underlying
      - total_pnl sum
    """
    df = pd.read_csv(pnl_under_csv)

    if "underlying" not in df.columns:
        raise ValueError("pnl_by_underlying.csv missing 'underlying' column")
    if "total_pnl" not in df.columns:
        raise ValueError("pnl_by_underlying.csv missing 'total_pnl' column")

    tbl = df[["underlying", "total_pnl"]].copy()
    tbl["total_pnl"] = pd.to_numeric(tbl["total_pnl"], errors="coerce").fillna(0.0)
    tbl = tbl.sort_values("total_pnl", ascending=False)

    total = float(tbl["total_pnl"].sum())

    total_row = pd.DataFrame([{"underlying": "TOTAL", "total_pnl": total}])
    tbl2 = pd.concat([tbl, total_row], ignore_index=True)

    tbl2["total_pnl"] = tbl2["total_pnl"].map(lambda x: f"{x:,.2f}")
    underlying_width = max(10, int(tbl2["underlying"].astype(str).map(len).max()))
    pnl_width = max(12, int(tbl2["total_pnl"].astype(str).map(len).max()))

    lines: list[str] = []
    header = f"{'UNDERLYING'.ljust(underlying_width)}  {'TOTAL_PNL'.rjust(pnl_width)}"
    lines.append(header)
    lines.append("-" * (underlying_width + 2 + pnl_width))

    for _, r in tbl2.iterrows():
        u = str(r["underlying"]).ljust(underlying_width)
        p = str(r["total_pnl"]).rjust(pnl_width)
        lines.append(f"{u}  {p}")

    return "\n".join(lines), total


def ensure_ledger_dir() -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Net Exposure helpers (beta-normalized)
# ---------------------------------------------------------------
def canonical_symbol(sym: str) -> str:
    """Mirror the normalization used in ibkr_accounting.py."""
    if sym is None:
        return ""
    s = str(sym).strip().upper()
    m = re.match(r"^([A-Z]{1,5})[ \-\.]([A-Z])$", s)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return s


def parse_positions_for_exposure(pos_xml: Path) -> pd.DataFrame:
    """
    Parse flex_positions.xml to extract fields needed for exposure calc:
    symbol, position, markPrice, fxRateToBase, positionValue.
    """
    r = ET.parse(pos_xml).getroot()
    op = r.find(".//OpenPositions")
    if op is None:
        return pd.DataFrame(columns=["symbol", "position", "markPrice", "fxRateToBase"])
    rows = []
    for node in op:
        a = node.attrib
        sym = canonical_symbol(a.get("symbol", ""))
        if not sym:
            continue
        rows.append({
            "symbol": sym,
            "position": float(a.get("position", "0") or 0),
            "markPrice": float(a.get("markPrice", "0") or 0),
            "fxRateToBase": float(a.get("fxRateToBase", "1") or 1),
        })
    return pd.DataFrame(rows)


def load_etf_beta_map(screened_csv: Path) -> tuple[dict, dict]:
    """
    Load ETF -> Underlying mapping and ETF -> Beta from etf_screened_today.csv.
    Returns (etf_to_under, etf_to_beta) dicts keyed by canonical symbol.

    Beta represents the ETF's sensitivity to its underlying (e.g. 2.0 for a
    2× levered ETF, -1.0 for an inverse, 1.0 for a plain wrapper).
    """
    u = pd.read_csv(screened_csv)
    cols_lc = {c.lower(): c for c in u.columns}

    # Find columns flexibly
    etf_col = None
    for cand in ["etf", "symbol", "ticker", "etf_symbol"]:
        if cand in cols_lc:
            etf_col = cols_lc[cand]
            break
    under_col = None
    for cand in ["underlying", "underlyingsymbol", "underlying_symbol", "root"]:
        if cand in cols_lc:
            under_col = cols_lc[cand]
            break
    beta_col = None
    for cand in ["beta", "leverage", "lev"]:
        if cand in cols_lc:
            beta_col = cols_lc[cand]
            break

    if etf_col is None or under_col is None:
        return {}, {}

    u = u[[etf_col, under_col] + ([beta_col] if beta_col else [])].dropna(subset=[etf_col, under_col])
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)

    etf_to_under = dict(zip(u[etf_col], u[under_col]))

    if beta_col:
        u[beta_col] = pd.to_numeric(u[beta_col], errors="coerce").fillna(1.0)
        etf_to_beta = dict(zip(u[etf_col], u[beta_col]))
    else:
        etf_to_beta = {k: 1.0 for k in etf_to_under}

    return etf_to_under, etf_to_beta


def compute_net_exposure(
    pos_xml: Path,
    screened_csv: Path,
    pnl_underlyings: set[str] | None = None,
) -> pd.DataFrame:
    """
    Compute beta-normalized net exposure by underlying.

    For each position:
      mv_base = position × beta × markPrice × fxRateToBase

    Beta is the ETF's sensitivity to its underlying (e.g. 2.0 for a 2×
    levered ETF).  Spot / 1× holdings have beta = 1.0.

    Then group by underlying to get net_notional_usd and gross_notional_usd.

    If pnl_underlyings is provided, only include underlyings in that set
    (to keep the same universe as the PnL report).

    Returns DataFrame with columns:
      underlying, net_notional_usd, gross_notional_usd, n_legs
    """
    pos = parse_positions_for_exposure(pos_xml)
    if pos.empty:
        return pd.DataFrame(columns=["underlying", "net_notional_usd", "gross_notional_usd", "n_legs"])

    etf_to_under, etf_to_beta = load_etf_beta_map(screened_csv)

    # Map each position to its underlying and beta
    pos["is_etf"] = pos["symbol"].isin(etf_to_under)
    pos["underlying"] = np.where(
        pos["is_etf"],
        pos["symbol"].map(etf_to_under),
        pos["symbol"],
    )
    pos["beta"] = np.where(
        pos["is_etf"],
        pos["symbol"].map(etf_to_beta).astype(float),
        1.0,
    )
    pos["beta"] = pd.to_numeric(pos["beta"], errors="coerce").fillna(1.0)

    # Beta-adjusted signed market value in base currency
    pos["mv_base"] = pos["position"] * pos["beta"] * pos["markPrice"] * pos["fxRateToBase"]
    pos["gross_mv_base"] = pos["mv_base"].abs()

    # Filter to PnL universe if provided
    if pnl_underlyings is not None:
        pos = pos[pos["underlying"].isin(pnl_underlyings)].copy()

    if pos.empty:
        return pd.DataFrame(columns=["underlying", "net_notional_usd", "gross_notional_usd", "n_legs"])

    exposure = (
        pos.groupby("underlying", as_index=False)
        .agg(
            net_notional_usd=("mv_base", "sum"),
            gross_notional_usd=("gross_mv_base", "sum"),
            n_legs=("symbol", "nunique"),
        )
        .sort_values("net_notional_usd", ascending=False)
    )

    return exposure


def format_exposure_table(exposure_df: pd.DataFrame) -> tuple[str, float, float]:
    """
    Format net exposure by underlying as a plain-text table.
    Returns (table_str, total_net, total_gross).
    """
    if exposure_df.empty:
        return "(no exposure data)", 0.0, 0.0

    tbl = exposure_df[["underlying", "net_notional_usd", "gross_notional_usd"]].copy()
    total_net = float(tbl["net_notional_usd"].sum())
    total_gross = float(tbl["gross_notional_usd"].sum())

    total_row = pd.DataFrame([{
        "underlying": "TOTAL",
        "net_notional_usd": total_net,
        "gross_notional_usd": total_gross,
    }])
    tbl2 = pd.concat([tbl, total_row], ignore_index=True)

    tbl2["net_notional_usd"] = tbl2["net_notional_usd"].map(lambda x: f"{x:,.2f}")
    tbl2["gross_notional_usd"] = tbl2["gross_notional_usd"].map(lambda x: f"{x:,.2f}")

    u_w = max(10, int(tbl2["underlying"].astype(str).map(len).max()))
    net_w = max(14, int(tbl2["net_notional_usd"].astype(str).map(len).max()))
    gross_w = max(16, int(tbl2["gross_notional_usd"].astype(str).map(len).max()))

    lines: list[str] = []
    header = (
        f"{'UNDERLYING'.ljust(u_w)}  "
        f"{'NET_NOTIONAL'.rjust(net_w)}  "
        f"{'GROSS_NOTIONAL'.rjust(gross_w)}"
    )
    lines.append(header)
    lines.append("-" * (u_w + 2 + net_w + 2 + gross_w))

    for _, r in tbl2.iterrows():
        u = str(r["underlying"]).ljust(u_w)
        n = str(r["net_notional_usd"]).rjust(net_w)
        g = str(r["gross_notional_usd"]).rjust(gross_w)
        lines.append(f"{u}  {n}  {g}")

    return "\n".join(lines), total_net, total_gross


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
    pnl_under_csv, totals_json_path, totals = load_outputs(run_date)
    total_pnl = float(totals.get("total_pnl", 0.0))

    # 4) Create underlying breakdown table
    underlying_table, underlying_total = format_underlying_table(pnl_under_csv)

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
            exposure_df = compute_net_exposure(pos_xml, screened_csv, pnl_underlyings)
            exposure_table_str, total_net, total_gross = format_exposure_table(exposure_df)

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

    body = (
        f"As of: {asof}\n"
        f"Run date: {run_date}\n\n"
        f"TOTAL PnL (base): {underlying_total:,.2f}\n\n"
        "TOTAL PnL by underlying:\n"
        "----------------------------------------\n"
        f"{underlying_table}\n"
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
        "- totals.json\n"
        f"- {plot_path.name}\n"
        "- net_exposure_by_underlying.csv\n"
    )

    # 7) Send (attach CSV + totals + plot + exposure)
    attachments = [pnl_under_csv, totals_json_path, plot_path]
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
