#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from datetime import date, datetime
import smtplib
from email.message import EmailMessage

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent  # adjust if needed

IBKR_FLEX_SCRIPT = PROJECT_ROOT / "ibkr_flex.py"
IBKR_ACCT_SCRIPT = PROJECT_ROOT / "ibkr_accounting.py"

# New: history / plot outputs
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

    # Keep just what we need for the email
    tbl = df[["underlying", "total_pnl"]].copy()
    tbl["total_pnl"] = pd.to_numeric(tbl["total_pnl"], errors="coerce").fillna(0.0)

    # Sort biggest contributors first
    tbl = tbl.sort_values("total_pnl", ascending=False)

    total = float(tbl["total_pnl"].sum())

    # Add TOTAL row
    total_row = pd.DataFrame([{"underlying": "TOTAL", "total_pnl": total}])
    tbl2 = pd.concat([tbl, total_row], ignore_index=True)

    # Format to fixed-width text table
    tbl2["total_pnl"] = tbl2["total_pnl"].map(lambda x: f"{x:,.2f}")
    underlying_width = max(10, int(tbl2["underlying"].astype(str).map(len).max()))
    pnl_width = max(12, int(tbl2["total_pnl"].astype(str).map(len).max()))

    lines = []
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
            # If schema changed or file corrupted, start fresh
            hist = row
        else:
            # Drop existing same-date row, then append
            hist["date"] = hist["date"].astype(str)
            hist = hist[hist["date"] != run_date]
            hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row

    # Normalize/sort
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date")
    hist["total_pnl"] = pd.to_numeric(hist["total_pnl"], errors="coerce").fillna(0.0)

    # Save back
    hist_out = hist.copy()
    hist_out["date"] = hist_out["date"].dt.strftime("%Y-%m-%d")
    hist_out.to_csv(PNL_HISTORY_CSV, index=False)

    # Return filtered history from START_DATE onward
    start_dt = pd.to_datetime(START_DATE)
    hist = hist[hist["date"] >= start_dt].copy()
    return hist


def make_pnl_plot(history: pd.DataFrame) -> Path:
    """
    Creates a PNG plot showing daily PnL and cumulative PnL since START_DATE.
    Saves to PLOT_PNG and returns that Path.
    """
    ensure_ledger_dir()

    if history.empty:
        # Create an empty placeholder plot
        plt.figure()
        plt.title(f"PnL since {START_DATE} (no data yet)")
        plt.xlabel("Date")
        plt.ylabel("PnL")
        plt.tight_layout()
        plt.savefig(PLOT_PNG, dpi=150)
        plt.close()
        return PLOT_PNG

    x = history["date"]
    daily = history["total_pnl"]
    cum = daily.cumsum()

    # Plot cumulative (primary)
    plt.figure()
    plt.plot(x, cum)
    plt.title(f"Cumulative PnL since {START_DATE}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (base)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=150)
    plt.close()

    # If you also want a daily plot, uncomment and attach a second PNG:
    # daily_png = LEDGER_DIR / f"daily_pnl_since_{START_DATE}.png"
    # plt.figure()
    # plt.plot(x, daily)
    # plt.title(f"Daily PnL since {START_DATE}")
    # plt.xlabel("Date")
    # plt.ylabel("Daily PnL (base)")
    # plt.xticks(rotation=30, ha="right")
    # plt.tight_layout()
    # plt.savefig(daily_png, dpi=150)
    # plt.close()

    return PLOT_PNG


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

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
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
        s.send_message(msg)


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

    # 5) Update history + plot since 2026-02-19
    hist = update_pnl_history(run_date, total_pnl=underlying_total)
    plot_path = make_pnl_plot(hist)

    # 6) Compose email
    recipients = [r.strip() for r in os.environ["PNL_RECIPIENTS"].split(",") if r.strip()]
    asof = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"EOD PnL by Underlying — {run_date} — Total: {underlying_total:,.2f}"

    # Include a short summary of the plotted series
    cum_total = float(hist["total_pnl"].cumsum().iloc[-1]) if not hist.empty else 0.0
    n_days = int(hist.shape[0])

    body = (
        f"As of: {asof}\n"
        f"Run date: {run_date}\n\n"
        f"TOTAL PnL (base): {underlying_total:,.2f}\n\n"
        "TOTAL PnL by underlying:\n"
        "----------------------------------------\n"
        f"{underlying_table}\n"
        "----------------------------------------\n\n"
        f"Since {START_DATE}: {n_days} day(s) logged | Cumulative PnL: {cum_total:,.2f}\n\n"
        "Attachments:\n"
        "- pnl_by_underlying.csv\n"
        "- totals.json\n"
        f"- {plot_path.name}\n"
    )

    # 7) Send (attach CSV + totals + plot)
    send_email(
        subject=subject,
        body=body,
        attachments=[pnl_under_csv, totals_json_path, plot_path],
        recipients=recipients,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
