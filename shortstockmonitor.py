#!/usr/bin/env python3
import ftplib
import io
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import smtplib
from email.mime.text import MIMEText

# -----------------------------
#  CONFIG
# -----------------------------

FTP_HOST = "ftp2.interactivebrokers.com"
FTP_USER = "shortstock"
FTP_PASS = ""

# Which FTP file to fetch (usa.txt is the default US short list)
FTP_FILE = os.getenv("IBKR_FTP_FILE", "usa.txt")

# Base workspace (GitHub Actions sets this; otherwise use current dir)
WORKSPACE = Path(os.getenv("GITHUB_WORKSPACE", ".")).resolve()

# Folder (relative to WORKSPACE) where daily snapshots go
# e.g. WORKSPACE / "data/shortstock_snapshots"
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", "data/shortstock_snapshots"))

# Optional override for snapshot date (e.g. "20251207")
# If not set, we use today's UTC date.
SNAPSHOT_DATE = os.getenv("SNAPSHOT_DATE", "")

# Ticker file used for watchlist alerts
TICKER_FILE = os.getenv("TICKER_FILE", "/config/tickers.csv")

# Thresholds (can tune via env vars)
BORROW_ABS_THRESHOLD = float(os.getenv("BORROW_ABS_THRESHOLD", "0.30"))      # 30%+
BORROW_CHANGE_THRESHOLD = float(os.getenv("BORROW_CHANGE_THRESHOLD", "0.05")) # +/- 5% change
AVAIL_ABS_THRESHOLD = int(os.getenv("AVAIL_ABS_THRESHOLD", "10000"))          # < 10k shares
AVAIL_CHANGE_THRESHOLD = int(os.getenv("AVAIL_CHANGE_THRESHOLD", "50000"))    # big change

# Email config (use env vars / k8s Secret)
EMAIL_FROM = os.getenv("EMAIL_FROM", "dag5wd@virginia.edu")
EMAIL_TO = os.getenv("EMAIL_TO", "dag5wd@virginia.edu")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")


# -----------------------------
#  HELPERS
# -----------------------------

def load_ticker_list(path: str) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ticker list file not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df[df.columns[0]].dropna().astype(str).str.upper().tolist()

    if path.suffix == ".json":
        with open(path, "r") as f:
            lst = json.load(f)
        return [str(x).upper() for x in lst]

    raise ValueError("Ticker file must be .csv or .json")


WATCH_TICKERS = load_ticker_list(TICKER_FILE)
print("Loaded watch tickers:", WATCH_TICKERS)


# -----------------------------
#  FETCH + PARSE IBKR FTP FILE
# -----------------------------

def fetch_ibkr_shortstock_file(filename: str = FTP_FILE) -> pd.DataFrame:
    """
    Connect to IBKR FTP, download the given shortstock file, and parse it into a DataFrame.

    The file format is pipe-delimited with a header line starting with "#SYM|...".
    We:
      - Find the header line
      - Use it to name the columns
      - Parse the following lines as data
    """
    print(f"Connecting to FTP: {FTP_HOST}, retrieving file: {filename}")
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login(user=FTP_USER, passwd=FTP_PASS)

    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {filename}", buf.write)
    ftp.quit()
    print("Download complete, parsing...")

    buf.seek(0)
    text = buf.getvalue().decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # Find header line (starts with "#SYM|")
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("#SYM|"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header line starting with '#SYM|' in FTP file")

    header_line = lines[header_idx]
    data_lines = lines[header_idx + 1:]

    header_cols = [c.strip().lstrip("#").lower() for c in header_line.split("|")]

    data_buf = io.StringIO("\n".join(data_lines))
    df = pd.read_csv(data_buf, sep="|", header=None, engine="python")

    # Align columns in case of mismatch
    n_cols = min(len(header_cols), df.shape[1])
    df = df.iloc[:, :n_cols]
    df.columns = header_cols[:n_cols]

    # Drop any unnamed junk columns if present
    df = df.drop(
        columns=[c for c in df.columns if not c or str(c).lower().startswith("unnamed")],
        errors="ignore",
    )

    print(f"Parsed {df.shape[0]} rows and {df.shape[1]} columns from FTP file.")
    return df


# -----------------------------
#  SAVE DAILY SNAPSHOT
# -----------------------------

def get_current_snapshot_date() -> datetime:
    """Return the datetime corresponding to this run's snapshot date."""
    if SNAPSHOT_DATE:
        # Expecting format YYYYMMDD
        return datetime.strptime(SNAPSHOT_DATE, "%Y%m%d")
    return datetime.utcnow()


def save_daily_snapshot(short_df: pd.DataFrame, as_of: Optional[datetime] = None) -> Path:
    """
    Save the full shortstock FTP file as a date-stamped CSV snapshot.

    Example output path:
      $WORKSPACE/data/shortstock_snapshots/usa_20251207.csv

    Returns the path to the saved file.
    """
    if as_of is None:
        as_of = get_current_snapshot_date()

    # Use override date if provided, else today's UTC date (YYYYMMDD)
    if SNAPSHOT_DATE:
        date_str = SNAPSHOT_DATE
    else:
        date_str = as_of.strftime("%Y%m%d")

    # Use the FTP file stem ("usa" from "usa.txt") in the filename
    ftp_stem = Path(FTP_FILE).stem
    snapshot_dir = WORKSPACE / SNAPSHOT_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_dir / f"{ftp_stem}_{date_str}.csv"
    short_df.to_csv(snapshot_path, index=False)

    print(f"Saved full daily IBKR shortstock snapshot → {snapshot_path}")
    return snapshot_path


# -----------------------------
#  BUILD SHORT MAPS (WATCHLIST)
# -----------------------------

def build_short_maps(tickers: List[str], df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Given a full short_df from the FTP file, build a dict for the specified tickers:
        {
          "ABNY": {"borrow": 0.12, "rebate": 0.01, "available": 6000},
          ...
        }
    """
    tickers = [t.upper() for t in tickers]

    df = df.copy()
    df["sym"] = df["sym"].astype(str).str.upper().str.strip()
    df["rebate_annual"] = pd.to_numeric(df["rebaterate"], errors="coerce") / 100.0
    df["fee_annual"] = pd.to_numeric(df["feerate"], errors="coerce") / 100.0
    df["available_int"] = pd.to_numeric(df["available"], errors="coerce")

    df["net_borrow_annual"] = df["fee_annual"] - df["rebate_annual"]
    df["net_borrow_annual"] = df["net_borrow_annual"].clip(lower=0)

    sub = df[df["sym"].isin(tickers)].copy()

    out: Dict[str, Dict] = {}
    for _, row in sub.iterrows():
        sym = row["sym"]
        out[sym] = {
            "borrow": float(row["net_borrow_annual"]) if not pd.isna(row["net_borrow_annual"]) else None,
            "rebate": float(row["rebate_annual"]) if not pd.isna(row["rebate_annual"]) else None,
            "available": int(row["available_int"]) if not pd.isna(row["available_int"]) else None,
        }
    return out


# -----------------------------
#  PREVIOUS SNAPSHOT LOADER
# -----------------------------

def load_previous_watch_state(current_snapshot_date: datetime) -> Dict[str, Dict]:
    """
    Look at SNAPSHOT_DIR, find the most recent snapshot CSV (same FTP file stem)
    with a date strictly before current_snapshot_date, and build a watchlist map
    from it.

    If no previous file exists, return {}.
    """
    snapshot_dir = WORKSPACE / SNAPSHOT_DIR
    ftp_stem = Path(FTP_FILE).stem

    if not snapshot_dir.exists():
        print("No snapshot directory yet; treating as first run (no previous state).")
        return {}

    # Find all files matching ftp_stem_YYYYMMDD.csv
    candidates = []
    for p in snapshot_dir.glob(f"{ftp_stem}_*.csv"):
        name = p.stem  # e.g. "usa_20251207"
        parts = name.split("_")
        if len(parts) < 2:
            continue
        date_str = parts[-1]
        if len(date_str) != 8 or not date_str.isdigit():
            continue
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue
        if dt < current_snapshot_date:
            candidates.append((dt, p))

    if not candidates:
        print("No previous snapshot file found; treating as first run (no previous state).")
        return {}

    # Pick the latest one before current date
    candidates.sort(key=lambda x: x[0])
    prev_date, prev_path = candidates[-1]
    print(f"Loading previous snapshot from {prev_path} (date {prev_date.date()})")

    prev_df = pd.read_csv(prev_path)
    prev_map = build_short_maps(WATCH_TICKERS, prev_df)
    return prev_map


# -----------------------------
#  ALERT LOGIC
# -----------------------------

def generate_alerts(current: Dict[str, Dict], previous: Dict[str, Dict]) -> str:
    lines = []

    for sym, cur in current.items():
        borrow = cur.get("borrow")
        avail = cur.get("available")

        prev = previous.get(sym, {})
        prev_borrow = prev.get("borrow")
        prev_avail = prev.get("available")

        msgs = []

        # Absolute levels
        if borrow is not None and borrow > BORROW_ABS_THRESHOLD:
            msgs.append(f"borrow {borrow:.1%} > {BORROW_ABS_THRESHOLD:.1%}")
        if avail is not None and avail < AVAIL_ABS_THRESHOLD:
            msgs.append(f"available {avail:,} < {AVAIL_ABS_THRESHOLD:,}")

        # Changes vs previous snapshot
        if prev_borrow is not None and borrow is not None:
            delta_b = borrow - prev_borrow
            if abs(delta_b) > BORROW_CHANGE_THRESHOLD:
                msgs.append(
                    f"borrow change {delta_b:+.1%} (prev {prev_borrow:.1%} → now {borrow:.1%})"
                )

        if prev_avail is not None and avail is not None:
            delta_a = avail - prev_avail
            if abs(delta_a) > AVAIL_CHANGE_THRESHOLD:
                msgs.append(
                    f"available change {delta_a:+,} (prev {prev_avail:,} → now {avail:,})"
                )

        if msgs:
            lines.append(f"{sym}: " + "; ".join(msgs))

    if not lines:
        return ""  # no alerts

    header = "IBKR Short Stock Monitor – Alerts\n\n"
    return header + "\n".join(lines)


# -----------------------------
#  EMAIL SENDER
# -----------------------------

def send_email(subject: str, body: str):
    if not SMTP_USER or not SMTP_PASS:
        print("No SMTP creds set; skipping email. Message would have been:\n", body)
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())


# -----------------------------
#  MAIN
# -----------------------------

def main():
    current_date = get_current_snapshot_date()

    # 1) Fetch today’s short file from IBKR
    short_df = fetch_ibkr_shortstock_file()

    # 2) Save full daily snapshot (for historical analysis later)
    save_daily_snapshot(short_df, as_of=current_date)

    # 3) Build current watchlist map
    current_map = build_short_maps(WATCH_TICKERS, short_df)

    # 4) Build previous watchlist map from the last snapshot file (if any)
    previous_map = load_previous_watch_state(current_date)

    # 5) Generate alerts
    alert_text = generate_alerts(current_map, previous_map)

    if alert_text:
        print(alert_text)
        send_email("IBKR Short Borrow / Availability Alert", alert_text)
    else:
        print("No alerts for watchlist today.")

    print("Done — snapshot saved and alert check complete.")


if __name__ == "__main__":
    main()
