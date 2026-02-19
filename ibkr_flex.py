#!/usr/bin/env python3
"""
pull_ibkr_flex.py

Download multiple IBKR Flex Queries via Flex Web Service (v3).

How it works:
1) SendRequest -> returns a ReferenceCode
2) GetStatement -> poll until report is ready, then save content (CSV/XML/TXT)

Required env vars:
  IBKR_FLEX_TOKEN=...               # your Flex Web Service token
  IBKR_FLEX_Q_TRADES=123456         # query id for Trades+Commissions
  IBKR_FLEX_Q_CASH=234567           # query id for Cash Transactions (dividends, fees, borrow)
  IBKR_FLEX_Q_POSITIONS=345678      # query id for Positions / NAV / open positions snapshot

Optional env vars:
  IBKR_FLEX_BASE_URL=https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService
  IBKR_FLEX_POLL_SEC=5
  IBKR_FLEX_TIMEOUT_SEC=180
  RUN_DATE=YYYY-MM-DD

Notes:
- ErrorCode 1019 ("Statement generation in progress") is NORMAL. We treat it as "keep polling".
- ErrorCode 1018 ("Too many requests") can happen on SendRequest and sometimes GetStatement.
  We use exponential backoff + jitter for resilience.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import requests


DEFAULT_BASE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"


def today_str() -> str:
    return date.today().isoformat()


@dataclass
class FlexQuery:
    name: str
    query_id: str


class FlexError(RuntimeError):
    pass


def _env_required(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise FlexError(f"Missing required environment variable: {key}")
    return v.strip()


def _extract_tag(xml_text: str, tag: str) -> Optional[str]:
    """
    Minimal tag extractor for <Tag>value</Tag>.
    Works for the simple XML IBKR returns from SendRequest/GetStatement error envelopes.
    """
    open_t = f"<{tag}>"
    close_t = f"</{tag}>"
    a = xml_text.find(open_t)
    b = xml_text.find(close_t)
    if a == -1 or b == -1 or b <= a:
        return None
    return xml_text[a + len(open_t) : b].strip()


def _is_flex_envelope(body: str) -> bool:
    return "<FlexStatementResponse" in body or "<FlexStatementResponse>" in body


def _get_error_code(body: str) -> Optional[str]:
    return _extract_tag(body, "ErrorCode")


def _get_status(body: str) -> Optional[str]:
    return _extract_tag(body, "Status")


def _is_processing(body: str) -> bool:
    """
    Treat these as "still generating / try again":
    - Status=Warn, ErrorCode=1019 (statement generation in progress)
    - Status=Fail with messages like "not ready" / "processing" / "try again"
    """
    if not _is_flex_envelope(body):
        return False

    status = (_get_status(body) or "").strip()
    code = (_get_error_code(body) or "").strip()
    msg = (_extract_tag(body, "ErrorMessage") or "").lower()

    # The key case you reported:
    if status.lower() == "warn" and code == "1019":
        return True

    # Some accounts get "Fail" while still processing
    if status.lower() == "fail":
        if code in {"1019"}:
            return True
        if ("not ready" in msg) or ("processing" in msg) or ("please try again" in msg) or ("in progress" in msg):
            return True

    return False


def _detect_extension(body: str) -> str:
    """
    Best-effort output type detection.
    If user configured Flex output as CSV/TXT/XML, body will reflect it.
    """
    b = body.lstrip()
    if b.startswith("<?xml") or b.startswith("<FlexQueryResponse") or b.startswith("<FlexStatementResponse"):
        return "xml"
    first = b.splitlines()[0] if b.splitlines() else ""
    if "," in first:
        return "csv"
    if "|" in first:
        return "txt"
    return "txt"


def _requests_get_with_1018_backoff(url: str, params: dict, timeout: float, max_attempts: int = 8, base_sleep: float = 2.0) -> str:
    """
    Wrapper for requests.get that retries on IBKR throttling (1018) and transient HTTP issues.
    Returns response.text.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            text = r.text

            # IBKR sometimes returns 200 with FlexStatementResponse containing throttling
            if _is_flex_envelope(text):
                code = _get_error_code(text)
                if code == "1018":
                    sleep = min(base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 1.5), 120.0)
                    print(f"[FLEX] Throttled (1018). Backing off {sleep:.1f}s (attempt {attempt}/{max_attempts})")
                    time.sleep(sleep)
                    continue

            return text

        except requests.RequestException as e:
            last_err = e
            sleep = min(base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 1.5), 60.0)
            print(f"[FLEX] HTTP error. Backing off {sleep:.1f}s (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(sleep)

    raise FlexError(f"HTTP request failed after {max_attempts} attempts: {last_err}")


def send_request(base_url: str, token: str, query_id: str, version: int = 3) -> str:
    """
    Returns ReferenceCode string if Success, else raises FlexError.
    """
    url = f"{base_url}/SendRequest"
    params = {"t": token, "q": query_id, "v": str(version)}

    text = _requests_get_with_1018_backoff(url, params=params, timeout=30)

    if "<Status>Success</Status>" not in text:
        err_code = _extract_tag(text, "ErrorCode") or "UNKNOWN"
        err_msg = _extract_tag(text, "ErrorMessage") or text[:400]
        raise FlexError(f"SendRequest failed for q={query_id}: {err_code} {err_msg}")

    ref = _extract_tag(text, "ReferenceCode")
    if not ref:
        raise FlexError(f"SendRequest succeeded but no ReferenceCode found for q={query_id}")
    return ref.strip()


def send_request_with_backoff(
    base_url: str,
    token: str,
    query_id: str,
    version: int = 3,
    max_attempts: int = 8,
    base_sleep: float = 5.0,
) -> str:
    """
    Retains your explicit SendRequest backoff behavior.
    (send_request already handles 1018 via _requests_get_with_1018_backoff, but we keep this
     wrapper so your existing structure stays intact and you can tune separately if desired.)
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return send_request(base_url, token, query_id, version=version)
        except FlexError as e:
            msg = str(e)
            last_err = e

            if " 1018 " in msg or "ErrorCode>1018" in msg or "Too many requests" in msg:
                sleep = base_sleep * (2 ** (attempt - 1))
                sleep = min(sleep, 120.0)
                sleep += random.uniform(0, 2.0)
                print(f"[FLEX] Throttled (1018). Backing off {sleep:.1f}s (attempt {attempt}/{max_attempts})")
                time.sleep(sleep)
                continue

            raise

    raise FlexError(f"SendRequest failed after {max_attempts} attempts for q={query_id}: {last_err}")


def get_statement(base_url: str, token: str, reference_code: str, version: int = 3) -> str:
    """
    Returns statement content as text.
    If not ready, IBKR returns an XML response with Status Warn/Fail and error codes (e.g., 1019).
    """
    url = f"{base_url}/GetStatement"
    params = {"t": token, "q": reference_code, "v": str(version)}
    return _requests_get_with_1018_backoff(url, params=params, timeout=60)


def fetch_and_save(
    *,
    base_url: str,
    token: str,
    q: FlexQuery,
    out_dir: Path,
    poll_every_sec: float = 5.0,
    max_wait_sec: float = 180.0,
) -> Path:
    """
    SendRequest -> poll GetStatement -> save to disk.
    """
    ref = send_request_with_backoff(base_url, token, q.query_id)
    t0 = time.time()
    last_body = None

    while True:
        body = get_statement(base_url, token, ref)
        last_body = body

        # Correct handling for 1019 (generation in progress) and similar "not ready" envelopes
        if _is_processing(body):
            if time.time() - t0 > max_wait_sec:
                code = _get_error_code(body) or "UNKNOWN"
                msg = _extract_tag(body, "ErrorMessage") or "Timed out waiting for report"
                raise FlexError(
                    f"Timed out fetching {q.name} (q={q.query_id}, ref={ref}): {code} {msg}"
                )
            time.sleep(poll_every_sec)
            continue

        # If we still got a Flex envelope but it's NOT processing, treat as real error
        if _is_flex_envelope(body) and "<Status>Success</Status>" not in body:
            code = _get_error_code(body) or "UNKNOWN"
            msg = _extract_tag(body, "ErrorMessage") or body[:400]
            raise FlexError(f"GetStatement failed for {q.name} (ref={ref}): {code} {msg}")

        # Otherwise: it's report content
        ext = _detect_extension(body)
        out_path = out_dir / f"{q.name}.{ext}"
        out_path.write_text(body, encoding="utf-8", errors="ignore")
        return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=os.getenv("RUN_DATE") or today_str(), help="YYYY-MM-DD")
    ap.add_argument("--out-root", default="data/runs", help="output root folder")
    ap.add_argument("--poll", type=float, default=float(os.getenv("IBKR_FLEX_POLL_SEC", "5")), help="poll interval seconds")
    ap.add_argument("--timeout", type=float, default=float(os.getenv("IBKR_FLEX_TIMEOUT_SEC", "180")), help="max wait per report seconds")
    args = ap.parse_args()

    # REQUIRED: use env vars correctly
    #token = _env_required("IBKR_FLEX_TOKEN")
    token = "605237565772720861934459" # MICHAEL TOKEN
    #token = "195413602443563105417466" # DREW TOKEN
    base_url = os.getenv("IBKR_FLEX_BASE_URL", DEFAULT_BASE_URL).strip()

    # You can override these via env vars if you prefer
    q_trades = os.getenv("IBKR_FLEX_Q_TRADES", "").strip()
    q_cash = os.getenv("IBKR_FLEX_Q_CASH", "").strip()
    q_positions = os.getenv("IBKR_FLEX_Q_POSITIONS", "").strip()

    # If you want to hardcode IDs while testing, you can set them here,
    # but DO NOT hardcode the token in code.
    if not q_trades:
        q_trades = "1376360" # MICHAEL
        #q_trades = "1374436" # DREW
    if not q_cash:
        q_cash = "1376356" # MICHAEL
        #q_cash = "1374451" # DREW
    if not q_positions:
        q_positions = "1376362" # MICHAEL
        #q_positions = "1374454" # DREW
        

    queries = [
        FlexQuery("flex_trades", q_trades),
        FlexQuery("flex_cash", q_cash),
        FlexQuery("flex_positions", q_positions),
    ]

    out_dir = Path(args.out_root) / args.run_date / "ibkr_flex"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FLEX] base_url={base_url}")
    print(f"[FLEX] writing to {out_dir}")

    for q in queries:
        # Trades can take longer—give it extra time without requiring user to change global timeout
        timeout = args.timeout
        if q.name == "flex_trades":
            timeout = max(timeout, 600.0)  # 10 minutes (adjust as needed)

        print(f"[FLEX] fetching {q.name} (query_id={q.query_id}) ...")
        p = fetch_and_save(
            base_url=base_url,
            token=token,
            q=q,
            out_dir=out_dir,
            poll_every_sec=args.poll,
            max_wait_sec=timeout,
        )
        print(f"[FLEX] wrote {p}")
        time.sleep(3)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FlexError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(2)
