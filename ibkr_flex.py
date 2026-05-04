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
  IBKR_FLEX_Q_BORROW_DETAILS=456789 # query id for Borrow Fee Details (daily)

Optional env vars:
  IBKR_FLEX_BASE_URL=https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService
  IBKR_FLEX_POLL_SEC=5
  IBKR_FLEX_TIMEOUT_SEC=180            # default --timeout; per-query floors in main() can raise this
  IBKR_FLEX_TRADES_TIMEOUT_SEC=10800   # min max-wait for flex_trades (1019 can exceed 90+ min on large queries)
  IBKR_FLEX_POST_SEND_DELAY_SEC=1.25   # min wait after SendRequest before GetStatement (rate limit / 1020)
  IBKR_FLEX_MAX_REFERENCE_RENEWS=8     # on 1017/1020, how many times to obtain a new reference
  IBKR_FLEX_REFERENCE_RENEW_SLEEP_SEC=2.5
  # If 1019 persists for flex_trades, IBKR sometimes never completes for one reference; optional fresh SendRequest:
  IBKR_FLEX_1019_STUCK_RENEW_AFTER_SEC=900   # first renew after this many seconds in 1019 (0 disables)
  IBKR_FLEX_1019_STUCK_RENEW_INTERVAL_SEC=600
  IBKR_FLEX_1019_STUCK_MAX_RENEWS=0          # default 0: renewals can queue extra IB jobs; set 2–4 only if IBKR confirms stuck refs
  RUN_DATE=YYYY-MM-DD

Why Flex can sit on 1019 / "still generating" for a long time (usually NOT a client bug):
  - IBKR builds the statement on their servers. Trades + FIFO / MTM / settlement sections
    (Flex error families 1005–1008) are CPU-heavy; wide date ranges or very active accounts
    routinely take many minutes.
  - Error 1009 means IB is under heavy load; you still poll, but wall-clock stretches.
  - Error 1018 (rate limit) triggers extra backoff sleeps in this client — fewer polls, longer
    apparent wait (check logs for "Throttled (1018)").
  - Your Flex *query template* (in Account Management) controls scope: all-time history,
    many columns, or combined sections inflate generation time. Narrow the period or split
    queries if EOD latency matters.
  - Optional IBKR_FLEX_1019_STUCK_* SendRequest renewals may help a stuck reference, but can
    also queue a *new* generation job; if renewals correlate with even longer waits, set
    IBKR_FLEX_1019_STUCK_MAX_RENEWS=0 to disable and rely on a single reference + timeout.

Notes:
- ErrorCode 1019 ("Statement generation in progress") is NORMAL. We treat it as "keep polling".
- ErrorCode 1018 ("Too many requests") can happen on SendRequest and sometimes GetStatement.
  We use exponential backoff + jitter for resilience.
- ErrorCode 1020 ("Invalid request or unable to validate request") often appears if
  GetStatement is called within IBKR's per-token rate window (~1 req/s) right after
  SendRequest, or when the reference code is no longer valid; we wait before the first
  GetStatement and renew the reference on 1020/1017.
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
BACKUP_BASE_URL = "https://gdcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

# IBKR: max ~1 request/sec per token across SendRequest + GetStatement combined.
_DEFAULT_POST_SEND_DELAY_SEC = 1.25

# Error codes that mean "not ready yet; keep polling with the same reference code"
# (see https://www.ibkrguides.com/adminportal/performanceandstatements/flex3error.htm).
_FLEX_PENDING_CODES = frozenset(
    {"1001", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1019", "1021"}
)

# Error codes where a fresh SendRequest reference is required
_FLEX_RENEW_REF_CODES = frozenset({"1017", "1020"})


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
    """Minimal tag extractor for <Tag>value</Tag>."""
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

    if status.lower() == "warn" and code == "1019":
        return True

    if code in _FLEX_PENDING_CODES:
        return True

    if status.lower() == "fail":
        if code in {"1019"}:
            return True
        if ("not ready" in msg) or ("processing" in msg) or ("please try again" in msg) or ("in progress" in msg):
            return True

    return False


def _flex_response_log_hint(body: str, *, max_msg: int = 160) -> str:
    """One-line summary of FlexStatementResponse for operator logs."""
    if not _is_flex_envelope(body):
        return "non-envelope body"
    st = (_get_status(body) or "?").strip()
    code = (_get_error_code(body) or "?").strip()
    msg = (_extract_tag(body, "ErrorMessage") or "").replace("\n", " ").strip()
    if len(msg) > max_msg:
        msg = msg[: max_msg - 3] + "..."
    hint = ""
    if code in {"1006", "1007", "1008"}:
        hint = "; IBKR assembling FIFO/MTM/settlement slices for this query"
    elif code == "1009":
        hint = "; IBKR reports heavy load for Flex generation"
    elif code == "1019":
        hint = "; server-side report build still running (normal while queued)"
    return f"Status={st!r} ErrorCode={code!r}{hint} — {msg!r}"


def _detect_extension(body: str) -> str:
    """Best-effort output type detection."""
    b = body.lstrip()
    if b.startswith("<?xml") or b.startswith("<FlexQueryResponse") or b.startswith("<FlexStatementResponse"):
        return "xml"
    first = b.splitlines()[0] if b.splitlines() else ""
    if "," in first:
        return "csv"
    if "|" in first:
        return "txt"
    return "txt"


def _candidate_base_urls(base_url: str) -> list[str]:
    """
    Return unique candidate Flex hosts for failover.
    If user provides ndcdyn, include gdcdyn as backup (and vice versa).
    """
    urls = [base_url.strip()]
    if "ndcdyn.interactivebrokers.com" in base_url:
        urls.append(BACKUP_BASE_URL)
    elif "gdcdyn.interactivebrokers.com" in base_url:
        urls.append(DEFAULT_BASE_URL)
    else:
        urls.append(BACKUP_BASE_URL)
    # preserve order while de-duplicating
    return list(dict.fromkeys(u for u in urls if u))


def _requests_get_with_1018_backoff(
    url: str,
    params: dict,
    timeout: float,
    max_attempts: int | None = None,
    base_sleep: float | None = None,
) -> str:
    """Wrapper for requests.get that retries on IBKR throttling (1018) and transient HTTP issues."""
    if max_attempts is None:
        max_attempts = int(os.getenv("IBKR_FLEX_HTTP_MAX_ATTEMPTS", "8"))
    if base_sleep is None:
        base_sleep = float(os.getenv("IBKR_FLEX_HTTP_BASE_SLEEP_SEC", "2.0"))
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            text = r.text

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
    """Returns ReferenceCode string if Success, else raises FlexError."""
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
    """Retains explicit SendRequest backoff behavior."""
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
    """Returns statement content as text."""
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
    enable_stuck_1019_renew: bool = False,
) -> Path:
    """SendRequest -> poll GetStatement -> save to disk."""
    ref = None
    active_base_url = base_url
    last_send_err: FlexError | None = None
    candidates = _candidate_base_urls(base_url)
    for i, candidate in enumerate(candidates, start=1):
        try:
            if i > 1:
                print(f"[FLEX] retrying SendRequest via backup host: {candidate}")
            ref = send_request_with_backoff(candidate, token, q.query_id)
            active_base_url = candidate
            break
        except FlexError as e:
            last_send_err = e
            print(f"[FLEX] SendRequest failed on host {candidate}: {e}")
            continue

    if not ref:
        raise FlexError(f"SendRequest failed on all candidate hosts for q={q.query_id}: {last_send_err}")

    post_send_delay = float(os.getenv("IBKR_FLEX_POST_SEND_DELAY_SEC", str(_DEFAULT_POST_SEND_DELAY_SEC)))
    post_send_delay = max(post_send_delay, _DEFAULT_POST_SEND_DELAY_SEC)
    max_ref_renews = int(os.getenv("IBKR_FLEX_MAX_REFERENCE_RENEWS", "8"))
    ref_renew_sleep = float(os.getenv("IBKR_FLEX_REFERENCE_RENEW_SLEEP_SEC", "2.5"))
    ref_renews_used = 0

    stuck_1019_after = float(os.getenv("IBKR_FLEX_1019_STUCK_RENEW_AFTER_SEC", "900"))
    stuck_1019_interval = float(os.getenv("IBKR_FLEX_1019_STUCK_RENEW_INTERVAL_SEC", "600"))
    stuck_1019_max = int(os.getenv("IBKR_FLEX_1019_STUCK_MAX_RENEWS", "0"))
    stuck_1019_renews = 0
    last_stuck_renew_at = 0.0

    t0 = time.time()
    time.sleep(post_send_delay)
    poll_n = 0
    polls_this_ref = 0

    if q.name == "flex_trades":
        print(
            "[FLEX] flex_trades: IBKR generates this report on their servers; long 1019 waits "
            "usually mean a large query (history span, trade count, FIFO/MTM sections) or "
            "platform load — not a bad token. Logs will repeat Flex ErrorMessage while pending."
        )

    while True:
        body = get_statement(active_base_url, token, ref)
        polls_this_ref += 1

        if _is_processing(body):
            elapsed = time.time() - t0
            if elapsed > max_wait_sec:
                code = _get_error_code(body) or "UNKNOWN"
                msg = _extract_tag(body, "ErrorMessage") or "Timed out waiting for report"
                detail = _flex_response_log_hint(body)
                raise FlexError(
                    f"Timed out fetching {q.name} (q={q.query_id}, ref={ref}): {code} {msg} "
                    f"(last response: {detail}; GetStatement polls this ref={polls_this_ref})"
                )
            code = (_get_error_code(body) or "").strip()
            # Large Trades+Commissions Flex jobs can sit on 1019 for one reference indefinitely;
            # a fresh SendRequest often queues a new worker and unblocks (empirical / IBKR forums).
            if (
                enable_stuck_1019_renew
                and stuck_1019_max > 0
                and stuck_1019_after > 0
                and code == "1019"
                and stuck_1019_renews < stuck_1019_max
            ):
                now = time.time()
                if last_stuck_renew_at == 0.0:
                    should_stuck_renew = elapsed >= stuck_1019_after
                else:
                    should_stuck_renew = (now - last_stuck_renew_at) >= stuck_1019_interval
                if should_stuck_renew:
                    stuck_1019_renews += 1
                    last_stuck_renew_at = now
                    print(
                        f"[FLEX] prolonged 1019 on {q.name} (~{elapsed:.0f}s); "
                        f"SendRequest again ({stuck_1019_renews}/{stuck_1019_max}) "
                        f"(prior ref polled {polls_this_ref}×; {_flex_response_log_hint(body)}) ..."
                    )
                    time.sleep(ref_renew_sleep)
                    ref = send_request_with_backoff(active_base_url, token, q.query_id)
                    polls_this_ref = 0
                    time.sleep(post_send_delay)
                    continue
            # Gentle backoff while IBKR is still generating (1019): reduces request rate
            # and avoids hammering the same reference when "try again shortly" applies.
            poll_n += 1
            sleep_sec = min(30.0, poll_every_sec + 0.08 * min(elapsed, 600.0))
            if poll_n % 12 == 0:
                print(
                    f"[FLEX] still waiting on {q.name} — {elapsed:.0f}s / {max_wait_sec:.0f}s "
                    f"(ref polls={polls_this_ref}) — {_flex_response_log_hint(body)}"
                )
            time.sleep(sleep_sec)
            continue

        if _is_flex_envelope(body) and "<Status>Success</Status>" not in body:
            code = (_get_error_code(body) or "").strip()
            msg = _extract_tag(body, "ErrorMessage") or body[:400]

            if code in _FLEX_RENEW_REF_CODES and ref_renews_used < max_ref_renews:
                ref_renews_used += 1
                print(
                    f"[FLEX] GetStatement {code} ({msg[:120]!r}); "
                    f"SendRequest again for {q.name} ({ref_renews_used}/{max_ref_renews}) ..."
                )
                time.sleep(ref_renew_sleep)
                ref = send_request_with_backoff(active_base_url, token, q.query_id)
                polls_this_ref = 0
                time.sleep(post_send_delay)
                continue

            raise FlexError(f"GetStatement failed for {q.name} (ref={ref}): {code} {msg}")

        ext = _detect_extension(body)
        out_path = out_dir / f"{q.name}.{ext}"
        out_path.write_text(body, encoding="utf-8", errors="ignore")
        return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=os.getenv("RUN_DATE") or today_str(), help="YYYY-MM-DD")
    ap.add_argument("--out-root", default="data/runs", help="output root folder")
    ap.add_argument("--poll", type=float, default=float(os.getenv("IBKR_FLEX_POLL_SEC", "5")), help="poll interval seconds")
    ap.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("IBKR_FLEX_TIMEOUT_SEC", "180")),
        help="max wait per report seconds (flex_trades also uses IBKR_FLEX_TRADES_TIMEOUT_SEC floor, default 10800)",
    )
    args = ap.parse_args()

    token = _env_required("IBKR_FLEX_TOKEN")
    base_url = os.getenv("IBKR_FLEX_BASE_URL", DEFAULT_BASE_URL).strip()

    # You can override these via env vars if you prefer
    q_trades = os.getenv("IBKR_FLEX_Q_TRADES", "").strip()
    q_cash = os.getenv("IBKR_FLEX_Q_CASH", "").strip()
    q_positions = os.getenv("IBKR_FLEX_Q_POSITIONS", "").strip()
    q_borrow_details = os.getenv("IBKR_FLEX_Q_BORROW_DETAILS", "").strip()

    # For local/manual runs, these defaults can be overridden by env vars.
    if not q_trades:
        q_trades = "1376360"
    if not q_cash:
        q_cash = "1376356"
    if not q_positions:
        q_positions = "1376362"
    if not q_borrow_details:
        q_borrow_details = "1408970"
    # Trades last: largest / slowest Flex job; smaller snapshots first (same total time if all
    # succeed, but avoids tying up the only slow slot before cash & positions land).
    queries = [
        FlexQuery("flex_cash", q_cash),
        FlexQuery("flex_positions", q_positions),
        FlexQuery("flex_borrow_fee_details", q_borrow_details),
        FlexQuery("flex_trades", q_trades),
    ]

    out_dir = Path(args.out_root) / args.run_date / "ibkr_flex"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FLEX] base_url={base_url}")
    print(f"[FLEX] writing to {out_dir}")

    for q in queries:
        timeout = args.timeout
        if q.name == "flex_trades":
            trades_floor = float(os.getenv("IBKR_FLEX_TRADES_TIMEOUT_SEC", "10800"))
            timeout = max(timeout, trades_floor)  # 1019 often lasts 60–120+ min on very large trade queries
        if q.name == "flex_borrow_fee_details":
            timeout = max(timeout, 300.0)  # borrow details sometimes slower than cash/positions

        print(f"[FLEX] fetching {q.name} (query_id={q.query_id}) ...")
        p = fetch_and_save(
            base_url=base_url,
            token=token,
            q=q,
            out_dir=out_dir,
            poll_every_sec=args.poll,
            max_wait_sec=timeout,
            enable_stuck_1019_renew=(q.name == "flex_trades"),
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