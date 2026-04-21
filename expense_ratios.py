"""
expense_ratios.py — Fetch annualized fund expense ratios for the ETF universe.

Priority order (per ticker):
  1. Manual YAML override   (config/etf_expense_ratios.yml)
  2. Issuer-specific scrape (authoritative, where URL-scrapeable)
  3. yfinance .info fallback (broad coverage, occasionally stale)

Returns, for each ticker, a decimal ratio (e.g. 0.0099 == 99 bps) and a
``source`` tag:
    ``manual`` | ``yieldmax`` | ``proshares`` | ``rex_shares`` | ``roundhill``
    | ``tradr`` | ``yfinance`` | ``missing``.

Notes
-----
* Several issuers render their fund pages via client-side JavaScript
  (GraniteShares, Leverage Shares, Direxion) and cannot be scraped without a
  headless browser. Those fall through to the yfinance fallback — or, when
  yfinance is known-wrong, the manual YAML override.
* yfinance calls are rate-limited (serial, short sleep) to avoid the Yahoo
  quoteSummary endpoint's aggressive 429 throttle.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger("expense_ratios")

_DEFAULT_TIMEOUT_SEC = 12
_YF_MIN_INTERVAL_SEC = 0.25
_YF_LOCK = threading.Lock()
_YF_LAST_CALL_AT: float = 0.0


# ──────────────────────────────────────────────
# HTTP session (shared across issuer scrapers)
# ──────────────────────────────────────────────
def _build_session() -> requests.Session:
    retry = Retry(
        total=2,
        backoff_factor=0.35,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def _http_get(session: requests.Session, url: str) -> str | None:
    try:
        r = session.get(url, timeout=_DEFAULT_TIMEOUT_SEC, allow_redirects=True)
        # Some issuer sites redirect unknown tickers to the homepage — treat
        # that as a miss so we don't mis-attribute a wrong value.
        if r.status_code != 200 or not r.text:
            return None
        return r.text, r.url
    except Exception as e:
        LOGGER.debug("GET %s failed: %s", url, e)
        return None


def _plausible(v: float) -> bool:
    """Reject obvious non-expense-ratio numbers (NAV, leverage, % return)."""
    return 0.0005 < v < 0.05  # 5 bps .. 5 % covers all real-world ETF fees


# ──────────────────────────────────────────────
# Per-issuer scrapers
# ──────────────────────────────────────────────
# YieldMax: `https://yieldmaxetfs.com/our-etfs/{ticker}/`
# Look for the pattern  "Expense Ratio" <tags> "0.99%"
_YMAX_RE = re.compile(
    r"Expense\s+Ratio[^%\d]{0,200}?(\d{1,2}\.\d{1,3})\s*%",
    re.IGNORECASE | re.DOTALL,
)


def _scrape_yieldmax(session, ticker: str) -> float | None:
    for url in (
        f"https://www.yieldmaxetfs.com/our-etfs/{ticker.lower()}/",
        f"https://yieldmaxetfs.com/our-etfs/{ticker.lower()}/",
    ):
        res = _http_get(session, url)
        if not res:
            continue
        html, final_url = res
        if ticker.lower() not in final_url.lower():
            continue  # redirected to homepage → unknown ticker
        m = _YMAX_RE.search(html)
        if m:
            v = float(m.group(1)) / 100.0
            if _plausible(v):
                return round(v, 6)
    return None


# ProShares: official snapshot IDs in the rendered HTML
_PS_NET_RE = re.compile(
    r'id="snapshot-netExpenseRatio"[^>]*>\s*(\d{1,2}\.\d{1,3})\s*%',
    re.IGNORECASE,
)
_PS_GROSS_RE = re.compile(
    r'id="snapshot-grossExpenseRatio"[^>]*>\s*(\d{1,2}\.\d{1,3})\s*%',
    re.IGNORECASE,
)


def _scrape_proshares(session, ticker: str) -> float | None:
    for url in (
        f"https://www.proshares.com/our-etfs/leveraged-and-inverse/{ticker.lower()}",
        f"https://www.proshares.com/our-etfs/strategic/{ticker.lower()}",
        f"https://www.proshares.com/our-etfs/{ticker.lower()}",
    ):
        res = _http_get(session, url)
        if not res:
            continue
        html, _ = res
        m = _PS_NET_RE.search(html) or _PS_GROSS_RE.search(html)
        if m:
            v = float(m.group(1)) / 100.0
            if _plausible(v):
                return round(v, 6)
    return None


# REX Shares:  /<ticker>/  — "Expense Ratio" label nearby
def _scrape_rex_shares(session, ticker: str) -> float | None:
    for url in (
        f"https://www.rexshares.com/{ticker.lower()}/",
        f"https://rexshares.com/{ticker.lower()}/",
    ):
        res = _http_get(session, url)
        if not res:
            continue
        html, final_url = res
        if ticker.lower() not in final_url.lower():
            continue
        m = re.search(
            r"Expense\s+Ratio[^%\d]{0,200}?(\d{1,2}\.\d{1,3})\s*%",
            html, re.IGNORECASE | re.DOTALL,
        )
        if m:
            v = float(m.group(1)) / 100.0
            if _plausible(v):
                return round(v, 6)
    return None


# Roundhill:  /etf/<ticker>/
def _scrape_roundhill(session, ticker: str) -> float | None:
    url = f"https://www.roundhillinvestments.com/etf/{ticker.lower()}/"
    res = _http_get(session, url)
    if not res:
        return None
    html, final_url = res
    if ticker.lower() not in final_url.lower():
        return None
    m = re.search(
        r"Expense\s+Ratio[^%\d]{0,200}?(\d{1,2}\.\d{1,3})\s*%",
        html, re.IGNORECASE | re.DOTALL,
    )
    if m:
        v = float(m.group(1)) / 100.0
        if _plausible(v):
            return round(v, 6)
    return None


# TRADR (AXS):  https://www.tradretfs.com/<ticker>
def _scrape_tradr(session, ticker: str) -> float | None:
    url = f"https://www.tradretfs.com/{ticker.lower()}"
    res = _http_get(session, url)
    if not res:
        return None
    html, final_url = res
    if ticker.lower() not in final_url.lower():
        return None
    m = re.search(
        r"Expense\s+Ratio[^%\d]{0,150}?(\d{1,2}\.\d{1,3})\s*%",
        html, re.IGNORECASE | re.DOTALL,
    )
    if m:
        v = float(m.group(1)) / 100.0
        if _plausible(v):
            return round(v, 6)
    return None


# ──────────────────────────────────────────────
# Issuer routing
# ──────────────────────────────────────────────
# Heuristic: each ticker is tried against the issuer in this list in order.
# First non-None win short-circuits the search. If none return a value we
# fall through to yfinance. Issuer lists are *suggestions*, not whitelists;
# the scrapers themselves will reject a mismatched redirect.

ISSUER_FETCHERS: dict[str, Callable[[requests.Session, str], Optional[float]]] = {
    "yieldmax":   _scrape_yieldmax,
    "proshares":  _scrape_proshares,
    "rex_shares": _scrape_rex_shares,
    "roundhill":  _scrape_roundhill,
    "tradr":      _scrape_tradr,
}

TICKER_ISSUER_HINTS: dict[str, str] = {}

# YieldMax tickers (suffix 'Y'/'YY' and a few fixed names)
for _t in (
    "TSLY NVDY APLY AMDY MSFO GOOY AMZY FBY MRNY NFLY CONY COIY YBIT YMAG YMAX"
    " ULTY SMCY CRSH FIAT GDXY SDTY QDTY RDTY JPMO XOMO TSLP MSTY PLTY ABNY CVNY"
    " PYPY LFGY FIVY MARO SOXY BIGY AIYY DIPS BABO DISO BITO"
).split():
    TICKER_ISSUER_HINTS[_t] = "yieldmax"

# ProShares (partial list — covers the major leveraged/inverse names in the universe)
for _t in (
    "SSO QLD DDM MVV SAA UWM UVG UKF UVU UKW UVT UKK UYM UGE UCC UYG RXL UXI DIG"
    " URE ROM UPW SDS QID DXD MZZ SDD TWM SJF SFK SJL SDK SJH SKK SMN SZK SCC SKF"
    " RXD SIJ DUG SRS REW SDP BITU ETHT UXRP UVXY SPXU SPXL SCO DRIP TBT TBX TMV"
    " MSTZ NVDQ BTCZ ETHD CRCD CONI MSDD NVD TSDD CORD TSLQ ZSL SQQQ BITO"
).split():
    TICKER_ISSUER_HINTS.setdefault(_t, "proshares")

# REX Shares
for _t in (
    "FEPI AIPI MSTU NVDX TSLT BTCL ETU CCUP CRWU AAPX GOOX MSFX NFLU ROBN ARMU"
    " DJTU RBLU SNOU SMUP DKUP BULU GLXU AFRU AXUP KTUP TTDU BKNU PXIU BMNU SBTU"
    " CIFU SOLX XRPK XRPT APLZ AMZO BEZ NBIZ SMZ QBTZ IONZ CLSZ IREZ RGTZ PLTZ"
    " SMST SMCZ BMNZ HOOZ DAMD RKLZ STSM OKLS ASTN"
).split():
    TICKER_ISSUER_HINTS.setdefault(_t, "rex_shares")

# Roundhill
for _t in "MAGX MAGS YBTC XDTE QDTE RDTE YETH AAPW NVW METW TSLW MSTW".split():
    TICKER_ISSUER_HINTS.setdefault(_t, "roundhill")

# TRADR (AXS) — we don't have a static list; rely on the scraper short-circuit.


# ──────────────────────────────────────────────
# Manual YAML override
# ──────────────────────────────────────────────
_MANUAL_CACHE: dict[str, float] | None = None
_MANUAL_PATH = Path(__file__).resolve().parent / "config" / "etf_expense_ratios.yml"


def _load_manual_overrides() -> dict[str, float]:
    global _MANUAL_CACHE
    if _MANUAL_CACHE is not None:
        return _MANUAL_CACHE
    _MANUAL_CACHE = {}
    if not _MANUAL_PATH.exists():
        return _MANUAL_CACHE
    try:
        import yaml  # optional dep — PyYAML
    except ImportError:
        LOGGER.warning("PyYAML not installed; manual overrides disabled")
        return _MANUAL_CACHE
    try:
        with _MANUAL_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        LOGGER.warning("Failed to read %s: %s", _MANUAL_PATH, e)
        return _MANUAL_CACHE
    # Expected structure:
    #   overrides:
    #     NVDL: 0.0115
    #     TSLR: 0.0115
    # Accept values as decimal (0.0115) OR percent (1.15)
    src = data.get("overrides") if isinstance(data, dict) else None
    if not isinstance(src, dict):
        return _MANUAL_CACHE
    for k, v in src.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0.15:  # user wrote "1.15" meaning 1.15 %
            f = f / 100.0
        if _plausible(f):
            _MANUAL_CACHE[str(k).strip().upper()] = round(f, 6)
    return _MANUAL_CACHE


# ──────────────────────────────────────────────
# yfinance fallback (rate-limited)
# ──────────────────────────────────────────────
def _yfinance_ratio(ticker: str) -> float | None:
    global _YF_LAST_CALL_AT
    # Throttle: Yahoo's quoteSummary endpoint 429s aggressively under load.
    with _YF_LOCK:
        wait = _YF_MIN_INTERVAL_SEC - (time.monotonic() - _YF_LAST_CALL_AT)
        if wait > 0:
            time.sleep(wait)
        _YF_LAST_CALL_AT = time.monotonic()
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception as e:
        LOGGER.debug("yfinance info failed for %s: %s", ticker, e)
        return None
    if not isinstance(info, dict):
        return None
    for key in ("annualReportExpenseRatio", "netExpenseRatio", "expenseRatio"):
        v = info.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0.15:     # yfinance sometimes returns percent (0.99) not decimal
            f = f / 100.0
        if _plausible(f):
            return round(f, 6)
    return None


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def get_expense_ratio(
    ticker: str,
    session: requests.Session | None = None,
) -> tuple[Optional[float], str]:
    """Return (ratio_decimal_or_None, source) for one ticker. Never raises."""
    t = str(ticker).strip().upper().replace(".", "-")
    if not t:
        return None, "missing"

    overrides = _load_manual_overrides()
    if t in overrides:
        return overrides[t], "manual"

    sess = session or _build_session()
    issuer = TICKER_ISSUER_HINTS.get(t)
    if issuer:
        fn = ISSUER_FETCHERS.get(issuer)
        if fn is not None:
            try:
                v = fn(sess, t)
            except Exception as e:
                LOGGER.debug("%s scraper raised for %s: %s", issuer, t, e)
                v = None
            if v is not None:
                return v, issuer

    # TRADR is a catch-all for single-stock LETFs whose issuer we don't know
    # statically. Its scraper redirect-guards so a miss is safe.
    if issuer != "tradr":
        try:
            v = _scrape_tradr(sess, t)
        except Exception:
            v = None
        if v is not None:
            return v, "tradr"

    v = _yfinance_ratio(t)
    if v is not None:
        return v, "yfinance"
    return None, "missing"


def fetch_expense_ratios(
    tickers: list[str],
    *,
    max_workers: int = 4,
    progress_every: int = 50,
) -> dict[str, tuple[Optional[float], str]]:
    """Parallel fetch. Returns {ticker: (ratio_or_None, source)}.

    ``max_workers`` is deliberately small: issuer sites are fine with 4–8
    concurrent requests, but the yfinance quoteSummary endpoint starts 429ing
    above ~4 parallel callers. The module's internal lock further serializes
    yfinance calls with a ~250 ms gap.
    """
    unique = sorted({str(t).strip().upper().replace(".", "-") for t in tickers if t})
    if not unique:
        return {}

    _load_manual_overrides()  # warm cache once
    print(
        f"[expense_ratios] Fetching {len(unique)} tickers "
        f"(manual -> issuer -> yfinance, threads={max_workers})"
    )
    t0 = time.monotonic()
    out: dict[str, tuple[Optional[float], str]] = {}
    session = _build_session()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(get_expense_ratio, t, session): t for t in unique}
        done = 0
        for fut in as_completed(futs):
            tkr = futs[fut]
            try:
                out[tkr] = fut.result()
            except Exception as e:
                LOGGER.debug("get_expense_ratio(%s) raised: %s", tkr, e)
                out[tkr] = (None, "missing")
            done += 1
            if progress_every and done % progress_every == 0:
                elapsed = time.monotonic() - t0
                hits = sum(1 for v, _ in out.values() if v is not None)
                print(f"  ... {done}/{len(unique)}  hits={hits}  [{elapsed:.1f}s]")

    elapsed = time.monotonic() - t0
    by_src: dict[str, int] = {}
    for _, src in out.values():
        by_src[src] = by_src.get(src, 0) + 1
    print(
        f"[expense_ratios] Done in {elapsed:.1f}s — sources: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_src.items()))
    )
    return out


__all__ = [
    "ISSUER_FETCHERS",
    "TICKER_ISSUER_HINTS",
    "fetch_expense_ratios",
    "get_expense_ratio",
]
