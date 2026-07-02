"""
Bucket 5 ("positive-carry tail risk") data loaders.

This module provides the market data the Bucket 5 short-UVIX / short-SVIX carry
engine and (later) the SPX put-roll overlay need:

  * ``load_vol_panel`` -- daily UVIX / SVIX closes plus the VIX/VIX3M term-
    structure ratio. By default extends pre-2022 history via synthetic
    simulation from ^SHORTVOL (see ``bucket5_synthetic.py``).

  * ``load_spx_put_eod`` / ``ThetaSpxClient`` -- legacy REST client for a
    locally running ThetaTerminal v2 (http://127.0.0.1:25510). Prefer
    ``bucket5_theta.py`` (official ``thetadata`` gRPC client + API key).

Price series are pulled with yfinance (``auto_adjust=True`` so reverse splits in
UVIX are stitched into a continuous series) and cached to
``data/cache/bucket5``. Existing ``data/cache/beta_history`` CSVs (which already
hold SVIX, ^VIX, ^VIX3M) are used as a read fallback.
"""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # optional at import time so the module can be inspected without it
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


# ---------------------------------------------------------------------------
# Tickers / constants
# ---------------------------------------------------------------------------
UVIX = "UVIX"          # +2x daily Cboe LONGVOL (long short-term VIX futures)
SVIX = "SVIX"          # -1x daily Cboe SHORTVOL (short short-term VIX futures)
VIX = "^VIX"           # 30-day implied vol
VIX3M = "^VIX3M"       # 3-month implied vol

# UVIX/SVIX first live trade date; synthetic history extends to 2005-12-20.
INCEPTION = "2022-03-30"
SYNTH_START = "2005-12-20"

# Borrow defaults sourced from data/borrow_cache.csv (annual fee rate / 100).
# UVIX ~2.84%/yr (feerate 2.8428, >10M shares available), SVIX ~3.47%/yr.
# These are calm-regime point values -- widen them in stress (see
# risk_dashboard.vix_scenario.borrow_rate_vix_stress).
DEFAULT_BORROW_UVIX = 0.0284
DEFAULT_BORROW_SVIX = 0.0347

CACHE_DIR = Path("data/cache/bucket5")
FALLBACK_CACHE_DIRS = (Path("data/cache/beta_history"),)


# ---------------------------------------------------------------------------
# Price loading (cached yfinance)
# ---------------------------------------------------------------------------
def _read_cache(ticker: str) -> pd.Series | None:
    for base in (CACHE_DIR, *FALLBACK_CACHE_DIRS):
        p = base / f"{ticker}.csv"
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
        except Exception:
            continue
        cols = {str(c).lower(): c for c in df.columns}
        col = cols.get("close") or (df.columns[0] if len(df.columns) == 1 else None)
        if col is None:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        idx = pd.to_datetime(s.index, utc=True, errors="coerce")
        s.index = idx.tz_convert(None)
        s = s[~s.index.isna()].sort_index()
        if len(s):
            return s.rename(ticker)
    return None


def _write_cache(ticker: str, s: pd.Series) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = s.rename("close").to_frame()
    out.index.name = "date"
    out.to_csv(CACHE_DIR / f"{ticker}.csv")


def _download(ticker: str, start: str, end: str | None, *, retries: int = 3) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is not installed; cannot download " + ticker)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(
                start=start, end=end, auto_adjust=True, repair=True
            )
            if df is not None and len(df) and "Close" in df.columns:
                s = df["Close"].dropna()
                s.index = pd.to_datetime(s.index, utc=True).tz_convert(None)
                return s.rename(ticker)
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"No yfinance data for {ticker} (last error: {last_err})")


def _to_bday_index(s: pd.Series) -> pd.Series:
    """Normalize timestamps to calendar dates so mixed tz/clock times align."""
    s = s.copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s.groupby(level=0).last().sort_index()


def load_series(
    ticker: str,
    start: str = INCEPTION,
    end: str | None = None,
    *,
    refresh: bool = False,
) -> pd.Series:
    """Adjusted close for ``ticker``; cache-first unless ``refresh``."""
    if not refresh:
        cached = _read_cache(ticker)
        if cached is not None:
            s = cached.loc[cached.index >= pd.Timestamp(start)]
            if end:
                s = s.loc[: pd.Timestamp(end)]
            if len(s):
                covers_start = cached.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=5)
                if end:
                    # Cache must span up to the requested end.
                    fresh_enough = covers_start and s.index.max() >= pd.Timestamp(end)
                else:
                    # Open-ended: only trust cache if it is within ~5 days of
                    # today, else refetch so a stale CSV can't truncate the panel.
                    age = pd.Timestamp.today().normalize() - s.index.max().normalize()
                    fresh_enough = covers_start and age <= pd.Timedelta(days=5)
                if fresh_enough:
                    return _to_bday_index(s)
    try:
        s = _download(ticker, start, end)
    except RuntimeError:
        # Network unavailable -> fall back to whatever the cache holds.
        cached = _read_cache(ticker)
        if cached is not None and len(cached):
            s = cached.loc[cached.index >= pd.Timestamp(start)]
            if end:
                s = s.loc[: pd.Timestamp(end)]
            if len(s):
                return _to_bday_index(s)
        raise
    _write_cache(ticker, s)
    return _to_bday_index(s.loc[s.index >= pd.Timestamp(start)])


def load_vol_panel(
    start: str = SYNTH_START,
    end: str | None = None,
    *,
    refresh: bool = False,
    use_synthetic: bool = True,
    ratio_clip: tuple[float, float] = (0.6, 1.4),
) -> pd.DataFrame:
    """Daily UVIX/SVIX + VIX term-structure ratio.

    Returns a DataFrame indexed by date with columns:
      ``uvix``, ``svix``    -- adjusted closes (synthetic pre-2022 when enabled)
      ``vix``, ``vix3m``    -- implied vol levels
      ``ratio``             -- VIX / VIX3M (regime signal)
      ``synthetic``         -- True for pre-splice simulated UVIX/SVIX rows

    When ``use_synthetic=True`` and ``start < INCEPTION``, UVIX/SVIX are built
    from ^SHORTVOL via ``bucket5_synthetic.build_synthetic_prices`` and spliced
    to live yfinance data on 2022-03-30.
    """
    try:
        from scripts.bucket5_synthetic import load_extended_uvix_svix
    except ImportError:
        from bucket5_synthetic import load_extended_uvix_svix  # type: ignore

    etp = load_extended_uvix_svix(start, end, use_synthetic=use_synthetic)
    vix_start = start if not use_synthetic else min(start, SYNTH_START)
    vix = _to_bday_index(load_series(VIX, vix_start, end, refresh=refresh)).rename("vix")
    vix3m = _to_bday_index(load_series(VIX3M, vix_start, end, refresh=refresh)).rename("vix3m")

    panel = etp.join([vix.reindex(etp.index).ffill(), vix3m.reindex(etp.index).ffill()])
    panel["ratio"] = (panel["vix"] / panel["vix3m"]).clip(*ratio_clip)
    if "synthetic" not in panel.columns:
        panel["synthetic"] = panel.index < pd.Timestamp(INCEPTION)
    return panel.dropna(subset=["uvix", "svix", "vix", "vix3m", "ratio"])


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DatetimeIndex:
    """Trading days on/closest-before each calendar anchor in ``freq``.

    e.g. ``freq='W-FRI'`` -> the last trading day on/of each week. Always
    includes the first day so the book is established on day 1.
    """
    if len(index) == 0:
        return pd.DatetimeIndex([])
    anchors = pd.date_range(index.min(), index.max(), freq=freq)
    pos = index.searchsorted(anchors, side="right") - 1
    pos = np.unique(pos[(pos >= 0) & (pos < len(index))])
    out = index[pos]
    return out.union(index[:1])


# ---------------------------------------------------------------------------
# ThetaData SPX option loader (for the 6M->3M put-roll overlay)
# ---------------------------------------------------------------------------
SPX_OPT_CACHE = Path("data/cache/spx_options")


class ThetaSpxClient:
    """Minimal ThetaData Terminal REST client for SPX option EOD data.

    Requires a running ThetaTerminal (https://www.thetadata.net). Strikes in the
    ThetaData API are integers in 1/1000 dollars (e.g. 4000.0 -> 4000000); this
    client handles the conversion. Responses are cached as parquet under
    ``data/cache/spx_options`` keyed by (root, exp, strike, right).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:25510", root: str = "SPX"):
        self.base_url = base_url.rstrip("/")
        self.root = root

    @staticmethod
    def _ymd(d) -> int:
        return int(pd.Timestamp(d).strftime("%Y%m%d"))

    def _cache_file(self, exp: int, strike: float, right: str) -> Path:
        return SPX_OPT_CACHE / f"{self.root}_{exp}_{int(round(strike*1000))}_{right}.parquet"

    def list_expirations(self) -> list[int]:
        df = self._get("/v2/list/expirations", {"root": self.root})
        return sorted(int(x[0]) for x in df)

    def _get(self, path: str, params: dict):
        if requests is None:
            raise RuntimeError("requests is not installed; cannot reach ThetaTerminal")
        r = requests.get(self.base_url + path, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        return payload.get("response", [])

    def eod_put(
        self,
        exp,
        strike: float,
        start_date,
        end_date,
        right: str = "P",
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """EOD quote/greeks for one SPX option contract over a date range.

        Returns a DataFrame indexed by date with at least ``mid`` (and bid/ask,
        and greeks when ThetaData returns them). Cached per contract.
        """
        exp_i = self._ymd(exp)
        cache = self._cache_file(exp_i, strike, right)
        if cache.is_file() and not refresh:
            df = pd.read_parquet(cache)
        else:
            rows = self._get(
                "/v2/hist/option/eod",
                {
                    "root": self.root,
                    "exp": exp_i,
                    "strike": int(round(strike * 1000)),
                    "right": right,
                    "start_date": self._ymd(start_date),
                    "end_date": self._ymd(end_date),
                },
            )
            df = self._eod_to_frame(rows)
            SPX_OPT_CACHE.mkdir(parents=True, exist_ok=True)
            if not df.empty:
                df.to_parquet(cache)
        if df.empty:
            return df
        m = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        return df.loc[m]

    @staticmethod
    def _eod_to_frame(rows: list) -> pd.DataFrame:
        # ThetaData EOD columns (v2): ms_of_day, ms_of_day2, open, high, low,
        # close, volume, count, bid_size, bid_exchange, bid, bid_condition,
        # ask_size, ask_exchange, ask, ask_condition, date
        if not rows:
            return pd.DataFrame()
        cols = [
            "ms_of_day", "ms_of_day2", "open", "high", "low", "close", "volume",
            "count", "bid_size", "bid_exchange", "bid", "bid_condition",
            "ask_size", "ask_exchange", "ask", "ask_condition", "date",
        ]
        df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
        df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
        df = df.set_index("date").sort_index()
        bid = pd.to_numeric(df.get("bid"), errors="coerce")
        ask = pd.to_numeric(df.get("ask"), errors="coerce")
        df["mid"] = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0,
                             pd.to_numeric(df.get("close"), errors="coerce"))
        return df[["bid", "ask", "mid", "close", "volume"]]


def load_spx_put_eod(
    exp,
    strike: float,
    start_date,
    end_date,
    base_url: str = "http://127.0.0.1:25510",
    root: str = "SPX",
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper: EOD series for one SPX put contract via ThetaData."""
    return ThetaSpxClient(base_url, root).eod_put(
        exp, strike, start_date, end_date, right="P", refresh=refresh
    )


if __name__ == "__main__":  # pragma: no cover - smoke check
    panel = load_vol_panel()
    print(f"panel: {panel.index.min().date()} -> {panel.index.max().date()} "
          f"({len(panel)} rows)")
    print(panel.tail(3).round(3).to_string())
    print(f"ratio: min={panel['ratio'].min():.3f} "
          f"median={panel['ratio'].median():.3f} max={panel['ratio'].max():.3f}")
