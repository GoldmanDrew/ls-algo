"""
etf_analytics.py — Drop-in decay + volatility enrichment for etf_screener.py

Adds columns to the screened DataFrame:
  - vol_underlying_annual  : annualized realized vol of the underlying (total return)
  - vol_etf_annual         : annualized realized vol of the ETF (total return)
  - gross_decay_annual     : annualized gross decay of the pair trade (before borrow)

Decay measurement:
  On each day with $1 constant notional:
    Long  $1 of the underlying
    Short $1/|Beta| of the ETF

  daily_pnl = r_underlying − (1/|Beta|) × r_etf

  gross_decay_annual = mean(daily_pnl) × 252

  This is a SIMPLE (linear) annualized rate — same units as the borrow
  rate, which is also a simple annual rate (accrued daily as rate/360).

  Net spread = gross_decay_annual − borrow_net_annual
  Both are simple annual rates → subtraction is valid.

  For a perfect Beta× daily tracker:
    r_etf = Beta × r_und
    daily_pnl = r_und − (1/Beta)(Beta × r_und) = 0
  Vol drag / fees / tracking error → daily_pnl > 0 on average → positive decay.

Both legs use explicit total-return price series:
  TR_t = TR_{t-1} × (Close_t + Div_t) / Close_{t-1}
so dividends are correctly captured on both sides.

Usage in etf_screener.py:
    from etf_analytics import enrich_with_decay_and_vol

    df = enrich_with_decay_and_vol(df)

    df.to_csv(OUTPUT_FILE, index=False)
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


# ──────────────────────────────────────────────
# Total return price series (per-ticker)
# ──────────────────────────────────────────────
def _get_total_return_series(ticker: str, period: str = "2y") -> pd.Series:
    """
    Build a long-only total return price series for one ticker.

    Uses unadjusted close + explicit dividends:
        TR_t = TR_{t-1} × (Close_t + Div_t) / Close_{t-1}

    This correctly captures:
      - Cash dividends (including large ETF distributions)
      - Stock splits (handled by Yahoo's unadjusted close)
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=False, actions=True)

        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float, name=ticker)

        close = df["Close"]
        divs = df.get("Dividends", pd.Series(0.0, index=df.index))
        divs = divs.reindex(close.index, fill_value=0.0)

        rel = (close + divs) / close.shift(1)
        rel.iloc[0] = 1.0

        tr_price = close.iloc[0] * rel.cumprod()
        tr_price.name = ticker
        return tr_price

    except Exception:
        return pd.Series(dtype=float, name=ticker)


def _download_all_tr_series(
    tickers: list[str],
    period: str = "2y",
    max_workers: int = 8,
) -> dict[str, pd.Series]:
    """Download total-return series for all tickers in parallel."""
    print(f"[etf_analytics] Downloading {len(tickers)} total-return series "
          f"(period={period}, threads={max_workers}) ...")
    t0 = time.monotonic()

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_get_total_return_series, ticker, period): ticker
            for ticker in tickers
        }
        done = 0
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                series = future.result()
                if not series.empty:
                    results[ticker.upper().replace(".", "-")] = series
            except Exception:
                pass
            done += 1
            if done % 50 == 0:
                print(f"  ... {done}/{len(tickers)}")

    elapsed = time.monotonic() - t0
    print(f"[etf_analytics] Got {len(results)}/{len(tickers)} tickers [{elapsed:.1f}s]")

    missing = set(t.upper().replace(".", "-") for t in tickers) - set(results.keys())
    if missing:
        print(f"[etf_analytics] Missing: {sorted(missing)[:15]}"
              f"{'...' if len(missing) > 15 else ''}")

    return results


# ──────────────────────────────────────────────
# Annualized volatility
# ──────────────────────────────────────────────
def _annualized_vol(tr_series: pd.Series, min_days: int = 60) -> float | None:
    """Annualized realized vol from a total-return price series."""
    tr = tr_series.dropna()
    if len(tr) < min_days + 1:
        return None
    ret = tr.pct_change().dropna()
    if len(ret) < min_days:
        return None
    return round(float(ret.std() * np.sqrt(TRADING_DAYS)), 6)


# ──────────────────────────────────────────────
# Gross decay — simple (linear) annualized rate
# ──────────────────────────────────────────────
def _compute_gross_decay(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_days: int = 60,
) -> float | None:
    """
    Gross annualized decay for one ETF/underlying pair.
    Returns a SIMPLE (linear) annual rate, same units as borrow.

    On each day with $1 constant notional:
      Long  $1 underlying          → earns r_und
      Short $1/|Beta| of ETF       → earns −(1/|Beta|) × r_etf

      daily_pnl = r_und − (1/|Beta|) × r_etf

    gross_decay_annual = mean(daily_pnl) × 252

    No compounding — positions reset to $1 notional each day.
    This is directly comparable to the borrow rate, which is also
    a simple annual rate accrued daily (rate/360 per day).

    Positive = ETF underperforms the hedge → profitable to short.
    """
    # Align on common dates
    df = pd.concat([etf_tr.rename("etf"), und_tr.rename("und")], axis=1).dropna()
    if len(df) < min_days + 1:
        return None

    # Daily total returns (dividends included via TR series)
    r_etf = df["etf"].pct_change()
    r_und = df["und"].pct_change()

    # Drop first NaN and align
    valid = r_etf.notna() & r_und.notna()
    r_etf = r_etf[valid]
    r_und = r_und[valid]

    if len(r_etf) < min_days:
        return None

    abs_beta = abs(float(beta))
    if abs_beta < 0.1:
        return None

    # Daily hedged P&L on constant $1 notional
    daily_pnl = r_und - (1.0 / abs_beta) * r_etf

    # Simple (linear) annualized rate
    gross_decay = float(daily_pnl.mean()) * TRADING_DAYS

    return round(gross_decay, 6)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def enrich_with_decay_and_vol(
    df: pd.DataFrame,
    lookback: str = "2y",
    min_days: int = 60,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Add vol + decay columns to the screened DataFrame.

    New columns:
      vol_underlying_annual  - annualized realized vol of underlying (total return)
      vol_etf_annual         - annualized realized vol of ETF (total return)
      gross_decay_annual     - simple annualized decay (1/|Beta| hedge,
                               constant notional, no compounding)
                               Subtract borrow_net_annual for net spread.

    Args:
        df:          Screened DataFrame (needs ETF, Underlying, Beta)
        lookback:    yfinance period string ('1y', '2y', etc.)
        min_days:    Minimum overlapping trading days required
        max_workers: Thread pool size for parallel downloads

    Returns:
        Same DataFrame with new columns added.
    """
    print("=" * 60)
    print("[etf_analytics] Computing decay + volatility (total return)")
    print("=" * 60)

    norm = lambda s: str(s).strip().upper().replace(".", "-")

    etf_syms = df["ETF"].apply(norm).tolist()
    und_syms = df["Underlying"].dropna().apply(norm).tolist()
    all_tickers = sorted(set(etf_syms + und_syms))

    tr_map = _download_all_tr_series(all_tickers, period=lookback,
                                      max_workers=max_workers)

    vols_und = []
    vols_etf = []
    decays = []

    ok_decay = 0
    ok_vol = 0

    for _, row in df.iterrows():
        etf = norm(row["ETF"])
        und = norm(row["Underlying"]) if pd.notna(row.get("Underlying")) else None
        beta = row.get("Beta")
        beta_f = float(beta) if pd.notna(beta) else None

        # ── Volatilities ──
        vol_und = None
        vol_etf = None
        if und and und in tr_map:
            vol_und = _annualized_vol(tr_map[und], min_days)
        if etf in tr_map:
            vol_etf = _annualized_vol(tr_map[etf], min_days)

        vols_und.append(vol_und)
        vols_etf.append(vol_etf)
        if vol_und is not None:
            ok_vol += 1

        # ── Gross decay ──
        decay = None
        if (
            beta_f is not None
            and abs(beta_f) >= 0.1
            and und
            and etf in tr_map
            and und in tr_map
        ):
            decay = _compute_gross_decay(
                tr_map[etf],
                tr_map[und],
                beta_f,
                min_days,
            )

        decays.append(decay)
        if decay is not None:
            ok_decay += 1

    # Add columns
    df["vol_underlying_annual"] = vols_und
    df["vol_etf_annual"] = vols_etf
    df["gross_decay_annual"] = decays

    # ── Summary ──
    print(f"\n[etf_analytics] Volatility computed: {ok_vol}/{len(df)}")
    print(f"[etf_analytics] Decay computed:      {ok_decay}/{len(df)}")

    valid_decays = df["gross_decay_annual"].dropna()
    if len(valid_decays) > 0:
        print(f"\n[etf_analytics] Decay stats:")
        print(f"  Range:  {valid_decays.min()*100:.2f}% -- {valid_decays.max()*100:.2f}%")
        print(f"  Median: {valid_decays.median()*100:.2f}%")

    valid_vols = [v for v in vols_und if v is not None]
    if valid_vols:
        print(f"\n[etf_analytics] Underlying vol range: "
              f"{min(valid_vols)*100:.1f}% -- {max(valid_vols)*100:.1f}%")

    # Top 5 by decay
    temp = df[df["gross_decay_annual"].notna()].nlargest(5, "gross_decay_annual")
    if len(temp) > 0:
        print(f"\n[etf_analytics] Top 5 gross decay:")
        for _, r in temp.iterrows():
            bn = r.get("borrow_net_annual")
            sp = round(r["gross_decay_annual"] - bn, 4) if pd.notna(bn) else None
            print(
                f"  {r['ETF']:8s} decay={r['gross_decay_annual']*100:6.2f}%  "
                f"vol_und={r['vol_underlying_annual']*100:5.1f}%  "
                f"vol_etf={r['vol_etf_annual']*100:5.1f}%  "
                f"B={r['Beta']:.2f}  "
                f"borrow={bn*100 if pd.notna(bn) else 0:5.2f}%  "
                f"spread={sp*100 if sp is not None else 0:6.2f}%"
            )

    print("=" * 60)
    return df
