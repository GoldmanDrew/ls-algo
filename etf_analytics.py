"""
etf_analytics.py — Drop-in decay + volatility enrichment for etf_screener.py

Adds columns to the screened DataFrame:
  - vol_underlying_annual  : annualized realized vol of the underlying (total return)
  - vol_etf_annual         : annualized realized vol of the ETF (total return)
  - gross_decay_annual     : annualized gross decay of the pair trade (before borrow)
  - borrow_drag_annual     : borrow cost scaled to hedge notional = borrow_net / |Beta|
  - net_decay_annual       : gross_decay_annual − borrow_drag_annual

Decay measurement (works for BOTH bull and inverse ETFs):
  On each day with $1 constant notional:
    sign(Beta) × $1 of the underlying   (long for bull ETFs, short for inverse)
    Short      $1/|Beta| of the ETF

  daily_pnl = sign(Beta) × r_underlying  −  (1/|Beta|) × r_etf

  gross_decay_annual = mean(daily_pnl) × 252

  This is a SIMPLE (linear) annualized rate — same units as the borrow
  rate, which is also a simple annual rate (accrued daily as rate/360).

  For a perfect Beta× daily tracker (bull or inverse):
    r_etf = Beta × r_und
    daily_pnl = sign(B)×r_und − (1/|B|)(B×r_und)
              = sign(B)×r_und − sign(B)×r_und = 0
  Vol drag / fees / tracking error → daily_pnl > 0 on average → positive decay.

Both legs use explicit total-return price series so dividends are captured.

If Beta is missing (NaN) for a row, it is computed via OLS regression
of ETF returns on underlying returns.

Usage in etf_screener.py:
    from etf_analytics import enrich_with_decay_and_vol

    df = enrich_with_decay_and_vol(df)   # adds all analytics columns
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
# OLS beta from total-return series
# ──────────────────────────────────────────────
def _compute_beta_ols(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    min_days: int = 60,
) -> tuple[float | None, int]:
    """
    OLS regression:  r_etf = alpha + beta × r_und + eps

    Returns (beta, n_obs).  beta is None if insufficient data.
    """
    df = pd.concat([etf_tr.rename("etf"), und_tr.rename("und")], axis=1).dropna()
    if len(df) < min_days + 1:
        return None, 0

    r_etf = df["etf"].pct_change().dropna()
    r_und = df["und"].pct_change().dropna()

    valid = r_etf.index.intersection(r_und.index)
    r_etf = r_etf.loc[valid]
    r_und = r_und.loc[valid]

    if len(r_etf) < min_days:
        return None, 0

    # OLS: beta = cov(r_etf, r_und) / var(r_und)
    cov = np.cov(r_etf.values, r_und.values)
    var_und = cov[1, 1]
    if var_und < 1e-12:
        return None, 0

    beta = float(cov[0, 1] / var_und)
    return round(beta, 6), len(r_etf)


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

    Works for BOTH bull (beta > 0) and inverse (beta < 0) ETFs.

    On each day with $1 constant notional:
      sign(beta) × $1 underlying    → long for bull, short for inverse
      Short $1/|beta| of ETF        → always short the ETF

      daily_pnl = sign(beta) × r_und  −  (1/|beta|) × r_etf

    No compounding — positions reset to $1 notional each day.

    Positive = ETF underperforms the hedge → profitable to short.
    """
    df = pd.concat([etf_tr.rename("etf"), und_tr.rename("und")], axis=1).dropna()
    if len(df) < min_days + 1:
        return None

    r_etf = df["etf"].pct_change()
    r_und = df["und"].pct_change()

    valid = r_etf.notna() & r_und.notna()
    r_etf = r_etf[valid]
    r_und = r_und[valid]

    if len(r_etf) < min_days:
        return None

    abs_beta = abs(float(beta))
    if abs_beta < 0.1:
        return None

    # sign(beta): +1 for bull ETFs (long underlying), -1 for inverse (short underlying)
    beta_sign = 1.0 if beta > 0 else -1.0

    # Daily hedged P&L on constant $1 notional
    daily_pnl = beta_sign * r_und - (1.0 / abs_beta) * r_etf

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

    New / updated columns:
      Beta                  - computed via OLS when missing (NaN)
      Beta_n_obs            - observation count for OLS beta
      vol_underlying_annual - annualized realized vol of underlying (total return)
      vol_etf_annual        - annualized realized vol of ETF (total return)
      gross_decay_annual    - simple annualized decay (1/|Beta| hedge,
                              constant notional, sign-corrected for inverse ETFs)
      borrow_drag_annual    - borrow cost on hedge notional = borrow_net / |Beta|
      net_decay_annual      - gross_decay − borrow_drag

    Args:
        df:          Screened DataFrame (needs ETF, Underlying; Beta optional)
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
    betas_out = []
    beta_nobs_out = []

    ok_decay = 0
    ok_vol = 0
    betas_computed = 0

    for _, row in df.iterrows():
        etf = norm(row["ETF"])
        und = norm(row["Underlying"]) if pd.notna(row.get("Underlying")) else None

        # ── Beta: use existing or compute from OLS ──
        beta = row.get("Beta")
        beta_f = float(beta) if pd.notna(beta) else None
        n_obs = row.get("Beta_n_obs")
        n_obs_i = int(n_obs) if pd.notna(n_obs) else 0

        if beta_f is None and und and etf in tr_map and und in tr_map:
            beta_f, n_obs_i = _compute_beta_ols(tr_map[etf], tr_map[und], min_days)
            if beta_f is not None:
                betas_computed += 1

        betas_out.append(beta_f)
        beta_nobs_out.append(n_obs_i)

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

    # ── Write columns ──
    df["Beta"] = betas_out
    df["Beta_n_obs"] = beta_nobs_out
    df["vol_underlying_annual"] = vols_und
    df["vol_etf_annual"] = vols_etf
    df["gross_decay_annual"] = decays

    # ── Borrow drag & net decay ──
    beta_abs = df["Beta"].abs()
    borrow_net = pd.to_numeric(df.get("borrow_net_annual"), errors="coerce")

    # borrow_drag = borrow_net / |Beta|  (cost on the 1/|Beta| short ETF leg)
    df["borrow_drag_annual"] = np.where(
        beta_abs.notna() & (beta_abs > 0.1) & borrow_net.notna(),
        borrow_net / beta_abs,
        np.nan,
    )

    # net_decay = gross_decay − borrow_drag
    df["net_decay_annual"] = np.where(
        df["gross_decay_annual"].notna() & df["borrow_drag_annual"].notna(),
        df["gross_decay_annual"] - df["borrow_drag_annual"],
        np.nan,
    )

    # ── Summary ──
    print(f"\n[etf_analytics] Beta computed from OLS: {betas_computed}")
    print(f"[etf_analytics] Volatility computed: {ok_vol}/{len(df)}")
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
            nd = r.get("net_decay_annual")
            print(
                f"  {r['ETF']:8s} decay={r['gross_decay_annual']*100:6.2f}%  "
                f"vol_und={r['vol_underlying_annual']*100:5.1f}%  "
                f"vol_etf={r['vol_etf_annual']*100:5.1f}%  "
                f"B={r['Beta']:.2f}  "
                f"borrow={bn*100 if pd.notna(bn) else 0:5.2f}%  "
                f"net={nd*100 if pd.notna(nd) else 0:6.2f}%"
            )

    print("=" * 60)
    return df
