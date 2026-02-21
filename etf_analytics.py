"""
etf_analytics.py — Drop-in decay + volatility enrichment for etf_screener.py

Adds columns to the screened DataFrame:
  - vol_underlying_annual  : annualized realized vol of the underlying (total return)
  - vol_etf_annual         : annualized realized vol of the ETF (total return)
  - gross_decay_annual     : annualized gross decay per $1 ETF short (before borrow)
  - net_decay_annual       : gross_decay_annual − borrow_net_annual

All decay numbers are PER $1 OF ETF SHORT NOTIONAL.
Borrow is also per $1 ETF short. So: net = gross − borrow. No scaling needed.

Decay measurement — TWO formulas, both per $1 ETF short:

  ── Bull ETFs (Beta > 0) ──────────────────────────────────
  Hedged pair: short $1 ETF + long |β| × $1 underlying.

    daily_pnl = |β| × r_underlying − r_etf

  For a perfect β× daily tracker: daily_pnl = 0.
  Vol drag / fees / tracking error → daily_pnl > 0 → positive decay.

  ── Inverse ETFs (Beta < 0) ───────────────────────────────
  Unhedged short: short $1 of the inverse ETF only (no underlying leg).

    daily_pnl = −r_etf

  For a perfect −|β|× tracker: daily_pnl = |β| × r_und.
  Vol drag makes the inverse ETF lose more → −r_etf > |β|×r_und → excess is decay.

  ── Common ────────────────────────────────────────────────
  gross_decay_annual = mean(daily_pnl) × 252   (simple linear rate)
  net_decay_annual   = gross_decay_annual − borrow_net_annual

Both legs use explicit total-return price series:
  TR_t = TR_{t-1} × (Close_t + Div_t) / Close_{t-1}
so dividends are correctly captured on both sides.

If Beta is missing (NaN) for a row, it is computed via OLS regression.

Usage in etf_screener.py:
    from etf_analytics import enrich_with_decay_and_vol
    df = enrich_with_decay_and_vol(df)
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
# Gross decay — per $1 ETF short, simple annual
# ──────────────────────────────────────────────
def _compute_gross_decay(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_days: int = 60,
) -> float | None:
    """
    Gross annualized decay per $1 of ETF short notional.
    Returns a SIMPLE (linear) annual rate, same units as borrow.

    ── Bull ETFs (beta > 0) ────────────────────────
    Short $1 ETF + long |β| × $1 underlying.

      daily_pnl = |β| × r_und − r_etf

    For a perfect β× tracker: daily_pnl = 0.
    Positive = vol drag + fees → profitable to short.

    ── Inverse ETFs (beta < 0) ─────────────────────
    Short $1 inverse ETF only (no underlying leg).

      daily_pnl = −r_etf

    For a perfect −|β|× tracker: daily_pnl = |β| × r_und.
    Vol drag makes −r_etf > |β|×r_und → excess is decay.

    Both formulas give P&L per $1 ETF short.
    net_decay = gross_decay − borrow_net_annual (no scaling needed).
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

    if beta > 0:
        # Bull ETF: short $1 ETF + long |β| underlying
        daily_pnl = abs_beta * r_und - r_etf
    else:
        # Inverse ETF: short $1 ETF only (no underlying leg)
        daily_pnl = -r_etf

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
      gross_decay_annual    - per $1 ETF short, simple annualized
                              Bull:    |β|×r_und − r_etf  (0 for perfect tracker)
                              Inverse: −r_etf             (|β|×r_und for perfect tracker)
      net_decay_annual      - gross_decay_annual − borrow_net_annual

    No borrow_drag_annual column — borrow is already per $1 ETF short,
    decay is now also per $1 ETF short, so they subtract directly.
    """
    print("=" * 60)
    print("[etf_analytics] Computing decay + volatility (total return)")
    print("[etf_analytics] Basis: per $1 ETF short notional")
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

    # ── Net decay = gross − borrow (both per $1 ETF short) ──
    borrow_net = pd.to_numeric(df.get("borrow_net_annual"), errors="coerce")
    df["net_decay_annual"] = np.where(
        df["gross_decay_annual"].notna() & borrow_net.notna(),
        df["gross_decay_annual"] - borrow_net,
        np.nan,
    )

    # ── Summary ──
    print(f"\n[etf_analytics] Beta computed from OLS: {betas_computed}")
    print(f"[etf_analytics] Volatility computed: {ok_vol}/{len(df)}")
    print(f"[etf_analytics] Decay computed:      {ok_decay}/{len(df)}")

    valid_decays = df["gross_decay_annual"].dropna()
    if len(valid_decays) > 0:
        print(f"\n[etf_analytics] Decay stats (gross, per $1 ETF short):")
        print(f"  Range:  {valid_decays.min()*100:.2f}% -- {valid_decays.max()*100:.2f}%")
        print(f"  Median: {valid_decays.median()*100:.2f}%")

    valid_net = df["net_decay_annual"].dropna()
    if len(valid_net) > 0:
        print(f"\n[etf_analytics] Decay stats (net = gross − borrow):")
        print(f"  Range:  {valid_net.min()*100:.2f}% -- {valid_net.max()*100:.2f}%")
        print(f"  Median: {valid_net.median()*100:.2f}%")

    valid_vols = [v for v in vols_und if v is not None]
    if valid_vols:
        print(f"\n[etf_analytics] Underlying vol range: "
              f"{min(valid_vols)*100:.1f}% -- {max(valid_vols)*100:.1f}%")

    # Top 5 by net decay
    temp = df[df["net_decay_annual"].notna()].nlargest(5, "net_decay_annual")
    if len(temp) > 0:
        print(f"\n[etf_analytics] Top 5 net decay:")
        for _, r in temp.iterrows():
            bn = r.get("borrow_net_annual")
            print(
                f"  {r['ETF']:8s} net={r['net_decay_annual']*100:6.2f}%  "
                f"gross={r['gross_decay_annual']*100:6.2f}%  "
                f"borrow={bn*100 if pd.notna(bn) else 0:5.2f}%  "
                f"vol_und={r['vol_underlying_annual']*100:5.1f}%  "
                f"vol_etf={r['vol_etf_annual']*100:5.1f}%  "
                f"B={r['Beta']:.2f}"
            )

    print("=" * 60)
    return df
