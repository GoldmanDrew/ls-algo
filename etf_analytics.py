"""
etf_analytics.py — Drop-in decay + volatility enrichment for etf_screener.py

Adds columns to the screened DataFrame:
  - vol_underlying_annual  : annualized realized vol of the underlying (total return)
  - vol_etf_annual         : annualized realized vol of the ETF (total return)
  - gross_decay_annual     : annualized gross decay per $1 ETF short (before borrow)
  - net_decay_annual       : gross_decay_annual − borrow_current

All decay numbers are PER $1 OF ETF SHORT NOTIONAL.
Borrow is also per $1 ETF short. So: net = gross − borrow. No scaling needed.

Decay measurement — LOG RETURNS for both bull and inverse, per $1 ETF short:

  Hedged pair using signed beta in log space:

    weekly_pnl = β × ln(1+r_und_w) − ln(1+r_etf_w)

  For a perfect β× daily tracker:
    β × ln(1+r) − ln(1+βr) ≈ 0.5 × β(β−1) × r²  (always > 0 for |β| > 1)

  Log returns are required for BOTH bull and inverse because the
  simple-return hedge PnL (|β|×r_und − r_etf) is identically zero
  at daily frequency (by construction: r_etf = β×r_und) and ~zero
  at weekly. It cannot capture vol drag. Log returns do.

  Measured at weekly (W-FRI) frequency × 52 to reduce microstructure
  noise (bid-ask bounce, closing auctions, illiquid names).

  gross_decay_annual = mean(weekly_pnl) × 52
  net_decay_annual   = gross_decay_annual − borrow_current

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

# Common split/reverse-split ratios.  When yfinance's auto_adjust
# fails (or the adjustment hasn't propagated yet), a split shows up
# as a single-day return that is *exactly* one of these ratios.
# We detect returns within ±2 % of each ratio and correct the price
# series by dividing out the split factor from that day onward.
_SPLIT_RATIOS = sorted(set(
    [n / d for n in (1, 2, 3, 4, 5, 10, 15, 20, 25, 50)
           for d in (1, 2, 3, 4, 5, 10, 15, 20, 25, 50)
           if n != d and 0.05 <= n / d <= 20.0]
), reverse=True)

_SPLIT_TOL = 0.02          # ±2 % tolerance around each ratio
_JUMP_FLOOR = 0.40         # ignore daily moves < 40 %
_CONTEXT_WINDOW = 20       # days of context for local vol estimate
_ZSCORE_THRESHOLD = 4.0    # jump must be > 4σ vs local vol to be a split


def _clean_split_artifacts(prices: pd.Series) -> pd.Series:
    """Detect and correct unadjusted splits/reverse-splits in a price series.

    Approach:  walk through daily price ratios (p[t] / p[t-1]).
    If a ratio matches a known split factor (within tolerance) AND
    the jump is an extreme outlier relative to local volatility
    (z-score > 4), correct it by dividing all subsequent prices by
    the split factor.

    The z-score approach (instead of a fixed neighbor threshold) handles
    volatile penny/crypto stocks correctly: a 20:1 reverse split is
    still 10+ sigma even on a stock with 200% annualized vol.

    Returns a corrected copy of the series.
    """
    if len(prices) < 3:
        return prices.copy()

    prices = prices.copy()
    vals = prices.values.astype(float)

    # Pre-compute daily log returns for z-score calculation
    log_ratios = np.full(len(vals), np.nan)
    for i in range(1, len(vals)):
        if vals[i - 1] > 0 and np.isfinite(vals[i - 1]) and vals[i] > 0:
            log_ratios[i] = np.log(vals[i] / vals[i - 1])

    for i in range(1, len(vals) - 1):
        if vals[i - 1] == 0 or not np.isfinite(vals[i - 1]):
            continue
        ratio = vals[i] / vals[i - 1]
        daily_ret = ratio - 1.0

        # Skip small moves — not a split
        if abs(daily_ret) < _JUMP_FLOOR:
            continue

        # Check if this ratio matches a known split factor
        matched_factor = None
        for sf in _SPLIT_RATIOS:
            if abs(ratio - sf) / sf < _SPLIT_TOL:
                matched_factor = sf
                break
            # Also check inverse (reverse-split looks like 1/sf)
            inv_sf = 1.0 / sf
            if abs(ratio - inv_sf) / inv_sf < _SPLIT_TOL:
                matched_factor = inv_sf
                break

        if matched_factor is None:
            continue

        # Z-score test: is this jump an extreme outlier vs local vol?
        # Use a window of returns EXCLUDING the candidate split day.
        start = max(1, i - _CONTEXT_WINDOW)
        end = min(len(log_ratios), i + _CONTEXT_WINDOW + 1)
        context = [log_ratios[j] for j in range(start, end)
                   if j != i and np.isfinite(log_ratios[j])]

        if len(context) >= 5:
            local_std = float(np.std(context))
            if local_std > 0:
                log_jump = abs(np.log(ratio))
                zscore = log_jump / local_std
                if zscore < _ZSCORE_THRESHOLD:
                    # Jump is within normal vol range → real price move
                    continue

        # If we don't have enough context, fall back to accepting the
        # correction (better to fix a split than leave 839% vol).

        # Correct: divide everything from day i onward by the factor
        vals[i:] /= matched_factor

        # Recompute log_ratios for the corrected region so subsequent
        # iterations see clean data
        for j in range(max(1, i), min(len(vals), i + 2)):
            if vals[j - 1] > 0 and vals[j] > 0:
                log_ratios[j] = np.log(vals[j] / vals[j - 1])

    prices.iloc[:] = vals
    return prices


def _get_total_return_series(ticker: str, period: str = "2y") -> pd.Series:
    """
    Build a long-only total return price series for one ticker.

    Uses auto_adjust=True + repair=True so yfinance returns prices
    adjusted for both stock splits AND dividends, with Yahoo's own
    split-repair logic applied.  Then runs _clean_split_artifacts()
    as a second safety net for cases where yfinance's repair is
    incomplete or hasn't propagated yet.
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True, repair=True)

        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float, name=ticker)

        tr_price = df["Close"].dropna()
        tr_price = _clean_split_artifacts(tr_price)
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
    df = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
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
_VOL_CAP_ANNUAL = 5.0   # 500 % — loose backstop for truly broken data
                        # Split artifacts are now cleaned at the source
                        # (_clean_split_artifacts), so this should rarely
                        # bind.  Kept as a last-resort safety net only.


def _annualized_vol(tr_series: pd.Series, min_days: int = 60,
                    cap: float = _VOL_CAP_ANNUAL) -> float | None:
    """LEGACY: centered, simple-return annualized vol.

    Kept for diagnostics. New callers should use
    :func:`_annualized_second_moment_log`, which is the σ that aligns
    algebraically with the Itô decay identity on log returns. See
    notes in daily_screener.py.
    """
    tr = tr_series.dropna()
    if len(tr) < min_days + 1:
        return None
    ret = tr.pct_change().dropna()
    if len(ret) < min_days:
        return None
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    if cap and vol > cap:
        vol = cap
    return round(vol, 6)


def _annualized_second_moment_log(
    tr_series: pd.Series,
    min_days: int = 60,
    cap: float = _VOL_CAP_ANNUAL,
) -> float | None:
    """σ = √( mean(log_return_t²) · 252 ).

    The σ that makes ``0.5·|β|·|β−1|·σ²`` equal (in expectation) to the
    realized daily drag ``mean( β·ln(1+r_u) − ln(1+r_e) ) · 252`` under
    a noise-free β× daily tracker. Three deliberate differences from
    :func:`_annualized_vol`:

    1. Log returns, not simple returns (matches the realized-drag form).
    2. Uncentered second moment — keeps μ² in, since it IS part of
       E[r²] in the Itô correction.
    3. Same 252 annualization factor used everywhere else.
    """
    tr = tr_series.dropna()
    if len(tr) < min_days + 1:
        return None
    r = np.log(tr / tr.shift(1)).dropna()
    r = r[np.isfinite(r)]
    if len(r) < min_days:
        return None
    m2 = float((r ** 2).mean())
    if m2 <= 0:
        return None
    sigma = float(np.sqrt(m2 * TRADING_DAYS))
    if cap and sigma > cap:
        sigma = cap
    return round(sigma, 6)


# ──────────────────────────────────────────────
# Gross decay — per $1 ETF short
# ──────────────────────────────────────────────
_WEEKS_PER_YEAR = 52
_MIN_WEEKS = 12            # ~60 trading days (legacy)
_MIN_DAYS_DECAY = 40       # min aligned daily returns for realized gross decay


def _compute_gross_decay_weekly(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_weeks: int = _MIN_WEEKS,
) -> float | None:
    """LEGACY weekly × 52 realized-decay estimator — kept for diagnostics.

    Carries a +3.2 % annualization wedge vs the daily × 252 form
    (52·5 = 260 ≠ 252). Do not use for new production code; use
    :func:`_compute_gross_decay` (aliased to the daily form).
    """
    combined = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
    if len(combined) < min_weeks * 5:
        return None

    if abs(float(beta)) < 0.1:
        return None

    weekly = combined.resample("W-FRI").last().dropna()
    if len(weekly) < min_weeks + 1:
        return None

    r_etf = np.log(weekly["etf"] / weekly["etf"].shift(1))
    r_und = np.log(weekly["und"] / weekly["und"].shift(1))

    valid = r_etf.notna() & r_und.notna() & np.isfinite(r_etf) & np.isfinite(r_und)
    r_etf = r_etf[valid]
    r_und = r_und[valid]

    if len(r_etf) < min_weeks:
        return None

    weekly_pnl = float(beta) * r_und - r_etf
    return round(float(weekly_pnl.mean()) * _WEEKS_PER_YEAR, 6)


def _compute_gross_decay_daily(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_days: int = _MIN_DAYS_DECAY,
) -> float | None:
    """Realized gross annualized decay — daily log form, Itô-aligned.

        daily_drag_t = β · ln(1+r_und_t) − ln(1+r_etf_t)
        gross_decay_annual = mean(daily_drag_t) · 252

    Matches ``0.5·β·(β−1) · mean(r_und_t²) · 252`` in expectation under
    a noise-free β× daily-rebalance tracker, where ``mean(r_und_t²)``
    is the non-central second moment produced by
    :func:`_annualized_second_moment_log` squared over 252.
    """
    combined = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
    if len(combined) < min_days + 1:
        return None
    if abs(float(beta)) < 0.1:
        return None

    r_etf = np.log(combined["etf"] / combined["etf"].shift(1))
    r_und = np.log(combined["und"] / combined["und"].shift(1))
    valid = r_etf.notna() & r_und.notna() & np.isfinite(r_etf) & np.isfinite(r_und)
    r_etf, r_und = r_etf[valid], r_und[valid]
    if len(r_etf) < min_days:
        return None

    daily_drag = float(beta) * r_und - r_etf
    return round(float(daily_drag.mean()) * TRADING_DAYS, 6)


def _compute_gross_decay(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_weeks: int | None = None,
    min_days: int = _MIN_DAYS_DECAY,
) -> float | None:
    """Default realized-decay estimator (aliases the daily form).

    Kept as the public name so existing callers keep working. The
    legacy weekly form is available at :func:`_compute_gross_decay_weekly`.
    """
    return _compute_gross_decay_daily(etf_tr, und_tr, beta, min_days=min_days)


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
      gross_decay_annual    - per $1 ETF short
                              Bull: simple returns |β|×r_und − r_etf
                              Inverse: log returns β×ln(1+r_und) − ln(1+r_etf)
      net_decay_annual      - gross_decay_annual − borrow_current

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

    vols_etf_raw = []
    decays = []
    betas_out = []
    beta_nobs_out = []

    ok_decay = 0
    ok_vol = 0
    betas_computed = 0

    # ── PASS 1: betas, raw ETF vols, realized decay ──
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

        # Itô-aligned σ: √(mean(log_return²) · 252). Matches the measure
        # used by the new _compute_gross_decay (daily log form).
        vol_etf = _annualized_second_moment_log(tr_map[etf], min_days) if etf in tr_map else None
        vols_etf_raw.append(vol_etf)

        # ── Gross decay ──
        decay = None
        if (
            beta_f is not None
            and abs(beta_f) >= 0.1
            and und
            and etf in tr_map
            and und in tr_map
        ):
            decay = _compute_gross_decay(tr_map[etf], tr_map[und], beta_f)

        decays.append(decay)
        if decay is not None:
            ok_decay += 1

    df["Beta"] = betas_out
    df["Beta_n_obs"] = beta_nobs_out
    df["vol_etf_annual"] = vols_etf_raw
    df["gross_decay_annual"] = decays

    # ── PASS 2: resolve underlying vol per ticker ──
    # For each underlying, compute raw vol from its price series, then
    # cross-check against implied vols from its ETFs (vol_etf / |β|).
    # If raw vol is corrupted (e.g. unadjusted splits), use the best
    # ETF-implied vol instead — picking the ETF with the most history
    # and highest |β| (tightest leverage relationship).
    # All ETFs sharing the same underlying get the SAME vol_und.
    _VOL_RATIO_MAX = 2.0

    unique_unds = df["Underlying"].dropna().apply(norm).unique()
    resolved_vol_und = {}

    for und in unique_unds:
        raw_vol = _annualized_second_moment_log(tr_map[und], min_days) if und in tr_map else None

        mask = df["Underlying"].apply(
            lambda x, u=und: norm(x) == u if pd.notna(x) else False)
        implied_candidates = []
        for idx in df.index[mask]:
            b = betas_out[idx]
            ve = vols_etf_raw[idx]
            nobs = beta_nobs_out[idx]
            if b and abs(b) >= 0.5 and ve and ve > 0 and nobs:
                implied = ve / abs(b)
                weight = nobs * abs(b)
                implied_candidates.append((implied, weight))

        if not implied_candidates:
            resolved_vol_und[und] = raw_vol
            continue

        total_w = sum(w for _, w in implied_candidates)
        best_implied = sum(v * w for v, w in implied_candidates) / total_w

        if raw_vol is None or raw_vol <= 0:
            resolved_vol_und[und] = round(best_implied, 6)
        elif best_implied > 0 and raw_vol / best_implied > _VOL_RATIO_MAX:
            resolved_vol_und[und] = round(best_implied, 6)
            print(f"  [VOL-FIX] {und}: raw={raw_vol*100:.1f}% -> implied={best_implied*100:.1f}% "
                  f"(from {len(implied_candidates)} ETF(s))")
        else:
            resolved_vol_und[und] = raw_vol

    vols_und = []
    for i, row in df.iterrows():
        und = norm(row["Underlying"]) if pd.notna(row.get("Underlying")) else None
        vol_und = resolved_vol_und.get(und) if und else None
        vols_und.append(vol_und)
        if vol_und is not None:
            ok_vol += 1

    df["vol_underlying_annual"] = vols_und

    # ── Net decay = gross − borrow (both per $1 ETF short) ──
    borrow_net = pd.to_numeric(
        df.get("borrow_current", df.get("borrow_fee_annual")),
        errors="coerce",
    )
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
            bn = r.get("borrow_current")
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
