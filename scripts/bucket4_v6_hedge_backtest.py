#!/usr/bin/env python3
"""
Simulate Bucket 4 v6 hedge ratios over time (Rentec Option-2 composite) and backtest vs static h.

Uses the same pair filters and sizing mechanics as notebooks/Simple_Pair_Backtest.ipynb:
- Short inverse ETF (leg A), short underlying (leg B)
- denom = 1 + h * |beta_ETF|; split gross between legs

v6 policy (Bucket4_Hedge_Ratio_v6_Rentec_Feature_Bank.html):
- z_r10d = -robust_z_xsec(r_10d), z_rexp = +robust_z_xsec(range_expansion)
- z_comp = mean(z_r10d, z_rexp)
- h_star = clip(0.5 - K*z_comp, H_MIN, H_MAX)
- h_applied = (1-alpha)*h_prev + alpha*h_star on each rebalance
- Rebalance / signal cadence: every N business days (default 10)

Outputs:
- CSV of h_applied by (date, underlying)
- Summary table: per-pair CAGR, max DD, Sharpe for v6 vs h=0.5 vs optional static grid
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Defaults from v6 HTML
V6_K = 0.05
V6_ALPHA = 0.25
V6_H_MIN = 0.10
V6_H_MAX = 1.10
V6_BDAY_STEP = 10


def _norm_sym(x: str) -> str:
    return str(x).strip().upper()


def load_bucket4_pairs(
    screened_csv: Path,
    *,
    min_underlying_vol: float = 0.60,
    min_net_decay: float = 0.20,
) -> list[tuple[str, str]]:
    df = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in df.columns}
    etf_col = cols.get("etf")
    und_col = cols.get("underlying")
    beta_col = cols.get("beta")
    bucket_col = cols.get("bucket")

    if etf_col is None or und_col is None or beta_col is None:
        raise ValueError("screened_csv must include ETF, Underlying, Beta columns")

    vol_candidates = [
        "vol_underlying_annual",
        "underlying_vol_annual",
        "underlying_vol",
        "underlying_volatility_annual",
        "underlying_realized_vol_annual",
    ]
    decay_candidates = [
        "bucket4_net_edge_annual",
        "net_decay_annual",
        "net_edge_annual",
    ]

    vol_col = next((cols[c] for c in vol_candidates if c in cols), None)
    decay_col = next((cols[c] for c in decay_candidates if c in cols), None)
    if vol_col is None:
        raise ValueError(f"Could not find underlying vol column. Tried: {vol_candidates}")
    if decay_col is None:
        raise ValueError(f"Could not find net decay column. Tried: {decay_candidates}")

    use_cols = [etf_col, und_col, beta_col, vol_col, decay_col] + ([bucket_col] if bucket_col else [])
    tmp = df[use_cols].copy()
    tmp[etf_col] = tmp[etf_col].astype(str).map(_norm_sym)
    tmp[und_col] = tmp[und_col].astype(str).map(_norm_sym)
    tmp[beta_col] = pd.to_numeric(tmp[beta_col], errors="coerce")
    tmp[vol_col] = pd.to_numeric(tmp[vol_col], errors="coerce")
    tmp[decay_col] = pd.to_numeric(tmp[decay_col], errors="coerce")

    mask = tmp[beta_col].notna() & (tmp[beta_col] < 0) & tmp[etf_col].ne("") & tmp[und_col].ne("")
    if bucket_col:
        b = tmp[bucket_col].astype(str).str.lower()
        mask = mask & b.isin(["bucket_4", "bucket_3_inverse", "bucket_3"])

    mask = mask & (tmp[vol_col] > min_underlying_vol) & (tmp[decay_col] > min_net_decay)

    out = tmp.loc[mask, [etf_col, und_col]].drop_duplicates()
    pairs = [(r[etf_col], r[und_col]) for _, r in out.iterrows()]

    manual_candidates = [("UVIX", "SVIX")]
    for cand in reversed(manual_candidates):
        if cand in pairs:
            pairs.remove(cand)
        pairs.insert(0, cand)

    print(
        f"[FILTER] vol>{min_underlying_vol:.0%} decay>{min_net_decay:.0%}: {len(pairs)} pairs from {screened_csv}"
    )
    if not pairs:
        raise RuntimeError("No bucket-4 candidate pairs after filters")
    return pairs


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float, name=ticker)
    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = [str(x).lower() for x in raw.columns.get_level_values(0)]
        lvl1 = [str(x).lower() for x in raw.columns.get_level_values(1)]
        if "close" in lvl0:
            close = raw.xs("Close", axis=1, level=0)
        elif "close" in lvl1:
            close = raw.xs("Close", axis=1, level=1)
    else:
        if "Close" in raw.columns:
            close = raw["Close"]
    if close is None:
        raise RuntimeError(f"Missing Close for {ticker}")
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            s = close[ticker]
        else:
            s = close.iloc[:, 0]
    else:
        s = close
    return pd.to_numeric(s, errors="coerce").rename(ticker)


def load_close_from_cache(cache_dir: Path, ticker: str) -> pd.Series | None:
    """Load `{ticker}.csv` with Date index and `close` (case-insensitive) or a single price column."""
    p = cache_dir / f"{ticker}.csv"
    if not p.exists():
        pq = cache_dir / f"{ticker}.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            c = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            s.index = pd.to_datetime(df.index).tz_localize(None)
            return s.rename(ticker).sort_index()
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    cols = {str(c).lower(): c for c in df.columns}
    col = cols.get("close", df.columns[0] if len(df.columns) == 1 else None)
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s.sort_index()


def download_close(
    ticker: str,
    start: str,
    end: str | None,
    *,
    cache_dir: Path | None = None,
    retries: int = 4,
) -> pd.Series:
    if cache_dir is not None:
        cached = load_close_from_cache(cache_dir, ticker)
        if cached is not None and len(cached.dropna()) >= 60:
            if end:
                cached = cached.loc[: pd.Timestamp(end)]
            return cached.loc[cached.index >= pd.Timestamp(start)]
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            tkr = yf.Ticker(ticker)
            df = tkr.history(start=start, end=end, auto_adjust=True, repair=True)
            if df is not None and len(df) > 0 and "Close" in df.columns:
                s = df["Close"].dropna().rename(ticker)
                s.index = pd.to_datetime(s.index).tz_localize(None)
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    s.to_csv(cache_dir / f"{ticker}.csv", header=True)
                return s
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                timeout=60,
            )
            s = _extract_close(raw, ticker).dropna()
            if len(s) > 0:
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    s.to_csv(cache_dir / f"{ticker}.csv", header=True)
                return s
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (attempt + 1))
    if last_err:
        raise last_err
    return pd.Series(dtype=float, name=ticker)


def write_synthetic_price_cache(
    cache_dir: Path,
    symbols: list[str],
    *,
    start: str = "2022-03-31",
    n_days: int = 700,
    seed: int = 42,
) -> None:
    """Deterministic random-walk prices for pipeline testing when Yahoo is unavailable."""
    rng = np.random.default_rng(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start=start, periods=n_days)
    for sym in symbols:
        r = rng.normal(0.0008, 0.025, len(dates))
        px = 50.0 * np.exp(np.cumsum(r))
        pd.Series(px, index=dates, name="close").to_csv(cache_dir / f"{sym}.csv", header=True)


def daily_log_returns(close: pd.Series) -> pd.Series:
    c = close.astype(float).replace(0.0, np.nan)
    return np.log(c / c.shift(1))


def robust_z_cross_section(s: pd.Series) -> pd.Series:
    """Robust z (median/MAD), clipped; NaN preserved for missing names."""
    v = pd.to_numeric(s, errors="coerce")
    m = v.median(skipna=True)
    mad = (v - m).abs().median(skipna=True)
    scale = 1.4826 * float(mad) if pd.notna(mad) and mad > 0 else float(v.std(skipna=True) or 1.0)
    z = (v - m) / scale if scale > 0 else v * 0.0
    return z.clip(lower=-3.0, upper=3.0)


def compute_features_asof(close: pd.Series, asof: pd.Timestamp) -> tuple[float | None, float | None]:
    """r_10d and range_expansion (vol5/vol63) using data up to and including asof."""
    sub = close.loc[:asof].dropna()
    if len(sub) < 64:
        return None, None
    r = daily_log_returns(sub)
    if len(sub) < 11:
        return None, None
    r_10d = float(np.log(sub.iloc[-1] / sub.iloc[-11]))
    tail = r.iloc[-63:]
    if tail.notna().sum() < 40:
        return r_10d, None
    vol5 = float(tail.iloc[-5:].std())
    vol63 = float(tail.std())
    if vol63 <= 1e-12:
        return r_10d, None
    rexp = vol5 / vol63
    return r_10d, rexp


def build_v6_h_panel(
    closes_by_underlying: dict[str, pd.Series],
    *,
    bday_step: int = V6_BDAY_STEP,
    k: float = V6_K,
    alpha: float = V6_ALPHA,
    h_min: float = V6_H_MIN,
    h_max: float = V6_H_MAX,
    warmup_bdays: int = 65,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Returns long DataFrame with columns: date, underlying, r_10d, range_expansion,
    z_r10d, z_rexp, z_comp, h_star, h_applied
    and the rebalance DatetimeIndex used.
    """
    if not closes_by_underlying:
        raise ValueError("empty closes")

    # Master calendar: union of all dates
    all_idx = None
    for s in closes_by_underlying.values():
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    all_idx = all_idx.sort_values()
    # Business days approximation: use dates present (already trading days from yfinance)
    if len(all_idx) <= warmup_bdays:
        raise RuntimeError(
            f"Not enough history for warmup: calendar_len={len(all_idx)} (need >{warmup_bdays}). "
            "Check Yahoo data downloads / network."
        )

    start_i = warmup_bdays
    rebal_dates = all_idx[start_i::bday_step]
    underlyings = sorted(closes_by_underlying.keys())

    h_prev = {u: 0.5 for u in underlyings}
    rows = []

    for dt in rebal_dates:
        r10 = {}
        rexp = {}
        for u in underlyings:
            r_10d, rx = compute_features_asof(closes_by_underlying[u], dt)
            if r_10d is not None:
                r10[u] = r_10d
            if rx is not None:
                rexp[u] = rx

        s10 = pd.Series(r10, dtype=float)
        srx = pd.Series(rexp, dtype=float)

        if len(s10) >= 2:
            z_cs_10 = robust_z_cross_section(s10)
            z_r10d_aligned = -1.0 * z_cs_10
        elif len(s10) == 1:
            z_r10d_aligned = pd.Series({s10.index[0]: 0.0})
        else:
            z_r10d_aligned = pd.Series(dtype=float)

        if len(srx) >= 2:
            z_cs_rx = robust_z_cross_section(srx)
            z_rexp_aligned = +1.0 * z_cs_rx
        elif len(srx) == 1:
            z_rexp_aligned = pd.Series({srx.index[0]: 0.0})
        else:
            z_rexp_aligned = pd.Series(dtype=float)

        for u in underlyings:
            a = z_r10d_aligned.get(u, np.nan)
            b = z_rexp_aligned.get(u, np.nan)
            if pd.isna(a) and pd.isna(b):
                h_new = h_prev[u]
                rows.append(
                    {
                        "date": dt,
                        "underlying": u,
                        "r_10d": s10.get(u, np.nan),
                        "range_expansion": srx.get(u, np.nan),
                        "z_r10d": np.nan,
                        "z_rexp": np.nan,
                        "z_comp": np.nan,
                        "h_star": np.nan,
                        "h_applied": h_new,
                    }
                )
                continue
            if pd.isna(a):
                a = 0.0
            if pd.isna(b):
                b = 0.0
            z_comp = 0.5 * float(a) + 0.5 * float(b)
            h_star = float(np.clip(0.5 - k * z_comp, h_min, h_max))
            h_new = (1.0 - alpha) * h_prev[u] + alpha * h_star
            h_prev[u] = h_new
            rows.append(
                {
                    "date": dt,
                    "underlying": u,
                    "r_10d": s10.get(u, np.nan),
                    "range_expansion": srx.get(u, np.nan),
                    "z_r10d": float(a),
                    "z_rexp": float(b),
                    "z_comp": z_comp,
                    "h_star": h_star,
                    "h_applied": h_new,
                }
            )

    panel = pd.DataFrame(rows)
    return panel, rebal_dates


def panel_to_h_daily(
    panel: pd.DataFrame,
    master_index: pd.DatetimeIndex,
    underlyings: list[str],
) -> dict[str, pd.Series]:
    """Forward-filled daily h per underlying."""
    out = {}
    if panel.empty:
        for u in underlyings:
            out[u] = pd.Series(0.5, index=master_index)
        return out

    pivot = panel.pivot_table(index="date", columns="underlying", values="h_applied", aggfunc="last")
    for u in underlyings:
        if u in pivot.columns:
            s = pivot[u].reindex(master_index).ffill()
        else:
            s = pd.Series(np.nan, index=master_index)
        # start at 0.5 before first signal
        s = s.fillna(0.5)
        # if still nan (empty), 0.5
        s = s.fillna(0.5)
        out[u] = s
    return out


def run_bucket4_backtest_dynamic_h(
    prices: pd.DataFrame,
    h_daily: pd.Series,
    rebal_dates: pd.DatetimeIndex,
    *,
    initial_capital: float = 100_000.0,
    gross_multiplier: float = 1.0,
    beta_a: float = -2.0,
    beta_b: float = 1.0,
    borrow_a_annual: float = 0.0,
    borrow_b_annual: float = 0.0,
    short_proceeds_annual: float = 0.0,
    fee_bps: float = 0.0,
) -> pd.DataFrame:
    bt = prices.copy()
    h_aligned = h_daily.reindex(bt.index).ffill().fillna(0.5)
    bt["rebalance"] = bt.index.isin(rebal_dates)
    bt.iloc[0, bt.columns.get_loc("rebalance")] = True

    a_sh, b_sh = 0.0, 0.0
    cash = float(initial_capital)
    fee_rate = fee_bps / 10_000.0
    borrow_a_daily = float(borrow_a_annual) / 252.0
    borrow_b_daily = float(borrow_b_annual) / 252.0
    short_proceeds_daily = float(short_proceeds_annual) / 252.0
    beta_inv_abs = abs(float(beta_a))

    rows = []
    for dt, row in bt.iterrows():
        ap = float(row["a_px"])
        bp = float(row["b_px"])
        h = float(h_aligned.loc[dt])

        a_pos_notional = a_sh * ap
        b_pos_notional = b_sh * bp

        borrow_cost = 0.0
        short_proceeds_credit = 0.0
        if a_pos_notional < 0:
            borrow_cost += abs(a_pos_notional) * borrow_a_daily
            short_proceeds_credit += abs(a_pos_notional) * short_proceeds_daily
        if b_pos_notional < 0:
            borrow_cost += abs(b_pos_notional) * borrow_b_daily
            short_proceeds_credit += abs(b_pos_notional) * short_proceeds_daily

        financing_pnl = short_proceeds_credit - borrow_cost
        cash += financing_pnl
        equity = cash + a_pos_notional + b_pos_notional

        if bool(row["rebalance"]):
            target_gross = max(0.0, float(gross_multiplier) * equity)
            denom = 1.0 + h * beta_inv_abs
            n_a = target_gross / denom if denom > 1e-12 else 0.5 * target_gross
            n_b = max(0.0, target_gross - n_a)

            target_a_pos = -n_a
            target_b_pos = -n_b

            delta_a = target_a_pos - a_pos_notional
            delta_b = target_b_pos - b_pos_notional
            traded_gross = abs(delta_a) + abs(delta_b)
            fee = traded_gross * fee_rate

            cash -= delta_a
            cash -= delta_b
            cash -= fee

            a_sh = target_a_pos / ap if ap > 0 else 0.0
            b_sh = target_b_pos / bp if bp > 0 else 0.0

            a_pos_notional = a_sh * ap
            b_pos_notional = b_sh * bp
            equity = cash + a_pos_notional + b_pos_notional

        beta_notional = (-1.0) * float(beta_a) * abs(a_pos_notional) + (-1.0) * float(beta_b) * abs(b_pos_notional)

        rows.append(
            {
                "date": dt,
                "a_px": ap,
                "b_px": bp,
                "cash": cash,
                "a_shares": a_sh,
                "b_shares": b_sh,
                "equity": equity,
                "h_used": h,
                "rebalance": bool(row["rebalance"]),
                "beta_notional": beta_notional,
            }
        )

    out = pd.DataFrame(rows).set_index("date")
    out["ret"] = out["equity"].pct_change().fillna(0.0)
    out["cum_return"] = out["equity"] / out["equity"].iloc[0] - 1.0
    out["drawdown"] = out["equity"].div(out["equity"].cummax()).sub(1.0)
    out["beta_exposure_frac"] = np.where(
        out["equity"].abs() > 1e-9, out["beta_notional"] / out["equity"], np.nan
    )
    return out


def load_pair_betas_and_borrow(
    etf: str,
    und: str,
    screened: pd.DataFrame,
) -> tuple[float, float, float]:
    cols = {c.lower(): c for c in screened.columns}
    etf_col, und_col = cols["etf"], cols["underlying"]
    beta_col = cols["beta"]
    m = (
        screened[etf_col].astype(str).map(_norm_sym).eq(_norm_sym(etf))
        & screened[und_col].astype(str).map(_norm_sym).eq(_norm_sym(und))
    )
    row = screened.loc[m]
    if row.empty:
        return -2.0, 1.0, 0.10
    r = row.iloc[0]
    beta_a = float(pd.to_numeric(r[beta_col], errors="coerce") or -2.0)
    for key in ("borrow_current", "borrow_fee_annual", "borrow_net_annual"):
        if key in r.index and pd.notna(r[key]):
            borrow_a = float(r[key])
            break
    else:
        borrow_a = 0.10
    # underlying borrow internalized 0 for B4 test
    beta_b = 1.0
    if (etf, und) == ("UVIX", "SVIX"):
        beta_a, beta_b = 2.0, -1.0
    return beta_a, beta_b, borrow_a


def perf_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"cagr": np.nan, "total_return": np.nan, "max_drawdown": np.nan, "sharpe": np.nan}
    ret = eq.pct_change().dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = len(eq) / 252.0
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    peak = eq.cummax()
    dd = float(((eq / peak) - 1.0).min())
    sharpe = float(np.sqrt(252.0) * ret.mean() / ret.std()) if ret.std() > 1e-12 else np.nan
    return {"cagr": cagr, "total_return": total, "max_drawdown": dd, "sharpe": sharpe}


def load_prices_pair(
    etf: str,
    und: str,
    start: str,
    end: str | None,
    *,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    a = download_close(etf, start, end, cache_dir=cache_dir).rename("a_px")
    b = download_close(und, start, end, cache_dir=cache_dir).rename("b_px")
    first_a = a.dropna().index.min()
    first_b = b.dropna().index.min()
    if pd.isna(first_a) or pd.isna(first_b):
        raise RuntimeError(f"No data {etf}/{und}")
    aligned_start = max(first_a, first_b)
    px = pd.concat([a, b], axis=1).loc[lambda x: x.index >= aligned_start].dropna()
    return px


def main() -> None:
    p = argparse.ArgumentParser(description="Bucket 4 v6 hedge simulation + backtest")
    p.add_argument("--screened-csv", type=Path, default=Path("data/etf_screened_today.csv"))
    p.add_argument("--start", default="2022-03-31")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--min-vol", type=float, default=0.60)
    p.add_argument("--min-decay", type=float, default=0.20)
    p.add_argument("--out-dir", type=Path, default=Path("data/runs/bucket4_v6_sim"))
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/price_cache_v6"),
        help="Directory for per-ticker CSV caches (and Yahoo save-on-download).",
    )
    p.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Write deterministic random-walk CSVs for all pair symbols into cache-dir, then run (offline test).",
    )
    p.add_argument("--static-h-compare", type=float, nargs="*", default=[0.5, 1.0])
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    screened_path = args.screened_csv if args.screened_csv.is_absolute() else root / args.screened_csv
    pairs = load_bucket4_pairs(screened_path, min_underlying_vol=args.min_vol, min_net_decay=args.min_decay)
    screened_df = pd.read_csv(screened_path)

    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else root / args.cache_dir
    if args.synthetic_demo:
        syms = sorted({e for e, _ in pairs} | {u for _, u in pairs})
        write_synthetic_price_cache(cache_dir, syms, start=args.start)
        print(f"[INFO] Synthetic demo: wrote {len(syms)} price series to {cache_dir}")

    unds = sorted({u for _, u in pairs})
    end = args.end

    print("[INFO] Loading underlying closes for v6 cross-section (cache or Yahoo)...")
    closes: dict[str, pd.Series] = {}
    for u in unds:
        try:
            s = download_close(u, args.start, end, cache_dir=cache_dir)
            if len(s.dropna()) >= 120:
                closes[u] = s
            else:
                print(f"[WARN] skip {u}: only {len(s)} rows (need >=120)")
        except Exception as e:
            print(f"[WARN] skip underlying {u}: {e}")

    if len(closes) < 2:
        raise SystemExit(
            "Need at least 2 underlyings with sufficient Yahoo history for cross-sectional z. "
            "Retry later or check connectivity."
        )

    panel, rebal_dates = build_v6_h_panel(closes, bday_step=V6_BDAY_STEP)
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "v6_h_panel.csv"
    panel.to_csv(panel_path, index=False)
    print(f"[OK] Wrote {panel_path} ({len(panel)} rows)")

    all_idx = None
    for s in closes.values():
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    all_idx = all_idx.sort_values()
    h_daily_map = panel_to_h_daily(panel, all_idx, unds)

    summaries = []

    for etf, und in pairs:
        if und not in closes:
            print(f"[SKIP] {etf}/{und} — no underlying history")
            continue
        try:
            px = load_prices_pair(etf, und, args.start, end, cache_dir=cache_dir)
        except Exception as e:
            print(f"[SKIP] {etf}/{und} — {e}")
            continue

        beta_a, beta_b, borrow_a = load_pair_betas_and_borrow(etf, und, screened_df)
        if beta_a >= 0:
            print(f"[SKIP] {etf}/{und} — beta_a not negative inverse ({beta_a})")
            continue

        h_d = h_daily_map[und].reindex(px.index).ffill().fillna(0.5)
        pair_rebal = rebal_dates.intersection(px.index)
        if len(pair_rebal) == 0:
            pair_rebal = pd.DatetimeIndex([px.index[0]])

        bt_v6 = run_bucket4_backtest_dynamic_h(
            px,
            h_d,
            pair_rebal,
            beta_a=beta_a,
            beta_b=beta_b,
            borrow_a_annual=borrow_a,
            borrow_b_annual=0.0,
        )
        st_v6 = perf_stats(bt_v6["equity"])
        st_v6["pair"] = f"{etf}/{und}"
        st_v6["run"] = "v6_dynamic"
        summaries.append(st_v6)

        for hh in args.static_h_compare:
            bt_s = run_bucket4_backtest_dynamic_h(
                px,
                pd.Series(float(hh), index=px.index),
                pair_rebal,
                beta_a=beta_a,
                beta_b=beta_b,
                borrow_a_annual=borrow_a,
                borrow_b_annual=0.0,
            )
            st = perf_stats(bt_s["equity"])
            st["pair"] = f"{etf}/{und}"
            st["run"] = f"static_h_{hh:g}"
            summaries.append(st)

        # Save one equity curve for inspection (v6)
        safe_name = f"{etf}_{und}".replace("/", "-")
        bt_v6.to_csv(out_dir / f"equity_{safe_name}.csv")

    summ_df = pd.DataFrame(summaries)
    if not summ_df.empty:
        summ_path = out_dir / "summary_by_pair.csv"
        summ_df.to_csv(summ_path, index=False)
        print("\n=== Average metrics by run (equal-weight across pairs) ===")
        agg = summ_df.groupby("run", as_index=False).agg(
            n=("cagr", "count"),
            mean_cagr=("cagr", "mean"),
            med_cagr=("cagr", "median"),
            mean_mdd=("max_drawdown", "mean"),
            mean_sharpe=("sharpe", "mean"),
        )
        print(agg.to_string(index=False))
        print(f"\n[OK] Wrote {summ_path}")

    # Pivot: pair x run
    if not summ_df.empty:
        pv = summ_df.pivot_table(index="pair", columns="run", values="cagr")
        pv_path = out_dir / "cagr_pivot.csv"
        pv.to_csv(pv_path)
        print(f"[OK] Wrote {pv_path}")


if __name__ == "__main__":
    main()
