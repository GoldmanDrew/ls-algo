#!/usr/bin/env python3
"""
Compare alternative ETF→underlying beta estimators vs the screener's default
(OLS on aligned daily *simple* returns: cov/var).

Uses the same Yahoo v8 total-return series as daily_screener.download_all_tr_series.

Examples:
  python scripts/compare_beta_methods.py --etf ORCX HIMZ MSTX NVDL
  python scripts/compare_beta_methods.py --etf ORCX --lookback 1y --min-days 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from daily_screener import compute_beta_ols, download_all_tr_series  # noqa: E402
from ibkr_accounting import load_etf_to_under_map  # noqa: E402


def _norm_sym(s: str) -> str:
    return str(s).upper().replace(".", "-").strip()


def _aligned_levels(etf_tr: pd.Series, und_tr: pd.Series) -> pd.DataFrame:
    etf_tr = etf_tr[~etf_tr.index.duplicated(keep="last")]
    und_tr = und_tr[~und_tr.index.duplicated(keep="last")]
    df = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, join="inner"
    ).dropna()
    return df.sort_index()


def _pct_returns(levels: pd.DataFrame) -> pd.DataFrame:
    r = levels.pct_change().dropna()
    return r


def beta_ols_log(levels: pd.DataFrame, min_days: int) -> tuple[float | None, int]:
    """OLS slope of log returns: cov(dlog etf, dlog und) / var(dlog und)."""
    le = np.log(levels["etf"]).diff().dropna()
    lu = np.log(levels["und"]).diff().dropna()
    valid = le.index.intersection(lu.index)
    le, lu = le.loc[valid], lu.loc[valid]
    if len(le) < min_days:
        return None, len(le)
    cov = np.cov(le.values, lu.values)
    var_u = cov[1, 1]
    if var_u < 1e-12:
        return None, len(le)
    return round(float(cov[0, 1] / var_u), 6), len(le)


def beta_ols_simple_on_levels(levels: pd.DataFrame, min_days: int) -> tuple[float | None, int]:
    r = _pct_returns(levels)
    if len(r) < min_days:
        return None, len(r)
    cov = np.cov(r["etf"].values, r["und"].values)
    var_u = cov[1, 1]
    if var_u < 1e-12:
        return None, len(r)
    return round(float(cov[0, 1] / var_u), 6), len(r)


def beta_ols_tail(levels: pd.DataFrame, tail_days: int, min_days: int) -> tuple[float | None, int]:
    """Simple-return OLS using only the last *tail_days* overlapping closes (or all if shorter)."""
    if len(levels) < min_days + 2:
        return None, len(levels)
    span = min(int(tail_days), len(levels))
    sub = levels.iloc[-span:]
    return beta_ols_simple_on_levels(sub, min_days=min_days)


def beta_roll_median_simple(levels: pd.DataFrame, window: int, min_days: int) -> tuple[float | None, int]:
    """Median of rolling window OLS betas (simple returns), windows that satisfy min_days."""
    r = _pct_returns(levels)
    if len(r) < window:
        return None, len(r)
    vals: list[float] = []
    for i in range(window, len(r) + 1):
        sub = r.iloc[i - window : i]
        if len(sub) < min_days:
            continue
        cov = np.cov(sub["etf"].values, sub["und"].values)
        vu = cov[1, 1]
        if vu < 1e-12:
            continue
        vals.append(float(cov[0, 1] / vu))
    if len(vals) < 5:
        return None, len(vals)
    return round(float(np.median(vals)), 6), len(vals)


def beta_weekly_simple(levels: pd.DataFrame, min_weeks: int) -> tuple[float | None, int]:
    """Friday-to-Friday levels, then simple weekly returns; OLS on weeks."""
    if not isinstance(levels.index, pd.DatetimeIndex):
        levels = levels.copy()
        levels.index = pd.to_datetime(levels.index)
    w = levels.resample("W-FRI").last().dropna()
    r = w.pct_change().dropna()
    if len(r) < min_weeks:
        return None, len(r)
    cov = np.cov(r["etf"].values, r["und"].values)
    vu = cov[1, 1]
    if vu < 1e-12:
        return None, len(r)
    return round(float(cov[0, 1] / vu), 6), len(r)


def shrinkage_toward_leverage(ols_beta: float | None, leverage: float, w_ols: float) -> float | None:
    """w_ols * beta_ols + (1-w_ols) * leverage (clip to finite)."""
    if ols_beta is None or not np.isfinite(ols_beta):
        return None
    w = float(np.clip(w_ols, 0.0, 1.0))
    return round(w * float(ols_beta) + (1.0 - w) * float(leverage), 6)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare beta estimation methods for ETF vs underlying")
    ap.add_argument(
        "--etf",
        nargs="+",
        default=["ORCX", "HIMZ", "MSTX", "NVDL", "FBYY"],
        help="ETF tickers (underlyings from etf_screened_today.csv if present, else ETF=underlying guess fails)",
    )
    ap.add_argument("--lookback", default="2y", help="Yahoo range string (default 2y)")
    ap.add_argument("--min-days", type=int, default=60, help="Min overlapping return days for full-sample OLS")
    ap.add_argument("--tail-days", type=int, default=252, help="Trailing calendar span for tail OLS")
    ap.add_argument("--roll-window", type=int, default=60, help="Rolling window (trading days) for median beta")
    ap.add_argument("--min-weeks", type=int, default=52, help="Min weeks for weekly OLS")
    ap.add_argument(
        "--screened",
        type=Path,
        default=_ROOT / "data" / "etf_screened_today.csv",
        help="CSV for ETF→underlying map + optional screened Beta column",
    )
    ap.add_argument("--leverage", type=float, default=2.0, help="Listed leverage for shrinkage column (default 2)")
    ap.add_argument("--shrink-w", type=float, default=0.5, help="Weight on OLS in shrink toward leverage (0..1)")
    ap.add_argument("--out", type=Path, default=None, help="Write CSV (default: print only)")
    args = ap.parse_args()

    etf_list = [_norm_sym(x) for x in args.etf]
    und_map: dict[str, str] = {}
    screened_beta: dict[str, float | None] = {}
    if args.screened.exists():
        und_map = load_etf_to_under_map(args.screened)
        sc = pd.read_csv(args.screened)
        cols = {c.lower(): c for c in sc.columns}
        etf_c = cols.get("etf") or cols.get("symbol")
        beta_c = cols.get("beta")
        if etf_c and beta_c:
            for _, row in sc.iterrows():
                sym = _norm_sym(str(row[etf_c]))
                screened_beta[sym] = float(row[beta_c]) if pd.notna(row.get(beta_c)) else None

    missing_under: list[str] = []
    pairs: list[tuple[str, str]] = []
    for etf in etf_list:
        u = und_map.get(etf)
        if not u:
            missing_under.append(etf)
            continue
        pairs.append((etf, _norm_sym(u)))

    if missing_under:
        print(
            "[WARN] No underlying in screened CSV for: "
            + ", ".join(missing_under)
            + " — skipped.",
            file=sys.stderr,
        )
    if not pairs:
        print("No valid pairs.", file=sys.stderr)
        return 1

    tickers = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    tr_map = download_all_tr_series(tickers, period=args.lookback, max_workers=8)

    rows: list[dict[str, object]] = []
    for etf, und in pairs:
        s_e = tr_map.get(etf)
        s_u = tr_map.get(und)
        if s_e is None or s_u is None or s_e.empty or s_u.empty:
            rows.append(
                {
                    "ETF": etf,
                    "Underlying": und,
                    "error": "missing_tr",
                }
            )
            continue

        lv = _aligned_levels(s_e, s_u)
        b_default, n_def = compute_beta_ols(s_e, s_u, min_days=args.min_days)
        b_simple_tbl, n_simple = beta_ols_simple_on_levels(lv, args.min_days)
        b_log, n_log = beta_ols_log(lv, args.min_days)
        b_tail, n_tail = beta_ols_tail(lv, tail_days=args.tail_days, min_days=min(60, args.tail_days - 5))
        b_roll, n_roll = beta_roll_median_simple(lv, window=args.roll_window, min_days=args.min_days)
        b_week, n_week = beta_weekly_simple(lv, min_weeks=args.min_weeks)
        b_shrink = shrinkage_toward_leverage(b_simple_tbl, args.leverage, args.shrink_w)

        rows.append(
            {
                "ETF": etf,
                "Underlying": und,
                "n_level_days": int(len(lv)),
                "screened_Beta_csv": screened_beta.get(etf),
                "ols_simple_default": b_default,
                "n_obs_default": n_def,
                "ols_simple_aligned": b_simple_tbl,
                "n_obs_simple_aligned": n_simple,
                "ols_log_returns": b_log,
                "n_obs_log": n_log,
                f"ols_simple_last_{args.tail_days}d": b_tail,
                "n_obs_tail": n_tail,
                f"roll_median_{args.roll_window}d_simple": b_roll,
                "n_roll_windows": n_roll,
                "weekly_simple_FRI": b_week,
                "n_weeks": n_week,
                "listed_leverage_assumption": args.leverage,
                f"shrink_{args.shrink_w}_ols_{1 - args.shrink_w}_lev": b_shrink,
            }
        )

    out_df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(out_df.to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nWrote: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
