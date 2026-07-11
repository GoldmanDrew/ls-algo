"""Bucket 4: backtest returns for every base_days (TR/VCR cadence frequency sweep).

For each pair in bucket4_pairs.csv and each base_days in [1..21]:
  * Build TR/VCR signals (vol_shape_history + underlying recompute, 1-day shift)
  * Rebalance schedule: policy_continuous_interval(base_days, k_tr=+2.25, m_vcr=2.5, cap=21)
  * Engine: run_bucket4_backtest_dynamic_h (20bps slippage, 0 fee, borrow from screener)
  * Hedge h = partial_hedge_ratio (0.75 default from accounting)

Also runs fixed cadences {1,5,10,21}d for reference (labelled base_days = nan, block = N).

Outputs (notebooks/output/):
  bucket4_base_days_returns_long.csv
  bucket4_base_days_cagr_matrix.csv   (pairs x base_days)
  bucket4_base_days_heatmap.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h  # noqa: E402
from scripts.bucket4_vol_shape_signals import (  # noqa: E402
    get_pair_signal,
    load_vol_shape_history,
    policy_continuous_interval,
    rebalance_cadence_stats,
)

TRADING_DAYS = 252
START_SIM = "2025-10-07"
SLIPPAGE_BPS = 20.0
FEE_BPS = 0.0
K_TR = 2.25
M_VCR = 2.5
CAP_DAYS = 21
SIGNAL_WINDOW = 45
BASE_DAYS_RANGE = list(range(1, 22))
FIXED_BLOCKS = [1, 5, 10, 21]


def norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def pair_metrics(bt: pd.DataFrame) -> dict:
    nav = bt["equity"].astype(float)
    rets = nav.pct_change().iloc[1:].dropna()
    span_days = max(int((nav.index[-1] - nav.index[0]).days), 1)
    years = max(span_days / 365.25, 1.0 / 365.25)
    n0, n1 = float(nav.iloc[0]), float(nav.iloc[-1])
    cagr = (n1 / n0) ** (1 / years) - 1.0 if n0 > 1e-9 else np.nan
    vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 2 else np.nan
    dd = nav / nav.cummax() - 1.0
    reb = bt.index[bt["rebalance"].astype(bool)] if "rebalance" in bt.columns else pd.DatetimeIndex([])
    cad = rebalance_cadence_stats(reb)
    return {
        "CAGR": cagr,
        "total_return": n1 / n0 - 1.0 if n0 > 1e-9 else np.nan,
        "vol": vol,
        "max_dd": float(dd.min()),
        "final_equity": n1,
        "borrow_paid": float(bt["borrow_cost"].sum()) if "borrow_cost" in bt.columns else np.nan,
        "slippage_paid": float(bt["rebalance_fee"].sum()) if "rebalance_fee" in bt.columns else np.nan,
        "n_rebalances": cad.get("n_rebalances", np.nan),
        "mean_interval_days": cad.get("mean_interval_days", np.nan),
    }


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        metrics_path,
        usecols=["date", "ticker", "close_price", "nav", "etf_adj_close", "underlying_adj_close"],
        low_memory=False,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].map(norm_sym)
    for c in ("close_price", "nav", "etf_adj_close", "underlying_adj_close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    etf = df["etf_adj_close"].where(df["etf_adj_close"] > 0)
    etf = etf.fillna(df["close_price"].where(df["close_price"] > 0))
    etf = etf.fillna(df["nav"].where(df["nav"] > 0))
    df["etf_px"] = etf
    return df


def build_prices(metrics: pd.DataFrame, etf: str, start: pd.Timestamp) -> pd.DataFrame | None:
    sub = metrics[metrics["ticker"] == etf].dropna(subset=["date", "etf_px", "underlying_adj_close"])
    sub = sub.drop_duplicates("date").sort_values("date")
    # Keep pre-start history for split adjustment continuity, then clip.
    sub = sub[sub["etf_px"] > 0]
    sub = sub[sub["underlying_adj_close"] > 0]
    if len(sub) < 40:
        return None
    a = pd.Series(sub["etf_px"].to_numpy(dtype=float), index=pd.DatetimeIndex(sub["date"]))
    b = pd.Series(sub["underlying_adj_close"].to_numpy(dtype=float), index=a.index)
    try:
        from scripts.pair_price_panel import apply_flex_splits_to_series

        a = apply_flex_splits_to_series(a, etf)
    except Exception:
        pass
    out = pd.DataFrame({"a_px": a.to_numpy(dtype=float), "b_px": b.reindex(a.index).to_numpy(dtype=float)}, index=a.index)
    out = out.dropna()
    out = out.loc[out.index >= start]
    if len(out) < 40:
        return None
    return out


def run_one(
    prices: pd.DataFrame,
    sig: pd.DataFrame,
    *,
    beta_a: float,
    borrow_a: float,
    partial_h: float,
    base_days: float | None,
    fixed_block: int | None,
) -> dict | None:
    cal = prices.index
    h = pd.Series(float(partial_h), index=cal)
    if fixed_block is not None:
        rd = cal[:: max(1, int(fixed_block))]
        mode = f"fixed_{fixed_block}d"
        bd = np.nan
    else:
        rd, _ = policy_continuous_interval(
            cal, sig,
            base_days=float(base_days),
            k_tr=K_TR,
            m_vcr=M_VCR,
            min_interval=1,
            max_interval=CAP_DAYS,
        )
        mode = "tr_vcr_cadence"
        bd = float(base_days)
    rd = pd.DatetimeIndex(rd).intersection(cal)
    if len(rd) == 0:
        rd = pd.DatetimeIndex([cal[0]])
    try:
        bt = run_bucket4_backtest_dynamic_h(
            prices,
            h,
            rd,
            beta_a=float(beta_a),
            beta_b=1.0,
            borrow_a_annual=float(borrow_a),
            fee_bps=FEE_BPS,
            slippage_bps=SLIPPAGE_BPS,
            opt2_h_base=float(partial_h),
        )
    except Exception:
        return None
    if bt is None or bt.empty:
        return None
    m = pair_metrics(bt)
    m["cadence_mode"] = mode
    m["base_days"] = bd
    m["fixed_block"] = fixed_block if fixed_block is not None else np.nan
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=REPO / "data/runs/2026-06-03/accounting/bucket4_pairs.csv")
    ap.add_argument("--metrics", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/etf_metrics_daily.csv")
    ap.add_argument("--vol-shape", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/vol_shape_history.json")
    ap.add_argument("--screened", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/etf_screened_today.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output")
    ap.add_argument("--start", default=START_SIM)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    pairs["etf"] = pairs["etf"].map(norm_sym)
    pairs["underlying"] = pairs["underlying"].map(norm_sym)
    metrics = load_metrics(args.metrics)
    vs_hist = load_vol_shape_history(args.vol_shape)
    screened = pd.read_csv(args.screened)
    screened["ETF"] = screened["ETF"].map(norm_sym)
    borrow_map = {
        r["ETF"]: float(r.get("borrow_current") or r.get("borrow_fee_annual") or np.nan)
        for _, r in screened.iterrows()
        if pd.notna(r.get("borrow_current", r.get("borrow_fee_annual", np.nan)))
    }

    start = pd.Timestamp(args.start)
    rows: list[dict] = []
    keys = list(zip(pairs["etf"], pairs["underlying"], pairs["delta"], pairs["partial_hedge_ratio"]))
    if args.max_pairs > 0:
        keys = keys[: args.max_pairs]

    n_tasks = len(keys) * (len(BASE_DAYS_RANGE) + len(FIXED_BLOCKS))
    done = 0
    for etf, und, delta, ph in keys:
        prices = build_prices(metrics, etf, start)
        if prices is None:
            continue
        sig = get_pair_signal(
            etf, und, prices.index, history=vs_hist,
            underlying_prices=prices["b_px"],
            window=SIGNAL_WINDOW, lookahead_shift=1,
            prefer_underlying_recompute=True, norm_sym=norm_sym,
        )
        beta_a = float(delta)
        borrow_a = float(borrow_map.get(etf, 0.0))
        partial_h = float(ph) if pd.notna(ph) else 0.75
        pair_lab = f"{etf}/{und}"

        for bd in BASE_DAYS_RANGE:
            m = run_one(prices, sig, beta_a=beta_a, borrow_a=borrow_a, partial_h=partial_h, base_days=bd, fixed_block=None)
            done += 1
            if m:
                rows.append({"pair": pair_lab, "etf": etf, "underlying": und, **m})
            if done % 50 == 0:
                print(f"  ... {done}/{n_tasks}", flush=True)

        for blk in FIXED_BLOCKS:
            m = run_one(prices, sig, beta_a=beta_a, borrow_a=borrow_a, partial_h=partial_h, base_days=None, fixed_block=blk)
            done += 1
            if m:
                rows.append({"pair": pair_lab, "etf": etf, "underlying": und, **m})

    if not rows:
        print("No backtests completed.", file=sys.stderr)
        return 1

    long = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    long_path = args.outdir / "bucket4_base_days_returns_long.csv"
    long.to_csv(long_path, index=False)

    cad = long[long["cadence_mode"] == "tr_vcr_cadence"].copy()
    mat = cad.pivot_table(index="pair", columns="base_days", values="CAGR", aggfunc="first")
    mat = mat.reindex(columns=sorted(mat.columns, key=lambda x: float(x)))
    mat_path = args.outdir / "bucket4_base_days_cagr_matrix.csv"
    mat.to_csv(mat_path)

    # equal-weight mean CAGR by base_days
    ew = cad.groupby("base_days")["CAGR"].agg(["mean", "median", "std", "count"]).reset_index()
    ew_path = args.outdir / "bucket4_base_days_ew_summary.csv"
    ew.to_csv(ew_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ew["base_days"], ew["mean"] * 100, "o-", label="equal-weight mean CAGR", lw=2)
    ax.plot(ew["base_days"], ew["median"] * 100, "s--", label="equal-weight median CAGR", lw=1.5)
    ref = long[long["cadence_mode"].str.startswith("fixed_")]
    for blk in FIXED_BLOCKS:
        sub = ref[ref["fixed_block"] == blk]
        if not sub.empty:
            ax.axhline(sub["CAGR"].mean() * 100, ls=":", alpha=0.7, label=f"fixed {int(blk)}d (mean)")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("base_days (PR cadence)")
    ax.set_ylabel("CAGR %")
    ax.set_title(f"Bucket 4 frequency sweep | k_tr={K_TR}, cap={CAP_DAYS}d, slippage={SLIPPAGE_BPS}bps")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "bucket4_base_days_ew_curve.png", dpi=130)
    plt.close(fig)

    if len(mat) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.22 * len(mat))))
        im = ax.imshow(mat.values * 100, aspect="auto", cmap="RdYlGn", vmin=-50, vmax=50)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([str(int(c)) if c == int(c) else str(c) for c in mat.columns], rotation=0)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=7)
        ax.set_xlabel("base_days")
        ax.set_title("CAGR % by pair (TR/VCR cadence backtest)")
        plt.colorbar(im, ax=ax, label="CAGR %")
        fig.tight_layout()
        fig.savefig(args.outdir / "bucket4_base_days_heatmap.png", dpi=120)
        plt.close(fig)

    print(f"Pairs run: {cad['pair'].nunique()} | rows: {len(long)}")
    print(f"Wrote {long_path}")
    print(f"Wrote {mat_path}")
    print(f"Wrote {ew_path}")
    print("\nEqual-weight CAGR by base_days (%):")
    print(ew.assign(mean_pct=lambda x: x["mean"] * 100, median_pct=lambda x: x["median"] * 100)[
        ["base_days", "mean_pct", "median_pct", "count"]
    ].to_string(index=False, formatters={"mean_pct": "{:.2f}", "median_pct": "{:.2f}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
