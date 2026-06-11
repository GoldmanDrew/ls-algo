"""Per-stock return / vol / drawdown distributions for Bucket 4 pairs + borrow & t-costs.

Uses:
  * ls-algo data/runs/<date>/accounting/bucket4_pairs.csv  (universe)
  * etf-dashboard data/etf_metrics_daily.csv             (prices)
  * etf-dashboard data/etf_screened_today.csv            (borrow quotes)
  * optional accounting pnl_bucket_4_by_pair.csv         (realized borrow fees YTD)

Pair P&L model (daily, short-favorable positive):
  d_t = |delta| * r_underlying - r_etf   (log returns)

Transaction costs (notebook defaults):
  fee_bps=0, slippage_bps=20 per rebalance leg; estimate at 5d rebalance cadence.

Outputs:
  notebooks/output/bucket4_per_stock_stats.csv
  notebooks/output/bucket4_per_stock_distributions.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252
START_SIM = "2025-10-07"
SLIPPAGE_BPS = 20.0
FEE_BPS = 0.0
REBAL_BLOCK = 5


def norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["etf"] = df["etf"].map(norm_sym)
    df["underlying"] = df["underlying"].map(norm_sym)
    return df


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
    df["px"] = df["etf_adj_close"].where(df["etf_adj_close"] > 0)
    df["px"] = df["px"].fillna(df["close_price"].where(df["close_price"] > 0))
    df["px"] = df["px"].fillna(df["nav"].where(df["nav"] > 0))
    return df


def price_series(metrics: pd.DataFrame, sym: str) -> pd.Series:
    sub = metrics[metrics["ticker"] == sym].dropna(subset=["date", "px"])
    sub = sub.drop_duplicates("date").sort_values("date")
    return sub.set_index("date")["px"]


def und_series_from_etf(metrics: pd.DataFrame, etf: str) -> pd.Series:
    """Underlying close is stored on each ETF row in etf_metrics_daily."""
    sub = metrics[metrics["ticker"] == etf].dropna(subset=["date", "underlying_adj_close"])
    sub = sub.drop_duplicates("date").sort_values("date")
    u = sub.set_index("date")["underlying_adj_close"]
    return u[u > 0]


def dist_stats(r: pd.Series) -> dict:
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 5:
        return {k: np.nan for k in (
            "n_days", "ret_mean_d", "ret_std_d", "ret_p05", "ret_p25", "ret_p50",
            "ret_p75", "ret_p95", "vol_ann", "edge_ann", "max_dd",
        )}
    eq = np.exp(r.cumsum())  # r is log-return / log-drag
    dd = eq / eq.cummax() - 1
    span = max((r.index[-1] - r.index[0]).days, 1)
    years = span / 365.25
    # Annualized log-edge (same convention as gross_decay_annual / study)
    edge_ann = float(r.mean() * TRADING_DAYS)
    return {
        "n_days": int(len(r)),
        "ret_mean_d": float(r.mean()),
        "ret_std_d": float(r.std(ddof=1)),
        "ret_p05": float(r.quantile(0.05)),
        "ret_p25": float(r.quantile(0.25)),
        "ret_p50": float(r.quantile(0.50)),
        "ret_p75": float(r.quantile(0.75)),
        "ret_p95": float(r.quantile(0.95)),
        "vol_ann": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "edge_ann": edge_ann,
        "max_dd": float(dd.min()),
    }


def tcost_annual(n_days: int, *, block: int = REBAL_BLOCK, slip_bps: float = SLIPPAGE_BPS) -> dict:
    """Rough annualized slippage drag from restriking both legs each rebalance."""
    if n_days < block:
        return {"n_rebal": 0, "slippage_ann": np.nan, "fee_ann": 0.0}
    n_rebal = max(1, n_days // block)
    per_rebal = 2.0 * (slip_bps / 10000.0)  # two legs, full notional
    total = n_rebal * per_rebal
    ann = total * TRADING_DAYS / n_days
    return {"n_rebal": n_rebal, "slippage_ann": ann, "fee_ann": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--pairs", type=Path,
                    default=repo / "data/runs/2026-06-03/accounting/bucket4_pairs.csv")
    ap.add_argument("--metrics", type=Path,
                    default=repo.parent / "Levered ETFs/etf-dashboard/data/etf_metrics_daily.csv")
    ap.add_argument("--screened", type=Path,
                    default=repo.parent / "Levered ETFs/etf-dashboard/data/etf_screened_today.csv")
    ap.add_argument("--pnl-pair", type=Path,
                    default=repo / "data/runs/2026-06-03/accounting/pnl_bucket_4_by_pair.csv")
    ap.add_argument("--outdir", type=Path, default=repo / "notebooks/output")
    ap.add_argument("--start", default=START_SIM)
    args = ap.parse_args()

    pairs = load_pairs(args.pairs)
    metrics = load_metrics(args.metrics)
    screened = pd.read_csv(args.screened)
    screened["ETF"] = screened["ETF"].astype(str).map(norm_sym)
    borrow_map = {}
    for _, row in screened.iterrows():
        e = row["ETF"]
        b = row.get("borrow_avg_annual", row.get("borrow_current", row.get("borrow_fee_annual", np.nan)))
        if pd.notna(b):
            borrow_map[e] = float(b)

    acct = None
    if args.pnl_pair.exists():
        acct = pd.read_csv(args.pnl_pair)
        acct["etf"] = acct["etf"].map(norm_sym)

    rows = []
    pair_rets: dict[str, pd.Series] = {}
    start = pd.Timestamp(args.start)

    for _, pr in pairs.iterrows():
        etf, und = pr["etf"], pr["underlying"]
        beta = abs(float(pr["delta"]))
        h = float(pr.get("partial_hedge_ratio", 0.75))

        pe = price_series(metrics, etf)
        pu = und_series_from_etf(metrics, etf)
        if pe.empty or pu.empty:
            continue
        cal = pe.index.intersection(pu.index)
        cal = cal[cal >= start]
        if len(cal) < 30:
            continue
        pe, pu = pe.reindex(cal).ffill(), pu.reindex(cal).ffill()
        r_e = np.log(pe / pe.shift(1)).iloc[1:]
        r_u = np.log(pu / pu.shift(1)).iloc[1:]
        r_pair = beta * r_u - r_e
        r_pair = r_pair.replace([np.inf, -np.inf], np.nan).dropna()
        r_pair.name = f"{etf}/{und}"
        pair_rets[r_pair.name] = r_pair

        st_pair = dist_stats(r_pair)
        st_etf = dist_stats(r_e)
        tc = tcost_annual(st_pair["n_days"])
        borrow = borrow_map.get(etf, np.nan)
        borrow_drag_ann = borrow if np.isfinite(borrow) else np.nan

        row = {
            "pair": f"{etf}/{und}",
            "etf": etf,
            "underlying": und,
            "delta": float(pr["delta"]),
            "hedge_beta": beta,
            "partial_h": h,
            "borrow_annual": borrow,
            "borrow_drag_ann": borrow_drag_ann,
            "slippage_ann_est": tc["slippage_ann"],
            "fee_ann_est": tc["fee_ann"],
            "n_rebal_est_5d": tc["n_rebal"],
            "net_edge_ann_est": (
                st_pair["edge_ann"] - borrow_drag_ann - tc["slippage_ann"]
                if np.isfinite(st_pair["edge_ann"]) and np.isfinite(borrow_drag_ann)
                else np.nan
            ),
            **{f"pair_{k}": v for k, v in st_pair.items()},
            **{f"etf_{k}": v for k, v in st_etf.items()},
        }
        if acct is not None:
            m = acct[acct["etf"] == etf]
            if len(m):
                row["acct_borrow_fees"] = float(m["borrow_fees"].iloc[0])
                row["acct_total_pnl"] = float(m["total_pnl"].iloc[0])
        rows.append(row)

    if not rows:
        print("No pairs with sufficient metrics history.", file=__import__("sys").stderr)
        return 1
    out = pd.DataFrame(rows).sort_values("pair_edge_ann", ascending=False)
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "bucket4_per_stock_stats.csv"
    out.to_csv(csv_path, index=False)

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    valid = out.dropna(subset=["pair_edge_ann"])
    if not valid.empty:
        axes[0, 0].barh(valid["pair"], valid["pair_edge_ann"], color="#4C78A8")
        axes[0, 0].axvline(0, color="#888", lw=0.8)
        axes[0, 0].set_title("Pair gross edge ann. (mean daily log-drag x 252)")
        axes[0, 0].set_xlabel("annualized")
        axes[0, 1].barh(valid["pair"], valid["pair_vol_ann"], color="#F58518")
        axes[0, 1].set_title("Pair realized vol (ann.)")
        axes[1, 0].barh(valid["pair"], valid["pair_max_dd"], color="#E45756")
        axes[1, 0].set_title("Pair max drawdown")
        sc = axes[1, 1].scatter(
            valid["borrow_annual"] * 100, valid["pair_edge_ann"] * 100,
            alpha=0.7, c=valid["slippage_ann_est"] * 100, cmap="viridis",
        )
        axes[1, 1].set_xlabel("borrow % ann.")
        axes[1, 1].set_ylabel("pair edge % ann.")
        axes[1, 1].set_title("Edge vs borrow (color=est. slippage % ann.)")
        plt.colorbar(sc, ax=axes[1, 1], label="slippage % ann.")
    fig.tight_layout()
    fig.savefig(args.outdir / "bucket4_per_stock_overview.png", dpi=120)
    plt.close(fig)

    if pair_rets:
        fig, ax = plt.subplots(figsize=(12, 5))
        data = [pair_rets[k].dropna().values for k in sorted(pair_rets)]
        labels = [k for k in sorted(pair_rets)]
        ax.boxplot(data, labels=labels, vert=False, showfliers=False)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlabel("daily pair return (short-favorable +)")
        ax.set_title(f"Daily return distributions | start={args.start}")
        fig.tight_layout()
        fig.savefig(args.outdir / "bucket4_per_stock_distributions.png", dpi=110)
        plt.close(fig)

    # --- console summary ---
    show_cols = [
        "pair", "pair_edge_ann", "pair_vol_ann", "pair_max_dd",
        "pair_ret_p05", "pair_ret_p50", "pair_ret_p95",
        "borrow_annual", "slippage_ann_est", "net_edge_ann_est",
        "acct_borrow_fees", "acct_total_pnl",
    ]
    show_cols = [c for c in show_cols if c in out.columns]
    print(f"Bucket 4 per-stock stats | n={len(out)} pairs | start={args.start}")
    print(f"TC assumptions: slippage={SLIPPAGE_BPS}bps, fee={FEE_BPS}bps, rebalance every {REBAL_BLOCK}d (estimate)")
    print(f"Wrote {csv_path}")
    print()
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.width", 200)
    fmt = {
        "pair_edge_ann": "{:.2%}", "pair_vol_ann": "{:.2%}", "pair_max_dd": "{:.2%}",
        "pair_ret_p05": "{:.3%}", "pair_ret_p50": "{:.3%}", "pair_ret_p95": "{:.3%}",
        "borrow_annual": "{:.2%}", "slippage_ann_est": "{:.2%}", "net_edge_ann_est": "{:.2%}",
    }
    disp = out[show_cols].copy()
    for c, f in fmt.items():
        if c in disp.columns:
            disp[c] = disp[c].map(lambda x, ff=f: ff.format(x) if pd.notna(x) else "")
    print(disp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
