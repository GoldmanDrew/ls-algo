#!/usr/bin/env python3
"""
plot_proposed_trades.py — Visualise bucket-level proposed trades for a run date.

Also appends **screened** ETFs with IBKR ``shares_available``≈0 (or ``exclude_no_shares``) that
do not appear in the sized plan (often purgatory / no-locate): they get a **proxy** hatched
optimal bar (median structural gross in the bucket × edge rank) so you can see where the
sleeve would like to be if inventory existed — see plot footnote †. Disable with
``--skip-no-ibkr-proxy``.

Outputs:
  - proposed_trades_bucket_1_plot.png   (beta > 1.5, stock sleeves only)
  - proposed_trades_bucket_2_plot.png   (yieldboost sleeve)
  - proposed_trades_bucket_4_plot.png   (inverse_decay_bucket4 sleeve: short ETF + hedge)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
B4_SLEEVE = "inverse_decay_bucket4"
BETA_FLOOR_PLOT = 0.1
B4_PARTIAL_HEDGE_DEFAULT = 1.0


def _to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)


def load_proposed(run_date: str) -> pd.DataFrame:
    path = RUNS_DIR / run_date / "proposed_trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"proposed_trades.csv not found for {run_date}: {path}")
    df = pd.read_csv(path)

    numeric_cols = (
        "long_usd",
        "short_usd",
        "borrow_current",
        "net_decay_annual",
        "Beta",
        # Optional optimal-target columns (added by generate_trade_plan dual pipeline). When
        # absent (older runs), these default to NaN; render falls back to executable-only bars.
        "optimal_long_usd",
        "optimal_short_usd",
        "optimal_gross_target_usd",
        "gross_target_usd",
        "liquidity_gap_usd",
        "executable_pct_of_optimal",
        "shares_available",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "purgatory" in df.columns:
        df["purgatory"] = _to_bool_series(df["purgatory"])
    else:
        df["purgatory"] = False

    return df


def load_screened(run_date: str) -> pd.DataFrame | None:
    """Same-day screened universe (IBKR ``shares_available`` / ``exclude_no_shares``)."""
    for path in (RUNS_DIR / run_date / "etf_screened_today.csv", PROJECT_ROOT / "data" / "etf_screened_today.csv"):
        if path.is_file():
            return pd.read_csv(path)
    return None


def _no_ibkr_shares_mask(df: pd.DataFrame) -> pd.Series:
    """True when FTP/screener reports no shortable inventory (or ``exclude_no_shares`` is set)."""
    sh = pd.to_numeric(df.get("shares_available"), errors="coerce")
    no_sh = sh.isna() | (sh.fillna(0.0) <= 0.0)
    if "exclude_no_shares" in df.columns:
        ex = _to_bool_series(df["exclude_no_shares"])
        return no_sh | ex
    return no_sh


def _median_positive(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").replace(0.0, np.nan).dropna()
    v = v[v > 0]
    if v.empty:
        return float("nan")
    return float(v.median())


def _bucket_median_structural_gross(bucket_df: pd.DataFrame) -> float:
    """Typical structural gross $ in this bucket (for proxy scaling of no-locate names)."""
    if bucket_df is None or bucket_df.empty:
        return 150_000.0
    if "optimal_gross_target_usd" in bucket_df.columns:
        m = _median_positive(bucket_df["optimal_gross_target_usd"])
        if np.isfinite(m) and m > 0:
            return m
    ol = pd.to_numeric(bucket_df.get("optimal_long_usd"), errors="coerce").fillna(0.0).abs()
    os_ = pd.to_numeric(bucket_df.get("optimal_short_usd"), errors="coerce").fillna(0.0).abs()
    combo = ol + os_
    m = _median_positive(combo)
    if np.isfinite(m) and m > 0:
        return m
    return 150_000.0


def _edge_signal(row: pd.Series) -> float:
    ne = pd.to_numeric(row.get("net_edge_p50_annual"), errors="coerce")
    if np.isfinite(ne) and ne > 0:
        return float(ne)
    nd = pd.to_numeric(row.get("net_decay_annual"), errors="coerce")
    if np.isfinite(nd):
        return max(float(nd), 1e-6)
    return 0.05


def _legs_from_gross_stock(*, gross: float, beta: float) -> tuple[float, float]:
    """Match stock-sleeve split: long>0, short<0 (same as ``_populate_long_short_columns`` non-B4)."""
    ba = max(abs(float(beta)), BETA_FLOOR_PLOT)
    hr = 1.0 / ba
    lg = gross / (1.0 + hr)
    sh = -(hr * lg)
    return float(lg), float(sh)


def _legs_from_gross_b4(*, gross: float, beta: float, partial: float = B4_PARTIAL_HEDGE_DEFAULT) -> tuple[float, float]:
    """Default B4 convention from generate_trade_plan (both legs negative USD)."""
    ba = max(abs(float(beta)), BETA_FLOOR_PLOT)
    sh = -float(gross)
    lg = -float(partial) * ba * float(gross)
    return float(lg), float(sh)


def append_no_ibkr_share_screened_rows(
    bucket_df: pd.DataFrame,
    screened: pd.DataFrame | None,
    *,
    bucket: str,
) -> pd.DataFrame:
    """Append screened names with IBKR ``shares_available``≈0 that are missing from the bucket.

    Those rows are often **purgatory** or otherwise absent from the sized book, so
    ``proposed_trades.csv`` carries no structural ``optimal_*`` dollars. We still plot a
    **proxy** optimal (hatched bar): median structural gross in this bucket × (edge signal /
    median edge among appended rows), capped at 2× the bucket median — strictly a visualization
    aid until ``generate_trade_plan`` emits true optimal targets for no-locate names.
    """
    if screened is None or screened.empty:
        return bucket_df
    need = {"ETF", "Underlying", "Beta"}
    if not need.issubset(set(screened.columns)):
        return bucket_df

    sc = screened.copy()
    sc["ETF"] = sc["ETF"].astype(str).str.strip()
    sc["Underlying"] = sc["Underlying"].astype(str).str.strip()
    beta = pd.to_numeric(sc["Beta"], errors="coerce")
    beta_abs = beta.abs()
    yb = sc.get("is_yieldboost", False)
    if hasattr(yb, "fillna"):
        yb = yb.fillna(False).astype(bool)
    else:
        yb = pd.Series(False, index=sc.index)
    if "inverse_shortable" in sc.columns:
        inv_ok = sc["inverse_shortable"].fillna(False).astype(bool)
    else:
        inv_ok = beta < 0

    if bucket == "b2":
        bucket_mask = yb & (beta > 0)
    elif bucket == "b1":
        bucket_mask = (~yb) & (beta_abs > 1.5) & (beta > 0)
    elif bucket == "b4":
        bucket_mask = (~yb) & (beta < 0) & inv_ok
    else:
        return bucket_df

    no_loc = _no_ibkr_shares_mask(sc) & bucket_mask
    if not bool(no_loc.any()):
        return bucket_df

    have = set()
    if not bucket_df.empty and "ETF" in bucket_df.columns and "Underlying" in bucket_df.columns:
        for _, r in bucket_df.iterrows():
            have.add((str(r["ETF"]).strip(), str(r["Underlying"]).strip()))

    med_g = _bucket_median_structural_gross(bucket_df)
    extras: list[dict[str, object]] = []
    edge_list = [_edge_signal(sc.loc[i]) for i in sc.index[no_loc]]
    med_edge = float(np.median(edge_list)) if edge_list else 1.0
    med_edge = max(med_edge, 1e-9)

    for idx in sc.index[no_loc]:
        row = sc.loc[idx]
        key = (str(row["ETF"]).strip(), str(row["Underlying"]).strip())
        if key in have:
            continue
        edge = _edge_signal(row)
        proxy_gross = med_g * (edge / med_edge)
        proxy_gross = float(np.clip(proxy_gross, 5_000.0, 2.0 * med_g))
        b = float(row["Beta"])
        if bucket == "b4":
            olg, osg = _legs_from_gross_b4(gross=proxy_gross, beta=b)
            sleeve = B4_SLEEVE
        else:
            olg, osg = _legs_from_gross_stock(gross=proxy_gross, beta=b)
            sleeve = "yieldboost" if bucket == "b2" else "core_leveraged"
        extras.append(
            {
                "ETF": key[0],
                "Underlying": key[1],
                "Beta": b,
                "sleeve": sleeve,
                "long_usd": 0.0,
                "short_usd": 0.0,
                "gross_target_usd": 0.0,
                "optimal_long_usd": olg,
                "optimal_short_usd": osg,
                "optimal_gross_target_usd": proxy_gross,
                "liquidity_gap_usd": proxy_gross,
                "executable_pct_of_optimal": 0.0,
                "shares_available": float(pd.to_numeric(row.get("shares_available"), errors="coerce") or 0.0),
                "_no_ibkr_shares_proxy": True,
            }
        )
        have.add(key)

    if not extras:
        return bucket_df
    add = pd.DataFrame(extras)
    add["_no_ibkr_shares_proxy"] = np.ones(len(add), dtype=bool)
    if bucket_df.empty:
        return add
    bucket_df = bucket_df.copy()
    if "_no_ibkr_shares_proxy" not in bucket_df.columns:
        bucket_df["_no_ibkr_shares_proxy"] = np.zeros(len(bucket_df), dtype=bool)
    out = pd.concat([bucket_df, add], axis=0, ignore_index=True)
    out.loc[out["_no_ibkr_shares_proxy"].isna(), "_no_ibkr_shares_proxy"] = False
    out["_no_ibkr_shares_proxy"] = out["_no_ibkr_shares_proxy"].astype(bool)
    return out


def active_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that are either being traded **today** (executable nonzero) OR have a
    standing **optimal** position (so we render the hatched bar even when day-liquidity
    pushed the executable to zero)."""
    long_exe = df["long_usd"].fillna(0).abs() > 1
    short_exe = df["short_usd"].fillna(0).abs() > 1
    long_opt = df.get("optimal_long_usd", pd.Series(0.0, index=df.index)).fillna(0).abs() > 1
    short_opt = df.get("optimal_short_usd", pd.Series(0.0, index=df.index)).fillna(0).abs() > 1
    return df[
        (df["purgatory"] == False)  # noqa: E712
        & (long_exe | short_exe | long_opt | short_opt)
    ].copy()


def split_buckets(df_active: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "sleeve" not in df_active.columns:
        stock = df_active.copy()
        b4 = df_active.iloc[0:0].copy()
        b = pd.to_numeric(stock["Beta"], errors="coerce")
        b1 = stock[b > 1.5].copy()
        b2 = stock[(b > 0) & (b <= 1.5)].copy()
        return b1, b2, b4

    is_b4 = df_active["sleeve"].astype(str).eq(B4_SLEEVE)
    stock = df_active[~is_b4].copy()
    b4 = df_active[is_b4].copy()
    slv = stock["sleeve"].astype(str)
    b1 = stock[slv.eq("core_leveraged")].copy()
    b2 = stock[slv.eq("yieldboost")].copy()
    return b1, b2, b4


def _has_optimal(df: pd.DataFrame) -> bool:
    if "optimal_long_usd" not in df.columns or "optimal_short_usd" not in df.columns:
        return False
    o_l = pd.to_numeric(df["optimal_long_usd"], errors="coerce").abs().fillna(0.0)
    o_s = pd.to_numeric(df["optimal_short_usd"], errors="coerce").abs().fillna(0.0)
    return bool((o_l + o_s).sum() > 1.0)


def plot_allocation(ax: plt.Axes, df: pd.DataFrame, title: str, *, bucket4: bool = False) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No proposed trades in this bucket", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_axis_off()
        return

    show_optimal = _has_optimal(df)
    if bucket4:
        # Both legs stored as negative USD; plot magnitudes symmetric about zero like B1/B2.
        long_usd = -pd.to_numeric(df["long_usd"], errors="coerce").fillna(0)
        short_usd = pd.to_numeric(df["short_usd"], errors="coerce").fillna(0)
        opt_long_usd = (
            -pd.to_numeric(df["optimal_long_usd"], errors="coerce").fillna(0) if show_optimal
            else pd.Series(0.0, index=df.index)
        )
        opt_short_usd = (
            pd.to_numeric(df["optimal_short_usd"], errors="coerce").fillna(0) if show_optimal
            else pd.Series(0.0, index=df.index)
        )
        df = df.assign(_plot_long=long_usd, _plot_short=short_usd,
                       _plot_opt_long=opt_long_usd, _plot_opt_short=opt_short_usd)
        sort_col = "_plot_opt_long" if show_optimal else "_plot_long"
        legend_long = mpatches.Patch(color="steelblue", alpha=0.85, label="Executable underlying hedge")
        legend_short = mpatches.Patch(color="tomato", alpha=0.85, label="Executable inverse ETF")
    else:
        df = df.assign(
            _plot_long=pd.to_numeric(df["long_usd"], errors="coerce").fillna(0),
            _plot_short=pd.to_numeric(df["short_usd"], errors="coerce").fillna(0),
            _plot_opt_long=pd.to_numeric(df.get("optimal_long_usd", 0.0), errors="coerce").fillna(0),
            _plot_opt_short=pd.to_numeric(df.get("optimal_short_usd", 0.0), errors="coerce").fillna(0),
        )
        sort_col = "_plot_opt_long" if show_optimal else "long_usd"
        legend_long = mpatches.Patch(color="steelblue", alpha=0.85, label="Executable Underlying (long)")
        legend_short = mpatches.Patch(color="tomato", alpha=0.85, label="Executable ETF (short)")
    df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    xmax = max(
        df["_plot_long"].abs().max(),
        df["_plot_short"].abs().max(),
        df["_plot_opt_long"].abs().max(),
        df["_plot_opt_short"].abs().max(),
        1,
    )
    pad = xmax * 0.012

    if "_no_ibkr_shares_proxy" not in df.columns:
        df["_no_ibkr_shares_proxy"] = False
    else:
        df.loc[df["_no_ibkr_shares_proxy"].isna(), "_no_ibkr_shares_proxy"] = False
        df["_no_ibkr_shares_proxy"] = df["_no_ibkr_shares_proxy"].astype(bool)
    n_proxy = int(df["_no_ibkr_shares_proxy"].sum())

    if show_optimal:
        for yi, (_, row) in enumerate(df.iterrows()):
            pr = bool(row["_no_ibkr_shares_proxy"])
            h_pat = "xxx" if pr else "///"
            ec_l, ec_s = (("#1b5e20", "#b71c1c") if pr else ("steelblue", "tomato"))
            lw = 1.35 if pr else 1.0
            alpha_h = 0.75 if pr else 0.55
            ax.barh(
                yi,
                row["_plot_opt_long"],
                color="none",
                edgecolor=ec_l,
                linewidth=lw,
                hatch=h_pat,
                height=0.85,
                alpha=alpha_h,
            )
            ax.barh(
                yi,
                row["_plot_opt_short"],
                color="none",
                edgecolor=ec_s,
                linewidth=lw,
                hatch=h_pat,
                height=0.85,
                alpha=alpha_h,
            )
        ax.barh(y, df["_plot_long"], color="steelblue", alpha=0.95, height=0.55)
        ax.barh(y, df["_plot_short"], color="tomato", alpha=0.95, height=0.55)
    else:
        ax.barh(y, df["_plot_long"], color="steelblue", alpha=0.85, height=0.7)
        ax.barh(y, df["_plot_short"], color="tomato", alpha=0.85, height=0.7)

    for i, row in df.iterrows():
        lu = row["_plot_long"]
        su = row["_plot_short"]
        opt_lu = row.get("_plot_opt_long", 0.0)
        opt_su = row.get("_plot_opt_short", 0.0)
        # Use the wider (optimal) extent for label placement so labels don't collide with the
        # outer hatched bar. Falls back to executable when optimal is unavailable.
        long_extent = opt_lu if show_optimal and abs(opt_lu) > abs(lu) else lu
        short_extent = opt_su if show_optimal and abs(opt_su) > abs(su) else su
        if bucket4:
            lu_abs = abs(float(row["long_usd"])) if pd.notna(row["long_usd"]) else 0.0
            su_abs = abs(float(row["short_usd"])) if pd.notna(row["short_usd"]) else 0.0
            if lu_abs > 1 or abs(opt_lu) > 1:
                ax.text(long_extent + pad, i, str(row.get("Underlying", "")),
                        ha="left", va="center", fontsize=6.5, color="steelblue")
            if su_abs > 1 or abs(opt_su) > 1:
                ax.text(short_extent - pad, i, str(row.get("ETF", "")),
                        ha="right", va="center", fontsize=6.5, color="tomato")
        else:
            if abs(row["long_usd"]) > 1 or abs(opt_lu) > 1:
                ax.text(long_extent + pad, i, str(row.get("Underlying", "")),
                        ha="left", va="center", fontsize=6.5, color="steelblue")
            if abs(row["short_usd"]) > 1 or abs(opt_su) > 1:
                ax.text(short_extent - pad, i, str(row.get("ETF", "")),
                        ha="right", va="center", fontsize=6.5, color="tomato")

    labels = []
    for _, r in df.iterrows():
        base = f"{r.ETF} -> {r.Underlying}"
        gap = float(r.get("liquidity_gap_usd", 0.0) or 0.0)
        if bool(r.get("_no_ibkr_shares_proxy", False)):
            og = float(r.get("optimal_gross_target_usd", 0.0) or 0.0)
            base += f"  [IBKR avail≈0; proxy opt ${og/1000:,.0f}k*]"
        elif show_optimal and abs(gap) > 1000:
            base += f"  [gap ${gap/1000:,.0f}k]"
        labels.append(base)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Notional (USD)", fontsize=9)
    ax.set_title(title, fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    handles = [legend_long, legend_short]
    if show_optimal:
        handles.extend([
            mpatches.Patch(facecolor="none", edgecolor="steelblue", hatch="///",
                           label="Optimal Underlying (structural-only)"),
            mpatches.Patch(facecolor="none", edgecolor="tomato", hatch="///",
                           label="Optimal ETF (structural-only)"),
        ])
    if n_proxy > 0:
        handles.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="#1b5e20",
                hatch="xxx",
                label="Proxy optimal (IBKR locate≈0; scaled†)",
            )
        )
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    if show_optimal:
        # Bottom annotation: total executable vs optimal gross (sum of |long| + |short|) for the bucket.
        exe_total = float(df["_plot_long"].abs().sum() + df["_plot_short"].abs().sum())
        opt_total = float(df["_plot_opt_long"].abs().sum() + df["_plot_opt_short"].abs().sum())
        pct = 100.0 * exe_total / max(opt_total, 1e-9)
        foot = (
            f"Bucket totals — executable=${exe_total/1000:,.0f}k / "
            f"optimal=${opt_total/1000:,.0f}k ({pct:.0f}%)"
        )
        if n_proxy > 0:
            foot += (
                f"  |  † {n_proxy} name(s) with IBKR locate≈0: green/red 'xxx' hatch = **proxy** "
                f"structural scale (median bucket optimal × edge rank); not from live sizing CSV."
            )
        ax.text(
            0.005,
            -0.10,
            foot,
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            color="dimgray",
        )


def plot_decay_score(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No decay data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    d = df.copy()
    d["_score"] = d["net_decay_annual"].fillna(0) * 100
    d = d.sort_values("_score", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    ax.barh(y, d["_score"], color="seagreen", alpha=0.85, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["ETF"].astype(str), fontsize=7.5)
    ax.set_xlabel("Net Decay Score (% annual)", fontsize=9)
    ax.set_title("Net Decay Score", fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def plot_borrow(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No borrow data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    d = df.copy()
    d["_borrow"] = d["borrow_current"].fillna(0) * 100
    d = d.sort_values("_borrow", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    ax.barh(y, d["_borrow"], color="mediumpurple", alpha=0.85, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["ETF"].astype(str), fontsize=7.5)
    ax.set_xlabel("Borrow Rate (% annual)", fontsize=9)
    ax.set_title("Current Borrow Rate by ETF", fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def render_stock_bucket(
    run_date: str,
    bucket_df: pd.DataFrame,
    bucket_label: str,
    out_path: Path,
    *,
    bucket4: bool = False,
) -> Path:
    n = len(bucket_df)
    row_height = 0.22
    panel_h = max(7, n * row_height)
    fig, ax = plt.subplots(1, 1, figsize=(20, panel_h + 1.5))

    # Keep only the proposed position allocation panel.
    subtitle = (
        "Allocation — inverse ETF (short) vs underlying hedge (short)"
        if bucket4
        else "Allocation — ETF short vs Underlying long"
    )
    plot_allocation(ax, bucket_df, f"{bucket_label} {subtitle}", bucket4=bucket4)

    fig.suptitle(f"{bucket_label} Proposed Trades — {run_date} ({n} rows)", fontsize=13, fontweight="bold", y=1.002)
    fig.tight_layout(pad=2.2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_bucket_plots(run_date: str, *, include_no_ibkr_proxy: bool = True) -> list[Path]:
    proposed = load_proposed(run_date)
    act = active_rows(proposed)
    b1, b2, b4 = split_buckets(act)
    screened = load_screened(run_date) if include_no_ibkr_proxy else None
    if include_no_ibkr_proxy:
        b1 = append_no_ibkr_share_screened_rows(b1, screened, bucket="b1")
        b2 = append_no_ibkr_share_screened_rows(b2, screened, bucket="b2")
        b4 = append_no_ibkr_share_screened_rows(b4, screened, bucket="b4")

    out_dir = RUNS_DIR / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_b1 = out_dir / "proposed_trades_bucket_1_plot.png"
    out_b2 = out_dir / "proposed_trades_bucket_2_plot.png"
    out_b4 = out_dir / "proposed_trades_bucket_4_plot.png"

    render_stock_bucket(run_date, b1, "Bucket 1 (core_leveraged)", out_b1)
    render_stock_bucket(run_date, b2, "Bucket 2 (yieldboost sleeve)", out_b2)
    render_stock_bucket(run_date, b4, "Bucket 4 (inverse decay)", out_b4, bucket4=True)

    return [out_b1, out_b2, out_b4]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_date", nargs="?", default=date.today().isoformat(),
                        help="Run date YYYY-MM-DD (default: today)")
    parser.add_argument("--show", action="store_true",
                        help="Retained for compatibility; plotting is saved to files in headless mode.")
    parser.add_argument(
        "--skip-no-ibkr-proxy",
        action="store_true",
        help="Do not append screened ETFs with IBKR shares_available≈0 (no-locate) for proxy optimal bars.",
    )
    args = parser.parse_args()

    try:
        outputs = make_bucket_plots(args.run_date, include_no_ibkr_proxy=not args.skip_no_ibkr_proxy)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    for p in outputs:
        print(f"[OK] Saved -> {p}")
    if args.show:
        print("[INFO] --show requested; script runs headless and saves PNG files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
