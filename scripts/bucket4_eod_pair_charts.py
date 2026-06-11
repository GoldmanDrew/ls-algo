"""Per-pair Bucket 4 PnL + hedge-ratio charts for the EOD email.

For each active B4 pair, one chart row with two panels:
  left  — cumulative PnL since accounting inception: pair total (bold),
          ETF short leg, underlying short leg. Leg split is the accounting's
          own allocation: ETF leg from pnl_bucket_4_by_symbol.csv, underlying
          leg = pair total - ETF leg (pnl_bucket_4_by_pair.csv already
          pro-rates shared underlying shorts across pairs).
  right — underlying adj close (left axis) + the LIVE model hedge ratio h
          (right axis): v7 closed form + v9 cross-sectional tilt, computed by
          the production engine (build_h_series), clip band shaded. The
          realized book h on the run date is marked with a dot so model-vs-
          book drift is visible at a glance.

Fail-soft: ``make_b4_pair_pnl_hedge_chart`` returns (None, None) on any data
problem rather than raising — the EOD email must always go out.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_METRICS_CSV = REPO.parent / "Levered ETFs" / "etf-dashboard" / "data" / "etf_metrics_daily.csv"
MIN_ACTIVE_GROSS_USD = 500.0


def _norm(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


# ---------------------------------------------------------------------------
# Accounting history (cumulative per run date)
# ---------------------------------------------------------------------------
def list_run_dates(runs_root: Path) -> list[str]:
    out = []
    for d in sorted(runs_root.iterdir()):
        if d.is_dir() and (d / "accounting" / "pnl_bucket_4_by_pair.csv").is_file():
            out.append(d.name)
    return out


def load_b4_pair_leg_history(runs_root: Path) -> pd.DataFrame:
    """Long frame per (date, pair): cumulative pair / ETF-leg / underlying-leg PnL."""
    rows = []
    for ds in list_run_dates(runs_root):
        acct = runs_root / ds / "accounting"
        try:
            by_pair = pd.read_csv(acct / "pnl_bucket_4_by_pair.csv")
            by_sym = pd.read_csv(acct / "pnl_bucket_4_by_symbol.csv")
        except Exception:
            continue
        if by_pair.empty or "etf" not in by_pair.columns:
            continue
        by_pair["etf"] = by_pair["etf"].map(_norm)
        by_pair["underlying"] = by_pair["underlying"].map(_norm)
        sym_pnl = (
            by_sym.assign(symbol=by_sym["symbol"].map(_norm))
            .set_index("symbol")["total_pnl"]
            .astype(float)
            if not by_sym.empty else pd.Series(dtype=float)
        )
        for _, r in by_pair.iterrows():
            pair_total = float(r.get("total_pnl", np.nan))
            etf_leg = float(sym_pnl.get(r["etf"], np.nan))
            rows.append({
                "date": pd.Timestamp(ds),
                "pair": f"{r['etf']}|{r['underlying']}",
                "etf": r["etf"],
                "underlying": r["underlying"],
                "delta": float(r.get("delta", np.nan)),
                "pair_pnl_cum": pair_total,
                "etf_leg_pnl_cum": etf_leg,
                "und_leg_pnl_cum": pair_total - etf_leg if np.isfinite(etf_leg) else np.nan,
            })
    return pd.DataFrame(rows)


def load_pair_gross_and_realized_h(runs_root: Path, run_date: str) -> pd.DataFrame:
    """Per pair on run_date: ETF-leg gross + realized hedge ratio from the book.

    realized_h = und_gross_allocated / (|delta| * etf_gross); a shared
    underlying short is allocated pro-rata across that underlying's ETF legs.
    """
    p = runs_root / run_date / "accounting" / "net_exposure_bucket_4_detail.csv"
    by_pair_p = runs_root / run_date / "accounting" / "pnl_bucket_4_by_pair.csv"
    if not (p.is_file() and by_pair_p.is_file()):
        return pd.DataFrame()
    det = pd.read_csv(p)
    det["underlying"] = det["underlying"].map(_norm)
    det["symbol"] = det["symbol"].map(_norm)
    det["gross_notional_usd"] = pd.to_numeric(det["gross_notional_usd"], errors="coerce").fillna(0.0)
    etf_rows = det[det["leg_type"].astype(str).str.lower() == "etf"]
    und_rows = det[det["leg_type"].astype(str).str.lower() == "underlying"]
    und_gross = und_rows.groupby("underlying")["gross_notional_usd"].sum()

    deltas = pd.read_csv(by_pair_p)
    deltas["etf"] = deltas["etf"].map(_norm)
    delta_by_etf = deltas.set_index("etf")["delta"].astype(float).abs()

    rows = []
    for u, grp in etf_rows.groupby("underlying"):
        tot_etf = float(grp["gross_notional_usd"].sum())
        ug = float(und_gross.get(u, 0.0))
        for _, r in grp.iterrows():
            eg = float(r["gross_notional_usd"])
            share = eg / tot_etf if tot_etf > 0 else 0.0
            beta = float(delta_by_etf.get(r["symbol"], np.nan))
            h_real = (ug * share) / (beta * eg) if eg > 0 and np.isfinite(beta) and beta > 0 else np.nan
            rows.append({
                "etf": r["symbol"], "underlying": u,
                "pair": f"{r['symbol']}|{u}",
                "etf_gross_usd": eg,
                "realized_h": h_real,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model hedge overlay (the live engine path)
# ---------------------------------------------------------------------------
def build_model_h_overlay(
    underlyings: list[str],
    etf_by_und: dict[str, str],
    metrics_csv: Path,
    start: str,
) -> dict[str, dict[str, pd.Series]]:
    """Per underlying: {'close': px series, 'h': model h series} via the production engine."""
    from scripts.bucket4_hedge_cadence import (
        build_h_series,
        build_xsec_z_panel,
        load_name_tilts,
        load_policy_from_config,
    )
    from scripts.bucket4_phase345_backtest import load_metrics_filtered
    from scripts.bucket4_vol_shape_signals import get_pair_signal
    from strategy_config import load_config

    cfg = load_config()
    knobs, tilts, _src = load_policy_from_config(cfg)

    etfs = {etf_by_und[u] for u in underlyings if u in etf_by_und}
    metrics = load_metrics_filtered(metrics_csv, etfs)
    metrics = metrics[metrics["date"] >= pd.Timestamp(start)]

    closes: dict[str, pd.Series] = {}
    for u in underlyings:
        e = etf_by_und.get(u)
        if e is None:
            continue
        sub = metrics[metrics["ticker"] == e].dropna(subset=["underlying_adj_close"])
        if sub.empty:
            continue
        px = sub.set_index("date")["underlying_adj_close"].astype(float).sort_index()
        px = px[~px.index.duplicated(keep="last")]
        if len(px) >= 70:
            closes[u] = px
    if not closes:
        return {}

    xsec = build_xsec_z_panel(pd.DataFrame(closes)) if knobs.k_z != 0.0 else None

    out: dict[str, dict[str, pd.Series]] = {}
    for u, px in closes.items():
        cal = pd.DatetimeIndex(px.index)
        with warnings.catch_warnings():
            # warmup windows produce all-NaN slices in the signal rollups
            warnings.simplefilter("ignore", RuntimeWarning)
            sig = get_pair_signal(u, u, cal, history={}, underlying_prices=px,
                                  window=60, lookahead_shift=1)
        if xsec is not None and u in xsec.columns:
            sig = sig.copy()
            sig["xsec_z"] = xsec[u].reindex(sig.index)
        tilt = tilts.get(etf_by_und.get(u, "")) or tilts.get(u)
        h = build_h_series(sig, cal, knobs=knobs, name_tilt=tilt)
        out[u] = {"close": px, "h": h}
    return out


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_b4_pair_pnl_hedge_chart(
    run_date: str,
    *,
    runs_root: Path,
    out_dir: Path,
    metrics_csv: Path = DEFAULT_METRICS_CSV,
    max_pairs: int = 12,
) -> tuple[Path | None, Path | None]:
    """Build the per-pair PnL + hedge chart PNG and companion CSV (fail-soft)."""
    try:
        return _make_chart_inner(run_date, runs_root=runs_root, out_dir=out_dir,
                                 metrics_csv=metrics_csv, max_pairs=max_pairs)
    except Exception as exc:  # the EOD email must never die on a chart
        print(f"[B4-pair-charts] skipped: {type(exc).__name__}: {exc}")
        return None, None


def _make_chart_inner(
    run_date: str,
    *,
    runs_root: Path,
    out_dir: Path,
    metrics_csv: Path,
    max_pairs: int,
) -> tuple[Path | None, Path | None]:
    hist = load_b4_pair_leg_history(runs_root)
    if hist.empty:
        print("[B4-pair-charts] no per-pair accounting history")
        return None, None

    gross = load_pair_gross_and_realized_h(runs_root, run_date)
    active = (
        gross[gross["etf_gross_usd"] > MIN_ACTIVE_GROSS_USD]
        .sort_values("etf_gross_usd", ascending=False)
        if not gross.empty else pd.DataFrame()
    )
    if active.empty:
        # fall back: most-recently-moving pairs from history
        latest = hist[hist["date"] == hist["date"].max()]
        active = (
            latest.assign(etf_gross_usd=latest["pair_pnl_cum"].abs(), realized_h=np.nan)
            .sort_values("etf_gross_usd", ascending=False)[["etf", "underlying", "pair", "etf_gross_usd", "realized_h"]]
        )
    pairs = active.head(int(max_pairs))
    if pairs.empty:
        print("[B4-pair-charts] no active pairs")
        return None, None

    start = str(hist["date"].min().date())
    etf_by_und = dict(zip(pairs["underlying"], pairs["etf"]))
    overlay = build_model_h_overlay(
        sorted(set(pairs["underlying"])), etf_by_und, metrics_csv, start,
    )

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(14, max(3.2 * n, 4.0)), squeeze=False)
    knob_tag = _knob_tag()
    fig.suptitle(
        f"Bucket 4 per-pair PnL & hedge — run {run_date}  |  {knob_tag}",
        fontsize=12, y=1.0,
    )

    for i, (_, pr) in enumerate(pairs.iterrows()):
        pair, etf, und = pr["pair"], pr["etf"], pr["underlying"]
        sub = hist[hist["pair"] == pair].sort_values("date")

        ax = axes[i][0]
        ax.plot(sub["date"], sub["pair_pnl_cum"], color="#1f77b4", lw=1.8, label="pair total")
        ax.plot(sub["date"], sub["etf_leg_pnl_cum"], color="#ff7f0e", lw=1.1, label=f"{etf} leg")
        ax.plot(sub["date"], sub["und_leg_pnl_cum"], color="#2ca02c", lw=1.1, label=f"{und} leg")
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_title(f"{etf} / {und} — cumulative PnL $", fontsize=10, loc="left")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=8)

        ax2 = axes[i][1]
        ov = overlay.get(und)
        if ov is not None:
            px = ov["close"]
            ax2.plot(px.index, px.values, color="#444444", lw=1.1, label=f"{und} adj close")
            ax2.set_ylabel("price $", fontsize=8)
            axh = ax2.twinx()
            h = ov["h"]
            axh.axhspan(0.30, 0.80, color="#1f77b4", alpha=0.06)
            axh.plot(h.index, h.values, color="#d62728", lw=1.3, label="model h (v9)")
            rh = pr.get("realized_h")
            if rh is not None and np.isfinite(rh):
                axh.scatter([pd.Timestamp(run_date)], [float(rh)], color="#9467bd",
                            s=42, zorder=5, label=f"book h={rh:.2f}")
            axh.set_ylim(0.0, 1.0)
            axh.set_ylabel("hedge ratio h", fontsize=8)
            axh.tick_params(labelsize=8)
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = axh.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
        else:
            ax2.text(0.5, 0.5, "no price/h data", ha="center", va="center",
                     transform=ax2.transAxes, color="#999")
        ax2.set_title(f"{und} price + hedge ratio", fontsize=10, loc="left")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"b4_pair_pnl_hedge_{run_date}.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    csv = out_dir / f"b4_pair_leg_pnl_history_{run_date}.csv"
    hist[hist["pair"].isin(set(pairs["pair"]))].to_csv(csv, index=False)
    print(f"[B4-pair-charts] wrote {png.name} ({n} pairs) + {csv.name}")
    return png, csv


def _knob_tag() -> str:
    try:
        from scripts.bucket4_hedge_cadence import load_policy_from_config
        from strategy_config import load_config

        knobs, _, _ = load_policy_from_config(load_config())
        return (
            f"model h: v7+v9 h_mid={knobs.h_mid:.2f} k_z={knobs.k_z:.2f} "
            f"clip[{knobs.h_min:.2f},{knobs.h_max:.2f}] | cadence base={knobs.base_days:.0f}d cap={knobs.max_interval}d"
        )
    except Exception:
        return "model h: config unavailable"
