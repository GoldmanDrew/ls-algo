#!/usr/bin/env python3
"""
Compare Bucket 5 insurance collateral policies.

Baseline (production):
  short-vol sleeve + idle cash in T-bills + SPX put ladder / monetize

SPY hybrid (this test):
  same sleeve + puts, but idle cash is split 50% SPY / 50% T-bill collateral

Also reports a 50/50 capital-split book (half capital buy-and-hold SPY,
half capital running full B5) as a second reference.

Usage::

    python scripts/bucket5_spy_hybrid_bt.py
    python scripts/bucket5_spy_hybrid_bt.py --era extended
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_data import INCEPTION, load_series  # noqa: E402
from scripts.bucket5_insurance_bt import (  # noqa: E402
    EXTENDED_START,
    production_config,
    run_insurance,
    summarize,
)
from scripts.bucket5_put_overlay import load_spx_spot  # noqa: E402
from scripts.bucket5_data import load_vol_panel  # noqa: E402


def _ann_stats(equity: pd.Series) -> dict:
    eq = equity.dropna()
    ret = eq.pct_change().fillna(0.0)
    n = len(eq)
    if n < 3:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = (1.0 + total) ** (252.0 / (n - 1)) - 1.0 if total > -1 else np.nan
    vol = float(ret.std() * np.sqrt(252))
    sharpe = float(ret.mean() * 252 / vol) if vol > 0 else np.nan
    dd = float(eq.div(eq.cummax()).sub(1.0).min())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": dd}


def _window_return(eq: pd.Series, start: str, end: str) -> float:
    w = eq.loc[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
    return float(w.iloc[-1] / w.iloc[0] - 1.0) if len(w) >= 2 else float("nan")


def load_spy(start: str, end: str | None = None) -> pd.Series:
    return load_series("SPY", start=start, end=end).rename("spy")


def run_idle_spy_hybrid(
    panel: pd.DataFrame,
    spx: pd.Series,
    iv: pd.Series,
    spy: pd.Series,
    *,
    spy_idle_frac: float = 0.50,
    cfg=None,
) -> dict:
    """B5 insurance where idle cash earns a SPY/T-bill mix instead of 100% T-bills.

    ``spy_idle_frac=0.50`` means half of undeployed cash is in SPY, the other half
    stays T-bill collateral. Short-vol sleeve sizing and the put ladder are unchanged.
    """
    cfg = cfg or production_config()
    # Run the stock engine once to get sleeve returns / deployed path / puts funded
    # off the *baseline* base_equity path, then rebuild base_equity with SPY idle mix.
    # Cleaner: replicate run_insurance with a patched idle return.
    from scripts.bucket5_carry_bt import run_carry_backtest
    from scripts.bucket5_data import DEFAULT_BORROW_SVIX, DEFAULT_BORROW_UVIX
    from scripts.bucket5_insurance_bt import adaptive_rebal_dates, run_hedge_layer

    ratio = panel["ratio"]
    vix = panel["vix"]
    rebal = adaptive_rebal_dates(ratio, base_days=cfg.base_days, k_stress=cfg.cadence_k)
    rho_s, gross_regime = cfg.regime.series(ratio)
    gross_s = (gross_regime * cfg.sleeve_frac).rename("gross")

    carry = run_carry_backtest(
        panel,
        rho_s,
        rebal,
        gross_daily=gross_s,
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.uvix_slip_bps,
        borrow_uvix_annual=cfg.borrow_uvix_annual
        if cfg.borrow_uvix_annual is not None
        else DEFAULT_BORROW_UVIX,
        borrow_svix_annual=cfg.borrow_svix_annual
        if cfg.borrow_svix_annual is not None
        else DEFAULT_BORROW_SVIX,
        short_proceeds_annual=cfg.short_proceeds_annual,
    )
    sleeve_ret = carry["ret"]
    deployed = (carry["gross"] / carry["equity"].replace(0, np.nan)).clip(0, 1).fillna(0.0)
    idle = (1.0 - deployed).clip(0, 1)

    spy_px = spy.reindex(carry.index).ffill().bfill()
    spy_ret = spy_px.pct_change().fillna(0.0)
    tbill_daily = cfg.tbill_rate / 252.0
    idle_ret = idle * (
        float(spy_idle_frac) * spy_ret + (1.0 - float(spy_idle_frac)) * tbill_daily
    )
    base_ret = sleeve_ret + idle_ret
    base_equity = (1.0 + base_ret).cumprod() * cfg.initial_capital
    base_equity.iloc[0] = cfg.initial_capital

    ladder = run_hedge_layer(panel.index, base_equity, spx, iv, vix, ratio, cfg)
    lad = ladder.reindex(base_equity.index).ffill().fillna(0.0)
    cum_cash = lad["put_cash_flow"].cumsum()
    realized = lad["realized"] if "realized" in lad else pd.Series(0.0, index=base_equity.index)

    redeploy_extra = pd.Series(0.0, index=base_equity.index)
    if cfg.redeploy is not None and float(realized.abs().sum()) > 0:
        ratio_at = ratio.reindex(base_equity.index).ffill()
        sleeve_w = ratio_at.map(cfg.redeploy.sleeve_weight).clip(0.0, 1.0)
        inj_sleeve = (realized * sleeve_w).fillna(0.0)
        inj_tbill = (realized * (1.0 - sleeve_w)).fillna(0.0)
        sret = sleeve_ret.reindex(base_equity.index).fillna(0.0)
        tret = tbill_daily
        sb = tb = cum_p = 0.0
        extra_vals = []
        for dt in base_equity.index:
            sb = sb * (1.0 + float(sret.loc[dt])) + float(inj_sleeve.loc[dt])
            tb = tb * (1.0 + tret) + float(inj_tbill.loc[dt])
            cum_p += float(realized.loc[dt])
            extra_vals.append(sb + tb - cum_p)
        redeploy_extra = pd.Series(extra_vals, index=base_equity.index)

    combined_equity = base_equity + lad["put_mtm"] + cum_cash + redeploy_extra
    out = pd.DataFrame(
        {
            "ratio": ratio.reindex(base_equity.index),
            "rho": rho_s.reindex(base_equity.index),
            "gross_frac": gross_s.reindex(base_equity.index),
            "deployed": deployed.reindex(base_equity.index),
            "sleeve_equity": carry["equity"],
            "base_equity": base_equity,
            "put_mtm": lad["put_mtm"],
            "put_cash_cum": cum_cash,
            "realized_cum": realized.cumsum(),
            "redeploy_extra": redeploy_extra,
            "combined_equity": combined_equity,
            "spy": spy_px,
        }
    )
    out["combined_ret"] = out["combined_equity"].pct_change().fillna(0.0)
    out["drawdown"] = out["combined_equity"].div(out["combined_equity"].cummax()).sub(1.0)
    out.attrs["rebalances"] = len(rebal)
    out.attrs["ladder"] = ladder.attrs
    out.attrs["carry"] = carry
    out.attrs["spy_idle_frac"] = float(spy_idle_frac)
    return {"bt": out, "carry": carry, "ladder": ladder, "rebal": rebal, "cfg": cfg}


def run_capital_split_spy(
    panel: pd.DataFrame,
    spx: pd.Series,
    iv: pd.Series,
    spy: pd.Series,
    *,
    spy_capital_frac: float = 0.50,
    cfg=None,
) -> dict:
    """Half capital buy-and-hold SPY; remaining capital runs full B5 insurance."""
    cfg = cfg or production_config()
    spy_w = float(spy_capital_frac)
    b5_w = 1.0 - spy_w
    b5_cfg = replace(cfg, initial_capital=cfg.initial_capital * b5_w)
    b5 = run_insurance(panel, spx, iv, b5_cfg)
    b5_eq = b5["bt"]["combined_equity"]

    spy_px = spy.reindex(b5_eq.index).ffill().bfill()
    spy_eq = cfg.initial_capital * spy_w * (spy_px / float(spy_px.iloc[0]))
    combined = b5_eq + spy_eq

    out = b5["bt"].copy()
    out["spy_equity"] = spy_eq
    out["b5_equity"] = b5_eq
    out["combined_equity"] = combined
    out["combined_ret"] = combined.pct_change().fillna(0.0)
    out["drawdown"] = combined.div(combined.cummax()).sub(1.0)
    return {
        "bt": out,
        "carry": b5["carry"],
        "ladder": b5["ladder"],
        "rebal": b5["rebal"],
        "cfg": cfg,
        "b5_res": b5,
    }


def _row(name: str, equity: pd.Series, extra: dict | None = None) -> dict:
    stats = _ann_stats(equity)
    row = {
        "variant": name,
        "final_$": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "CAGR": stats["CAGR"],
        "Vol": stats["Vol"],
        "Sharpe": stats["Sharpe"],
        "MaxDD": stats["MaxDD"],
        "aug24": _window_return(equity, "2024-08-01", "2024-08-31"),
        "2022_bear": _window_return(equity, "2022-03-30", "2022-10-14"),
        "2023": _window_return(equity, "2023-01-01", "2023-12-31"),
        "2025_ytd": _window_return(equity, "2025-01-01", "2025-12-31"),
    }
    if extra:
        row.update(extra)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--era", choices=("live", "extended"), default="live")
    ap.add_argument("--spy-idle-frac", type=float, default=0.50)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    start = INCEPTION if args.era == "live" else EXTENDED_START
    use_synthetic = args.era == "extended"
    dest = REPO / "data" / "runs" / pd.Timestamp.today().strftime("%Y-%m-%d") / "bucket5" / "spy_hybrid"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Loading panel era={args.era} start={start} ...")
    panel = load_vol_panel(start=start, end=args.end, use_synthetic=use_synthetic)
    spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"), args.end)
    spy = load_spy(panel.index.min().strftime("%Y-%m-%d"), args.end)
    iv = (panel["vix"] / 100.0).rename("iv")
    cfg = production_config()

    print("Running baseline B5 (idle = T-bills) ...")
    base = run_insurance(panel, spx, iv, cfg)

    print(f"Running idle-mix hybrid (idle = {args.spy_idle_frac:.0%} SPY / {1-args.spy_idle_frac:.0%} T-bills) ...")
    idle_hyb = run_idle_spy_hybrid(
        panel, spx, iv, spy, spy_idle_frac=args.spy_idle_frac, cfg=cfg
    )

    print("Running capital-split (50% SPY B&H + 50% full B5) ...")
    cap_hyb = run_capital_split_spy(panel, spx, iv, spy, spy_capital_frac=0.50, cfg=cfg)

    spy_px = spy.reindex(base["bt"].index).ffill().bfill()
    spy_only = cfg.initial_capital * (spy_px / float(spy_px.iloc[0]))

    base_sum = summarize(base, spx, panel)
    idle_sum = summarize(idle_hyb, spx, panel)
    # capital-split crash/put metrics are not directly comparable via summarize on
    # the combined book; still record B5-half metrics.
    cap_b5_sum = summarize(cap_hyb["b5_res"], spx, panel)

    rows = [
        _row(
            "A_baseline_B5_tbill_collateral",
            base["bt"]["combined_equity"],
            {
                "sleeve_carry_%/yr": base_sum.get("sleeve_carry_%/yr"),
                "realized_$": float(base["ladder"].attrs.get("realized_total", 0.0)),
                "crash_mild_-20%": base_sum.get("crash_mild_-20%"),
                "crash_severe_-30%": base_sum.get("crash_severe_-30%"),
                "crash_volmageddon_-40%": base_sum.get("crash_volmageddon_-40%"),
            },
        ),
        _row(
            f"B_idle_{int(args.spy_idle_frac*100)}pct_SPY_{int((1-args.spy_idle_frac)*100)}pct_tbill",
            idle_hyb["bt"]["combined_equity"],
            {
                "sleeve_carry_%/yr": idle_sum.get("sleeve_carry_%/yr"),
                "realized_$": float(idle_hyb["ladder"].attrs.get("realized_total", 0.0)),
                "crash_mild_-20%": idle_sum.get("crash_mild_-20%"),
                "crash_severe_-30%": idle_sum.get("crash_severe_-30%"),
                "crash_volmageddon_-40%": idle_sum.get("crash_volmageddon_-40%"),
            },
        ),
        _row(
            "C_50pct_SPY_plus_50pct_B5",
            cap_hyb["bt"]["combined_equity"],
            {
                "sleeve_carry_%/yr": cap_b5_sum.get("sleeve_carry_%/yr"),
                "realized_$": float(cap_hyb["ladder"].attrs.get("realized_total", 0.0)),
                "crash_mild_-20%": np.nan,  # combined book; not same as B5-only crash helper
                "crash_severe_-30%": np.nan,
                "crash_volmageddon_-40%": np.nan,
            },
        ),
        _row("D_SPY_buy_and_hold", spy_only, {"sleeve_carry_%/yr": np.nan, "realized_$": 0.0}),
    ]
    summary = pd.DataFrame(rows)

    # equity series for plotting / export
    series = pd.DataFrame(
        {
            "A_baseline": base["bt"]["combined_equity"],
            "B_idle_spy_mix": idle_hyb["bt"]["combined_equity"],
            "C_capital_split": cap_hyb["bt"]["combined_equity"],
            "D_SPY": spy_only,
        }
    ).dropna(how="any")

    summary_path = dest / "spy_hybrid_summary.csv"
    series_path = dest / "spy_hybrid_equity.csv"
    summary.to_csv(summary_path, index=False)
    series.to_csv(series_path)

    # charts
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True, sharex=True)
    for col, color in [
        ("A_baseline", "#0f766e"),
        ("B_idle_spy_mix", "#2563eb"),
        ("C_capital_split", "#b45309"),
        ("D_SPY", "#64748b"),
    ]:
        axes[0].plot(series.index, series[col] / 1e6, label=col, lw=1.5, color=color)
    axes[0].set_ylabel("Equity ($M)")
    axes[0].set_title(
        f"Bucket 5 collateral test — {args.era} era "
        f"({series.index.min().date()} → {series.index.max().date()})"
    )
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(alpha=0.3)

    for col, color in [
        ("A_baseline", "#0f766e"),
        ("B_idle_spy_mix", "#2563eb"),
        ("C_capital_split", "#b45309"),
        ("D_SPY", "#64748b"),
    ]:
        dd = series[col].div(series[col].cummax()).sub(1.0) * 100
        axes[1].plot(series.index, dd, label=col, lw=1.2, color=color)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title("Drawdowns")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=0.3)
    plot_path = dest / "spy_hybrid_equity.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    meta = {
        "era": args.era,
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "spy_idle_frac": args.spy_idle_frac,
        "interpretation": {
            "B": (
                "Same B5 short-vol sleeve + SPX puts; idle cash split "
                f"{args.spy_idle_frac:.0%} SPY / {1-args.spy_idle_frac:.0%} T-bill collateral"
            ),
            "C": "50% capital buy-and-hold SPY + 50% capital running full B5 insurance",
        },
        "paths": {
            "summary": str(summary_path),
            "series": str(series_path),
            "plot": str(plot_path),
        },
    }
    (dest / "spy_hybrid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # console report
    show = summary.copy()
    pct_cols = ("CAGR", "Vol", "MaxDD", "total_return", "aug24", "2022_bear", "2023", "2025_ytd")
    for c in pct_cols:
        if c in show.columns:
            show[c] = summary[c].map(lambda x: f"{100 * x:.2f}%" if pd.notna(x) else "")
    if "Sharpe" in show.columns:
        show["Sharpe"] = summary["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    show["final_$"] = summary["final_$"].map(lambda x: f"${x:,.0f}")
    if "realized_$" in show.columns:
        show["realized_$"] = summary["realized_$"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

    print("\n=== Bucket 5 vs SPY collateral hybrid ===")
    print(
        show[["variant", "final_$", "CAGR", "Vol", "Sharpe", "MaxDD", "aug24", "2022_bear"]].to_string(
            index=False
        )
    )
    print(f"\nsaved summary -> {summary_path}")
    print(f"saved series  -> {series_path}")
    print(f"saved plot    -> {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
