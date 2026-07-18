#!/usr/bin/env python3
"""
Bucket 5 extended-era grid: SPY idle share × put-budget multiplier.

Grid
----
* SPY idle fraction of undeployed cash: 10%, 20%, 30%, 40%, 50%
* Put premium vs production 2.4%/roll ladder: 1.00×, 1.25× (+25%), 1.50× (+50%)

Era defaults to extended (synthetic UVIX/SVIX from 2008-01-01).

Usage::

    python scripts/bucket5_spy_put_grid_bt.py
    python scripts/bucket5_spy_put_grid_bt.py --era live
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

from scripts.bucket5_carry_bt import run_carry_backtest  # noqa: E402
from scripts.bucket5_data import (  # noqa: E402
    DEFAULT_BORROW_SVIX,
    DEFAULT_BORROW_UVIX,
    INCEPTION,
    load_series,
    load_vol_panel,
)
from scripts.bucket5_insurance_bt import (  # noqa: E402
    EXTENDED_START,
    LadderRung,
    adaptive_rebal_dates,
    crash_payoff,
    production_config,
    run_hedge_layer,
)
from scripts.bucket5_put_overlay import load_spx_spot  # noqa: E402
from scripts.bucket5_spy_hybrid_bt import _ann_stats, _window_return  # noqa: E402

SPY_FRACS = (0.10, 0.20, 0.30, 0.40, 0.50)
PUT_MULTS = (1.00, 1.25, 1.50)


def _scale_rungs(rungs: list[LadderRung], mult: float) -> list[LadderRung]:
    return [LadderRung(r.otm_pct, float(r.per_roll_frac) * float(mult)) for r in rungs]


def _build_idle_base(
    panel: pd.DataFrame,
    spy: pd.Series,
    *,
    spy_idle_frac: float,
    cfg,
) -> dict:
    """Carry sleeve + idle SPY/T-bill mix → base_equity (no puts yet)."""
    ratio = panel["ratio"]
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
    idle_ret = idle * (float(spy_idle_frac) * spy_ret + (1.0 - float(spy_idle_frac)) * tbill_daily)
    base_ret = sleeve_ret + idle_ret
    base_equity = (1.0 + base_ret).cumprod() * cfg.initial_capital
    base_equity.iloc[0] = cfg.initial_capital

    return {
        "carry": carry,
        "sleeve_ret": sleeve_ret,
        "deployed": deployed,
        "rho_s": rho_s,
        "gross_s": gross_s,
        "rebal": rebal,
        "base_equity": base_equity,
        "spy_px": spy_px,
    }


def _attach_puts(
    panel: pd.DataFrame,
    spx: pd.Series,
    iv: pd.Series,
    base: dict,
    cfg,
) -> dict:
    """Put ladder + monetize/redeploy on top of a prebuilt base_equity path."""
    ratio = panel["ratio"]
    vix = panel["vix"]
    base_equity = base["base_equity"]
    sleeve_ret = base["sleeve_ret"]

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
        tret = cfg.tbill_rate / 252.0
        sb = tb = cum_p = 0.0
        extra_vals = []
        for dt in base_equity.index:
            sb = sb * (1.0 + float(sret.loc[dt])) + float(inj_sleeve.loc[dt])
            tb = tb * (1.0 + tret) + float(inj_tbill.loc[dt])
            cum_p += float(realized.loc[dt])
            extra_vals.append(sb + tb - cum_p)
        redeploy_extra = pd.Series(extra_vals, index=base_equity.index)

    combined = base_equity + lad["put_mtm"] + cum_cash + redeploy_extra
    bt = pd.DataFrame(
        {
            "ratio": ratio.reindex(base_equity.index),
            "rho": base["rho_s"].reindex(base_equity.index),
            "gross_frac": base["gross_s"].reindex(base_equity.index),
            "base_equity": base_equity,
            "put_mtm": lad["put_mtm"],
            "put_cash_cum": cum_cash,
            "realized_cum": realized.cumsum(),
            "redeploy_extra": redeploy_extra,
            "combined_equity": combined,
        }
    )
    bt["combined_ret"] = bt["combined_equity"].pct_change().fillna(0.0)
    bt["drawdown"] = bt["combined_equity"].div(bt["combined_equity"].cummax()).sub(1.0)
    bt.attrs["rebalances"] = len(base["rebal"])
    bt.attrs["ladder"] = ladder.attrs
    return {"bt": bt, "carry": base["carry"], "ladder": ladder, "rebal": base["rebal"], "cfg": cfg}


def _metrics(res: dict, spx: pd.Series, panel: pd.DataFrame, *, spy_frac: float, put_mult: float) -> dict:
    eq = res["bt"]["combined_equity"]
    stats = _ann_stats(eq)
    crash = crash_payoff(res, spx, panel)
    ladder_pct = sum(r.per_roll_frac for r in res["cfg"].rungs) * 100.0
    return {
        "spy_idle_frac": spy_frac,
        "put_mult": put_mult,
        "ladder_per_roll_%": round(ladder_pct, 3),
        "final_$": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "CAGR": stats["CAGR"],
        "Vol": stats["Vol"],
        "Sharpe": stats["Sharpe"],
        "MaxDD": stats["MaxDD"],
        "Calmar": (stats["CAGR"] / abs(stats["MaxDD"])) if stats["MaxDD"] and stats["MaxDD"] < 0 else np.nan,
        "realized_$": float(res["ladder"].attrs.get("realized_total", 0.0)),
        "gfc_2008_09": _window_return(eq, "2008-01-01", "2009-03-31"),
        "covid_2020": _window_return(eq, "2020-02-15", "2020-04-30"),
        "bear_2022": _window_return(eq, "2022-01-01", "2022-10-14"),
        "aug24": _window_return(eq, "2024-08-01", "2024-08-31"),
        "crash_mild_-20%": crash.get("mild_-20%"),
        "crash_severe_-30%": crash.get("severe_-30%"),
        "crash_volmageddon_-40%": crash.get("volmageddon_-40%"),
    }


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    p = df.pivot(index="spy_idle_frac", columns="put_mult", values=value)
    p.index = [f"{100 * i:.0f}% SPY" for i in p.index]
    p.columns = [f"puts {c:.2f}×" for c in p.columns]
    return p


def _save_heatmap(mat: pd.DataFrame, title: str, path: Path, fmt: str = ".1%") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    data = mat.astype(float).values
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(list(mat.columns))
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(list(mat.index))
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if fmt.endswith("%"):
                label = f"{100 * v:.1f}%"
            else:
                label = f"{v:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--era", choices=("live", "extended"), default="extended")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    start = INCEPTION if args.era == "live" else EXTENDED_START
    use_synthetic = args.era == "extended"
    dest = (
        REPO
        / "data"
        / "runs"
        / pd.Timestamp.today().strftime("%Y-%m-%d")
        / "bucket5"
        / "spy_put_grid"
    )
    dest.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading panel era={args.era} start={start} ...")
    panel = load_vol_panel(start=start, end=args.end, use_synthetic=use_synthetic)
    spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"), args.end)
    spy = load_series("SPY", start=panel.index.min().strftime("%Y-%m-%d"), end=args.end).rename("spy")
    iv = (panel["vix"] / 100.0).rename("iv")
    base_cfg = production_config()

    print(
        f"Grid: SPY idle {list(SPY_FRACS)} × put mult {list(PUT_MULTS)} "
        f"({len(SPY_FRACS) * len(PUT_MULTS)} runs) "
        f"on {panel.index.min().date()} → {panel.index.max().date()} ({len(panel)} days)"
    )

    rows: list[dict] = []
    equity_cols: dict[str, pd.Series] = {}

    for spy_frac in SPY_FRACS:
        print(f"\n=== SPY idle {spy_frac:.0%} — building carry/base path ===")
        base = _build_idle_base(panel, spy, spy_idle_frac=spy_frac, cfg=base_cfg)
        for put_mult in PUT_MULTS:
            label = f"spy{int(spy_frac*100):02d}_puts{put_mult:.2f}x"
            print(f"  -> puts {put_mult:.2f}× ({sum(r.per_roll_frac for r in base_cfg.rungs)*put_mult*100:.2f}%/roll) ...")
            cfg = replace(
                base_cfg,
                rungs=_scale_rungs(base_cfg.rungs, put_mult),
            )
            res = _attach_puts(panel, spx, iv, base, cfg)
            m = _metrics(res, spx, panel, spy_frac=spy_frac, put_mult=put_mult)
            m["label"] = label
            rows.append(m)
            equity_cols[label] = res["bt"]["combined_equity"]
            print(
                f"     CAGR={100*m['CAGR']:.2f}%  Sharpe={m['Sharpe']:.2f}  "
                f"MaxDD={100*m['MaxDD']:.2f}%  realized=${m['realized_$']:,.0f}"
            )

    summary = pd.DataFrame(rows).sort_values(["spy_idle_frac", "put_mult"]).reset_index(drop=True)
    series = pd.DataFrame(equity_cols).dropna(how="any")

    summary_path = dest / "grid_summary.csv"
    series_path = dest / "grid_equity.csv"
    summary.to_csv(summary_path, index=False)
    series.to_csv(series_path)

    # pivots + heatmaps
    for metric, title, fmt, fname in [
        ("CAGR", "CAGR", ".1%", "heatmap_cagr.png"),
        ("Sharpe", "Sharpe", ".2f", "heatmap_sharpe.png"),
        ("MaxDD", "Max drawdown", ".1%", "heatmap_maxdd.png"),
        ("Calmar", "Calmar (CAGR/|MaxDD|)", ".2f", "heatmap_calmar.png"),
        ("gfc_2008_09", "GFC window return (2008→2009-03)", ".1%", "heatmap_gfc.png"),
        ("aug24", "Aug 2024 return", ".1%", "heatmap_aug24.png"),
    ]:
        mat = _pivot(summary, metric)
        mat.to_csv(dest / f"pivot_{metric}.csv")
        _save_heatmap(mat, f"Bucket 5 grid — {title} ({args.era})", dest / fname, fmt=fmt)

    # equity chart: best Sharpe / best CAGR / worst MaxDD extremes + mid cell
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    pick = [
        summary.sort_values("Sharpe", ascending=False).iloc[0]["label"],
        summary.sort_values("CAGR", ascending=False).iloc[0]["label"],
        summary.sort_values("MaxDD", ascending=False).iloc[0]["label"],  # least negative
        "spy30_puts1.25x",
    ]
    pick = list(dict.fromkeys([p for p in pick if p in series.columns]))  # unique, existing
    for col in pick:
        ax.plot(series.index, series[col] / 1e6, lw=1.4, label=col)
    ax.set_ylabel("Equity ($M)")
    ax.set_title(
        f"Bucket 5 SPY×puts grid — selected paths ({series.index.min().date()} → {series.index.max().date()})"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plot_path = dest / "grid_selected_equity.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    # ranked tables
    rank_sharpe = summary.sort_values("Sharpe", ascending=False).head(5)
    rank_cagr = summary.sort_values("CAGR", ascending=False).head(5)
    rank_calmar = summary.sort_values("Calmar", ascending=False).head(5)
    rank_dd = summary.sort_values("MaxDD", ascending=False).head(5)  # best (= least bad) DD

    meta = {
        "era": args.era,
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "n_days": int(len(panel)),
        "spy_fracs": list(SPY_FRACS),
        "put_mults": list(PUT_MULTS),
        "baseline_ladder_per_roll": 0.024,
        "elapsed_sec": round(time.time() - t0, 1),
        "best_sharpe": rank_sharpe.iloc[0][["label", "CAGR", "Sharpe", "MaxDD"]].to_dict(),
        "best_cagr": rank_cagr.iloc[0][["label", "CAGR", "Sharpe", "MaxDD"]].to_dict(),
        "best_calmar": rank_calmar.iloc[0][["label", "CAGR", "Sharpe", "MaxDD", "Calmar"]].to_dict(),
        "best_maxdd": rank_dd.iloc[0][["label", "CAGR", "Sharpe", "MaxDD"]].to_dict(),
        "paths": {
            "summary": str(summary_path),
            "series": str(series_path),
            "selected_plot": str(plot_path),
        },
    }
    (dest / "grid_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["spy"] = out["spy_idle_frac"].map(lambda x: f"{100*x:.0f}%")
        out["puts"] = out["put_mult"].map(lambda x: f"{x:.2f}×")
        out["CAGR"] = out["CAGR"].map(lambda x: f"{100*x:.2f}%")
        out["Vol"] = out["Vol"].map(lambda x: f"{100*x:.2f}%")
        out["Sharpe"] = out["Sharpe"].map(lambda x: f"{x:.2f}")
        out["MaxDD"] = out["MaxDD"].map(lambda x: f"{100*x:.2f}%")
        out["Calmar"] = out["Calmar"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        out["gfc"] = out["gfc_2008_09"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "")
        out["aug24"] = out["aug24"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "")
        return out[["spy", "puts", "CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "gfc", "aug24"]]

    print("\n" + "=" * 72)
    print(f"FULL GRID SUMMARY  ({meta['start']} → {meta['end']}, {meta['elapsed_sec']}s)")
    print("=" * 72)
    print(_fmt(summary).to_string(index=False))
    print("\n-- Top 5 by Sharpe --")
    print(_fmt(rank_sharpe).to_string(index=False))
    print("\n-- Top 5 by CAGR --")
    print(_fmt(rank_cagr).to_string(index=False))
    print("\n-- Top 5 by Calmar --")
    print(_fmt(rank_calmar).to_string(index=False))
    print("\n-- Best 5 MaxDD (least severe) --")
    print(_fmt(rank_dd).to_string(index=False))
    print(f"\nCAGR heatmap:\n{_pivot(summary, 'CAGR').map(lambda x: f'{100*x:.1f}%')}")
    print(f"\nSharpe heatmap:\n{_pivot(summary, 'Sharpe').map(lambda x: f'{x:.2f}')}")
    print(f"\nMaxDD heatmap:\n{_pivot(summary, 'MaxDD').map(lambda x: f'{100*x:.1f}%')}")
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
