"""
Regenerate all live-era Bucket 5 charts into data/runs/<today>/bucket5/live/.

Produces:
  - bucket5_equity_curves.png       carry-only (multiple rho)
  - bucket5_full_equity.png         carry + single-strike put overlay
  - insurance_sf20_ladder24_*.png   insurance product (buy-and-roll ladder)
  - insurance_monetize_*.png        insurance product WITH profit-taking (primary)
  - insurance_compare_equity.png    monetize vs buy-and-roll on one chart

Usage::

    python scripts/bucket5_live_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from scripts.bucket5_carry_bt import RunConfig, run_all
    from scripts.bucket5_data import INCEPTION, load_vol_panel, rebalance_dates
    from scripts.bucket5_full_bt import PutOverlayConfig, run_full
    from scripts.bucket5_insurance_bt import (
        InsuranceConfig,
        LADDER_2P4,
        MonetizeConfig,
        production_config,
        run_insurance,
        save_insurance_plots,
        summarize,
    )
except ImportError:
    from bucket5_carry_bt import RunConfig, run_all  # type: ignore
    from bucket5_data import INCEPTION, load_vol_panel, rebalance_dates  # type: ignore
    from bucket5_full_bt import PutOverlayConfig, run_full  # type: ignore
    from bucket5_insurance_bt import (  # type: ignore
        InsuranceConfig,
        LADDER_2P4,
        MonetizeConfig,
        production_config,
        run_insurance,
        save_insurance_plots,
        summarize,
    )
try:
    from scripts.bucket5_put_overlay import load_spx_spot
except ImportError:
    from bucket5_put_overlay import load_spx_spot  # type: ignore


def _dest() -> Path:
    d = Path("data/runs") / pd.Timestamp.today().strftime("%Y-%m-%d") / "bucket5" / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot_carry(out: dict, dest: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for name, bt in out["results"].items():
        ax.plot(bt.index, bt["equity"] / 1e6, label=name, lw=1.4)
    ax.set_title("Bucket 5 — short UVIX / short SVIX carry (live era)")
    ax.set_ylabel("Equity ($M)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = dest / "bucket5_equity_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def _plot_full(out: dict, dest: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for name, bundle in out["results"].items():
        c = bundle["carry"]
        p = bundle["puts"]
        axes[0].plot(c.index, c["equity"] / 1e6, label=f"{name} carry", lw=1.2)
        axes[1].plot(p.index, p["combined_equity"] / 1e6, label=f"{name} + puts", lw=1.2)
    axes[0].set_title("Bucket 5 carry equity (live era)")
    axes[0].set_ylabel("$M")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Combined (carry + SPX 6M→3M puts, single strike)")
    axes[1].set_ylabel("$M")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    p = dest / "bucket5_full_equity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def _plot_insurance_compare(res_no: dict, res_mon: dict, dest: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bt0 = res_no["bt"]
    bt1 = res_mon["bt"]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.plot(bt0.index, bt0["combined_equity"] / 1e6, lw=1.4, color="#64748b",
            label="A baseline (fixed budget)")
    ax.plot(bt1.index, bt1["combined_equity"] / 1e6, lw=1.6, color="#0f766e",
            label="B dynamic budget (production)")
    ax.plot(bt1.index, bt1["base_equity"] / 1e6, lw=1.0, ls="--", color="#2563eb", alpha=0.7,
            label="Sleeve + T-bills (no puts)")
    ax.set_ylabel("Equity ($M)")
    ax.set_title("Insurance — A baseline vs B dynamic budget (live era)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    p = dest / "insurance_compare_equity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main() -> int:
    dest = _dest()
    panel = load_vol_panel(start=INCEPTION, use_synthetic=False)
    spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"))
    iv = (panel["vix"] / 100.0).rename("iv")
    print(f"live sample: {panel.index.min().date()} -> {panel.index.max().date()} ({len(panel)} rows)")
    print(f"output dir: {dest}\n")

    saved: list[Path] = []

    # 1) Carry-only curves
    carry_out = run_all(RunConfig())
    carry_out["summary"].to_csv(dest / "bucket5_carry_summary.csv")
    saved.append(_plot_carry(carry_out, dest))

    # 2) Full product (single-strike puts)
    full_out = run_full(RunConfig(), PutOverlayConfig())
    full_out["summary"].to_csv(dest / "bucket5_full_summary.csv")
    saved.append(_plot_full(full_out, dest))

    # 3) Insurance — buy-and-roll
    cfg_roll = InsuranceConfig(sleeve_frac=0.20, rungs=LADDER_2P4, monetize=None)
    res_roll = run_insurance(panel, spx, iv, cfg_roll)
    res_roll["bt"].to_csv(dest / "insurance_sf20_ladder24_path.csv")
    saved.extend(save_insurance_plots(res_roll, dest, tag="insurance_sf20_ladder24"))

    # 4) Insurance — production default (B: dynamic budget + liquidate + redeploy)
    res_a = run_insurance(panel, spx, iv, production_config(hedge_budget=None))
    res_a["bt"].to_csv(dest / "baseline_A_path.csv")
    cfg_prod = production_config()
    res_prod = run_insurance(panel, spx, iv, cfg_prod)
    res_prod["bt"].to_csv(dest / "production_B_path.csv")
    saved.extend(save_insurance_plots(res_prod, dest, tag="production_B"))
    saved.append(_plot_insurance_compare(res_a, res_prod, dest))

    rows = []
    for label, res in [("baseline_A", res_a), ("production_B", res_prod)]:
        s = summarize(res, spx, panel)
        s["variant"] = label
        s["realized_$"] = float(res["ladder"].attrs.get("realized_total", 0.0))
        rows.append(s)
    pd.DataFrame(rows).to_csv(dest / "insurance_production_compare.csv", index=False)

    manifest = dest / "PLOTS.txt"
    manifest.write_text("\n".join(str(p) for p in saved) + "\n", encoding="utf-8")

    print("=== REGENERATED LIVE BUCKET 5 PLOTS ===")
    for p in saved:
        print(f"  {p}")
    print(f"\nmanifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
