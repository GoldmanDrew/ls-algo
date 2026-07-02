"""
Sweep tail-hedge structures + dynamic budget sizing.

Production stack (all variants): liquidate + redeploy, sleeve_frac=20%.

Usage::

    python scripts/bucket5_hedge_sweep.py           # live era
    python scripts/bucket5_hedge_sweep.py extended  # 2008+ synthetic history + charts
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.bucket5_insurance_bt import (
        EXTENDED_START,
        BackspreadHedge,
        HedgeBudgetPolicy,
        InsuranceConfig,
        LADDER_2P4,
        MonetizeConfig,
        RedeployPolicy,
        build_ladder,
        production_config,
        run_insurance,
        save_insurance_plots,
        summarize,
    )
    from scripts.bucket5_data import INCEPTION, load_vol_panel
    from scripts.bucket5_put_overlay import load_spx_spot
except ImportError:
    from bucket5_insurance_bt import (  # type: ignore
        EXTENDED_START,
        BackspreadHedge,
        HedgeBudgetPolicy,
        InsuranceConfig,
        LADDER_2P4,
        MonetizeConfig,
        RedeployPolicy,
        build_ladder,
        production_config,
        run_insurance,
        save_insurance_plots,
        summarize,
    )
    from bucket5_data import INCEPTION, load_vol_panel  # type: ignore
    from bucket5_put_overlay import load_spx_spot  # type: ignore


PROD_MON = MonetizeConfig(runner_frac=0.0)
PROD_REDEPLOY = RedeployPolicy()
SLEEVE = 0.20
TOTAL_24 = 0.024
TOTAL_30 = 0.030


def _cfg(**kw) -> InsuranceConfig:
    base = dict(
        sleeve_frac=SLEEVE,
        monetize=PROD_MON,
        redeploy=PROD_REDEPLOY,
        hedge_kind="ladder",
    )
    base.update(kw)
    return InsuranceConfig(**base)


SWEEP: list[tuple[str, InsuranceConfig]] = [
    ("A_baseline_ladder24", _cfg(rungs=LADDER_2P4, hedge_budget=None)),
    ("B_dynamic_budget24", production_config()),
    ("C_deep_skew24", _cfg(rungs=build_ladder([(0.10, 0.15), (0.25, 0.35), (0.35, 0.50)], TOTAL_24), hedge_budget=None)),
    ("D_wide5_ladder24", _cfg(rungs=build_ladder(
        [(0.08, 1), (0.15, 1), (0.22, 1), (0.28, 1), (0.35, 1)], TOTAL_24,
    ), hedge_budget=None)),
    ("E_near_gamma24", _cfg(rungs=build_ladder([(0.05, 0.40), (0.10, 0.35), (0.15, 0.25)], TOTAL_24), hedge_budget=None)),
    ("F_dynamic_deep30", _cfg(
        rungs=build_ladder([(0.10, 0.15), (0.25, 0.35), (0.35, 0.50)], TOTAL_30),
        hedge_budget=HedgeBudgetPolicy(),
    )),
    ("G_backspread_12x30_24", _cfg(
        hedge_kind="backspread",
        hedge_budget=None,
        backspread=BackspreadHedge(otm_near=0.12, otm_far=0.30, far_ratio=3, premium_frac=TOTAL_24),
    )),
    ("H_backspread_10x35_24", _cfg(
        hedge_kind="backspread",
        hedge_budget=None,
        backspread=BackspreadHedge(otm_near=0.10, otm_far=0.35, far_ratio=3, premium_frac=TOTAL_24),
    )),
    ("I_hybrid_ladder70_bs30_24", _cfg(
        rungs=LADDER_2P4,
        hedge_kind="hybrid",
        hedge_budget=None,
        hybrid_ladder_frac=0.70,
        backspread=BackspreadHedge(otm_near=0.12, otm_far=0.30, far_ratio=3, premium_frac=TOTAL_24),
    )),
    ("J_hybrid_dynamic_deep24", _cfg(
        rungs=build_ladder([(0.10, 0.15), (0.25, 0.35), (0.35, 0.50)], TOTAL_24),
        hedge_kind="hybrid",
        hybrid_ladder_frac=0.65,
        hedge_budget=HedgeBudgetPolicy(),
        backspread=BackspreadHedge(otm_near=0.12, otm_far=0.30, far_ratio=3, premium_frac=TOTAL_24),
    )),
]

SHORT = {
    "A_baseline_ladder24": "A baseline",
    "B_dynamic_budget24": "B dynamic ★",
    "C_deep_skew24": "C deep-skew",
    "D_wide5_ladder24": "D wide-5",
    "E_near_gamma24": "E near-γ",
    "F_dynamic_deep30": "F dyn deep",
    "G_backspread_12x30_24": "G bs 12x30",
    "H_backspread_10x35_24": "H bs 10x35",
    "I_hybrid_ladder70_bs30_24": "I hybrid",
    "J_hybrid_dynamic_deep24": "J hyb dyn",
}


def _load_panel(extended: bool) -> pd.DataFrame:
    if extended:
        return load_vol_panel(start=EXTENDED_START, use_synthetic=True)
    return load_vol_panel(start=INCEPTION, use_synthetic=False)


def plot_sweep(
    df: pd.DataFrame,
    equity_paths: dict[str, pd.Series],
    dest: Path,
    *,
    era: str,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved: list[Path] = []
    labels = [SHORT.get(v, v) for v in df["variant"]]

    # 1) All equity curves
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    cmap = plt.cm.tab10
    for i, (name, eq) in enumerate(equity_paths.items()):
        lbl = SHORT.get(name, name)
        lw = 2.2 if "B_dynamic" in name else 1.0
        ls = "-" if "B_dynamic" in name else "-"
        alpha = 1.0 if name in ("A_baseline_ladder24", "B_dynamic_budget24") else 0.55
        ax.plot(eq.index, eq.values / 1e6, label=lbl, lw=lw, alpha=alpha, color=cmap(i % 10))
    ax.set_ylabel("Equity ($M)")
    ax.set_title(f"Hedge structure sweep — combined equity ({era})")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    p1 = dest / f"hedge_sweep_equity_{era}.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    saved.append(p1)

    # 2) A vs B highlight (production default)
    if "A_baseline_ladder24" in equity_paths and "B_dynamic_budget24" in equity_paths:
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax.plot(equity_paths["A_baseline_ladder24"].index,
                equity_paths["A_baseline_ladder24"].values / 1e6,
                label="A baseline (fixed budget)", lw=1.4, color="#64748b")
        ax.plot(equity_paths["B_dynamic_budget24"].index,
                equity_paths["B_dynamic_budget24"].values / 1e6,
                label="B dynamic budget (production ★)", lw=2.0, color="#0f766e")
        ax.set_ylabel("Equity ($M)")
        ax.set_title(f"Production default B vs baseline A ({era})")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        p2 = dest / f"hedge_sweep_B_vs_A_{era}.png"
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        saved.append(p2)

    # 3) Metrics bars
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(df))
    colors = ["#0f766e" if "B_dynamic" in v else "#6366f1" for v in df["variant"]]
    for ax, col, title in zip(
        axes,
        ["combined_CAGR", "combined_MaxDD", "combined_Sharpe"],
        ["CAGR", "Max drawdown", "Sharpe"],
    ):
        vals = df[col].astype(float)
        if col == "combined_MaxDD":
            vals = vals * 100
            ax.set_ylabel("%")
        elif col == "combined_CAGR":
            vals = vals * 100
            ax.set_ylabel("%")
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Live metrics by hedge structure ({era})", fontsize=11)
    p3 = dest / f"hedge_sweep_metrics_{era}.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    saved.append(p3)

    # 4) Crash scenario bars
    crash_cols = ["crash_mild_-20%", "crash_severe_-30%", "crash_volmageddon_-40%"]
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    w = 0.25
    for j, col in enumerate(crash_cols):
        offset = (j - 1) * w
        ax.bar(x + offset, df[col].astype(float) * 100, width=w, label=col.replace("crash_", ""), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Stylized crash P&L (% equity)")
    ax.set_title(f"Crash payouts by structure ({era})")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    p4 = dest / f"hedge_sweep_crash_{era}.png"
    fig.savefig(p4, dpi=130)
    plt.close(fig)
    saved.append(p4)

    # 5) Drawdown panel for top ladder variants only (exclude catastrophic backspreads)
    ladder_variants = [
        v for v in equity_paths
        if df.loc[df["variant"] == v, "combined_MaxDD"].iloc[0] > -0.25
    ]
    if ladder_variants:
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        for i, name in enumerate(ladder_variants):
            eq = equity_paths[name]
            dd = eq / eq.cummax() - 1.0
            ax.plot(dd.index, dd.values * 100, label=SHORT.get(name, name), lw=1.1, alpha=0.8)
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(f"Drawdown — ladder/hybrid variants ({era})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        p5 = dest / f"hedge_sweep_drawdown_{era}.png"
        fig.savefig(p5, dpi=130)
        plt.close(fig)
        saved.append(p5)

    return saved


def run_sweep(*, extended: bool = False) -> Path:
    era = "extended" if extended else "live"
    dest = Path("data/runs") / pd.Timestamp.today().strftime("%Y-%m-%d") / "bucket5" / era
    dest.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(extended)
    spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"))
    iv = (panel["vix"] / 100.0).rename("iv")
    synth = int(panel.get("synthetic", pd.Series(False, index=panel.index)).sum()) if "synthetic" in panel else 0

    print(f"{era} sample: {panel.index.min().date()} -> {panel.index.max().date()} ({len(panel)} rows)")
    if synth:
        print(f"  synthetic UVIX/SVIX days: {synth}")
    print("stack: liquidate + redeploy | sleeve_frac=20%\n")

    rows = []
    equity_paths: dict[str, pd.Series] = {}
    for name, cfg in SWEEP:
        print(f"  running {name}...", flush=True)
        res = run_insurance(panel, spx, iv, cfg)
        s = summarize(res, spx, panel)
        s["variant"] = name
        s["hedge_kind"] = cfg.hedge_kind
        s["dynamic_budget"] = cfg.hedge_budget is not None and cfg.hedge_budget.enabled
        s["realized_$"] = float(res["ladder"].attrs.get("realized_total", 0.0))
        s["redeploy_extra_$"] = float(res["bt"]["redeploy_extra"].iloc[-1])
        rows.append(s)
        equity_paths[name] = res["bt"]["combined_equity"].copy()
        if name == "B_dynamic_budget24":
            res["bt"].to_csv(dest / "production_B_path.csv", index=True)
            save_insurance_plots(res, dest, tag="production_B")

    df = pd.DataFrame(rows)
    cols = [
        "variant", "hedge_kind", "dynamic_budget", "ladder_per_roll_%",
        "combined_CAGR", "combined_MaxDD", "combined_Sharpe", "combined_Calmar",
        "realized_$", "redeploy_extra_$",
        "crash_mild_-20%", "crash_severe_-30%", "crash_volmageddon_-40%",
    ]
    df = df[[c for c in cols if c in df.columns]]
    out = dest / "hedge_structure_sweep.csv"
    df.to_csv(out, index=False)

    plots = plot_sweep(df, equity_paths, dest, era=era)
    manifest = dest / "PLOTS.txt"
    manifest.write_text("\n".join(str(p) for p in plots) + "\n", encoding="utf-8")

    pd.set_option("display.width", 240)
    print(f"\n=== HEDGE STRUCTURE SWEEP ({era}) ===")
    print(df.round(4).to_string(index=False))
    print(f"\nsaved -> {out}")
    for p in plots:
        print(f"plot  -> {p}")
    return dest


def main() -> None:
    extended = len(sys.argv) > 1 and sys.argv[1].lower() in ("extended", "ext", "2008")
    run_sweep(extended=extended)


if __name__ == "__main__":
    main()
