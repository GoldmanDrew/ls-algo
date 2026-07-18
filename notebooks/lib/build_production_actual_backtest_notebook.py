"""Build notebooks/Production_Actual_Backtest.ipynb"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "Production_Actual_Backtest.ipynb"


def md(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """# Production Actual Backtest (2026-02-27 → now)

**Single method (`prod` only):** each archived `etf_screened_today.csv` is sized with
**today's** full `generate_trade_plan` stack (B1/B2 + B4 opt2 → crash → smooth →
ratchet), carrying isolated state forward day-to-day. Spot `borrow_current` and
screener edge/opt2 inputs match production (no avg-borrow overlay). Archived
`proposed_trades.csv` is **not** used.

```text
screened(as-of D) + state[D-1] + held shorts from plan[D-1]
  → size_book_from_screened (full GTP)
  → plan[D] + state[D]
  → scale sleeve legs to YAML budget
  → simulate_book_from_plan_timeline (next-close, operator_5d Phase-2b, costs)
```

Capital / sleeve budgets come from live `strategy_config.yml`. B3 flow is excluded.
B1/B2 retarget on `stock_rebalance_clock=operator_5d` (every 5 business days, same
as live `rebalance_strategy`). Between operator days the book **share-holds**
(`stock_midweek_mode=rebal_only` — no midweek ramp or hedge repair). Each sleeve with
positive plan gross is scaled so sleeve gross equals the YAML sleeve budget.

**Sim controller** (`production_actual_backtest.turnover_pace.mode=hedge_safe_v1`):
persistent pair targets, ADV caps, and operator-day structural fills. Off-clock
B1/B2 trades are disabled under `rebal_only`. Hard exits still flatten immediately.
`legacy` retains EMA/per-leg pacing and `off` is full chase. Live Phase-2b is unchanged.

**Notebook-only exceptions:** `BLACKLIST_EXCEPT = [APLD, SMR, CBRS]` re-admits those
underlyings for this backtest without editing live YAML. `PRICE_PANEL_MIN_DAYS = 20`
relaxes the default 40-day panel gate so short-history names (e.g. CBRZ) can mark.

**Start:** `2026-02-27` (first screened archive after the Dec→Feb gap; sparse thereafter).
B5 only when GTP sizes it; no live locates / execution rejects.

When today's GTP cannot size an old screened file, the timeline **falls back** to that
date's archived `proposed_trades.csv` (and also ingests plan-only archive dates).
"""
    ),
    code(
        """from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.4f}".format

REPO = Path.cwd()
if not (REPO / "generate_trade_plan.py").exists():
    if (REPO.parent / "generate_trade_plan.py").exists():
        REPO = REPO.parent
sys.path.insert(0, str(REPO))

RUN_DATE = "2026-07-13"          # price panel date
START = "2026-02-27"
PRE_ARCHIVE_POLICY = "cash"      # "cash" | "skip"
# Temporarily re-admit these underlyings for this notebook backtest only
# (live strategy_config.yml blacklist is unchanged).
BLACKLIST_EXCEPT = ["APLD", "SMR", "CBRS"]
# CBRZ has <40 metrics days as of RUN_DATE; default panel gate drops it.
PRICE_PANEL_MIN_DAYS = 20
# Notebook-only B4 borrow band (does NOT write strategy_config.yml).
# Live remains 70/90; this run uses 60/80 and shifts opt2 borrow_ramp by the
# same delta (80/120 → 70/110). Requires REUSE_PLANS=False to rebuild plans.
NOTEBOOK_B4_BORROW = {
    "entry_borrow_cap": 0.60,
    "keep_borrow_cap": 0.80,
    "shift_ramp_with_band": True,
}
# True = rebuild/resim. REUSE_PLANS=False when borrow overrides change so GTP
# re-sizes under 60/80. True = skip GTP and resim from cached plans only.
FORCE_RERUN = True
REUSE_PLANS = False  # must be False for NOTEBOOK_B4_BORROW to affect plans
OUT_BASE = REPO / "notebooks" / "output" / "production_actual_bt"

# Embed plots inline so figures show after Run All (including after rebuild).
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.max_open_warning"] = 100
from IPython.display import display as _ip_display
# Prefer explicit display(fig) for multi-figure cells; plt.show() also works inline.

SLEEVE_LABELS = {
    "core_leveraged": "B1 core",
    "yieldboost": "B2 yieldboost",
    "inverse_decay_bucket4": "B4 inverse",
    "volatility_etp_bucket5": "B5 vol ETP",
}
SLEEVE_ORDER = list(SLEEVE_LABELS)

print("REPO", REPO)
print("RUN_DATE", RUN_DATE, "START", START, "mode=prod")
print("BLACKLIST_EXCEPT", BLACKLIST_EXCEPT, "PRICE_PANEL_MIN_DAYS", PRICE_PANEL_MIN_DAYS)
print("FORCE_RERUN", FORCE_RERUN, "REUSE_PLANS", REUSE_PLANS)
print("NOTEBOOK_B4_BORROW", NOTEBOOK_B4_BORROW)
print(
    "B4 realism: membership_clock=operator_5d, resize_bands=on, ratchet_guard=on, "
    "empty_plan=hold, purgatory_model_zero=hold; charts=ALL B4 pairs"
)
"""
    ),
    md("## Archive coverage"),
    code(
        """from scripts.production_actual_backtest import archive_coverage_summary, list_archived_plan_dates, list_archived_screened_dates

cov = archive_coverage_summary(START)
display(cov)
print("n proposed_trades dates:", len(list_archived_plan_dates()), "(unused by prod)")
print("n screened dates:", len(list_archived_screened_dates()))
print("first screened:", list_archived_screened_dates()[0].date() if list_archived_screened_dates() else None)
"""
    ),
    md(
        """## Production-accounting parity checklist

`Implemented` means the ledger models the behavior directly. `Proxy` is
intentionally visible in the cost and sensitivity charts. `Gap` should not be
read as production-accurate until the missing artifact is archived.
"""
    ),
    code(
        """assumption_audit = pd.DataFrame([
    ["Gross sizing", "Dynamic NAV × target gross", "Scale sleeve legs to YAML budget, then × NAV/capital", "Implemented"],
    ["Plan source", "Live generate_trade_plan", "Full GTP sizing on archived screened (isolated state)", "Implemented"],
    ["Plan timing", "Signal known after close", "T plan executes next available close; P&L starts next session", "Implemented"],
    ["Between rebals", "Hold shares; legs drift", "Signed ETF/underlying notionals drift with each leg", "Implemented"],
    ["Leg schema", "Underlying target + ETF target", "long_usd=underlying; short_usd=ETF; explicit columns preferred", "Implemented"],
    ["Rebalance", "Phase-2b hysteresis + operator clock", "B1/B2 operator_5d + rebal_only midweek; 12%/4%/$250 bands", "Implemented"],
    ["Purgatory", "Reduce toward model target; never increase pair gross", "Holdings-aware reduce_only constraint; zero target exits", "Implemented"],
    ["Same-run churn", "Pre-net Phase 2b against projected Phase 3; audited risk overrides only", "One terminal target per pair/day; avoided round trips and override turnover reported", "Implemented"],
    ["Slippage", "Broker/fill dependent", "20 bp on every traded dollar, including opening trades", "Proxy"],
    ["Commission", "Clear Street low-touch", "$0.0035/share by leg", "Implemented"],
    ["Borrow", "Point-in-time by short symbol", "Screened spot borrow_current carried until the next plan", "Implemented"],
    ["Short credit", "IBKR interest on short proceeds", "3.8% annual on short notional / Actual-360", "Implemented"],
    ["Margin debit", "OBFR + 45 bp, Actual/360", "4.00% benchmark fallback + 45 bp, Actual/360", "Proxy"],
    ["Prices", "Total-return / split-safe marks", "Adjusted-close panel + Flex/override/heuristic split repair", "Implemented"],
    ["Missing bars", "Carry last mark; cannot trade", "Zero-return stale mark; blocked close/entry is audited", "Implemented"],
    ["Price integrity", "Yahoo referee + price_patches + flex/overrides", "RDWU-class phantoms patched before mark", "Implemented"],
    ["B4 cadence", "TR/VCR h + interval in book ledger", "b4_execution=cadence (default); weekly_plan_legs legacy", "Implemented"],
    ["Delistings", "last_trade flatten via data/delistings.csv", "Force exit on/after last print; panel cutoff", "Implemented"],
    ["Share rounding", "Whole shares / broker lots", "Dollar-notional targets", "Gap"],
    ["Locates", "Can reject or resize shorts", "Screened universe at sizing time; no execution reject", "Gap"],
    ["B4 stack", "opt2 → crash → smooth → ratchet", "Same stack; state isolated + ratchet from prior plan", "Implemented"],
    ["Archive window", "Full history", "Start 2026-02-27; pre-Apr-25 edge shimmed from net_decay when net_edge_p50 missing", "Implemented"],
], columns=["Item", "Production", "This backtest", "Status"])
display(assumption_audit)
"""
    ),
    md(
        """## B4 crash-budget + scale-to-budget snapshot (live GTP)

`crash_budget_mult < 1` means the pair was trimmed by ρ·budget/L before the
sleeve refill. With `scale_to_budget: true`, those trims are then scaled
pro-rata so sleeve gross ≈ YAML `target_weight`.
"""
    ),
    code(
        """cb_candidates = [
    REPO / "data" / "runs" / RUN_DATE / "b4_crash_budget.csv",
    REPO / "data" / "runs" / RUN_DATE / "b4_hedge_cadence" / "b4_crash_budget.csv",
]
cb_path = next((p for p in cb_candidates if p.is_file()), cb_candidates[0])
if cb_path.is_file():
    cb = pd.read_csv(cb_path)
    cols = [c for c in [
        "ETF", "Underlying", "weight_solved", "weight_capped", "weight_final",
        "gross_solved_usd", "gross_capped_usd", "gross_final_usd", "cap_usd",
        "crash_budget_mult", "scale_to_budget", "scale_mult",
        "L", "C", "runup", "tail", "hedge_ratio", "crash_l_source",
    ] if c in cb.columns]
    display(cb[cols].sort_values("crash_budget_mult").round(4))

    solved = float(cb["gross_solved_usd"].sum()) if "gross_solved_usd" in cb.columns else np.nan
    capped = float(cb["gross_capped_usd"].sum()) if "gross_capped_usd" in cb.columns else np.nan
    final = float(cb["gross_final_usd"].sum()) if "gross_final_usd" in cb.columns else capped
    scale_mult = float(cb["scale_mult"].iloc[0]) if "scale_mult" in cb.columns and len(cb) else np.nan
    scale_on = bool(cb["scale_to_budget"].iloc[0]) if "scale_to_budget" in cb.columns and len(cb) else False
    print(
        f"solved ${solved:,.0f} → post-cap ${capped:,.0f} → final ${final:,.0f} "
        f"(scale_to_budget={scale_on}, scale_mult={scale_mult:.3f}); "
        f"n_capped={(cb['crash_budget_mult'] < 0.999).sum()}/{len(cb)}"
        if "crash_budget_mult" in cb.columns else f"rows={len(cb)}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    if {"ETF", "Underlying", "crash_budget_mult"}.issubset(cb.columns):
        top = cb.sort_values("crash_budget_mult").head(12)
        ax.barh(top["ETF"] + "/" + top["Underlying"], top["crash_budget_mult"], color="#c44e52")
        ax.axvline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("crash_budget_mult (1 = uncapped before scale)")
        ax.set_title("B4 crash-budget multipliers")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
    ax = axes[1]
    if {"C", "L", "crash_budget_mult", "Underlying"}.issubset(cb.columns):
        ax.scatter(cb["C"], cb["L"], c=cb["crash_budget_mult"], cmap="RdYlGn", s=60, edgecolor="k", lw=0.4)
        for _, r in cb.iterrows():
            ax.annotate(r["Underlying"], (r["C"], r["L"]), fontsize=7, alpha=0.8)
        ax.set_xlabel("conditional crash C")
        ax.set_ylabel("pair loss per gross $ (L)")
        ax.set_title("Crash exposure vs loss intensity")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(f"no crash-budget telemetry at {cb_path} — re-run generate_trade_plan.py for {RUN_DATE}")
"""
    ),
    md(
        """## B4 sizing waterfall (opt2 → cap/scale → smooth → legs)

`b4_sizing_waterfall.csv` is the per-pair audit of the four-step stack.
"""
    ),
    code(
        """wf_candidates = [
    REPO / "data" / "runs" / RUN_DATE / "b4_sizing_waterfall.csv",
    REPO / "data" / "runs" / RUN_DATE / "b4_hedge_cadence" / "b4_sizing_waterfall.csv",
]
wf_path = next((p for p in wf_candidates if p.is_file()), wf_candidates[0])
if wf_path.is_file():
    wf = pd.read_csv(wf_path)
    show = [c for c in [
        "ETF", "Underlying", "weight_solved", "weight_capped", "weight_smoothed",
        "gross_decay_score_usd", "gross_after_caps_usd", "gross_after_smooth_usd",
        "gross_final_usd", "inverse_short_final_usd", "underlying_short_final_usd",
        "L", "C", "rho_effective", "crash_l_source",
    ] if c in wf.columns]
    sort_col = "weight_smoothed" if "weight_smoothed" in wf.columns else show[0]
    display(wf[show].sort_values(sort_col, ascending=False).round(4))

    stages = [
        ("decay score", "gross_decay_score_usd"),
        ("after cap/scale", "gross_after_caps_usd"),
        ("after smooth", "gross_after_smooth_usd"),
        ("final", "gross_final_usd"),
    ]
    totals = {label: float(wf[col].sum()) for label, col in stages if col in wf.columns}
    print("sleeve gross by stage:")
    for k, v in totals.items():
        print(f"  {k:16s} ${v:,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if {"weight_solved", "weight_capped", "weight_smoothed"}.issubset(wf.columns):
        plot_df = wf.sort_values("weight_smoothed", ascending=True).tail(12)
        y = np.arange(len(plot_df))
        h = 0.25
        ax.barh(y - h, plot_df["weight_solved"], height=h, label="solved", color="#9ecae1")
        ax.barh(y, plot_df["weight_capped"], height=h, label="capped+scale", color="#6baed6")
        ax.barh(y + h, plot_df["weight_smoothed"], height=h, label="smoothed", color="#2171b5")
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["ETF"] + "/" + plot_df["Underlying"], fontsize=8)
        ax.set_xlabel("pair weight (fraction of B4 sleeve)")
        ax.set_title("Weight path: solved → capped → smoothed")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    if totals:
        ax.bar(list(totals.keys()), list(totals.values()), color=["#9ecae1", "#6baed6", "#2171b5", "#08306b"][:len(totals)])
        ax.set_ylabel("Sleeve gross ($)")
        ax.set_title("Book gross by sizing stage")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(f"no waterfall at {wf_path} — re-run generate_trade_plan.py for {RUN_DATE}")
"""
    ),
    md(
        """## Run production historical sizing

Outputs land under `notebooks/output/production_actual_bt/` (`prod_sizing_diag.csv`,
`pair_stats.csv`, `daily_diagnostics.csv`, …). Full GTP per screened day is
slower than the old mirror path.
"""
    ),
    code(
        """from scripts.production_actual_backtest import run_production_actual_backtest
from scripts.sizing_tilt_cadence_bt import perf
import json

outdir = OUT_BASE
summary_path = outdir / "sleeve_summary.csv"
report_path = outdir / "report.json"
if FORCE_RERUN or not summary_path.exists():
    print("===== running mode=prod =====")
    report = run_production_actual_backtest(
        run_date=RUN_DATE,
        start=START,
        outdir=outdir,
        mode="prod",
        pre_archive_policy=PRE_ARCHIVE_POLICY,
        blacklist_except=BLACKLIST_EXCEPT,
        price_panel_min_days=PRICE_PANEL_MIN_DAYS,
        reuse_plans=bool(REUSE_PLANS) and (outdir / "plans").is_dir() and any((outdir / "plans").glob("*.csv")),
        notebook_b4_borrow=NOTEBOOK_B4_BORROW,
    )
else:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"loaded cached prod from {outdir}")
    be = (report.get("prod_stats") or {}).get("blacklist_except") or []
    if sorted(x.upper() for x in BLACKLIST_EXCEPT) != sorted(str(x).upper() for x in be):
        print(
            "WARN: cached report blacklist_except", be,
            "!= notebook", BLACKLIST_EXCEPT, "— set FORCE_RERUN=True"
        )

summary = pd.read_csv(summary_path)
nav = pd.read_csv(outdir / "daily_nav.csv", index_col=0, parse_dates=True)
pair_stats_df = pd.read_csv(outdir / "pair_stats.csv") if (outdir / "pair_stats.csv").exists() else pd.DataFrame()
sleeve_pnl_df = pd.read_csv(outdir / "sleeve_pnl.csv") if (outdir / "sleeve_pnl.csv").exists() else pd.DataFrame()
pair_daily_df = (
    pd.read_csv(outdir / "pair_daily_pnl.csv", parse_dates=["date"])
    if (outdir / "pair_daily_pnl.csv").exists()
    else pd.DataFrame()
)

display(summary)
print("Book:", report.get("book"))
print("prod_stats:", report.get("prod_stats"))
print("pair_stats", len(pair_stats_df), "pair_daily rows", len(pair_daily_df))
"""
    ),
    md(
        """## Hedge-safe calibration and pending-target diagnostics

The isolated calibration replays the **same cached plans and price panel** through
`off`, `legacy`, and `hedge_safe_v1`. The comparison shows the execution/risk
trade-off rather than treating minimum turnover as the only objective. Hedge-safe
freezes B1/B2 pair-gross destinations between weekly decisions while refreshing
daily hedge and liquidity metadata. Weekly destinations convexly blend prior and
new confirmed pair gross, smoothing pair switches without sleeve renormalization.
Its sensitivity grid gates missing Delta, orphan legs, deployment fidelity, B4
cadence, and the $10m turnover ceiling. It then ranks stock-residual breach
group-days first, maximum residual drift second, turnover/cost next, and stability
last, without using PnL. Hedge-safe daily budget uses the
larger of deployed and confirmed desired gross, avoiding self-starvation.
Phase-3 hedge repair and acceptance use the B1/B2 stock residual only. Raw
all-sleeve net drift is displayed separately as context; intentional B4/B5
structural exposure does not trigger repair. Drift diagnostics use each simulated
position's persisted Delta, including decaying pairs absent from the active plan;
missing-Delta rows and underlying groups are explicit hard-gate diagnostics.

`pending_target_audit.csv` is the pair-session controller ledger. It records the
current, desired, and next gross; hedge drift before/after; target age; allocated
and deferred turnover; block reason; and priority. Weekly target-formation rows
also show raw plan, prior, and blended structural pair gross.
"""
    ),
    code(
        """CAL_DIR = OUT_BASE / "hedge_safe_calibration"
cal_path = CAL_DIR / "comparison.csv"
if cal_path.exists():
    cal = pd.read_csv(cal_path)
    show_cols = [c for c in [
        "arm", "turnover_usd", "txn_cost_usd", "end_usd", "total_return",
        "cagr", "vol", "sharpe", "maxdd", "hedge_breach_group_days",
        "missing_delta_rows", "missing_delta_group_days",
        "raw_all_sleeve_hedge_breach_group_days",
        "raw_all_sleeve_missing_delta_rows",
        "raw_all_sleeve_missing_delta_group_days",
        "median_deployed_desired_gross_ratio",
        "p10_deployed_desired_gross_ratio",
        "ending_deployed_desired_gross_ratio", "n_b4_cadence_rebals",
        "max_abs_hedge_net_pct", "orphan_pair_days", "max_deferred_age",
        "deferred_turnover_usd", "blocked_reason_counts_json",
    ] if c in cal.columns]
    display(cal[show_cols])
    if {"arm", "turnover_usd", "hedge_breach_group_days"}.issubset(cal.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(cal["arm"], cal["turnover_usd"] / 1e6, color=["#999999", "#4c72b0", "#55a868"])
        axes[0].set_ylabel("Turnover ($mm)")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].set_title("Execution turnover")
        axes[1].bar(cal["arm"], cal["hedge_breach_group_days"], color=["#999999", "#4c72b0", "#55a868"])
        axes[1].set_ylabel("Underlying-group breach days")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].set_title("B1/B2 stock-residual hedge breaches")
        plt.tight_layout()
        plt.show()
else:
    print("no calibration comparison at", cal_path)

sensitivity_path = CAL_DIR / "sensitivity.csv"
if sensitivity_path.exists():
    sensitivity = pd.read_csv(sensitivity_path)
    sensitivity_cols = [c for c in [
        "target_blend_alpha", "max_daily_turnover_pct", "remaining_gap_rate", "turnover_usd",
        "txn_cost_usd", "hedge_breach_group_days", "orphan_pair_days",
        "missing_delta_rows", "missing_delta_group_days",
        "raw_all_sleeve_hedge_breach_group_days",
        "median_deployed_desired_gross_ratio",
        "p10_deployed_desired_gross_ratio",
        "ending_deployed_desired_gross_ratio", "n_b4_cadence_rebals",
        "hedge_safety_pass", "deployment_pass", "b4_cadence_pass",
        "hard_gates_pass", "turnover_target_pass", "p10_deployment_preferred",
        "stable_region_pass", "selected",
    ] if c in sensitivity.columns]
    display(sensitivity[sensitivity_cols].sort_values(
        ["target_blend_alpha", "max_daily_turnover_pct", "remaining_gap_rate"]
    ))
else:
    print("no hedge-safe sensitivity grid at", sensitivity_path)

pending_candidates = [
    OUT_BASE / "pending_target_audit.csv",
    CAL_DIR / "hedge_safe_v1" / "pending_target_audit.csv",
]
pending_path = next((p for p in pending_candidates if p.exists()), pending_candidates[0])
pending = pd.read_csv(pending_path, parse_dates=["date"]) if pending_path.exists() else pd.DataFrame()
if pending.empty:
    print("no pending-target ledger found")
else:
    print("pending ledger:", pending_path, "rows=", len(pending))
    if "block_reason" in pending.columns:
        display(pending["block_reason"].fillna("allocated").value_counts().to_frame("pair_sessions"))
    age = pd.to_numeric(pending.get("target_age"), errors="coerce")
    deferred = pd.to_numeric(pending.get("deferred_turnover_usd"), errors="coerce").fillna(0)
    print(
        f"max target age={age.max():.0f} | p95 age={age.quantile(.95):.1f} | "
        f"cumulative daily deferred demand=${deferred.sum():,.0f}"
    )
    display(pending.sort_values(["target_age", "deferred_turnover_usd"], ascending=False).head(20))

drift_path = CAL_DIR / "hedge_safe_v1" / "hedge_drift_daily.csv"
drift = pd.read_csv(drift_path, parse_dates=["date"]) if drift_path.exists() else pd.DataFrame()
if not drift.empty:
    breaches = drift[drift["breach"].fillna("").ne("")]
    print(
        f"hedge drift groups={len(drift):,} breaches={len(breaches):,} "
        f"max |net/gross|={drift['hedge_net_pct'].abs().max():.2%}"
    )
    display(breaches.reindex(breaches["hedge_net_pct"].abs().sort_values(ascending=False).index).head(20))
"""
    ),
    md(
        """## Book NAV, bucket returns, and risk stats

Book NAV path and cumulative sleeve PnL (synthetic NAV = capital + cum sleeve PnL).

**Primary return metrics (deployed capital only):**
- **rog_deployed** = PnL on days with gross > $1 / mean(gross on those days)
- **rog_deployed_ann** = simple annualization × 252 / n_days
- **roc_deployed** = same with mean(|net|) on deployed days
- **deployed_day_frac** = fraction of sessions with capital at work

Calendar ``rog`` / ``roc`` (all days, including flat) are legacy — do not use them
to judge under-ramped sleeves. Book ``roc`` uses netted net and is not "capital at work"
when `net_shared_underlyings=true` — prefer book **rog_deployed**.

The CAGR / vol / Sharpe table on synthetic sleeve NAV paths is secondary — do not
read sleeve `total_ret` as return on deployed capital.
"""
    ),
    code(
        """from scripts.production_actual_backtest import compute_sleeve_return_metrics

def _series_stats(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 2:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan,
                "total_ret": np.nan, "start_usd": np.nan, "end_usd": np.nan}
    out = perf(s)
    out["total_ret"] = float(s.iloc[-1] / s.iloc[0] - 1.0) if float(s.iloc[0]) > 0 else np.nan
    out["start_usd"] = float(s.iloc[0])
    out["end_usd"] = float(s.iloc[-1])
    return out

if nav is None or nav.empty:
    print("no daily_nav.csv — re-run the prod backtest cell above")
else:
    book_col = "BOOK_NAV" if "BOOK_NAV" in nav.columns else nav.columns[0]
    book = nav[book_col].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1.0]})
    ax = axes[0]
    if len(book):
        ax.plot(book.index, book.values, color="black", lw=2.0, label="BOOK")
    ax.set_ylabel("NAV ($)")
    ax.set_title(f"Prod book NAV  {START} → now")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for s in SLEEVE_ORDER:
        if s not in nav.columns:
            continue
        series = nav[s].dropna()
        if len(series) < 2 or float(series.iloc[0]) == 0:
            continue
        r = series / float(series.iloc[0]) - 1.0
        ax.plot(r.index, r.values, label=SLEEVE_LABELS[s], lw=1.6)
    if len(book) >= 2 and float(book.iloc[0]) > 0:
        br = book / float(book.iloc[0]) - 1.0
        ax.plot(br.index, br.values, label="BOOK", color="k", lw=2.0)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("Cumulative return")
    ax.set_title("Per-bucket cumulative returns (synthetic NAV = capital + cum PnL)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # EOD-style ROC / ROG (primary)
    ret_path = OUT_BASE / "sleeve_return_metrics.csv"
    if ret_path.exists():
        ret_tbl = pd.read_csv(ret_path)
    else:
        diag_tmp = (
            pd.read_csv(OUT_BASE / "daily_diagnostics.csv", parse_dates=["date"])
            if (OUT_BASE / "daily_diagnostics.csv").exists()
            else pd.DataFrame()
        )
        ret_tbl = compute_sleeve_return_metrics(diag_tmp) if not diag_tmp.empty else pd.DataFrame()

    if not ret_tbl.empty:
        show = ret_tbl.copy()
        show["sleeve"] = show["sleeve"].map(lambda x: SLEEVE_LABELS.get(x, x))
        for c in (
            "pnl_usd",
            "pnl_deployed_usd",
            "avg_gross_deployed",
            "avg_abs_net_deployed",
            "avg_gross_cap",
        ):
            if c in show.columns:
                show[c] = show[c].map(lambda v: f"{float(v):,.0f}" if pd.notna(v) else "")
        for c in ("rog_deployed", "rog_deployed_ann", "roc_deployed", "deployed_day_frac", "rog", "roc"):
            if c in show.columns:
                show[c] = show[c].map(lambda v: f"{float(v):.2%}" if pd.notna(v) else "n/a")
        print("Bucket returns on DEPLOYED capital (primary):")
        display(show.set_index("sleeve")[
            [c for c in [
                "pnl_deployed_usd", "avg_gross_deployed", "rog_deployed", "rog_deployed_ann",
                "roc_deployed", "deployed_day_frac", "n_deployed_days", "n_days",
            ] if c in show.columns]
        ])
        print("Legacy calendar averages (includes flat days — secondary):")
        display(show.set_index("sleeve")[
            [c for c in ["pnl_usd", "avg_gross_cap", "rog", "roc"] if c in show.columns]
        ])
    else:
        print("no sleeve_return_metrics — re-run prod cell")

    # Secondary path stats on synthetic sleeve NAV (not ROC/ROG)
    stat_rows = [{"sleeve": "BOOK", **_series_stats(book)}]
    for s in SLEEVE_ORDER:
        if s in nav.columns:
            stat_rows.append({"sleeve": SLEEVE_LABELS.get(s, s), **_series_stats(nav[s])})
    stats_tbl = pd.DataFrame(stat_rows)
    b = report.get("book") or {}
    for k in ("cagr", "vol", "sharpe", "maxdd", "end_usd"):
        if b.get(k) is not None and len(stats_tbl):
            stats_tbl.loc[stats_tbl["sleeve"] == "BOOK", k] = b.get(k)
    cols = [c for c in ["start_usd", "end_usd", "total_ret", "cagr", "vol", "sharpe", "maxdd"] if c in stats_tbl.columns]
    print("Path stats on synthetic NAV (secondary; total_ret ≠ deployed ROG):")
    display(stats_tbl.set_index("sleeve")[cols].round(4))
"""
    ),
    md(
        """## Production-debug ledger checks

These checks fail loudly on accounting drift. The daily ledger must reconcile
price P&L less borrow, margin, and transaction costs exactly to the NAV change.
"""
    ),
    code(
        """diag_dir = OUT_BASE
diag_path = diag_dir / "daily_diagnostics.csv"
rebalance_path = diag_dir / "rebalance_audit.csv"

daily_diag = pd.read_csv(diag_path, parse_dates=["date"]) if diag_path.exists() else pd.DataFrame()
rebalance_diag = pd.read_csv(rebalance_path, parse_dates=["date"]) if rebalance_path.exists() else pd.DataFrame()

if daily_diag.empty:
    print(f"No production-debug ledger at {diag_path}; re-run with FORCE_RERUN=True.")
else:
    max_resid = float(daily_diag["pnl_recon_residual"].abs().max())
    print(f"mode=prod days={len(daily_diag):,}  max |P&L residual|=${max_resid:,.8f}")
    assert max_resid < 0.01, f"P&L reconciliation failed: ${max_resid:,.4f}"
    checks = pd.DataFrame({
        "check": ["P&L reconciliation", "max gross leverage", "stale-mark days", "rebalance events", "opening/resize costs"],
        "value": [
            max_resid,
            daily_diag["gross_leverage"].max(),
            int(((daily_diag["n_stale_etf"] + daily_diag["n_stale_underlying"]) > 0).sum()),
            len(rebalance_diag),
            daily_diag["daily_txn_cost"].sum(),
        ],
    })
    display(checks)
"""
    ),
    md(
        """## Risk, leverage, and exposure dashboard

Fastest graph for an unintended leverage reset, directional drift, stuck
position count, or a drawdown that coincides with a plan switch.
"""
    ),
    code(
        """if not daily_diag.empty:
    d = daily_diag.set_index("date")
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.2, 1.2, 1.2]})

    ax = axes[0]
    ax.plot(d.index, d["book_equity"], color="black", lw=1.8, label="book NAV")
    ax2 = ax.twinx()
    ax2.fill_between(d.index, d["drawdown"], 0, color="#c44e52", alpha=0.22, label="drawdown")
    ax.set_ylabel("NAV ($)")
    ax2.set_ylabel("Drawdown")
    ax.set_title("prod: NAV and drawdown")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(d.index, d["gross_leverage"], color="#4c72b0", label="gross / NAV")
    cfg_lev = float(report.get("gross_leverage") or np.nan)
    ax.axhline(cfg_lev, color="black", ls="--", lw=1, label=f"configured {cfg_lev:g}x")
    ax2 = ax.twinx()
    ax2.plot(d.index, d["net_exposure_pct"], color="#dd8452", alpha=0.8, label="net / NAV")
    ax.set_ylabel("Gross leverage")
    ax2.set_ylabel("Net exposure / NAV")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    eq = d["book_equity"].replace(0, np.nan)
    ax.plot(d.index, d["long_notional"] / eq, label="long / NAV", color="#55a868")
    ax.plot(d.index, d["short_notional"] / eq, label="short / NAV", color="#c44e52")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylabel("Leg leverage")
    ax.grid(True, alpha=0.25)

    ax = axes[3]
    ax.bar(d.index, d["turnover_usd"], color="#8172b2", alpha=0.7, label="turnover")
    ax2 = ax.twinx()
    ax2.plot(d.index, d["n_positions"], color="black", lw=1.1, label="positions")
    ax.set_ylabel("Daily turnover ($)")
    ax2.set_ylabel("Open pairs")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """## P&L bridge and explicit cost drag

Gross price P&L must bridge to net P&L after borrow, margin debit, and execution
costs.
"""
    ),
    code(
        """if not daily_diag.empty:
    d = daily_diag.set_index("date")
    gross_cum = d["daily_price_pnl"].cumsum()
    borrow_cum = d["daily_borrow_cost"].cumsum()
    credit_cum = d["daily_short_credit"].cumsum() if "daily_short_credit" in d.columns else 0.0
    margin_cum = d["daily_margin_cost"].cumsum()
    txn_cum = d["daily_txn_cost"].cumsum()
    net_cum = d["daily_net_pnl"].cumsum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(d.index, gross_cum, label="gross price P&L", lw=1.8)
    ax.plot(d.index, net_cum, label="net P&L", lw=2.0, color="black")
    ax.plot(d.index, -borrow_cum, label="− borrow", ls="--")
    if "daily_short_credit" in d.columns:
        ax.plot(d.index, credit_cum, label="+ short credit 3.8%", ls="--", color="#55a868")
    ax.plot(d.index, -margin_cum, label="− margin", ls="--")
    ax.plot(d.index, -txn_cum, label="− transaction", ls="--")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Cumulative P&L bridge")
    ax.set_ylabel("Cumulative dollars")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)

    totals = pd.Series({
        "gross price": d["daily_price_pnl"].sum(),
        "borrow": -d["daily_borrow_cost"].sum(),
        "short credit": float(d["daily_short_credit"].sum()) if "daily_short_credit" in d.columns else 0.0,
        "margin": -d["daily_margin_cost"].sum(),
        "transaction": -d["daily_txn_cost"].sum(),
        "net": d["daily_net_pnl"].sum(),
    })
    colors = ["#4c72b0" if x >= 0 else "#c44e52" for x in totals]
    axes[1].bar(totals.index, totals.values, color=colors)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_title("Full-period attribution")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    display(totals.rename("usd").to_frame())
"""
    ),
    md(
        """## Cost-assumption sensitivity (first-order)

Holds the executed path fixed and restates only cost dollars.
"""
    ),
    code(
        """if not daily_diag.empty:
    d = daily_diag.set_index("date")
    rep = report
    end_nav = float(rep["book"]["end_usd"])
    knobs = rep.get("rebalance_knobs") or {}
    current_slip_bps = float(knobs.get("slippage_bps", 20.0))
    turnover = float(d["turnover_usd"].sum())
    current_slip = turnover * current_slip_bps / 1e4
    total_txn = float(d["daily_txn_cost"].sum())
    commission = max(0.0, total_txn - current_slip)
    current_margin_rate = float(knobs.get("margin_rate_annual", 0.0445))
    current_margin = float(d["daily_margin_cost"].sum())
    current_borrow = float(d["daily_borrow_cost"].sum())
    current_credit = float(d["daily_short_credit"].sum()) if "daily_short_credit" in d.columns else 0.0

    slip_grid = np.array([5, 10, 20, 30], dtype=float)
    slip_ends = end_nav + current_slip - turnover * slip_grid / 1e4
    margin_grid = np.array([0.025, 0.035, 0.0445, 0.055, 0.07])
    margin_ends = end_nav + current_margin - current_margin * margin_grid / max(current_margin_rate, 1e-9)
    borrow_mult = np.array([0.5, 1.0, 1.5, 2.0])
    borrow_ends = end_nav + current_borrow - current_borrow * borrow_mult

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(slip_grid, slip_ends, marker="o")
    axes[0].axvline(current_slip_bps, color="black", ls="--")
    axes[0].set_title("Slippage sensitivity")
    axes[0].set_xlabel("Slippage (bp / traded $)")
    axes[0].set_ylabel("Restated end NAV ($)")
    axes[1].plot(100 * margin_grid, margin_ends, marker="o", color="#dd8452")
    axes[1].axvline(100 * current_margin_rate, color="black", ls="--")
    axes[1].set_title("Margin-rate sensitivity")
    axes[1].set_xlabel("Annual debit rate (%)")
    axes[2].plot(borrow_mult, borrow_ends, marker="o", color="#c44e52")
    axes[2].axvline(1.0, color="black", ls="--")
    axes[2].set_title("Borrow sensitivity")
    axes[2].set_xlabel("Plan borrow × multiplier")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    print(f"turnover=${turnover:,.0f}; implied slippage=${current_slip:,.0f}; "
          f"commission=${commission:,.0f}; margin=${current_margin:,.0f}; "
          f"borrow=${current_borrow:,.0f}; short_credit=${current_credit:,.0f}")
"""
    ),
    md(
        """## Plan deployment, churn, and open pair count

Middle panel now shows **open pairs** (`n_positions` from the daily ledger), not
just resize counts. High turnover previously came from retargeting on every
screened-day plan change; GTP now trades weekly (Friday) with the latest plan.
"""
    ),
    code(
        """if not rebalance_diag.empty:
    r = rebalance_diag.set_index("date")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(r.index, r["target_planned_gross_usd"], label="planned gross", lw=1.8)
    axes[0].plot(r.index, r["target_tradeable_gross_usd"], label="panel-tradeable target", lw=1.4)
    axes[0].plot(r.index, r["deployed_gross_usd"], label="deployed after close", lw=1.4)
    axes[0].fill_between(r.index, r["target_tradeable_gross_usd"], r["target_planned_gross_usd"],
                         color="#c44e52", alpha=0.2, label="missing-panel gross")
    axes[0].set_ylabel("Gross ($)")
    axes[0].legend(loc="best", ncol=2, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    width = 1.5
    axes[1].bar(r.index, r.get("n_added", 0), width=width, label="adds", color="#55a868")
    axes[1].bar(r.index, -r.get("n_exited", 0), width=width, label="exits", color="#c44e52")
    if not daily_diag.empty and "n_positions" in daily_diag.columns:
        pos = daily_diag.set_index("date")["n_positions"]
        axes[1].plot(pos.index, pos.values, label="open pairs", color="black", lw=1.6)
    axes[1].plot(r.index, r.get("n_resized", 0), label="resizes", color="#4c72b0", alpha=0.7)
    axes[1].set_ylabel("Pair count")
    axes[1].legend(loc="best", ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].bar(r.index, r["turnover_usd"], width=width, color="#8172b2", label="turnover")
    ax2 = axes[2].twinx()
    ax2.plot(r.index, r["blocked_pairs"], color="black", marker=".", label="blocked pairs")
    axes[2].set_ylabel("Turnover ($)")
    ax2.set_ylabel("Blocked pairs")
    axes[2].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    display(r.sort_values("turnover_usd", ascending=False).head(15)[[
        "plan_date", "turnover_usd", "txn_cost_usd", "n_added", "n_exited",
        "n_resized", "blocked_pairs", "untradeable_plan_gross_usd"
    ]])
"""
    ),
    md(
        """## Data quality, concentration, and event-day triage

Stale marks should cluster around genuine listing/calendar gaps, not unexplained
P&L events. Concentration should be reviewed alongside large daily moves.
"""
    ),
    code(
        """if not daily_diag.empty:
    d = daily_diag.set_index("date")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(d.index, d["n_stale_etf"], label="ETF stale")
    axes[0].plot(d.index, d["n_stale_underlying"], label="underlying stale", alpha=0.8)
    axes[0].set_title("Stale marks")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(d.index, d["largest_pair_gross_share"], label="largest pair")
    axes[1].plot(d.index, d["top5_gross_share"], label="top 5")
    axes[1].plot(d.index, d["gross_hhi"], label="HHI")
    axes[1].set_title("Gross concentration")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    stale_total = d["n_stale_etf"] + d["n_stale_underlying"]
    sc = axes[2].scatter(stale_total, d["daily_net_pnl"].abs(), c=d["gross_leverage"],
                         cmap="viridis", alpha=0.65, s=24)
    axes[2].set_xlabel("Stale leg marks")
    axes[2].set_ylabel("|daily net P&L| ($)")
    axes[2].set_title("P&L events vs data gaps")
    plt.colorbar(sc, ax=axes[2], label="gross leverage")
    axes[2].grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    event_cols = [c for c in [
        "book_equity", "daily_price_pnl", "daily_borrow_cost", "daily_margin_cost",
        "daily_txn_cost", "daily_net_pnl", "gross_leverage", "n_positions",
        "n_stale_etf", "n_stale_underlying", "active_plan_date",
    ] if c in d.columns]
    events = d.loc[d["daily_net_pnl"].abs().nlargest(20).index, event_cols].sort_values("daily_net_pnl")
    display(events)
"""
    ),
    md(
        """## Per-bucket position composition (Diamond Creek style)

For each sleeve, stacked time series of **% of that sleeve's gross** in:
1. each **pair** (ETF/underlying), and
2. each **underlying** (sum of pair gross sharing the same underlying).

Same idea as the Diamond Creek / IBKR `ALL_PAIR_GROSS` stackplot. When a sleeve
has many names, the chart keeps the top 12 by average share and rolls the rest
into `Other` so the evolution stays readable.
"""
    ),
    code(
        """import matplotlib.dates as mdates

COMPOSITION_TOP_N = 12  # top series kept; remainder -> Other


def _pair_leg_gross(df: pd.DataFrame) -> pd.Series:
    if {"etf_usd", "underlying_usd"}.issubset(df.columns):
        return df["etf_usd"].abs() + df["underlying_usd"].abs()
    if {"long_usd", "short_usd"}.issubset(df.columns):
        return df["long_usd"].abs() + df["short_usd"].abs()
    return pd.Series(0.0, index=df.index)


def _gross_share_wide(
    day_df: pd.DataFrame,
    key_col: str,
    *,
    top_n: int = COMPOSITION_TOP_N,
) -> pd.DataFrame:
    # Daily % of sleeve gross by key_col; keep top_n by mean share, else Other.
    if day_df.empty or key_col not in day_df.columns:
        return pd.DataFrame()
    g = (
        day_df.groupby(["date", key_col], as_index=False)["pair_gross"]
        .sum()
        .sort_values(["date", key_col])
    )
    wide = g.pivot(index="date", columns=key_col, values="pair_gross").fillna(0.0)
    wide.index = pd.to_datetime(wide.index)
    total = wide.sum(axis=1).replace(0.0, np.nan)
    pct = wide.div(total, axis=0).fillna(0.0) * 100.0
    if pct.empty:
        return pct
    order = pct.mean(axis=0).sort_values(ascending=False)
    keep = order.index[: max(1, int(top_n))].tolist()
    if len(order) > len(keep):
        out = pct[keep].copy()
        out["Other"] = pct.drop(columns=keep).sum(axis=1)
        return out
    return pct[order.index.tolist()]


def _stack_share(ax, pct: pd.DataFrame, title: str) -> None:
    if pct.empty:
        ax.set_title(title + " (empty)")
        ax.set_ylim(0, 100)
        return
    cols = list(pct.columns)
    # Stable palette; "Other" last and muted.
    cmap = plt.get_cmap("tab20")
    colors = []
    for i, c in enumerate(cols):
        if str(c) == "Other":
            colors.append("#bdbdbd")
        else:
            colors.append(cmap(i % 20))
    ax.stackplot(pct.index, pct.T.values, labels=cols, colors=colors, linewidth=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Gross share (%)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.grid(axis="y", alpha=0.25)
    # Legend only when few series (B2/B4/B5); B1 would be unreadable.
    if len(cols) <= 14:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)


if pair_daily_df is None or pair_daily_df.empty:
    print("no pair_daily_pnl.csv — re-run with FORCE_RERUN=True")
else:
    pdf = pair_daily_df.copy()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf["ETF"] = pdf["ETF"].astype(str).str.upper()
    pdf["Underlying"] = pdf["Underlying"].astype(str).str.upper()
    pdf["pair"] = pdf["ETF"] + "/" + pdf["Underlying"]
    pdf["pair_gross"] = _pair_leg_gross(pdf)

    for sleeve in SLEEVE_ORDER:
        sub = pdf.loc[pdf["sleeve"] == sleeve].copy()
        if sub.empty or float(sub["pair_gross"].sum()) <= 0:
            print(f"{SLEEVE_LABELS.get(sleeve, sleeve)}: no deployed gross")
            continue

        pair_pct = _gross_share_wide(sub, "pair")
        und_pct = _gross_share_wide(sub, "Underlying")
        label = SLEEVE_LABELS.get(sleeve, sleeve)
        n_pairs = int(sub["pair"].nunique())
        n_und = int(sub["Underlying"].nunique())

        fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
        _stack_share(
            axes[0],
            pair_pct,
            f"{label}: each pair as % of sleeve gross  (n={n_pairs})",
        )
        _stack_share(
            axes[1],
            und_pct,
            f"{label}: each underlying as % of sleeve gross  (n={n_und})",
        )
        axes[1].set_xlabel("Date")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        avg_pair = pair_pct.mean().sort_values(ascending=False).head(10).to_frame("avg_gross_share_pct")
        avg_und = und_pct.mean().sort_values(ascending=False).head(10).to_frame("avg_gross_share_pct")
        print(f"{label}: top pairs by avg sleeve-gross share")
        display(avg_pair)
        print(f"{label}: top underlyings by avg sleeve-gross share")
        display(avg_und)
"""
    ),
    md("## Monthly sleeve P&L heatmap"),
    code(
        """if not daily_diag.empty:
    sd = daily_diag.set_index("date")
    sleeve_cols = [s for s in SLEEVE_ORDER if s in sd.columns]
    monthly = sd[sleeve_cols].groupby(sd.index.to_period("M")).sum()
    if not monthly.empty:
        vals = monthly.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(vals))) or 1.0
        fig, ax = plt.subplots(figsize=(10, max(3.2, 0.42 * len(monthly))))
        im = ax.imshow(vals, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_yticks(np.arange(len(monthly)))
        ax.set_yticklabels(monthly.index.astype(str))
        ax.set_xticks(np.arange(len(sleeve_cols)))
        ax.set_xticklabels([SLEEVE_LABELS.get(s, s) for s in sleeve_cols], rotation=20, ha="right")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]/1000:.0f}k", ha="center", va="center", fontsize=7)
        ax.set_title("prod: monthly net P&L by sleeve")
        plt.colorbar(im, ax=ax, label="Monthly P&L ($)")
        plt.tight_layout()
        plt.show()
        display(monthly.round(0))
"""
    ),
    md("## Per-bucket PnL contribution"),
    code(
        """if sleeve_pnl_df.empty and pair_stats_df.empty:
    print("no pair/sleeve PnL artifacts — re-run with FORCE_RERUN=True")
else:
    if not sleeve_pnl_df.empty:
        sp = sleeve_pnl_df.copy()
    else:
        sp = (pair_stats_df.groupby("sleeve", as_index=False)
              .agg(n_pairs=("ETF", "count"), pnl_usd=("pnl_usd", "sum")))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if not sp.empty:
        sp = sp.set_index("sleeve").reindex(SLEEVE_ORDER).dropna(how="all")
        colors = ["#4c72b0" if v >= 0 else "#c44e52" for v in sp["pnl_usd"]]
        labels = [SLEEVE_LABELS.get(i, i) for i in sp.index]
        ax.barh(labels, sp["pnl_usd"], color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("prod: sleeve PnL ($)")
        ax.grid(True, axis="x", alpha=0.3)
        display(sp.assign(label=labels)[["label", "n_pairs", "pnl_usd"]].reset_index(drop=True)
                if "n_pairs" in sp.columns else sp)
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """## Per-pair PnL by bucket (price / borrow / live divs)

For each sleeve: chart first (worst 8 + best 8 by `pnl_usd`), then the detail
table. Bars are **sim `pnl_usd`** (price − borrow + short credit − margin − txn).
Markers: **borrow** (sim fee, plotted as −cost) and **live IBKR div/PIL** from
`data/ledger/dividend_cash_history.csv` (informational — not added into sim PnL;
useful for yieldboost shorts).
"""
    ),
    code(
        """div_path = REPO / "data" / "ledger" / "dividend_cash_history.csv"
div_by_etf = {}
if div_path.exists():
    _div = pd.read_csv(div_path, parse_dates=["date"])
    _div["symbol"] = _div["symbol"].astype(str).str.upper()
    _div["amount_usd"] = pd.to_numeric(_div["amount_usd"], errors="coerce").fillna(0.0)
    # Window to backtest sample
    if not pair_daily_df.empty:
        d0, d1 = pair_daily_df["date"].min(), pair_daily_df["date"].max()
        _div = _div[(_div["date"] >= d0) & (_div["date"] <= d1)]
    div_by_etf = _div.groupby("symbol")["amount_usd"].sum().to_dict()
    print(f"loaded live div/PIL for {len(div_by_etf)} symbols from {div_path.name}")
else:
    print("no dividend_cash_history.csv — div column will be 0")

if pair_stats_df.empty:
    print("no pair_stats — re-run backtest")
else:
    ps = pair_stats_df.copy()
    ps["div_pil_live_usd"] = ps["ETF"].astype(str).str.upper().map(div_by_etf).fillna(0.0)
    for c in ("price_pnl_usd", "borrow_cost_usd", "short_credit_usd", "margin_cost_usd", "txn_cost_usd"):
        if c not in ps.columns:
            ps[c] = 0.0
    # Net after attaching live div/PIL (informational; sim pnl_usd excludes this)
    ps["pnl_plus_live_div"] = ps["pnl_usd"] + ps["div_pil_live_usd"]

    sleeves_present = [s for s in SLEEVE_ORDER if s in set(ps["sleeve"].astype(str))]
    for sleeve in sleeves_present:
        full = ps[ps["sleeve"] == sleeve].sort_values("pnl_usd")
        n_full = len(full)
        sub = full
        if n_full > 16:
            sub = pd.concat([full.head(8), full.tail(8)])
        labels = sub["ETF"].astype(str) + "/" + sub["Underlying"].astype(str)
        y = np.arange(len(sub))
        fig_h = max(3.2, 0.38 * len(sub) + 1.4)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        colors = ["#c44e52" if v < 0 else "#4c72b0" for v in sub["pnl_usd"]]
        ax.barh(y, sub["pnl_usd"], color=colors, alpha=0.85, label="sim pnl")
        # Borrow is a cost: plot as negative so it sits with losses / left of zero.
        ax.scatter(
            -pd.to_numeric(sub["borrow_cost_usd"], errors="coerce").fillna(0.0),
            y, color="#e6a817", s=42, zorder=3, label="−borrow $ (sim)",
        )
        ax.scatter(
            sub["div_pil_live_usd"], y, color="#2ca02c", s=42, zorder=3,
            label="live div/PIL $",
        )
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        title = f"{SLEEVE_LABELS.get(sleeve, sleeve)} — pair PnL + borrow / live div"
        if n_full > len(sub):
            title += f" (worst/best 8 of {n_full})"
        ax.set_title(title)
        ax.set_xlabel("USD")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        _ip_display(fig)
        plt.close(fig)

        print(f"=== {SLEEVE_LABELS.get(sleeve, sleeve)} ({n_full} pairs) ===")
        show = sub[[
            "ETF", "Underlying", "pnl_usd", "price_pnl_usd", "borrow_cost_usd",
            "short_credit_usd", "margin_cost_usd", "txn_cost_usd", "div_pil_live_usd",
            "pnl_plus_live_div",
        ]].sort_values("pnl_usd")
        display(show.round(2))
"""
    ),
    md(
        """## Pair exposure / hedge / rebals by sleeve

B1/B2/B5: worst 5 and best 5 by `pnl_usd`. **B4: every pair**.

Charts: PnL, long/short exposure, hedge ratio, plus rebalance-date table.
(`long_usd` = underlying target, `short_usd` = ETF target;
`hedge_ratio` = `|underlying| / |ETF|`.)
"""
    ),
    code(
        """need = {"long_usd", "short_usd", "hedge_ratio", "rebalance_dates", "pnl_usd"}
if pair_stats_df.empty:
    print("no pair_stats — re-run backtest")
elif not need.issubset(pair_stats_df.columns):
    print(
        "pair_stats missing exposure/rebal columns — re-run with FORCE_RERUN=True "
        f"(have={sorted(pair_stats_df.columns)})"
    )
else:
    ps = pair_stats_df.copy()
    show_cols = [
        c for c in [
            "rank", "ETF", "Underlying", "pnl_usd",
            "long_usd", "short_usd", "hedge_ratio", "Delta",
            "n_rebals", "rebalance_dates", "end_weight",
        ] if c == "rank" or c in ps.columns
    ]
    for sleeve in SLEEVE_ORDER:
        sub = ps[ps["sleeve"].astype(str) == sleeve].sort_values("pnl_usd")
        if sub.empty:
            continue
        is_b4 = sleeve == "inverse_decay_bucket4"
        if is_b4:
            plot = sub.copy()
            plot.insert(0, "rank", [f"p{i}" for i in range(1, len(plot) + 1)])
            out = plot.copy()
            title_sfx = f"all {len(plot)} pairs"
        else:
            bottom = sub.head(5).copy()
            top = sub.tail(5).iloc[::-1].copy()
            bottom.insert(0, "rank", [f"bottom_{i}" for i in range(1, len(bottom) + 1)])
            top.insert(0, "rank", [f"top_{i}" for i in range(1, len(top) + 1)])
            out = pd.concat([bottom, top], ignore_index=True)
            plot = pd.concat([bottom, top.iloc[::-1]], ignore_index=True)
            title_sfx = "top/bottom 5"
        labels = plot["ETF"].astype(str) + "/" + plot["Underlying"].astype(str)
        fig_h = max(4.2, 0.28 * len(plot) + 1.5)
        fig, axes = plt.subplots(1, 3, figsize=(15, fig_h))

        ax = axes[0]
        colors = ["#c44e52" if v < 0 else "#4c72b0" for v in plot["pnl_usd"]]
        ax.barh(labels, plot["pnl_usd"], color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("PnL ($)")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

        ax = axes[1]
        y = np.arange(len(plot))
        h = 0.35
        ax.barh(y - h / 2, plot["long_usd"], height=h, label="long (und)", color="#55a868")
        ax.barh(y + h / 2, plot["short_usd"], height=h, label="short (ETF)", color="#c44e52")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("Last target exposure ($)")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, axis="x", alpha=0.3)

        ax = axes[2]
        ax.barh(labels, plot["hedge_ratio"], color="#8172b2")
        ax.set_title("Hedge ratio |und|/|ETF|")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

        fig.suptitle(f"{SLEEVE_LABELS.get(sleeve, sleeve)} — {title_sfx}", y=1.02, fontsize=12)
        fig.tight_layout()
        _ip_display(fig)
        plt.close(fig)

        print(f"=== {SLEEVE_LABELS.get(sleeve, sleeve)} — {title_sfx} ===")
        disp = out[[c for c in show_cols if c in out.columns]].copy()
        for col in ("pnl_usd", "long_usd", "short_usd", "hedge_ratio", "Delta", "end_weight"):
            if col in disp.columns:
                disp[col] = pd.to_numeric(disp[col], errors="coerce").round(4)
        display(disp)
"""
    ),
    md(
        """## Cumulative pair PnL over time

B1/B2/B5: same top/bottom 5 as above. **B4: every pair.** From
`pair_daily_pnl.csv` (share-hold marks + financing + txn).
"""
    ),
    code(
        """pdaily_path = OUT_BASE / "pair_daily_pnl.csv"
need = {"long_usd", "short_usd", "hedge_ratio", "rebalance_dates", "pnl_usd"}
if pair_stats_df.empty:
    print("no pair_stats — re-run backtest")
elif not pdaily_path.exists():
    print(
        "no pair_daily_pnl.csv — re-run with FORCE_RERUN=True "
        "(simulator now exports pair×day series)"
    )
elif not need.issubset(pair_stats_df.columns):
    print("pair_stats missing columns for top/bottom selection")
else:
    pdaily = pd.read_csv(pdaily_path, parse_dates=["date"])
    ps = pair_stats_df.copy()
    for sleeve in SLEEVE_ORDER:
        sub = ps[ps["sleeve"].astype(str) == sleeve].sort_values("pnl_usd")
        if sub.empty:
            continue
        bottom = sub.head(5)
        top = sub.tail(5)
        focus = sub.copy() if sleeve == "inverse_decay_bucket4" else pd.concat([bottom, top], ignore_index=True)
        etfs = set(focus["ETF"].astype(str))
        day = pdaily[
            (pdaily["sleeve"].astype(str) == sleeve)
            & (pdaily["ETF"].astype(str).isin(etfs))
        ].copy()
        if day.empty:
            print(f"{SLEEVE_LABELS.get(sleeve, sleeve)}: no pair_daily rows")
            continue

        rank_map = {}
        if sleeve == "inverse_decay_bucket4":
            for r in focus.itertuples(index=False):
                rank_map[str(r.ETF)] = f"{r.ETF}/{r.Underlying}"
            title_sfx = f"all {len(focus)} pairs cum PnL"
        else:
            for i, r in enumerate(bottom.itertuples(index=False), 1):
                rank_map[str(r.ETF)] = f"bottom_{i} {r.ETF}/{r.Underlying}"
            for i, r in enumerate(top.iloc[::-1].itertuples(index=False), 1):
                rank_map[str(r.ETF)] = f"top_{i} {r.ETF}/{r.Underlying}"
            title_sfx = "top/bottom 5 cum PnL"

        day["label"] = day["ETF"].astype(str).map(rank_map)
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

        ax = axes[0]
        for lab, g in day.groupby("label", sort=True):
            g = g.sort_values("date")
            ax.plot(g["date"], g["cum_pnl"], label=lab, lw=1.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Cumulative PnL ($)")
        ax.set_title(f"{SLEEVE_LABELS.get(sleeve, sleeve)} — {title_sfx}")
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for lab, g in day.groupby("label", sort=True):
            g = g.sort_values("date")
            ax.plot(g["date"], g["underlying_usd"], label=lab, lw=1.2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Long (underlying) $")
        ax.set_title(f"{SLEEVE_LABELS.get(sleeve, sleeve)} — underlying exposure")
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        _ip_display(fig)
        plt.close(fig)

        end = (
            day.sort_values("date")
            .groupby("label", as_index=False)
            .tail(1)[["label", "date", "cum_pnl", "underlying_usd", "etf_usd", "hedge_ratio"]]
            .sort_values("cum_pnl", ascending=False)
        )
        print(f"=== {SLEEVE_LABELS.get(sleeve, sleeve)} — end-of-sample pair daily ===")
        display(end.round(2))
"""
    ),
    md(
        """## Why trading starts late (pre-April gap)

Prod mode re-sizes each archived `etf_screened_today.csv` with **today's** GTP.
Pre-2026-04-25 screened files are a thin schema (~27 cols vs ~97): they lack
`net_edge_p50_annual` and most edge/opt2 fields. The backtest shims
`net_edge_p50_annual` from `net_decay_annual`, which is enough for B1/B2 to size.

If `prod_sizing_diag.csv` still shows empty plans until late April, the last run
was before that shim (or hit a transient GTP failure). Re-run with
`FORCE_RERUN=True`. Archived `proposed_trades.csv` also exists sparsely from
2026-03-24 (unused by prod mode) and can seed a hybrid timeline.

### Method to estimate plans from 2026-02-27

1. **Re-run prod** with the edge shim (preferred counterfactual under today's GTP).
2. **Hybrid fill (now default):** where prod sizing still fails, insert that date's
   archived `proposed_trades.csv`; also ingest plan-only dates (no screened CSV).
3. **Schema backfill** (highest fidelity): re-screen Feb–Apr dates so archives
   include full edge bootstrap / B4 opt2 columns, then prod-replay again.
4. **Sanity check**: compare sim pair paths to live
   `data/runs/*/accounting/pnl_bucket_1.csv` for the same underlyings.
"""
    ),
    md("## Cumulative sleeve PnL"),
    code(
        """path = OUT_BASE / "sleeve_daily_pnl.csv"
if not path.exists():
    print("no sleeve_daily_pnl.csv")
else:
    sd = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for s in SLEEVE_ORDER:
        col = f"{s}_cum_pnl"
        if col in sd.columns:
            ax.plot(sd.index, sd[col], label=SLEEVE_LABELS[s], lw=1.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("prod: cumulative sleeve PnL ($)")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """## Prod sizing diagnostics

`prod_sizing_diag.csv` — one row per screened date (ok / fail, n pairs, B4 gross).
"""
    ),
    code(
        """diag_path = OUT_BASE / "prod_sizing_diag.csv"
rebalance_path = OUT_BASE / "rebalance_audit.csv"

if diag_path.exists():
    rd = pd.read_csv(diag_path, parse_dates=["date"])
    print("prod sizing sources:")
    display(rd["source"].value_counts())
    display(rd.tail(15))
    if "gross_b4" in rd.columns and "ok" in rd.columns:
        ok = rd[rd["ok"].astype(bool)]
        if len(ok):
            print(
                f"B4 gross median=${ok['gross_b4'].median():,.0f} "
                f"max=${ok['gross_b4'].max():,.0f}"
            )
            fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
            ax = axes[0]
            ax.plot(ok["date"], ok["gross_sum"], label="book plan gross", lw=1.6)
            ax.plot(ok["date"], ok["gross_b4"], label="B4 plan gross", lw=1.4)
            ax.set_ylabel("Plan gross ($)")
            ax.set_title("Prod sizing: plan gross over screened dates")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax = axes[1]
            ax.bar(ok["date"], ok["n_pairs"], width=1.5, color="#4c72b0", alpha=0.75)
            ax.set_ylabel("n pairs in plan")
            ax.set_xlabel("screened date")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()
    if "error" in rd.columns:
        fails = rd[~rd["ok"].astype(bool)] if "ok" in rd.columns else rd[rd["source"].astype(str).str.contains("fail", na=False)]
        if len(fails):
            print("sizing failures (sample):")
            display(fails[["date", "error"]].head(10))
else:
    print("no prod_sizing_diag.csv")

if rebalance_path.exists():
    ra = pd.read_csv(rebalance_path, parse_dates=["date"])
    print("rebalance events:", len(ra))
    display(ra.tail(12))
    if {"deployed_gross_usd", "target_planned_gross_usd"}.issubset(ra.columns):
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.plot(ra["date"], ra["target_planned_gross_usd"], label="target planned", marker="o", lw=1.4)
        ax.plot(ra["date"], ra["deployed_gross_usd"], label="deployed", marker="s", lw=1.4)
        ax.set_title("Rebalance days: planned vs deployed gross")
        ax.set_ylabel("Gross ($)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
"""
    ),
    md(
        """## B4 path diagnostics (sim book) — **every pair**

EOD-style multipanels for **all** B4 pairs from this notebook's artifacts
(`pair_daily_pnl.csv` + price panel + production cadence knobs):

1. Simulated cumulative PnL
2. Gross leg exposure (|ETF| / |underlying|)
3. Hedge h(t): book (share-held) vs model (TR/VCR cadence); gray = no-signal / h_mid
4. Days → model rebalance + cadence signal
5. Underlying price (metrics, Yahoo fill after panel end) + rebal markers

Also writes `b4_pair_paths.pdf` (one page per pair). Cadence uses the full
underlying calendar (proper warmup), then aligns to the book window.
"""
    ),
    code(
        """from scripts.b4_backtest_pair_charts import plot_b4_path_gallery
from scripts.sizing_tilt_cadence_bt import load_price_panel

pair_daily_path = OUT_BASE / "pair_daily_pnl.csv"
pair_stats_path = OUT_BASE / "pair_stats.csv"
b4_pdf_path = OUT_BASE / "b4_pair_paths.pdf"

if not pair_daily_path.exists() or not pair_stats_path.exists():
    print("skip B4 path diagnostics — need pair_daily_pnl.csv + pair_stats.csv")
else:
    _ps = pd.read_csv(pair_stats_path)
    _pd = pd.read_csv(pair_daily_path, parse_dates=["date"])
    _b4 = _ps[_ps["sleeve"].astype(str).eq("inverse_decay_bucket4")] if "sleeve" in _ps.columns else _ps.iloc[0:0]
    if _b4.empty:
        print("no inverse_decay_bucket4 rows in pair_stats — nothing to plot")
    else:
        print(f"B4 pairs in sim: {len(_b4)} — plotting ALL pairs (inline + PDF)")
        display(
            _b4.sort_values("pnl_usd", ascending=False)[
                [c for c in ("ETF", "Underlying", "pnl_usd", "hedge_ratio", "n_rebals") if c in _b4.columns]
            ].round(2)
        )
        _panel = load_price_panel(RUN_DATE, min_days=PRICE_PANEL_MIN_DAYS)
        _meta = plot_b4_path_gallery(
            pair_stats=_ps,
            pair_daily=_pd,
            panel=_panel,
            start=START,
            mode="all",
            sleeves=("inverse_decay_bucket4",),
            fill_yahoo=True,
            pdf_path=b4_pdf_path,
            show=True,
        )
        print("wrote", b4_pdf_path)
        if _meta:
            display(pd.DataFrame(_meta))
        else:
            print("gallery empty")
"""
    ),
    md(
        """## B4 per-pair trade ledger (sim vs live model)

Exact fills from `pair_stats.rebalance_dates` + legs/txn in `pair_daily_pnl`
(do **not** use `is_rebalance` — that flag is book-level).

Each trade is labeled `enter_operator` / `exit_operator` / `cadence_resize` /
`off_cadence_resize` (or `resize` if cadence overlay skipped). Writes:

- `b4_pair_trade_ledger.csv`
- `b4_pair_trade_summary.csv`
- `b4_sim_vs_live_checklist.csv`

Sim counters from `report.json`: `n_b4_membership_deferred`, `n_b4_ratchet_pins`,
`n_b4_cadence_rebals` (band skips are silent — no fill).
"""
    ),
    code(
        """from scripts.b4_pair_trade_audit import run_audit, realism_checklist

# Panel overlay is slow (full price load). Membership/enter-exit audit is the
# main realism check; set WITH_B4_CADENCE_OVERLAY=True to tag cadence hits.
WITH_B4_CADENCE_OVERLAY = False
_paths = run_audit(OUT_BASE, with_panel=WITH_B4_CADENCE_OVERLAY, fill_yahoo=False)
_sum = pd.read_csv(_paths["summary"])
_led = pd.read_csv(_paths["ledger"], parse_dates=["date"])
_rj = (report.get("book") if isinstance(report, dict) else {}) or {}
print(
    "sim B4 counters:",
    {k: _rj.get(k) for k in (
        "n_b4_membership_deferred",
        "n_b4_ratchet_pins",
        "n_b4_cadence_rebals",
        "b4_membership_clock",
        "b4_apply_resize_bands",
        "b4_ratchet_execution_guard",
    )},
)
display(realism_checklist())
display(
    _sum[
        [
            c
            for c in (
                "ETF",
                "n_trades",
                "n_enter",
                "n_exit",
                "n_cadence_resize",
                "n_cadence_rebal",
                "n_off_cadence_resize",
                "n_resize_unknown_cadence",
                "membership_trade_share",
                "median_days_between_trades",
                "rapid_churn_events",
                "pnl_usd",
                "realism_flags",
            )
            if c in _sum.columns
        ]
    ].round(2)
)
print("reason mix:")
display(_led["reason"].value_counts().to_frame("n"))
# Example: worst PnL pair path
if not _sum.empty:
    _worst = str(_sum.sort_values("pnl_usd").iloc[0]["ETF"])
    print(f"worst pair ledger: {_worst}")
    display(
        _led[_led["ETF"] == _worst][
            [c for c in ("date", "reason", "prev_gross", "gross", "turnover_usd", "txn_cost", "book_h") if c in _led.columns]
        ].round(1)
    )
"""
    ),
    md(
        """## B4 historical plan / crash / waterfall audit

Rebuilt from **this notebook's** `plans/*.csv` (gross, n pairs, median h over the
sim window) plus any archived `b4_crash_budget.csv` / `b4_sizing_waterfall.csv`
under `data/runs/` (often only the latest GTP day).
"""
    ),
    code(
        """from scripts.b4_historical_audit import (
    load_archived_crash_waterfall,
    load_b4_plan_history,
    plot_b4_plan_history,
    plot_crash_waterfall_snapshots,
    summarize_b4_plan_history,
)

_plans = OUT_BASE / "plans"
_hist = load_b4_plan_history(_plans)
_sum = summarize_b4_plan_history(_hist)
if _sum.empty:
    print("no B4 rows in cached plans — skip historical audit")
else:
    display(_sum.tail(12))
    _fig = plot_b4_plan_history(_sum)
    if _fig is not None:
        plt.show()
        plt.close(_fig)
    _hist.to_csv(OUT_BASE / "b4_plan_history.csv", index=False)
    _sum.to_csv(OUT_BASE / "b4_plan_history_daily.csv", index=False)
    print("wrote", OUT_BASE / "b4_plan_history_daily.csv")

_crash, _wf = load_archived_crash_waterfall()
print(f"archived crash rows={len(_crash)} waterfall rows={len(_wf)}")
for _f in plot_crash_waterfall_snapshots(_crash, _wf):
    plt.show()
    plt.close(_f)
"""
    ),
    md(
        """## B4 Monte Carlo tail / equity fan (dashboard-style)

Uses the same production cadence portfolio construction as
`scripts/build_bucket4_risk_sim.py` (proposed book on `RUN_DATE`), then plots
1y max-drawdown histograms and a block-bootstrap equity fan — analogous to the
risk-dashboard B4 simulator.
"""
    ),
    code(
        """from scripts.b4_notebook_risk_sim import plot_b4_risk_sim, run_b4_notebook_risk_sim

_b4risk = run_b4_notebook_risk_sim(run_date=RUN_DATE, start="2025-01-01", n_mc=2000)
if not _b4risk:
    print("B4 risk sim unavailable for", RUN_DATE, "(need proposed book + panel)")
else:
    print(
        f"pairs={len(_b4risk['pairs'])} book_gross=${_b4risk['book_gross']:,.0f} "
        f"realized CAGR={_b4risk['perf'].get('cagr')} maxDD={_b4risk['perf'].get('maxdd')}"
    )
    _f = plot_b4_risk_sim(_b4risk)
    if _f is not None:
        plt.show()
        plt.close(_f)
"""
    ),
    md(
        """### Live GTP cadence snapshots (optional)

If today's run folder has GTP cadence PNGs, show them as a **live as-of
`RUN_DATE`** reference. Separate from the historical sim paths above.
"""
    ),
    code(
        """from IPython.display import Image, display as _disp

_cad_dir = REPO / "data" / "runs" / RUN_DATE / "b4_hedge_cadence"
_pngs = [
    _cad_dir / "b4_hedge_ratio_over_time.png",
    _cad_dir / "b4_days_to_rebalance.png",
]
_shown = False
for _p in _pngs:
    if _p.is_file():
        print(_p.relative_to(REPO))
        _disp(Image(filename=str(_p)))
        _shown = True
if not _shown:
    print(f"no GTP cadence PNGs under {_cad_dir} — skip (optional)")
"""
    ),
    md(
        """## Single-pair path drilldown

Set `PAIR_ETF` (and optional `PAIR_UNDERLYING`) to plot one pair’s exposure,
hedge ratio, delta residual, and cum PnL over time. Black dots mark days with
material notional change or txn cost (sim `is_rebalance` is often always-on).
"""
    ),
    code(
        """PAIR_ETF = "SNDU"
PAIR_UNDERLYING = None  # e.g. "SNDK", or None to take the first match

pdaily_path = OUT_BASE / "pair_daily_pnl.csv"
if not pdaily_path.exists():
    print("need pair_daily_pnl.csv — re-run prod cell")
else:
    pdaily = pd.read_csv(pdaily_path, parse_dates=["date"])
    m = pdaily["ETF"].astype(str).str.upper() == str(PAIR_ETF).upper()
    if PAIR_UNDERLYING:
        m &= pdaily["Underlying"].astype(str).str.upper() == str(PAIR_UNDERLYING).upper()
    s = pdaily.loc[m].sort_values("date").copy()
    if s.empty:
        print(f"no rows for {PAIR_ETF}/{PAIR_UNDERLYING or '*'}")
    else:
        und = str(s["Underlying"].iloc[0])
        s["gross"] = s["etf_usd"].abs() + s["underlying_usd"].abs()
        s["net_dollar"] = s["underlying_usd"] + s["etf_usd"]
        s["delta_net"] = s["etf_usd"] * s["Delta"] + s["underlying_usd"]
        s["d_gross"] = s["gross"].diff().abs().fillna(s["gross"].abs())
        trade = (s["d_gross"] > 250) | (s["txn_cost"].abs() > 0.5)
        fri = s["date"].dt.weekday == 4

        fig, axes = plt.subplots(
            4, 1, figsize=(12, 11), sharex=True,
            gridspec_kw={"height_ratios": [1.4, 1.0, 0.9, 1.1]},
        )
        ax = axes[0]
        ax.fill_between(s["date"], 0, s["underlying_usd"], color="#55a868", alpha=0.35, label=f"long und ({und})")
        ax.plot(s["date"], s["underlying_usd"], color="#2b6e3f", lw=1.4)
        ax.fill_between(s["date"], 0, s["etf_usd"], color="#c44e52", alpha=0.35, label=f"short ETF ({PAIR_ETF})")
        ax.plot(s["date"], s["etf_usd"], color="#8b2e32", lw=1.4)
        ax.plot(s["date"], s["net_dollar"], color="k", lw=1.2, ls="--", label="net $")
        ax.scatter(s.loc[trade, "date"], s.loc[trade, "underlying_usd"], s=28, c="k", zorder=5, label="trade day")
        for d in s.loc[fri, "date"]:
            ax.axvline(d, color="steelblue", alpha=0.15, lw=1)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("Notional ($)")
        ax.set_title(f"{PAIR_ETF}/{und} exposure (blue bands = Fridays)")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(s["date"], s["hedge_ratio"], color="#8172b2", lw=1.6, label="|und|/|ETF|")
        ax.plot(s["date"], s["Delta"].abs(), color="gray", lw=1.2, ls="--", label="|Delta|")
        ax.scatter(s.loc[trade, "date"], s.loc[trade, "hedge_ratio"], s=22, c="k", zorder=5)
        ax.set_ylabel("Hedge ratio")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(s["date"], s["delta_net"], color="#dd8452", lw=1.5, label="delta-adj net")
        ax.axhline(0, color="k", lw=0.6)
        ax.scatter(s.loc[trade, "date"], s.loc[trade, "delta_net"], s=22, c="k", zorder=5)
        ax.set_ylabel("$ delta")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        ax.plot(s["date"], s["cum_pnl"], color="#4c72b0", lw=2.0, label="cum PnL")
        ax.bar(s["date"], s["daily_pnl"], color="#4c72b0", alpha=0.25, width=1.0, label="daily PnL")
        ax.scatter(s.loc[trade, "date"], s.loc[trade, "cum_pnl"], s=28, c="k", zorder=5, label="trade day")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("PnL ($)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"{PAIR_ETF}/{und} — production actual pair path", fontsize=13, y=0.995)
        fig.tight_layout()
        png = OUT_BASE / f"{str(PAIR_ETF).lower()}_{und.lower()}_pair_path.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.show()
        print(f"days={len(s)} trade_days={int(trade.sum())} fridays={int(fri.sum())}")
        print(
            f"end und=${float(s.underlying_usd.iloc[-1]):,.0f}  "
            f"etf=${float(s.etf_usd.iloc[-1]):,.0f}  "
            f"cum_pnl=${float(s.cum_pnl.iloc[-1]):,.0f}"
        )
        print("saved", png)
"""
    ),
    md(
        """## B1 deep-dive vs Buckets1-4Backtest

Why production-actual B1 (`core_leveraged`) may look weak vs the research notebook
`notebooks/Buckets1-4Backtest.ipynb`:

| Dimension | Prod actual BT | Buckets1-4Backtest |
|-----------|----------------|--------------------|
| Universe / sizing | Live GTP + YAML budgets + Phase-2b | Research screener + custom weight power / QCQP |
| Hedge | Plan legs; operator_5d retarget | Often dynamic / static-h research path |
| Costs | Borrow + margin + 20bp slip + commission | Often lighter / different borrow |
| Capital metric | Use **ROC / ROG** above | Often path return on sleeve notional |
| Overlap with B4 | Financing now nets shared underlyings | Usually sleeve-isolated |

Use the cell below to rank B1 pairs and compare price vs financing drag.
"""
    ),
    code(
        """b1 = "core_leveraged"
ret_m = (
    pd.read_csv(OUT_BASE / "sleeve_return_metrics.csv")
    if (OUT_BASE / "sleeve_return_metrics.csv").exists()
    else pd.DataFrame()
)
if not ret_m.empty:
    print("=== sleeve ROC / ROG ===")
    display(ret_m)

if pair_stats_df.empty:
    print("no pair_stats — re-run prod cell")
else:
    ps_b1 = pair_stats_df.loc[pair_stats_df["sleeve"].astype(str) == b1].copy()
    if ps_b1.empty:
        print("no B1 pairs in pair_stats")
    else:
        money_cols = [
            c for c in [
                "pnl_usd", "price_pnl_usd", "borrow_cost_usd", "short_credit_usd",
                "margin_cost_usd", "txn_cost_usd", "long_usd", "short_usd",
            ]
            if c in ps_b1.columns
        ]
        print(f"B1 pairs={len(ps_b1)}  total pnl=${ps_b1['pnl_usd'].sum():,.0f}")
        if money_cols:
            agg = ps_b1[money_cols].sum(numeric_only=True)
            print("B1 component totals:")
            display(agg.to_frame("usd").T)
        show_cols = [c for c in ["ETF", "Underlying", *money_cols, "n_rebal"] if c in ps_b1.columns]
        ranked = ps_b1.sort_values("pnl_usd")
        print("=== worst 10 B1 pairs ===")
        display(ranked.head(10)[show_cols])
        print("=== best 10 B1 pairs ===")
        display(ranked.tail(10)[show_cols].iloc[::-1])

# Financing internalization: how much und notional was netted away?
if (OUT_BASE / "daily_diagnostics.csv").exists():
    dd = pd.read_csv(OUT_BASE / "daily_diagnostics.csv", parse_dates=["date"])
    if "underlying_internalized_usd" in dd.columns:
        print(
            f"avg internalized und ${dd['underlying_internalized_usd'].mean():,.0f} | "
            f"avg book gross ${dd['gross_notional'].mean():,.0f} | "
            f"avg book net ${dd['net_notional'].mean():,.0f}"
        )
    if f"{b1}__gross_cap" in dd.columns:
        print(
            f"B1 avg gross ${dd[f'{b1}__gross_cap'].mean():,.0f} | "
            f"B1 avg net ${dd[f'{b1}__net_cap'].mean():,.0f} | "
            f"B1 cum pnl ${dd[b1].sum():,.0f}"
        )
"""
    ),
    md("## Limitations"),
    code(
        """print("### prod")
for line in report.get("limitations", []):
    print("-", line)
if report.get("prod_stats"):
    print("prod_stats:", report["prod_stats"])
print("\\nReport:", OUT_BASE / "REPORT.md")
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT)
