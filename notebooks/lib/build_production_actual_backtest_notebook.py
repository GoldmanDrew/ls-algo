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
        """# Production Actual Backtest (May 2025 → now)

**Single method:** each archived `etf_screened_today.csv` is sized with a
generate-trade-plan approximation — the day's listed universe, average borrow
(`borrow_avg_annual`, else spot), and current net-edge score
(`net_edge_p50_annual`) via `mirror_generate_trade_plan_sizing`. Archived
`proposed_trades.csv` is **not** used.

```text
screened(as-of D)
  → borrow := borrow_avg_annual (fallback borrow_current)
  → score  := net_edge − borrow_aversion × borrow   (per YAML)
  → plan[D] from GTP mirror (B1/B2/B4 gates + sleeve budgets)
  → simulate_book_from_plan_timeline (next-close, Phase-2b, costs)
```

This is the decay-score GTP path (not live B4 opt2 / crash / smooth / ratchet).
Capital / sleeve budgets come from live `strategy_config.yml`. B3 flow is excluded.

**Archive gap:** screened archives start ~2025-12-28. Pre-archive the book holds
cash (`PRE_ARCHIVE_POLICY="cash"`) unless you backfill screened history.
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

RUN_DATE = "2026-07-10"          # price panel date
START = "2025-05-01"
PRE_ARCHIVE_POLICY = "cash"      # "cash" | "skip"
FORCE_RERUN = True
OUT_BASE = REPO / "notebooks" / "output" / "production_actual_bt"

# Single GTP-approx path only
RUN_MODES = ["gtp"]
DIAG_MODE = "gtp"

SLEEVE_LABELS = {
    "core_leveraged": "B1 core",
    "yieldboost": "B2 yieldboost",
    "inverse_decay_bucket4": "B4 inverse",
    "volatility_etp_bucket5": "B5 vol ETP",
}
SLEEVE_ORDER = list(SLEEVE_LABELS)

print("REPO", REPO)
print("RUN_DATE", RUN_DATE, "START", START, "modes", RUN_MODES)
"""
    ),
    md("## Archive coverage"),
    code(
        """from scripts.production_actual_backtest import archive_coverage_summary, list_archived_plan_dates, list_archived_screened_dates

cov = archive_coverage_summary(START)
display(cov)
print("n proposed_trades dates:", len(list_archived_plan_dates()), "(unused by gtp mode)")
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
    ["Gross sizing", "Dynamic NAV × target gross", "GTP-approx plan gross/equity × current NAV", "Implemented"],
    ["Plan source", "Live generate_trade_plan", "Daily mirror on screened: avg borrow + net edge", "Proxy"],
    ["Plan timing", "Signal known after close", "T plan executes next available close; P&L starts next session", "Implemented"],
    ["Between rebals", "Hold shares; legs drift", "Signed ETF/underlying notionals drift with each leg", "Implemented"],
    ["Leg schema", "Underlying target + ETF target", "long_usd=underlying; short_usd=ETF; explicit columns preferred", "Implemented"],
    ["Rebalance", "Phase-2b hysteresis + plan changes", "12% enter / 4% exit bands + $250 floor; establish entries / close exits", "Implemented"],
    ["Slippage", "Broker/fill dependent", "20 bp on every traded dollar, including opening trades", "Proxy"],
    ["Commission", "Clear Street low-touch", "$0.0035/share by leg", "Implemented"],
    ["Borrow", "Point-in-time by short symbol", "Plan borrow_avg (sizing) carried until the next plan", "Proxy"],
    ["Margin debit", "OBFR + 45 bp, Actual/360", "4.00% benchmark fallback + 45 bp, Actual/360", "Proxy"],
    ["Short credit", "Disabled in Diamond Creek v15", "Disabled", "Implemented"],
    ["Prices", "Total-return / split-safe marks", "Adjusted-close panel + Flex/override/heuristic split repair", "Implemented"],
    ["Missing bars", "Carry last mark; cannot trade", "Zero-return stale mark; blocked close/entry is audited", "Implemented"],
    ["Share rounding", "Whole shares / broker lots", "Dollar-notional targets", "Gap"],
    ["Locates", "Can reject or resize shorts", "Screened universe at sizing time; no execution reject", "Gap"],
    ["B4 stack", "opt2 → crash → smooth → ratchet", "Decay-score mirror only (no opt2/crash/smooth/ratchet)", "Proxy"],
], columns=["Item", "Production", "This backtest", "Status"])
display(assumption_audit)
"""
    ),
    md(
        """## Run GTP-approx backtest

Outputs land under `notebooks/output/production_actual_bt/` (`gtp_sizing_diag.csv`,
`pair_stats.csv`, `daily_diagnostics.csv`, …).
"""
    ),
    code(
        """from scripts.production_actual_backtest import run_production_actual_backtest
import json

reports = {}
navs = {}
summaries = {}
pair_stats = {}
sleeve_pnls = {}

for mode in RUN_MODES:
    outdir = OUT_BASE  # gtp writes at the root
    summary_path = outdir / "sleeve_summary.csv"
    report_path = outdir / "report.json"
    if FORCE_RERUN or not summary_path.exists():
        print(f"\\n===== running mode={mode} =====")
        reports[mode] = run_production_actual_backtest(
            run_date=RUN_DATE,
            start=START,
            outdir=outdir,
            mode=mode,
            pre_archive_policy=PRE_ARCHIVE_POLICY,
        )
    else:
        reports[mode] = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"loaded cached {mode} from {outdir}")
    summaries[mode] = pd.read_csv(summary_path)
    navs[mode] = pd.read_csv(outdir / "daily_nav.csv", index_col=0, parse_dates=True)
    ps = outdir / "pair_stats.csv"
    sp = outdir / "sleeve_pnl.csv"
    if ps.exists():
        pair_stats[mode] = pd.read_csv(ps)
    if sp.exists():
        sleeve_pnls[mode] = pd.read_csv(sp)
    display(summaries[mode])
    print("Book:", reports[mode].get("book"))
    print("gtp_stats:", reports[mode].get("gtp_stats"))
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
    print(f"mode={DIAG_MODE} days={len(daily_diag):,}  max |P&L residual|=${max_resid:,.8f}")
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
    ax.set_title(f"{DIAG_MODE}: NAV and drawdown")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(d.index, d["gross_leverage"], color="#4c72b0", label="gross / NAV")
    cfg_lev = float(reports[DIAG_MODE].get("gross_leverage") or np.nan)
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
    margin_cum = d["daily_margin_cost"].cumsum()
    txn_cum = d["daily_txn_cost"].cumsum()
    net_cum = d["daily_net_pnl"].cumsum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(d.index, gross_cum, label="gross price P&L", lw=1.8)
    ax.plot(d.index, net_cum, label="net P&L", lw=2.0, color="black")
    ax.plot(d.index, -borrow_cum, label="− borrow", ls="--")
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
    rep = reports[DIAG_MODE]
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
          f"commission=${commission:,.0f}; margin=${current_margin:,.0f}; borrow=${current_borrow:,.0f}")
"""
    ),
    md(
        """## Plan deployment, churn, and blocked execution

Target-vs-deployed gross reveals missing panels and blocked close marks.
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
    axes[1].plot(r.index, r.get("n_resized", 0), label="resizes", color="#4c72b0")
    axes[1].set_ylabel("Pair count")
    axes[1].legend(loc="best", ncol=3)
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
    md("## Book NAV"),
    code(
        """fig, ax = plt.subplots(figsize=(11, 4.5))
for mode, nav in navs.items():
    col = "BOOK_NAV" if "BOOK_NAV" in nav.columns else nav.columns[0]
    s = nav[col].dropna()
    if len(s) == 0:
        continue
    ax.plot(s.index, s.values, label=f"{mode} (end ${s.iloc[-1]:,.0f})", lw=2)
ax.set_title(f"GTP-approx production backtest  {START} → now")
ax.set_ylabel("Book NAV ($)")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

cmp_rows = []
for mode, rep in reports.items():
    b = rep.get("book") or {}
    cmp_rows.append({
        "mode": mode,
        "cagr": b.get("cagr"),
        "vol": b.get("vol"),
        "sharpe": b.get("sharpe"),
        "maxdd": b.get("maxdd"),
        "end_usd": b.get("end_usd"),
        "n_plans": b.get("n_plans_used"),
        "cash_days": b.get("cash_days"),
        "first_plan": b.get("first_plan"),
    })
display(pd.DataFrame(cmp_rows))
"""
    ),
    md("## Sleeve NAV paths"),
    code(
        """mode = DIAG_MODE if DIAG_MODE in navs else next(iter(navs), None)
if mode is None:
    print("no modes loaded")
else:
    nav = navs[mode]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for s in SLEEVE_ORDER:
        if s in nav.columns:
            series = nav[s].dropna()
            if len(series):
                ax.plot(series.index, series.values, label=SLEEVE_LABELS[s], lw=1.6)
    if "BOOK_NAV" in nav.columns:
        ax.plot(nav.index, nav["BOOK_NAV"], label="BOOK", color="k", lw=2.2)
    ax.set_title(f"Sleeve NAVs — {mode}")
    ax.set_ylabel("NAV ($)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## Per-bucket PnL contribution"),
    code(
        """def _sleeve_pnl_table(mode: str) -> pd.DataFrame:
    if mode in sleeve_pnls and not sleeve_pnls[mode].empty:
        return sleeve_pnls[mode].copy()
    if mode in pair_stats and not pair_stats[mode].empty:
        return (pair_stats[mode].groupby("sleeve", as_index=False)
                .agg(n_pairs=("ETF", "count"), pnl_usd=("pnl_usd", "sum")))
    return pd.DataFrame()

if DIAG_MODE not in pair_stats and DIAG_MODE not in sleeve_pnls:
    print("no pair/sleeve PnL artifacts — re-run with FORCE_RERUN=True")
else:
    sp = _sleeve_pnl_table(DIAG_MODE)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if not sp.empty:
        sp = sp.set_index("sleeve").reindex(SLEEVE_ORDER).dropna(how="all")
        colors = ["#4c72b0" if v >= 0 else "#c44e52" for v in sp["pnl_usd"]]
        labels = [SLEEVE_LABELS.get(i, i) for i in sp.index]
        ax.barh(labels, sp["pnl_usd"], color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{DIAG_MODE}: sleeve PnL ($)")
        ax.grid(True, axis="x", alpha=0.3)
        display(sp.assign(label=labels)[["label", "n_pairs", "pnl_usd"]].reset_index(drop=True)
                if "n_pairs" in sp.columns else sp)
    plt.tight_layout()
    plt.show()
"""
    ),
    md("## Per-pair PnL by bucket"),
    code(
        """FOCUS_MODE = DIAG_MODE if DIAG_MODE in pair_stats else next(iter(pair_stats), None)
if FOCUS_MODE is None:
    print("no pair_stats — re-run backtest")
else:
    ps = pair_stats[FOCUS_MODE].copy()
    if "stats_corrupt" in ps.columns and ps["stats_corrupt"].any():
        print("WARNING: stats_corrupt rows:")
        display(ps[ps["stats_corrupt"]])
    print(f"pair stats mode={FOCUS_MODE}  n={len(ps)}")
    sleeves_present = [s for s in SLEEVE_ORDER if s in set(ps["sleeve"].astype(str))]
    n = max(1, len(sleeves_present))
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    for ax, sleeve in zip(axes[:, 0], sleeves_present):
        sub = ps[ps["sleeve"] == sleeve].sort_values("pnl_usd")
        if len(sub) > 16:
            sub = pd.concat([sub.head(8), sub.tail(8)])
        colors = ["#c44e52" if v < 0 else "#4c72b0" for v in sub["pnl_usd"]]
        labels = sub["ETF"].astype(str) + "/" + sub["Underlying"].astype(str)
        ax.barh(labels, sub["pnl_usd"], color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{SLEEVE_LABELS.get(sleeve, sleeve)} — pair PnL ($) [{FOCUS_MODE}]")
        ax.grid(True, axis="x", alpha=0.3)
        print(f"\\n=== {SLEEVE_LABELS.get(sleeve, sleeve)} ===")
        show_cols = [c for c in ["ETF", "Underlying", "start_usd", "end_usd", "pnl_usd", "ret", "cagr", "maxdd", "first_trade_date", "stats_corrupt"]
                     if c in sub.columns]
        display(sub.sort_values("pnl_usd")[show_cols].round(4))
    plt.tight_layout()
    plt.show()
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
    ax.set_title("gtp: cumulative sleeve PnL ($)")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """## GTP sizing diagnostics

`gtp_sizing_diag.csv` — one row per screened date (ok / fail, n pairs, borrow-avg hits).
"""
    ),
    code(
        """diag_path = OUT_BASE / "gtp_sizing_diag.csv"
rebalance_path = OUT_BASE / "rebalance_audit.csv"

if diag_path.exists():
    rd = pd.read_csv(diag_path, parse_dates=["date"])
    print("gtp sizing sources:")
    display(rd["source"].value_counts())
    display(rd.tail(15))
    if "error" in rd.columns:
        fails = rd[~rd["ok"].astype(bool)] if "ok" in rd.columns else rd[rd["source"].astype(str).str.contains("fail", na=False)]
        if len(fails):
            print("sizing failures (sample):")
            display(fails[["date", "error"]].head(10))
else:
    print("no gtp_sizing_diag.csv")

if rebalance_path.exists():
    ra = pd.read_csv(rebalance_path, parse_dates=["date"])
    print("rebalance events:", len(ra))
    display(ra.tail(12))
"""
    ),
    md("## Limitations"),
    code(
        """for mode, rep in reports.items():
    print(f"\\n### {mode}")
    for line in rep.get("limitations", []):
        print("-", line)
    if rep.get("gtp_stats"):
        print("gtp_stats:", rep["gtp_stats"])
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
