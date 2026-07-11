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

Three modes (see `scripts/production_actual_backtest.py`):

| Mode | What it does | Window fidelity |
|------|----------------|-----------------|
| **`frozen`** | Single anchor `proposed_trades` (today's book, **full B4 stack**) | Counterfactual on today's sized book |
| **`replay`** (Phase A) | Day-by-day weights from archived `data/runs/*/proposed_trades.csv` | Exact GTP output when archived (~Dec 2025+) — **prefer for PIT pair attribution** |
| **`recompute`** (Phase B) | `mirror_generate_trade_plan_sizing` on archived screened; else archived plan | Decay-score GTP (no live B4 opt2 / crash / smooth / ratchet) |

**Frozen caveats (post 2026-07-11 audit):** pair stats compound from each name's
first live date (no NaN-Friday wipeouts); prices are split-adjusted; B5 uses
`bucket5_carry_bt` (not B4 dynamic-h); B4 skips are logged in `pair_skip_audit.csv`.
Frozen still applies *today's* weights from `--start` — use **replay** for
true point-in-time history.

**Replay accounting (post 2026-07-11 parity audit):** archived gross is no
longer normalized down to 1x; plan legs and borrow stay point-in-time; positions
are marked as held shares between weekly rebalances; a T plan executes at the
next available close and starts earning P&L on the following session. Opening
and resize trades pay 20 bp slippage plus $0.0035/share. Borrow and margin debit
are reported separately and daily P&L reconciles to NAV.

**B4 sizing (production, 2026-07-11+)** — four-step stack (see README § Bucket 4 sizing):

```text
1. Score     opt2 decay/borrow (+ cov tilt, continuous borrow ramp) → relative weights
2. Crash-cap trim to ρ·budget/L (L uses asymmetric EMA), then scale_to_budget refill
3. Smooth    post-cap trim-only EMA: entry ramp, dilution fade, soft exits, hard cuts
4. Leg split inv / und via h·β (+ grow-only inverse ratchet)
```

Knobs live under `portfolio.sleeves.inverse_decay_bucket4.rules` in
`config/strategy_config.yml` (`crash_budget.*`, `weight_smoothing.*`).
Telemetry: `b4_crash_budget.csv`, `b4_sizing_waterfall.csv`.

Capital / sleeve budgets come from live `strategy_config.yml`. B3 flow is excluded from NAV.

**Archive gap:** live proposed/screened archives start ~2025-12-28. For recompute
before that, run `python -m scripts.backfill_screened_history --start 2025-05-01 --end 2025-12-26`
to write weekly PIT `etf_screened_today.csv` (prices truncated; borrow
carry-first-known/default; shares stubbed). Replay still holds cash pre-archive
unless `PRE_ARCHIVE_POLICY="skip"`.
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

RUN_DATE = "2026-07-10"          # price panel + frozen anchor (has waterfall + scale_to_budget)
START = "2025-05-01"
PRE_ARCHIVE_POLICY = "cash"      # "cash" | "skip"
FORCE_RERUN = True
OUT_BASE = REPO / "notebooks" / "output" / "production_actual_bt"

# Which modes to run in this notebook session
RUN_MODES = ["frozen", "replay", "recompute"]
DIAG_MODE = "replay"             # production-debug focus

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
print("n proposed_trades dates:", len(list_archived_plan_dates()))
print("n screened dates:", len(list_archived_screened_dates()))
print("first plan:", list_archived_plan_dates()[0].date() if list_archived_plan_dates() else None)
"""
    ),
    md(
        """## Production-accounting parity checklist

This is the explicit contract for the production-debug run. `Implemented`
means the replay ledger models the behavior directly. `Proxy` is intentionally
visible in the cost and sensitivity charts. `Gap` should not be interpreted as
production-accurate until the missing artifact is archived.
"""
    ),
    code(
        """assumption_audit = pd.DataFrame([
    ["Gross sizing", "Dynamic NAV × target gross", "Archived plan gross/equity × current NAV", "Implemented"],
    ["Plan timing", "Signal known after close", "T plan executes next available close; P&L starts next session", "Implemented"],
    ["Between rebals", "Hold shares; legs drift", "Signed ETF/underlying notionals drift with each leg", "Implemented"],
    ["Leg schema", "Underlying target + ETF target", "long_usd=underlying; short_usd=ETF; explicit columns preferred", "Implemented"],
    ["Rebalance", "Phase-2b hysteresis + plan changes", "12% enter / 4% exit bands + $250 floor; establish entries / close exits", "Implemented"],
    ["Slippage", "Broker/fill dependent", "20 bp on every traded dollar, including opening trades", "Proxy"],
    ["Commission", "Clear Street low-touch", "$0.0035/share by leg", "Implemented"],
    ["Borrow", "Point-in-time by short symbol", "ETF + underlying archived rates carried until the next plan", "Proxy"],
    ["Margin debit", "OBFR + 45 bp, Actual/360", "4.00% benchmark fallback + 45 bp, Actual/360", "Proxy"],
    ["Short credit", "Disabled in Diamond Creek v15", "Disabled", "Implemented"],
    ["Prices", "Total-return / split-safe marks", "Adjusted-close panel + Flex/override/heuristic split repair", "Implemented"],
    ["Missing bars", "Carry last mark; cannot trade", "Zero-return stale mark; blocked close/entry is audited", "Implemented"],
    ["Share rounding", "Whole shares / broker lots", "Dollar-notional targets", "Gap"],
    ["Locates", "Can reject or resize shorts", "Archived plans reflect sizing-time availability; no execution reject", "Gap"],
    ["Intraday fills", "Actual execution prices", "Close-to-close marks plus slippage proxy", "Gap"],
    ["B3 flow", "$1,300/week parallel sleeve", "Excluded", "Gap"],
], columns=["topic", "Diamond Creek / production intent", "production backtest", "status"])
display(assumption_audit)
print("Highest-priority remaining fidelity work: archive actual fills/locates, daily OBFR, and whole-share target snapshots.")
"""
    ),
    md(
        """## B4 crash-budget + scale-to-budget snapshot (live GTP)

`crash_budget_mult < 1` means the pair was trimmed by `ρ·budget/L` before the
sleeve refill. With `scale_to_budget: true`, those trims are then scaled
pro-rata so sleeve gross ≈ YAML `target_weight` — freed dollars do **not** sit
idle. `scale_mult` and `rho_effective ≈ ρ × scale_mult` show the refill
amplification. `weight_final` is post-scale (pre-smoothing on older rows;
post-smooth when waterfall is present).
"""
    ),
    code(
        """cb_path = REPO / "data" / "runs" / RUN_DATE / "b4_crash_budget.csv"
if cb_path.is_file():
    cb = pd.read_csv(cb_path)
    cols = [c for c in [
        "ETF", "Underlying", "weight_solved", "weight_capped", "weight_final",
        "gross_solved_usd", "gross_capped_usd", "gross_final_usd", "cap_usd",
        "crash_budget_mult", "scale_to_budget", "scale_mult",
        "L", "C", "runup", "tail", "hedge_ratio", "crash_l_source",
    ] if c in cb.columns]
    display(cb[cols].sort_values("crash_budget_mult").round(4))

    solved = float(cb["gross_solved_usd"].sum())
    capped = float(cb["gross_capped_usd"].sum())
    final = float(cb["gross_final_usd"].sum()) if "gross_final_usd" in cb.columns else capped
    scale_mult = float(cb["scale_mult"].iloc[0]) if "scale_mult" in cb.columns and len(cb) else np.nan
    scale_on = bool(cb["scale_to_budget"].iloc[0]) if "scale_to_budget" in cb.columns and len(cb) else False
    print(
        f"solved ${solved:,.0f} → post-cap ${capped:,.0f} → final ${final:,.0f} "
        f"(scale_to_budget={scale_on}, scale_mult={scale_mult:.3f}); "
        f"n_capped={(cb['crash_budget_mult'] < 0.999).sum()}/{len(cb)}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    top = cb.sort_values("crash_budget_mult").head(12)
    ax.barh(top["ETF"] + "/" + top["Underlying"], top["crash_budget_mult"], color="#c44e52")
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("crash_budget_mult (1 = uncapped before scale)")
    ax.set_title("B4 crash-budget multipliers")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
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
`weight_solved` → `weight_capped` (post scale-to-budget) → `weight_smoothed`
(post dilution-aware EMA). Gross columns mirror the same stages; final legs
are `inverse_short_final_usd` / `underlying_short_final_usd`.
"""
    ),
    code(
        """wf_path = REPO / "data" / "runs" / RUN_DATE / "b4_sizing_waterfall.csv"
if wf_path.is_file():
    wf = pd.read_csv(wf_path)
    show = [c for c in [
        "ETF", "Underlying", "weight_solved", "weight_capped", "weight_smoothed",
        "gross_decay_score_usd", "gross_after_caps_usd", "gross_after_smooth_usd",
        "gross_final_usd", "inverse_short_final_usd", "underlying_short_final_usd",
        "L", "C", "rho_effective", "crash_l_source",
    ] if c in wf.columns]
    display(wf[show].sort_values("weight_smoothed", ascending=False).round(4))

    # Stage totals
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

    # Entry/dilution signal: solved vs smoothed divergence
    if {"weight_solved", "weight_smoothed"}.issubset(wf.columns):
        delta = (wf["weight_smoothed"] - wf["weight_solved"]).abs()
        print(f"mean |smoothed − solved| = {delta.mean():.4f}  "
              f"(large when entry ramp / dilution fade is active)")
else:
    print(f"no waterfall at {wf_path} — re-run generate_trade_plan.py for {RUN_DATE}")
"""
    ),
    md(
        """## Run modes

Outputs land under:
- `notebooks/output/production_actual_bt/` (frozen) — also writes `pair_stats.csv`, `sleeve_pnl.csv`
- `.../production_actual_bt/replay/`
- `.../production_actual_bt/recompute/`

Frozen/replay use **plan dollars** (already through the full live B4 stack when
the plan was generated). Under-deployment vs YAML budget is mostly entry-ramp
cash while names fade in — not permanent crash-cap idle cash.
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
    outdir = OUT_BASE if mode == "frozen" else OUT_BASE / mode
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
"""
    ),
    md(
        """## Production-debug ledger checks

These checks fail loudly on accounting drift. The daily ledger must reconcile
price P&L less borrow, margin, and transaction costs exactly to the NAV change.
`DIAG_MODE="replay"` is the primary production-debug view.
"""
    ),
    code(
        """diag_dir = OUT_BASE if DIAG_MODE == "frozen" else OUT_BASE / DIAG_MODE
diag_path = diag_dir / "daily_diagnostics.csv"
rebalance_path = diag_dir / "rebalance_audit.csv"

daily_diag = pd.read_csv(diag_path, parse_dates=["date"]) if diag_path.exists() else pd.DataFrame()
rebalance_diag = pd.read_csv(rebalance_path, parse_dates=["date"]) if rebalance_path.exists() else pd.DataFrame()

if daily_diag.empty:
    print(f"No production-debug ledger at {diag_path}; run replay or recompute.")
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

This is the fastest graph for finding an unintended leverage reset, directional
drift, a stuck position count, or a drawdown that coincides with a plan switch.
The dashed line is configured gross leverage, not a promise that every sleeve
is fully deployed.
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
costs. A widening unexplained gap is an accounting bug; a widening explained
cost line is an economic assumption to revisit.
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

This keeps the executed holdings path fixed and restates only cost dollars. It
is not a full behavioral rerun, but it immediately shows whether the conclusion
is dominated by the 20 bp fill proxy, the 4.45% financing fallback, or borrow.
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
    axes[2].set_xlabel("Archived borrow × multiplier")
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

Target-vs-deployed gross reveals missing panels and blocked close marks. Adds,
exits, and resizes identify plan dates whose turnover deserves a pair-level
inspection.
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
    md(
        """## Data quality, concentration, and event-day triage

Stale marks should cluster around genuine listing/calendar gaps, not unexplained
P&L events. Concentration should be reviewed alongside large daily moves because
a clean accounting bridge can still reflect an economically unrealistic book.
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

    event_cols = ["book_equity", "daily_price_pnl", "daily_borrow_cost", "daily_margin_cost",
                  "daily_txn_cost", "daily_net_pnl", "gross_leverage", "n_positions",
                  "n_stale_etf", "n_stale_underlying", "active_plan_date"]
    events = d.loc[d["daily_net_pnl"].abs().nlargest(20).index, event_cols].sort_values("daily_net_pnl")
    display(events)
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
        ax.set_title(f"{DIAG_MODE}: monthly net P&L by sleeve")
        plt.colorbar(im, ax=ax, label="Monthly P&L ($)")
        plt.tight_layout()
        plt.show()
        display(monthly.round(0))
"""
    ),
    md("## Compare book NAVs"),
    code(
        """fig, ax = plt.subplots(figsize=(11, 4.5))
for mode, nav in navs.items():
    col = "BOOK_NAV" if "BOOK_NAV" in nav.columns else nav.columns[0]
    s = nav[col].dropna()
    if len(s) == 0:
        continue
    ax.plot(s.index, s.values, label=f"{mode} (end ${s.iloc[-1]:,.0f})", lw=2 if mode == "replay" else 1.5)
ax.set_title(f"Production actual backtest  {START} → now")
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
cmp = pd.DataFrame(cmp_rows)
display(cmp)
"""
    ),
    md(
        """## Sleeve NAV paths (frozen)

Each sleeve simulated independently at its YAML budget, then combined into book NAV.
B4 uses plan-deployed gross (post scale + smooth), so entry-ramp cash is not
force-filled up to the YAML ceiling.
"""
    ),
    code(
        """mode = "frozen" if "frozen" in navs else next(iter(navs), None)
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
    md(
        """## Per-bucket PnL contribution

Dollar PnL by sleeve (from `sleeve_pnl.csv` / `pair_stats.csv`). Frozen mode uses
start→end pair equity; replay/recompute use daily `w × r × equity` attribution.
"""
    ),
    code(
        """def _sleeve_pnl_table(mode: str) -> pd.DataFrame:
    if mode in sleeve_pnls and not sleeve_pnls[mode].empty:
        return sleeve_pnls[mode].copy()
    if mode in pair_stats and not pair_stats[mode].empty:
        return (pair_stats[mode].groupby("sleeve", as_index=False)
                .agg(n_pairs=("ETF", "count"), pnl_usd=("pnl_usd", "sum")))
    return pd.DataFrame()

n_modes = len(pair_stats) or len(sleeve_pnls)
if n_modes == 0:
    print("no pair/sleeve PnL artifacts — re-run with FORCE_RERUN=True")
else:
    modes_with = [m for m in RUN_MODES if m in pair_stats or m in sleeve_pnls]
    fig, axes = plt.subplots(1, len(modes_with), figsize=(5.2 * len(modes_with), 4.2), squeeze=False)
    for ax, mode in zip(axes[0], modes_with):
        sp = _sleeve_pnl_table(mode)
        if sp.empty:
            ax.set_title(f"{mode}: empty")
            continue
        sp = sp.set_index("sleeve").reindex(SLEEVE_ORDER).dropna(how="all")
        colors = ["#4c72b0" if v >= 0 else "#c44e52" for v in sp["pnl_usd"]]
        labels = [SLEEVE_LABELS.get(i, i) for i in sp.index]
        ax.barh(labels, sp["pnl_usd"], color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{mode}: sleeve PnL ($)")
        ax.grid(True, axis="x", alpha=0.3)
        display(sp.assign(label=labels)[["label", "n_pairs", "pnl_usd"]].reset_index(drop=True)
                if "n_pairs" in sp.columns else sp)
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """## Per-pair PnL by bucket

Top contributors / detractors within each sleeve.

**Prefer `FOCUS_MODE = \"replay\"` for historical attribution.** Frozen mode is
a counterfactual (today's book from May 2025). Rows with `stats_corrupt=True`
are flagged automatically. See `pair_skip_audit.csv` for B4 panel / short-hist drops.
"""
    ),
    code(
        """FOCUS_MODE = "replay" if "replay" in pair_stats else ("frozen" if "frozen" in pair_stats else next(iter(pair_stats), None))
if FOCUS_MODE is None:
    print("no pair_stats — re-run backtest")
else:
    ps = pair_stats[FOCUS_MODE].copy()
    if "stats_corrupt" in ps.columns and ps["stats_corrupt"].any():
        print("WARNING: stats_corrupt rows:")
        display(ps[ps["stats_corrupt"]])
    skip_path = (OUT_BASE / "pair_skip_audit.csv") if FOCUS_MODE == "frozen" else (OUT_BASE / FOCUS_MODE / "pair_skip_audit.csv")
    if skip_path.exists():
        print("skip audit:")
        display(pd.read_csv(skip_path))
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
    md(
        """## Cumulative sleeve PnL (replay / recompute)

Daily dollar attribution stacked over time — shows *when* each bucket contributed.
"""
    ),
    code(
        """for mode in ("replay", "recompute"):
    path = (OUT_BASE / mode / "sleeve_daily_pnl.csv")
    if not path.exists():
        print(f"{mode}: no sleeve_daily_pnl.csv")
        continue
    sd = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    cum_cols = [f"{s}_cum_pnl" for s in SLEEVE_ORDER if f"{s}_cum_pnl" in sd.columns]
    if not cum_cols:
        print(f"{mode}: no cum pnl columns")
        continue
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for s in SLEEVE_ORDER:
        col = f"{s}_cum_pnl"
        if col in sd.columns:
            ax.plot(sd.index, sd[col], label=SLEEVE_LABELS[s], lw=1.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{mode}: cumulative sleeve PnL ($)")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""
    ),
    md(
        """## Replay / recompute diagnostics

- **replay:** `rebalance_audit.csv` — each plan switch / weekly retarget
- **recompute:** `recompute_diag.csv` — mirror vs archived fallback per date

Note: recompute's mirror path still skips live B4 opt2 / crash / smooth / ratchet;
use **replay** (or frozen) to evaluate the production four-step stack as archived.
"""
    ),
    code(
        """replay_audit = OUT_BASE / "replay" / "rebalance_audit.csv"
recompute_diag = OUT_BASE / "recompute" / "recompute_diag.csv"

if replay_audit.exists():
    ra = pd.read_csv(replay_audit, parse_dates=["date"])
    print("replay rebalance events:", len(ra))
    display(ra.tail(12))
else:
    print("no replay audit (mode not run)")

if recompute_diag.exists():
    rd = pd.read_csv(recompute_diag, parse_dates=["date"])
    print("recompute sources:")
    display(rd["source"].value_counts())
    display(rd.tail(15))
    if "error" in rd.columns:
        fails = rd[rd["source"].astype(str).str.contains("fail", na=False)]
        if len(fails):
            print("mirror failures (sample):")
            display(fails[["date", "error"]].head(10))
else:
    print("no recompute diag (mode not run)")
"""
    ),
    md("## Limitations"),
    code(
        """for mode, rep in reports.items():
    print(f"\\n### {mode}")
    for line in rep.get("limitations", []):
        print("-", line)
    if rep.get("recompute_stats"):
        print("recompute_stats:", rep["recompute_stats"])
print("\\nReports:")
for mode in reports:
    p = OUT_BASE / ("REPORT.md" if mode == "frozen" else f"{mode}/REPORT.md")
    print(" ", p)
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
