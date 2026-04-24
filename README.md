# ls-algo — Leveraged ETF decay & pair strategy

Python toolkit for a **systematic long/short** book around leveraged and inverse ETFs: daily **screening** and decay analytics, **YAML-driven** portfolio sleeves, **IBKR** execution helpers, **Flex-based** accounting, and **research notebooks** (including Bucket‑4 / v6 hedge experiments). This repository is an **engineering workspace**; it is not investment advice.

---

## Table of contents

1. [What this repo does](#what-this-repo-does)
2. [Repository layout](#repository-layout)
3. [End-to-end pipeline](#end-to-end-pipeline)
4. [Configuration (`config/strategy_config.yml`)](#configuration-configstrategy_configyml)
5. [Sleeves vs accounting “buckets”](#sleeves-vs-accounting-buckets)
6. [Research & backtests](#research--backtests)
7. [Automation (GitHub Actions)](#automation-github-actions)
8. [Data layout](#data-layout)
9. [Requirements & IBKR](#requirements--ibkr)
10. [Disclaimer](#disclaimer)

---

## What this repo does

- **Decay capture (conceptual):** leveraged and inverse ETFs exhibit path-dependent **volatility drag** relative to a buy-and-hold of the underlying. The screener estimates **realised** and **expected** gross decay, blends them, and subtracts **borrow** to produce **`net_decay_annual`** (and related columns) for ranking and sizing.
- **Portfolio construction:** `generate_trade_plan.py` reads the screened universe and `strategy_config.yml`, assigns each non-purgatory row to **core**, **whitelist**, and/or **inverse decay Bucket‑4** sleeves (subject to rules), sizes **gross USD** targets, and writes **`data/proposed_trades.csv`** plus a dated copy under **`data/runs/<YYYY-MM-DD>/`**.
- **Execution:** `execute_trade_plan.py` and `rebalance_strategy.py` implement cleanup, establishment, hedging, short-availability gates, and adaptive orders against **TWS / IB Gateway** via **`ib_insync`**.
- **Flow sleeve:** `execute_flow_program.py` deploys the configured **inverse ETF short basket** on a calendar (e.g. weekly `goodAfterTime` orders), separate from the stock-sleeve plan, with cumulative tracking in **`flow_ledger_csv`**.
- **Accounting:** `ibkr_flex.py` pulls Flex XML; `ibkr_accounting.py` builds PnL and beta-normalised exposure reports under **`data/runs/<date>/accounting/`**; `run_eod_pnl_email.py` composes history, plots, and optional email.

---

## Repository layout

| Path | Role |
|------|------|
| **`config/strategy_config.yml`** | Single source of truth for strategy tag, capital, leverage, IBKR host, screener borrow bands, **portfolio sleeves**, rebalance thresholds, paths. |
| **`config/strategy_blacklist.txt`** | Optional extra symbol blacklist (referenced when present). |
| **`config/etf_expense_ratios.yml`** | Expense-ratio side data (used by enrichment paths where applicable). |
| **`daily_screener.py`** | Universe → prices → OLS betas → FTP borrow → decay / vol enrichment → **`paths.screened_csv`**. |
| **`etf_analytics.py`**, **`etf_screener.py`**, **`expense_ratios.py`** | Supporting analytics / scraping / expense helpers used by the screener and notebooks. |
| **`generate_trade_plan.py`** | Screened CSV → sleeve membership + decay-aware weights → proposed trades + flow ledger append. |
| **`execute_trade_plan.py`** | Proposed notionals → IBKR orders (parallel legs, purgatory rules, short availability). |
| **`rebalance_strategy.py`** | Three-phase hybrid rebalancer (cleanup / establish / hedge); reuses execution helpers. |
| **`execute_flow_program.py`** | Scheduled flow short deployment. |
| **`harvest_underexposed_shorts.py`** | Maintenance helper around short hedges (see script header). |
| **`plot_proposed_trades.py`** | Visualization helper for plan outputs. |
| **`baseline_snapshot.py`** | Baseline position snapshot utilities for execution deltas. |
| **`ibkr_flex.py`** | Flex Web Service pull → `data/runs/<date>/ibkr_flex/*.xml`. |
| **`ibkr_accounting.py`** | Flex + screened universe → accounting CSVs + `totals.json`. |
| **`run_eod_pnl_email.py`** | EOD orchestration: flex + accounting + ledger history + optional SMTP. |
| **`strategy_config.py`** | Shared YAML loader with **path resolution** relative to repo root (used by email script and others). |
| **`notebooks/`** | Research: **`Simple_Pair_Backtest.ipynb`** (Bucket‑4, v6 Option‑2 hedge research, regime overlays, grids), Diamond Creek / IBKR large-AUM studies, etc. |
| **`data/`** | Working CSVs, caches, **`ledger/`**, **`runs/<date>/`** run artifacts. |
| **`.github/workflows/eod_pnl_email.yml`** | Scheduled weekday **screener + trade plan** commit and **EOD PnL** pipeline. |

There is **no** in-repo `core/` Python package required for the main pipeline; `daily_screener.py` may optionally import helpers from a `core` package if you install one alongside this repo.

---

## End-to-end pipeline

High-level flow:

```text
daily_screener.py
  → etf_screened_today.csv (columns: ETF, Underlying, Beta, borrow, decay, purgatory, …)

generate_trade_plan.py
  → proposed_trades.csv (+ dated run copy)
  → flow_ledger.csv (append, when flow sleeve is used)

execute_trade_plan.py / rebalance_strategy.py
  → IBKR orders vs proposed_trades + baseline

execute_flow_program.py
  → IBKR flow sleeve shorts (schedule from YAML)

ibkr_flex.py → ibkr_accounting.py → run_eod_pnl_email.py
  → data/runs/<date>/accounting/* + optional email
```

### 1) Screen — `daily_screener.py`

Builds the tradable universe (hardcoded lists + optional YieldMax / Roundhill scraping), fetches prices, estimates **beta** (OLS on total-return aligned series), pulls **IBKR indicative borrow** via FTP (with cache / retry), computes **gross / blended / net decay**, underlying vol, Bucket labels used in downstream analytics, and writes the screened CSV.

Common flags (see module docstring for the full set):

```bash
python daily_screener.py
python daily_screener.py --skip-scrape          # use cached issuer pages
python daily_screener.py --skip-inverse        # skip inverse universe bucket
python daily_screener.py --lookback 1y
python daily_screener.py --output data/my_screen.csv
```

The GitHub Action uses **`--skip-ibkr-check`** for unattended runs (see workflow file).

### 2) Plan — `generate_trade_plan.py`

- Reads **`config/strategy_config.yml`** (fixed path `CONFIG_PATH` in script).
- Applies **global strategy blacklist** and sleeve rules.
- **Stock sleeves:** `core_leveraged`, `whitelist_stock`, optional **`inverse_decay_bucket4`** (toggle with **`enabled: false`** to remove all B4 targets; remaining gross goes to core/whitelist by their `target_weight` ratio).
- **Core net-decay selectivity (optional):** `min_net_decay_annual` and/or **`net_decay_hysteresis`** with sticky state file **`paths.core_leveraged_decay_state_json`** (reduces core names flickering in/out when decay hovers near a threshold). If **`min_net_decay_annual` > 0**, it is a **hard floor** applied after hysteresis (no NaN bypass, no “sticky” admission below the minimum). Whitelist is **not** gated by this hysteresis.
- **Flow program** weights are **fixed** in YAML (`weighting.method: fixed`, `normalize: true`); the script re-normalises if needed.
- Writes **`data/proposed_trades.csv`**, **`data/runs/<run-date>/proposed_trades.csv`**, and may append **`flow_ledger_csv`**.

```bash
python generate_trade_plan.py
python generate_trade_plan.py --run-date 2026-04-22
```

### 3) Execute — `execute_trade_plan.py`

Parallel IBKR workflow: cleanup of stale ETF legs (respecting purgatory), bucket execution (short ETF first where configured), post-pass hedging, FTP short-availability gate, adaptive / limit escalation, coordinator cancels, `DRY_RUN` env support.

```bash
python execute_trade_plan.py
DRY_RUN=1 python execute_trade_plan.py
```

`execution.dry_run` in YAML is also consulted when `DRY_RUN` is unset.

### 4) Rebalance — `rebalance_strategy.py`

Hybrid **Phase 1–3** rebalancer driven by **`portfolio.rebalance`** thresholds in YAML (drift, min trade, net long/short bands, etc.). Imports shared primitives from `execute_trade_plan.py`.

```bash
python rebalance_strategy.py --dry-run
python rebalance_strategy.py --run-date 2026-04-22 --skip-phase-1
```

### 5) Flow sleeve — `execute_flow_program.py`

Deploys the inverse short basket on **`frequency`** **`D`** or **`W`**, using **`schedule_days`** / **`schedule_time_et`** in Eastern time. Maintains cumulative notionals in the ledger path from config.

```bash
python execute_flow_program.py
```

### 6) Flex & accounting — `ibkr_flex.py`, `ibkr_accounting.py`, `run_eod_pnl_email.py`

```bash
python ibkr_flex.py --run-date 2026-04-22
python ibkr_accounting.py 2026-04-22
python run_eod_pnl_email.py
```

Secrets for Flex queries and SMTP are expected in the environment for CI (see workflow).

---

## Configuration (`config/strategy_config.yml`)

Everything operational reads from this file. **Do not treat the table below as authoritative numbers** — copy the structure and edit values in YAML.

| Section | Purpose |
|---------|---------|
| **`ibkr.*`** | TWS / Gateway host, port, client id, delayed data flags. |
| **`accounting.*`** | Mark-price behaviour, bucket split method for attribution. |
| **`strategy.*`** | Tag, capital, gross leverage, equity source (`ibkr` / `static` / `file`), global blacklist. |
| **`paths.*`** | All major CSV paths, run directories, optional **`core_leveraged_decay_state_json`**. |
| **`execution.*`** | Order style, timeouts, `dry_run`, parallelism, short-first behaviour. |
| **`screener.*`** | Borrow “soft” cap, purgatory margin, whitelist / flow hard caps, staleness rules. |
| **`portfolio.rebalance.*`** | Drift and net-exposure triggers used by `rebalance_strategy` / execution tooling. |
| **`portfolio.sleeves.core_leveraged`** | Core (levered long ETFs): `target_weight`, `min_beta_used`, optional **`min_net_decay_annual`**, **`net_decay_hysteresis`**, `weighting` (decay_score vs equal, caps, blend). |
| **`portfolio.sleeves.whitelist_stock`** | Explicit ETF list + weighting. |
| **`portfolio.sleeves.inverse_decay_bucket4`** | Inverse decay pairs: **`enabled`** master switch, borrow / vol / edge rules, **`partial_hedge_ratio`**, shares-outstanding cap, weighting. |
| **`portfolio.sleeves.flow_program`** | Flow shorts universe, schedule, **`fixed_usd_per_week`** (or deployment base from YAML), **fixed weights** summing to **1.0**. |

`generate_trade_plan.py` documents sleeve behaviour in its module docstring; keep YAML and that docstring aligned when you change rules.

---

## Sleeves vs accounting “buckets”

- **Sleeves** (`core_leveraged`, `whitelist_stock`, `inverse_decay_bucket4`, `flow_program`) are **portfolio construction** labels written into **`proposed_trades.csv`** and used by execution / rebalancing.
- **Accounting buckets** (`bucket_1` … `bucket_4`) in **`ibkr_accounting.py`** are **attribution / reporting** groupings (e.g. high-beta levered vs inverse decay). They are related concepts but **not identical** to YAML sleeve names. When interpreting `pnl_bucket_*.csv`, read the accounting script headers.

---

## Research & backtests

- **`notebooks/Simple_Pair_Backtest.ipynb`** — end-to-end **pair backtests** (Bucket‑4 hedge tests, **v6 Option‑2** dynamic hedge vs static **h=0.5**, optional **regime overlays** and **grid search** cells, equity / **h** plots). This is **research code**; parameters in the notebook are **not** automatically wired into `generate_trade_plan.py` unless you port them.
- **`notebooks/Diamond_Creek_*.ipynb`**, **`IBKR_Backtest_Large_AUM.ipynb`** — fund-style / attribution experiments.

---

## Automation (GitHub Actions)

**`.github/workflows/eod_pnl_email.yml`**

1. **Weekday schedule (`cron`)** — America/New_York calendar date for `RUN_DATE`.
2. **Job `screener`:** `pip install -r requirements.txt` → `daily_screener.py --run-date "$RUN_DATE" --skip-ibkr-check` → `generate_trade_plan.py --run-date "$RUN_DATE"` → commits `data/etf_screened_today.csv`, `data/proposed_trades.csv`, and dated run outputs.
3. **Job `eod`:** depends on `screener`; runs `ibkr_flex.py`, `ibkr_accounting.py`, `run_eod_pnl_email.py` with **secrets** for Flex queries and SMTP; may commit under `data/`.

Use **workflow_dispatch** to run screener-only, EOD-only, or both.

---

## Data layout

| Location | Contents |
|----------|-----------|
| **`data/etf_screened_today.csv`** | Latest screener output (canonical input to the trade planner). |
| **`data/proposed_trades.csv`** | Latest targets for execution. |
| **`data/baseline_snapshot.csv`** | Pre-strategy positions for delta math. |
| **`data/flow_ledger.csv`** | Cumulative flow deployments. |
| **`data/borrow_cache.csv`**, **`data/scrape_cache_*.csv`** | Caches to speed reruns. |
| **`data/runs/<YYYY-MM-DD>/`** | Per-day snapshots: `proposed_trades.csv`, `etf_screened_today.csv`, `ibkr_flex/`, `accounting/`, `rebalance/`, etc. |
| **`data/ledger/`** | Longer-horizon PnL history and plots used by the email script. |
| **`data/core_leveraged_decay_state.json`** | Sticky core net-decay state (created/updated when hysteresis is enabled). |

**`data/borrow_history.json` (optional):** If present before `daily_screener.py` runs, it is **auto-loaded** (no CLI flag required) from, in order: `BORROW_HISTORY_PATH` / `--borrow-history-path`, `ETF_DASHBOARD_ROOT/data/borrow_history.json`, sibling `../etf-dashboard/data/borrow_history.json`, or this repo’s `data/borrow_history.json`. The scheduled **GitHub Action** downloads it from `GoldmanDrew/etf-dashboard` raw when the curl succeeds. That enables **weighted borrow resampling** for `net_edge_*` plus `net_edge_hist_json` / p25 / p75 on the CSV.

---

## Requirements & IBKR

**`requirements.txt`** (install with `pip install -r requirements.txt`):

- `pandas`, `numpy`, `matplotlib`, `requests`, `beautifulsoup4`, `lxml`, `yfinance`, `pytz`, `PyYAML`, `pyarrow`

**Live IBKR trading scripts** (`execute_trade_plan.py`, `rebalance_strategy.py`, `execute_flow_program.py`) additionally require:

```bash
pip install ib_insync
```

Use **Python 3.10+** (CI uses 3.11). IBKR **TWS or Gateway** must accept API connections on the host/port in `strategy_config.yml`.

---

## Disclaimer

This repository is for **research and automation** of a specific trading workflow. Past backtests and decay formulas **do not** guarantee future results. Borrow, short availability, corporate actions, issuer-specific path dependency, and regulatory constraints can dominate small theoretical edges. You are responsible for compliance, risk limits, and testing in paper accounts before live trading.
