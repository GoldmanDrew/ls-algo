# ls-algo — Leveraged ETF decay & pair strategy

Python toolkit for a **systematic long/short** book around leveraged and inverse ETFs: daily **screening** and decay analytics, **YAML-driven** portfolio sleeves, **IBKR** execution helpers, **Flex-based** accounting, and **research notebooks** (including Bucket‑4 / v6 hedge experiments). This repository is an **engineering workspace**; it is not investment advice.

---

## Table of contents

1. [What this repo does](#what-this-repo-does)
2. [Repository layout](#repository-layout)
3. [End-to-end pipeline](#end-to-end-pipeline)
4. [Configuration (`config/strategy_config.yml`)](#configuration-configstrategy_configyml)
5. [Corporate-action splits — see [`SPLITS.md`](SPLITS.md)](#corporate-action-splits)
6. [Sleeves vs accounting “buckets”](#sleeves-vs-accounting-buckets)
7. [Research & backtests](#research--backtests)
8. [Automation (GitHub Actions)](#automation-github-actions)
9. [Data layout](#data-layout)
10. [Phase 2b setup — tax-aware resize & loss-harvest substitution](#phase-2b-setup--tax-aware-resize--loss-harvest-substitution)
11. [Requirements & IBKR](#requirements--ibkr)
12. [Disclaimer](#disclaimer)

---

## What this repo does

- **Decay capture (conceptual):** leveraged and inverse ETFs exhibit path-dependent **volatility drag** relative to a buy-and-hold of the underlying. The screener estimates **realised** and **expected** gross decay, blends them, and subtracts **borrow** to produce **`net_decay_annual`** (and related columns) for ranking and sizing.
- **Portfolio construction:** `generate_trade_plan.py` reads the screened universe and `strategy_config.yml`, assigns each non-purgatory row to **`core_leveraged`**, **`yieldboost`** (YieldBoost names only — `is_yieldboost` from the screener), and/or **inverse decay Bucket‑4** (subject to rules), sizes **gross USD** targets, and writes **`data/proposed_trades.csv`** plus a dated copy under **`data/runs/<YYYY-MM-DD>/`**.
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
| **`daily_screener.py`** | Production screener: universe → prices → OLS betas → FTP borrow → decay / vol enrichment → **`paths.screened_csv`**. Includes `--audit-splits` CLI for split-event review. |
| **`etf_analytics.py`**, **`expense_ratios.py`** | Supporting analytics / expense helpers used by the screener and notebooks. The old standalone `etf_screener.py` path has been retired. |
| **`splits.py`** | Multi-source corporate-action split detection / repair (Yahoo `events.splits`, IBKR Flex `<CorporateAction>`, ops CSV, legacy overrides, heuristic). See [`SPLITS.md`](SPLITS.md). |
| **`generate_trade_plan.py`** | Screened CSV → sleeve membership + decay-aware weights → proposed trades. |
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
| **`.github/workflows/eod_pnl_email.yml`** | Scheduled weekday **screener + trade plan** commit and **EOD PnL** pipeline; screener also runs on every push to `main`. |

There is **no** in-repo `core/` Python package required for the main pipeline; `daily_screener.py` may optionally import helpers from a `core` package if you install one alongside this repo.

---

## End-to-end pipeline

High-level flow:

```text
daily_screener.py
  → etf_screened_today.csv (columns: ETF, Underlying, Delta, borrow, decay, purgatory, …)

generate_trade_plan.py
  → proposed_trades.csv (+ dated run copy)

execute_trade_plan.py / rebalance_strategy.py
  → IBKR orders vs proposed_trades + baseline

harvest_underexposed_shorts.py
  → targeted short catch-up vs plan/position discrepancies

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

- Reads **`config/strategy_config.yml`** through the shared `strategy_config.load_config()` loader.
- Applies **global strategy blacklist** and sleeve rules.
- **Stock sleeves:** `core_leveraged`, **`yieldboost`** (YieldBoost-only bucket‑2 candidates), optional **`inverse_decay_bucket4`** (toggle with **`enabled: false`** to remove all B4 targets; remaining post‑B4 gross is split core vs yieldboost via their **`target_weight` ratio).
- **Core net-decay selectivity (optional):** `min_net_decay_annual` and/or **`net_decay_hysteresis`** with sticky state file **`paths.core_leveraged_decay_state_json`** (reduces core names flickering in/out when decay hovers near a threshold). If **`min_net_decay_annual` > 0**, it is a **hard floor** applied after hysteresis (no NaN bypass, no “sticky” admission below the minimum). The **`yieldboost`** sleeve is **not** gated by this hysteresis.
- **Flow program** weights are **fixed** in YAML (`weighting.method: fixed`, `normalize: true`) and executed separately by `execute_flow_program.py`.
- Writes **`data/proposed_trades.csv`** and **`data/runs/<run-date>/proposed_trades.csv`**. It no longer mutates the flow ledger.

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

Hybrid **Phase 1–3** rebalancer driven by **`portfolio.rebalance`** thresholds in YAML (minimum trade size, net long/short bands, establishment threshold, etc.). Imports shared primitives from `execute_trade_plan.py`.

```bash
python rebalance_strategy.py --dry-run
python rebalance_strategy.py --run-date 2026-04-22 --skip-phase-1
```

### 5) Harvest under-exposed shorts — `harvest_underexposed_shorts.py`

Targeted maintenance runner for ETF shorts that are under target versus the current plan. By default it builds discrepancies from live IBKR positions; it can fall back to accounting discrepancy CSVs.

```bash
python harvest_underexposed_shorts.py --dry-run
python harvest_underexposed_shorts.py --run-date 2026-04-22 --top-n 20
```

### 6) Flow sleeve — `execute_flow_program.py`

Deploys the inverse short basket on **`frequency`** **`D`** or **`W`**, using **`schedule_days`** / **`schedule_time_et`** in Eastern time. Maintains cumulative notionals in the ledger path from config.

```bash
python execute_flow_program.py
```

### 7) Flex & accounting — `ibkr_flex.py`, `ibkr_accounting.py`, `run_eod_pnl_email.py`

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
| **`strategy.*`** | Tag, capital, gross leverage, global blacklist, gross sizing caps, covariance balance. |
| **`paths.*`** | All major CSV paths, run directories, optional **`core_leveraged_decay_state_json`**. |
| **`execution.*`** | Order style, timeouts, `dry_run`, parallelism, short-first behaviour. |
| **`screener.*`** | Borrow “soft” cap, purgatory margin, whitelist hard cap, staleness rules. Flow borrow caps live under `portfolio.sleeves.flow_program.rules`. |
| **`portfolio.rebalance.*`** | Minimum trade and net-exposure triggers used by `rebalance_strategy` and `harvest_underexposed_shorts`. |
| **`portfolio.sleeves.core_leveraged`** | Core (levered long ETFs): `target_weight`, `min_delta_used`, optional **`min_net_decay_annual`**, **`net_decay_hysteresis`**, `weighting` (decay_score vs equal, caps, blend). |
| **`portfolio.sleeves.yieldboost`** | YieldBoost / bucket‑2 sleeve: **`is_yieldboost`** names from the screener only, `rules.min_net_edge_annual`, `weighting`; budget vs core from **`target_weight`**. |
| **`portfolio.sleeves.inverse_decay_bucket4`** | Inverse decay pairs: **`enabled`** master switch, borrow / vol / edge rules, **`partial_hedge_ratio`**, shares-outstanding cap, weighting. |
| **`portfolio.sleeves.flow_program`** | Flow shorts universe, schedule, **`fixed_usd_per_week`** (or deployment base from YAML), **fixed weights** summing to **1.0**. |

`generate_trade_plan.py` documents sleeve behaviour in its module docstring; keep YAML and that docstring aligned when you change rules.

---

## Corporate-action splits

Same-day reverse splits (BAIG, BMNG, FIGG, DUOG, CRWG, CRCG on 2026-05-05 Nasdaq ECA2026-298, etc.) historically blew up `vol_etf_annual` and `expected_gross_decay_annual` because Yahoo lags on retro-adjusting adjclose for some issuers. **`splits.py`** centralises a multi-source pipeline that handles this:

1. `flex` (IBKR `<CorporateAction type="RS">`)
2. `yahoo_events` (v8 chart `events.splits`)
3. `splits_overrides_csv` (operator-managed `data/splits_overrides.csv`)
4. `manual_override_dict` (legacy in-code overrides)
5. `heuristic` (z-score + integer-factor matched detector)

A self-healing pre/post boundary check makes re-runs idempotent once any source retro-adjusts. The screener also emits two structured columns (`vol_ratio_value`, `vol_ratio_outlier`) that gate sleeve sizing under `screener.vol_ratio_gate` in YAML, and `ibkr_accounting.override_mark_prices` reverts to Flex `markPrice` on a symbol's split day so Yahoo close × Flex pre-split quantity can never produce a 10× phantom MtM swing.

Audit + ops workflow:

```bash
python daily_screener.py --audit-splits --run-date 2026-05-05
python daily_screener.py --audit-splits --symbols BAIG,BMNG,FIGG --write-overrides
```

Full operator playbook, edge-case table, and verification checklist live in [`SPLITS.md`](SPLITS.md).

---

## Sleeves vs accounting “buckets”

- **Sleeves** (`core_leveraged`, `yieldboost`, `inverse_decay_bucket4`, `flow_program`) are **portfolio construction** labels written into **`proposed_trades.csv`** and used by execution / rebalancing.
- **Accounting buckets** (`bucket_1` … `bucket_4`) in **`ibkr_accounting.py`** are **attribution / reporting** groupings (e.g. high-delta levered vs inverse decay). They are related concepts but **not identical** to YAML sleeve names. When interpreting `pnl_bucket_*.csv`, read the accounting script headers.
- **Bucket 4 (inverse decay)** attributes PnL and exposure at the **pair** level: short inverse ETF **and** short underlying. The structural short underlying leg is sized at `held inverse ETF position × β × partial_hedge_ratio` (default `0.75`) and shows up as a **negative-signed** row in `net_exposure_bucket_4_detail.csv`, so `gross_exposure_bucket_4 > |net_exposure_bucket_4|`.
- **B4 underlying attribution rule** (`accounting.b4_underlying_attribution`):
  - **`etf_implied`** (default) — the latest `inverse_decay_bucket4` plan row wins when present; otherwise the structural short is derived from held inverse ETFs (`compute_implied_b4_short`). This recovers the short on names like BE / CLSK / CRCL even after the rebalancer nets the sleeve order into a single broker stock line and the FIFO `qty_b4` ledger loses the tag.
  - **`plan_only`** — legacy behaviour: structural short only attributed when the latest plan still carries the sleeve row.
  - **`ledger_fifo`** — disable structural shorts entirely; trust the FIFO share ledger only.
  See `accounting.b4_attribution_min_usd` (ignore noise below threshold) and `accounting.b4_partial_hedge_ratio_default` (registry fallback). The `b4_source` column in `b4_plan_ledger_reconciliation.csv` reports which signal won per underlying.
- **Stable B1/B2 ratio-split** (`accounting.b12_spot_split_method`, `accounting.b12_pnl_mode`):
  - **Spot ↔ ETF sleeve rule** — physical spot on an underlying may only offset **held** ETF sleeves on that name: B1 spot requires a levered ETF (β > 1.5), B2 spot requires a standard/yieldboost ETF (0 < β ≤ 1.5), B4 spot requires a non-flow inverse ETF. Order references that name an ETF apply only when that sleeve is present; stale ledger `qty_b2` with no B2 ETF is zeroed at report time.
  - **Full Flex replay** — with `ledger_full_replay_include_bucket2: true` (default), every underlying with a bucket-2 ETF in the screened map or open book replays all trades from the restate window; add extras via `ledger_full_replay_underlyings` (e.g. MSTR, INTC).
  - **`b12_spot_exposure_method`** (default `sleeve_balance`) — ratio-split **net exposure** only. **`sleeve_balance`**: long spot offsets each short ETF sleeve up to `|etf_net|` (flat B1/B2 when fully hedged); **unpaired stock** → `net_exposure_unbucketed` (not forced into B1). **`hedge_ratio`**: B2 hedge first, then B1; orphan → B1. **B4 ratio-split** (`totals.json` → `net_exposure_bucket_4`): inverse ETF legs (100% B4) **plus** structural short underlying carved from the spot line (plan or ETF-implied at `partial_hedge_ratio`, default **0.75**). Reconciliation: `B1 + B2 + B4 + unbucketed = book`. **`b12_spot_split_method`** (default `ledger_fifo`) still drives **PnL** spot (`ratio_spot_*`) and may differ from exposure until the B2 share ledger is replayed. Files: `net_exposure_bucket_{1,2,4}.csv`, `net_exposure_unbucketed.csv`, `net_exposure_bucket_4_detail.csv`.
  - **`ledger_fifo`** (exposure method) — exposure spot matches FIFO ledger (legacy).
  - **`held_exposure_waterfall`** — legacy ratio-split: spot is carved by held ETF hedge-residual + plan/ETF-implied B4 structural short (large B1↔B2 relabeling).
  - **`lot_timed_strict`** (default PnL) — FIFO lot ledger for realized and unrealized; plan-B4 `inject_slice` uses ledger unrealized weights (not held exposure when orphan). Yieldboost list (`yieldboost_spot_b2_underlyings`) forces spot→B2 when a B2 sleeve is held.
  - **B4 pair view** (`net_exposure_bucket_4_detail`, `gross_exposure_bucket_4_pair`) always uses `etf_implied` structural shorts — independent of `b12_*` knobs.
- **Combined stock-sleeve views** (`pnl_by_underlying.csv`, `net_exposure_by_underlying.csv`) sum **buckets 1 + 2 + 4** (flow inverse bucket 3 remains separate). Legacy bucket-1&2-only PnL is still written to `pnl_by_underlying_b12.csv`.

---

## Research & backtests

- **`notebooks/Simple_Pair_Backtest.ipynb`** — end-to-end **pair backtests** (Bucket‑4 hedge tests, **v6 Option‑2** dynamic hedge vs static **h=0.5**, optional **regime overlays** and **grid search** cells, equity / **h** plots). This is **research code**; production sizing parameters live in `config/strategy_config.yml`.
- **`notebooks/Diamond_Creek_*.ipynb`**, **`IBKR_Backtest_Large_AUM.ipynb`** — fund-style / attribution experiments.

---

## Automation (GitHub Actions)

**`.github/workflows/eod_pnl_email.yml`**

1. **Weekday schedule (`cron`)** — America/New_York calendar date for `RUN_DATE`.
2. **Push to `main`** — re-runs the screener + trade plan only (not the EOD email pipeline). Skips commits whose message starts with `screener:`, `eod:`, or `risk_dashboard:` so bot data refreshes do not loop.
3. **Job `screener`:** `pip install -r requirements.txt` → `daily_screener.py --run-date "$RUN_DATE" --skip-ibkr-check` → `generate_trade_plan.py --run-date "$RUN_DATE"` → commits `data/etf_screened_today.csv`, `data/proposed_trades.csv`, and dated run outputs.
4. **Job `eod`:** depends on `screener`; runs on schedule / manual dispatch only — `ibkr_flex.py`, `ibkr_accounting.py`, `run_eod_pnl_email.py` with **secrets** for Flex queries and SMTP; may commit under `data/`.

Use **workflow_dispatch** to run screener-only, EOD-only, or both.

**`.github/workflows/risk_dashboard.yml`** — runs after the EOD job and rebuilds the risk dashboard (`risk_dashboard/data/latest.json`), then redeploys the static SPA under `site/` to GitHub Pages (snapshot bundled as `site/data/latest.json`). Optional **login id + password** via `site/data/investors.json` (same pattern as etf-dashboard). See [`risk_dashboard/README.md`](risk_dashboard/README.md) for setup.

**`.github/workflows/universe_discovery.yml`** -- weekday/manual discovery for new single-stock leveraged ETFs. It crawls configured issuer and exchange sources in `config/levered_etf_discovery.yml`, verifies issuer + exchange evidence, resolves the underlying/proxy, checks live market data, patches `daily_screener.py` in this repo and sibling `GoldmanDrew/Diamond-Creek-Quant`, then opens PRs only when new verified pairs are found. Every run uploads a discovery report with rejection reasons; manual dispatch supports `allow_issuer_only`, `skip_market_data`, `allow_pending_market_data`, and `allow_future_listings` for operator review runs. Cross-repo PRs require `GH_PAT` with write access to both repositories.

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
| **`risk_dashboard/data/latest.json`** | JSON snapshot consumed by the static SPA in `site/` (deployed to GitHub Pages by the risk dashboard workflow). |
| **`risk_dashboard/data/<YYYY-MM-DD>.json`** | Historical per-day snapshots. |
| **`site/`** | Static SPA shell (HTML/CSS/JS); deployed to Pages with bundled `site/data/latest.json` and optional investor login. |

**`data/borrow_history.json` (optional):** If present before `daily_screener.py` runs, it is **auto-loaded** (no CLI flag required) from, in order: `BORROW_HISTORY_PATH` / `--borrow-history-path`, `ETF_DASHBOARD_ROOT/data/borrow_history.json`, sibling `../etf-dashboard/data/borrow_history.json`, or this repo’s `data/borrow_history.json`. The scheduled **GitHub Action** downloads it from private `magis-capital-partners/etf-dashboard` through the authenticated GitHub API when the token succeeds. That enables **weighted borrow resampling** for `net_edge_*` plus `net_edge_hist_json` / p25 / p75 on the CSV.

---

## Phase 2b setup — tax-aware resize & loss-harvest substitution

Phase 2b is a fourth phase in `rebalance_strategy.py` that bidirectionally trims/grows existing pair legs back to plan target sizes whenever the leg notional drifts outside a hysteresis band. It runs between Phase 2 (Establish) and Phase 3 (Hedge):

| Phase | Purpose |
| --- | --- |
| 1 | Cleanup — close pairs the new plan no longer wants |
| 2 | Establish — open new pairs from the plan |
| **2b** | **Resize — trim/grow surviving pairs back to plan targets (band-gated)** |
| 3 | Hedge — net-exposure correction via the configured hedge ETF |

### Stage 1 — bands (always on)

`portfolio.rebalance.resize` in `config/strategy_config.yml`:

```yaml
resize:
  enabled: true
  enter_band_pct: 0.15        # trigger when leg notional drifts >15%
  exit_band_pct:  0.05        # trim/grow until within 5% (hysteresis)
  min_trim_usd: 250
  min_grow_usd: 250
```

Telemetry is appended to `data/runs/<run_date>/rebalance/resize_decisions.csv` (one row per leg evaluated, including skips).

### Stage 2 — tax routing & ETF substitution (opt-in)

When enabled, Phase 2b TRIMs (SELL on long legs, BUY-to-cover on shorts) are routed through a tax-aware classifier (`tax_router.py`):

| Situation | Routing |
| --- | --- |
| GAIN + `prefer_long_term_lots: true` and LT inventory available | Limit qty to LT shares, prefer LT in lot ordering (`lt_only_trim`) |
| LOSS ≥ `min_loss_usd_to_substitute` and substitute available | SELL original + BUY equivalent-notional substitute (`harvest_sub_sell` / `harvest_sub_buy`) |
| LOSS < floor, or no substitute, or substitution disabled | Pure trim — book the realized P&L as-is |
| LOSS ≥ floor with no eligible substitute | Defer — skip the trim with `no_substitute_for_loss_trim` |

State for active swaps is persisted to `data/active_substitutions.json`. While a substitute is held, it is excluded from being chosen as a substitute for any other underlying.

> **IBKR API constraint:** The TWS API does **not** support per-order SpecID tax-lot designation. The router only **predicts** realized P&L under the assumed account default lot method. The broker still matches lots according to TWS's account-level setting. A one-time TWS configuration step is therefore required.

#### One-time TWS Account Configuration

1. Open **Trader Workstation → File → Global Configuration → API**, ensure socket clients are enabled (already required for the rest of this repo).
2. Open **Account → Account Configuration → Tax Optimizer Default Match Method** (in IBKR Client Portal: *Account → Settings → Tax Optimizer Default*).
3. Set the default match method to match `portfolio.rebalance.tax.lot_method_assumed` in `config/strategy_config.yml` — typically **HIFO** (highest in, first out) or **MaxLossUtilization**.
4. The account default applies to all new closes; you can still override per-close in the **Tax Optimizer** GUI post-trade.

#### One-time Flex Query setup (lot-level data)

`tax_lot_view.py` reads `data/runs/<run_date>/accounting/flex_positions.xml` (the same file already used by `ibkr_accounting.py`) but expects lot-level detail.

In **Account Management → Reports → Flex Queries** for your "Open Positions" Flex query:

- Section: **Open Positions**
- **Level of Detail: Lot** (not Summary)
- Required fields: `Symbol`, `Position`, `Cost Basis Price`, `Cost Basis Money`, `Open Date Time`, `Holding Period Date Time`, `Originating Order ID`, `Mark Price`

Re-run the Flex export. If the file is missing or still summary-level when Phase 2b runs, the tax router silently falls back to "no lot data → pure trim" behavior (graceful degradation).

#### Enabling Stage 2

Edit `config/strategy_config.yml`:

```yaml
portfolio:
  rebalance:
    tax:
      enabled: true
      lot_method_assumed: "HIFO"          # MUST match TWS Tax Optimizer default
      prefer_long_term_lots: true
      st_lt_holding_days: 365

  substitution:
    enabled: true
    min_loss_usd_to_substitute: 500.0
    hold_substitute_days: 31
    underlyings:
      IBIT:  ["FBTC", "BITB", "ARKB"]     # spot-BTC ETFs
      ETHA:  ["FETH", "ETHE"]             # spot-ETH ETFs
      # Add per-underlying pools you consider "not substantially identical"
```

> **Wash-sale judgment is yours.** The engine treats the configured pool as authoritative — it does not opine on whether two ETFs are "substantially identical" for IRS §1091 purposes. Consult tax counsel; conservative pools typically draw from different issuers tracking the same asset (e.g. iShares vs Fidelity vs Bitwise spot-BTC ETFs).

#### Stage 2 telemetry

`resize_decisions.csv` adds columns:

| Column | Meaning |
| --- | --- |
| `est_realized_pnl_usd` | Predicted P&L under the assumed lot method (signed) |
| `st_qty_consumed`, `lt_qty_consumed` | Predicted ST / LT split of consumed qty |
| `lots_consumed_count` | Number of distinct lots touched |
| `substitute_of` | If this row is the BUY-substitute leg, the symbol it replaced |
| `swap_with` | Partner symbol when the row is part of a swap |
| `harvested_loss_usd` | Magnitude of realized loss for `harvest_sub_*` rows |

`decision` widens to include `harvest_sub_sell`, `harvest_sub_buy`, `lt_only_trim` in addition to `trim` / `grow` / `skip`.

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
