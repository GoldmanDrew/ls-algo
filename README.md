# LS-ALGO

LS-ALGO is a Python-based pipeline for running a rules-driven long/short ETF strategy end to end. The system separates periodic research, daily screening and risk monitoring, baseline portfolio construction, and trade execution, enabling a controlled, repeatable workflow suitable for systematic deployment.

## Workflow Overview

### 1) Periodic Research (Weekly / Ad Hoc)
**Purpose:** Generate and refresh historical pair-level return statistics used by the strategy.

- Jupyter notebooks in `notebooks/` compute historical pair CAGRs and related analytics.
- Outputs are written to `config/etf_cagr.csv`.
- This step is not run daily and is intended to be updated periodically (e.g., weekly or when the universe changes).

**Key inputs:** ETF universe definitions, historical price data.

**Key output:** `config/etf_cagr.csv` — authoritative source of pair mappings and historical performance.

### 2) Daily ETF Screening
**Script:** `etf_screener.py`

**Purpose:** Build the daily tradable ETF universe based on borrow economics and availability.

- Loads ETF metadata and historical CAGRs from `config/etf_cagr.csv`.
- Downloads the IBKR short-stock file (via FTP).
- Computes net borrow (fee − rebate), shares available, and screening flags.
- Applies rule-based filters (borrow caps, minimum shares available, optional whitelist overrides).
- Writes the screened universe to `data/etf_screened_today.csv`.

This file is the single source of truth for all downstream daily steps.

### 3) Daily Short Borrow Monitoring
**Script:** `short_stock_monitor.py`

**Purpose:** Detect changes in short borrow conditions and surface risk alerts.

- Loads the active ETF watchlist from `data/etf_screened_today.csv`.
- Saves a dated snapshot of the IBKR short-stock file under `data/shortstock_snapshots/`.
- Compares against the most recent prior snapshot.
- Flags borrow rate spikes and availability drops (thresholds configurable via environment variables).
- Alerts are printed and can optionally be emailed.

This step runs daily and independently of trading, serving as a risk and ops monitor.

### 4) Baseline Portfolio Snapshot
**Script:** `baseline_snapshot.py`

**Purpose:** Establish the current “baseline” portfolio state against which new trades are generated.

- Captures existing positions and quantities.
- Persists the baseline to `data/baseline_snapshot.csv`.

This snapshot ensures that trade generation is delta-based, not absolute.

### 5) Trade Plan Generation
**Script:** `generate_trade_plan.py`

**Purpose:** Translate the screened universe into actionable trades.

- Reads `data/etf_screened_today.csv` and `data/baseline_snapshot.csv`.
- Applies strategy sizing, pairing, and exposure rules.
- Produces a proposed trade list at `data/proposed_trades.csv`.

No orders are sent at this stage.

### 6) Trade Execution (Optional)
**Script:** `execute_trade_plan.py`

**Purpose:** Execute the generated trade plan via Interactive Brokers.

- Reads `data/proposed_trades.csv`.
- Connects to IBKR using `ib_insync`.
- Submits Adaptive Passive limit orders.
- Supports `DRY_RUN=1` to log intent without placing orders.
- Saves execution and position state for auditability.

## Repository Layout

```
config/
  etf_cagr.csv              # Historical pair CAGRs (from notebooks)
  strategy_config.yml       # Strategy parameters

data/
  etf_screened_today.csv    # Daily screened universe
  baseline_snapshot.csv     # Current portfolio baseline
  proposed_trades.csv       # Generated trade plan
  shortstock_snapshots/     # Daily borrow snapshots

notebooks/
  leveraged_etf_research.ipynb
  etf_universe.ipynb        # Research notebooks for CAGR generation

scripts/
  run_daily_pipeline.sh     # Optional local wrapper

etf_screener.py
short_stock_monitor.py
baseline_snapshot.py
generate_trade_plan.py
execute_trade_plan.py

.github/workflows/
  daily_pipeline.yml        # CI for daily screening & monitoring

requirements.txt
README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure access to the IBKR FTP short-stock feed and, if trading live, an IBKR TWS/Gateway session with API enabled.

## Running Locally

### Individual scripts
```bash
python etf_screener.py
python short_stock_monitor.py
python baseline_snapshot.py
python generate_trade_plan.py
python execute_trade_plan.py
```

### Optional full daily sequence
Export any needed environment variables (see below), then run:
```bash
./scripts/run_daily_pipeline.sh
```

## Key Environment Variables

### Screener
- `CAGR_CSV` — Path to `etf_cagr.csv` (default `config/etf_cagr.csv`).
- `OUTPUT_DIR`, `OUTPUT_FILE` — Where to write the screened CSV (default `data/etf_screened_today.csv`).
- `IBKR_FTP_HOST`, `IBKR_FTP_USER`, `IBKR_FTP_PASS`, `IBKR_FTP_FILE` — FTP connection and filename (`usa.txt` by default).
- `BORROW_CAP` — Maximum acceptable borrow before exclusion (default `0.10`).
- `MIN_SHARES_AVAILABLE` — Minimum available shares required (default `1000`).

### Monitoring & Alerts
- `SCREENED_CSV` — Watchlist source (defaults to screener output).
- `SNAPSHOT_DIR` — Directory for dated short-stock CSV snapshots (default `data/shortstock_snapshots`).
- `IBKR_FTP_FILE` — FTP filename to monitor (`usa.txt` default).
- `BORROW_ABS_THRESHOLD`, `BORROW_CHANGE_THRESHOLD` — Borrow-level alert thresholds.
- `AVAIL_ABS_THRESHOLD`, `AVAIL_CHANGE_THRESHOLD` — Availability alert thresholds.
- `EMAIL_FROM`, `EMAIL_TO`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS` — Email settings.

### Trading
- `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID` — IBKR API connection details.
- `DRY_RUN` — Set to `1` to log actions without placing orders.
- `MAX_SHARES_PER_ORDER` — Clamp per-leg order size (default `500`).

## Outputs

- `data/etf_screened_today.csv` — Screened ETF universe with borrow metrics and inclusion flags.
- `data/baseline_snapshot.csv` — Current portfolio baseline.
- `data/proposed_trades.csv` — Generated trade plan.
- `data/shortstock_snapshots/` — Historical short borrow files and comparisons used for alerts.

## Automation

- **Daily ETF Pipeline** (`.github/workflows/daily_pipeline.yml`): runs screening and monitoring with secrets for FTP, IBKR, and email.
