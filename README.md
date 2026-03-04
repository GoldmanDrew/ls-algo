# ls-algo — Leveraged ETF Decay Arbitrage

A systematic long/short strategy that captures **volatility drag** (decay) from leveraged, inverse, and income ETFs while maintaining beta-neutral exposure to the underlying securities.

## Strategy Overview

Leveraged ETFs suffer from a mathematical certainty: daily rebalancing erodes value over time through the compounding of returns. For a leveraged ETF with leverage factor β and underlying annualised volatility σ, the expected annualised decay rate is:

```
expected_decay ≈ 0.5 × |β| × |β − 1| × σ²
```

This decay is significantly higher for **inverse ETFs** (β < 0) due to the |β−1| term. For example, a −2× inverse ETF on an underlying with 30% vol has ~3× the decay of its +2× bull counterpart.

The strategy:
1. **Short** the leveraged/inverse ETF to capture decay
2. **Long** the underlying to maintain beta neutrality
3. **Size** positions proportional to net decay (after borrow costs)
4. **Rebalance** to current equity for compounding

## Architecture

```
daily_screener.py          Build universe → betas → borrow → screen → decay → CSV
  ↓
generate_trade_plan.py     Screened CSV → decay-aware sizing → proposed_trades.csv
  ↓
execute_trade_plan.py      Proposed trades → IBKR orders (parallel, beta-hedged)
execute_flow_program.py    Daily flow sleeve deployment (inverse ETF shorts)
```

Shared logic lives in `core/`:

```
core/
├── symbols.py        Ticker normalisation, IB symbol mapping
├── config.py         YAML config loading, path resolution
├── portfolio.py      Decay scoring, position sizing, hedge ratios
└── ibkr.py           IBKR connection, pricing, order helpers
```

## Daily Pipeline

### 1. Screen (`daily_screener.py`)

Builds the full ETF universe (~400 pairs) from hardcoded lists plus scraped YieldMax/Roundhill products. For each pair:

- Computes **OLS beta** from total-return series (dividends included)
- Fetches **IBKR borrow rates** via FTP
- Calculates **realised gross decay** (historical path-dependent PnL from shorting ETF + hedging)
- Calculates **expected gross decay** from the theoretical volatility drag formula
- Screens by borrow cost thresholds, with purgatory band and protected ETF logic

```bash
python daily_screener.py                    # full run
python daily_screener.py --skip-ftp         # skip borrow fetch (use cached)
python daily_screener.py --lookback 1y      # shorter history
```

Output: `data/etf_screened_today.csv`

### 2. Generate Trade Plan (`generate_trade_plan.py`)

Two stock sleeves plus a flow overlay:

**Core Leveraged (88%)** — All eligible pairs with |β| ≥ 1.5 and borrow ≤ 8%. Sized by **decay score**: pairs with higher net decay (after borrow) get proportionally larger allocations, capped at 10% per underlying.

**Whitelist Stock (12%)** — Curated income/covered-call ETFs, zipf-weighted by list order with 20% per-name cap.

**Flow Program (overlay)** — Daily deployment of $650 across inverse ETF shorts (SDS, SQQQ, etc.) with fixed weights. Tracked via cumulative ledger.

**Compounding**: Target gross exposure = current equity × leverage multiplier. Equity is read from IBKR account data (configurable to static fallback).

```bash
python generate_trade_plan.py               # uses today's date
python generate_trade_plan.py --run-date 2025-06-01
```

Output: `data/proposed_trades.csv`

### 3. Execute (`execute_trade_plan.py`)

Parallel execution against IBKR TWS/Gateway:

1. **Cleanup pass** — Close ETF legs that are held but not in the current plan (respecting purgatory freeze)
2. **Bucket execution** — For each underlying group, trade ETF shorts first, then hedge with underlying longs. Uses Adaptive Market orders with escalating priority.
3. **Post-pass hedge** — Sweep all underlyings to ensure beta neutrality after all trades settle

Safety features: FTP short availability gate, IB Error 201 graceful handling, coordinator cancel service, SIGINT shutdown, position-confirm retry loops.

```bash
python execute_trade_plan.py                # live execution
DRY_RUN=1 python execute_trade_plan.py      # dry run (no orders)
```

### 4. Flow Program (`execute_flow_program.py`)

Independent daily runner for the flow sleeve. Computes today's deployment USD, converts to shares, and sells (increases short). Updates cumulative ledger.

```bash
python execute_flow_program.py              # daily flow deployment
```

## Position Sizing: Decay Score

Each eligible pair receives a **net decay score**:

```
realised_gross  = historical path-dependent decay (from screener)
expected_gross  = 0.5 × |β| × |β−1| × σ² × 252

blend_weight    = min(1, days_of_data / 252)   # trust realised more with more data
blended_gross   = blend_weight × realised + (1 − blend_weight) × expected

borrow_cost     = current annual borrow rate

net_score       = max(0, blended_gross − borrow_cost)
```

Scores are normalised to weights, capped at 10% per underlying, and re-normalised. This means:

- High-vol underlyings with cheap borrow → largest positions
- Pairs where borrow exceeds decay → zero allocation
- New ETFs with little history → sized conservatively via theoretical decay
- As history accumulates, realised decay dominates the estimate

## Configuration

All parameters in `config/strategy_config.yml`:

| Section | Key | Description |
|---------|-----|-------------|
| `strategy.capital_usd` | 330000 | Fallback equity if IBKR unavailable |
| `strategy.gross_leverage` | 4 | Target gross / equity ratio |
| `strategy.equity_source` | "static" | Where to read current equity: ibkr/static/file |
| `screener.borrow_low` | 0.08 | Max borrow for core sleeve (8%) |
| `screener.purgatory_margin` | 0.04 | Band above borrow_low for purgatory |
| `portfolio.rebalance.drift_trigger_pct` | 0.15 | Rebalance if name drifts >15% |
| `portfolio.rebalance.min_trade_usd` | 500 | Skip deltas below $500 |
| `core_leveraged.weighting.method` | "decay_score" | Sizing method: decay_score or equal |
| `core_leveraged.weighting.max_name_weight` | 0.10 | 10% concentration cap per underlying |
| `flow_program.fixed_usd_per_day` | 650 | Daily flow deployment USD |

## Requirements

```
python >= 3.10
ib_insync
pandas
numpy
yfinance
pyyaml
requests
beautifulsoup4
```

IBKR TWS or Gateway must be running with API enabled on the configured port.

## Data Files

| File | Description |
|------|-------------|
| `data/etf_screened_today.csv` | Full screened universe with betas, borrow, decay |
| `data/proposed_trades.csv` | Target notionals per pair for execution |
| `data/baseline_snapshot.csv` | Pre-strategy positions to subtract |
| `data/flow_ledger.csv` | Cumulative flow program deployment |
| `data/equity_snapshot.csv` | Latest IBKR account equity |
| `data/runs/<date>/` | Dated snapshots of each pipeline run |
