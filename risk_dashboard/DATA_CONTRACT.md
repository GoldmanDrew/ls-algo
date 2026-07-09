# Risk Dashboard Data Contract

This document defines what each published snapshot must satisfy and how
metrics relate to the EOD email.

## Source of truth

| Domain | Authoritative source |
|---|---|
| Sleeve / book PnL (YTD) | `data/runs/<date>/accounting/totals.json` → `total_pnl`, `bucket_pnl` |
| Per-name PnL | `pnl_<bucket>.csv`, `pnl_by_underlying.csv` |
| Exposures | `totals.json` gross/net fields + `net_exposure_<bucket>.csv` |
| Broker positions / borrow | `data/runs/<date>/ibkr_flex/*.xml` |
| Daily PnL move | `data/ledger/pnl_history.csv` consecutive `total_pnl` rows |
| NAV denominator | `totals.json` → `nav_usd` / `nav_source` (broker Flex preferred) |
| Screener borrow overlay | `data/runs/<date>/etf_screened_today.csv` (pinned copy) |

The dashboard is **read-only** over accounting; it never recomputes book PnL.

## Run manifest

Every production run must have `data/runs/<date>/manifest.json` written by
`scripts/run_data_contract.py` after EOD. It records checksums, NAV source,
git SHA, and workflow run id. Snapshots embed this under `manifest` + `data_quality.lineage`.

## Metric definitions

- **P&L YTD (`pnl_ytd_usd`)** — strategy cumulative PnL (= email headline).
- **P&L today (`pnl_daily_usd`)** — change in cumulative PnL vs prior accounting
  session row in `pnl_history.csv` (fallback: prior snapshot delta).
- **Gross exposure (book)** — B1 + B2 + B4 ratio-split gross + documented overlays;
  `gross_exposure_total` in totals **excludes B5** (vol ETP sleeve).
- **Sleeve target % (dashboard)** — derived from `config/strategy_config.yml`
  using the same budget waterfall as `generate_trade_plan.py`
  (`portfolio.sleeves.*.target_weight`, B4 vol-ETP carve-out). Denominator for
  drift is `capital_usd × gross_leverage` (book target gross), not deployed
  reconcile gross. Bucket 3 (flow overlay) has no % target.
- **Bucket 5** — shown in UI and email groups; not in B1+B2+B4 reconcile gate.
- **NAV % metrics** — `metric / nav_usd` where `nav_usd` is broker-derived when available
  (`flex_positions:percentOfNAV_median` or equity Flex tags).

## Publish gates (`scripts/verify_dashboard_snapshot.py`)

A snapshot is publishable only when:

1. `run_date` equals latest accounting run (unless `--allow-stale`).
2. `manifest.json` exists for that date.
3. NAV source is broker-derived (unless `--allow-config-nav`).
4. Exposure reconciliation passes (B1+B2+B4 + unbucketed net).
5. Each bucket CSV sum matches `bucket_pnl` and snapshot rows.
6. `data_quality.status` is not `hard`.
7. `latest.json` matches the built `run_date`.

## Pipeline order (production)

1. Screener → pin `etf_screened_today.csv` under run folder.
2. EOD Flex + accounting + email → commit `data/`.
3. Write run manifest + broker NAV on totals.
4. Dashboard build + verify → commit `risk_dashboard/data/`.
5. Deploy Pages (push trigger or manual) after deploy-time freshness check.

Recovery: `.github/workflows/dashboard_recovery.yml` at 10:00 UTC Mon–Sat if
manifest or snapshot lags accounting.
