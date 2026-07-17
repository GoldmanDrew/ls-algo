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
| Hedged vs unhedged PnL lens | `data/runs/<date>/accounting/hedged_pnl_split.json`, `hedged_pnl_b4_by_pair.csv` + `data/ledger/hedged_pnl_history.csv` (written by `hedged_pnl.py`) |
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
- **Bucket 5 (book sleeve)** — shown in UI and email groups; not in B1+B2+B4 reconcile gate.
- **Bucket 5 Product dashboard** — standalone JSON
  `risk_dashboard/data/bucket5_product.json`
  (`schema: bucket5_product_dashboard.v1`) from
  `scripts/build_bucket5_product_dashboard.py`. Fetched by the **B5 Product**
  tab (not embedded in `latest.json`). Contains strategy guide, full daily
  path, sparsified marks, regime panels, and live GTP sleeve day tags.
  Deploy copies it to `site/data/bucket5_product.json`. Legacy
  `bucket5_backtest` panel may still exist for rollback but is no longer the
  primary UI.
- **NAV % metrics** — `metric / nav_usd` where `nav_usd` is broker-derived when available
  (`flex_positions:percentOfNAV_median` or equity Flex tags).
- **Hedged vs unhedged PnL (`hedged_pnl_panel`)** — additional lens on top of
  bucket accounting (buckets unchanged). Hedged = B1 + B2 + the matched slice
  of each B4 pair (short underlying offsetting the short inverse ETF up to the
  realized book hedge ratio); unhedged = B3 + B5 + the B4 slice above each
  pair's hedge ratio. YTD values are daily-accumulated in
  `data/ledger/hedged_pnl_history.csv`; hedged + unhedged must tie to the
  bucket-sum total (checked as a publish gate when the panel is present).

## Publish gates (`scripts/verify_dashboard_snapshot.py`)

A snapshot is publishable only when:

1. `run_date` equals latest accounting run (unless `--allow-stale`).
2. `manifest.json` exists for that date.
3. NAV source is broker-derived (unless `--allow-config-nav`).
4. Exposure reconciliation passes (B1+B2+B4 + unbucketed net).
5. Each bucket CSV sum matches `bucket_pnl` and snapshot rows.
6. `data_quality.status` is not `hard`.
7. `latest.json` matches the built `run_date`.
8. When `hedged_pnl_panel` is available: hedged + unhedged YTD ties to the
   bucket-sum total within $1.

## Pipeline order (production)

1. Screener → pin `etf_screened_today.csv` under run folder.
2. EOD Flex + accounting + email → commit `data/`.
3. Write run manifest + broker NAV on totals.
4. Dashboard build + verify → commit `risk_dashboard/data/`.
5. Deploy Pages (push trigger or manual) after deploy-time freshness check.

Recovery: `.github/workflows/dashboard_recovery.yml` at 10:00 UTC Mon–Sat if
manifest or snapshot lags accounting.
