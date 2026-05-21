# Prompt: Net beta to SPY by bucket (risk dashboard)

Use this prompt to add or extend per-bucket factor exposure on the static risk dashboard.

## Goal

Break down **net beta to SPY** by accounting bucket (`bucket_1` … `bucket_4`) and show it in the Factor exposure section of the SPA.

## Definitions

- **Per-name beta-weighted net:** `beta_weighted_net_usd = net_notional_usd × beta_to_spy`
- **Bucket net beta to SPY:** `sum(beta_weighted_net_usd) / NAV`
- **Book net beta to SPY:** same formula on `net_exposure_by_underlying.csv` (underlying rollup)

Betas come from `risk_dashboard.beta_loader.compute_betas()` (252-day OLS log returns vs SPY), with curated fallback via `factor_map.lookup_underlying()`.

## Data sources

| File | Use |
|------|-----|
| `data/runs/<date>/accounting/net_exposure_bucket_1.csv` … `_4.csv` | Per-bucket net/gross by underlying |
| `data/runs/<date>/accounting/net_exposure_by_underlying.csv` | Book-level factor panel (may not sum to buckets) |
| `data/cache/beta_summary.json` | Cached OLS betas |

**Important:** Bucket CSVs can **double-count** legs vs the underlying rollup (especially bucket 3 flow overlay and bucket 4 pair structure). The UI must show a reconciliation note when bucket β-wtd net sum ≠ book total.

## Backend (Python)

1. Add `compute_factor_by_bucket(accounting_dir, nav_usd, beta_results=…)` in `risk_dashboard/metrics.py`.
2. For each bucket, read `net_exposure_{bucket}.csv`, resolve `beta_to_spy` per underlying (same logic as `compute_factor_panel`).
3. Emit per bucket:
   - `bucket`, `bucket_label`, `n_names`
   - `net_notional_usd`, `gross_notional_usd`
   - `beta_weighted_net_usd`, `beta_weighted_gross_usd`
   - `net_beta_to_spy`, `gross_beta_to_spy`
   - `implied_avg_beta` (= β-wtd net / net $)
   - `top_beta_names` (top 5 by |β-wtd net|)
   - `pct_of_portfolio_beta_net` (share of book β-wtd net)
4. In `build_snapshot()`, attach `factor_panel["by_bucket"]` and totals:
   - `by_bucket_beta_weighted_net_usd`
   - `by_bucket_net_beta_to_spy`
   - `by_bucket_reconciles` (within 2% of book β-wtd net)

## Frontend (static site)

In `site/index.html` Factor section, add table `#factor-by-bucket`.

In `site/assets/js/app.js` → `renderFactor()`:

| Column | Field |
|--------|-------|
| Bucket | `bucket_label` |
| Names | `n_names` |
| Net $ | `net_notional_usd` |
| Gross $ | `gross_notional_usd` |
| β-wtd net $ | `beta_weighted_net_usd` |
| Net β SPY | `net_beta_to_spy` (format as `0.35x`) |
| % of book β | `pct_of_portfolio_beta_net` |
| Avg β | `implied_avg_beta` |

Include a **Book total** footer row from `factor_panel.totals`.

Show a warning callout when `by_bucket_reconciles === false`.

## Verify

```bash
python -m risk_dashboard.build_site --run-date YYYY-MM-DD
python scripts/_bucket_beta_print.py
python -m pytest risk_dashboard/tests/test_metrics.py::test_compute_factor_by_bucket_aggregates_beta_weighted_net -q
```

Copy `risk_dashboard/data/latest.json` → `site/data/latest.json` for local preview.

## Bucket labels

- `bucket_1` → Bucket 1
- `bucket_2` → Bucket 2
- `bucket_3` → Bucket 3 (flow hedge overlay)
- `bucket_4` → Bucket 4
