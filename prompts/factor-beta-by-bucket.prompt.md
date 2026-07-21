# Prompt: Net beta to SPY by bucket (risk dashboard)

Use this prompt to add or extend per-bucket factor exposure on the static risk dashboard.

## Goal

Break down **net beta to SPY** by accounting bucket and show it in the Factor exposure section of the SPA — with an **additive** sleeve partition that reconciles to the book.

## Definitions

- **Per-name beta-weighted net:** `beta_weighted_net_usd = net_notional_usd × beta_to_spy`
- **Bucket net beta to SPY:** `sum(beta_weighted_net_usd) / NAV`
- **Book net beta to SPY:** same formula on `net_exposure_by_underlying.csv` (underlying rollup)

Betas come from `risk_dashboard.beta_loader.compute_betas()` (252-day OLS log returns vs SPY), with curated fallback via `factor_map.lookup_underlying()`.

## Data sources

| File | Use |
|------|-----|
| `data/runs/<date>/accounting/bucket_exposure_detail.csv` | Primary: ratio-split legs (`_ratio_b1/_b2/_b4`) for additive B1/B2/B4 |
| `data/runs/<date>/accounting/net_exposure_unbucketed.csv` | Additive residual sleeve |
| `data/runs/<date>/accounting/net_exposure_bucket_3.csv` | Overlay (flow program) — not in additive sum |
| `data/runs/<date>/accounting/net_exposure_bucket_5.csv` | Overlay (vol ETP pair view) — not in additive sum |
| `data/runs/<date>/accounting/net_exposure_by_underlying.csv` | Book-level factor panel |
| `data/cache/beta_summary.json` | Cached OLS betas |

**Additive partition:** B1 + B2 + B4 (ratio-split from detail) + unbucketed ≈ book.  
**Overlays:** B3 and B5 are informative only; excluded from `by_bucket_beta_weighted_net_usd`.  
Warn when additive sleeve β-wtd net ≠ book total (`by_bucket_reconciles === false`).

## Backend (Python)

1. `compute_factor_by_bucket(accounting_dir, nav_usd, beta_results=…)` in `risk_dashboard/metrics.py`.
2. Prefer `bucket_exposure_detail.csv`: scale each leg by `_ratio_bK`, aggregate by underlying, apply betas.
3. Emit per bucket:
   - `bucket`, `bucket_label`, `n_names`
   - `net_notional_usd`, `gross_notional_usd`
   - `beta_weighted_net_usd`, `beta_weighted_gross_usd` (+ QQQ/IWM/BTC variants)
   - `net_beta_to_spy`, `gross_beta_to_spy`
   - `implied_avg_beta` (= β-wtd net / net $)
   - `top_beta_names` (top 5 by |β-wtd net|)
   - `additive`, `role` (`sleeve` / `overlay` / `unbucketed`), `attribution_mode`
   - `pct_of_portfolio_beta_net` (share of book β-wtd net; overlays null)
4. In `build_snapshot()`, attach `factor_panel["by_bucket"]` and totals:
   - `by_bucket_beta_weighted_net_usd` (**additive only**)
   - `by_bucket_net_beta_to_spy` (+ QQQ/IWM/BTC)
   - `by_bucket_overlay_beta_weighted_net_usd`
   - `by_bucket_reconciles` (additive within 2% of book β-wtd net)

## Frontend (static site)

In `site/index.html` Factor section, table `#factor-by-bucket`.

In `site/assets/js/app.js` → `renderFactor()`:

| Column | Field |
|--------|-------|
| Bucket | `bucket_label` (+ ` (overlay)` for non-additive) |
| Net $ | `net_notional_usd` |
| Net β SPY/QQQ/IWM/BTC | `net_beta_to_*` (format as `0.35x`) |

Order: additive sleeves → **Additive sleeves** subtotal → overlays → **Book total**.

Show a warning callout when `by_bucket_reconciles === false`.

## Verify

```bash
python -m risk_dashboard.build_site --run-date YYYY-MM-DD
python -m pytest risk_dashboard/tests/test_metrics.py -q -k "factor_by_bucket or factor_panel"
```

Copy `risk_dashboard/data/latest.json` → `site/data/latest.json` for local preview.

## Bucket labels

- `bucket_1` → Bucket 1 (core leveraged)
- `bucket_2` → Bucket 2 (yield boost)
- `bucket_4` → Bucket 4 (inverse / decay)
- `unbucketed` → Unbucketed
- `bucket_3` → Bucket 3 (flow hedge overlay) — overlay
- `bucket_5` → Bucket 5 (volatility ETP) — overlay
