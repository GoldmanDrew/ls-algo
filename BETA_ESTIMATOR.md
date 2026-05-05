# Robust hierarchical hedge-beta estimator

`beta_estimator.py` replaces the old shrunk-OLS estimator
(`daily_screener.compute_beta_shrunk` — still kept around for back-compat
imports / tests) for the purpose of producing the **`Beta`** column in
`data/etf_screened_today.csv`.

The estimator is consumed by `ibkr_accounting.compute_net_exposure` to
size beta-adjusted notional exposure, by `rebalance_strategy.compute_beta_adjusted_net_notional`
to drive sleeve-level rebalancing, and by `screener_v2_fields` to derive
`product_class` etc.  This document captures the design choices.

## Why the old estimator was wrong for income sleeves

The legacy rule was

```
β̂ = w·β̂_OLS + (1−w)·L
k  = 60·max(1, L²)
w  = n_eff / (n_eff + k)
```

with `Leverage = 1.0` hardcoded for the entire **income sleeve**
(`covered_call_pairs ∪ YIELDBOOST_BUCKET2_PAIRS`, plus scraped
YieldMax / Roundhill rows).  That prior is structurally wrong:

- **L = 1 is not a “listed leverage”** for covered-call or YieldBOOST
  overlays.  Their hedge ratio to the underlying is **&lt; 1 by design**
  (option premium income, distribution-day NAV gaps, capped upside on
  call overwriting, distance-weighted weekly put-spreads).  Shrinking
  toward 1 systematically over-hedges those positions in the direction
  of the underlying.
- **Magnitude discontinuity** along the history axis — high `n_eff`
  rows kept β near OLS, low `n_eff` rows snapped to 1.0.  That is an
  artifact of the prior, not economics.
- **No peer/family information** across YieldBOOST siblings even
  though they share product structure (95/88 weeklies on a 2× sleeve)
  and only differ by underlying.
- **No uncertainty signal** for hedge sizing.
- **Single-horizon OLS** is brittle to non-synchronous closes,
  microstructure, and big single-day moves on the ETF (distribution
  days on income products in particular).

## What the new estimator does

`compute_beta_for_hedging(etf_tr, und_tr, prior, *, min_days=60)`
computes a Gaussian-conjugate posterior

```
β̂ = (τ·μ + n_eff·β_robust) / (τ + n_eff)
```

where

- **`β_robust`** is the slope of the ETF on the underlying using
  *aligned daily log returns*, EWMA-weighted (half-life 63 trading days
  with a 25 % floor on the long tail), iterated through Huber IRLS
  (`k = 1.345 · MAD`, three reweighting passes).
- **`n_eff`** is the sum of EWMA weights — naturally smaller for newly
  launched ETFs even when the calendar window is long.
- **`(μ, τ)`** are the prior mean and prior precision (in *effective
  trading days*).  See **prior router** below.
- **HAC SE** (Newey–West, lag 5) on the slope feeds `Beta_se` in the
  CSV.  This is the hedge-quality signal sizing code can read.
- **Multi-horizon stability gate.**  We fit at `h ∈ {1, 5}`
  (overlapping log-return sums for `h = 5`).  When
  ``|β₁ − β₅| > max(0.25·|μ|, 2·SE(β₅))`` we prefer **`h = 5`** and
  inflate `τ` by 2× — this corrects for non-synchronous closes
  (microstructure) and one-off liquidity events.
- **Distribution-day guard.**  When an `actions_mask` (Yahoo
  `actions` events) is supplied, daily returns whose residual
  `|r_etf − μ·r_und|` exceeds 25 % AND coincide with a same-day
  distribution event are dropped before the fit.  Without an actions
  mask the threshold is applied alone (unit tests exercise this
  path).
- **Sign conflict — graceful, not a snap.**  If
  ``sign(β_robust) ≠ sign(μ)`` and ``|β_robust| > 0.3``, we **inflate
  `τ` by 5×** and re-blend.  We never hard-snap to `μ = L`; the
  resulting posterior is strictly between the data and the prior.

The estimator returns a `BetaResult` with `(beta, beta_se, n_obs,
n_eff, resid_sigma_annual, horizon, quality, source, prior_*, extras)`.
All of this lands on the screener CSV.

## Prior router — `BetaPrior.for_row`

The **central contract**: nominal listed leverage `L` is used as the
prior mean **only** for the two LETF classes.  Everywhere else the
prior comes from a calibration constant or the YieldBOOST family
hierarchy.  `Beta_prior_source = "nominal_L"` MUST NEVER appear on a
non-LETF row — there is a unit test for this.

| `product_class`        | `μ_β`                                                         | `τ_β` | Prior source tag |
|------------------------|---------------------------------------------------------------|------:|-------------------|
| `letf_long`            | nominal `L` (typically 2)                                     | `60·L²` | `nominal_L` |
| `letf_inverse`         | nominal `L` (e.g. -2, -3, -1.5, signed)                       | `60·L²` | `nominal_L` |
| `covered_call_1x`      | calibrated constant (default **0.55**)                        | 30    | `covered_call_default` |
| `income_yieldboost`    | leave-one-out median of sibling raw-OLS β on same underlying  | min(120, n_sib · 30) | `yieldboost_peer_<UND>` / `yieldboost_global` |
| `volatility_etp`       | 0 (no shrinkage to a hedge ratio)                             | 0     | `volatility_etp` |
| `scraped_income`       | same as `covered_call_1x`                                     | 30    | `scraped_income_default` |
| `unknown`              | 0                                                             | 15    | `empirical_bayes_global` |

The classification itself is in `daily_screener.classify_beta_product_class`.
The order matters — vol-ETPs are flagged first, then negative-leverage,
then `is_yieldboost`, then membership in `covered_call_pairs`, then a
final `Leverage`-based fallthrough.

### YieldBOOST family priors

`build_yieldboost_family_priors(yieldboost_pairs, tr_map, …)`:

1. For every `(ETF, Underlying)` pair in `YIELDBOOST_BUCKET2_PAIRS`
   with ≥ `min_days` history, compute the **raw OLS** slope of aligned
   daily log returns.  This is the family-level signal — **it does not
   use the listed `Leverage = 1`** (we explicitly do not feed `L = 1`
   into the YieldBOOST prior).
2. Aggregate per underlying:

   ```
   μ_fam(u) = leave_one_out_median(siblings_on_u)  if ≥ 2
   μ_fam(u) = sibling_β                            if 1
   τ_fam(u) = min(120, n_sib · 30)
   ```
3. Also produce `__global__ = (median over ALL sibling betas, τ_cap)`
   used as the fallback.

Whenever `BetaPrior.for_row` is called for an `income_yieldboost` row
and the underlying has at least one sibling, the per-underlying entry
is used.  Otherwise the global fallback is used.  Newly-launched
YieldBOOST ETFs without siblings (rare today) drop through to a
documented constant — they never receive `L = 1`.

## Output schema (additions to `etf_screened_today.csv`)

All existing columns are preserved.  New columns:

| Column                    | Description |
|---------------------------|-------------|
| `Beta_se`                 | Posterior standard error of `Beta` (HAC, robust). |
| `Beta_resid_sigma_annual` | Annualized residual σ at the chosen horizon (hedge tracking error). |
| `Beta_horizon_chosen`     | 1 or 5 — which horizon the multi-horizon gate selected. |
| `Beta_quality`            | `ok`, `non_stationary`, `low_n`, `imputed_missing_prices`. |
| `Beta_prior_mu`           | μ used for the posterior (after τ-inflation if applicable). |
| `Beta_prior_tau`          | τ used (effective days; reflects sign-inflation/horizon-inflation). |
| `Beta_prior_source`       | One of `nominal_L`, `covered_call_default`, `yieldboost_peer_<UND>`, `yieldboost_global`, `scraped_income_default`, `volatility_etp`, `empirical_bayes_global`. |
| `Beta_product_class`      | Classification used to build the prior. |

`Beta_source` itself is now a dotted tag: `posterior.<prior_source>` for
the new path, `imputed_*` for fallthrough.  This lets old consumers
still grep on `imputed_*` / `shrunk_to_L` style strings while exposing
the prior provenance.

## Validation snapshot — 2026-05-05

Per-ETF β change for the YieldBOOST symbols held that day:

| ETF | Underlying | β_old | β_new | Δ | Prior source |
|---|---|---:|---:|---:|---|
| FBYY | META | +0.713 | +0.529 | -0.184 | yieldboost_peer_META |
| HMYY | HIMS | +0.555 | +0.270 | -0.285 | yieldboost_peer_HIMS |
| IOYY | IONQ | +0.562 | +0.324 | -0.238 | yieldboost_peer_IONQ |
| MAAY | MARA | +0.535 | +0.307 | -0.228 | yieldboost_peer_MARA |
| MTYY | MSTR | +0.548 | +0.367 | -0.182 | yieldboost_peer_MSTR |
| NVYY | NVDA | +0.708 | +0.619 | -0.090 | yieldboost_peer_NVDA |
| QBY  | QBTS | +0.540 | +0.285 | -0.255 | yieldboost_peer_QBTS |
| TQQY | QQQ  | +0.888 | +0.941 | +0.053 | yieldboost_peer_QQQ |
| TSYY | TSLA | +0.603 | +0.538 | -0.065 | yieldboost_peer_TSLA |
| XBTY | IBIT | +0.630 | +0.525 | -0.106 | yieldboost_peer_IBIT |

Beta-adjusted **net** exposure on the same book (signed USD):

| Underlying | net_old | net_new | Δ |
|---|---:|---:|---:|
| AMD   |  -7,827 |  -7,528 |    +299 |
| BE    |  -3,612 |  -3,551 |     +61 |
| GOOGL |  -3,136 |  -3,071 |     +65 |
| INTC  |  -9,614 |  -9,456 |    +157 |
| IONQ  |  -7,070 |     +26 |  +7,096 |
| MARA  | -11,779 |  +1,014 | +12,793 |
| META  |  -3,029 |    +785 |  +3,814 |
| MSTR  |  -7,645 |    +892 |  +8,537 |
| MU    |  -4,133 |  -4,053 |     +80 |
| NVTS  | +10,029 | +10,441 |    +413 |

The big movers (MSTR, MARA, IONQ, META) all carry YieldBOOST short
legs whose hedge-equivalence to the underlying was previously
*overstated*.  After the change, the implied beta-adjusted short
shrinks.  This is the expected direction: the income overlay’s
realized co-movement with the underlying is structurally below 1.

LETF betas barely move (Δ in the basis-point range), as they should —
they were already pinned near nominal `L` by the strong listed-leverage
prior.

## Calibration

`scripts/calibrate_covered_call_prior.py` re-derives the
`covered_call_default` constant from realized 3-year history of
QYLD / XYLD / JEPI / JEPQ / SPYI / RYLD vs SPY / QQQ / IWM and writes
`data/beta_priors_calibration.json`.  If the median β lands in
`[0.45, 0.65]` we keep the constant at `0.55`; otherwise the calibrated
value should be committed.

## Things deliberately not changed

- `ibkr_accounting.compute_net_exposure` is untouched — it consumes
  `Beta` through `load_etf_beta_map` and the new column drops in.
- `compute_beta_adjusted_net_notional` and the rebalancer were not
  modified; they read the same `Beta` column.
- The legacy `compute_beta_shrunk` function is retained for the
  existing test suite (`tests/test_beta_shrinkage.py`) and is still
  used by `enrich_with_decay_and_vol` as a *fill-in for missing β*.
  That fallback applies only when `add_betas` produced a NaN (it
  doesn’t today; see `add_betas`) — for safety it is left in place
  exactly as before so a bad release cannot accidentally null out
  decay rows.

## Limitations / future work

- The current `Leverage` column in `dx_df` is uniformly `2.0` even for
  3× LETFs (e.g. `TQQQ`, `SPXL`).  That is a pre-existing classification
  bug — the new estimator faithfully shrinks toward `2.0` for those
  rows, which still gives a sensible posterior because the data
  dominates after enough history, but the **prior source tag is
  honest** (`nominal_L`) and downstream callers can read the prior
  value to detect the inconsistency.  Fixing the universe lists to
  carry true nominal `L` is a separate change.
- The distribution-day guard is currently exercised in tests; the
  production wiring inside `daily_screener` does not yet pass
  `actions_mask` through.  The estimator is ready for it as soon as
  `download_all_tr_series` returns the actions stream.
