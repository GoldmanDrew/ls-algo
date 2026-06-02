# Vol→VIX β (`beta_vol_vix`)

Used by the risk dashboard slide-risk strip **12M expected carry vs VIX (SPX 0%)**.

## Estimator (v2)

1. **Realized vol:** 20-day rolling annualized σ from underlying log returns.
2. **VIX:** `^VIX` close ÷ 100 (decimal).
3. **Regression:** first differences over up to **252** sessions:
   `Δσ_t = α + β · Δ(VIX_decimal)_t + ε`
4. **Shrinkage** (same shape as `daily_screener.compute_beta_shrunk`):
   - `n_eff` from AR(1) on Δσ
   - `k = 60 · max(1, β_prior²)`
   - `β_final = w·β_OLS + (1−w)·β_prior`, `w = n_eff/(n_eff+k)`
5. **Prior:** sector mean if ≥5 names in sector else product-class (`volatility_etp`→1.0, `broad`→0.5, else 0.75).
6. **Guards:** clip β to [0, 2]; large negative OLS β snaps toward prior.

## Shock map

`σ_shocked = σ_base + β · (ΔVIX_pts / 100)`, floor 5% annualized.

## Code

- `risk_dashboard/vol_vix_beta.py` — `compute_vol_vix_betas`, `ESTIMATOR_VERSION = v2_diff_ols`
- `risk_dashboard/metrics.py` — builds pack with screener product class + sector metadata
