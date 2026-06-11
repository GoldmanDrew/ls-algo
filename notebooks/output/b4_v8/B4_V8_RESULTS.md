# Bucket 4 v8 experiment: hedge clip bounds (h_mid / h_min / h_max)

Window 2025-10-07 -> latest | 45 pairs | split 2026-02-03 | k_vcr=1.0, EMA=0.25, cadence base 12/2.25/2.5/cap21, slip 20.0bps

## Why 0.55/0.30/0.80 existed

Inherited from the v6 calibration (V7_DEFAULT_H_MID, V7_GLOBAL_H_MIN/MAX in scripts/bucket4_hedge_v7.py) -- guardrails chosen to mimic v6's average hedge level, never jointly optimized.

## Selection protocol (pre-declared)

Top quartile EW mean CAGR in BOTH time halves required to qualify; best composite (full winsor mean, full median, H1 mean, H2 mean, ret/vol) among qualifiers wins; plateau and 20-fold pair CV reported as overfit checks.

## Top 10 combos

```
 h_mid  h_min  h_max  stable  ew_mean_full  ew_median_full  ew_mean_h1  ew_mean_h2  ew_mean_vol  ew_mean_dd  ret_over_vol  plateau_ew_mean  composite_rank
  0.20   0.15   0.70   False      0.515395        0.190776   -0.206541    2.036254     0.842723   -0.351517      0.611583         0.476998            45.7
  0.20   0.15   0.80   False      0.502783        0.182746   -0.204117    2.019093     0.834690   -0.348268      0.602360         0.481275            46.3
  0.20   0.15   0.90   False      0.502783        0.182746   -0.204117    2.019093     0.834690   -0.348268      0.602360         0.478122            46.3
  0.20   0.15   0.95   False      0.502783        0.182746   -0.204117    2.019093     0.834690   -0.348268      0.602360         0.469902            46.3
  0.25   0.10   0.70   False      0.411638        0.172521   -0.203010    2.222193     0.728475   -0.328367      0.565068         0.412827            48.9
  0.25   0.10   0.90   False      0.402885        0.159757   -0.200235    2.208846     0.720555   -0.325197      0.559131         0.405355            50.0
  0.25   0.10   0.80   False      0.402885        0.159757   -0.200235    2.208846     0.720555   -0.325197      0.559131         0.407105            50.0
  0.25   0.10   0.95   False      0.402885        0.159757   -0.200235    2.208846     0.720555   -0.325197      0.559131         0.405972            50.0
  0.25   0.15   0.70   False      0.406536        0.172521   -0.202362    1.776142     0.716638   -0.326866      0.567282         0.409355            50.9
  0.25   0.15   0.95   False      0.397649        0.159757   -0.199551    1.762430     0.708706   -0.323670      0.561092         0.402038            51.0
```

## v7 baseline

```
 h_mid  h_min  h_max  stable  ew_mean_full  ew_median_full  ew_mean_h1  ew_mean_h2  ew_mean_vol  ew_mean_dd  ret_over_vol  plateau_ew_mean  composite_rank
  0.55    0.3    0.8   False      0.058536        0.048605   -0.153603    0.353562     0.327206   -0.215975      0.178898         0.059384           136.2
```

## v8 candidate: h_mid=0.45 h_min=0.3 h_max=0.8

- full EW mean 13.53% vs v7 5.85%

- full EW median 8.07% vs v7 4.86%

- H1 -16.78% vs -15.36% | H2 60.32% vs 35.36%

- vol 39.66% vs 32.72% | dd -24.30% vs -21.60%

- plateau (neighbor mean) 13.60%

- pair-fold CV: mean OOS uplift +43.87%, win-rate 100%, modal pick (0.2, 0.1, 0.7)
