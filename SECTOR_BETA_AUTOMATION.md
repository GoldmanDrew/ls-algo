# Auto Sector + Live Beta Attribution

A spec for replacing the hand-curated `risk_dashboard/factor_map.py`
SPY-beta and sector map with a tiered sector resolver + live OLS
betas to SPY/QQQ/IWM with shrinkage to a sector-mean prior.

## 1. Current state (as of 2026-05-20)

- `risk_dashboard/factor_map.py` is fully hand-curated (~150 sectors,
  ~140 SPY betas).
- `risk_dashboard/beta_loader.py` exists and runs every build (60-day
  OLS, yfinance, on-disk cache, shrinkage toward 1.0).
- Latest snapshot provenance: 165 `curated_fallback`, 9
  `default_fallback`, **0 `computed`** — yfinance is failing in CI and
  the cache directory is not committed.
- 9 names fall to the generic 1.2 prior: `LCID, TER, SOEZ, QS, URA,
  ZETA, DUOL, COPX, SOUN`.

## 2. Goals and success criteria

| # | Goal | Pass criterion |
|---|---|---|
| G1 | Every book underlying has a sector with provenance | `sector_source` populated for ≥99% of gross |
| G2 | Every book underlying has β to SPY/QQQ/IWM with provenance + SE | `beta_source = computed` for ≥90% of gross |
| G3 | Build is deterministic and offline-resilient | CI succeeds with no network; never below ≥80% computed when cache is warm |
| G4 | UI shows confidence honestly | Per-row pill: `computed` / `shrunk` / `fallback` |
| G5 | Estimates are stable for sizing | 30d rolling β drift < 0.4 for ≥75% of names; SE reported |

## 3. Architecture

### 3.1 Sector attribution — tiered (first hit wins)

| Tier | Source | Use for |
|------|--------|---------|
| 1. Override | `OVERRIDE_SECTOR_MAP` in `factor_map.py` | Thematic buckets GICS can't express (quantum, crypto-equity, nuclear, evtol, drones, space, insurtech) |
| 2. Screener | `data/etf_screened_today.csv` underlying theme/bucket | Reuse pipeline metadata |
| 3. Vendor | `yfinance.Ticker(sym).info["sector"]` mapped to our taxonomy | GICS-style sectors |
| 4. Heuristic | Keyword regex on `longName` / `industry` (`nuclear`, `uranium`, `crypto`, `miner`, `quantum`, `drone`, `eVTOL`, `satellite`, `lithium`, `rare earth`) | New thematics not yet in override map |
| 5. Default | `"other"` | Last resort, flagged in UI |

Output: `{sector, sector_source, sector_confidence}` per underlying.

### 3.2 Live beta math

Daily adjusted log returns over a **252 trading-day window**. Three
independent OLS regressions per underlying:

```
r_i = α + β_SPY · r_SPY + ε     (window 252d)
r_i = α + β_QQQ · r_QQQ + ε
r_i = α + β_IWM · r_IWM + ε
```

Each returns `(β, β_se, n_obs, R²)`.

**Shrinkage** (sector-mean prior, AR(1) effective sample size):

```
β_final = w · β_OLS + (1 − w) · β_prior
w       = n_eff / (n_eff + k)
n_eff   = n · (1 − ρ_AR1) / (1 + ρ_AR1)
k       = K_BASE · max(1, β_prior²)        # K_BASE = 60
β_prior = sector_mean_β if sector has ≥5 computed names
         else 1.2 (single-name) / 1.0 (broad)
```

Sign guard: if `sign(β_OLS) ≠ sign(β_prior)` and `|β_OLS| > 0.3` and
prior is from an override, hard-snap to prior, mark
`imputed_sign_mismatch`.

Regime σ persisted from the trailing 60d separately.

### 3.3 Data sources — fail-over chain

| Order | Source |
|---|---|
| 1 | yfinance batch download (`chunk_size=25`) |
| 2 | Stooq CSV (`https://stooq.com/q/d/l/?s={sym}.us&i=d`) |
| 3 | On-disk cache (≤14 days stale) |
| 4 | Skip — mark `no_data`, fall to curated/default |

Circuit breaker: if both yfinance and stooq return 0 for a chunk,
abort further network calls this run.

### 3.4 Caching & CI

- Per-symbol closes at `data/cache/beta_history/<SYM>.csv` (already
  exists).
- `data/cache/beta_summary.json` with snapshot date + all
  `BetaResult.to_dict()`.
- CI commits both back to the repo after a successful build (separate
  `[skip ci]` commit so it doesn't loop).

## 4. File changes

```
NEW   risk_dashboard/sector_loader.py
EDIT  risk_dashboard/beta_loader.py
EDIT  risk_dashboard/factor_map.py       (SECTOR_MAP → OVERRIDE_SECTOR_MAP)
EDIT  risk_dashboard/metrics.py
EDIT  site/assets/js/app.js
EDIT  .github/workflows/risk_dashboard.yml
NEW   risk_dashboard/tests/test_sector_loader.py
NEW   risk_dashboard/tests/test_beta_loader_live.py
EDIT  risk_dashboard/tests/test_metrics.py
EDIT  risk_dashboard/README.md
```

## 5. Tests

- Unit: each sector tier; heuristic regex set.
- Unit: shrinkage matches the AR(1) shape from
  `daily_screener.compute_beta_shrunk` on a shared fixture.
- Unit: stooq adapter parses a recorded CSV.
- Integration: full `build_snapshot` against a fixture run-date with
  mocked fetch; assert ≥90% computed.
- Determinism: same fixture inputs → same `beta_summary.json` bytes.

## 6. Rollout

1. Ship behind existing `enable_computed_betas` flag (default True).
2. Shadow run for 5 trading days: emit both curated and computed,
   log diff to `data/runs/<date>/beta_diff.csv`.
3. Gate switch once `mean_abs_diff(β) < 0.3` and no sign mismatches.
4. Delete `BETA_TO_SPY` from `factor_map.py` after shadow confirms.
5. Update `risk_dashboard/README.md`.
