# Prompt: Institutional-grade VIX → expected decay model

Replace the current additive diff-OLS vol→VIX β estimator with the model a multi-strategy book PM (Millennium-style) would actually use to forecast 12M expected decay under VIX shocks. The goal is to make the **12M expected decay vs VIX** table responsive to realistic VIX moves while keeping the math defensible.

---

## Why the current model is too quiet

Current pipeline (`risk_dashboard/vol_vix_beta.py` + `_build_vix_decay_matrix` in `metrics.py`):

1. 20d rolling realized vol per name.
2. **Diff-OLS** of Δσ on ΔVIX over 252 days.
3. AR(1)-shrunk to sector-mean prior (which is itself a mean of low diff-OLS betas).
4. **Additive** σ mapping: `σ_new = σ_base + β × (ΔVIX/100)`.
5. Floor at β = 0; clip at β = 2.

Empirically on 2026-05-19:

| Stat | Value |
|------|------:|
| Names with computed β | 172/173 |
| Median β | **0.08** |
| Mean β | 0.13 |
| Names with β = 0 | **49** |
| +30 VIX → median σ change | **+2.6 vol pts** |
| +30 VIX → 12M total carry | 21.5% → **24.2%** (+2.7 pp) |

Structural reasons it underprices vol sensitivity:

- Diffing 20d realized vol is dominated by sampling noise (R² ≈ 0.001 on AAL, etc.); OLS slope collapses toward 0.
- Sector priors are circular (means of the same low estimates).
- Hard floor at 0 nukes 49 names entirely.
- Additive σ is linear in VIX pts; loses convexity for big moves.
- σ is treated as a single terminal point; LETF decay is actually a path integral of σ²(t).
- Borrow is held constant — no margin/funding tightening in a vol spike.
- Correlations are static — no cross-sectional correlation lift.

---

## Target model (institutional standard)

A PM running a leveraged ETF / vol-sensitive book would build this as **three independent layers** that can be calibrated and shocked separately. The dashboard should expose the levers, not bury them.

### Layer 1 — Single-name **vol elasticity** (replaces diff-OLS)

Log–log regression of EWMA realized vol on VIX level (or σ_SPX):

```
log σ_i,t  =  α_i  +  β_i · log VIX_t  +  ε_i,t          (1)
```

with EWMA-weighted least squares, λ = 0.94 (RiskMetrics), 504-day window.

**Why:** captures the multiplicative nature of vol (a quiet → stressed regime is a *ratio* change, not an additive points change). Coefficient β_i is a **vol-of-vol elasticity** in [0, ~2]; equities cluster 0.6–1.2, single-stock vol ETPs 1.5–2.0, broad indices 0.9–1.0, gold/treasuries 0.2–0.5.

Shock mapping becomes multiplicative:

```
σ_i,new  =  σ_i,base  ·  (VIX_new / VIX_now)^β_i          (2)
```

This is naturally convex: +30 VIX from 17.5 → 47.5 is a 2.7× ratio, so β = 1 → σ scales 2.7×; β = 0.5 → σ scales 1.65×. No additive coefficient required.

### Layer 2 — **Variance decomposition** (market + idio)

Drop the assumption that every name regresses individually on VIX. Decompose:

```
σ_i²  =  β_i,SPX²  ·  σ_SPX²   +   σ_i,idio²              (3)
```

with σ_SPX driven by VIX (≈ VRP-adjusted, see Layer 3) and σ_i,idio modeled by EWMA on the residual of the SPX-beta regression already used by the factor panel.

Shock then becomes:

```
σ_SPX,new       =  f_VRP(VIX_new)
σ_i,idio,new    =  σ_i,idio,now · (corr_lift_factor)
σ_i,new         =  sqrt(β_i,SPX² · σ_SPX,new² + σ_i,idio,new²)
```

**Why:** reuses the same SPY OLS β you already trust (`beta_loader`), and respects the “in a crash everything correlates” effect by lifting σ_idio proportionally (correlation lift α = 1.2–1.5 in regime ≥ 25 VIX).

### Layer 3 — **VIX → realized σ_SPX** mapping (VRP correction)

VIX is the *risk-neutral* 30d expected vol of SPX; realized SPX vol is typically 70–85% of VIX (the variance risk premium):

```
σ_SPX,t  ≈  k · VIX_t / 100         (4)
```

Calibrate `k` from rolling regression (k ≈ 0.80 historical mean, regime-dependent).

For path-dependent decay, also need **VIX mean-reversion**:

```
dVIX_t  =  κ (θ - VIX_t) dt  +  ν · sqrt(VIX_t) dW_t       (5)
```

with κ ≈ 5 (half-life ~50d), θ ≈ long-run mean ≈ 18–20.

### Layer 4 — **Path-integrated decay**, not terminal σ

LETF decay over horizon T is:

```
decay_T  =  ½ · L(L-1) · ∫₀ᵀ σ²(t) dt                     (6)
```

Under VIX shock, model σ(t) reverting from σ_shock back to σ_base via Layer 3 mean-reversion. The realized integral is what actually shows up in 12M carry, not σ_terminal².

Closed-form for OU mean reversion of variance:

```
∫₀ᵀ σ²(t) dt  =  σ_∞² · T  +  (σ_0² - σ_∞²) · (1 - e^(-κT)) / κ
```

For T = 1y, κ = 5: ~80% of integral comes from σ_∞², ~20% from σ_0² half-life decay. This single change typically **doubles** the response to a one-shot VIX spike vs the current “σ stays elevated all year” assumption — wait, actually the opposite: it *reduces* terminal-σ response because the spike decays. Net effect depends on whether you’re shocking VIX-now or VIX-path.

**Recommended convention:** the table should support **two columns of intuition**:

| Scenario type | Interpretation |
|---------------|----------------|
| **Sustained VIX** | VIX shifts to new level and *stays* there for 12M (current convention) |
| **Spike & revert** | VIX jumps now, mean-reverts to long-run θ over horizon |

PMs need both: regime change vs single event.

### Layer 5 — **Correlated leg moves & borrow stress**

Two adjustments PMs always insist on:

1. **Borrow widening with VIX:** in stress, hard-to-borrow rates spike. Add a simple multiplicative bump: `borrow_rate(VIX) = borrow_base × (1 + γ × max(0, VIX - 20)/10)` with γ ≈ 0.3 for HTB names, 0.05 broad.
2. **Distribution/yield-boost roll cost:** option-overlay ETFs (XYLD, JEPI, single-stock yield-boost) lose roll yield in spikes; encode an `iv_premium_shock` per product class (already partially in `scenario_engine`; just needs VIX wiring).

---

## Concrete implementation plan (phased)

### Phase 1 — Replace estimator: vol elasticity (Layer 1)

**Files:**
- New `risk_dashboard/vol_vix_beta_v3.py` (don’t modify v2 in place; keep both behind a config flag)
- `config/strategy_config.yml`: add `vol_vix_beta.estimator: v3_log_elasticity`

**Logic:**
1. EWMA realized vol per name (`λ = 0.94`, decay weights), 504d history, min 252d.
2. Log–log OLS: `log σ = α + β · log VIX`.
3. Weighted by EWMA decay → recent regime weighted more.
4. Shrink toward **product-class anchors** (data-anchored, not circular sector means):

| Product class | Prior β |
|---------------|--------:|
| `volatility_etp` | 1.50 |
| `single_stock_yield_boost` | 1.20 |
| `broad` (SPY/QQQ/IWM clones) | 1.00 |
| `semis / tech / growth` | 1.10 |
| `defensives / staples / utilities` | 0.60 |
| `gold / commodity` | 0.40 |
| `bonds` | 0.30 |
| `crypto-equity proxies` | 1.30 |

5. Shrinkage: `β = w · β_OLS + (1−w) · β_prior` with `w = n_eff / (n_eff + 30)` (lighter k vs v2; OLS on smoother EWMA is more reliable).
6. **Remove zero floor**; clip to [0.1, 2.5]. Negative β snaps to prior, not zero.

**Sigma shock:** equation (2) — multiplicative.

**Tests:**
- `test_vol_elasticity_recovers_known_beta` — synthetic log VIX → log σ.
- `test_priors_anchor_low_data_names`.
- `test_multiplicative_shock_convexity` — +30 VIX from VIX=15 produces larger σ change than from VIX=25.

### Phase 2 — Path-integrated decay (Layer 4)

**Files:** `risk_dashboard/scenario_engine.py`, `metrics.py::_build_vix_decay_matrix`.

**Logic:**
1. Add `σ_path_integral(σ_0, σ_∞, κ, T)` helper using OU closed-form.
2. Two scenario modes in `_build_vix_decay_matrix`:
   - `mode = "sustained"`: σ_0 = σ_∞ = σ_shock (current behavior, σ² T).
   - `mode = "spike_revert"`: σ_0 = σ_shock, σ_∞ = σ_base, κ = `vix_mean_reversion_kappa` (config, default 5.0 → 50d half-life).
3. UI: two side-by-side matrices (or a `mode` toggle).

**Acceptance:**
- Sustained mode reproduces existing numbers when β = current values (sanity check).
- Spike-revert mode shows materially smaller terminal impact (~30–50% of sustained for κ=5, T=1).

### Phase 3 — Variance decomposition + correlation lift (Layer 2)

**Files:** new `risk_dashboard/variance_decomp.py`.

**Logic:**
1. Compute σ_i,idio from existing SPY β + EWMA residual vol (cache in `beta_summary.json`).
2. On VIX shock, compute σ_SPX,new from Layer 3 mapping, scale σ_i,idio by `corr_lift(VIX_new)`:

```
corr_lift(VIX) = 1.0                    if VIX ≤ 18
                = 1.0 + 0.05·(VIX-18)    if 18 < VIX ≤ 35
                = 1.85                   if VIX > 35   (cap)
```

3. New σ_i,new from equation (3).
4. Use σ_i,new in path integral (Phase 2) instead of equation (2).

Phase 1 result becomes a **fallback** for names without trusted SPY β; Layer 2 is preferred when SPY β is `computed` or `shrunk`.

### Phase 4 — VIX term-structure & regime conditioning

**Inputs:** add `^VIX9D`, `^VIX3M`, `^VVIX` to the yfinance fetcher.

**Outputs in matrix:**
- Contango/backwardation flag (VIX9D/VIX vs VIX3M/VIX) — affects roll-cost assumption.
- Vol-of-vol shock column for derivative books.
- **Regime split:** estimate β separately on `VIX ≤ 20` and `VIX > 20` subsamples; use the appropriate β based on `VIX_new`.

### Phase 5 — Stress scenarios beyond parallel shifts

Replace the single +/- pts row with named historical analogs:

| Scenario | VIX path | Corr lift | Borrow lift |
|----------|----------|----------:|------------:|
| **Aug 2015 China** | 13 → 28 → 18 over 30d | 1.4 | 1.2× |
| **Feb 2018 XIV** | 12 → 37 → 19 over 14d | 1.6 | 1.3× |
| **Mar 2020 COVID** | 14 → 82 → 35 over 45d | 1.9 | 2.0× |
| **Sep 2022 inflation** | 22 → 33 → 25 over 60d | 1.3 | 1.1× |
| **Aug 2024 yen carry** | 16 → 65 → 22 over 7d | 1.7 | 1.5× |

Each scenario runs through Phase 2 path engine → reports 12M total decay impact.

This is the column a PM will actually look at.

---

## UI changes

### New panel: **VIX-scenario expected decay (12M)**

Sections, in order:

1. **Headline strip** (current snapshot)
   - σ_book (EWMA-median), VIX-now, VIX9D/3M, VVIX, expected 12M carry at status quo.

2. **Mode toggle:** Sustained / Spike & revert (Phase 2).

3. **Matrix:** rows = scenarios (current + parallel shifts + named historical), columns = book total decay, borrow drag, total carry %NAV, top-3 contributor names.

4. **Per-name table** (collapsible): symbol, β_vol (Phase 1 elasticity), σ_base, σ_shocked at +20 VIX, decay $ contribution.

5. **Diagnostics callout:** estimator version, EWMA λ, sample size summary, prior provenance counts. Current dashboard already has this — extend with new fields.

---

## Calibration & validation

### Reference points (PMs will sanity-check)

- **SPY/QQQ** β_vol should land 0.9–1.1 (well-known empirical fact).
- **TLT** β_vol around 0.3 (rates-driven, weakly correlated to VIX).
- **GLD** β_vol around 0.4–0.5.
- **MSTR/COIN** β_vol around 1.3–1.6.
- **Vol ETPs (SVIX/SVXY/UVXY)** β_vol around 1.5–2.0.

Add a `tests/test_vol_vix_v3_anchors.py` regression test that runs the estimator on these tickers (cached fixtures, deterministic) and asserts β within tolerance.

### Backtests

For each historical analog in Phase 5:
1. Reconstruct book state at T-30d.
2. Run model with VIX trajectory.
3. Compare predicted 30/60/90d decay to actual decay from `pnl_history.csv` (filtered to vol-driver legs).
4. Report MAPE per scenario; target < 25%.

---

## Out of scope

- Full Heston/SABR calibration per name (too heavy for daily snapshot).
- Volatility surface modelling (skew, term structure per name).
- Implied vol from options chains (would require IBKR option-chain pull; nice-to-have for later).
- Replacing the equity-price β panel.

---

## Acceptance criteria

1. New estimator produces median β_vol in [0.7, 1.1] for current book (vs 0.08 today).
2. +30 VIX shock under **sustained** mode moves median σ by ≥ +20 vol points (vs +2.6 today).
3. +30 VIX shock under **spike-revert** mode produces a clearly smaller (but non-trivial) effect, demonstrating the path integral works.
4. Per-name table shows SPY ≈ 1.0, TLT < 0.5, vol ETPs > 1.3.
5. All Phase 5 historical scenarios produce a numeric decay forecast and a deviation vs actual.
6. Old v2 estimator remains callable behind `estimator: v2_diff_ols` config switch for A/B.
7. All new tests pass; snapshot rebuild succeeds; existing factor panel and slide-risk strips unchanged.

---

## Phase 1 only — minimum viable upgrade

If you want a single first pass that delivers most of the value without all five phases:

- Implement **Phase 1 only** (log–log elasticity, EWMA σ, product-class priors, no zero floor, multiplicative σ shock).
- Keep terminal-σ decay (no path integral yet).
- Add new estimator alongside v2, default v2 still, expose via config flag.
- Rebuild snapshot and have the user compare side-by-side before flipping default.

This is the smallest credible step. Expected outcome: +30 VIX moves 12M total carry from 21.5% to ~30–35% (instead of 24.2%), which is closer to what historical realized vol regimes would imply.

---

## Reference files

- `risk_dashboard/vol_vix_beta.py` — current v2 estimator (keep)
- `risk_dashboard/scenario_engine.py` — leg-level LETF decay model
- `risk_dashboard/metrics.py` — `_build_vix_decay_matrix`, `_slide_horizon_scenario_totals`
- `risk_dashboard/beta_loader.py` — SPY β (reused in Layer 2)
- `daily_screener.py` — `BETA_SHRINK_K_BASE`, AR(1) n_eff helpers
- `site/index.html`, `site/assets/js/app.js` — slide-risk panel UI
