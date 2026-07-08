# Bucket 4 / 5 engine notes

Reference for the inverse-decay (Bucket 4) and volatility-ETP (Bucket 5) sizing
engine. This file holds the backtest rationale and the disabled-subsystem
explanations that used to live as long comment blocks in
`config/strategy_config.yml`. The config now keeps only one-line comments and
points here.

---

## Hedge ratio + cadence engine (`scripts/bucket4_hedge_cadence.py`)

For each B4 pair on a given day the engine emits two numbers, both closed-form
functions of the pair's own vol-shape signals (TR = trend ratio,
VCR = variance-contribution ratio):

- `h` — hedge ratio (underlying short per unit inverse-ETF beta exposure)
- `interval_days` — trading days until the next rebalance

```
v7 hedge (production, hedge_ratio_model: v7):
    h_raw = h_mid + k_vcr * (VCR - VCR_med)
    h     = clip(h_raw * tilt.h_mult + tilt.h_shift, h_min, h_max)
    h_ema = (1 - alpha) * prev_h + alpha * h     # optional smoothing

v9 cross-sectional tilt (k_z > 0):
    h_raw -= k_z * z_composite
    z_composite = 0.5 * (-z(r10d)) + 0.5 * (+z(sig5/sig63))   # ranked daily across B4

cadence (source: tr_vcr):
    denom    = 1 + k_tr*(TR - 1) + m_vcr*(VCR - VCR_med)
    interval = clip(round(base_days / denom * tilt.interval_mult), min, max)
```

The same functions back production (`generate_trade_plan` /
`bucket4_weekly_opt2`) and the backtest, so the two cannot diverge.

### Leg split (how gross becomes two shorts)

```
gross = pair_weight * sleeve_budget_usd
denom = 1 + h * beta_used
inv_short_usd = gross / denom                       # inverse-ETF short leg
und_short_usd = (gross - inv_short_usd) * partial_hedge_ratio   # underlying hedge leg
```

So `und/inv = h * beta_used` (before `partial_hedge_ratio`), and the operator
pair overrides (see below) recompute exactly these two legs.

---

## Backtest rationale for the live knobs

- **`h_mid: 0.45`** — "v8" recalibration (`scripts/bucket4_v8_clip_experiment.py`,
  213-combo grid, 45 pairs): performance is monotone in `h_mid` (`h_min`/`h_max`
  barely bind); 0.45 picked under a risk budget (vol/dd <= 1.25x the 0.55
  baseline). EW mean CAGR 13.5% vs 5.9%, median 8.1% vs 4.9%; vol 39.9% vs 32.7%.
  Unconstrained optimum (0.20) rejected as a regime bet (84% vol). The old 0.55
  was inherited from v6 and never optimized.
- **`hedge_ratio_model: v7`** — Phase-3/4/5 lab
  (`scripts/bucket4_phase345_backtest.py`, 45 pairs, 2025-10 -> 2026-06, 20bps
  slip): v7 VCR closed-form `h` beat fixed 0.75 by ~+10pp EW mean CAGR.
  h-hysteresis and a regime bump both HURT (not adopted); realized-20d beta was
  noise (+0.5pp mean, -0.1pp median; not adopted).
- **`k_z: 0.20`** — "v9" cross-sectional tilt ported from the v6 panel
  (`scripts/bucket4_v9_xsec_experiment.py`). v6 SIGN confirmed (rally -> hedge
  up, vol expansion -> hedge down). At `k_z=0.20`: EW mean 16.4% vs 13.5% (v8),
  median 11.0% vs 8.1%, better in BOTH halves, vol/DD unchanged; broad plateau
  `k_z` 0.10-0.60. 0 = off (pure v8).
- **`base_days: 12`, `k_tr: 2.25`, `m_vcr: 2.5`** — full theta-grid under v7
  hedge: (2.25, 2.5, 12) EW mean 5.9% vs 3.3% at `base_days=10`, same vol/DD
  (`b4_param_scorecard.csv` 2026-06-10). `m_vcr` is the mean-best A7 cadence
  sensitivity (39.4% CAGR @ 20bps).
- **`max_interval: 21`** — allow calm regimes to stretch toward 3-week spacing.
- **`vol_etp_weight_penalty: 0.333`** — flat sizing haircut on volatility-ETP
  pairs (VIX complex: UVIX/SVIX). Final weight `*= (1 - penalty)`, renormalized
  so other pairs absorb the freed budget. The built-in tail-risk penalty is
  relative (median-normalized) and the B4 book is full of ~100%-vol crypto
  names, so short-vol blowup risk on a -1x VIX product doesn't stand out enough;
  this is an explicit operator prior on top.

---

## Disabled B4 subsystems (kept in config, OFF)

These four blocks ship in `strategy_config.yml` with `enabled: false`. They are
documented here so the config stays terse. Code paths exist but are skipped
while disabled.

### `ratchet` (grow-only inverse leg + continuous trim) — **LIVE**
- **What:** floors the inverse-ETF short leg at
  `max(solved, currently_held, persisted_floor)` so the short side only grows;
  continuous trim gradually closes the creep gap when enabled, re-solving the
  underlying leg to preserve hedge ratio `h`.
- **Applied in:** `scripts/bucket4_weekly_opt2.py` (planning),
  `generate_trade_plan.py` (`_b4_*_ratchet_state` helpers),
  `scripts/b4_reconstruct_state.py` (held legs from accounting detail),
  `phase2b_resize.py` (execution guard / trim allowance).
- **State:** `data/b4_inverse_ratchet_state.json` (trim mode tracks current target,
  not high-water).
- **Artifacts:** `data/runs/<date>/b4_hedge_cadence/b4_ratchet_targets.csv`,
  `ratchet_released` / `ratchet_trim_lambda` / `ratchet_trim_usd` on
  `proposed_trades.csv`.
- **Execution:** `ratchet.execution.allow_inverse_cover: true` permits capped
  inverse ETF covers when `ratchet_released=True` on the plan row.

### `drawdown_governor` (sleeve drawdown brake)
- **What:** on worst-drawdown pairs, pause new inverse adds (`dd_freeze`) and
  bias the underlying hedge toward `h_max` (`dd_derisk` + `dd_hedge_bump`) when
  trailing drawdown breaches thresholds.
- **Status:** config block only — **not consumed by any Python code today**.
  This is a stub for a future feature; enabling it has no effect until the
  governor logic is wired into `generate_trade_plan`.
- **State (planned):** `data/b4_governor_state.json`.

### `pair_lifecycle` (demotion ladder)
- **What:** turns report-only flags from `scripts/bucket4_pair_monitor.py` into
  sizing actions: `normal -> half (0.5x) -> freeze (keep-open, no new size) ->
  exit (dropped, re-entry blocked for a cooldown)`. Escalation is immediate;
  recovery needs `recover_obs_days` consecutive clean runs to promote one level.
- **Applied in:** `scripts/bucket4_pair_lifecycle.py` (advances state), consumed
  by `generate_trade_plan.py` when enabled.
- **State:** `data/b4_pair_lifecycle_state.json`.
- **Why off:** intended to flip on after a week of report-only shadowing.

### `concentration` + `cluster_caps` (Diamond-Creek WS4/WS5)
- **What:** `concentration` keeps only the best `top_n_pairs` by
  `(net_edge - borrow) / underlying_vol` (held leftovers become keep-open rows so
  they roll off via cadence rather than force-close). `cluster_caps` caps the
  combined weight of a named cluster (e.g. crypto underlyings) and redistributes
  the excess to non-cluster pairs.
- **Applied in:** `scripts/bucket4_sizing.py`
  (`apply_concentration_to_b4`, `apply_cluster_caps_to_b4`).
- **Backtest basis:** Stage C — top-10/15 equal-weight beat the all-pairs book on
  CAGR and Calmar; Kelly-style tilting and looser books underperformed; cluster
  cap was neutral-to-cheap insurance in the test window.
- **Why off:** `concentration` waits for shadow review of the kept list;
  `cluster_caps` members are defined but only bind once `concentration` (or the
  cap path) is active.
