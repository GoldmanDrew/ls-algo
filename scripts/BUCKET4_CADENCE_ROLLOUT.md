# Bucket 4 cadence + dynamic hedge rollout plan

Branch: **`feat/bucket4-hedge-cadence-engine`** (do not merge until paper-live validation completes).

## What changed (this rollout)

| Knob | Old | New |
|------|-----|-----|
| `hedge_cadence_policy.base_days` | 4 | **10** |
| `hedge_cadence_policy.max_interval` | 10 | **21** |
| Operator script cadence | weekly implicit | **every 5 business days** (`operator_check_days: 5`) |
| Per-pair defer | none | **`bucket4_cadence_gate.py`** + `data/b4_cadence_state.json` |

B1/B2/flow are **unchanged**.

---

## Production architecture (end-to-end)

```mermaid
flowchart TD
  subgraph daily [Daily EOD]
    DS[daily_screener.py]
    GTP[generate_trade_plan.py]
    ACCT[ibkr_accounting.py]
  end
  subgraph every5d [Every 5 business days]
    RS[rebalance_strategy.py]
  end
  DS --> GTP
  GTP --> ACCT
  GTP --> RS
  RS --> EXEC[execute_trade_plan legs]
```

### 1. Daily screener + trade plan (`generate_trade_plan.py`)

When `bucket4_weekly_opt2.enabled: true`:

1. **Universe** — inverse β<0 names pass B4 gates (`min_net_edge`, `min_underlying_vol`, borrow caps).
2. **Tail-risk weights** — `compute_bucket4_weights` (v6 covariance penalty on pair proxies).
3. **Signals** — per underlying: **TR** (trend ratio), **VCR** (variance-contribution ratio), **VCR_med** (cross-sectional median). Built from underlying prices with **1-day shift** (no look-ahead).
4. **Hedge ratio `h`** (default model: **v6 Opt-2**, not v7):
   - `z_composite` = blend of 10d return rank + range expansion (+ optional regime mult)
   - `h_star = h_base - opt2_k * z_composite`  (default `h_base≈0.55`, `opt2_k=0.05`)
   - EMA smooth: `h = (1-α)*h_prev + α*h_star` (`α=0.25` in panel)
   - Clipped to `[h_min, h_max]` when using v7 path; v6 uses its own bounds
5. **Cadence `interval_days`** (TR/VCR engine):
   - `denom = 1 + k_tr*(TR-1) + m_vcr*(VCR-VCR_med)`
   - `interval = round(base_days / denom)` clipped to `[1, 21]` with **`base_days=10`**
   - Trending (TR>1) → **faster** rebalance; choppy → **slower**
6. **Targets** — `compute_bucket4_targets`:
   - Inverse ETF short USD from weights × budget
   - Underlying short USD = `h × |β| × inverse_short × partial_hedge_ratio` (default **0.75**)
   - **Ratchet** (when enabled): inverse short never **shrinks**; delta cuts flow through underlying leg only
7. **Outputs** — `data/proposed_trades.csv` + `data/runs/<date>/b4_hedge_cadence/b4_hedge_cadence_explain.csv`

Sizing columns in plan: `b4_opt2_hedge_ratio`, `b4_opt2_inverse_etf_short_usd`, underlying `long_usd` (negative = short).

### 2. Rebalance every 5 business days (`rebalance_strategy.py`)

You run the **same** rebalance script on a **5-business-day** calendar (Mon/Wed/Fri or cron). Inside each run:

#### Phase 1 — Cleanup
Close ETFs not in plan (unchanged).

#### Phase 2 — Establish
Open new pairs below `establish_threshold_usd` (unchanged).

#### Phase 2b — Resize (where B4 cadence lives)

**Cadence gate** (`scripts/bucket4_cadence_gate.py`):

For each B4 pair `(ETF, Underlying)`:

| Condition | Action |
|-----------|--------|
| No `last_rebalance` in state | **DUE** (first resize) |
| `trading_days_since(last) >= interval_days` | **DUE** |
| `trading_days_since(last) >= max_interval` (21) | **FORCE DUE** |
| Otherwise | **DEFER** — pair rows removed from resize plan |

Deferred pairs keep positions; plan targets update daily but **no trades** until due.

Telemetry: `data/runs/<date>/rebalance/b4_cadence_decisions.csv`

**Hedge ratio hysteresis** — two layers:

1. **Target level** — `h` moves gradually via EMA in GTP (not re-traded every day).
2. **Execution band** — Phase 2b only trades when leg drift exceeds **12% enter / 4% exit** (`portfolio.rebalance.resize`). Small hedge drift → **no trade** even on a due day.

**Ratchet** — inverse ETF short leg: **BUY-to-cover blocked** in resize; only underlying leg adjusts down.

After successful fills, `data/b4_cadence_state.json` updates `last_rebalance` for traded pairs.

#### Phase 3 — Hedge (B1/B2 only)
Directional net-exposure correction on `core_leveraged` + `yieldboost`. **B4 is not Phase-3 hedged** — its hedge is structural (inverse ETF + partial underlying short).

---

## Hedge ratio: how it is calculated and used

| Layer | What | Where |
|-------|------|-------|
| Dynamic `h` | v6 Opt-2 cross-section score (default) | `bucket4_weekly_opt2.build_hedge_panel_opt2` |
| Scale | `partial_hedge_ratio = 0.75` | `strategy_config.yml` + `compute_bucket4_targets` |
| Structural short | `und_short = h × β × inv_short × 0.75` | `proposed_trades.csv` `long_usd` |
| Accounting | Implied B4 underlying exposure | `ibkr_accounting` `etf_implied` mode |
| Execution deadband | 12% / 4% resize bands | `phase2b_resize.py` |
| Inverse protection | Grow-only ratchet | GTP + `phase2b_resize` |

**Optional v7 hedge** (`hedge_ratio_model: v7`): `h = clip(h_mid + k_vcr*(VCR-VCR_med), 0.30, 0.80)` — only switch after v6 paper-live baseline.

---

## Rollout checklist (staged, on branch)

### Stage 0 — Config only (current commit)
- [x] `base_days: 10`, `max_interval: 21`
- [x] Cadence gate code + tests
- [ ] `bucket4_weekly_opt2.enabled: false` — **leave off** until Stage 1

### Stage 1 — Shadow mode (1–2 weeks)
1. Set `bucket4_weekly_opt2.enabled: true` but **do not** enable ratchet yet.
2. Run daily: `daily_screener` → `generate_trade_plan` → inspect `b4_hedge_cadence_explain.csv`.
3. Run `python -m scripts.bucket4_hedge_cadence --run-date <today> --plots` every 5bd alongside rebalance (read-only).
4. Compare proposed B4 targets vs current holdings; **no B4 resize** (`--skip-phase-2b` or gate with empty state + manual review).

### Stage 2 — Cadence gate live, small resize
1. Enable gate (automatic when `enabled: true` + `source: tr_vcr`).
2. Run full `rebalance_strategy.py` every **5 business days**.
3. Verify `b4_cadence_decisions.csv`: mix of DUE/DEFER; deferred pairs untouched.
4. Enable `ratchet.enabled: true` once inverse inventory is stable.

### Stage 3 — Full production
1. Remove `--skip-phase-2b` shortcuts.
2. Monitor slippage + borrow via accounting runs.
3. Optional: `drawdown_governor` for tail pairs.
4. Open PR to `main` after 4+ successful 5-day cycles.

---

## Optimization plan — Phases 1 & 2 (this branch)

### Phase 1 — Measure (report-only, always on)
* `scripts/bucket4_pair_monitor.py` — per-pair trailing 20/60bd realized PnL,
  annualized return on pair gross, borrow paid vs gross decay captured, and
  `edge_capture_ratio` vs the screener's `bucket4_net_edge_annual`. Emits the
  demotion-ladder flags (half/freeze/exit/vol-floor) WITHOUT acting on them.
  Writes `data/runs/<date>/b4_monitor/b4_pair_monitor.csv` and appends one
  summary line per run to `data/b4_observations.jsonl` (knobs + EW results —
  the raw material for the nudge loop).
* `scripts/bucket4_param_scorecard.py` — replays the real pair backtest over a
  theta-grid around the current `(k_tr, m_vcr, base_days)` and ranks every theta
  on the same fixed metrics (winsorized mean CAGR, median CAGR, vol, max DD).
  Current knobs are marked `is_current`; the script prints HOLD or a one-knob
  nudge suggestion and appends to `data/b4_param_scorecard_history.jsonl`.
  Use `--quick` weekly (7 thetas), full 3x3x3 grid monthly.

### Phase 2 — Cut losers (gated by `pair_lifecycle.enabled`)
* Demotion ladder `normal -> half -> freeze -> exit` in
  `scripts/bucket4_pair_lifecycle.py`; state in `data/b4_pair_lifecycle_state.json`.
  - **half**: edge capture < 0.25 and realized < +10% ann -> 0.5x decay-score weight.
  - **freeze**: trailing 60bd < -15% ann (or vol < keep floor) -> weight 0,
    `purgatory=True` (keep-open, no auto-close).
  - **exit**: trailing 60bd < -30% ann OR material borrow>decay (>= 5% ann drag
    on gross) -> row dropped from plan; Phase 1 cleanup closes it; 45bd re-entry
    cooldown.
  - Escalation is immediate; recovery promotes ONE level per 10 consecutive
    clean monitor runs.
* Underlying-vol floors split into entry/keep: new pairs need >= 0.5 annual vol,
  held pairs (tracked in lifecycle state) may stay down to 0.4 — index-like
  pairs roll off instead of being churned.
* `generate_trade_plan.py` applies the ladder inside the B4 core slice after
  decay-score weights, renormalizing so the budget shifts to surviving pairs.

Rollout: run monitor + lifecycle `--dry-run` daily for ~1 week alongside Stage 1
shadow mode; sanity-check the flagged names; then set `pair_lifecycle.enabled: true`.

---

## Optimization plan — Phases 3-5 (backtest-driven; this branch)

All decisions below come from `scripts/bucket4_phase345_backtest.py` (45 pairs,
2025-10-07 → 2026-06, 20 bps slippage, production TR/VCR cadence). Full tables:
`notebooks/output/b4_phase345_stage{A,B,C}.csv` + `b4_phase345_summary.md`.

### Adopted (config/live)
* **v7 VCR closed-form hedge ratio** (`hedge_ratio_model: v7`): EW mean CAGR
  **+3.8%** vs **-6.6%** for fixed h=0.75 — a ~10 pp improvement, and positive
  median. The biggest single lever found.
* **"v8" clip recalibration — `h_mid` 0.55 → 0.45** (`scripts/bucket4_v8_clip_experiment.py`,
  213-combo grid over h_mid × h_min × h_max, 45 pairs): the old 0.55/0.30/0.80
  were inherited from the v6 calibration and never jointly optimized. Findings:
  (a) `h_min`/`h_max` barely bind — performance is a monotone function of
  `h_mid` alone (lower hedge = more net-short = more return AND more risk);
  (b) the unconstrained optimum (h_mid 0.20: 51.5% EW mean) was REJECTED as a
  regime bet — 84% vol, -35% DD, all of the gain from the Feb-Jun half;
  (c) under a pre-declared risk budget (vol/DD ≤ 1.25× baseline), every top
  combo is h_mid 0.45, so v8 = (0.45, 0.30, 0.80) by minimal change.
  EW mean 13.5% vs 5.9%, median 8.1% vs 4.9%, vol 39.9% vs 32.7%, DD -24.4%
  vs -21.6%, ret/vol 0.34 vs 0.18. Independently corroborated by the h* lab
  (exp-2 refit also landed on h_mid 0.45). Caveat: slightly worse in the
  Oct-Jan half (-16.8% vs -15.4% EW mean); 20-fold pair CV win-rate 100%.
  Full artifacts: `notebooks/output/b4_v8/`.
* **"v9" cross-sectional tilt — `k_z: 0.20`** (`scripts/bucket4_v9_xsec_experiment.py`):
  ports the v6 Opt-2 panel's signal core into the v8 closed form:
  `h_raw -= k_z * z_composite`, `z_composite = 0.5*(-z(r10d)) + 0.5*(+z(sig5/sig63))`
  ranked daily across B4 underlyings (robust median/MAD z, shifted 1 day,
  dedup by underlying). The v6 regime overlay was NOT ported (rejected in the
  phase-3 lab). Findings: the v6 SIGN convention wins (rally -> hedge up, vol
  expansion -> hedge down — the h* lab's opposite-sign momentum idea loses);
  composite beats momentum-only and range-only on balance; broad plateau over
  k_z 0.10-0.60 with interior optimum ~0.20. At k_z=0.20 vs v8: EW mean 16.4%
  vs 13.5%, median 11.0% vs 8.1%, better in BOTH halves, vol/DD ~unchanged
  (39.5%/-24.2% vs 39.7%/-24.3%), ret/vol 0.41 vs 0.34. Honesty caveats: the
  fixed pick wins on only 25/45 pairs (gains concentrated: ASTN +81pp, QBTZ
  +31pp vs NBIZ -51pp, DAMD -42pp); 20-fold pair CV median uplift +0.8pp,
  win-rate 65%, mean ~0 (dragged by folds picking outlier combos). All five
  pre-declared gates passed -> adopted, reversible with `k_z: 0` (pure v8).
  Engine support in `bucket4_hedge_cadence.py` (`build_xsec_z_panel`,
  `xsec_z` arg); GTP path injects `sig["xsec_z"]` only when `k_z != 0`.
  Full artifacts: `notebooks/output/b4_v9/`.
* **`base_days` 10 → 12** (one-knob nudge): the full 3x3x3 theta grid re-run
  UNDER the v7 hedge (`b4_param_scorecard.csv`) ranks (k_tr 2.25, m_vcr 2.5,
  base_days 12) first — EW mean 5.9% / median 4.9% vs 3.3%/3.7% at 10, with the
  same vol and DD. base_days 8 is negative across the entire grid.
* **WS4 concentration** (`concentration.top_n_pairs: 15`, gated off until the
  kept-list is reviewed): Stage C top-15 equal-weight portfolio: **29.6% CAGR,
  Calmar 2.08** vs materially worse for the all-45 book. Top-10 was slightly
  higher CAGR (37%) but more concentrated; 15 is the risk-aware pick. Held pairs
  that fall out of the top-N become keep-open (no force-close churn).
* **WS5 crypto cluster cap** (`cluster_caps.crypto: 0.35`): neutral in the test
  window (the cap rarely bound) — kept as cheap insurance against the
  MSTR/COIN/IBIT/CLSK/IREN/BMNR/CRCL monoculture.
* **`max_name_weight` 0.15 → 0.20** in lockstep with concentration.
* **WS5 Phase-3 visibility**: `[B4-DELTA]` read-only line in the Phase 3 header
  shows B4's residual delta contribution to the book (Phase 3 still never
  trades B4).

### Tested and REJECTED (negative results, documented to prevent re-litigation)
* **Hybrid drift+time cadence (WS3)**: every drift-trigger variant (4-12% leg-share
  drift, 10/21d clock floor) *cut vol nearly in half* (33% → 13-15%) but destroyed
  the return (best drift variant -9.5% vs +3.8% EW mean for the production clock).
  The TR/VCR clock stays. Drift support remains in the engine
  (`drift_threshold_share_of_gross`, `force_rebalance_after_days`) for re-testing
  in a different regime.
* **h hysteresis 0.05**: -1.4 pp EW mean CAGR. Not adopted.
* **Regime bump (+0.10 toward h_max when rv pctile > 0.8)**: -2.4 pp in this
  (calm) window. The book-level overlay idea stays parked until there is a
  crash window to test it on.
* **Realized 20d beta**: +0.5 pp mean, -0.1 pp median — noise. Static screener
  |Δ| stays.
* **Kelly-lite weights (WS4)**: λ=0 (pure equal weight over the selected top-N)
  beat λ=0.5 and λ=1.0 at every book size. Selection does the work; trailing
  per-pair momentum tilts only added vol.

---

## Operator commands

```powershell
# Daily (unchanged)
python daily_screener.py
python generate_trade_plan.py
python ibkr_accounting.py

# Every 5 business days
python rebalance_strategy.py --run-date 2026-06-03

# Inspect cadence (human-readable)
python -m scripts.bucket4_hedge_cadence --run-date 2026-06-03 --plots

# Phase 1: monitor + lifecycle shadow (daily, after ibkr_accounting.py)
python -m scripts.bucket4_pair_monitor
python -m scripts.bucket4_pair_lifecycle --dry-run     # drop --dry-run once enabled

# Phase 1: parameter scorecard (weekly quick, monthly full grid)
python -m scripts.bucket4_param_scorecard --quick

# Tests
python -m pytest tests/test_bucket4_hedge_cadence.py tests/test_bucket4_cadence_gate.py tests/test_bucket4_pair_lifecycle.py -q
```

---

## FAQ

**Why run every 5 days if intervals are 1–21?**  
The script is the *polling* interval. Each pair has its own `interval_days`; most runs will trade only the subset that is due. Five days balances ops burden vs catching trending pairs (interval can drop to 1–3d).

**Do we hedge only outside a hysteresis band?**  
Yes — at execution. Plan targets move daily; trades fire only when resize bands breach (12%/4%) **and** cadence says due. `h` itself is smoothed by EMA so targets do not jump daily.

**What if TR/VCR signals are missing?**  
Neutral policy: `h ≈ h_mid (0.45, v8 recalibration)`, `interval ≈ base_days (12)`.

**Merge policy**  
Stay on `feat/bucket4-hedge-cadence-engine` until Stage 3 sign-off. Do not merge to `main` early.
