# B4 sizing: conditional crash exposure (tail risk × run-up) — design note

Date: 2026-07-10. Research direction only; no production changes proposed
without the validation gates at the end.

Motivating question: size B4 pairs as a combination of tail risk and how
far the underlying has already run up / drawn down. A name like AMD after
an enormous rally can plausibly be cut in half; its pair should be sized
much smaller than a name already sitting on a deep drawdown, even if their
historical vol looks similar.

## 1. What the drawdown_63 battery actually established

The 36-run battery (`notebooks/output/b4_ta_drawdown63_fixes/`) ruled out a
family of *mechanics*, not the run-up signal itself:

- Hard gross cap with daily end-of-day rescaling creates ~15× trade-cost
  churn and dominates every other effect (F03–F06 all worse than baseline).
- Daily retargeting is ruled out (F04/F06: −70% CAGR).
- Regime gating that turns the sleeve off most days is ruled out (F08/F10).
- **The sizing signal itself was likely never isolated.** F03 (pure cap) and
  F05/F11–F14 plus all 20 α×floor sweep combos produced byte-identical
  results. Two candidate causes, both mechanical:
  1. `apply_ta` multiplies `pair_weight` by the TA multiplier and then
     **renormalizes weights to sum to 1 and redeploys the full budget**.
     Under trim-only, if most names get similar multipliers, renormalization
     cancels the overlay exactly; whatever survives is *redistribution into
     the un-trimmed names* — concentration, the opposite of the intent.
  2. Possible silent lookup misses (`KeyError → mult = 1.0`) would also
     produce identical paths across α/floor.
  Either way: "size risky names down" has not actually been falsified. What
  failed was cap-churn plus renormalize-and-redeploy.
- F01 (daily, no cap) at +7pp vs baseline hints the cross-sectional signal
  has directional content if the sizing plumbing doesn't destroy it.

## 2. Gaps in the current production sizing (`v6_b4_pf_weights.py`)

The opt2 stack is: `decay / borrow penalties → tail-risk penalty →
covariance-concentration penalty → bounds → vol-ETP haircut`, plus dynamic
hedge ratios `h` per underlying. Reviewing it against the "AMD problem":

1. **Tail risk is evaluated as of `start_sim`, not the run date.**
   `risk_raw = _tail_risk_raw(risk_symbol, start_sim, ...)` and inside,
   `s = px.loc[px.index <= as_of]`. `start_sim` is the first date on which
   `min_pairs` pairs have price rows — with long-history vol-ETP pairs in
   the cache this can be many years back. The covariance window is likewise
   clipped to `index < sim_cut`. This is correct hygiene for the notebook
   backtest the module was factored from (weights frozen at sim start), but
   in production GTP it means **the tail estimate can ignore the most
   recent years of data entirely** — AMD's rally, 2025 crashes, all of it.
   *Action: print `meta["start_sim"]` from the next GTP run to confirm how
   stale the as-of actually is; if stale, an `as_of=run_date` mode is a
   candidate production fix (with its own A/B validation, since it changes
   live weights).*

2. **The tail measure is unconditional, not state-dependent.** Worst
   historical 20-day return (70% weighted to FULL history) + downside vol
   is the same number whether the stock sits at all-time highs or is
   already down 60%. The user's intuition — "how far can it fall from
   *here*" — is exactly the conditioning the measure lacks.

3. **Cross-sectional compression.** `risk_adj` is median-normalized, then
   penalized with `exp(−2.5·norm)` floored at 0.25. `exp(−2.5·0.55) ≈ 0.25`,
   so every name above ~0.55× the median risk — i.e. most of a book made of
   high-vol crypto/AI names — saturates at the same floor. Differentiation
   survives only through the `1 + 3·norm^1.5` denominator. Extreme names do
   not stand out proportionally to their risk.

4. **The risk penalty ignores the hedge ratio.** Pair crash exposure per
   gross dollar is approximately `(1−h)` (delta-matched hedge), yet
   `risk_adj = tail × |beta|` uses the raw underlying tail regardless of
   whether the pair runs h=0.30 or h=0.80. A tightly hedged pair on a risky
   name is penalized the same as a loosely hedged one.

## 3. Proposed design: size on conditional pair-level crash loss

Define, per pair *i* at each weekly opt2 rebalance (as of the run date):

**(a) Conditional crash estimate for the underlying**

```
runup_i   = max(0, P_i / anchor_i − 1)          anchor = 252d median close (or 200dma)
retrace_i = θ · runup_i / (1 + runup_i)          θ ≈ 0.5 = "half the run-up can retrace"
tail_i    = realized tail (current _tail_risk_raw, as_of = run date,
            trailing window emphasized rather than 70% full-history)
C_i       = max(tail_i, retrace_i)               conditional crash, in return units
```

`max` (not blend) so a name with no crash history but a huge run-up still
gets flagged, and a name with crash history but already flat on its anchor
keeps its historical floor. AMD at 1.8× its 252d median: `runup=0.8 →
retrace≈0.22` with θ=0.5 — a ~22% conditional crash floor even if its
recent realized tail is quiet.

**(b) Pair loss per gross dollar**

```
L_i ≈ (1 − h_i) · C_i · (1 + φ·C_i)
```

`(1−h_i)` is the unhedged fraction (short inverse ETF loses `≈ L·r` on an
underlying move `r`; the short-underlying hedge recovers `h` of it).
`(1+φ·C_i)` is a convexity bump (φ ≈ 0.5–1.0) for the fact that a short
daily-rebalanced inverse loses *more* than linearly on a sustained
multi-day crash. Using `h_i` here finally makes the risk penalty consistent
with the dynamic hedge targets the same engine already computes.

**(c) Three sizing levers (test separately, then combined)**

1. **Inverse-ES weights**: replace the exp-penalty/denominator pair with
   `score_i = base_score_i / max(L_i, L_floor)` — dollars scale inversely
   with conditional crash loss. No median normalization, so extreme names
   are penalized in proportion, not relative to an already-risky median.
2. **Per-name crash budget (the direct "AMD rule")**: cap
   `gross_i · L_i ≤ ρ · sleeve_equity` with ρ ≈ 0.5–1.0% — "if this name
   gets cut in half tomorrow, the pair loses at most ρ of the sleeve."
   Hard, interpretable, and independent of the optimizer internals.
3. **h-tilt instead of (or before) w-cut**: raise `h_i` with `C_i`
   (e.g. `h_i' = min(h_max, h_i + κ·retrace_i)`). Hedging more preserves
   the decay harvest while cutting crash beta — often better economics than
   shrinking the position, at the cost of extra underlying borrow/margin.

**(d) Mechanics constraints (the actual lessons of the battery)**

- Apply at **weekly opt2 rebalance dates only** — never daily rescaling.
- **Trim-only relative to the opt2 baseline weights** — this overlay only
  sizes down; it never sizes up drawn-down decay names (the half of
  drawdown_63 that both failed and is dangerous — knife-catching).
- **Do not renormalize freed weight back into the book.** If risky names
  are cut, sleeve gross deploys below budget. Redistribution concentrates
  into the remaining names and neutralizes the overlay (see §1).
- Enforce budget in the *targets* (weights), not via end-of-day position
  rescale; Phase 2b's hysteresis band then suppresses churn naturally.

## 4. Experiment spec (reuse `scripts/b4_ta_research_lib.py` harness)

Variant family `G*` on the same window (2025-01-01 → 2026-07-08, $126k,
20 bps):

| ID | Config |
|----|--------|
| G0 | baseline = production opt2 weights, rebalance-only (same as R0) |
| G1 | inverse-ES weights, θ=0.5, no renorm |
| G2 | per-name crash cap ρ=0.75%, no renorm |
| G3 | h-tilt κ=0.5, h_max=0.85 |
| G4 | G1 + G2 |
| G5 | G2 + G3 |
| Sweep | θ ∈ {0.3, 0.5, 0.7} × ρ ∈ {0.5%, 0.75%, 1.0%} |

Diagnostics required per run (to avoid the F03/F05 identical-results trap):

- distribution of multipliers actually applied per rebalance date (assert
  not all ≈ 1.0, assert cross-sectional dispersion > 0);
- gross deployed vs budget over time (expect < 100% sometimes — fine);
- per-name contribution: did AMD-type names (top-quartile run-up) actually
  shrink, and what did that cost/save;
- trade cost vs R0 (must stay same order of magnitude, not 15×).

Acceptance: same Popper gates (≥2pp CAGR lift, maxDD not >25% worse, mean
gross ≤110% budget) **plus** split-half H1/H2 both non-catastrophic, plus a
crash-day study: on the worst 10 underlying-crash days in-window, the G
variants must lose materially less than R0 on the affected pairs (that is
the entire point of the overlay; a CAGR tie with lower crash loss is a
pass, since this is insurance, not alpha).

## 5. Ordered next steps

1. One-line diagnostic in GTP log: print `start_sim` from opt2 meta
   (confirms or clears finding §2.1). Zero risk. **DONE 2026-07-10.**
2. If confirmed stale: add `tail_as_of="run_date"` option to
   `V6PfParams`, A/B the weight diff on the next few plans before enabling.
   **DONE 2026-07-10** (see §6).
3. Build the `G*` variant family in the research harness with the
   multiplier-dispersion diagnostics baked in. **DONE 2026-07-10** (see §7).
4. Review F15 (blacklist lift) as its own decision — borrow/locate/risk
   review, not bundled with sizing mechanics. **Open.**

## 6. Implementation status (2026-07-10)

**Production diagnostics (behavior-neutral by default):**

- `V6PfParams.tail_as_of` added (`"start_sim"` default = legacy, `"latest"` =
  full history up to run date). Configurable from `bucket4_weekly_opt2` yaml
  via `from_opt2_dict`. Both variants are always computed into the weights
  diagnostics frame (`risk_raw_start_sim` / `risk_raw_latest`), so every GTP
  run now carries the A/B for free.
- GTP now logs after the opt2 solve:
  `[INFO] B4 opt2 tail-risk as_of=... start_sim=... (Nd before run date);
  median risk_raw start_sim=... vs latest=...` — watch the next live run to
  quantify staleness before flipping `tail_as_of: latest` in config.

**Research harness fixes (`scripts/b4_ta_research_lib.py`):**

- Confirmed and fixed §1's suspicion: `simulate_sleeve` applied `apply_ta` to
  a **single-row slice**, whose renormalize-to-1 cancels any multiplier
  exactly. That is why F03/F05/F11–F14 and all 20 sweep combos were
  byte-identical. Overlays are now applied to the full snapshot once per day.
- Phase-2b realism added: `SimSpec(enter_band_pct, min_trade_usd)` models the
  production hysteresis (12% band / $250 floor) — an already-open leg only
  trades when drift exceeds the band, for baseline and variants alike.
- Telemetry: per-pair daily P&L frame and per-rebalance overlay events
  (multiplier, run-up, tail, C, L, h) on every `SimResult`.

## 7. G* battery results (2026-07-08 window, $126k, 20 bps)

`python -m scripts.b4_crash_sizing_suite --run-date 2026-07-08` →
`notebooks/output/b4_crash_sizing/CRASH_SIZING_REPORT.md`. All runs:
rebalance-only, trim-only, no renorm, production ratchet ON, hysteresis
band ON (so the baseline G0 = −36.0% CAGR is itself more realistic, and
slightly worse, than the old R0 = −28.8%).

| Variant | CAGR | MaxDD | Mean gross | Crash-day P&L | Cost |
|---------|------|-------|------------|---------------|------|
| G0 baseline | −36.0% | −66.5% | 23% | −$7,727 | $1,313 |
| G1 inverse-ES | −35.7% | −61.7% | 20% | −$7,036 | $1,417 |
| **G2 crash cap ρ=0.75%** | **+2.8%** | **−46.7%** | 7% | −$5,213 | **$366** |
| G3 h-tilt | −43.8% | −70.6% | 28% | −$11,644 | $2,052 |
| G4 = G1+G2 | −1.3% | −43.9% | 7% | −$5,464 | $358 |
| **G6 = G2 + conservative missing-signal default** | **−9.0%** | −57% | 5% | **−$3,210** | $229 |
| Sweep best: ρ=0.50% | +4.2% | −41.7% | 6% | −$4,741 | — |

What holds up:

- **The per-name crash budget (G2) is the lever that works.** It cuts gross
  in the highest-conditional-loss names (UVIX/SVIX mult 0.15, ASTS 0.09,
  IREN 0.12; COIN 0.94, AMD 1.0 — already small), turns the sleeve CAGR
  positive, reduces maxDD by ~20pp, and cuts trade costs 4× *below* baseline
  — the no-renorm + rebalance-only mechanics avoided the churn that killed
  the drawdown_63 cap variants. Tighter ρ monotonically better in-window
  (0.50% > 0.75% > 1.00%).
- **h-tilt (G3) is ruled out**: raising the underlying short into names that
  keep rallying loses more, not less (−43.8%).
- **Inverse-ES alone (G1) does nothing material** — with `max mult = 1` and
  no renorm it is a milder version of the cap.

Honest caveats (why this is not a deploy recommendation yet):

- **The run-up leg was inert in this window.** `C > tail` never bound
  (retrace 0% of events): realized trailing tail dominates for these
  high-vol names, and the 252d-median anchor needs ~126 obs that several
  new ETP underlyings don't have. The battery validates *conditional-loss
  budgeting*, not the run-up signal specifically; θ sweep is flat.
- **Missing-signal concentration — quantified with the G6 control.** Under
  neutral policy, pairs with <40d history (LITE, SNDK) end up holding the
  largest gross after everything else is trimmed, and they happened to do
  well. G6 re-runs G2 with those names assigned the book's 75th-percentile
  conditional loss instead: CAGR drops from +2.8% to **−9.0%**, so roughly
  a third of the headline lift was that mix shift. The honest read is G6,
  not G2: crash budgeting still adds **+27pp CAGR vs baseline**, cuts
  crash-day dollar loss 58% (−$3.2k vs −$7.7k) and trade costs ~6×, but it
  does not make the sleeve positive by itself in this window. Which default
  to use for short-history names in any production version is a judgment
  call. [HUMAN REVIEW]
- **In return space, crash days are not better** (−49% vs −44% summed
  crash-day returns); the dollar improvement comes from carrying ~3× less
  gross in risky names. That is exactly what a crash *budget* is supposed
  to do, but it means the sleeve harvests less decay when nothing crashes.
- **Split-half is not clean**: H1 +40%/H2 −24% for G2 (baseline H1 +40%/H2
  −70%). H2 improves hugely but is still negative; only ~2 months of real
  B4 gross in-window remains the binding data limitation.

Recommended sequencing: (1) watch the `start_sim` staleness log on the next
live GTP run and A/B `tail_as_of: latest` weight diffs from the diagnostics
CSV; (2) with the G6 control now run, treat −9%/−$3.2k as the honest
in-window estimate of the crash budget's value; (3) if that trade-off is
acceptable (this is insurance: less decay harvest in calm periods for much
smaller crash losses), consider a production `crash_budget` knob in the
opt2 stack, starting shadow-only (log the would-be caps next to actual
weights), plus a decision on the short-history default.

## 8. Production implementation (LIVE 2026-07-10)

Implemented in `scripts/b4_crash_budget.py`, wired into
`generate_trade_plan.py`, configured under
`bucket4_weekly_opt2.crash_budget` in `config/strategy_config.yml`.
Tests: `tests/test_b4_crash_budget.py` (14 cases).

**The old tail is REMOVED.** The unconditional opt2 tail penalty
(``dd_risk_lambda`` / ``risk_denom_coeff`` / ``_tail_risk_raw`` /
``tail_as_of``) was deleted from ``scripts/v6_b4_pf_weights.py`` on
2026-07-10. The score is decay/borrow (+ covariance tilt) only. Stale YAML
keys are silently ignored by ``V6PfParams.from_opt2_dict``. Crash risk is
sized in exactly one place — the budget below.

**The rule.** Per pair, as of the run date, from underlying closes:

1. `runup = max(0, P / median(close, 252d) − 1)`; needs ≥126 obs.
2. `retrace = θ · runup / (1 + runup)` with θ = 0.5.
3. `tail = worst trailing 20d drop over 756d + 0.45 · downside vol (126d)`;
   needs ≥40 obs.
4. `C = max(tail, retrace)` — the conditional crash.
5. `L = (1−h) · β / (1 + h·β) · C · (1 + φ·C)` with φ = 0.5 — the pair's
   loss per gross dollar if C hits, using the pair's live hedge ratio `h`
   and inverse-ETF beta `β`. (Note: this is the *exact* leg-split loss; the
   battery used the `(1−h)·C·(1+φC)` approximation, which is ~5% looser at
   β=2, h=0.45. Production caps are therefore marginally tighter than G2.)
6. `cap_usd = ρ · sleeve_budget / max(L, 0.02)` with ρ = 0.75% — so a
   realized conditional crash costs at most ρ of the sleeve.
7. `gross = min(solved_gross, cap_usd)` — trim-only, never sizes up.

Short-history names (no run-up AND no tail) get the book's 75th-percentile
`L` (the validated G6 control), not a free pass.

**No-renorm mechanics** (the property the battery showed is essential):

- `cap_pair_weights` caps the opt2 weights and returns a *shrunken
  effective budget* (`budget · Σw_capped`), which is what
  `compute_bucket4_targets` receives — its internal renormalization then
  reproduces `min(solved, cap)` exactly instead of redeploying freed gross.
- `b4_core_reserved` (the sleeve-rescale target) is shrunk by the freed
  dollars so the book-level budget rescale cannot re-inflate B4.
- A final `clamp_sized_to_crash_budget` pass (driven by the
  `crash_budget_clamp_usd` row column) re-clamps after the notional-cap
  redistribution / covariance / rescale stack. Usually a no-op.
- Freed dollars stay in cash (they are not shifted to B1/B2 either — the
  stock-sleeve budgets are set before B4 sizing).

**Interaction with the ratchet.** Caps apply to *targets*; the grow-only
ratchet and phase2b's `ratchet_released` gate still govern execution, so a
held position above its cap unwinds gradually via the continuous trim, not
in one forced cover.

**Telemetry.** Every run writes `data/runs/<date>/b4_crash_budget.csv`
(per pair: solved vs capped gross, cap, L, C, runup, tail, h, source) and
logs each capped pair. First live run (2026-07-08 data): 10/10 pairs
capped, sleeve gross $115.5k solved → $16.4k deployed — consistent with
the battery's 5–7% mean gross. If that is too tight, raise `rho`
(sweep showed 0.50% < 0.75% < 1.00% monotone in-window risk/return).
