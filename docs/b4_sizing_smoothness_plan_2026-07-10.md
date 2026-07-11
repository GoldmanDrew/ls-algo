# B4 sizing: smoothness + robustness plan — design note

Date: 2026-07-10. Motivated by the CORD/CRWV incident: the pair with the worst
borrow (127% spot) and a top-10 crash-day underlying escaped the crash budget
entirely and became the largest proposed position in the sleeve (~3x peers),
because the opt2 high-borrow exclusion fails OPEN into the legacy decay-score
path.

## Guiding principles

1. **One risk law, zero bypasses.** The conditional-crash budget is the risk
   law of the sleeve. It must bind on every dollar of B4 gross regardless of
   which code path sized it. Risk limits are properties of the book, not of
   the sizing engine that happened to produce a row.
2. **Fail closed.** When a gate rejects a name ("borrow too high to size"),
   the consequence must be less exposure, never more. Rejection paths that
   produce larger positions than acceptance paths are inverted by design.
3. **No cliffs.** Sizing must be approximately continuous in its inputs. A
   2-point change in borrow (89% -> 91%) must not flip a name from
   strictly-capped ~$660 to uncapped ~$3,300. Every hard threshold is a
   discontinuity that data noise will eventually find.
4. **One estimate per quantity.** Spot pair-cache borrow (127%) and screener
   posterior borrow (62%) disagreeing by 2x must not route the same name
   through different regimes. Pick the posterior (it is what nets the edge),
   use it everywhere, and treat disagreement as uncertainty -> smaller size.
5. **Auditable layering.** Every multiplicative stage between raw weight and
   final gross must be visible in one artifact. Today the effective crash
   budget is ~0.38% of sleeve (rho=0.75% then a ~0.51 book-cap haircut) and
   nobody chose that number.

## Phases

### Phase 0 — Invariant telemetry (immediate, no behavior change)

- Emit `b4_sizing_waterfall.csv` per run: one row per pair, one column per
  stage (raw score weight -> bounds -> cov tilt -> crash cap -> book caps ->
  overrides -> final), so any future CORD is visible as a row whose crash-cap
  column is blank.
- Hard assertion in GTP: every B4 row with `gross_target_usd > 0` must have a
  row in the crash-caps table. Violation = loud `[WARN]` + telemetry flag
  (not a crash) for one week of shadow, then upgrade to failure.
- Log effective per-name crash loss after ALL caps:
  `gross_final * L / budget` per pair. This is the real rho actually running.

### Phase 1 — Close the seam (the CORD fix)

Two changes, both small:

- `compute_crash_caps` must cover **every pair in the B4 sleeve frame**, not
  just pairs that survived the opt2 solve. Pairs missing from the opt2
  pair-cache get the `book_quantile` (q75) conservative L — the same
  already-validated G6 mechanism used for short-history names.
- `clamp_sized_to_crash_budget` (the final pass at line ~3468 of
  `generate_trade_plan.py`) then clamps ALL B4 rows, including decay-score
  fallback rows and pair-override rows.

Decision: keep the high-borrow name in the sleeve (capped) or eject it
entirely? Recommendation: keep-open for held inventory (CORD holds ~$77k
inverse), cap new size at the crash budget. Ejecting held names creates
forced covers in hostile borrow — the exact thing the ratchet exists to
prevent.

### Phase 2 — Remove the cliff (continuous borrow treatment)

- Delete `exclude_if_borrow_annual_gt` as a binary router. Replace with the
  continuous penalty already in the score stack: the linear borrow aversion
  (1.5x) + uncertainty penalty (3.0x on posterior variance) already
  downweight expensive names smoothly. If a hard ceiling is still wanted,
  implement it as a weight cap that decays to zero over a band (e.g. linear
  ramp 80%->120% borrow), not a step.
- Single borrow source: posterior everywhere (`borrow_aversion_source:
  posterior` already does this for the score; extend to the exclusion/cap
  logic). Spot-vs-posterior disagreement feeds the uncertainty penalty
  instead of routing.
- Same treatment for the other hard gates worth softening later (entry vol
  floor, min net edge): keep as entry gates for NEW names (hysteresis
  already exists: entry 50% / keep 40%), but never as sizing cliffs on held
  names.

### Phase 3 — One leg-split convention

- The legacy fallback splits `inv = gross, und = beta*gross` (2x hedge
  overshoot at beta=2) while opt2 uses `inv = gross/(1+h*beta)`,
  `und = h*beta*inv`. Nobody should get the legacy split: route every B4 row
  through the opt2 leg solver with the pair's dynamic `h` (or `h_mid=0.45`
  when no signal history exists). Kill or quarantine `_legs_from_gross`
  legacy path for B4.

### Phase 4 — Explicit layering (fix the hidden 0.51 haircut)

- Today: crash cap (rho=0.75%) -> book cap stack halves everything -> final
  crash exposure ~0.38% of sleeve. Either intent A: "rho is the final number"
  -> book caps should be trim-only relative to crash-capped targets and never
  rescale below them uniformly; or intent B: "rho is pre-book-cap" -> then
  document rho_effective in config and telemetry. Recommendation: intent A —
  crash budget is the risk law, book caps handle liquidity/concentration,
  and uniform sleeve rescale should not silently halve an insurance premium
  someone chose deliberately.
- The budget waterfall (Phase 0 artifact) makes this decision verifiable
  every run.

### Phase 5 — Temporal smoothness (lower priority)

- Weight EMA at rebalance: `w_t = (1-a)*w_{t-1} + a*w_solved` with trim-only
  override (risk cuts apply immediately, size increases smooth in). The
  drawdown_63 battery's lesson stands: apply at weekly rebalance only, never
  daily rescale, never renormalize freed weight.
- Cadence hysteresis already handles trade suppression (12% band / $250
  floor); weight smoothing rides on top, it does not replace it.

## Validation gates (per the crash-sizing battery discipline)

- Re-run `scripts/b4_crash_sizing_suite.py` harness with Phase 1-3 variants
  vs current production baseline. Gates: Popper (CAGR lift >= 0, maxDD not
  >25% worse) AND insurance (crash-day loss <= baseline's).
- One week of shadow telemetry (Phase 0 artifacts) before flipping each
  phase live.
- Diagnostics per run must include the identical-results trap check from the
  drawdown_63 battery (distinct multipliers per pair, no byte-identical
  series across variants).

## Explicit non-goals

- No daily retargeting, no regime gating, no redeploy of freed dollars
  (all falsified by the F/G batteries).
- No new risk measures: C = max(tail, theta-retrace) stays; this plan is
  about making the existing law universal and continuous, not changing it.
