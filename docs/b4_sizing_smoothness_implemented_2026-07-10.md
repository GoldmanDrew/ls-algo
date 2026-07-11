# B4 sizing smoothness + robustness — IMPLEMENTED 2026-07-10

Status: all phases of `b4_sizing_smoothness_plan_2026-07-10.md` are live.
Verification run: `data/runs/2026-07-10/` (proposed_trades.csv,
b4_sizing_waterfall.csv, b4_crash_budget.csv).

## What changed, by phase

### Phase 0 — Invariant telemetry (`b4_sizing_waterfall.csv`)

`generate_trade_plan.py` now writes one row per B4 pair with every sizing
stage: solved weight → smoothed weight → opt2 solved gross → crash cap →
post-crash gross → final gross (+ final legs, L, C, `crash_l_source`,
`book_cap_mult`) and the number that matters most:

    rho_effective = gross_final * L / sleeve_budget

the crash loss actually running after ALL caps. The run log prints the max
per name and the book total next to the config rho. A loud
`[WARN] B4 INVARIANT VIOLATION` fires for any sized B4 row that carries no
crash clamp (a CORD-class bypass).

### Phase 1 — One risk law, zero bypasses (the CORD fix)

- The crash-clamp map is now built from the FULL caps table
  (`compute_crash_caps` over the whole pair cache), not from the subset of
  pairs that survived the opt2 solve. Rows the solver rejects still get their
  cap.
- Rows missing from the caps table entirely get
  `book_default_cap_usd` (new in `scripts/b4_crash_budget.py`): the
  conservative book-quantile-L cap — the same validated G6 mechanism used
  for short-history names.
- Fallback (non-opt2) rows are clamped **at plan time**, not only in the
  final re-clamp, so an oversized fallback row can no longer eat the sleeve
  budget in the rescale and dilute properly-capped pairs.

### Phase 2 — No cliffs (continuous borrow ramp)

`exclude_if_borrow_annual_gt` (the binary router that failed OPEN) is
deprecated and ignored. Replacement in `scripts/v6_b4_pf_weights.py`:

    score multiplier = 1.0 below borrow_ramp_lo (80%)
                     → linear → 0.0 at borrow_ramp_hi (120%)

using the SAME posterior borrow estimate the aversion terms use (one
estimate per quantity — no more spot-vs-posterior regime routing). A name at
the ramp boundary has already faded to ~zero weight, so dropping it is
continuous; anything that still gets sized by any path is crash-clamped
(Phase 1), so the ramp fails closed.

### Phase 3 — One leg-split convention

B4 rows without opt2 legs no longer get the legacy split
(`inv = gross` AND `und = beta*gross`, i.e. real exposure `gross*(1+beta)` —
CORD's 2x hedge overshoot). Every B4 row now uses the opt2 convention
`inv = gross/(1+h*beta)`, `und = h*beta*inv`, with the config `h_mid` (0.45)
as the hedge fallback when no dynamic-h signal exists. The legacy split
remains only for the separately-budgeted B5 vol-ETP sleeve.

### Phase 4 — Explicit layering (rho is the final number)

The B4 sleeve-rescale target is now set to what the risk law actually
allows: crash-capped opt2 targets + clamped fallback rows (bounded by the
YAML budget). The book-cap rescale is therefore ~1.0x on B4 instead of the
hidden ~0.51x haircut. Verified: `rho_effective = 0.750%` exactly (config
0.75%) for every full-size pair; `b4_sizing_waterfall.csv` `book_cap_mult`
column proves it every run. Deployed = 100% of the law-allowed budget.

### Phase 5 — Temporal smoothness (trim-only weight EMA)

`smooth_pair_weights_trim_only` (`scripts/bucket4_weekly_opt2.py`), state in
`data/b4_weight_ema_state.json`, config `bucket4_weekly_opt2.weight_smoothing`
(alpha 0.5). Risk cuts and pair exits apply immediately; size increases move
alpha of the way per run. Deliberately NOT renormalized: renormalizing would
scale cut names back above their solved weight (redeploying trimmed weight —
the drawdown_63 lesson) and prevent convergence. Downstream normalizes.
First run / new pairs are a no-op; identical solves are a fixed point
(verified: consecutive-run state diff = 0.0 for all pairs). Applied BEFORE
the crash budget, so smoothing can never lift a name above its cap.

## Result on the 2026-07-10 book (before → after)

| Pair | Before | After | Why |
|---|---|---|---|
| CORD/CRWV | $3,272 gross, legs −$3,272 inv / −$6,549 und (~$9.8k exposure), NO cap | $1,310, capped, dynamic h split | borrow ramp admits it (posterior 62%); crash budget caps it (L=0.66) |
| DAMD/AMD | $1,165 | $2,297 (largest) | lowest L in book (0.377) — least crash-risky per dollar |
| MSTZ/MSTR | $666 | $1,313 | hidden 0.51x book haircut removed |
| MUZ/MU | absent | $895 (0.5 low-n haircut) | ramp admits it; capped + haircut |
| Sleeve total | $10,859 (rho_eff ~0.38%, one bypass) | $17,190 (rho_eff = 0.75% every full-size name, zero bypasses) | rho is now the law |

Sizes are now ordered by inverse crash-riskiness (cap = rho·budget/L), which
is the design intent: every pair contributes the same worst-case dollar loss.

## Validation state

- `tests/test_b4_crash_budget.py`, `tests/test_b4_crash_sizing.py` extended
  (default cap, ramp knobs, EMA semantics); all pass.
- Backtest gate (crash-sizing suite re-run with ramp + EMA variants) still
  owed before the next config-tightening round; the mechanical fixes above
  are risk-reducing by construction (trim-only everywhere).
