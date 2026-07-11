# Rebalance pipeline: fixes shipped 2026-07-10 + improvement backlog

## What was wrong (verified in code + run artifacts)

1. **Deferred B4 rows were dropped from the resize net.**
   `filter_resize_plan_for_b4_cadence` removed cadence-deferred
   `inverse_decay_bucket4` rows from the Phase 2b plan entirely. On any
   underlying shared with B1/B2, the netted underlying target then became
   B1/B2-only, so Phase 2b could trade the stock *against* the held B4
   short intent. This was the core "buckets are not independent" bug.

2. **Cadence gate never read the plan's own cadence outputs.**
   `generate_trade_plan` writes `data/runs/<D>/b4_hedge_cadence/b4_cadence_explain.csv`
   (keyed by Underlying); the gate looked for `b4_hedge_cadence_explain.csv`
   (per-pair, only written by the manual CLI tool). Result: every
   `rebalance_strategy` run recomputed TR/VCR live per underlying — slow,
   network-dependent, error-prone, and able to diverge from the plan.

3. **The cadence state had never been written.** `data/b4_cadence_state.json`
   did not exist and no `data/runs/*/rebalance/` folder exists after
   2026-03-31, so every pair has always been "due" and the per-pair cadence
   has effectively never gated anything in production.

4. **Cadence marking was heuristic and dry-run-unsafe.** Any due pair on an
   underlying that traded was marked rebalanced — even if its own hedge leg
   was blocked (no locate) or unfilled — and dry-run fill records could
   persist state.

5. **B4 opt2 sizing silently fell back on 2026-07-08.** The GTP log shows
   `bucket4_weekly_opt2 disabled for this run (V6PfParams ...
   'borrow_aversion_source')` and the July 8 plan has no
   `b4_opt2_hedge_ratio` / opt2 leg columns — B4 traded decay-score
   fallback targets that day. Commit `dd385214` fixed `V6PfParams`
   (verified constructing from current YAML) and current code raises
   instead of silently falling back. Watch the next GTP run's log for
   `bucket4_weekly_opt2: tail-risk weights + dynamic hedge targets`.

6. **Phase 2b execution was fully serial** through the coordinator IB
   connection (up to 90 s timeout per leg).

## What changed

| File | Change |
|------|--------|
| `scripts/bucket4_cadence_gate.py` | `filter_resize_plan_for_b4_cadence` now KEEPS deferred rows and flags them `b4_cadence_deferred=True`. New `policies_from_gtp_explain_csv` reads GTP's underlying-keyed explain CSV and broadcasts to pairs; the gate prefers it, then the CLI per-pair CSV, and only recomputes TR/VCR live for pairs missing from both (with a log line). |
| `phase2b_resize.py` | `build_resize_trades` skips ETF legs of deferred rows with reason `b4_cadence_deferred` while the underlying net still includes their signed `long_usd`. Single-trade execution extracted to `_execute_one_resize_trade`; new `execute_resize_parallel` runs two waves (all SELLs, then all BUYs) across per-worker IB connections (clientId +500..); `execute_resize_serial` retained. |
| `rebalance_strategy.py` | New `compute_b4_pairs_to_mark`: a due pair's clock resets only when its ETF leg filled, or was evaluated and found at target (within band / floor / ratchet hold). Blocked/unfilled legs stay due and retry next run. State never written on `--dry-run`. At-target due pairs are also marked when Phase 2b produces zero trades. Phase 2b uses the parallel executor when `execution.resize_max_workers > 1`. |
| `config/strategy_config.yml` | `execution.resize_max_workers: 8` (set to 1 for the legacy serial loop). |
| `rebalance_strategy.py` (Phase 3, B4-residual hedging) | Phase 3 now hedges the **B1/B2 residual**, not the raw delta-adjusted net: `build_b4_under_targets` collects per-underlying B4/B5 planned short-underlying USD from the plan; `compute_hedge_net_residual` subtracts actual inverse-ETF legs + that plan target from the raw net at every decision point (build, per-trade re-verification in serial and parallel workers, post-pass checks, underlying reconciliation, pre/post summaries). A healthy partially hedged B4 pair no longer looks like an imbalance to correct. `cap_buy_qty_for_b4_short` additionally caps any hedge/reconcile BUY of the underlying so it can never cover the stock below B4's planned short. |
| `rebalance_strategy.py` (establish/resize seam) | `build_establish_trades` now gates per leg instead of per row: only the *underlying* being positioned hands the name to Phase 2b. Already-positioned ETF legs on a near-zero underlying become non-traded "reuse" legs — their targets count in the coverage denominator and their existing short shares count as coverage — so half-opened buckets (naked ETF short with no underlying) are repaired by establish instead of falling between the phases. Open legs are sized to the residual of their target after any tiny existing short. |
| `rebalance_strategy.py` (prices) | The batched price prefetch now runs **before Phase 1** over the full plan/position universe (cleanup reuses the same snapshot); the old post-Phase-1 call remains as a cheap top-up for newly relevant symbols. |
| `scripts/rebalance_run_report.py` (new) | Post-run anomaly report written to `data/runs/<D>/rebalance/run_report.md` and echoed at end of every run (non-fatal on failure): decision counts, unresolved drift (wanted to trade, could not), fill shortfalls, no-locate/201 blocks, and pairs blocked on consecutive prior runs. Uses only local artifacts — works on any machine that ran the rebalance, degrades gracefully when history is missing. Also runnable standalone: `python scripts/rebalance_run_report.py --run-date <D>` (exit code 2 when anomalies exist, for scheduling wrappers). |

Semantics now: **each bucket keeps its own target and its own hedge-leg
cadence; only the shared underlying stock leg is netted across buckets to
avoid double trading** — exactly the intended "independent, but net the
underlying" contract. The hysteresis band prevents daily churn of the
underlying leg from deferred pairs' small target drift.

## How to verify on the next live cycle

1. `python generate_trade_plan.py --run-date <D>` — confirm log shows
   `bucket4_weekly_opt2: tail-risk weights ...` (not a fallback WARN) and
   `data/runs/<D>/b4_hedge_cadence/b4_cadence_explain.csv` exists.
2. `python rebalance_strategy.py --run-date <D> --dry-run` — check
   `data/runs/<D>/rebalance/b4_cadence_decisions.csv` (all pairs should be
   `no_prior_rebalance` on the first run) and `resize_decisions.csv` for
   `b4_cadence_deferred` reasons in later runs.
3. After the first real (non-dry) run, `data/b4_cadence_state.json` should
   exist with one `ETF|UND: date` entry per reconciled pair.
4. Phase 3 log should show `[HEDGE] B4 structural underlying-short targets
   loaded for N underlyings ... hedging on B4-residual basis.` and, on
   shared names, per-underlying lines with `b4=$... raw=...` next to the
   residual net. B4-only names (healthy pairs) should read `ok`, not
   `TRIGGERED`.
5. End of run prints `[REPORT] Clean run -> .../run_report.md` (or
   `N item(s) to review`). Read that file — it lists unresolved drift,
   fill shortfalls, no-locate blocks, and chronic blockers.

## Improvement backlog (not yet implemented)

### Correctness / bucket isolation

- **Per-bucket attribution of the netted underlying order.** Resize
  underlying orders carry only the B1/B2 `LSB` token; the B4 share is
  reconstructed later (`qty_b4_structural`). Emitting the B4 permille in
  the order ref too would make accounting reconciliation exact.
- **`drawdown_governor` config block is a stub** — YAML exists but no
  Python reads it. Either implement or delete to avoid false confidence.
- **GTP explain CSV should carry ETF.** `_emit_b4_cadence_outputs` writes
  per-underlying rows; adding the pair's ETF (available in `tgt_df`) would
  remove the broadcast step and let two ETFs on one underlying (e.g.
  SPCG/SSPC on SPCX) carry distinct hedge ratios if that ever diverges.

### Speed / robustness

- **Preflight all locates once.** Build the FTP short map + screener
  availability once per run and pass a frozen snapshot to all phases
  (Phase 2 and 2b both re-check per leg today).
- **Tighten `timeout_sec`.** 90 s per leg × serial retries dominated long
  runs. With parallel waves this matters less, but adaptive orders on
  liquid ETFs rarely need more than ~30 s to reach a terminal state.
- **Connection pool.** Establish opens and closes up to 25 worker + 10 leg
  sockets per run; a persistent pool with clientId leasing would cut
  connect/handshake overhead and TWS connection-slot churn.
- **Idempotent re-run / resume.** Persist a per-run ledger of submitted
  order refs so a crashed run can resume without re-submitting fills
  already made (order refs are already unique per leg — a lookup at
  startup against today's executions would suffice).

### Process / trading the strategy generally

- **Run the rebalancer on a schedule with telemetry checks.** The gap in
  `data/runs/*/rebalance/` (nothing since 2026-03-31) means the operator
  cadence (`operator_check_days: 5`) has not been honored. A scheduled
  wrapper that runs GTP → contract check → rebalance `--dry-run` → prompt
  (or auto) would keep the cadence real and produce the audit trail the
  gate depends on.
- **Fail loud on plan-quality regressions.** The July 8 silent opt2
  fallback traded the wrong B4 targets. `scripts/b4_plan_contract.py`
  already validates artifacts; run it (and fail the pipeline) after every
  GTP, not just ad hoc.
- **Full one-shot target-diff pass.** The per-leg establish qualification
  shipped today removes the half-opened-bucket seam surgically; a future
  refactor could still collapse establish + resize into one "diff current
  vs target, trade what's outside band" pass to unify the two code paths
  entirely. Recommend only after several clean sessions of the new
  telemetry (`run_report.md`) on live runs.
