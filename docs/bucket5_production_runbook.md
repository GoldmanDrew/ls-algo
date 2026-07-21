# Bucket 5 Production B — operator runbook

**Status:** implemented 2026-07-20; shipped in `mode: shadow`.
**Plan of record:** [bucket5_production_b_integration_plan_2026-07-18.md](bucket5_production_b_integration_plan_2026-07-18.md) (2026-07-20 amendment: real account only, wired through `generate_trade_plan` → `rebalance_strategy`).

## Components

| File | Role |
|---|---|
| `config/bucket5_production.yml` | Frozen economics, capital, modes, kill modes, execution gates |
| `scripts/bucket5_policy.py` | Pure Production B rules (parity-tested against `bucket5_insurance_bt.py`) |
| `scripts/bucket5_ledger.py` | Append-only event ledger; conId lot book of record (`data/bucket5_ledger/events.jsonl`) |
| `scripts/bucket5_gtp_ext.py` | Called by `generate_trade_plan.py`: carry targets + option intents + decision manifest |
| `scripts/bucket5_rebalance_ext.py` | Called by `rebalance_strategy.py`: OPT limit-order phase (`B5P\|` namespace) |
| `scripts/bucket5_flex_options.py` | Contract-safe Flex option accounting (called by `ibkr_accounting.py`) |
| `scripts/bucket5_reconcile.py` | Daily lot identity + NAV identity + health status |
| `scripts/bucket5_monitor.py` | `bucket5_live.json` for the dashboard's B5 Product tab live panel |

## Modes (`bucket5_production.mode`)

- `placeholder` — legacy behavior; no B5 option intents anywhere.
- `shadow` (current) — GTP writes carry/option targets + manifest to
  `data/runs/<date>/bucket5_production/`; the rebalancer option phase runs in
  forced dry-run. Zero submissions. Plan requires ≥40 reconciled shadow days.
- `production` — GTP re-sizes the UVIX/SVIX pair row from Production B policy
  and stamps `b5_owner=production` (the rebalancer fails closed without the
  stamp). Option intents submit as limit orders under `B5P|<intent_id>` only
  when the intent id is listed in
  `data/runs/<date>/bucket5_production/approved_intents.csv` (one `intent_id`
  column). Pilot caps: 1 contract per intent, 6 open contracts total.

## Daily flow

```text
generate_trade_plan.py --run-date D          # emits B5 targets + manifest
rebalance_strategy.py  --run-date D [--dry-run]   # stocks phases + B5 option phase
ibkr_accounting.py     D                     # next day: contract-safe put book
python scripts/bucket5_reconcile.py --run-date D  # lot + NAV identity, health
python scripts/bucket5_monitor.py   --run-date D  # dashboard live panel JSON
python -m risk_dashboard.build_site --run-date D  # merges bucket5_live into latest.json
```

Manual approval (production only): copy the intent ids you release from
`option_intents.csv` into `approved_intents.csv` in the same folder, then run
the rebalancer. Unapproved intents are skipped and logged in
`option_execution_audit.csv`.

## Kill modes (`bucket5_production.kill_mode`)

`normal | no_new_risk | reduce_only | flatten_carry | exit_options | halt_all` —
checked by GTP at plan time and by the option phase pre-submit. `halt_all`
cancels working `B5P|` orders (never any other namespace) and stops.

## Fail-closed behaviors to expect

- Stale VIX/VIX3M caches (> `max_signal_age_days`) → manifest `health=no_new_risk`, no intents.
- No two-sided quote / spread over `max_spread_frac_of_mid` → intent skipped, audited.
- Executable ask that breaches the rung budget → 0 contracts + under-coverage alert (never rounds up).
- Duplicate submit after restart → refused via ledger `ORDER_SUBMITTED`/`FILL` events and open `B5P|` orderRefs.
- Production plan without `b5_owner=production` stamp → rebalancer aborts (dual-ownership guard).
- Broker option position without a ledger lot (orphan) → reconcile `no_new_risk`.

## 0DTE isolation

`spx-0dte/live/session_recovery.py` now filters positions AND orphan-order
cancels to same-day SPX/SPXW expiries and never touches `B5P|` orderRefs.
B5 puts prefer XSP; even same-family SPX long-dated puts are invisible to the
0DTE engine.

## Tests

`tests/test_bucket5_production.py` (parity, monetization state machine, ledger
idempotency, GTP shadow/production/stale-signal behavior, Flex namespace
isolation) and `spx-0dte/live/test_session_recovery.py::StrategyIsolationTests`.
