# Prompt: align Bucket 4 on **once-weekly** rebalancing

Use this as an agent or human checklist when switching Bucket 4 from **twice-weekly** (Tue+Fri) or **fortnightly** (`2W-FRI`) to **one anchor per week** (typically Friday close: `W-FRI`).

## Goal

Scheduled B4 refreshes should occur **at most once per calendar week** on a single pandas offset (e.g. `W-FRI`), consistent with:

- `scripts.bucket4_weekly_opt2.weekly_rebalance_dates` / `Bucket4WeeklyConfig.weekly_rebalance_freq`
- Hedge panel construction in `Bucket_4_Backtest.ipynb` (cell using `V6_OPT2_WEEKLY_REBAL_FREQ`)
- Cached `v6_opt2_rebal_index` in pickles consumed by `Buckets1-4_v2.ipynb`

## Copy-paste prompt (for Cursor / another agent)

```
Make Bucket 4 rebalance once per week (single weekly anchor, default W-FRI) everywhere it matters:

1. Notebooks: Bucket_4_Backtest — parameters `REBALANCE_FREQ` / `REBALANCE_FREQ_MAP` / `BUCKET4_REBAL_FREQ`, v6 cell `V6_OPT2_WEEKLY_REBAL_FREQ`, and any legacy `TWICE_WEEKLY` (W-TUE ∪ W-FRI) paths in `run_pair_backtest`-style helpers — ensure defaults and docs describe weekly, not Tue+Fri or 2W-FRI unless explicitly requested.

2. Combined book: notebooks/Buckets1-4_v2.ipynb (and Buckets1-4.ipynb if still used) — `EXP["rebalance_freq_bucket_4"]`, B4 grid `rebalance_freq`, and markdown that explains standalone vs combined calendars.

3. Scripts: generate_trade_plan / gtp_sizing_mirror / bucket4_tail_portfolio — any hard-coded rebalance frequencies or assumptions about B4 cadence.

4. Pickles / caches: if `v6_opt2_rebal_index` was built with a stale calendar (e.g. 10 business-day steps), set `EXP["b4_resync_v6_rebal_weekly"] = True` or rebuild state via `bucket4_weekly_opt2.build_bucket4_state` so scheduled dates match weekly_rebalance_dates.

5. Tests: tests/test_bucket4_weekly_opt2.py and tests/test_bucket4_parity.py — update fixtures if expectations assumed non-weekly schedules.

6. After edits: grep the repo for TWICE_WEEKLY, W-TUE, 2W-FRI, V6_BDAY_STEP, and “twice” in B4 context; fix stragglers or document intentional exceptions.
```

## Quick grep targets

```bash
rg "TWICE_WEEKLY|W-TUE|2W-FRI|V6_BDAY_STEP|bi.weekly" notebooks scripts tests generate_trade_plan.py
```

## Related EXP keys (`Buckets1-4_v2`)

| Key | Role |
|-----|------|
| `rebalance_freq` | Base combined-book calendar (often `W-FRI`) |
| `rebalance_freq_bucket_4` | Standalone B4 sim when grid omits `rebalance_freq` |
| `combined_union_v6_b4_rebalance` | Union base dates with `v6_opt2_rebal_index` |
| `b4_weekly_rebalance_freq` | String passed to weekly resample when resyncing v6 |
| `b4_resync_v6_rebal_weekly` | Overwrite pickled `v6_opt2_rebal_index` with weekly module calendar |

## Covariance (Bucket 4 v6 vs book-wide B1/B2)

**Bucket 4 v6 internal weights** (`scripts/v6_b4_pf_weights.py`): build **pair-level** return proxies (dynamic-hedge inverse ETF leg vs underlying fallback), estimate a **shrunk covariance** of those proxies over `cov_lookback_days`, then penalize weights where the pair’s contribution to **weighted variance** (`w ⊙ (Σw)`) is large — multiplier `1 / (1 + cov_penalty * normalized_contrib)`, then renormalize and clip.

**Buckets 1–2 (and full book after sizing)** use `generate_trade_plan.apply_covariance_balance`: operates on **`gross_target_usd`** rows, aggregates exposure **per underlying** as `|β|`-weighted gross, uses **underlying** log-return covariance with the same shrink / marginal-risk idea, applies multipliers per underlying, preserves gross, then runs caps again.

So: **B4 v6** concentrates penalty at the **pair sleeve weight** step using **pair return proxies**; **B1/B2-style** penalty runs later on **dollar gross targets** keyed by **underlying** exposure — complementary layers, not identical math.
