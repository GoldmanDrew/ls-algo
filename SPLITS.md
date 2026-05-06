# Corporate-action splits — `ls-algo` operator playbook

This document explains how `ls-algo` detects, repairs, and audits stock-split
corporate actions (especially same-day reverse splits where Yahoo lags on
retro-adjusting history). Source-of-truth code lives in
[`splits.py`](splits.py); the screener and accounting paths consume it via
thin shims.

## Why this matters

A 1-for-10 reverse split that is NOT corrected in the upstream price feed
inflates `vol_etf_annual` by ~+200 %, which then:

- pushes `expected_gross_decay_annual` to nonsensical levels via the Itô
  identity,
- flips `vol_ratio_outlier` (`vol_etf / (|β| · vol_und)` outside the
  per-leverage band),
- corrupts the OLS β estimate when both legs of the pair share the corruption
  with different magnitudes,
- and, when `accounting.use_yfinance: true` is enabled, causes a 10× phantom
  MtM swing on the ex-date.

The 2026-05-05 LeverageShares 1-for-10 reverse splits (BAIG, BMNG, FIGG, DUOG,
CRWG, CRCG — Nasdaq ECA2026-298) were the canonical failure case that drove
this module.

## Sources, in precedence order

| # | Source                  | Where it comes from                                                                                         | Trigger                                  |
| - | ----------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 1 | `flex`                  | IBKR Flex `<CorporateAction type="RS">` in `data/runs/<date>/ibkr_flex/flex_cash.xml`                       | Auto-merged by `ibkr_accounting` into `data/splits_from_flex.csv`. |
| 2 | `yahoo_events`          | `events.splits` payload in the v8 chart API (`https://query1.finance.yahoo.com/v8/finance/chart/<sym>`)     | Captured per-fetch in `daily_screener._get_total_return_series`. |
| 3 | `splits_overrides_csv`  | Operator-managed `data/splits_overrides.csv` (path resolved via `paths.splits_overrides_csv` in YAML)       | CSV must exist; missing file is silent.   |
| 4 | `manual_override_dict`  | Legacy in-code `_LEGACY_MANUAL_OVERRIDES` (SMUP, EOSU)                                                      | Always loaded; preserved for parity.      |
| 5 | `heuristic`             | `splits.detect_heuristic_splits` — z-score outlier + integer-factor matched detector                        | Always run; catches same-day events Yahoo hasn't yet pushed. |

When two sources disagree on the same `(symbol, ex_date)` (within ±2 days),
the **higher-precedence source wins**.

## Convention

Every detected event becomes a `SplitEvent`:

```python
@dataclass(frozen=True)
class SplitEvent:
    symbol: str
    ex_date: pd.Timestamp
    factor: float          # PRICE-multiplier applied to pre-split history
    source: str            # "flex" | "yahoo_events" | "splits_overrides_csv" |
                           # "manual_override_dict" | "heuristic"
    note: str
```

`factor` is the **price multiplier** that takes pre-split history onto the
post-split basis:

| Split                  | Pre-split price | Post-split price | `factor`  |
| ---------------------- | --------------- | ---------------- | --------- |
| 1-for-10 reverse       | $1              | $10              | 10.0      |
| 1-for-25 reverse       | $4              | $100             | 25.0      |
| 5-for-1 forward        | $100            | $20              | 0.2       |
| 10-for-1 forward       | $50             | $5               | 0.1       |

The CSV uses the share-multiplier form (`numerator,denominator`) which is the
inverse: a 1-for-10 reverse split is `numerator=1, denominator=10` →
`factor = denominator/numerator = 10`.

## How ops adds a new split

1. Open `data/splits_overrides.csv` (the path is configured under
   `paths.splits_overrides_csv` in `config/strategy_config.yml`). Schema:

   ```
   symbol,ex_date,numerator,denominator,source,note
   BAIG,2026-05-05,1,10,manual,LeverageShares 1-for-10 reverse (ECA2026-298)
   ```

2. Use the share-multiplier form. For a 1-for-N reverse split, set
   `numerator=1, denominator=N`. For an M-for-1 forward split, set
   `numerator=M, denominator=1`.

3. The next `daily_screener.py` run will pick it up automatically. The screener
   prints `[SPLITS] overrides CSV: ...` at startup so you can confirm the file
   was found.

4. When the IBKR Flex CSV (`data/splits_from_flex.csv`) lands a `flex` row
   for the same symbol, the manual override is superseded — `flex` is the
   highest-precedence source. You can leave the manual row in place safely;
   it's a no-op.

## How to verify the fix took

Run the dedicated audit CLI:

```bash
python daily_screener.py --audit-splits --run-date 2026-05-05
```

This is a read-only mode that:

- Builds the universe (or uses `--symbols A,B,C` for a subset),
- Pulls each TR series through the multi-source split pipeline,
- Prints, for every symbol where any source fired in the last
  `--audit-window 5` trading days:

  ```
  symbol  ex_date     factor  source        sigma_raw  sigma_clean  delta  note
  BAIG    2026-05-05  10.0    yahoo_events  3.1900     1.2500       1.94   yahoo splitRatio=1:10
  ...
  ```

  `sigma_raw` is from the un-cleaned series; `sigma_clean` from the corrected
  series. A large positive `delta` means the cleaner did real work.

- Optionally appends newly-detected `heuristic` events into the operator CSV
  with `--write-overrides`. Use this to backfill `splits_overrides.csv` for
  future re-runs (heuristics on thin histories can be flaky; promoting them
  to `splits_overrides_csv` makes them deterministic).

## Edge cases handled

- **Same-day reverse split** (the BAIG bug). The legacy detector iterated
  `range(1, len(vals) - 1)` and skipped the most recent bar; the new pipeline
  uses `range(1, len(vals))` and additionally consumes Yahoo `events.splits`
  / Flex `<CorporateAction>` so the heuristic isn't the only line of defence.
- **Same-day forward split** (5-for-1, 10-for-1 …). Symmetric handling via
  `_INTEGER_SPLIT_FACTORS` and their reciprocals.
- **Already-Yahoo-adjusted history.** The self-healing check in
  `splits.apply_split_events` compares the raw boundary ratio. If
  `1/3 < ratio < 3` we treat the source as already adjusted and SKIP. Re-runs
  the next day are idempotent.
- **News-spike non-split.** A 7× single-bar spike on a meme stock that does
  NOT match an integer factor is rejected by the matched-factor gate. A 10×
  spike that DOES match is rejected by the local-vol z-score gate when the
  symbol has high baseline volatility (z < 4σ).
- **Multiple splits in sequence.** Cumulative pre-factors compose:
  `apply_split_events` walks events oldest-first and stacks factors on the
  pre-portion of the series.
- **Yahoo lag on ex-day.** When `events.splits` is present but adjclose has
  not yet been retro-divided, the heuristic AND the Yahoo event both fire for
  the same boundary. Merge picks `yahoo_events` (higher precedence) and
  applies the announced ratio verbatim.
- **Flex CA reportDate vs Yahoo ex-date drift.** Sources are merged with
  ±2-day tolerance; the higher-precedence source's ex-date wins.
- **`SYM` and `SYM.OLD` paired Flex rows.** `parse_flex_corporate_action_splits`
  collapses both rows for the same `actionID` to the canonical symbol.

## Accounting safeguard

`ibkr_accounting.override_mark_prices` accepts `corp_action_split_dates`
(a `dict[symbol → 'YYYY-MM-DD']` built from the freshly-parsed Flex
`<CorporateAction type="RS">` set). On any symbol whose ex-date matches
`run_date`, the function:

1. SKIPS the Yahoo close override for that symbol (keeps Flex `markPrice`),
2. SKIPS the corresponding `unrealized_pnl` adjustment (Flex FIFO PnL is
   self-consistent with the pre-split position quantity),
3. Logs `[ACCOUNTING][corp-action] {sym} reverted to Flex mark on split day`.

This protection is in place even when `accounting.use_yfinance: false`
(default) so flipping the flag is safe.

## Vol-ratio gate (downstream consumer)

`recompute_vol_ratio_gate` in `daily_screener.py` adds two structured columns
to `data/etf_screened_today.csv`:

- `vol_ratio_value: float` — `vol_etf / (|β| · vol_und)` (NaN when β or σ
  inputs are missing/below 0.1).
- `vol_ratio_outlier: bool` — True when `vol_ratio_value` is outside the
  per-leverage band configured under `screener.vol_ratio_gate.by_abs_beta`
  (default `[0.5, 1.5]` for all leverages).

When `vol_ratio_outlier == True` AND the gate is enabled
(`screener.vol_ratio_gate.purgatory_on_outlier: true`),
`recompute_purgatory_by_bucket` forces `purgatory = True` on that row. This
prevents a still-corrupted ETF from sneaking into `proposed_trades.csv`
before the operator has had a chance to add an override.

The console diagnostic warn-band (default `[0.20, 1.75]`) is looser than the
gate so reviewers see borderline rows even when they're inside the
purgatory-forcing band.

## File map

| File                                          | Role                                                                                |
| --------------------------------------------- | ----------------------------------------------------------------------------------- |
| `splits.py`                                   | **Single source of truth**: dataclass, source loaders, merge, apply, heuristic.     |
| `daily_screener.py`                           | Routes every TR fetch through the multi-source pipeline; vol-ratio gate; audit CLI. |
| `etf_analytics.py`                            | Mirrors `daily_screener` via the same `splits.clean_split_artifacts` entry point.   |
| `ibkr_flex.py`                                | After Flex download, writes `<CorporateAction type="RS">` rows to `data/splits_from_flex.csv`. |
| `ibkr_accounting.py`                          | Merges Flex CA rows into the splits CSV; passes today's split set to `override_mark_prices`. |
| `config/strategy_config.yml`                  | `paths.splits_overrides_csv`, `paths.flex_splits_csv`, `screener.vol_ratio_gate`.   |
| `data/splits_overrides.csv`                   | Operator-managed canonical override file (opt-in; safe defaults if absent).         |
| `data/splits_from_flex.csv`                   | Auto-generated from Flex on every accounting run (idempotent).                      |
| `tests/test_split_detection.py`               | Regression: same-day reverse, forward, multi-split, news-spike, Yahoo events, Flex parser. |
| `tests/test_vol_winsorization.py`             | σ winsorization (1/99 clip) — single bad bar no longer dominates σ.                 |
| `tests/test_vol_ratio_gate.py`                | Gate column + per-leverage YAML config.                                              |
