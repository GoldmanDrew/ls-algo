# Production actual backtest (prod) — 2026-02-27 → 2026-07-13

Mode: **prod**
Capital: $1,050,000 × 4.0x

## Sleeve / book summary
```
sleeve mode  n_rebal  turnover_l1  turnover_usd  cash_days first_plan  n_plans_used  start_usd      end_usd  execution_lag_sessions target_notional_mode  scale_sleeves_to_budget  commission_per_share  margin_rate_annual  financing_daycount  short_proceeds_credit_annual  retarget_on_plan_change  use_resize_bands  enter_band_pct  exit_band_pct  min_trade_usd b4_execution  n_b4_cadence_rebals  n_delist_flat  n_b4_cadence_pairs  n_purgatory_reductions  purgatory_blocked_add_usd     cagr      vol   sharpe    maxdd
  BOOK prod       60    27.793428  3.025310e+07          2 2026-02-27            82  1050000.0 1.210351e+06                       1        equity_scaled                     True                0.0035              0.0445               360.0                         0.038                    False              True            0.12           0.04          250.0      cadence                   32              5                  56                       0                        0.0 0.464755 0.151604 2.644801 -0.04565
```

## Book
- CAGR: 0.46475469773607525
- Vol: 0.1516040773039633
- Sharpe: 2.644800898886203
- MaxDD: -0.045649535515711714
- End NAV: $1210351.406850975

## Limitations
- Daily targets from full generate_trade_plan on archived etf_screened_today.csv (opt2 → crash → smooth → ratchet) with isolated state carried forward.
- Borrow/edge inputs: screened spot borrow_current + production edge/opt2 path (no avg-borrow overlay).
- Ratchet floors from prior-day sized plan (simulated), not live Flex holdings.
- Does not prefer archived proposed_trades.csv, but falls back to them when prod sizing fails or on plan-only archive dates.
- Archive gap ~Dec 2025 / sparse screened: pre-2026-04-25 archives lack net_edge_p50_annual — prod replay shims from net_decay_annual (backtest-only).
- B5 included only when GTP sizes it; no live locates / execution rejects.
- B1/B2 retarget weekly (W-FRI) with Phase-2b hysteresis; purgatory may reduce toward model targets but cannot increase pair gross.
- B4 retargets on production TR/VCR cadence with dynamic h (b4_execution=cadence); legacy weekly_plan_legs available via config.
- Price panel: flex splits + overrides + price_patches + Yahoo referee; Yahoo tail extend; delist cutoff from data/delistings.csv.
- Sleeve legs scaled to YAML sleeve budgets (scale_sleeves_to_budget) then equity-scaled with NAV.
- IBKR short-sale proceeds credit modelled at 3.8% annual on short notional (Actual/360); borrow fee from screened rates with optional borrow_history overlay.
- Point-in-time legs use next-close execution, share-hold marking, 20 bp slippage, $0.0035/share commissions, and 4.45% margin debit / Actual-360.
- B3 flow excluded.
- Screened archives begin 2026-02-27 for this run window (sparse thereafter).
- Ratchet floors from prior-day sized plan (simulated), not live Flex holdings.
