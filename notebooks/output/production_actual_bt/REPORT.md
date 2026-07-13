# Production actual backtest (prod) — 2026-02-27 → 2026-07-09

Mode: **prod**
Capital: $1,050,000 × 4.0x

## Sleeve / book summary
```
sleeve mode  n_rebal  turnover_l1  turnover_usd  cash_days first_plan  n_plans_used  start_usd      end_usd  execution_lag_sessions target_notional_mode  scale_sleeves_to_budget  commission_per_share  margin_rate_annual  financing_daycount  short_proceeds_credit_annual  retarget_on_plan_change  use_resize_bands  enter_band_pct  exit_band_pct  min_trade_usd     cagr      vol  sharpe     maxdd
  BOOK prod       16    25.604664  2.773565e+07          2 2026-02-27            80  1050000.0 1.323562e+06                       1        equity_scaled                     True                0.0035              0.0445               360.0                         0.038                    False              True            0.12           0.04          250.0 0.897756 0.248319 2.73608 -0.047143
```

## Book
- CAGR: 0.8977563816700336
- Vol: 0.24831933737094683
- Sharpe: 2.7360796081532786
- MaxDD: -0.04714303954199206
- End NAV: $1323562.253566446

## Limitations
- Daily targets from full generate_trade_plan on archived etf_screened_today.csv (opt2 → crash → smooth → ratchet) with isolated state carried forward.
- Borrow/edge inputs: screened spot borrow_current + production edge/opt2 path (no avg-borrow overlay).
- Ratchet floors from prior-day sized plan (simulated), not live Flex holdings.
- Does not prefer archived proposed_trades.csv, but falls back to them when prod sizing fails or on plan-only archive dates.
- Archive gap ~Dec 2025 / sparse screened: pre-2026-04-25 archives lack net_edge_p50_annual — prod replay shims from net_decay_annual (backtest-only).
- B5 included only when GTP sizes it; no live locates / execution rejects.
- Plans sized every screened day; book retargets weekly (W-FRI) with the latest plan (share-hold between Fridays — no daily OLS hedge rebuild).
- Phase-2b hysteresis on existing legs (12%/4%/$250); purgatory keep-open holds shares.
- Sleeve legs scaled to YAML sleeve budgets (scale_sleeves_to_budget) then equity-scaled with NAV.
- IBKR short-sale proceeds credit modelled at 3.8% annual on short notional (Actual/360); borrow fee still charged from screened/IBKR rates.
- Point-in-time legs use next-close execution, share-hold marking, 20 bp slippage, $0.0035/share commissions, and 4.45% margin debit / Actual-360.
- B3 flow excluded.
- Screened archives begin 2026-02-27 for this run window (sparse thereafter).
