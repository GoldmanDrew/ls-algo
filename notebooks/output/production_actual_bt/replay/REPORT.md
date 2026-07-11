# Production actual backtest (replay) — 2025-05-01 → 2026-07-09

Mode: **replay**
Capital: $1,050,000 × 4.0x

## Sleeve / book summary
```
sleeve   mode  n_rebal  turnover_l1    turnover_usd  cash_days first_plan  n_plans_used      start_usd      end_usd  execution_lag_sessions target_notional_mode  commission_per_share  margin_rate_annual  financing_daycount  use_resize_bands  enter_band_pct  exit_band_pct  min_trade_usd    cagr    vol  sharpe   maxdd
  BOOK replay       74      38.8700 39,546,906.3047        167 2025-12-28            61 1,050,000.0000 985,525.2652                       1        equity_scaled                0.0035              0.0445            360.0000              True          0.1200         0.0400       250.0000 -0.0519 0.0952 -0.5169 -0.0797
```

## Book
- CAGR: -0.05193491290650376
- Vol: 0.09518779153340871
- Sharpe: -0.5169171902903447
- MaxDD: -0.07972360782930354
- End NAV: $985525.2651618898

## Limitations
- Phase A: replays archived proposed_trades (exact GTP output when archived).
- Archives begin 2025-12-28; before that policy=cash.
- Schema normalized across eras (gross from |long|+|short|; whitelist_stock→yieldboost).
- Point-in-time plan legs are held as signed close notionals between weekly rebalances; no latest-plan look-ahead.
- Plans are known after their run-date close, execute at the next available close, and earn P&L from the following session.
- Gross targets preserve the archived plan gross/equity multiple as NAV changes; missing panels stay undeployed.
- Existing legs use production Phase-2b enter/exit hysteresis and the configured minimum trade; new pairs establish and exits close.
- Costs include 20 bp slippage per traded dollar, $0.0035/share commissions, archived borrow, and 4.45% margin debit / Actual-360.
- The 4.45% financing input is the Diamond Creek fallback (4.00% benchmark + 45 bp), not a point-in-time OBFR curve.
- Replay mirrors Phase-2b resize math but not broker execution sequencing or the B4 intra-pair cadence engine.
- B4 full stack (opt2 → crash+scale → post-cap smooth → legs) appears only on dates whose archived plan was generated after those features shipped.
- B3 flow excluded. Mirror recompute still skips live B4 opt2/crash/smooth/ratchet.
