# Production actual backtest (recompute) — 2025-05-01 → 2026-07-09

Mode: **recompute**
Capital: $1,050,000 × 4.0x

## Sleeve / book summary
```
sleeve      mode  n_rebal  turnover_l1    turnover_usd  cash_days first_plan  n_plans_used      start_usd      end_usd  execution_lag_sessions target_notional_mode  commission_per_share  margin_rate_annual  financing_daycount  use_resize_bands  enter_band_pct  exit_band_pct  min_trade_usd    cagr    vol  sharpe   maxdd
  BOOK recompute       93      55.8172 55,745,948.0962        167 2025-12-28            84 1,050,000.0000 950,698.3099                       1        equity_scaled                0.0035              0.0445            360.0000              True          0.1200         0.0400       250.0000 -0.0802 0.0958 -0.8320 -0.1060
```

## Book
- CAGR: -0.08021085891307322
- Vol: 0.09575820540540832
- Sharpe: -0.8319982653569671
- MaxDD: -0.10599765716942666
- End NAV: $950698.3099117998

## Limitations
- Phase B: mirror_generate_trade_plan_sizing on archived screened CSVs (decay-score path).
- Mirror does NOT run live B4 opt2 / crash-budget / scale_to_budget / post-cap smooth / ratchet — B4 hedge is plan long/short fractions.
- B1 hysteresis state is carried across recompute days in an isolated temp file (prod state untouched).
- Days without screened fall back to archived proposed_trades.
- Existing legs use production Phase-2b enter/exit hysteresis and the configured minimum trade.
- Point-in-time legs use next-close execution, share-hold marking, 20 bp slippage, $0.0035/share commissions, borrow, and 4.45% margin debit / Actual-360.
- The 4.45% financing input is a visible fallback, not a point-in-time OBFR curve.
- Pre-archive window uses scripts/backfill_screened_history.py (PIT prices; borrow carry-first-known/default; shares stubbed; no live FTP locates).
- B3 flow excluded.
