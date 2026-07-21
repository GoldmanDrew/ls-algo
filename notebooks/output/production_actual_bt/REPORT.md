# Production actual backtest (prod) — 2026-02-27 → 2026-07-17

Mode: **prod**
Capital: $1,200,000 × 4.0x

## Sleeve / book summary
```
sleeve mode  n_rebal  turnover_l1  turnover_usd  same_run_churn_enabled  avoided_round_trip_usd  risk_override_turnover_usd  one_terminal_target_per_symbol  cash_days first_plan  n_plans_used  start_usd      end_usd  execution_lag_sessions target_notional_mode  scale_sleeves_to_budget  commission_per_share  margin_rate_annual  financing_daycount  short_proceeds_credit_annual  retarget_on_plan_change  use_resize_bands  enter_band_pct  exit_band_pct  min_trade_usd b4_execution purgatory_model_zero_policy b4_membership_clock stock_rebalance_clock  operator_check_days  b4_apply_resize_bands  b4_ratchet_execution_guard  b4_allow_inverse_cover b4_empty_plan_policy  net_shared_underlyings  turnover_pace_enabled turnover_pace_mode turnover_pace_version  confirmation_count  entry_ramp_sessions  reduction_ramp_sessions  remaining_gap_rate stock_midweek_mode  midweek_hedge_repair  hedge_reserve_frac  adv_participation_pct  sleeve_gross_ema_alpha  max_leg_step_pct  pair_gross_ramp_pct  max_daily_turnover_pct  target_blend_alpha  establish_budget_frac  pace_bootstrap_enabled  pace_bootstrap_until_ratio  pace_bootstrap_target_blend_alpha  pace_bootstrap_remaining_gap_rate  pace_bootstrap_establish_budget_frac pace_bootstrap_max_daily_turnover_pct pace_bootstrap_stock_rebalance_clock  pace_bootstrap_latched  n_bootstrap_sessions  resize_age_boost_days  n_deferred_pace  n_hedge_repairs  n_growth_blocked_hedge_infeasible  n_b4_cadence_rebals  n_b4_membership_deferred  n_b4_empty_plan_holds  n_b4_ratchet_pins  n_delist_flat  n_b4_cadence_pairs  n_purgatory_reductions  purgatory_blocked_add_usd     cagr      vol   sharpe     maxdd
  BOOK prod       55     6.318405  7.742039e+06                    True                     0.0                2.108300e+06                            True          2 2026-02-27            87  1200000.0 1.283971e+06                       1        equity_scaled                     True                0.0035              0.0445               360.0                         0.038                    False              True            0.12           0.04          250.0      cadence                        hold         operator_5d           operator_5d                    5                   True                        True                    True                 hold                    True                   True      hedge_safe_v1                     1                   1                    1                        1                0.15         rebal_only                 False                0.15                    0.1                    0.35              0.25                 0.25                    0.08                0.05                    0.5                    True                         0.9                                1.0                                1.0                                   1.0                                  None                                 None                    True                     2                      5              694             1446                                  0                   23                       191                     64                  0              0                  57                     250               1.547350e+07 0.192985 0.228717 0.888902 -0.066953
```

## Book
- CAGR: 0.19298486869961362
- Vol: 0.22871687777849134
- Sharpe: 0.8889017329053848
- MaxDD: -0.06695325685542697
- End NAV: $1283971.3954822682

## Limitations
- Daily targets from full generate_trade_plan on archived etf_screened_today.csv (opt2 → crash → smooth → ratchet) with isolated state carried forward.
- Borrow/edge inputs: screened spot borrow_current + production edge/opt2 path (no avg-borrow overlay).
- Ratchet floors from prior-day sized plan (simulated), not live Flex holdings.
- Archived proposed_trades.csv fallback (GTP fail or plan-only dates) is sanitized to current YAML blacklist + preferred_wrappers before use (see production_actual_backtest.archive_fallback).
- Archive gap ~Dec 2025 / sparse screened: pre-2026-04-25 archives lack net_edge_p50_annual — prod replay shims from net_decay_annual (backtest-only).
- B5 included only when GTP sizes it; no live locates / execution rejects.
- B1/B2 retarget on stock_rebalance_clock (default operator_5d) with Phase-2b hysteresis; purgatory is reduce-only toward model_* (trim, never increase gross); missing/zero model_* share-holds (purgatory_model_zero_policy=hold) — executable 0 is not a flatten.
- B4: TR/VCR cadence + Phase-2b bands on resize; membership add/drop gated by b4_membership_clock (default operator_5d); inverse ratchet pin/trim-cap on covers (b4_ratchet_execution_guard); empty B4 plan (exec gross~0) share-holds the open sleeve (b4_empty_plan_policy=hold).
- Shared underlyings net for financing when net_shared_underlyings=true: borrow / short-credit / margin use residual net short/long only (B1/B2 long vs B4 short internalization). Price PnL still marks each pair leg; book gross/net notionals are netted.
- Sim-only turnover pacing (turnover_pace): EMA stock-sleeve gross, per-leg max step toward plan, soft daily book turnover budget (exits first, then establishes, then resizes).
- Price panel: flex splits + overrides + price_patches + Yahoo referee; Yahoo tail extend; delist cutoff from data/delistings.csv.
- Sleeve legs scaled to YAML sleeve budgets (scale_sleeves_to_budget) then equity-scaled with NAV.
- IBKR short-sale proceeds credit modelled at 3.8% annual on short notional (Actual/360); borrow fee from screened rates with optional borrow_history overlay.
- Point-in-time legs use next-close execution, share-hold marking, 20 bp slippage, $0.0035/share commissions, and 4.45% margin debit / Actual-360.
- B3 flow excluded.
- Screened archives begin 2026-02-27 for this run window (sparse thereafter).
- Ratchet floors from prior-day sized plan (simulated), not live Flex holdings.
