# Production actual backtest (prod) — 2026-02-27 → 2026-07-13

Mode: **prod**
Capital: $1,050,000 × 4.0x

## Sleeve / book summary
```
sleeve mode  n_rebal  turnover_l1   turnover_usd  same_run_churn_enabled  avoided_round_trip_usd  risk_override_turnover_usd  one_terminal_target_per_symbol  cash_days first_plan  n_plans_used      start_usd        end_usd  execution_lag_sessions target_notional_mode  scale_sleeves_to_budget  commission_per_share  margin_rate_annual  financing_daycount  short_proceeds_credit_annual  retarget_on_plan_change  use_resize_bands  enter_band_pct  exit_band_pct  min_trade_usd b4_execution purgatory_model_zero_policy b4_membership_clock stock_rebalance_clock  operator_check_days  b4_apply_resize_bands  b4_ratchet_execution_guard  b4_allow_inverse_cover b4_empty_plan_policy  net_shared_underlyings  turnover_pace_enabled turnover_pace_mode turnover_pace_version  confirmation_count  entry_ramp_sessions  reduction_ramp_sessions  remaining_gap_rate stock_midweek_mode  hedge_reserve_frac  adv_participation_pct  sleeve_gross_ema_alpha  max_leg_step_pct  pair_gross_ramp_pct  max_daily_turnover_pct  target_blend_alpha  establish_budget_frac  resize_age_boost_days  n_deferred_pace  n_hedge_repairs  n_growth_blocked_hedge_infeasible  n_b4_cadence_rebals  n_b4_membership_deferred  n_b4_empty_plan_holds  n_b4_ratchet_pins  n_delist_flat  n_b4_cadence_pairs  n_purgatory_reductions  purgatory_blocked_add_usd   cagr    vol  sharpe   maxdd
  BOOK prod       32       3.5167 3,796,610.1159                    True                  0.0000                186,709.2083                            True          2 2026-02-27            82 1,050,000.0000 1,126,148.1145                       1        equity_scaled                     True                0.0035              0.0445            360.0000                        0.0380                    False              True          0.1200         0.0400       250.0000      cadence                        hold         operator_5d           operator_5d                    5                   True                        True                    True                 hold                    True                   True      hedge_safe_v1                     1                   1                    1                        1              1.0000         rebal_only              0.1500                 0.1000                  0.3500            0.2500               0.2500                  0.0800              0.1500                 0.5000                      5             3230              201                                  0                    7                       361                     30                  0              2                  58                      30               155,426.2265 0.2069 0.0544  3.5513 -0.0139
```

## Book
- CAGR: 0.20687088100820028
- Vol: 0.054434260194527725
- Sharpe: 3.5512567699699025
- MaxDD: -0.013866949021536512
- End NAV: $1126148.1144682101

## Limitations
- Daily targets from full generate_trade_plan on archived etf_screened_today.csv (opt2 → crash → smooth → ratchet) with isolated state carried forward.
- Borrow/edge inputs: screened spot borrow_current + production edge/opt2 path (no avg-borrow overlay).
- Ratchet floors from prior-day sized plan (simulated), not live Flex holdings.
- Does not prefer archived proposed_trades.csv, but falls back to them when prod sizing fails or on plan-only archive dates.
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
- Notebook blacklist exception (not live YAML): re-admitted APLD, CBRS, SMR.
- Price panel min_days overridden to 20 (default 40) so short-history names like CBRZ can mark.
