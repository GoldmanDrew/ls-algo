# Production actual backtest (frozen) — 2025-05-01 → 2026-07-10

Mode: **frozen**
Capital: $1,050,000 × 4.0x

## Sleeve / book summary
```
                sleeve  n_pairs  n_skipped                                                                                                                                                                                 skip_reasons  n_rebal  turnover_l1      start_usd  yaml_budget_usd  plan_deployed_usd        end_usd                                                                    engine   cagr    vol  sharpe   maxdd  admitted_gross_usd  cadence_base_days  cadence_k_tr  min_equity_pre_floor    rho
        core_leveraged      218    10.0000 TXXH:not_in_panel; AMKL:not_in_panel; AXTU:not_in_panel; XNDX:not_in_panel; DRNL:not_in_panel; AAOG:not_in_panel; CATG:not_in_panel; COHH:not_in_panel; SNDG:not_in_panel; STXU:not_in_panel  59.0000       1.9659 1,801,148.2120   2,142,000.0000     1,801,148.2120 1,823,717.0480 pair_daily_returns + weekly retarget (cash for inactive; no pair_eq wipe) 0.0105 0.0420  0.2722 -0.0224                 NaN                NaN           NaN                   NaN    NaN
            yieldboost        8     2.0000                                                                                                                                                           CRY:not_in_panel; CWY:not_in_panel  55.0000       1.8894   535,480.2387   1,932,000.0000       535,480.2387   700,484.7881 pair_daily_returns + weekly retarget (cash for inactive; no pair_eq wipe) 0.2634 0.1128  2.1807 -0.0627                 NaN                NaN           NaN                   NaN    NaN
 inverse_decay_bucket4        9     2.0000                                                                                                                                                          MUZ:not_in_panel; SSPC:not_in_panel      NaN          NaN    27,977.0860     115,500.0000       105,861.9611   116,094.4607       bucket4_dynamic_bt + production cadence/v7 + admitted-only notional 2.3213 2.4560  1.2966 -0.7865         96,223.9222            14.0000       -1.0000                   NaN    NaN
volatility_etp_bucket5        1     0.0000                                                                                                                                                                                          NaN      NaN          NaN    10,500.0000      10,500.0000        10,500.0000    10,727.8530                         bucket5_carry_bt short-UVIX/short-SVIX (plan rho) 0.0182 0.0566  0.3488 -0.0311                 NaN                NaN           NaN           10,272.1707 1.9921
                  BOOK      236        NaN                                                                                                                                                                                          NaN      NaN          NaN 1,050,000.0000              NaN                NaN 1,171,983.9820                                sum sleeve budgets, rescale to capital_usd 0.0967 0.0645  1.4732 -0.0551                 NaN                NaN           NaN                   NaN    NaN
```

## Book
- CAGR: 0.09667700469482932
- Vol: 0.06451021469225339
- Sharpe: 1.4732036862205293
- MaxDD: -0.05510730313518086
- End NAV: $1171983.9819841343

## Limitations
- Universe/weights frozen to run_date proposed_trades (full B4 stack when GTP wrote them).
- B1/B2: plan leg fractions + weekly retarget; inactive names held as cash (no pair_eq wipe on NaN Fridays).
- Prices split-adjusted via data/splits_from_flex.csv before returns.
- B4: production cadence + v7 dynamic hedge; per-pair sim_start; skipped gross stays cash.
- B4 sizing: opt2 → crash-cap + scale_to_budget → post-cap dilution-aware smooth → legs/ratchet.
- B5: bucket5_carry_bt short-UVIX/short-SVIX at plan rho (not B4 dynamic-h).
- B1/B2 ETF shorts pay borrow_current; explicit underlying borrow is charged when that leg is short.
- B3 flow ($1,300/wk) excluded from NAV.
- Frozen is counterfactual (today's book from --start); prefer --mode replay for PIT history.
