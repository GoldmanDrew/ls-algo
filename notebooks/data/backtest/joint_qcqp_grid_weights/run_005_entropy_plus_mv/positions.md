# positions: entropy_plus_mv

## solve
- status: `optimal` (primary `infeasible`)
- fallback: `no_book_variance`
- solver: `CLARABEL`
- sigma_p_book: 32.09% (target 20.00%)

## diversification
- n_pairs_active: **155** / 155
- eff_n_pair: **79.37**
- top1_pair_share: 3.66%
- top5_pair_share: 15.58%
- hhi_pair: 0.0126
- gini_pair: -0.419

## knobs
- entropy_lambda: `0.5`
- entropy_reference: `prior_w_pre`
- edge_temperature: `1.25`
- mv_lambda: `5.0`
- eff_n_min_pairs: `None`
- weight_ridge_lambda: `0.0`
- mu_shrink_intensity: `0.25`
- turnover_lambda: `1.0`
- confidence_haircut: `True`

## sleeve totals
- core_leveraged: 67.50%
- whitelist_stock: 20.00%
- inverse_decay_bucket4: 12.50%

## perf
- CAGR: 25.62%
- Vol: 16.01%
- Sharpe: 1.60
- Max_DD: -11.76%
- Sortino: 2.71
- Calmar: 2.18

## top 30 positions
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| whitelist_stock       | XBTY | IBIT       | 3.66%  | 0.503   | 0.210     | 0.033   | 479         |
| whitelist_stock       | IOYY | IONQ       | 3.42%  | 0.593   | 0.265     | 0.031   | 479         |
| whitelist_stock       | MAAY | MARA       | 3.13%  | 0.573   | 0.249     | 0.032   | 479         |
| whitelist_stock       | FBYY | META       | 2.79%  | 0.409   | 0.228     | 0.026   | 479         |
| whitelist_stock       | MTYY | MSTR       | 2.59%  | 0.527   | 0.262     | 0.028   | 479         |
| inverse_decay_bucket4 | QBTZ | QBTS       | 2.50%  | 2.364   | 0.506     | 0.029   | 479         |
| inverse_decay_bucket4 | ETHD | ETHA       | 2.50%  | 1.092   | 0.278     | 0.025   | 426         |
| inverse_decay_bucket4 | CORD | CRWV       | 2.50%  | 1.257   | 0.726     | 0.013   | 255         |
| whitelist_stock       | SMYY | SMCI       | 2.30%  | 0.526   | 0.272     | 0.028   | 479         |
| whitelist_stock       | COYY | COIN       | 2.12%  | 0.372   | 0.249     | 0.022   | 479         |
| inverse_decay_bucket4 | PLTZ | PLTR       | 1.80%  | 0.577   | 0.243     | 0.017   | 479         |
| inverse_decay_bucket4 | MSDD | MSTR       | 1.65%  | 0.889   | 0.348     | 0.018   | 479         |
| core_leveraged        | ETHU | ETHA       | 1.29%  | 0.519   | 0.086     | 0.010   | 426         |
| core_leveraged        | ASTX | ASTS       | 1.15%  | 0.898   | 0.115     | 0.012   | 479         |
| core_leveraged        | ETU  | ETHA       | 1.12%  | 0.478   | 0.089     | 0.009   | 426         |
| core_leveraged        | QBTX | QBTS       | 1.06%  | 0.920   | 0.146     | 0.010   | 479         |
| core_leveraged        | INTW | INTC       | 0.96%  | 0.370   | 0.089     | 0.008   | 479         |
| core_leveraged        | NEBX | NBIS       | 0.95%  | 0.817   | 0.118     | 0.011   | 363         |
| core_leveraged        | ETHT | ETHA       | 0.94%  | 0.422   | 0.089     | 0.008   | 426         |
| core_leveraged        | SMU  | SMR        | 0.90%  | 0.846   | 0.132     | 0.011   | 479         |
| core_leveraged        | CWVX | CRWV       | 0.88%  | 0.799   | 0.140     | 0.010   | 255         |
| core_leveraged        | RKLX | RKLB       | 0.88%  | 0.567   | 0.092     | 0.010   | 479         |
| core_leveraged        | CONL | COIN       | 0.87%  | 0.382   | 0.086     | 0.008   | 479         |
| core_leveraged        | CRWG | CRWV       | 0.87%  | 0.798   | 0.128     | 0.010   | 255         |
| core_leveraged        | NUGT | GDX        | 0.86%  | 0.215   | 0.073     | 0.006   | 479         |
| core_leveraged        | RGTX | RGTI       | 0.85%  | 0.704   | 0.118     | 0.010   | 479         |
| core_leveraged        | BABX | BABA       | 0.81%  | 0.196   | 0.073     | 0.005   | 479         |
| core_leveraged        | NVTX | NVTS       | 0.80%  | 1.078   | 0.169     | 0.011   | 479         |
| inverse_decay_bucket4 | CONI | COIN       | 0.79%  | 0.529   | 0.405     | 0.010   | 479         |
| core_leveraged        | MUU  | MU         | 0.78%  | 0.337   | 0.085     | 0.007   | 479         |

## smallest active positions (long tail)
| bucket         | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| -------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged | AVGU | AVGO       | 0.12%  | 0.040   | 0.093     | 0.001   | 479.0       |
| core_leveraged | ASMG | ASML       | 0.12%  | 0.031   | 0.085     | 0.001   | 479.0       |
| core_leveraged | MRAL | MARA       | 0.11%  | 0.082   | 0.142     | 0.002   | 479.0       |
| core_leveraged | JOBX | JOBY       | 0.11%  | 0.112   | 0.154     | 0.002   | 479.0       |
| core_leveraged | SRPU | SRPT       | 0.10%  | 0.277   | 0.316     | 0.002   | 479.0       |
| core_leveraged | MSTU | MSTR       | 0.09%  | 0.043   | 0.145     | 0.001   | 479.0       |
| core_leveraged | TARK | ARKK       | 0.06%  | 0.016   | 0.123     | 0.000   | 479.0       |
| core_leveraged | TTDU | TTD        | 0.06%  | 0.017   | 0.139     | 0.000   | 479.0       |
| core_leveraged | QPUX | IONQ       | 0.06%  | 0.186   | 0.522     | 0.001   | 479.0       |
| core_leveraged | SMCL | SMCI       | 0.06%  | 0.061   | 0.325     | 0.001   | 479.0       |
