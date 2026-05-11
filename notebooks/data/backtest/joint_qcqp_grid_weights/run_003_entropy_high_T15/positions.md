# positions: entropy_high_T15

## solve
- status: `optimal` (primary `infeasible`)
- fallback: `no_book_variance`
- solver: `CLARABEL`
- sigma_p_book: 37.69% (target 20.00%)

## diversification
- n_pairs_active: **155** / 155
- eff_n_pair: **75.54**
- top1_pair_share: 3.29%
- top5_pair_share: 15.48%
- hhi_pair: 0.0132
- gini_pair: -0.470

## knobs
- entropy_lambda: `1.0`
- entropy_reference: `prior_w_pre`
- edge_temperature: `1.5`
- mv_lambda: `0.0`
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
- CAGR: 25.08%
- Vol: 16.97%
- Sharpe: 1.48
- Max_DD: -14.16%
- Sortino: 2.53
- Calmar: 1.77

## top 30 positions
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| whitelist_stock       | MAAY | MARA       | 3.29%  | 0.573   | 0.249     | 0.031   | 479         |
| whitelist_stock       | IOYY | IONQ       | 3.28%  | 0.593   | 0.265     | 0.031   | 479         |
| whitelist_stock       | XBTY | IBIT       | 3.20%  | 0.503   | 0.210     | 0.032   | 479         |
| whitelist_stock       | MTYY | MSTR       | 2.89%  | 0.527   | 0.262     | 0.029   | 479         |
| whitelist_stock       | SMYY | SMCI       | 2.82%  | 0.526   | 0.272     | 0.028   | 479         |
| inverse_decay_bucket4 | QBTZ | QBTS       | 2.50%  | 2.364   | 0.506     | 0.027   | 479         |
| inverse_decay_bucket4 | ETHD | ETHA       | 2.50%  | 1.092   | 0.278     | 0.024   | 426         |
| whitelist_stock       | FBYY | META       | 2.43%  | 0.409   | 0.228     | 0.026   | 479         |
| inverse_decay_bucket4 | CORD | CRWV       | 2.26%  | 1.257   | 0.726     | 0.014   | 255         |
| whitelist_stock       | COYY | COIN       | 2.09%  | 0.372   | 0.249     | 0.023   | 479         |
| inverse_decay_bucket4 | PLTZ | PLTR       | 1.67%  | 0.577   | 0.243     | 0.017   | 479         |
| core_leveraged        | NVTX | NVTS       | 1.62%  | 1.078   | 0.169     | 0.009   | 479         |
| core_leveraged        | ASTX | ASTS       | 1.60%  | 0.898   | 0.115     | 0.011   | 479         |
| inverse_decay_bucket4 | MSDD | MSTR       | 1.58%  | 0.889   | 0.348     | 0.018   | 479         |
| core_leveraged        | QBTX | QBTS       | 1.40%  | 0.920   | 0.146     | 0.009   | 479         |
| core_leveraged        | IRE  | IREN       | 1.36%  | 0.865   | 0.133     | 0.010   | 479         |
| core_leveraged        | NEBX | NBIS       | 1.36%  | 0.817   | 0.118     | 0.010   | 363         |
| core_leveraged        | NBIG | NBIS       | 1.35%  | 0.818   | 0.121     | 0.010   | 363         |
| core_leveraged        | SMU  | SMR        | 1.33%  | 0.846   | 0.132     | 0.009   | 479         |
| core_leveraged        | NBIL | NBIS       | 1.31%  | 0.769   | 0.110     | 0.010   | 363         |
| core_leveraged        | CRWG | CRWV       | 1.24%  | 0.798   | 0.128     | 0.009   | 255         |
| core_leveraged        | OKLL | OKLO       | 1.23%  | 0.745   | 0.114     | 0.010   | 479         |
| core_leveraged        | CWVX | CRWV       | 1.18%  | 0.799   | 0.140     | 0.009   | 255         |
| core_leveraged        | RGTX | RGTI       | 1.12%  | 0.704   | 0.118     | 0.009   | 479         |
| inverse_decay_bucket4 | CONI | COIN       | 1.08%  | 0.529   | 0.405     | 0.012   | 479         |
| core_leveraged        | CLSX | CLSK       | 1.07%  | 0.740   | 0.139     | 0.008   | 479         |
| core_leveraged        | RKLX | RKLB       | 1.03%  | 0.567   | 0.092     | 0.009   | 479         |
| core_leveraged        | LABX | ALAB       | 1.01%  | 0.731   | 0.147     | 0.008   | 479         |
| core_leveraged        | ETHU | ETHA       | 0.98%  | 0.519   | 0.086     | 0.009   | 426         |
| core_leveraged        | CRDU | CRDO       | 0.95%  | 0.682   | 0.142     | 0.008   | 479         |

## smallest active positions (long tail)
| bucket         | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| -------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged | MRAL | MARA       | 0.15%  | 0.082   | 0.142     | 0.002   | 479.0       |
| core_leveraged | CRWL | CRWD       | 0.12%  | 0.037   | 0.084     | 0.002   | 479.0       |
| core_leveraged | AVGU | AVGO       | 0.12%  | 0.040   | 0.093     | 0.002   | 479.0       |
| core_leveraged | QPUX | IONQ       | 0.12%  | 0.186   | 0.522     | 0.001   | 479.0       |
| core_leveraged | BOEG | BA         | 0.11%  | 0.034   | 0.089     | 0.001   | 479.0       |
| core_leveraged | ASMG | ASML       | 0.10%  | 0.031   | 0.085     | 0.001   | 479.0       |
| core_leveraged | MSTU | MSTR       | 0.09%  | 0.043   | 0.145     | 0.001   | 479.0       |
| core_leveraged | SMCL | SMCI       | 0.07%  | 0.061   | 0.325     | 0.001   | 479.0       |
| core_leveraged | TARK | ARKK       | 0.06%  | 0.016   | 0.123     | 0.001   | 479.0       |
| core_leveraged | TTDU | TTD        | 0.06%  | 0.017   | 0.139     | 0.001   | 479.0       |
