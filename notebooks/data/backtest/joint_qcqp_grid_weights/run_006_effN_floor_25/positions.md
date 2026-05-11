# positions: effN_floor_25

## solve
- status: `optimal` (primary `infeasible`)
- fallback: `no_book_variance`
- solver: `CLARABEL`
- sigma_p_book: 47.88% (target 20.00%)

## diversification
- n_pairs_active: **155** / 155
- eff_n_pair: **48.69**
- top1_pair_share: 4.79%
- top5_pair_share: 19.11%
- hhi_pair: 0.0205
- gini_pair: -0.655

## knobs
- entropy_lambda: `0.25`
- entropy_reference: `prior_w_pre`
- edge_temperature: `1.25`
- mv_lambda: `0.0`
- eff_n_min_pairs: `25.0`
- weight_ridge_lambda: `0.0`
- mu_shrink_intensity: `0.25`
- turnover_lambda: `1.0`
- confidence_haircut: `True`

## sleeve totals
- core_leveraged: 67.50%
- whitelist_stock: 20.00%
- inverse_decay_bucket4: 12.50%

## perf
- CAGR: 25.26%
- Vol: 18.92%
- Sharpe: 1.33
- Max_DD: -19.90%
- Sortino: 2.25
- Calmar: 1.27

## top 30 positions
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged        | NVTX | NVTS       | 4.79%  | 1.078   | 0.169     | 0.011   | 479         |
| whitelist_stock       | IOYY | IONQ       | 3.86%  | 0.593   | 0.265     | 0.031   | 479         |
| whitelist_stock       | MAAY | MARA       | 3.71%  | 0.573   | 0.249     | 0.032   | 479         |
| core_leveraged        | ASTX | ASTS       | 3.52%  | 0.898   | 0.115     | 0.012   | 479         |
| whitelist_stock       | XBTY | IBIT       | 3.24%  | 0.503   | 0.210     | 0.033   | 479         |
| core_leveraged        | QBTX | QBTS       | 3.19%  | 0.920   | 0.146     | 0.010   | 479         |
| whitelist_stock       | MTYY | MSTR       | 2.92%  | 0.527   | 0.262     | 0.028   | 479         |
| whitelist_stock       | SMYY | SMCI       | 2.81%  | 0.526   | 0.272     | 0.028   | 479         |
| core_leveraged        | IRE  | IREN       | 2.76%  | 0.865   | 0.133     | 0.011   | 479         |
| core_leveraged        | SMU  | SMR        | 2.62%  | 0.846   | 0.132     | 0.011   | 479         |
| core_leveraged        | NEBX | NBIS       | 2.56%  | 0.817   | 0.118     | 0.011   | 363         |
| core_leveraged        | NBIG | NBIS       | 2.52%  | 0.818   | 0.121     | 0.011   | 363         |
| inverse_decay_bucket4 | QBTZ | QBTS       | 2.50%  | 2.364   | 0.506     | 0.029   | 479         |
| inverse_decay_bucket4 | ETHD | ETHA       | 2.50%  | 1.092   | 0.278     | 0.025   | 426         |
| inverse_decay_bucket4 | CORD | CRWV       | 2.50%  | 1.257   | 0.726     | 0.013   | 255         |
| core_leveraged        | NBIL | NBIS       | 2.25%  | 0.769   | 0.110     | 0.011   | 363         |
| core_leveraged        | CRWG | CRWV       | 2.25%  | 0.798   | 0.128     | 0.010   | 255         |
| core_leveraged        | CWVX | CRWV       | 2.12%  | 0.799   | 0.140     | 0.010   | 255         |
| core_leveraged        | OKLL | OKLO       | 1.99%  | 0.745   | 0.114     | 0.011   | 479         |
| whitelist_stock       | FBYY | META       | 1.96%  | 0.409   | 0.228     | 0.026   | 479         |
| inverse_decay_bucket4 | MSDD | MSTR       | 1.93%  | 0.889   | 0.348     | 0.018   | 479         |
| core_leveraged        | CLSX | CLSK       | 1.69%  | 0.740   | 0.139     | 0.009   | 479         |
| core_leveraged        | RGTX | RGTI       | 1.67%  | 0.704   | 0.118     | 0.010   | 479         |
| inverse_decay_bucket4 | PLTZ | PLTR       | 1.63%  | 0.577   | 0.243     | 0.017   | 479         |
| core_leveraged        | LABX | ALAB       | 1.58%  | 0.731   | 0.147     | 0.009   | 479         |
| whitelist_stock       | COYY | COIN       | 1.50%  | 0.372   | 0.249     | 0.022   | 479         |
| core_leveraged        | WULX | WULF       | 1.45%  | 0.736   | 0.167     | 0.008   | 479         |
| core_leveraged        | CRDU | CRDO       | 1.33%  | 0.682   | 0.142     | 0.008   | 479         |
| core_leveraged        | IREX | IREN       | 1.24%  | 0.705   | 0.172     | 0.007   | 479         |
| core_leveraged        | CRWU | CRWV       | 1.19%  | 0.657   | 0.143     | 0.008   | 255         |

## smallest active positions (long tail)
| bucket         | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| -------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged | NFXL | NFLX       | 0.06%  | 0.068   | 0.113     | 0.002   | 479.0       |
| core_leveraged | QPUX | IONQ       | 0.06%  | 0.186   | 0.522     | 0.001   | 479.0       |
| core_leveraged | CRWL | CRWD       | 0.06%  | 0.037   | 0.084     | 0.001   | 479.0       |
| core_leveraged | AVGU | AVGO       | 0.06%  | 0.040   | 0.093     | 0.001   | 479.0       |
| core_leveraged | BOEG | BA         | 0.06%  | 0.034   | 0.089     | 0.001   | 479.0       |
| core_leveraged | ASMG | ASML       | 0.06%  | 0.031   | 0.085     | 0.001   | 479.0       |
| core_leveraged | MSTU | MSTR       | 0.06%  | 0.043   | 0.145     | 0.001   | 479.0       |
| core_leveraged | SMCL | SMCI       | 0.06%  | 0.061   | 0.325     | 0.001   | 479.0       |
| core_leveraged | TARK | ARKK       | 0.06%  | 0.016   | 0.123     | 0.000   | 479.0       |
| core_leveraged | TTDU | TTD        | 0.06%  | 0.017   | 0.139     | 0.000   | 479.0       |
