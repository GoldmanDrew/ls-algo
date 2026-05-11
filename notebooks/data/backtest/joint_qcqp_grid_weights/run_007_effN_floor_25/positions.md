# positions: effN_floor_25

## solve
- status: `optimal` (primary `infeasible`)
- fallback: `no_book_variance`
- solver: `CLARABEL`
- sigma_p_book: 55.93% (target 20.00%)

## diversification
- n_pairs_active: **155** / 155
- eff_n_pair: **30.91**
- top1_pair_share: 5.00%
- top5_pair_share: 25.00%
- hhi_pair: 0.0323
- gini_pair: -0.765

## knobs
- entropy_lambda: `0.1`
- entropy_reference: `prior_w_pre`
- edge_temperature: `1.0`
- mv_lambda: `0.0`
- eff_n_min_pairs: `25.0`
- weight_ridge_lambda: `0.0`
- mu_shrink_intensity: `0.05`
- turnover_lambda: `1.0`
- confidence_haircut: `True`

## sleeve totals
- core_leveraged: 67.50%
- whitelist_stock: 20.00%
- inverse_decay_bucket4: 12.50%

## perf
- CAGR: 22.91%
- Vol: 18.04%
- Sharpe: 1.27
- Max_DD: -22.77%
- Sortino: 1.96
- Calmar: 1.01

## top 30 positions
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged        | NVTX | NVTS       | 5.00%  | 1.078   | 0.169     | 0.012   | 479         |
| core_leveraged        | QBTX | QBTS       | 5.00%  | 0.921   | 0.145     | 0.012   | 479         |
| core_leveraged        | ASTX | ASTS       | 5.00%  | 0.899   | 0.113     | 0.015   | 479         |
| core_leveraged        | SMU  | SMR        | 5.00%  | 0.845   | 0.132     | 0.012   | 479         |
| core_leveraged        | IRE  | IREN       | 5.00%  | 0.849   | 0.133     | 0.012   | 479         |
| core_leveraged        | NEBX | NBIS       | 4.58%  | 0.817   | 0.117     | 0.013   | 363         |
| core_leveraged        | NBIG | NBIS       | 4.37%  | 0.818   | 0.120     | 0.013   | 363         |
| whitelist_stock       | IOYY | IONQ       | 4.00%  | 0.593   | 0.274     | 0.032   | 479         |
| whitelist_stock       | MAAY | MARA       | 4.00%  | 0.573   | 0.257     | 0.032   | 479         |
| core_leveraged        | CRWG | CRWV       | 3.68%  | 0.797   | 0.127     | 0.012   | 255         |
| core_leveraged        | CWVX | CRWV       | 3.67%  | 0.802   | 0.137     | 0.011   | 255         |
| whitelist_stock       | XBTY | IBIT       | 3.57%  | 0.503   | 0.217     | 0.034   | 479         |
| whitelist_stock       | MTYY | MSTR       | 3.47%  | 0.533   | 0.270     | 0.029   | 479         |
| core_leveraged        | NBIL | NBIS       | 3.17%  | 0.769   | 0.109     | 0.013   | 363         |
| whitelist_stock       | SMYY | SMCI       | 2.97%  | 0.522   | 0.277     | 0.027   | 479         |
| core_leveraged        | OKLL | OKLO       | 2.51%  | 0.746   | 0.114     | 0.012   | 479         |
| inverse_decay_bucket4 | QBTZ | QBTS       | 2.50%  | 2.358   | 0.496     | 0.032   | 479         |
| inverse_decay_bucket4 | CORD | CRWV       | 2.50%  | 1.325   | 0.719     | 0.012   | 255         |
| inverse_decay_bucket4 | ETHD | ETHA       | 2.50%  | 1.094   | 0.269     | 0.027   | 426         |
| inverse_decay_bucket4 | MSDD | MSTR       | 2.39%  | 0.888   | 0.339     | 0.017   | 479         |
| core_leveraged        | CLSX | CLSK       | 2.02%  | 0.739   | 0.140     | 0.010   | 479         |
| inverse_decay_bucket4 | PLTZ | PLTR       | 1.89%  | 0.579   | 0.233     | 0.017   | 479         |
| core_leveraged        | LABX | ALAB       | 1.87%  | 0.732   | 0.147     | 0.009   | 479         |
| core_leveraged        | RGTX | RGTI       | 1.80%  | 0.707   | 0.117     | 0.011   | 479         |
| core_leveraged        | WULX | WULF       | 1.68%  | 0.735   | 0.165     | 0.008   | 479         |
| whitelist_stock       | FBYY | META       | 1.29%  | 0.412   | 0.235     | 0.025   | 479         |
| core_leveraged        | IREX | IREN       | 1.26%  | 0.710   | 0.168     | 0.008   | 479         |
| core_leveraged        | CRDU | CRDO       | 1.19%  | 0.682   | 0.141     | 0.009   | 479         |
| core_leveraged        | CRWU | CRWV       | 0.87%  | 0.655   | 0.142     | 0.009   | 255         |
| whitelist_stock       | COYY | COIN       | 0.69%  | 0.376   | 0.264     | 0.021   | 479         |

## smallest active positions (long tail)
| bucket         | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| -------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged | UNHG | UNH        | 0.06%  | 0.063   | 0.103     | 0.001   | 479.0       |
| core_leveraged | MSFU | MSFT       | 0.06%  | 0.048   | 0.072     | 0.001   | 479.0       |
| core_leveraged | AVGU | AVGO       | 0.06%  | 0.040   | 0.090     | 0.001   | 479.0       |
| core_leveraged | CRWL | CRWD       | 0.06%  | 0.035   | 0.084     | 0.001   | 479.0       |
| core_leveraged | BOEG | BA         | 0.06%  | 0.035   | 0.088     | 0.001   | 479.0       |
| core_leveraged | MSTU | MSTR       | 0.06%  | 0.049   | 0.144     | 0.001   | 479.0       |
| core_leveraged | ASMG | ASML       | 0.06%  | 0.031   | 0.085     | 0.001   | 479.0       |
| core_leveraged | SMCL | SMCI       | 0.06%  | 0.061   | 0.324     | 0.000   | 479.0       |
| core_leveraged | TARK | ARKK       | 0.06%  | 0.017   | 0.122     | 0.000   | 479.0       |
| core_leveraged | TTDU | TTD        | 0.06%  | 0.018   | 0.138     | 0.000   | 479.0       |
