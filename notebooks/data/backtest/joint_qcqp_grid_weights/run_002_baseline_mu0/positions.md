# positions: baseline_mu0

## solve
- status: `optimal` (primary `infeasible`)
- fallback: `no_book_variance`
- solver: `CLARABEL`
- sigma_p_book: 66.04% (target 20.00%)

## diversification
- n_pairs_active: **155** / 155
- eff_n_pair: **13.52**
- top1_pair_share: 10.00%
- top5_pair_share: 50.00%
- hhi_pair: 0.0739
- gini_pair: -0.918

## knobs
- entropy_lambda: `0.0`
- entropy_reference: `prior_w_pre`
- edge_temperature: `1.0`
- mv_lambda: `0.0`
- eff_n_min_pairs: `None`
- weight_ridge_lambda: `0.0`
- mu_shrink_intensity: `0.0`
- turnover_lambda: `1.0`
- confidence_haircut: `True`

## sleeve totals
- core_leveraged: 67.50%
- whitelist_stock: 20.00%
- inverse_decay_bucket4: 12.50%

## perf
- CAGR: 19.70%
- Vol: 20.75%
- Sharpe: 0.95
- Max_DD: -32.61%
- Sortino: 1.25
- Calmar: 0.60

## top 30 positions
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged        | NVTX | NVTS       | 10.00% | 1.078   | 0.169     | 0.012   | 479         |
| core_leveraged        | QBTX | QBTS       | 10.00% | 0.921   | 0.145     | 0.012   | 479         |
| core_leveraged        | ASTX | ASTS       | 10.00% | 0.899   | 0.113     | 0.015   | 479         |
| core_leveraged        | IRE  | IREN       | 10.00% | 0.849   | 0.133     | 0.012   | 479         |
| core_leveraged        | SMU  | SMR        | 10.00% | 0.845   | 0.132     | 0.012   | 479         |
| core_leveraged        | NBIG | NBIS       | 10.00% | 0.818   | 0.120     | 0.013   | 363         |
| whitelist_stock       | IOYY | IONQ       | 4.00%  | 0.593   | 0.274     | 0.032   | 479         |
| whitelist_stock       | MAAY | MARA       | 4.00%  | 0.573   | 0.257     | 0.032   | 479         |
| whitelist_stock       | MTYY | MSTR       | 4.00%  | 0.533   | 0.270     | 0.029   | 479         |
| whitelist_stock       | SMYY | SMCI       | 4.00%  | 0.522   | 0.277     | 0.027   | 479         |
| whitelist_stock       | XBTY | IBIT       | 4.00%  | 0.503   | 0.217     | 0.034   | 479         |
| core_leveraged        | CWVX | CRWV       | 4.00%  | 0.802   | 0.137     | 0.011   | 255         |
| core_leveraged        | NEBX | NBIS       | 3.50%  | 0.817   | 0.117     | 0.013   | 363         |
| inverse_decay_bucket4 | QBTZ | QBTS       | 2.50%  | 2.358   | 0.496     | 0.032   | 479         |
| inverse_decay_bucket4 | CORD | CRWV       | 2.50%  | 1.325   | 0.719     | 0.012   | 255         |
| inverse_decay_bucket4 | ETHD | ETHA       | 2.50%  | 1.094   | 0.269     | 0.027   | 426         |
| inverse_decay_bucket4 | MSDD | MSTR       | 2.50%  | 0.888   | 0.339     | 0.017   | 479         |
| inverse_decay_bucket4 | PLTZ | PLTR       | 2.50%  | 0.579   | 0.233     | 0.017   | 479         |
| core_leveraged        | CRWG | CRWV       | 0.00%  | 0.797   | 0.127     | 0.012   | 255         |
| inverse_decay_bucket4 | CONI | COIN       | 0.00%  | 0.529   | 0.393     | 0.009   | 479         |
| whitelist_stock       | FBYY | META       | 0.00%  | 0.412   | 0.235     | 0.025   | 479         |
| core_leveraged        | NBIL | NBIS       | 0.00%  | 0.769   | 0.109     | 0.013   | 363         |
| whitelist_stock       | COYY | COIN       | 0.00%  | 0.376   | 0.264     | 0.021   | 479         |
| core_leveraged        | WULX | WULF       | 0.00%  | 0.735   | 0.165     | 0.008   | 479         |
| core_leveraged        | CLSX | CLSK       | 0.00%  | 0.739   | 0.140     | 0.010   | 479         |
| core_leveraged        | LABX | ALAB       | 0.00%  | 0.732   | 0.147     | 0.009   | 479         |
| core_leveraged        | OKLL | OKLO       | 0.00%  | 0.746   | 0.114     | 0.012   | 479         |
| core_leveraged        | IREX | IREN       | 0.00%  | 0.710   | 0.168     | 0.008   | 479         |
| core_leveraged        | CRDU | CRDO       | 0.00%  | 0.682   | 0.141     | 0.009   | 479         |
| core_leveraged        | RGTX | RGTI       | 0.00%  | 0.707   | 0.117     | 0.011   | 479         |

## smallest active positions (long tail)
| bucket         | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| -------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged | NFLU | NFLX       | 0.00%  | 0.069   | 0.075     | 0.002   | 479.0       |
| core_leveraged | NFXL | NFLX       | 0.00%  | 0.067   | 0.113     | 0.001   | 479.0       |
| core_leveraged | AMZU | AMZN       | 0.00%  | 0.058   | 0.072     | 0.002   | 479.0       |
| core_leveraged | BOEU | BA         | 0.00%  | 0.056   | 0.077     | 0.001   | 479.0       |
| core_leveraged | AVGU | AVGO       | 0.00%  | 0.040   | 0.090     | 0.001   | 479.0       |
| core_leveraged | BOEG | BA         | 0.00%  | 0.035   | 0.088     | 0.001   | 479.0       |
| core_leveraged | NVDG | NVDA       | 0.00%  | 0.101   | 0.082     | 0.002   | 479.0       |
| core_leveraged | NVDU | NVDA       | 0.00%  | 0.078   | 0.072     | 0.002   | 479.0       |
| core_leveraged | NVDL | NVDA       | 0.00%  | 0.073   | 0.074     | 0.002   | 479.0       |
| core_leveraged | NVDB | NVDA       | 0.00%  | 0.058   | 0.084     | 0.001   | 479.0       |
