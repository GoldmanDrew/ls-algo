# positions: entropy_plus_mv_sharp

## solve
- status: `optimal` (primary `infeasible`)
- fallback: `no_book_variance`
- solver: `CLARABEL`
- sigma_p_book: 31.23% (target 20.00%)

## diversification
- n_pairs_active: **155** / 155
- eff_n_pair: **69.07**
- top1_pair_share: 4.00%
- top5_pair_share: 16.49%
- hhi_pair: 0.0145
- gini_pair: -0.519

## knobs
- entropy_lambda: `0.15`
- entropy_reference: `prior_w_pre`
- edge_temperature: `1.0`
- mv_lambda: `5.0`
- eff_n_min_pairs: `None`
- weight_ridge_lambda: `0.0`
- mu_shrink_intensity: `0.05`
- turnover_lambda: `1.0`
- confidence_haircut: `True`

## sleeve totals
- core_leveraged: 67.50%
- whitelist_stock: 20.00%
- inverse_decay_bucket4: 12.50%

## perf
- CAGR: 25.35%
- Vol: 15.51%
- Sharpe: 1.63
- Max_DD: -11.72%
- Sortino: 2.73
- Calmar: 2.16

## top 30 positions
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| whitelist_stock       | XBTY | IBIT       | 4.00%  | 0.503   | 0.217     | 0.034   | 479         |
| whitelist_stock       | IOYY | IONQ       | 3.75%  | 0.593   | 0.274     | 0.032   | 479         |
| whitelist_stock       | MAAY | MARA       | 3.21%  | 0.573   | 0.257     | 0.032   | 479         |
| whitelist_stock       | FBYY | META       | 3.02%  | 0.412   | 0.235     | 0.025   | 479         |
| inverse_decay_bucket4 | QBTZ | QBTS       | 2.50%  | 2.358   | 0.496     | 0.032   | 479         |
| inverse_decay_bucket4 | ETHD | ETHA       | 2.50%  | 1.094   | 0.269     | 0.027   | 426         |
| inverse_decay_bucket4 | CORD | CRWV       | 2.50%  | 1.325   | 0.719     | 0.012   | 255         |
| whitelist_stock       | MTYY | MSTR       | 2.43%  | 0.533   | 0.270     | 0.029   | 479         |
| inverse_decay_bucket4 | MSDD | MSTR       | 2.16%  | 0.888   | 0.339     | 0.017   | 479         |
| inverse_decay_bucket4 | PLTZ | PLTR       | 2.13%  | 0.579   | 0.233     | 0.017   | 479         |
| core_leveraged        | ETHU | ETHA       | 2.05%  | 0.519   | 0.085     | 0.012   | 426         |
| whitelist_stock       | SMYY | SMCI       | 1.94%  | 0.522   | 0.277     | 0.027   | 479         |
| core_leveraged        | QBTX | QBTS       | 1.85%  | 0.921   | 0.145     | 0.012   | 479         |
| core_leveraged        | ASTX | ASTS       | 1.70%  | 0.899   | 0.113     | 0.015   | 479         |
| whitelist_stock       | COYY | COIN       | 1.64%  | 0.376   | 0.264     | 0.021   | 479         |
| core_leveraged        | ETU  | ETHA       | 1.55%  | 0.478   | 0.089     | 0.010   | 426         |
| core_leveraged        | NUGT | GDX        | 1.34%  | 0.214   | 0.072     | 0.006   | 479         |
| core_leveraged        | CWVX | CRWV       | 1.27%  | 0.802   | 0.137     | 0.011   | 255         |
| core_leveraged        | CWEB | KWEB       | 1.25%  | 0.096   | 0.072     | 0.003   | 479         |
| core_leveraged        | INTW | INTC       | 1.24%  | 0.368   | 0.088     | 0.008   | 479         |
| core_leveraged        | BABX | BABA       | 1.20%  | 0.196   | 0.072     | 0.005   | 479         |
| core_leveraged        | SMU  | SMR        | 1.19%  | 0.845   | 0.132     | 0.012   | 479         |
| core_leveraged        | NEBX | NBIS       | 1.18%  | 0.817   | 0.117     | 0.013   | 363         |
| core_leveraged        | ETHT | ETHA       | 1.06%  | 0.422   | 0.088     | 0.009   | 426         |
| core_leveraged        | BITX | IBIT       | 1.06%  | 0.170   | 0.072     | 0.004   | 479         |
| core_leveraged        | CRWG | CRWV       | 1.06%  | 0.797   | 0.127     | 0.012   | 255         |
| core_leveraged        | NVTX | NVTS       | 1.02%  | 1.078   | 0.169     | 0.012   | 479         |
| core_leveraged        | LLYX | LLY        | 1.00%  | 0.097   | 0.073     | 0.003   | 479         |
| core_leveraged        | CONL | COIN       | 0.99%  | 0.381   | 0.083     | 0.009   | 479         |
| core_leveraged        | UBRL | UBER       | 0.99%  | 0.128   | 0.072     | 0.003   | 479         |

## smallest active positions (long tail)
| bucket         | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| -------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged | MSTU | MSTR       | 0.06%  | 0.049   | 0.144     | 0.001   | 479.0       |
| core_leveraged | GLGG | GLXY       | 0.06%  | 0.243   | 0.147     | 0.003   | 221.0       |
| core_leveraged | OPEX | OPEN       | 0.06%  | 0.257   | 0.183     | 0.003   | 479.0       |
| core_leveraged | MRAL | MARA       | 0.06%  | 0.086   | 0.140     | 0.001   | 479.0       |
| core_leveraged | SRPU | SRPT       | 0.06%  | 0.278   | 0.315     | 0.002   | 479.0       |
| core_leveraged | JOBX | JOBY       | 0.06%  | 0.114   | 0.153     | 0.001   | 479.0       |
| core_leveraged | SMCL | SMCI       | 0.06%  | 0.061   | 0.324     | 0.000   | 479.0       |
| core_leveraged | QPUX | IONQ       | 0.06%  | 0.186   | 0.520     | 0.001   | 479.0       |
| core_leveraged | TTDU | TTD        | 0.06%  | 0.018   | 0.138     | 0.000   | 479.0       |
| core_leveraged | TARK | ARKK       | 0.06%  | 0.017   | 0.122     | 0.000   | 479.0       |
