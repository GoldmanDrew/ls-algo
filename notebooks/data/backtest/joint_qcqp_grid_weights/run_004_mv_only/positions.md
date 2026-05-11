# positions: mv_only

## solve
- status: `optimal_inaccurate` (primary `optimal_inaccurate`)
- solver: `SCS`
- sigma_p_book: 20.20% (target 20.00%)

## diversification
- n_pairs_active: **41** / 155
- eff_n_pair: **21.78**
- top1_pair_share: 10.00%
- top5_pair_share: 35.06%
- hhi_pair: 0.0459
- gini_pair: -0.861

## knobs
- entropy_lambda: `0.0`
- entropy_reference: `prior_w_pre`
- edge_temperature: `1.0`
- mv_lambda: `10.0`
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
- CAGR: 25.66%
- Vol: 13.57%
- Sharpe: 1.89
- Max_DD: -10.45%
- Sortino: 3.30
- Calmar: 2.46

## top 30 positions
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged        | MSFU | MSFT       | 10.00% | 0.048   | 0.073     | 0.001   | 479         |
| core_leveraged        | CWEB | KWEB       | 10.00% | 0.096   | 0.073     | 0.003   | 479         |
| core_leveraged        | TARK | ARKK       | 5.98%  | 0.016   | 0.123     | 0.000   | 479         |
| core_leveraged        | BITX | IBIT       | 4.57%  | 0.170   | 0.073     | 0.004   | 479         |
| core_leveraged        | BITU | IBIT       | 4.51%  | 0.148   | 0.073     | 0.004   | 479         |
| core_leveraged        | BTCL | IBIT       | 4.42%  | 0.155   | 0.078     | 0.004   | 479         |
| core_leveraged        | BOEU | BA         | 4.10%  | 0.055   | 0.078     | 0.001   | 479         |
| whitelist_stock       | FBYY | META       | 4.00%  | 0.409   | 0.228     | 0.025   | 479         |
| whitelist_stock       | XBTY | IBIT       | 4.00%  | 0.503   | 0.210     | 0.034   | 479         |
| whitelist_stock       | COYY | COIN       | 4.00%  | 0.372   | 0.249     | 0.021   | 479         |
| whitelist_stock       | MTYY | MSTR       | 4.00%  | 0.527   | 0.262     | 0.028   | 479         |
| core_leveraged        | GOOX | GOOGL      | 3.30%  | 0.077   | 0.078     | 0.002   | 479         |
| whitelist_stock       | SMYY | SMCI       | 2.67%  | 0.526   | 0.272     | 0.027   | 479         |
| core_leveraged        | NVDU | NVDA       | 2.55%  | 0.077   | 0.073     | 0.002   | 479         |
| inverse_decay_bucket4 | PLTZ | PLTR       | 2.50%  | 0.577   | 0.243     | 0.016   | 479         |
| inverse_decay_bucket4 | ETHD | ETHA       | 2.50%  | 1.092   | 0.278     | 0.027   | 426         |
| inverse_decay_bucket4 | CONI | COIN       | 2.50%  | 0.529   | 0.405     | 0.009   | 479         |
| core_leveraged        | NVDL | NVDA       | 2.46%  | 0.072   | 0.075     | 0.002   | 479         |
| inverse_decay_bucket4 | CORD | CRWV       | 2.09%  | 1.257   | 0.726     | 0.012   | 255         |
| core_leveraged        | MSTU | MSTR       | 2.06%  | 0.043   | 0.145     | 0.001   | 479         |
| core_leveraged        | MSTX | MSTR       | 2.03%  | 0.148   | 0.128     | 0.002   | 479         |
| core_leveraged        | NFXL | NFLX       | 1.92%  | 0.068   | 0.113     | 0.001   | 479         |
| inverse_decay_bucket4 | MSDD | MSTR       | 1.89%  | 0.889   | 0.348     | 0.018   | 479         |
| core_leveraged        | NVDG | NVDA       | 1.84%  | 0.101   | 0.082     | 0.002   | 479         |
| core_leveraged        | PALU | PANW       | 1.43%  | 0.091   | 0.073     | 0.002   | 479         |
| core_leveraged        | PANG | PANW       | 1.33%  | 0.067   | 0.082     | 0.002   | 479         |
| core_leveraged        | UBRL | UBER       | 1.30%  | 0.129   | 0.073     | 0.003   | 479         |
| core_leveraged        | NUGT | GDX        | 0.94%  | 0.215   | 0.073     | 0.006   | 479         |
| whitelist_stock       | MAAY | MARA       | 0.70%  | 0.573   | 0.249     | 0.033   | 479         |
| whitelist_stock       | IOYY | IONQ       | 0.63%  | 0.593   | 0.265     | 0.032   | 479         |

## smallest active positions (long tail)
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged        | NFLU | NFLX       | 0.57%  | 0.070   | 0.077     | 0.002   | 479.0       |
| core_leveraged        | CONL | COIN       | 0.56%  | 0.382   | 0.086     | 0.009   | 479.0       |
| core_leveraged        | LLYX | LLY        | 0.54%  | 0.095   | 0.074     | 0.002   | 479.0       |
| inverse_decay_bucket4 | QBTZ | QBTS       | 0.41%  | 2.364   | 0.506     | 0.032   | 479.0       |
| core_leveraged        | TSMG | TSM        | 0.37%  | 0.089   | 0.085     | 0.002   | 479.0       |
| core_leveraged        | TSMU | TSM        | 0.25%  | 0.080   | 0.074     | 0.002   | 479.0       |
| core_leveraged        | BOEG | BA         | 0.24%  | 0.034   | 0.089     | 0.001   | 479.0       |
| core_leveraged        | COIG | COIN       | 0.23%  | 0.310   | 0.101     | 0.006   | 479.0       |
| core_leveraged        | ASMG | ASML       | 0.00%  | 0.031   | 0.085     | 0.001   | 479.0       |
| core_leveraged        | CRWL | CRWD       | 0.00%  | 0.037   | 0.084     | 0.001   | 479.0       |
