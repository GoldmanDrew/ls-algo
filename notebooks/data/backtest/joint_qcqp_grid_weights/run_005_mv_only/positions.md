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
- CAGR: 25.63%
- Vol: 13.54%
- Sharpe: 1.89
- Max_DD: -10.40%
- Sortino: 3.30
- Calmar: 2.46

## top 30 positions
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged        | MSFU | MSFT       | 10.00% | 0.048   | 0.072     | 0.001   | 479         |
| core_leveraged        | CWEB | KWEB       | 10.00% | 0.096   | 0.072     | 0.003   | 479         |
| core_leveraged        | TARK | ARKK       | 5.99%  | 0.017   | 0.122     | 0.000   | 479         |
| core_leveraged        | BITX | IBIT       | 4.57%  | 0.170   | 0.072     | 0.004   | 479         |
| core_leveraged        | BITU | IBIT       | 4.51%  | 0.147   | 0.072     | 0.004   | 479         |
| core_leveraged        | BTCL | IBIT       | 4.42%  | 0.155   | 0.077     | 0.004   | 479         |
| core_leveraged        | BOEU | BA         | 4.10%  | 0.056   | 0.077     | 0.001   | 479         |
| whitelist_stock       | FBYY | META       | 4.00%  | 0.412   | 0.235     | 0.025   | 479         |
| whitelist_stock       | XBTY | IBIT       | 4.00%  | 0.503   | 0.217     | 0.034   | 479         |
| whitelist_stock       | COYY | COIN       | 4.00%  | 0.376   | 0.264     | 0.021   | 479         |
| whitelist_stock       | MTYY | MSTR       | 4.00%  | 0.533   | 0.270     | 0.029   | 479         |
| core_leveraged        | GOOX | GOOGL      | 3.30%  | 0.077   | 0.077     | 0.002   | 479         |
| whitelist_stock       | SMYY | SMCI       | 2.66%  | 0.522   | 0.277     | 0.027   | 479         |
| core_leveraged        | NVDU | NVDA       | 2.56%  | 0.078   | 0.072     | 0.002   | 479         |
| inverse_decay_bucket4 | PLTZ | PLTR       | 2.50%  | 0.579   | 0.233     | 0.017   | 479         |
| inverse_decay_bucket4 | ETHD | ETHA       | 2.50%  | 1.094   | 0.269     | 0.027   | 426         |
| inverse_decay_bucket4 | CONI | COIN       | 2.50%  | 0.529   | 0.393     | 0.009   | 479         |
| core_leveraged        | NVDL | NVDA       | 2.46%  | 0.073   | 0.074     | 0.002   | 479         |
| inverse_decay_bucket4 | CORD | CRWV       | 2.10%  | 1.325   | 0.719     | 0.012   | 255         |
| core_leveraged        | MSTU | MSTR       | 2.07%  | 0.049   | 0.144     | 0.001   | 479         |
| core_leveraged        | MSTX | MSTR       | 2.03%  | 0.155   | 0.126     | 0.002   | 479         |
| core_leveraged        | NFXL | NFLX       | 1.92%  | 0.067   | 0.113     | 0.001   | 479         |
| inverse_decay_bucket4 | MSDD | MSTR       | 1.90%  | 0.888   | 0.339     | 0.017   | 479         |
| core_leveraged        | NVDG | NVDA       | 1.84%  | 0.101   | 0.082     | 0.002   | 479         |
| core_leveraged        | PALU | PANW       | 1.42%  | 0.091   | 0.072     | 0.002   | 479         |
| core_leveraged        | PANG | PANW       | 1.33%  | 0.068   | 0.081     | 0.002   | 479         |
| core_leveraged        | UBRL | UBER       | 1.29%  | 0.128   | 0.072     | 0.003   | 479         |
| core_leveraged        | NUGT | GDX        | 0.94%  | 0.214   | 0.072     | 0.006   | 479         |
| whitelist_stock       | MAAY | MARA       | 0.70%  | 0.573   | 0.257     | 0.032   | 479         |
| whitelist_stock       | IOYY | IONQ       | 0.64%  | 0.593   | 0.274     | 0.032   | 479         |

## smallest active positions (long tail)
| bucket                | etf  | underlying | w_book | mu_used | sigma_eff | q_prior | n_obs_decay |
| --------------------- | ---- | ---------- | ------ | ------- | --------- | ------- | ----------- |
| core_leveraged        | NFLU | NFLX       | 0.56%  | 0.069   | 0.075     | 0.002   | 479.0       |
| core_leveraged        | CONL | COIN       | 0.56%  | 0.381   | 0.083     | 0.009   | 479.0       |
| core_leveraged        | LLYX | LLY        | 0.54%  | 0.097   | 0.073     | 0.003   | 479.0       |
| inverse_decay_bucket4 | QBTZ | QBTS       | 0.40%  | 2.358   | 0.496     | 0.032   | 479.0       |
| core_leveraged        | TSMG | TSM        | 0.37%  | 0.090   | 0.085     | 0.002   | 479.0       |
| core_leveraged        | BOEG | BA         | 0.25%  | 0.035   | 0.088     | 0.001   | 479.0       |
| core_leveraged        | TSMU | TSM        | 0.24%  | 0.080   | 0.073     | 0.002   | 479.0       |
| core_leveraged        | COIG | COIN       | 0.23%  | 0.311   | 0.102     | 0.006   | 479.0       |
| core_leveraged        | ASMG | ASML       | 0.00%  | 0.031   | 0.085     | 0.001   | 479.0       |
| core_leveraged        | CRWL | CRWD       | 0.00%  | 0.035   | 0.084     | 0.001   | 479.0       |
