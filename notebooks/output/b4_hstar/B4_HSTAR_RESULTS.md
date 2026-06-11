# Bucket 4 hedge-ratio (h*) optimization results

Window start: 2025-10-07 | pairs: 45 | slippage 20bps, fee 0, static beta, screener borrow.
Cadence fixed at production optimum: `policy_continuous_interval(base_days=12.0, k_tr=2.25, m_vcr=2.5, min=1, max=21)`, schedule computed once per pair and reused.

## Method

The engine (`run_bucket4_backtest_dynamic_h`) shorts the inverse ETF and shorts `h * |beta|` dollars of the underlying per dollar of inverse short; `h` is consumed as a daily series but only matters on rebalance days. Everything except the h-rule is held at the production configuration above.

**Baseline (v7 closed form):**

```
h_t = clip(0.55 + 1.0*(VCR_t - VCRmed_t), 0.30, 0.80);  NaN -> partial_h;
h <- EMA(h, alpha=0.25, adjust=False)
```

Experiments 1-3 are **in-sample** (parameters chosen on the same window they are scored on). Experiment 4 is the **out-of-sample** test: every 21 trading days, each pair adopts the constant h in {0.30..0.95} with the best trailing-63-day equity growth (`eq[t]/eq[t-63]`, from a precomputed constant-h panel; ties -> lower h), applied from the next day. Warmup (first 63bd) uses the v7 baseline h.

## Baseline score

- EW mean CAGR: **0.0585**
- EW median CAGR: **0.0486**
- winsorized mean: 0.0695 | mean vol: 0.3272 | mean max-DD: -0.2160

## Experiment 1 — per-pair constant h* (IS)

Constant-h grid 0.30..0.95 step 0.05 per pair (`b4_hstar_perpair_profiles.csv`). Per-pair argmax (`b4_hstar_perpair_argmax.csv`):

```
     pair  h_star  cagr_at_hstar  cagr_at_h055  abs_beta  und_vol  borrow_a  net_edge
ASTN/ASTS    0.30       1.920989      1.036258  2.005816 1.042652  0.000000  3.790747
NVDS/NVDA    0.30       0.198951      0.088892  1.494417 0.336746  0.092849  0.194797
UVIX/SVIX    0.30       0.730389      0.487083  1.984718 0.558302  0.010745  1.965851
QBTZ/QBTS    0.30       0.951549      0.788587  2.006780 1.012809  0.438380  3.629417
  QID/QQQ    0.30       0.234544      0.088945  1.996938 0.158316  0.025606 -0.032967
  REW/XLK    0.30       0.297905      0.085567  2.008636 0.207798  0.123102 -0.051458
RKLZ/RKLB    0.30       3.034770      0.524312  2.013770 0.868185  0.682612  2.121285
 NVD/NVDA    0.30       0.292151      0.137732  2.004502 0.336746  0.092975  0.427299
  SCO/USO    0.30       0.564051      0.133774  1.735150 0.327522  0.007021 -0.036647
 SPXS/SPY    0.30       0.129978      0.031145  2.996011 0.116284  0.078868 -0.125620
 SPXU/SPY    0.30       0.158628      0.050705  2.994171 0.116284  0.029989 -0.072889
 STSM/TSM    0.30      -0.411177     -0.447998  2.009731 0.355040  2.141965 -1.681890
 TECS/XLK    0.30       0.456105      0.140023  3.000334 0.207798  0.050753  0.135158
  TWM/IWM    0.30       0.173099      0.060755  2.002032 0.164075  0.057228 -0.038333
  TZA/IWM    0.30       0.248864      0.086126  2.997767 0.164075  0.071362  0.021919
 SDOW/DIA    0.30       0.034690     -0.014215  2.999147 0.118231  0.044570 -0.101086
NBIZ/NBIS    0.30       7.598826      1.609596  2.042842 0.980837  0.190857  3.563449
 WEBS/XLK    0.30      -0.227969     -0.282745  2.795899 0.207798  0.050658 -0.159960
CLSZ/CLSK    0.30       6.607139      1.671080  2.027275 0.943786  0.424849  1.696714
IREZ/IREN    0.30       1.188563      0.645141  2.017457 1.063209  0.864429  2.643906
   BEZ/BE    0.30       2.455639      0.503492  2.010773 1.039419  1.223303  2.330827
HIBS/SPHB    0.30       0.383442      0.100049  3.006812 0.220311  0.404552 -0.185115
 FNGD/XLK    0.30      -0.062511     -0.168156  2.916645 0.207798  0.173168  0.064659
  DXD/DIA    0.30       0.040179     -0.004695  2.002234 0.118231  0.019588 -0.088337
  DUG/XLE    0.30       0.190214      0.038088  1.988816 0.203935  0.213762 -0.205290
 DRIP/XOP    0.30       0.291785      0.114155  1.996282 0.275626  0.083403  0.028455
 DAMD/AMD    0.30       0.979850      0.181862  2.032590 0.595838  0.305774  1.101554
CORD/CRWV    0.55       0.126668      0.126668  2.013856 0.947885  0.304421  2.299679
  TBT/TLT    0.95      -0.046325     -0.071781  1.997764 0.097358  0.058070 -0.121509
  TBX/TLT    0.95      -0.179333     -0.233832  1.372029 0.097358  0.430262 -0.457164
MSTZ/MSTR    0.95       0.035460     -0.287256  2.001739 0.647882  0.155255  1.941048
  TMV/TLT    0.95      -0.119986     -0.176910  2.989784 0.097358  0.370524 -0.451583
BMNZ/BMNR    0.95       0.608588      0.296258  1.991842 0.965302  0.966770  1.900261
BTCZ/IBIT    0.95       0.055696     -0.201320  1.997947 0.412520  0.000000  0.612390
TSDD/TSLA    0.95      -0.108036     -0.157486  2.010756 0.443958  0.497918  0.417678
SMCZ/SMCI    0.95      -0.940131     -0.984281  2.039517 0.701594  8.580133 -6.655443
SMST/MSTR    0.95      -0.270342     -0.553646  1.976358 0.647882  1.201543  0.614811
MSDD/MSTR    0.95       0.014469     -0.306381  1.989542 0.647882  0.156930  1.145425
  SKF/XLF    0.95      -0.055365     -0.093292  1.996919 0.141044  0.075743 -0.115918
CRCD/CRCL    0.95      -0.010697     -0.199856  2.010421 0.899666  0.977448  1.562439
RGTZ/RGTI    0.95       0.230934     -0.004545  1.998962 1.010379  1.089416  3.230792
HOOZ/HOOD    0.95      -0.659073     -0.822933  2.003926 0.674833  3.538119 -1.926103
IONZ/IONQ    0.95      -0.344529     -0.446814  2.031859 0.889428  1.553649  1.640555
CONI/COIN    0.95      -0.242176     -0.565280  1.925575 0.682267  0.649468  0.552436
OKLS/OKLO    0.95       0.051218     -0.049207  2.004924 0.989165  1.542238  1.754736
```

Oracle (pick each pair's IS-best constant h): EW mean CAGR **0.5913**, median **0.1586** — this is the fully overfit upper bound, not attainable.

Cross-section of h* vs pair characteristics:

```
  driver  n   pearson  spearman
abs_beta 45 -0.277840 -0.321729
 und_vol 45  0.207119  0.154425
borrow_a 45  0.348387  0.435127
net_edge 45 -0.124667 -0.011562
```

## Experiment 2 — global v7 refit (IS)

Grid: h_mid in [0.45, 0.55, 0.65, 0.75], k_vcr in [0.0, 0.5, 1.0, 1.5, 2.0], h_max in [0.8, 0.95] (h_min 0.30, EMA 0.25 fixed). Top 10 by composite rank:

```
 h_mid  k_vcr  h_max  n_pairs  ew_mean_cagr  ew_median_cagr  winsor_mean_cagr  ew_mean_vol  ew_mean_max_dd  mean_trades_per_pair  total_slippage  total_borrow  ret_over_vol  calmar_ew  composite_rank  d_mean_cagr_vs_baseline
  0.45    0.0   0.80       45      0.141634        0.072713          0.146165     0.381806       -0.238153             11.555556    21392.513802 571187.770283      0.370957   0.594717           3.500                 0.083097
  0.45    0.0   0.95       45      0.141634        0.072713          0.146165     0.381806       -0.238153             11.555556    21392.513802 571187.770283      0.370957   0.594717           3.500                 0.083097
  0.45    0.5   0.80       45      0.137920        0.076904          0.142225     0.387968       -0.240560             11.555556    21759.196629 569080.906306      0.355494   0.573329           4.500                 0.079384
  0.45    0.5   0.95       45      0.137920        0.076904          0.142225     0.387968       -0.240560             11.555556    21759.196629 569080.906306      0.355494   0.573329           4.500                 0.079384
  0.45    1.0   0.80       45      0.135334        0.080741          0.139404     0.396627       -0.242990             11.555556    22283.733089 567726.131663      0.341211   0.556952           5.500                 0.076797
  0.45    1.0   0.95       45      0.135329        0.080741          0.139404     0.396579       -0.243003             11.555556    22284.148133 567520.081893      0.341241   0.556903           5.500                 0.076792
  0.45    1.5   0.80       45      0.130224        0.082353          0.132681     0.404238       -0.244796             11.555556    22798.555629 566378.618033      0.322146   0.531968           6.375                 0.071687
  0.45    1.5   0.95       45      0.130201        0.082353          0.132683     0.404191       -0.244863             11.555556    22799.648484 565934.495326      0.322128   0.531732           6.625                 0.071665
  0.45    2.0   0.80       45      0.124907        0.086477          0.125460     0.410415       -0.246280             11.555556    23313.391429 565046.666935      0.304342   0.507172           7.125                 0.066370
  0.45    2.0   0.95       45      0.124794        0.086477          0.125370     0.410346       -0.246316             11.555556    23325.564254 564555.113768      0.304117   0.506640           7.875                 0.066257
```

**Best refit:** `{'h_mid': 0.45, 'k_vcr': 0.0, 'h_max': 0.8}` -> EW mean CAGR 0.1416 (+0.0831 vs baseline).

## Experiment 3 — one extra signal on top of the refit optimum (IS)

h_t = clip(h_mid* + k_vcr*(VCR-VCRmed) + k_sig*X, 0.30, h_max*), X in {TR-1, rv_pct-0.5, sign(20d underlying return, shifted)}; signal NaNs contribute 0.

```
signal  coef  n_pairs  ew_mean_cagr  ew_median_cagr  winsor_mean_cagr  ew_mean_vol  ew_mean_max_dd  mean_trades_per_pair  total_slippage  total_borrow  d_mean_cagr_vs_best2
 mom20  -0.2       45      0.206705        0.050659          0.147903     0.394204       -0.228904             11.555556    27711.553690 566901.175144              0.065071
 mom20  -0.1       45      0.183309        0.053013          0.150456     0.391652       -0.234196             11.555556    24981.807491 571441.773620              0.041675
 mom20   0.0       45      0.141634        0.072713          0.146165     0.381806       -0.238153             11.555556    21392.513802 571187.770283              0.000000
 mom20   0.1       45      0.127552        0.090254          0.121349     0.395004       -0.248927             11.555556    21466.058887 574348.338744             -0.014082
 mom20   0.2       45      0.112890        0.088293          0.090221     0.398248       -0.254337             11.555556    22721.411370 572130.764251             -0.028744
rv_pct  -0.2       45      0.169776        0.107693          0.182828     0.375177       -0.234020             11.555556    22114.553591 574884.545851              0.028143
rv_pct  -0.1       45      0.154617        0.093680          0.163527     0.376965       -0.235514             11.555556    21573.013051 572765.934057              0.012983
rv_pct   0.0       45      0.141634        0.072713          0.146165     0.381806       -0.238153             11.555556    21392.513802 571187.770283              0.000000
rv_pct   0.1       45      0.130422        0.052545          0.130294     0.389744       -0.241462             11.555556    21749.476716 570122.340793             -0.011212
rv_pct   0.2       45      0.120814        0.035957          0.115691     0.400917       -0.245487             11.555556    22388.479424 569556.191715             -0.020820
    tr  -0.2       45      0.128707        0.057389          0.126285     0.371147       -0.231700             11.555556    22021.364439 565961.784893             -0.012926
    tr  -0.1       45      0.134743        0.064375          0.135629     0.376094       -0.234811             11.555556    21674.795427 568490.594719             -0.006891
    tr   0.0       45      0.141634        0.072713          0.146165     0.381806       -0.238153             11.555556    21392.513802 571187.770283              0.000000
    tr   0.1       45      0.149425        0.084803          0.156306     0.388339       -0.241710             11.555556    21244.491472 574065.858713              0.007791
    tr   0.2       45      0.158200        0.097741          0.164423     0.395761       -0.245524             11.555556    21238.626753 577140.122627              0.016567
```

Best coefficient per signal (marginal EW mean CAGR vs exp-2 optimum):

```
signal  coef  ew_mean_cagr  d_mean_cagr_vs_best2
 mom20  -0.2      0.206705              0.065071
rv_pct  -0.2      0.169776              0.028143
    tr   0.2      0.158200              0.016567
```

## Experiment 4 — walk-forward adaptive per-pair h* (OOS)

```
          variant  n_pairs  ew_mean_cagr  ew_median_cagr  winsor_mean_cagr  ew_mean_vol  ew_mean_max_dd  mean_trades_per_pair  total_slippage  total_borrow  d_mean_cagr_vs_baseline
      v7_baseline       45      0.058536        0.048605          0.069540     0.327206       -0.215975             11.555556    21065.400051 543471.057324                 0.000000
walkforward_hstar       45      0.114562       -0.037329          0.088947     0.328414       -0.210943             11.555556    26990.035236 521920.049568                 0.056026
```

## Conclusion

- Baseline v7 EW mean CAGR 0.0585 (median 0.0486); best IS refit mean 0.1416 (+0.0831, median +0.0241); OOS walk-forward mean 0.1146 (+0.0560, median -0.0859).
- **Per-pair h* is bimodal at the grid edges**: 27/45 pairs pin at h*=0.30 and 17/45 at h*=0.95. h* is mostly a label for whether the pair made or lost money this window (winners want minimal hedging, losers want maximal), i.e. it encodes the realized outcome rather than an exploitable ex-ante characteristic. The strongest cross-sectional driver is borrow cost (expensive-to-carry pairs prefer high h).
- The best IS refit sets `k_vcr=0` — on this window the VCR tilt adds nothing and the gain comes almost entirely from a lower average h (more residual delta in a period when shorts of inverse ETFs were profitable). That is direction-of-market exposure, not signal alpha.
- The best extra signal is `mom20` (k=-0.2, +0.0651 EW mean CAGR vs the refit optimum), but check the median column in `b4_hstar_signal_marginals.csv` before trusting it — mean improvements driven by a few high-vol pairs while the median deteriorates are an overfit signature.
- **IS vs OOS:** experiments 1-3 select parameters on the same window they are scored on and therefore overstate attainable performance; the per-pair oracle h* is pure overfit. Only the experiment-4 walk-forward delta is an honest estimate of what per-pair h adaptation would have delivered out of sample on this (short, ~8-month) window — and there the mean improves while the **median deteriorates**, meaning the typical pair is hurt and the aggregate gain rides on a handful of momentum-y winners.
- **Recommendation:** keep the production v7 rule. Do not adopt per-pair h* or the walk-forward adapter on this evidence (median-negative OOS, h* pinned at grid edges). The only candidates worth a longer-window / paper-live validation are (a) a modestly lower h_mid and (b) a small negative rv_pct tilt (rv_pct k=-0.2 improved both mean and median in-sample); both add residual delta, so size them against the sleeve's delta budget rather than on backtest CAGR alone.

## Files

- `b4_hstar_perpair_profiles.csv`, `b4_hstar_perpair_argmax.csv`, `b4_hstar_crosssection_corr.csv`
- `b4_hstar_refit_grid.csv`, `b4_hstar_signal_marginals.csv`
- `b4_hstar_walkforward_summary.csv`, `b4_hstar_walkforward_picks.csv`
- `b4_hstar_heatmap_pair_by_h.png`, `b4_hstar_scatter_drivers.png`, `b4_hstar_refit_heatmap.png`,
  `b4_hstar_wf_h_timeseries.png`, `b4_hstar_portfolio_equity.png`, `b4_hstar_signal_marginal_bars.png`
