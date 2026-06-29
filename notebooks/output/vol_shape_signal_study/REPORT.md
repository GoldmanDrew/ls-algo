# Vol-shape trend-ratio signal-quality study (Section A)

- Underlyings: **180**, pooled obs: **195,868**, window: 2021-06..2026 (per-name history).
- IC = Spearman rank correlation. `xs` = cross-sectional (per date, across names; this is the form the sizing tilt consumes). `ts` = time-series (per name, across time; the form the cadence engine consumes).

## A2 - Does forward `tr_est` predict better than raw TR?

| target | signal | pooled IC | xs IC (t) | ts IC (t) |
|---|---|---:|---:|---:|
| fwd_tr60 | tr_raw60 | -0.0074 | -0.0028 (-0.9) | -0.0633 (-6.3) |
| fwd_tr60 | tr_est60 | -0.0151 | -0.0157 (-4.5) | -0.0602 (-6.2) |
| fwd_tr60 | vcr60 | +0.0138 | +0.0141 (+4.9) | +0.0074 (+0.7) |
| fwd_tr60 | eff_z | -0.0075 | -0.0082 (-2.4) | -0.0318 (-3.3) |
| fwd_tr60 | cons_z | -0.0251 | -0.0305 (-9.9) | -0.0362 (-4.0) |
| fwd_tr60 | r2_z | -0.0043 | -0.0091 (-2.6) | -0.0147 (-1.6) |
| fwd_tr60 | persistence_z | -0.0145 | -0.0184 (-6.4) | -0.0173 (-2.3) |
| fwd_tr20 | tr_raw60 | -0.0162 | -0.0087 (-2.8) | -0.0510 (-7.8) |
| fwd_tr20 | tr_est60 | -0.0239 | -0.0174 (-5.2) | -0.0530 (-7.3) |
| fwd_tr20 | vcr60 | +0.0131 | +0.0161 (+5.9) | +0.0105 (+1.5) |
| fwd_tr20 | eff_z | -0.0143 | -0.0062 (-1.8) | -0.0304 (-4.3) |
| fwd_tr20 | cons_z | -0.0211 | -0.0211 (-7.3) | -0.0358 (-4.7) |
| fwd_tr20 | r2_z | -0.0118 | -0.0095 (-2.8) | -0.0189 (-2.2) |
| fwd_tr20 | persistence_z | -0.0176 | -0.0152 (-5.1) | -0.0199 (-4.6) |
| fwd_rv60 | tr_raw60 | +0.0372 | +0.0443 (+11.0) | +0.0628 (+5.1) |
| fwd_rv60 | tr_est60 | +0.0527 | +0.0559 (+11.9) | +0.0905 (+6.4) |
| fwd_rv60 | vcr60 | -0.0405 | -0.0314 (-8.3) | -0.0688 (-3.8) |
| fwd_rv60 | eff_z | +0.0390 | +0.0405 (+8.1) | +0.0822 (+5.5) |
| fwd_rv60 | cons_z | +0.0471 | +0.0488 (+9.8) | +0.0367 (+2.4) |
| fwd_rv60 | r2_z | +0.0154 | +0.0175 (+3.6) | +0.0427 (+3.1) |
| fwd_rv60 | persistence_z | +0.0122 | +0.0111 (+2.7) | +0.0380 (+3.3) |

## A1 - Horizon blend  w*tr_est20 + (1-w)*tr_est60  vs fwd_tr60

| w20 | pooled IC | xs IC (t) | ts IC (t) |
|---:|---:|---:|---:|
| 0.0 | -0.0151 | -0.0157 (-4.5) | -0.0602 (-6.2) |
| 0.1 | -0.0167 | -0.0170 (-4.9) | -0.0634 (-6.6) |
| 0.2 | -0.0183 | -0.0178 (-5.1) | -0.0665 (-7.0) |
| 0.3 | -0.0197 | -0.0183 (-5.3) | -0.0680 (-7.3) |
| 0.4 | -0.0209 | -0.0187 (-5.5) | -0.0681 (-7.6) |
| 0.5 | -0.0216 | -0.0186 (-5.6) | -0.0666 (-7.8) |
| 0.6 | -0.0217 | -0.0181 (-5.6) | -0.0636 (-7.8) |
| 0.7 | -0.0214 | -0.0174 (-5.4) | -0.0593 (-7.8) |
| 0.8 | -0.0208 | -0.0165 (-5.3) | -0.0547 (-7.6) |
| 0.9 | -0.0200 | -0.0153 (-5.0) | -0.0498 (-7.4) |
| 1.0 | -0.0192 | -0.0144 (-4.7) | -0.0449 (-7.0) |

(`a1_horizon_blend_tr_raw_fwd_tr60.csv` has the raw-TR blend for comparison.)

## A3 - Re-fit evidence weights (walk-forward, target fwd_tr60)

- Out-of-sample mean xs IC, production linear weights: **-0.0299**
- Out-of-sample mean xs IC, refit weights: **+0.0138**
- Out-of-sample mean xs IC, full production tr_est: **-0.0298**

Suggested weights (full-sample fit, L1-normalized to production scale):

| component | prod | refit (norm) |
|---|---:|---:|
| tr_z | +0.40 | +0.013 |
| eff_z | +0.22 | +0.221 |
| cons_z | +0.20 | -0.957 |
| r2_z | +0.20 | +0.016 |
| persistence_z | +0.22 | -0.134 |
| jump_penalty | -0.35 | +0.248 |

## Decay-capture target (production joint-metrics basis)

Sign note: a *negative* IC supports the thesis (lower TR -> more forward decay).

| signal | pooled IC | xs IC (t) | ts IC (t) |
|---|---:|---:|---:|
| tr_raw60 | +0.0079 | +0.0016 (+0.5) | +0.0102 (+0.7) |
| tr_est60 | -0.0164 | +0.0002 (+0.1) | -0.0066 (-0.4) |
