# SPY30 Puts 2x
## Standalone strategy research summary and return-verification appendix

**Status:** Research variant. Not the current production primary.  
**Dashboard run:** `R_spy30_puts2x`  
**As-of data build:** 2026-07-14  
**Purpose:** Product-evaluation document, not an offer document or a claim of live investable performance.

---

## Executive summary

SPY30 Puts 2x is a proposed standalone positive-carry tail-insurance strategy. It combines a regime-managed short UVIX / short SVIX carry sleeve, a large idle-collateral allocation that is partly invested in SPY, and a systematic ladder of long SPX puts. Its intended return sources are: short-volatility ETP carry in favorable term-structure regimes, SPY return on a stated share of idle collateral, T-bill yield on the remaining idle collateral, realized gains from monetizing long puts in market stress, and reinvestment of harvested put cash.

The research run places 30% of **idle** collateral in SPY and doubles the production put ladder budget. It is therefore not simply a more aggressive version of the production insurance stack: it adds directional equity-market exposure to the idle collateral and buys materially more convexity. That combination raised the extended-sample return and reduced the modeled peak drawdown versus `B_extended`, but it also raised realized volatility and depends substantially on synthetic pre-inception UVIX/SVIX history and modeled option prices.

The likely buyer is an allocator seeking a systematic, liquid, equity-sensitive alternative-return allocation with explicit crash monetization rules, and who can tolerate substantial model, short-volatility, and equity-beta risk. It is not a capital-preservation product, a substitute for a cash reserve, or a fully validated market-neutral tail-risk fund.

### Dashboard headline metrics

| Metric | `R_spy30_puts2x` research run |
|---|---:|
| Sample | 2008-01-02 to 2026-07-14 |
| Trading days / rebalances | 4,644 / 807 |
| CAGR | 16.80% |
| Annualized volatility | 23.80% |
| Sharpe | 0.75 |
| Maximum drawdown | -39.35% |
| Calmar | 0.43 |
| Harvested put cash | $15.69m |
| Redeployment contribution at end | $3.44m |
| Synthetic UVIX/SVIX days | 3,569 |

These figures are from the dashboard model starting with $1.0m, so the dollar figures are model-path outputs rather than audited fund P&L.

---

## Strategy architecture

### 1. Collateral and exposures

The carry sleeve has a stated 20% equity gross allocation. The remainder is “idle” collateral after the dynamic carry gross is considered. In this research variant, 30% of that idle collateral earns SPY total-price return and the other 70% accrues the model’s T-bill rate. The configured T-bill rate is 4.30% annually.

The SPY allocation is not a hedge. It adds long equity beta precisely when the put ladder is expected to be out of the money most of the time. The strategy should therefore be evaluated as a hybrid of short-vol carry, long SPY exposure, cash collateral, and a systematic long-put program.

### 2. Short UVIX / short SVIX carry sleeve

The model shorts both volatility ETPs. It uses the VIX/VIX3M term-structure ratio as the regime signal and sizes SVIX short relative to UVIX short through `rho = SVIX-short notional / UVIX-short notional`.

| Regime rule from model config | Policy |
|---|---|
| Ratio at or below 0.88, deep contango | `rho = 1.0`; full carry sleeve gross |
| Ratio at or above 1.00, backwardation | `rho = 2.0`; carry sleeve gross cut to 35% of calm size |
| Between thresholds | Interpolated regime exposure |
| Rebalance cadence | 14 trading-day base clock, accelerated by the stress signal |

The economic rationale is that UVIX is a leveraged long-volatility ETP and may decay in contango, while the SVIX leg offsets part of the adverse volatility-spike exposure. This is a basis and path-dependent construction, not a reliable one-for-one hedge: either ETP can gap, trade dislocated from its indicative value, become difficult to borrow, or be subject to trading interruptions.

### 3. SPX put ladder: 2x research budget

The strategy buys approximately six-month SPX puts and rolls when approximately three months remain. It holds three out-of-the-money rungs: 10%, 20%, and 30% below spot.

| Put rung | `R_spy30_puts2x` budget per roll | Production `B_extended` budget per roll |
|---|---:|---:|
| 10% out of the money | 1.60% of equity | 0.80% of equity |
| 20% out of the money | 1.60% of equity | 0.80% of equity |
| 30% out of the money | 1.60% of equity | 0.80% of equity |
| Total stated ladder budget | 4.80% of equity | 2.40% of equity |

The dashboard’s dynamic hedge budget applies a 1.20x premium multiplier in contango and 0.85x in stress. That changes spending around the stated per-roll amounts. The production logic also limits actual contract counts to the available budget and uses whole contracts, so the realized purchase amount can differ from the percentage target.

### 4. Monetization and redeployment

The put program is designed to realize cash in a crisis rather than simply report a temporary mark-to-market gain.

* Profit tiers sell 34% of remaining exposure at 3x cost, 50% at 5x, and all remaining exposure at 8x.
* VIX overrides sell 50% at VIX 45 and all remaining exposure at VIX 65.
* After a put has reached at least 2x cost, the model exits if its mark falls 35% from its peak.
* On a full exit, 60% of proceeds are banked; the program re-arms a fresh put position.
* Realized proceeds are redeployed 20% to the carry sleeve in contango and 65% in backwardation, with the balance held in the T-bill collateral account.

These are research rules. They require executable option liquidity and disciplined order handling in real trading. A sudden V-shaped reversal can make tiered monetization look either too early or too late.

---

## Modeled performance and stress profile

### Extended sample results

The dashboard identifies this run as `pricing_mode = mixed`: the extended option path contains modeled Black-Scholes-with-skew prices where live option marks are not available and can use cached Theta EOD mids when available. The UVIX/SVIX panel is synthetic for 3,569 of 4,644 days.

| Measure | Research run |
|---|---:|
| Annualized carry-sleeve return | 1.57% |
| Carry-sleeve financing drag | -0.54% per year |
| August 2024 return | 1.53% |
| Modeled mild -20% crash payoff | 71.81% of equity |
| Modeled severe -30% crash payoff | 189.16% of equity |
| Modeled “volmageddon” -40% payoff | 369.42% of equity |

The crash figures are scenario outputs calculated from the ending configuration, not observed realized fund returns. They should be treated as sensitivity illustrations. They do not establish that the strategy would have captured those payouts after bid-ask spreads, execution latency, short-sale constraints, or a discontinuous ETP event.

### Risk discussion

The 39.35% modeled maximum drawdown confirms that this is not a low-volatility or capital-stable strategy. Important risks are:

1. **Short-volatility gap risk.** UVIX can gap higher and SVIX can behave differently from a simple inverse-volatility proxy.
2. **Equity beta.** The 30% SPY allocation to idle collateral is long equity exposure. It creates ordinary equity-market drawdown risk even before accounting for the carry sleeve.
3. **Put-basis and monetization risk.** SPX puts are not a perfect hedge for the ETP book. The monetization rules may miss a short-lived spike or sell before a continued decline.
4. **Borrow and financing risk.** Modeled annual borrow assumptions are 2.84% for UVIX and 3.47% for SVIX. Actual borrow, locate availability, recalls, margin, and short proceeds can be materially worse.
5. **Option-model risk.** Pre-live and cache-missing option marks use Black-Scholes with VIX as an ATM-vol proxy plus a fixed skew adjustment. This is not a full historical option surface or executable quote simulation.
6. **Synthetic-history risk.** Most of the extended sample precedes the live UVIX/SVIX overlap used by the product. Synthetic history is useful for hypothesis generation and stress testing, not for a performance claim.

---

## Comparison with production `B_extended`

Both rows below are the dashboard’s extended period, 2008-01-02 to 2026-07-14. They share the 20% dual-short carry sleeve, term-structure regime policy, borrow assumptions, adaptive cadence, and base monetization/redeployment rules.

| Dimension | `R_spy30_puts2x` | `B_extended` production primary |
|---|---|---|
| Classification | Research variant | Current primary research product run |
| Idle collateral | 30% of idle collateral in SPY; remainder in T-bills | T-bills |
| Put ladder budget | 4.80% of equity per roll before dynamic multiplier | 2.40% per roll before dynamic multiplier |
| Option pricing label | Mixed | Black-Scholes with skew |
| CAGR | 16.80% | 12.79% |
| Volatility | 23.80% | 19.15% |
| Sharpe | 0.75 | 0.72 |
| Maximum drawdown | -39.35% | -41.61% |
| Calmar | 0.43 | 0.31 |
| Harvested put cash | $15.69m | $7.89m |
| Model severe -30% crash payoff | 189.16% | 81.21% |
| Synthetic UVIX/SVIX days | 3,569 | 3,569 |

The comparison is descriptive, not proof that the research variant is superior. The run changes two economically meaningful variables at the same time: equity-beta exposure through SPY and the amount spent on puts. A proper investment decision needs attribution and out-of-sample testing of each change separately.

### Relationship to the live GTP book

The dashboard explicitly states that the live GTP book holds only a small volatility-ETP sleeve of approximately 0.25% of gross as a placeholder risk budget. It does **not** run the full dual-short-plus-put product represented here. This document therefore presents SPY30 Puts 2x as a separate research strategy/fund concept, not as a sleeve nested inside the GTP book.

---

## Classification and caveats

**Classification:** Research variant; standalone product concept; not currently the production primary.

**What is a fact in this document:** the run identifier, source code behavior, config values, dashboard statistics, and limitations described in the cited repository artifacts.

**What remains a research assumption:** that synthetic UVIX/SVIX history is representative; that the fixed borrow, financing, slippage, and commission inputs are attainable; that Black-Scholes-with-skew or sparse Theta mids proxy executable option marks; and that the stated monetization/redeployment rules can be followed at the modeled prices and timings.

Do not market the extended results as live, audited, or investable performance. The most defensible evidence available today is the no-synthetic-instrument `B_live` window, which is still a model-backed reconstruction rather than a live funded track record.

---

# Appendix A: Return verification plan for the B strategy family

## Objective

The question is not whether the backtest can produce a high return. The question is whether the **magnitude** of the modeled return survives independently verifiable prices, realistic execution, and conservative financing assumptions. The most credible process separates the return engines before recombining them.

## Required split verification

Run every verification test in three independent books, with identical dates, capital conventions, and daily reconciliation:

1. **Short UVIX/SVIX carry sleeve alone.** Include each leg’s price P&L, borrow, short-proceeds treatment, commissions, slippage, margin/financing, rebalances, and all forced exits.
2. **SPX put ladder alone.** Include premium paid, option MTM, sales from every monetization rule, roll proceeds, contract rounding, bid/ask or conservative execution price, and remaining inventory.
3. **Combined portfolio.** Add actual cash flows from the two sleeves, T-bill/financing, SPY collateral where applicable, and redeployment. Reconcile combined NAV exactly to the sum of component NAVs and cash.

The combined result should never be the first or only report. A high combined CAGR can conceal a carry assumption, option-pricing convention, or cash-accounting error.

## Most realistic verification methods, ranked

| Rank | Method | Evidence produced | Main limitation |
|---:|---|---|---|
| 1 | Live or paper trading with broker order/fill, locate, borrow, and margin records | Actual fill prices, availability, borrow changes, financing, rejected orders, and operational timing | Takes time; paper fills can still be optimistic |
| 2 | Historical replay using actual SPX option marks from Theta/OPRA-quality data | Contract-level mid/bid/ask comparisons, real strike/expiry selection, roll and event-day MTM | Cache currently has finite and incomplete coverage; EOD data is not fill data |
| 3 | Instrument-live window only: `B_live` after UVIX/SVIX listing | Removes synthetic UVIX/SVIX instrument history; dashboard labels this window `theta_mid` and reports 0 synthetic days | Still model-backed and a short sample |
| 4 | Event-day comparison of modeled versus actual SPX option marks | Direct error distribution during stress, where the hedge matters most | Does not validate continuous portfolio execution |
| 5 | Borrow/financing replay using historical locate and broker data | Realistic net carry after borrow spikes, availability, recalls, and margin | Historical locate series may be unavailable; must not fill missing data with a benign constant |

### A. Live / paper implementation

Use a dedicated, small notional account or broker paper environment configured to preserve every order and reject. Each trading day archive:

* target notional, actual order, fill, partial fill, cancellation, and timestamp for UVIX and SVIX;
* locate quantity, borrow annual rate, recall/availability flags, margin requirement, and short-credit treatment;
* exact SPX option contract, bid, ask, mid, fill, commission/exchange fee, and remaining inventory;
* the model’s signal time and the actual order-release time;
* daily NAV and P&L attribution that ties to broker records.

Evaluate the model-versus-execution wedge by sleeve, not only in total. A live pilot should be considered a gating exercise, not an optimization loop.

### B. Option-mark replay

The repository already has a Theta cache under `data/cache/spx_options/theta/`. The helper `scripts/bucket5_theta.py` reads cached EOD option data and uses bid/ask mid where both exist, otherwise close. The current cache audit found 95 nonempty parquet files, with observations spanning 2022-03-25 through 2026-03-19. That is usable as a starting validation sample but insufficient as a full 2008-to-present history.

For each historical roll:

1. Reconstruct the exact contract selected by the strategy, including target DTE, strike rounding, and actual expiry.
2. Price entry at ask or a stressed ask-side execution, not mid.
3. Price exit/monetization at bid or a stressed bid-side execution, not mid.
4. Mark open positions at conservative mids only where both quotes are valid; flag stale, crossed, zero-volume, and missing quotes.
5. Compare the model’s Black-Scholes/skew price with the observed mark on every available day and separately on crash days.
6. Report mean/median error and tail error by DTE, moneyness, VIX regime, and event window.

This is the most realistic way to validate the SPX-put magnitude before live trading. The current code falls back to Black-Scholes with VIX/100 as the ATM-volatility proxy plus a fixed skew bump whenever a Theta mark is unavailable. That fallback should be measured, not assumed harmless.

### C. UVIX/SVIX carry verification

Treat short UVIX and short SVIX as separate executable instruments. Rebuild their P&L from unadjusted tradable closes or, preferably, NBBO/VWAP-like data around the actual rebalance time. Apply:

* independently sourced borrow fees by day and security;
* actual locate availability and zeroing of new short exposure on no-locate days;
* recall/forced-cover scenarios;
* short proceeds and margin financing treatment that matches the broker;
* 5 bp modeled slippage plus a sensitivity grid such as 5/15/30/50 bp;
* commissions, corporate actions, splits, halts, reverse splits, and ETP termination rules.

Run this only over the instrument-live window first. `B_live` begins 2022-03-30, has 1,075 days, 0 synthetic days, and dashboard results of 7.71% CAGR, 1.19 Sharpe, -5.26% maximum drawdown, and $48,336 harvested put cash. This is a more relevant validation benchmark than the extended synthetic period, although it remains short and not live P&L.

## Specific tests and reporting metrics

### Daily reconciliation and attribution

* Reconcile NAV daily: prior NAV + UVIX P&L + SVIX P&L + SPY P&L + T-bill/interest + option MTM change + option cash flows - borrow - financing - fees = ending NAV.
* Report gross and net P&L by UVIX, SVIX, SPY collateral, T-bills, put premium, put MTM, monetization cash, and redeployment.
* Tie every realized put cash flow to a contract, event rule, quantity, price, and source quote.
* Separate realized put cash from unrealized put MTM. Do not call a mark “harvested cash.”

### Stress and crash windows

Create event packets for at least the COVID selloff, the 2022 bear market, August 2024, and every available 2022-to-2026 VIX spike. For each packet, show the daily path of:

* UVIX and SVIX exposure, price P&L, borrow, and forced de-risking;
* each put contract’s bid, ask, modeled mark, actual mark, multiple of cost, and monetization event;
* combined NAV, peak-to-trough drawdown, net beta, and cash realized;
* model-versus-realized or model-versus-quote differences.

### Sensitivities and acceptance gates

Run a predeclared grid, not a single favorable point:

* option execution: mid, bid/ask, and stressed bid/ask;
* UVIX/SVIX slippage: 5, 15, 30, and 50 basis points;
* borrow: actual history where available, then 1.5x and 2.0x stressed scenarios;
* no-locate/recall: zero new shorts or forced covers on the documented dates;
* financing: broker actuals, then conservative spread stress;
* puts: observed Theta marks only versus model fallback only versus hybrid;
* monetization: same rules, but executions delayed one session and one intraday interval where data allows.

Report CAGR, volatility, Sharpe, maximum drawdown, Calmar, turnover, total borrow/financing, option premium paid, realized put cash, residual put MTM, and the proportion of return attributable to each sleeve.

## What would falsify the modeled magnitude

The magnitude is not supported if one or more of the following occurs:

1. Actual bid/ask-based SPX replay materially reduces put monetization cash or produces persistent negative mark error in the stress windows that drive the result.
2. The UVIX/SVIX sleeve becomes uneconomic after actual borrow, locates, recalls, financing, and conservative execution costs.
3. The performance is concentrated in synthetic pre-inception years and does not remain economically meaningful in the `B_live` instrument window.
4. Component-level reconciliation does not tie: combined NAV differs from the exact sum of cash and sleeve P&L, or harvested cash cannot be traced to contract sales.
5. Small changes in slippage, borrow, option execution side, or rebalancing delay erase the strategy’s return advantage.
6. Stress packets show the puts fail to offset the carry sleeve when actual instrument prices and option marks are used.

No single pass establishes truth. The decision should depend on a documented error budget: how much of the extended return survives after each realism adjustment, and whether the surviving live-era and paper/live results meet the allocator’s required return and drawdown threshold.

## Existing repository evidence to use

* `docs/production_actual_realism_plan.md` documents price-integrity, cadence, delisting, borrow, margin, locate, and Flex-reconciliation gaps for the broader production replay. Its stated residual gaps include the Flex P&L sample gate, point-in-time financing curve, and live locate rejects.
* `notebooks/output/production_actual_bt/replay/REPORT.md` demonstrates the production-replay reporting convention: archived plans with one-session execution lag, cost and financing assumptions, and explicit limitations. It also states that broker sequencing is not replicated.
* `tests/test_bucket5_product_export.py`, `tests/test_bucket5_backtest_api.py`, and Bucket-5-related production tests provide regression coverage for export/schema behavior and sleeve plumbing. They are software checks, not economic validation.
* `scripts/bucket5_theta.py` and `scripts/bucket5_put_overlay.py` disclose the actual option-price hierarchy: cached Theta EOD marks first where available, then Black-Scholes with VIX-derived implied volatility and a fixed skew adjustment.

---

## Source record

* Dashboard run metrics and scenario outputs: `risk_dashboard/data/bucket5_product.json`, run `R_spy30_puts2x` (generated 2026-07-15; run date 2026-07-14).
* Product construction and run IDs: `scripts/build_bucket5_product_dashboard.py`.
* Dashboard strategy guide and exported assumptions: `scripts/bucket5_product_export.py`.
* SPY-idle and put-budget mechanics: `scripts/bucket5_spy_put_grid_bt.py`.
* Carry, ladder, monetization, redeployment, and scenario mechanics: `scripts/bucket5_insurance_bt.py`.
* Theta cache and fallback hierarchy: `scripts/bucket5_theta.py`, `scripts/bucket5_put_overlay.py`.
* Broader production realism plan: `docs/production_actual_realism_plan.md`.
* Production replay limitations: `notebooks/output/production_actual_bt/replay/REPORT.md`.
