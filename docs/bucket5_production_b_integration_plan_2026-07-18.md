# Bucket 5 Production B — real integration and controlled rollout plan

**Date:** 2026-07-18

**Status:** Proposed implementation and capital-governance plan

**Scope:** Integrate the `B_production` dual-short volatility-ETP plus long-put insurance strategy into `ls-algo` without treating the existing Bucket 5 placeholder as if it were already the full product.

## Approved implementation amendment — 2026-07-20

This amendment supersedes the paper-account path and the separate B5 executor design
elsewhere in this document.

### Real account only (no paper trading)

Do **not** use an IBKR paper account for B5. All shadow, pilot, and production work runs
against the live account. Paper fills do not validate borrow, locates, or option liquidity
anyway; the compensating controls are:

1. **Longer shadow on live state** before any order is allowed (≥40 trading days of
   target generation + hypothetical fills using live quotes; zero submissions).
2. **`--dry-run` must succeed end-to-end** through `generate_trade_plan` →
   `rebalance_strategy` for both carry and option intents before the first live order.
3. **Manual approval required** for every risk-increasing B5 order (new shorts, put buys,
   re-arms) until automation is explicitly enabled. Sell-only monetization and covers may
   automate later under the same kill modes.
4. **One-contract put pilot first**, then staged ladder growth. Cap carry at or below the
   current live gross during the pilot so the first live action cannot enlarge short-vol
   risk.
5. **Fail closed on dual ownership, stale quotes, missing locates, or reconciliation
   failure** — health drops to `no_new_risk` / `halt_all` rather than falling back to a
   model price or a second owner.

Remove `mode: paper` from the operational mode set. Allowed modes are:

```yaml
bucket5_production:
  mode: placeholder | shadow | production
  account: live   # fixed; paper path is not implemented
```

| Mode | Behavior |
|---|---|
| `placeholder` | Current GTP + rebalancer own UVIX/SVIX stocks only; no B5 option intents |
| `shadow` | GTP emits B5 carry + option targets into `proposed_trades` / B5 manifest; rebalancer runs with `--dry-run` only; no live submits |
| `production` | Same path submits live orders under `orderRef` namespace `B5P|…` (options) and existing `ETF_LS|…` carry tags, subject to manual-approval and kill modes |

### Wire into `generate_trade_plan` and `rebalance_strategy`

Do **not** build a separate `bucket5_execute.py` as the primary order path. B5 Production B
is an extension of the existing daily plan → rebalance pipeline:

| Layer | Responsibility |
|---|---|
| `scripts/bucket5_policy.py` | Pure regime / ladder / monetization / redeployment rules (shared with research) |
| `scripts/bucket5_ledger.py` | Event-sourced option lots (`conId`), tiers fired, peak marks, rolls |
| `generate_trade_plan.py` | Owns B5 targets: sizes UVIX/SVIX carry from Production B policy; selects exact XSP/SPX put contracts; writes option rows + carry rows into the plan artifacts |
| `rebalance_strategy.py` (+ `execute_trade_plan.py`) | Executes plan rows: stocks via existing `make_stock` path; options via a new `make_option` / OPT limit-order path gated by quote quality and `orderRef` |
| `ibkr_accounting.py` + `bucket5_reconcile` | Contract-safe Flex attribution and daily B5 NAV identity |
| `config/bucket5_production.yml` | Frozen economics, ramp, kill modes, automation flags |

Plan artifact rules:

- Carry legs remain on the `UVIX` / `SVIX` pair row in `proposed_trades.csv` with
  `sleeve=volatility_etp_bucket5`.
- Option intents are first-class plan rows (or a sibling `proposed_trades_b5_options.csv`
  loaded by the same rebalancer) with fields: `conId`, expiry, strike, right,
  tradingClass, multiplier, action, qty, limit_px, rung_id, intent_id, `order_ref_prefix=B5P`.
- The rebalancer must **ignore B5 option rows unless** `bucket5_production.mode` is
  `shadow` (dry-run) or `production` (live), and must refuse option submits when mode is
  `placeholder`.
- When mode is `production`, GTP is the sole owner of UVIX/SVIX **and** B5 options. The
  generic B1/B2/B4 phases must not resize those symbols. Fail closed if a second owner is
  detected.
- Cancel hygiene: stock cancels stay scoped to `ETF_LS|…`; option cancels stay scoped to
  `B5P|…`. Never cancel 0DTE or other strategy orders.

0DTE isolation is unchanged and mandatory: prefer **XSP** for B5 puts; require `B5P|`
  orderRef; patch spx-0dte to same-day SPXW only so long-dated puts are never treated as
  0DTE inventory.

### Revised rollout (real account)

| Stage | What runs | Live orders? |
|---|---|---|
| Phase 0–1 | Charter + build policy/ledger + GTP/rebalancer option path | No |
| Phase 2 Shadow | Live account state + live quotes; GTP writes targets; rebalancer `--dry-run` only | No (≥40 days) |
| Phase 3 *(replaces paper)* | Live dry-run drills: restart, partial-fill simulation, monetization injection, kill modes on the real code path | No |
| Phase 4 Live put pilot | Manual release, one XSP (or SPX) contract → staged ladder; carry reduce-only / no-grow | Yes (puts first) |
| Phase 5 Carry cutover | GTP Production B owns UVIX/SVIX under no-grow ceiling | Yes |
| Phase 6 Automate | Same automation ladder as before, still on the live account | Yes |

The old Phase 3 “IBKR paper account, 63–90 days” is **deleted**. Its mechanical goals move
into Phase 2/3 dry-run drills on the live code path. Capital and put-sizing gates from the
2026-07-18 amendment still apply.

---

## Approved implementation amendment — 2026-07-18

This section supersedes the older sizing examples below. The approved book is now based on
**$1.2 million NAV** and **$4.8 million target gross**:

| Sleeve | Production budget | Rule |
|---|---:|---|
| B1 | $2,737,000 | residual; this is the only sleeve increased by the NAV change |
| B2 | $1,900,000 | fixed-dollar ceiling, rounded down |
| B4 | $115,000 | fixed-dollar ceiling, rounded down; excludes B5 |
| B5 | $48,000 | independent 1% of total target gross; true two-leg pair gross |

For B5, `gross_target_usd` means `abs(vol-ETP short) + abs(underlying short)`. With hedge
beta `b`, the leg solver is:

```text
vol_ETP_short = B5_pair_gross / (1 + b)
underlying_short = b × vol_ETP_short
```

Production B's calm carry sleeve is 20% of its effective capital, so the $48,000 pair-gross
ceiling maps to **$240,000 effective B5 NAV**. Put quantities must be sized from this effective
B5 NAV—not from the full $1.2 million account—so the hedge cannot silently become five times
larger than the carry sleeve. The approved ladder doubles each rung's existing integer quantity
exactly; 4.8% is the nominal doubled budget, while executable premium determines actual cost:

```text
baseline roll budget = $240,000 × 2.4% = $5,760
baseline contracts_rung = max(1, floor(0.8% × $240,000 × regime multiplier / (ask × 100)))
target contracts_rung = 2 × baseline contracts_rung
nominal doubled roll budget = $11,520 before the dynamic regime multiplier
```

The dashboard's modeled contract count is a planning estimate only. Immediately before an
order, recompute with executable asks, reject stale/wide markets, and never round up beyond the
premium budget.

### Slower production ramp and learning loop

1. **Two weeks shadow:** calculate B5 legs and SPX quantities daily; place nothing. Require
   stable identifiers, no interaction with 0DTE orders, and complete reconciliation.
2. **Two weeks one-contract pilot:** permit at most one contract in the cheapest eligible rung;
   carry stays at or below 25% of the $48,000 ceiling. Manual approval is mandatory.
3. **Two weeks 50% ladder / 50% carry:** cap pair gross at $24,000 and each rung at half of its
   solved quantity. Advance only with clean fills, settlement, and risk attribution.
4. **Four weeks full puts / 50% carry:** establish the complete 2× put ladder before allowing
   the short-vol carry sleeve to reach its full ceiling.
5. **Full $48,000 carry ceiling:** enable only after eight clean weeks, zero 0DTE namespace
   collisions, premium slippage inside limits, and stress loss within the approved budget.

Every Friday, compare modeled versus executable option cost, realized hedge beta, carry P&L,
put P&L, total B5 drawdown, and 0DTE P&L independently. Change only one control per review cycle;
hold or step down automatically on reconciliation failures, stale quotes, missing inventory,
margin breach, or unexplained cross-strategy P&L.

### New backtest evidence: 2× quantity is implemented, but not approved for full live rollout

The rebuilt single full-history run (2008-01-02 through 2026-07-17) produced 14.77% CAGR,
0.60 Sharpe, and **-64.54% maximum drawdown** under the exact 2× integer-quantity rule. This is
materially worse than the immediately preceding doubled-budget interpretation (-40.19% maximum
drawdown). The current modeled snapshot also requires about **$36,459** of premium for six SPX
contracts versus a regime-adjusted nominal doubled budget of about **$13,524**, mainly because
the 10% OTM SPX contract is indivisible and expensive at this B5 scale.

Therefore, the code and dashboard should retain the requested six-contract research target, but
the live gate is **blocked above the one-contract pilot** until at least one of these is approved:

- use XSP or another smaller multiplier for granular sizing;
- omit a rung whose minimum one-contract cost breaches its rung budget;
- increase effective B5 capital without increasing B5 risk concentration; or
- replace “exactly 2× contracts” with “up to 2×, subject to premium and stress-loss caps.”

No automated SPX order should be routed from this change. The pilot must use a separate B5 order
namespace, strategy-owned inventory ledger, and order reference; the 0DTE engine must filter its
positions, P&L, Greeks, open orders, and flatten logic to its own namespace and same-day expiries.
Account-level margin remains shared and must be monitored globally, but B5 long puts must never
be treated as 0DTE inventory or available contracts for 0DTE closing orders.

## Executive decision

Build Production B as a **ring-fenced, stateful Bucket 5 sub-book** that reports into the L/S account and dashboard. Do not implement it by changing `volatility_etp_bucket5.target_weight` from `0.0025` to a larger number.

The full product is not one stock pair. It has:

- a dynamic short-UVIX / short-SVIX carry book;
- a three-contract-family SPX put inventory with rolls and partial monetization;
- cash, premium, settlement, and redeployment state;
- two independent borrow/locate paths;
- its own NAV and collateral conventions; and
- risk that must be aggregated with, but not confused with, the rest of the four-times-gross L/S book.

The rollout should proceed in this order:

1. **Freeze and harden the economic specification.**
2. **Run it at full proposed notional in shadow and paper.**
3. **Add live put coverage before increasing short-vol carry risk.**
4. **Migrate the existing carry placeholder into the dynamic Production B policy under a no-grow ceiling.**
5. **Ramp operational automation first, then capital.**
6. **Treat every later capital increase as a separate allocator decision.**

The initial production ceiling should be sized from the lower current planned exposure, not from the backtest's $1 million example. On 2026-07-18, the proposed B5 row calls for approximately $10,500 short UVIX and $20,875 short SVIX, or **$31,375 two-leg gross**. At Production B's 20% calm carry allocation, that maps to an initial B5 sleeve NAV of approximately:

```text
A_B5_initial = $31,375 / 20% = $156,875 ≈ 14.9% of the $1.05m account NAV
```

This is deliberately below the 2026-07-17 accounting gross of approximately $41,024. It makes the first live migration a trim, not a size increase.

At that initial sleeve NAV:

| Item | Initial ceiling |
|---|---:|
| Calm carry gross | $31,375 |
| Backwardation carry gross at the 35% policy floor | $10,981 |
| Base put premium per roll, 2.4% of B5 NAV | $3,765 |
| Dynamic total roll budget, 0.85x–1.32x | approximately $3,200–$4,970 |
| Dynamic budget per equal rung | approximately $1,067–$1,657 |

“100% rolled out” must initially mean **100% of this approved $156,875 B5 allocation**, not applying Production B to the full $1.05 million account. Increasing the approved B5 allocation is a later decision.

## 1. What exists today versus what Production B requires

### 1.1 Current live placeholder

The repository currently:

- allocates Bucket 5 `0.25%` of target L/S gross in [`config/strategy_config.yml`](../config/strategy_config.yml);
- emits one `UVIX | SVIX` row through [`generate_trade_plan.py`](../generate_trade_plan.py);
- trades both instruments as stocks through the general rebalancer and stock-only contract helper in [`execute_trade_plan.py`](../execute_trade_plan.py);
- attributes the two stock legs to Bucket 5 in [`ibkr_accounting.py`](../ibkr_accounting.py); and
- displays the full insurance product as research while explicitly labeling the live GTP book as only a small placeholder.

The latest evidence inspected for this plan is:

| Evidence | Current value |
|---|---:|
| Configured account NAV | $1,050,000 |
| Configured target gross | $4,200,000 |
| B5 pair anchor in the 2026-07-18 plan | $10,500 |
| Planned UVIX / SVIX leg notionals | approximately $10,500 / $20,875 short |
| Planned two-leg gross | approximately $31,375 |
| 2026-07-17 B5 accounting gross / net | approximately $41,024 / -$462 |
| 2026-07-17 B5 cumulative accounting P&L | approximately $358 |

This live history is useful execution evidence for the two ETPs, but it is **not** a live track record for Production B because there is no live put ladder, no Production B regime sizing, and no Production B cash ledger.

### 1.2 Production B economic specification to preserve

The source of truth is `production_config()` in [`scripts/bucket5_insurance_bt.py`](../scripts/bucket5_insurance_bt.py), selected by `B_production` in [`config/bucket5_presets.yml`](../config/bucket5_presets.yml).

| Policy | Production B v1 |
|---|---|
| B5 carry gross | 20% of **B5 allocated NAV** in deep contango |
| Regime signal | VIX / VIX3M |
| Deep contango | ratio at or below 0.88; rho = 1.0; 100% of carry gross |
| Backwardation | ratio at or above 1.00; rho = 2.0; 35% of carry gross |
| Between thresholds | linear interpolation |
| Carry rebalance cadence | 14 trading-day base, accelerating toward 2 days in stress |
| Put ladder | approximately six months to expiry; roll near three months |
| Put rungs | 10%, 20%, and 30% out of the money |
| Base premium per roll | 0.8% of B5 NAV per rung; 2.4% total |
| Dynamic premium multiplier | 1.20x in contango, 0.85x in stress, with up to a 1.10x low-VIX boost |
| Profit monetization | sell 34% at 3x, 50% of remaining at 5x, all remaining at 8x |
| VIX monetization | sell 50% at VIX 45; all remaining at VIX 65 |
| Giveback rule | after at least 2x, sell on a 35% decline from peak |
| Full-exit policy | bank 60%; re-arm fresh puts |
| Redeployment | 20% to carry in contango; 65% in backwardation; balance to bills |

The live engine must consume these same pure policy functions. It must not carry a second hand-copied implementation whose behavior can drift from the backtest.

### 1.3 Evidence that prevents an immediate config flip

1. **The live executor is stock-oriented.** `make_stock()` creates `Stock` contracts and the normal plan has no exact option contract, expiry, strike, right, multiplier, or `conId`.
2. **Existing accounting is not contract-safe for options.** Flex parsers aggregate primarily by `symbol`; multiple SPX expiries and strikes cannot be safely represented that way. Option accounting must key on broker `conId` plus expiry, strike, right, trading class, and multiplier.
3. **Research marks are not executable marks.** The current return-verification artifact has 402 Theta observations, a mean model-minus-quote error of 26.88 index points, and a 95th-percentile absolute error of 131.06. It correctly raises a soft tail-error flag.
4. **Important replay controls are still unsupported.** Bid/ask-only execution, forced Theta-only versus model-only modes, delayed monetization, no-locate days, and forced-cover recalls are listed as future gates in [`data/runs/2026-07-18/b5_return_verify/REPORT.md`](../data/runs/2026-07-18/b5_return_verify/REPORT.md).
5. **Borrow assumptions do not currently match the live row.** Production B assumes 2.84% annual UVIX borrow and 3.47% SVIX borrow. The 2026-07-18 proposed row shows approximately 2.17% for UVIX and 12.35% for the SVIX underlying. The live SVIX observation is about 3.6 times the research assumption.
6. **Most extended history is synthetic.** `B_extended` uses synthetic UVIX/SVIX data on 3,569 of 4,644 days. The 1,075-day `B_live` window has no synthetic ETP days and is the relevant economic starting point, but it remains reconstructed rather than broker-realized.
7. **The collateral assumption conflicts with a four-times-gross account unless it is explicitly reserved.** Production B earns bill yield on most B5 capital. The L/S account already uses capital and short proceeds to support a large book. Bill yield cannot be credited to both books, and B5 premium cash cannot be assumed free.

## 2. Capital and accounting model

### 2.1 Define a real B5 sleeve NAV

Introduce three separate quantities:

```text
account_nav       = broker account net liquidation value
b5_allocated_nav  = capital specifically approved and reserved for Bucket 5
ramp_factor       = 0.0 to 1.0 within that approved allocation
```

Every Production B target is based on `effective_b5_nav = b5_allocated_nav × ramp_factor`, not on total account NAV and not on total L/S gross.

For a given effective B5 NAV:

```text
carry_gross = 20% × effective_b5_nav × regime_gross_multiplier
uvix_short  = carry_gross / (1 + rho)
svix_short  = carry_gross - uvix_short
roll_budget = 2.4% × effective_b5_nav × hedge_budget_multiplier
```

The planner must round down to whole shares and whole option contracts without exceeding either the rung budget or the total premium budget. Unspent rounding cash remains B5 cash; it is not redistributed into more carry.

### 2.2 Reserve collateral without double counting

Preferred implementation:

- use an IBKR paper account during testing and a dedicated production subaccount or separately tracked allocation for live B5;
- hold the intended collateral in actual eligible bills/cash where operationally appropriate; and
- allocate the observed broker interest, financing, and margin effects to B5.

If the same account must be used, maintain a strict virtual capital ledger and report two return views:

1. **Incremental B5 return:** actual ETP P&L, borrow, option P&L/cash flows, fees, and incremental financing. No assumed bill yield.
2. **Allocated-capital B5 return:** the same result plus only the bill/interest income that can be directly attributed to the reserved B5 collateral.

Do not compare live results with `B_live` or `B_extended` using an assumed 4.3% bill yield unless that yield was actually earned on separately identified B5 collateral.

### 2.3 Initial capital ceiling and risk acceptance

Use approximately **$156,875** as the first proposed ceiling because it maps Production B's calm carry gross to the lower current planned two-leg gross. Final approval should use the actual cutover-day broker positions and NAV.

Before approving this ceiling, explicitly accept these approximate, pre-put risk references:

- an instantaneous 80% volatility-futures shock from a calm `rho = 1` state produces roughly a 40% loss on the carry gross, or about **8% of B5 allocated NAV** before option benefit;
- at the proposed $156,875 B5 allocation, that is about **$12,550**, or 1.2% of current account NAV, before any put offset; and
- the reconstructed extended Production B run has a **-41.6% maximum drawdown** on B5 capital. Scaled to a 14.9% account allocation, that is approximately a 6.2% account-NAV contribution if the relationship were linear.

These are not forecasts. They are sizing constraints. The allocator should set a maximum B5 contribution to account drawdown and gap loss, then solve the maximum allowed `b5_allocated_nav`. A preset label must not determine account risk.

## 3. Target production architecture

### 3.1 Keep exactly one owner of UVIX and SVIX

Add an explicit mode (see **2026-07-20 amendment** — paper mode removed):

```yaml
bucket5_production:
  mode: placeholder | shadow | production
  account: live
```

- `placeholder`: current `generate_trade_plan` and rebalancer own UVIX/SVIX stocks only.
- `shadow`: GTP writes B5 carry + option targets; rebalancer may only `--dry-run` those intents; no live submits.
- `production`: GTP + rebalancer exclusively own UVIX, SVIX, and B5 option contracts on the **live** account; B1/B2/B4 phases must exclude those symbols.

The cutover must fail closed if two owners are active. A rerun must not duplicate an order.

### 3.2 Proposed components

Wire through the existing plan → rebalance path (see **2026-07-20 amendment**). Do not
stand up a separate primary executor:

| Component | Responsibility |
|---|---|
| `config/bucket5_production.yml` | Frozen economic parameters, capital ceiling, ramp, execution limits, automation flags, and kill modes |
| `scripts/bucket5_policy.py` | Pure, side-effect-free regime, cadence, ladder, monetization, and redeployment rules shared by research and live |
| `generate_trade_plan.py` (B5 extension) | Broker-aware B5 target generation and exact contract selection; emits carry + option plan rows |
| `rebalance_strategy.py` / `execute_trade_plan.py` (OPT path) | Idempotent stock + option limit-order execution with dry-run and manual-approval modes |
| `scripts/bucket5_ledger.py` | Event-sourced position, cost-basis, peak-mark, tier-fired, roll, cash, and redeployment state |
| `scripts/bucket5_reconcile.py` | Flex/broker reconciliation and daily NAV identity |
| `data/runs/<date>/bucket5_production/` | Immutable inputs, decision manifest, targets, intents, fills, rejects, positions, marks, P&L, reconciliation, and approvals |

`proposed_trades.csv` (plus optional `proposed_trades_b5_options.csv`) is the execution
input the rebalancer already understands. The B5 ledger remains the authoritative option
inventory; plan rows are intents, not the lot book.

### 3.3 Exact contract identity

Every option lot must retain:

- IBKR `conId`;
- local symbol;
- underlying;
- security type;
- expiry;
- strike;
- right;
- trading class;
- multiplier;
- settlement style/session;
- entry order/fill IDs and timestamps;
- original quantity and remaining quantity;
- entry premium and allocated fees;
- peak executable mark since entry;
- monetization tiers already fired; and
- roll/re-arm parent intent.

IBKR's current API documentation says derivatives require fields such as expiration, trading class, multiplier, and strike, and identifies `conId` as the exact contract identifier. That is the appropriate production key: [IBKR TWS API contracts documentation](https://ibkrcampus.com/campus/ibkr-api-page/twsapi-doc/).

### 3.4 Option choice and pilot granularity

Standard SPX has a $100 multiplier. At an initial total roll budget of roughly $3,200–$4,970, whole SPX contracts can materially distort the three-rung weights. The implementation should run a predeclared SPX-versus-XSP execution study.

XSP is one-tenth the size of SPX, cash-settled, and European exercise, which makes it a natural pilot candidate if its spreads and fills are acceptable: [Cboe XSP overview](https://www.cboe.com/tradable-products/sp-500/xsp-options/). SPX itself has a $100 multiplier and product-specific trading/expiration hours: [Cboe SPX specifications](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications/).

Rules:

- do not substitute SPY options without a separate strategy decision; their exercise and physical-delivery mechanics differ;
- do not mix SPX and XSP marks as if they were the same contract;
- compare spread cost, depth, fill quality, commissions, and tracking by rung;
- use the instrument that stays within the premium budget with acceptable executable liquidity; and
- transition from XSP to SPX only through an explicit roll plan, never by silently changing the contract family.

### 3.5 Live prices and fail-closed behavior

Research may compare Theta marks and Black-Scholes estimates. Live decisions may not use a model fallback as an executable quote.

For every live option order require:

- an exact qualified contract;
- a current, two-sided broker quote;
- positive bid, ask not below bid, and a configurable maximum quote age;
- a configurable maximum spread as a fraction of mid and premium budget;
- a limit order with a bounded cancel/replace schedule; and
- an archived pre-trade quote and final fill.

If any check fails, skip the order and alert. Never send a market order merely to keep the backtest schedule.

### 3.6 State and order idempotency

Give every decision and order a deterministic identity, for example:

```text
strategy_version + asof_timestamp + action_type + conId/symbol + parent_lot + target_stage
```

Before submitting, search the ledger and broker open orders/fills for that identity. A restart can resume or cancel an existing intent, but it cannot create a second economic order.

## 4. Execution policy

### 4.1 Carry legs

Carry execution must treat UVIX and SVIX as a coordinated risk unit even though they are separate orders.

1. Obtain current positions, quotes, short availability, borrow rates, and margin impact for both legs.
2. Compute a feasible paired target under whole shares and available locates.
3. Execute all risk-reducing actions before risk-increasing actions.
4. During de-risking, cover adverse short-UVIX exposure before adding or retaining offsetting short-SVIX exposure.
5. Do not increase either short if the other required locate is absent and the partial fill would breach the net-vol-risk limit.
6. Recompute the second order from actual first-order fills.
7. Stop and leave the book reduce-only if a partial fill, rejection, halt, or quote failure breaches the pair-risk limit.

The current generic short-availability logic can be reused, but pilot controls should be stricter than Bucket 5's existing 70% borrow cap. At minimum:

- missing locate or borrow data blocks increases;
- either borrow rate above two times the frozen research assumption blocks increases until a live-cost replay approves the trade;
- displayed availability use is capped conservatively during the pilot; and
- reductions and covers remain allowed.

The current approximately 12.35% SVIX borrow observation would therefore block a new SVIX increase under the initial rule and force an explicit review instead of silently accepting a 3.6x assumption error.

### 4.2 Put entry and roll

- Select actual listed expiries around the target DTE; do not invent a synthetic expiry.
- Round strikes to listed strikes and archive the distance from the intended 10% / 20% / 30% OTM levels.
- Allocate whole contracts across rungs with an optimization that minimizes weight error while never exceeding total or per-rung budgets.
- Buy replacement protection before selling expiring protection when cash and limits permit.
- Use limit orders only.
- If a rung cannot be purchased inside its limit, leave the budget in cash and record an under-coverage alert.
- Do not increase carry capital while any required rung is materially under-covered.

### 4.3 Monetization

Monetization depends on **executable liquidation value**, not a stale mid or model mark.

- Compute multiples of cost using the current bid net of estimated fees.
- Persist each fired tier so a restart cannot fire it twice.
- Round partial sales down, but ensure the final tier can clear the remaining lot.
- Treat VIX-tier and profit-tier collisions as one ordered state transition.
- At pilot scale, generate and manually release monetization orders.
- After successful drills and live reconciliation, automate sell-only monetization before automating risk-increasing re-arm and redeployment.

Re-arm and redeployment are separate actions. A put sale can succeed while re-arm or carry redeployment is blocked by quotes, locates, margin, or a kill switch.

## 5. Accounting and reconciliation

### 5.1 Required daily identity

Reconcile each day:

```text
prior B5 NAV
+ UVIX price P&L
+ SVIX price P&L
- security-specific borrow
+ attributable short-credit interest
+ attributable bill/cash interest
+ option mark change
+ option sale and settlement cash
- option premium paid
- commissions and exchange/regulatory fees
+ realized redeployment P&L
= ending B5 NAV
```

The residual must be less than `max($10, 1 basis point of B5 allocated NAV)` before any capital or automation ramp. Every fill and cash flow must trace to an intent and contract lot.

### 5.2 Flex changes

Extend or isolate the Flex parsing so that it preserves:

- `assetCategory` / security type;
- `conId`;
- expiry, strike, right, multiplier, and trading class;
- trade, commission, cash-settlement, and open-position records; and
- stock borrow separately by UVIX and SVIX.

Do not allow SPX option rows to flow through stock bucket logic keyed only by `symbol`.

### 5.3 Three mandatory views

Produce independent daily books:

1. **Carry book:** UVIX and SVIX price P&L, borrow, short credit, fees, turnover, and exposures.
2. **Put book:** premium, contract MTM, realized sale cash, residual inventory, fees, and settlement.
3. **Combined B5 book:** exact sum of carry, puts, attributable collateral, and redeployment.

Then aggregate B5 into account-level gross, net, margin, beta, drawdown, and P&L. The combined book is not allowed to hide a component reconciliation failure.

## 6. Risk controls and kill modes

### 6.1 Pre-trade controls

The planner must enforce:

- approved B5 NAV and current ramp factor;
- carry gross, UVIX gross, SVIX gross, and net-vol-risk limits;
- total and per-rung premium budgets;
- maximum open put premium and contract count;
- locate availability and borrow gates on both ETPs;
- maximum share of displayed availability and average daily volume;
- minimum post-trade excess liquidity;
- account gross-leverage ceiling after B5 and all other planned orders;
- stress loss under a predeclared UVIX/SVIX gap grid;
- stress option value at executable bid with a conservative basis haircut; and
- stale-data, crossed-market, halt, and corporate-action gates.

If Production B were applied to the full $1.05 million account NAV, its 20% calm carry target would be about $210,000, or 5% of the $4.2 million L/S target gross. That is larger than the entire current 3% B4 budget and cannot be implemented by simply increasing the `0.25%` pair anchor. Even at the smaller initial ring-fenced allocation, any capital increase must identify which existing gross and collateral fund it or demonstrate that total account gross and margin limits still pass. B5 is not additive free leverage.

### 6.2 Kill modes

Implement explicit, tested modes:

| Mode | Behavior |
|---|---|
| `normal` | All approved actions allowed |
| `no_new_risk` | No new shorts, no re-arm, no carry redeployment; reductions and put sales allowed |
| `reduce_only` | Cover/trim carry toward a lower approved target; retain or monetize puts |
| `flatten_carry` | Cover UVIX first according to risk; then SVIX; do not liquidate puts automatically |
| `exit_options` | Sell option lots by limit order; no market liquidation |
| `halt_all` | Cancel working B5 orders; retain positions; require operator decision |

Triggers for at least `no_new_risk` include:

- stale or missing signal data;
- unresolved contract identity;
- daily reconciliation failure;
- duplicate or orphan order/fill;
- missing locate or recall;
- margin/excess-liquidity breach;
- position mismatch with the broker;
- quote-quality failure;
- state-file corruption or incompatible strategy version; and
- an unapproved parameter/config hash.

Any hard breach automatically resets the capital-ramp decision to the prior approved stage. It does not erase state or market-liquidate positions.

## 7. Phased rollout

Time gates and observation gates both apply. A phase does not pass merely because its calendar time elapsed.

### Phase 0 — charter, capital, and frozen specification (1–2 weeks)

Deliver:

- approved B5 capital ceiling and account loss budget;
- decision on dedicated subaccount versus strict virtual ledger;
- collateral and interest-attribution policy;
- SPX/XSP pilot policy;
- exact Production B v1 config and hash;
- automation/approval matrix;
- kill-switch runbook and named operator coverage; and
- confirmation that account permissions, market data, and current OCC disclosures are in place. OCC maintains the current standardized-options disclosure document here: [OCC options disclosure document](https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document).

Exit gates:

- risk owner accepts the proposed allocation's gap and drawdown references;
- no ambiguity remains about whether bill yield and margin are shared or reserved;
- the initial ceiling cannot increase current carry gross; and
- every economic parameter has one source of truth.

### Phase 1 — build and deterministic replay (2–4 weeks)

Deliver:

- shared pure policy module;
- exact-contract target generator;
- event-sourced B5 ledger;
- option-capable execution path in permanent dry-run mode;
- contract-safe accounting and daily reconciliation;
- pre-trade risk engine;
- golden replays over the live instrument window and selected stress dates; and
- tests for restarts, duplicate intents, partial fills, crossed quotes, missing quotes, no locates, recalls, halts, splits, roll collisions, tier collisions, and cash settlement.

Exit gates:

- research and live target engines match on frozen inputs to rounding tolerance;
- every historical decision is deterministic from archived inputs and state;
- contract-level NAV identity and harvested-cash trace pass;
- no live decision can consume a Black-Scholes fallback;
- bid/ask execution, delayed monetization, locate, and recall cases are injectable; and
- a clean rollback to placeholder mode is proven in a test account.

### Phase 2 — full-notional shadow (at least 40 trading days)

Run Production B every day using actual account state and live quotes, but submit no orders.

Archive:

- signal timestamp and VIX/VIX3M ratio;
- B5 NAV and target gross;
- exact carry and put targets;
- selected contracts and rejected alternatives;
- hypothetical limit orders and fills using conservative bid/ask rules;
- expected borrow, margin, and excess liquidity;
- all monetization and re-arm signals; and
- old-placeholder versus new-policy exposure differences.

Exit gates:

- at least 40 reconciled trading days;
- at least three carry cadence decisions;
- no unexplained target drift or duplicate intent;
- all data-quality failures correctly produce skips;
- shadow gross never breaches the approved ceiling;
- current borrow is incorporated into after-cost results; and
- operators complete a no-new-risk and flatten-carry drill.

### Phase 3 — live-account dry-run drills (replaces paper; at least 20 trading days overlapping late shadow)

**Amendment 2026-07-20:** there is no IBKR paper phase. Validate the full GTP →
rebalancer → ledger → reconcile loop on the **live account** with `--dry-run` only.

Required observations (all non-submitting):

- end-to-end dry-run of three-rung put entry intents with live two-sided quotes;
- dry-run of a scheduled roll (buy replacement before sell, archived conIds);
- at least six carry cadence decisions or stress-driven decisions in shadow outputs;
- restart / duplicate-intent / cancel-replace recovery without creating a second economic order;
- injected monetization at 3x, 5x, 8x, VIX 45/65, and giveback through controlled test inputs;
- simulated recall and margin breach driving `no_new_risk` / `flatten_carry`; and
- next-day Flex parse of *existing* UVIX/SVIX (and any prior test option lots if present)
  through the contract-safe accounting path.

Exit gates:

- zero unresolved dry-run intent / ledger mismatches;
- every injected monetization event fires once and only once in the ledger;
- emergency runbook works without hand-editing state;
- rebalancer cancel hygiene proven: only `ETF_LS|` and `B5P|` scopes touched; and
- a go-live report lists residual differences between dry-run assumptions and live
  borrow/quote reality (paper is not used as a substitute).

### Phase 4 — live coverage-first pilot (approximately 6–10 weeks)

The existing carry placeholder remains capped and reduce-only while long-put mechanics are introduced. This is a risk-reducing pilot, not a justification to add short carry.

Suggested sub-stages:

1. **25% of the approved put budget**, manual release, XSP if the execution study supports it.
2. **50% of the approved put budget** after 20 clean trading days and correct contract reconciliation.
3. **100% of the approved put budget** after another 20 clean trading days.

Whole-contract feasibility overrides percentage labels. Never overspend a rung to hit a nominal stage.

Exit gates before changing the carry policy:

- all live option positions reconcile by `conId` on the next broker statement;
- fill prices remain inside approved limits and quote-quality thresholds;
- realized commissions and exchange fees are captured;
- no duplicate orders, orphan lots, or unexplained cash;
- at least one live entry or partial roll has completed; and
- the operator has successfully run sell-only monetization and halt drills in paper using the live code version.

### Phase 5 — migrate carry to Production B under a no-grow ceiling (4–8 weeks)

Once full approved put coverage is live:

- switch UVIX/SVIX ownership atomically from the generic placeholder to the B5 engine;
- use `b5_allocated_nav ≈ $156,875` as the initial maximum;
- compute the current Production B regime target;
- execute risk-reducing changes first;
- prohibit any carry gross above the lower of the cutover-day live gross and approved Production B target; and
- keep all risk-increasing orders manually approved.

Using the latest cached 2026-07-17 VIX 18.77 and VIX3M 20.54 only as an illustration, the ratio is about 0.914. At the proposed B5 allocation, the Production B formula would imply approximately:

| Illustration | Value |
|---|---:|
| rho | 1.282 |
| carry gross multiplier | 0.817 |
| total carry gross | $25,626 |
| UVIX short | $11,230 |
| SVIX short | $14,396 |

Those numbers are not an order recommendation; the live engine must recompute them from cutover-time data. They show why the migration can reduce gross while changing the leg mix.

Exit gates:

- at least 40 live trading days;
- at least two live cadence changes;
- 100% next-day B5 reconciliations pass;
- median and tail ETP slippage remain inside predeclared limits;
- actual borrow and financing do not erase positive expected carry under the stressed cost model;
- option under-coverage never coincides with a carry increase; and
- no hard risk or operational breach remains open.

### Phase 6 — automate in risk order, not all at once (3–6 months)

Enable automation in this sequence:

1. automatic data validation and target generation;
2. automatic carry reductions/covers;
3. automatic sell-only put monetization with hard limits;
4. automatic carry increases inside approved locate and margin limits;
5. automatic scheduled put rolls;
6. automatic re-arm; and
7. automatic harvested-cash redeployment.

Each capability requires at least 20 clean live trading days at the prior level, a replay/drill of the new path, and explicit approval. Re-arm and carry redeployment are last because they add risk during stressed markets.

### Phase 7 — capital expansion, if earned (not before six live months)

Do not increase the initial B5 allocation until:

- at least six live months have elapsed;
- at least two scheduled live put rolls have reconciled;
- at least one sell-only monetization has occurred live or the same production version has passed repeated controlled drills;
- borrow, slippage, quote coverage, and financing evidence support the economics;
- account-level stress and excess-liquidity tests pass at the proposed new size; and
- a frozen challenger version has run in shadow for at least 40 trading days if parameters changed.

Capital increases should normally be no more than **25% of the prior B5 allocation per gate**. For example, $156.9k → $196.1k → $245.1k, rather than jumping from the placeholder to full-account Production B. Every increase must identify the gross and collateral funding source elsewhere in the account.

## 8. Quantitative gates

The following are proposed starting gates. They should be approved before shadow begins and not relaxed after seeing unfavorable results.

### 8.1 Hard gates

- Daily B5 reconciliation residual ≤ `max($10, 1 bp of allocated B5 NAV)`.
- 100% of live fills map to one approved intent and one broker contract.
- 100% of option inventory maps to exact `conId`-level ledger lots.
- Zero duplicated economic orders after restart or rerun.
- Zero risk-increasing orders with stale signals, invalid quotes, missing locate, or an unapproved config hash.
- No option order above its rung or total premium budget.
- No carry increase when required put coverage is below its approved threshold.
- Post-trade account gross, margin, and excess-liquidity limits pass both current and stressed views.
- No live decision uses synthetic ETP history or a modeled option price as a substitute for an executable quote.

### 8.2 Economic gates

- Re-run the `B_live` window with actual bid/ask execution, contract rounding, observed borrow where available, and delayed execution.
- The 50 bp ETP-slippage / 2x-borrow case must remain economically positive before live capital grows. The current sensitivity artifact remains positive in that cell, but it does not yet include execution-side and delayed-monetization effects.
- At least 95% of required live contract-days in the validation sample must have valid two-sided quotes; missing days cannot silently fall back to Black-Scholes.
- Stressed bid/ask replay should retain at least 75% of mid-based harvested cash and keep B5 live-era maximum drawdown inside the allocator-approved limit.
- Actual weighted borrow above two times the frozen assumption pauses increases; the strategy must demonstrate positive after-cost edge using observed rates before resuming.
- No single component—bill yield, synthetic history, model-marked puts, or one stress episode—may account for the investment case by itself.

### 8.3 Execution-quality gates

- ETP arrival slippage: predeclare median and 95th-percentile limits by order size; use 15 bp median and 50 bp 95th percentile as initial review thresholds, not promises.
- Options: every fill must be at or better than its submitted limit and within the approved fraction of the contemporaneous spread.
- Reject or skip quotes whose age or width breaches the rung-specific threshold.
- Track cancel/replace count, fill latency, partial-fill duration, and unfilled protection budget.
- Track real borrow availability, rejected short orders, recalls, and broker margin—not only FTP indications.

## 9. Improvement loop after launch

### Daily

- Reconcile positions, cash, orders, fills, option lots, and B5 NAV.
- Compare target versus actual carry gross, rho, put coverage, premium budget, and Greeks.
- Review borrow changes, recalls, quote gaps, stale data, and margin headroom.
- Publish one B5 health status: `green`, `no_new_risk`, `reduce_only`, or `halted`.

### Weekly operations review

- Review every rejection, partial fill, manual override, and limit escalation.
- Compare live fills with arrival mid, bid/ask, and shadow assumptions.
- Review actual versus assumed borrow and interest.
- Confirm the next roll window and operator coverage.
- Close or assign every reconciliation exception.

### Monthly economic review

Report both absolute dollars and return on allocated B5 NAV:

- UVIX price P&L;
- SVIX price P&L;
- borrow and short credit by security;
- bill/cash interest actually attributable to B5;
- option premium paid;
- unrealized put MTM;
- realized monetization cash;
- redeployment P&L;
- fees and slippage;
- carry-only, put-only, and combined drawdown; and
- contribution to total account risk and return.

Capital gates should depend more on reconciliation, execution, borrow, and stress behavior than on one or two months of favorable P&L.

### Quarterly model review

- Maintain the live version as the **champion** and run proposed changes as shadow **challengers**.
- Change one economic family at a time: regime thresholds, cadence, rung weights, monetization, or redeployment—not several together.
- Pre-register the test period and acceptance metrics.
- Require at least 40 forward shadow days for a changed policy.
- Version every parameter set and retain old-state replay compatibility.
- Update cost and borrow assumptions from live evidence without automatically optimizing the signal to the same sample.
- Re-run stress packets and the full live-era replay after every material change.

Do not tune the strategy in response to one crash, one missed monetization, or one profitable month. Operational fixes can ship immediately behind tests; economic changes require a new frozen version and forward shadow period.

## 10. First implementation backlog

1. Approve `b5_allocated_nav`, collateral treatment, and account risk budget (live account; no paper).
2. Extract the Production B rules into a pure shared policy module with golden parity tests.
3. Extend `generate_trade_plan.py` with exact-contract B5 put target generation (XSP/SPX) and Production B carry sizing.
4. Build the event-sourced B5 ledger and deterministic intent IDs.
5. Extend `rebalance_strategy.py` / `execute_trade_plan.py` with an OPT limit-order path, `B5P|` orderRef, dry-run, and manual approval.
6. Add exclusive-ownership guards so B1/B2/B4 cannot resize UVIX/SVIX when mode is `production`.
7. Extend Flex ingestion and accounting to contract-level option records.
8. Add daily B5 NAV identity, fill/cash trace, and account-level risk aggregation (dashboard live panel).
9. Add kill modes, stale-data/quote/locate/margin gates, and operator drills; patch spx-0dte same-day filter.
10. Run ≥40 shadow days + ≥20 live dry-run drill days, then the one-contract live put pilot (no paper phase).

## 11. Go/no-go summary

### Go to shadow when

- the capital and collateral model is approved;
- Production B has one versioned source of truth;
- target generation is deterministic; and
- all outputs are read-only.

### Go to live dry-run drills when *(replaces “go to paper”)*

- exact contracts and order intents are stable in GTP output;
- accounting works by contract on Flex;
- failures are injectable in the live `--dry-run` path; and
- rollback to `placeholder` mode is tested.

### Go to the live coverage pilot when

- shadow + dry-run drills have passed their exit gates;
- a full dry-run roll has reconciled in the ledger;
- live quotes never fall back to model prices;
- option budgets fit whole contracts without overspending;
- 0DTE isolation (XSP and/or same-day SPXW filter + `B5P|` refs) is verified; and
- every risk-increasing order remains manually released.

### Go to Production B carry ownership when

- approved put coverage is live;
- the new target is no larger than the existing carry gross;
- observed borrow passes the after-cost gate; and
- there is exactly one owner of UVIX/SVIX.

### Go to larger capital when

- six or more live months and two live rolls have passed;
- execution and accounting evidence support the economics;
- the account risk budget supports the new allocation;
- no hard gate is open; and
- the increase is no more than 25% of the prior approved B5 allocation.

This sequence turns Production B into a measurable, reversible operating process. It does not assume that a successful research backtest, the current tiny live pair, or a paper fill proves that the full product is ready for account-scale capital.
