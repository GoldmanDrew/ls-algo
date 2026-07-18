# B5 return verification

## Scope
This artifact implements Appendix A's non-live checks. Paper/live broker fills, locate archives, recalls, and broker financing records are intentionally out of scope and remain future gating requirements.

## Hard accounting gates
- **nav_identity:** PASS (max_error=2.874003257602453e-10, base_attribution_diagnostic=136.5038407783537)
- **harvested_cash_trace:** PASS (untraced_cash=0.0)

## Combined portfolio metrics
- **CAGR:** 7.58%
- **Vol:** 6.40%
- **Sharpe:** 117.41%
- **MaxDD:** -5.26%
- **Calmar:** 144.07%

## Split books and attribution
- `carry_book.csv`, `put_book.csv`, and `combined_book.csv` all use the same initial capital.
- `daily_attribution.csv` separates gross/net UVIX and SVIX price P&L, allocated borrow, T-bills, put premium, unrealized put MTM, monetization cash, and redeployment.
- `daily_reconciliation.csv` reconciles combined NAV from sleeve, T-bill, put MTM/cash, and redeployment components.
- `harvest_audit.csv` ties banked put cash to monetization contract-sale events where engine event detail exists.
- **Locate/recall replay:** SKIP. The carry engine cannot yet zero new shorts on no-locate days or force-cover recalls; this is an explicit future gate.

## Theta option-mark replay
- **Status:** ok
- **Observations:** 402
- **Mean model-minus-quote error:** 26.88
- **Median error:** 6.41
- **95th percentile absolute error:** 131.06
- **Soft flag, Theta tail error large:** True

## Stress packets
- Wrote 13 available packets. Fixed packets require data coverage; UVIX spike packets cover available live-era stress dates.

## Sensitivity and soft falsifiers
- `sensitivity.csv` covers 5/15/30/50 bp ETP slippage and 1.0/1.5/2.0× borrow.
- Option execution-side, Theta-only/BS-only forced modes, delayed monetization, and no-locate/recall cannot be injected into the current production engine. They are reported as future soft verification gates, not treated as passes.
- **Soft flag, costs erase edge:** False

## Gate classification
- **Hard fail:** combined NAV identity; harvested cash that cannot be traced to a monetization event.
- **Soft flag/report:** cost-stressed edge erased, weak B_live performance, large Theta stress error, missing cache coverage, and engine capabilities not yet available.
