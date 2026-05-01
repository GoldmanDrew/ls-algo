"""Quick smoke test of trailing-partial-year crystallization for LP fees."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.lp_fees_v15 import apply_lp_fees_quarterly, build_lp_fee_daily_cashflow_usd


def main() -> None:
    rng = pd.bdate_range("2025-01-02", "2026-03-31")
    rs = np.full(len(rng), 0.0006)
    s = pd.Series(rs, index=rng)

    _, fee_diag = apply_lp_fees_quarterly(
        s, mgmt_fee_q=0.005, incentive_fee=0.20, crystallize_trailing_partial_year=True
    )
    print("=== with trailing partial year crystallization ===")
    print(fee_diag[["QuarterEnd", "MgmtFee_amt", "PerfFee_amt"]].to_string(index=False))

    _, fee_diag2 = apply_lp_fees_quarterly(
        s, mgmt_fee_q=0.005, incentive_fee=0.20, crystallize_trailing_partial_year=False
    )
    print()
    print("=== without (legacy behaviour) ===")
    print(fee_diag2[["QuarterEnd", "MgmtFee_amt", "PerfFee_amt"]].to_string(index=False))

    # NAV-driven cashflow path
    nav = (1.0 + s).cumprod() * 10_000_000.0
    cf_on = build_lp_fee_daily_cashflow_usd(
        nav,
        rng,
        attribution_base_capital=10_000_000.0,
        management_fee_annual=0.02,
        incentive_fee=0.20,
        crystallize_trailing_partial_year=True,
    )
    cf_off = build_lp_fee_daily_cashflow_usd(
        nav,
        rng,
        attribution_base_capital=10_000_000.0,
        management_fee_annual=0.02,
        incentive_fee=0.20,
        crystallize_trailing_partial_year=False,
    )
    nz_on = cf_on.loc[(cf_on["mgmt_usd"].abs() > 0) | (cf_on["perf_usd"].abs() > 0)]
    nz_off = cf_off.loc[(cf_off["mgmt_usd"].abs() > 0) | (cf_off["perf_usd"].abs() > 0)]
    print()
    print("=== build_lp_fee_daily_cashflow_usd (ON) — non-zero rows ===")
    print(nz_on.to_string())
    print()
    print("=== build_lp_fee_daily_cashflow_usd (OFF, legacy) — non-zero rows ===")
    print(nz_off.to_string())


if __name__ == "__main__":
    main()
