"""Post-process ``data/etf_screened_today.csv`` with the refined product-class
taxonomy without rerunning the full network/data pipeline.

What this script does:
  1. Re-derives ``product_class`` from the (Beta, Leverage, is_yieldboost)
     triplet using the new ``screener_v2_fields._product_class`` taxonomy.
  2. Re-derives ``expected_decay_available`` from that taxonomy.
  3. For ``passive_low_beta`` rows, nulls out the family of expected /
     distributional decay columns (Itô identity gives ~0 around β=1, so any
     value there is at best noise; the dashboard falls back to realized).
  4. Writes the file back in place AND mirrors to today's dated run dir.

This intentionally does NOT re-run the rest of ``daily_screener.py`` — vol,
borrow, net-edge, etc. all stay exactly as the previous run produced them.
Use this when you have changed taxonomy logic and want the dashboard build
to pick the new taxonomy up immediately without a full refresh.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Make sibling modules in the repo root importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

# Re-use the same module so the post-processor and the live pipeline can never
# drift apart on classification rules.
from screener_v2_fields import _expected_decay_available, _product_class


_PASSIVE_NULL_COLS = (
    "expected_gross_decay_annual",
    "expected_gross_decay_annual_legacy",
    "expected_gross_decay_adjusted_annual",
    "expected_gross_decay_simple_ito_annual",
    "expected_decay_adjustment_annual",
    "blended_gross_decay",
    "expected_gross_decay_p10_annual",
    "expected_gross_decay_p50_annual",
    "expected_gross_decay_p90_annual",
    "expected_gross_decay_mean_annual",
    "expected_logIV_mu_annual",
    "expected_logIV_sigma_annual",
    "expected_gross_decay_dist_n_obs",
    "expected_gross_decay_dist_horizon_days",
    "mechanical_decay_annual",
)


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v) and not (isinstance(v, float) and np.isnan(v))
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "t", "yes", "y")


def reapply(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if "is_yieldboost" not in df.columns:
        df["is_yieldboost"] = False

    new_class: list[str] = []
    new_avail: list[bool] = []
    for _, row in df.iterrows():
        beta = row.get("Beta")
        lev = row.get("Leverage") if "Leverage" in df.columns else None
        is_yb = _truthy(row.get("is_yieldboost", False))
        cls = _product_class(lev, beta, is_yieldboost=is_yb)
        new_class.append(cls)
        new_avail.append(_expected_decay_available(cls))

    # Volatility ETPs were already routed to product_class=='volatility_etp'
    # by ``apply_volatility_etp_expected_decay_adjustment``; preserve that.
    prev_class = df.get("product_class")
    if prev_class is not None:
        for j, prev in enumerate(prev_class):
            if str(prev).strip() == "volatility_etp":
                new_class[j] = "volatility_etp"
                new_avail[j] = _expected_decay_available("volatility_etp")

    df["product_class"] = new_class
    df["expected_decay_available"] = new_avail

    passive_mask = df["product_class"].astype(str).eq("passive_low_beta")
    n_passive = int(passive_mask.sum())
    if n_passive:
        for col in _PASSIVE_NULL_COLS:
            if col in df.columns:
                df.loc[passive_mask, col] = np.nan
        if "expected_gross_decay_dist_model" in df.columns:
            df.loc[passive_mask, "expected_gross_decay_dist_model"] = (
                "passive_low_beta_na"
            )
        if "expected_gross_decay_reliable" in df.columns:
            df.loc[passive_mask, "expected_gross_decay_reliable"] = False

    counts = df["product_class"].value_counts(dropna=False).to_dict()
    print(f"[REAPPLY] product_class counts: {counts}")
    print(f"[REAPPLY] passive_low_beta rows nulled: {n_passive}")

    return df


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    primary = repo_root / "data" / "etf_screened_today.csv"
    if not primary.exists():
        print(f"[REAPPLY] {primary} not found", file=sys.stderr)
        return 1

    out = reapply(primary)
    out.to_csv(primary, index=False)
    print(f"[REAPPLY] Wrote {primary}")

    today = date.today().isoformat()
    dated = repo_root / "data" / "runs" / today / "etf_screened_today.csv"
    if dated.exists():
        out.to_csv(dated, index=False)
        print(f"[REAPPLY] Wrote {dated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
