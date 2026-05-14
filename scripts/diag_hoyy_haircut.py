"""One-shot diagnostic: show YieldBoost sleeve weights with new borrow_aversion + haircut."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from generate_trade_plan import (
    _decay_score_weights,
    apply_gross_sizing_book_caps,
    load_shares_outstanding_map,
)
from strategy_config import load_config


def main() -> None:
    cfg = load_config()
    strategy = cfg["strategy"]
    paths = cfg["paths"]
    yb_w = cfg["portfolio"]["sleeves"]["yieldboost"]["weighting"]
    print("YB borrow_aversion now:", yb_w["borrow_aversion"])
    print(
        "YB pre_cap_score_haircut_multiplier:",
        strategy["gross_sizing_caps"]["per_sleeve"]["yieldboost"].get(
            "pre_cap_score_haircut_multiplier"
        ),
    )

    screened = pd.read_csv(paths["screened_csv"])
    yb_etfs = [
        "AMYY", "BBYY", "COYY", "HOYY", "IOYY", "FBYY", "MTYY", "QBY", "SMYY", "XBTY",
    ]
    sub = screened[screened["ETF"].isin(yb_etfs)].copy().reset_index(drop=True)
    sub["beta_abs"] = sub["Beta"].astype(float).abs()

    w = _decay_score_weights(sub, yb_w, sleeve_name="yieldboost")
    sub["gross_target_usd"] = 1_600_000.0 * w
    sub["sleeve"] = "yieldboost"
    sub["_pre_cap_score_weight"] = w

    shares_out_map, _ = load_shares_outstanding_map(paths)

    print("\nDECAY SCORE weights (post borrow_aversion=0.5):")
    disp = sub[["ETF", "borrow_current", "net_edge_p50_annual"]].copy()
    disp["decay_w"] = w
    print(disp.sort_values("decay_w", ascending=False).to_string(index=False))

    out, diag = apply_gross_sizing_book_caps(
        sub,
        target_gross_usd=3_200_000.0,
        beta_floor=0.1,
        strategy=strategy,
        shares_out_map=shares_out_map,
    )
    g = out["gross_target_usd"]
    sleeve_sum = float(g.sum())
    final_frac = g / sleeve_sum

    print("\nAFTER cap stack (haircut x2.0 active):")
    res = out[["ETF"]].copy()
    res["frac_of_sleeve"] = final_frac
    res["gross_usd"] = g
    print(res.sort_values("frac_of_sleeve", ascending=False).to_string(index=False))

    hc = (diag.get("pre_cap_score_haircut") or {}).get("yieldboost") or {}
    print(f"\nsleeve_sum = ${sleeve_sum:,.0f}; rows_capped_by_haircut = {hc.get('n_rows_capped_by_haircut')}")
    print(
        "HOYY final frac:",
        float(final_frac[res.ETF == "HOYY"].iloc[0]),
    )


if __name__ == "__main__":
    main()
