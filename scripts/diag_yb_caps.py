"""Print, for each YB row, the binding cap among (mp_f, liquidity, haircut) and resulting cap_v."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from generate_trade_plan import (
    _decay_score_weights,
    _liquidity_tight_book_weights,
    load_shares_outstanding_map,
)
from strategy_config import load_config


def main() -> None:
    cfg = load_config()
    strategy = cfg["strategy"]
    paths = cfg["paths"]
    yb_w = cfg["portfolio"]["sleeves"]["yieldboost"]["weighting"]
    yb_caps_cfg = strategy["gross_sizing_caps"]["per_sleeve"]["yieldboost"]
    root_caps = strategy["gross_sizing_caps"]
    mp_f = float(yb_caps_cfg["max_pair_weight"])
    ha_f = float(yb_caps_cfg["pre_cap_score_haircut_multiplier"])
    target_gross = float(strategy["capital_usd"] * strategy["gross_leverage"])
    yb_budget = target_gross * 0.5

    yb_etfs = ["AMYY","BBYY","COYY","HOYY","IOYY","FBYY","MTYY","QBY","SMYY","XBTY"]
    screened = pd.read_csv(paths["screened_csv"])
    sub = screened[screened["ETF"].isin(yb_etfs)].copy().reset_index(drop=True)
    sub["beta_abs"] = sub["Beta"].astype(float).abs()
    w = _decay_score_weights(sub, yb_w, sleeve_name="yieldboost")
    sub["gross_target_usd"] = yb_budget * w
    sub["sleeve"] = "yieldboost"
    sub["_pre_cap_score_weight"] = w

    shares_out_map, _ = load_shares_outstanding_map(paths)

    # Liquidity caps as the production code computes them: anchor T = deployed_book
    # (here, sum of pre-cap gross_target_usd which equals yb_budget when fully deployed)
    deployed_T = float(sub["gross_target_usd"].sum())
    liq = _liquidity_tight_book_weights(
        sub,
        target_gross_usd=deployed_T,
        beta_floor=float(strategy.get("beta_floor", 0.1)),
        caps={**root_caps},  # uses aum/sh_av/missing_shares
        shares_out_map=shares_out_map,
    )
    s_b = 1.0  # only YB sleeve in this synthetic
    cap_mp = np.full(len(sub), mp_f)
    cap_liq = liq / max(s_b, 1e-18)
    cap_haircut = ha_f * w
    cap_eff = np.minimum.reduce([cap_mp, cap_liq, np.maximum(cap_haircut, 1e-18)])

    out = sub[["ETF", "borrow_current"]].copy()
    out["decay_w"] = w
    out["cap_mp_pair"] = cap_mp
    out["cap_liquidity"] = cap_liq
    out["cap_haircut"] = cap_haircut
    out["cap_effective"] = cap_eff
    binders = []
    for i in range(len(sub)):
        b = []
        if abs(cap_eff[i] - cap_mp[i]) < 1e-9: b.append("mp_pair")
        if abs(cap_eff[i] - cap_liq[i]) < 1e-9: b.append("liquidity")
        if abs(cap_eff[i] - cap_haircut[i]) < 1e-9: b.append("haircut")
        binders.append(",".join(b) if b else "?")
    out["binding"] = binders
    out["max_dollars_at_cap"] = cap_eff * yb_budget
    print("Per-row caps (YB sleeve, deployed_book anchor=$%.0f):" % deployed_T)
    print(out.sort_values("decay_w", ascending=False).to_string(index=False))
    print()
    print("Sum of caps (sleeve-internal):", round(cap_eff.sum(), 4))
    print("Sum of max dollars at cap: $%.0f vs YB budget $%.0f" % (cap_eff.sum() * yb_budget, yb_budget))


if __name__ == "__main__":
    main()
