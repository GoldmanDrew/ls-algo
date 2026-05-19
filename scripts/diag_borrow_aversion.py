"""Diagnostic: trace what changing ``borrow_aversion`` actually does to sleeve weights.

For each sleeve (core_leveraged, yieldboost, inverse_decay_bucket4 core, inverse_decay_bucket4 vol_etp)
this script:
  1. Reproduces the production eligibility / sleeve mask from ``generate_trade_plan.main``.
  2. Sweeps ``borrow_aversion`` ∈ {0, 0.25, 0.5, 1, 2, 3, 5, 10}.
  3. Prints, for each level:
       - n active rows (positive weight after clip)
       - max weight, top-3 names, weights on the 5 highest-borrow names
       - delta vs aversion=0 baseline (max abs weight delta, sum of |Δ| across names)
  4. Pinpoints whether any rows were *zeroed out* by the borrow penalty
     (raw_score = signal - a*borrow goes ≤ 0 then clipped).

Usage:
    python -m scripts.diag_borrow_aversion
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_trade_plan import (  # noqa: E402
    _b2_b4_universe_masks,
    _b4_eligibility_edge_column,
    _decay_score_weights,
    _norm_sym,
    compute_borrow_annual_series,
    get_borrow_col,
)

CFG_PATH = ROOT / "config" / "strategy_config.yml"
SCREENED_PATH = ROOT / "data" / "etf_screened_today.csv"

AVERSIONS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def _load_inputs() -> tuple[dict, pd.DataFrame]:
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    df = pd.read_csv(SCREENED_PATH)
    df["Delta"] = pd.to_numeric(df["Delta"], errors="coerce")
    df["delta_abs"] = df["Delta"].abs()
    for c in ("blended_gross_decay", "borrow_current", "net_decay_annual",
              "net_edge_p50_annual", "vol_underlying_annual", "bucket4_net_edge_annual"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    bcol = get_borrow_col(df)
    df["borrow_annual"] = compute_borrow_annual_series(df, bcol)
    return cfg, df


def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    if "net_edge_p50_annual" in df.columns:
        ne_ok = df["net_edge_p50_annual"].fillna(-1.0).ge(0.0)
    else:
        ne_ok = pd.Series(True, index=df.index)
    return df.loc[(df["purgatory"] != True) & ne_ok].copy()  # noqa: E712


def _build_sleeve_frames(cfg: dict, eligible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sleeves = cfg.get("portfolio", {}).get("sleeves", {})
    flow = sleeves.get("flow_program", {}) or {}
    flow_shorts = (flow.get("universe", {}) or {}).get("shorts", []) or []
    flow_set = {_norm_sym(x) for x in flow_shorts}

    is_yb, in_b2_universe, in_flow_program = _b2_b4_universe_masks(
        eligible, flow_program_etfs=flow_set
    )
    pos_beta = eligible["Delta"].gt(0)
    neg_beta = eligible["Delta"].lt(0)

    if "inverse_shortable" in eligible.columns:
        inv_shortable = eligible["inverse_shortable"].fillna(False).astype(bool)
    else:
        inv_shortable = neg_beta

    per_bucket = (cfg.get("screener", {}) or {}).get("per_bucket", {}) or {}
    b1_cap = float((per_bucket.get("bucket_1") or {}).get("entry_borrow_cap", 1.0))
    b2_cap = float((per_bucket.get("bucket_2") or {}).get("entry_borrow_cap", b1_cap))
    b4_cap = float((per_bucket.get("bucket_4") or {}).get("entry_borrow_cap", np.inf))

    b = eligible["borrow_annual"]
    core_borrow_ok = (~np.isfinite(b)) | (b <= b1_cap)
    yb_borrow_ok = (~np.isfinite(b)) | (b <= b2_cap)
    b4_borrow_ok = (~np.isfinite(b)) | (b <= b4_cap)

    core = sleeves.get("core_leveraged", {}) or {}
    yb_sleeve = sleeves.get("yieldboost", {}) or {}
    b4 = sleeves.get("inverse_decay_bucket4", {}) or {}

    core_rules = core.get("rules", {}) or {}
    core_delta_min = float(core_rules.get("min_delta_used", 1.5))

    yb_rules = yb_sleeve.get("rules", {}) or {}
    yb_min_edge = float(yb_rules.get("min_net_edge_annual", 0.0) or 0.0)

    b4_rules = b4.get("rules", {}) or {}
    b4_min_edge = float(b4_rules.get("min_net_edge_annual", 0.0))
    b4_min_vol = float(b4_rules.get("min_underlying_vol", 0.50))
    b4_excluded = {_norm_sym(x) for x in (b4_rules.get("excluded_etfs") or [])}

    edge_col = _b4_eligibility_edge_column(eligible)
    b4_edge = pd.to_numeric(eligible.get(edge_col), errors="coerce")
    b4_edge_ok = (~np.isfinite(b4_edge)) | (b4_edge >= b4_min_edge)
    b4_und_vol = pd.to_numeric(eligible.get("vol_underlying_annual"), errors="coerce")
    b4_vol_ok = np.isfinite(b4_und_vol) & (b4_und_vol >= b4_min_vol)
    b4_not_excluded = ~eligible["ETF"].isin(b4_excluded)

    ne = pd.to_numeric(eligible.get("net_edge_p50_annual"), errors="coerce")
    yb_edge_ok = np.isfinite(ne) & (ne >= yb_min_edge) if yb_min_edge > 0 else pd.Series(True, index=eligible.index)

    in_core = pos_beta & eligible["delta_abs"].ge(core_delta_min) & core_borrow_ok & ~in_b2_universe
    in_yb = pos_beta & in_b2_universe & ~in_flow_program & yb_borrow_ok & yb_edge_ok
    in_b4_core = neg_beta & inv_shortable & b4_borrow_ok & b4_edge_ok & b4_vol_ok & b4_not_excluded & ~in_flow_program

    is_vol_etp = pd.Series(False, index=eligible.index)
    for col in ("Delta_product_class", "product_class"):
        if col in eligible.columns:
            is_vol_etp |= eligible[col].astype(str).str.lower().eq("volatility_etp")
    in_b4_vol = is_vol_etp & inv_shortable & b4_borrow_ok & b4_edge_ok & b4_vol_ok & b4_not_excluded & ~in_flow_program & ~in_b4_core

    return {
        "core_leveraged": eligible.loc[in_core].copy(),
        "yieldboost": eligible.loc[in_yb].copy(),
        "b4_core_inverse": eligible.loc[in_b4_core].copy(),
        "b4_volatility_etp": eligible.loc[in_b4_vol].copy(),
    }


def _gini(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    if w.size == 0 or w.sum() <= 0:
        return float("nan")
    x = np.sort(w)
    n = x.size
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _sweep_sleeve(name: str, sleeve_df: pd.DataFrame, cfg: dict, sleeve_cfg_key: str) -> None:
    print("=" * 92)
    print(f"  SLEEVE: {name}   (n_eligible_rows = {len(sleeve_df)})")
    print("=" * 92)
    if sleeve_df.empty:
        print("  (no eligible rows; nothing to sweep)\n")
        return

    sleeves = cfg.get("portfolio", {}).get("sleeves", {})
    base_w_cfg = dict((sleeves.get(sleeve_cfg_key, {}) or {}).get("weighting", {}) or {})
    yaml_a = base_w_cfg.get("borrow_aversion", "?")
    print(f"  YAML borrow_aversion = {yaml_a}   "
          f"sizing_signal={base_w_cfg.get('sizing_signal','?')}  "
          f"sizing_edge_column={base_w_cfg.get('sizing_edge_column','?')}  "
          f"score_concavity_p={base_w_cfg.get('score_concavity_p',1.0)}  "
          f"margin_efficiency_power={base_w_cfg.get('margin_efficiency_power',0.0)}  "
          f"eq_blend={base_w_cfg.get('eq_blend',0.0)}  "
          f"max_name_weight={base_w_cfg.get('max_name_weight',1.0)}")

    sig_col = base_w_cfg.get("sizing_edge_column", "net_edge_p50_annual")
    if sig_col not in sleeve_df.columns:
        sig_col = "net_edge_p50_annual"
    signal = pd.to_numeric(sleeve_df.get(sig_col), errors="coerce")
    borrow = pd.to_numeric(sleeve_df["borrow_current"], errors="coerce").fillna(0.0)
    print(f"  signal={sig_col}: min={float(signal.min()):.4f} med={float(signal.median()):.4f} "
          f"max={float(signal.max()):.4f} nan={int(signal.isna().sum())}")
    print(f"  borrow_current: min={float(borrow.min()):.4f} med={float(borrow.median()):.4f} "
          f"max={float(borrow.max()):.4f} std={float(borrow.std()):.4f}")
    print(f"  ratio std(borrow)/std(signal) = "
          f"{float(borrow.std()/(signal.std() if signal.std()>0 else np.nan)):.3f}\n")

    if "ETF" in sleeve_df.columns:
        labels = sleeve_df["ETF"].astype(str).to_numpy()
    else:
        labels = sleeve_df.index.astype(str).to_numpy()

    high_borrow_idx = borrow.sort_values(ascending=False).head(5).index.tolist()
    high_borrow_pos = [int(sleeve_df.index.get_loc(i)) for i in high_borrow_idx]
    high_borrow_etfs = [labels[p] for p in high_borrow_pos]
    high_borrow_vals = [float(borrow.loc[i]) for i in high_borrow_idx]

    base_w = None
    rows = []
    for a in AVERSIONS:
        cfg_a = dict(base_w_cfg)
        cfg_a["borrow_aversion"] = a
        w = _decay_score_weights(sleeve_df, cfg_a, sleeve_name=sleeve_cfg_key)
        if base_w is None:
            base_w = w.copy()
        n_active = int((w > 1e-9).sum())
        n_zeroed_now = int(((base_w > 1e-9) & (w <= 1e-9)).sum())
        max_w = float(w.max()) if w.size else 0.0
        top_idx = np.argsort(-w)[:3]
        top3 = ", ".join(f"{labels[k]}={w[k]*100:.1f}%" for k in top_idx)
        hb = ", ".join(f"{labels[p]}={w[p]*100:.2f}%" for p in high_borrow_pos)
        delta_max = float(np.max(np.abs(w - base_w))) if base_w is not None else 0.0
        l1 = float(np.sum(np.abs(w - base_w))) if base_w is not None else 0.0
        rows.append({
            "a": a, "n_active": n_active, "n_zeroed_vs_a0": n_zeroed_now,
            "max_w": max_w, "gini": _gini(w),
            "top3": top3, "high_borrow_weights": hb,
            "delta_max_vs_a0": delta_max, "L1_vs_a0": l1,
        })

    print(f"  HIGH-BORROW probes (top-5 by borrow_current): "
          + ", ".join(f"{e}={b*100:.1f}%" for e, b in zip(high_borrow_etfs, high_borrow_vals)))
    print()
    print(f"  {'a':>5}  {'n_act':>5}  {'zeroed':>6}  {'max_w':>6}  {'gini':>6}  "
          f"{'dMax':>7}  {'L1d':>7}  top3 / high-borrow weights")
    for r in rows:
        print(
            f"  {r['a']:>5.2f}  {r['n_active']:>5d}  {r['n_zeroed_vs_a0']:>6d}  "
            f"{r['max_w']*100:>5.2f}%  {r['gini']:>6.3f}  "
            f"{r['delta_max_vs_a0']*100:>6.3f}%  {r['L1_vs_a0']*100:>6.2f}%  "
            f"top: {r['top3']}"
        )
        print(f"  {' '*38}        high-borrow: {r['high_borrow_weights']}")
    print()


def main() -> None:
    cfg, df = _load_inputs()
    elig = _eligible(df)
    print(f"\n[INPUT] screened rows = {len(df)}, eligible after purgatory + net_edge>=0 = {len(elig)}\n")
    frames = _build_sleeve_frames(cfg, elig)
    pairs = [
        ("core_leveraged",      "core_leveraged"),
        ("yieldboost",          "yieldboost"),
        ("b4_core_inverse",     "inverse_decay_bucket4"),
        ("b4_volatility_etp",   "inverse_decay_bucket4"),
    ]
    for diag_name, cfg_key in pairs:
        _sweep_sleeve(diag_name, frames[diag_name], cfg, cfg_key)


if __name__ == "__main__":
    main()
