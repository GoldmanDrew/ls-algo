"""Bucket 4 WS4/WS5 sizing overlays: top-N concentration + cluster caps.

Applied by ``generate_trade_plan`` to the B4 core slice AFTER decay-score
weights (and after the pair-lifecycle ladder), gated by YAML:

  rules.concentration.enabled / top_n_pairs
      Keep only the best N pairs by ``(bucket4_net_edge_annual - borrow) /
      vol_underlying_annual``. Pairs outside the top-N that are currently HELD
      (present in the lifecycle state) become keep-open rows (weight 0,
      purgatory=True) so they roll off via cadence/lifecycle instead of being
      force-closed; never-held pairs are simply dropped.

  rules.cluster_caps.<name>.cap / members
      Cap the combined weight of a cluster (members matched on UNDERLYING),
      redistributing the excess to non-cluster pairs.

Backtest basis (scripts/bucket4_phase345_backtest.py Stage C): top-10/15
selection with equal weights beat the all-pairs book on CAGR and Calmar;
Kelly-style tilting and looser books underperformed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _norm(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def concentration_scores(b4c: pd.DataFrame) -> pd.Series:
    """(net edge - borrow) / underlying vol, NaN-safe (NaN edge -> -inf, ranked last)."""
    edge = pd.to_numeric(b4c.get("bucket4_net_edge_annual"), errors="coerce")
    borrow = pd.to_numeric(b4c.get("borrow_current"), errors="coerce").fillna(0.0)
    vol = pd.to_numeric(b4c.get("vol_underlying_annual"), errors="coerce").clip(lower=0.05)
    score = (edge - borrow) / vol
    return score.fillna(-np.inf)


def apply_concentration_to_b4(
    b4c: pd.DataFrame,
    w: np.ndarray,
    *,
    top_n: int,
    held: set[str],
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Keep top-N pairs by score; held leftovers go keep-open, others drop."""
    if b4c.empty or top_n <= 0 or len(b4c) <= top_n:
        return b4c, w, {"n_dropped": 0, "n_keep_open": 0}
    score = concentration_scores(b4c)
    keep_idx = set(score.nlargest(int(top_n)).index)
    in_top = b4c.index.isin(keep_idx)
    etfs = b4c["ETF"].astype(str).map(_norm)
    is_held = etfs.isin(held).to_numpy()

    drop_mask = ~in_top & ~is_held
    keep_open_mask = ~in_top & is_held

    w = np.asarray(w, dtype=float)
    out = b4c.loc[~drop_mask].copy()
    w_out = w[~drop_mask]
    ko = keep_open_mask[~drop_mask]
    w_out = np.where(ko, 0.0, w_out)
    if ko.any() and "purgatory" in out.columns:
        out.loc[ko, "purgatory"] = True
    tot = float(w_out.sum())
    if tot > 1e-12:
        w_out = w_out / tot
    info = {"n_dropped": int(drop_mask.sum()), "n_keep_open": int(keep_open_mask.sum())}
    return out, w_out, info


def apply_cluster_caps_to_b4(
    b4c: pd.DataFrame,
    w: np.ndarray,
    cluster_caps: dict | None,
) -> tuple[np.ndarray, dict]:
    """Cap each configured cluster's combined weight; redistribute to the rest."""
    w = np.asarray(w, dtype=float)
    info: dict = {}
    if b4c.empty or not cluster_caps or w.sum() <= 1e-12:
        return w, info
    unds = b4c["Underlying"].astype(str).map(_norm)
    for name, spec in cluster_caps.items():
        spec = spec or {}
        cap = float(spec.get("cap", 1.0))
        members = {_norm(m) for m in (spec.get("members") or [])}
        if not members or cap >= 1.0:
            continue
        in_cl = unds.isin(members).to_numpy()
        cl_w = float(w[in_cl].sum())
        if cl_w <= cap or not in_cl.any() or in_cl.all():
            info[name] = {"weight": cl_w, "capped": False}
            continue
        w = w.copy()
        w[in_cl] *= cap / cl_w
        rest = float(w[~in_cl].sum())
        if rest > 1e-12:
            w[~in_cl] *= (1.0 - cap) / rest
        info[name] = {"weight": cl_w, "capped": True, "cap": cap}
    return w, info


__all__ = ["apply_cluster_caps_to_b4", "apply_concentration_to_b4", "concentration_scores"]
