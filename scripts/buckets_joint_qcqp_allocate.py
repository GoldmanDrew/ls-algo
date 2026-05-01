# -*- coding: utf-8 -*-
"""Shared helpers for ``notebooks/Buckets1-4Backtest.ipynb`` joint QCQP + grid runs.

Builds the same sleeve splits and book-wide cap inversion as the notebook joint
cell, runs ``dcq.sizing.sizing_v2.pair_weights_qcqp_joint``, and merges outputs
to ``PAIR_WEIGHTS`` / ``PAIR_FRAC_BY_KEY``-shaped dicts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def load_notebook_cell_source(nb_path: Path | str, cell_id: str) -> str:
    """Return concatenated source for the first code cell with Jupyter ``id``."""
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for c in nb.get("cells", []):
        if c.get("id") == cell_id and c.get("cell_type") == "code":
            return "".join(c.get("source") or [])
    raise KeyError(f"code cell id={cell_id!r} not found in {nb_path}")


def build_joint_bundle(
    *,
    gtp_mirror_df: pd.DataFrame,
    gtp_mirror_diag: Mapping[str, Any],
    universe: list[tuple[Any, Any, Any]],
    prices: Mapping[str, Any],
    norm_sym: Callable[[Any], str],
    sleeve_order: tuple[str, ...] = (
        "core_leveraged",
        "whitelist_stock",
        "inverse_decay_bucket4",
    ),
    budget_key: Mapping[str, str] | None = None,
    price_min_obs: int = 65,
) -> dict[str, Any]:
    """Return bucket_universes, mirror dfs, sleeve_targets, and price frames."""
    budget_key = dict(budget_key or {})
    if not budget_key:
        budget_key = {
            "core_leveraged": "core_budget",
            "whitelist_stock": "wl_budget",
            "inverse_decay_bucket4": "b4_budget",
        }

    mdf = gtp_mirror_df.copy()
    mdf["ETF"] = mdf["ETF"].astype(str).map(norm_sym)
    mdf["Underlying"] = mdf["Underlying"].astype(str).map(norm_sym)
    gross = pd.to_numeric(mdf.get("gross_target_usd"), errors="coerce").fillna(0.0)
    universe_set = {(norm_sym(e), norm_sym(u)) for e, u, _ in universe}

    bucket_universes: dict[str, list[tuple[str, str, float]]] = {}
    bucket_mirror_dfs: dict[str, pd.DataFrame] = {}
    sleeve_targets: dict[str, float] = {}
    raw_budget: dict[str, float] = {}

    for sleeve in sleeve_order:
        mk = budget_key[sleeve]
        raw_budget[sleeve] = float(gtp_mirror_diag.get(mk, 0.0) or 0.0)
        sub = mdf.loc[(mdf["sleeve"].astype(str) == sleeve) & (gross > 0.0)].copy()
        if sub.empty or raw_budget[sleeve] <= 0:
            continue
        keys = {(norm_sym(r["ETF"]), norm_sym(r["Underlying"])) for _, r in sub.iterrows()}
        univ: list[tuple[str, str, float]] = []
        for e, u, b in universe:
            ek, uk = norm_sym(e), norm_sym(u)
            if (ek, uk) in keys and (ek, uk) in universe_set:
                univ.append((ek, uk, float(b)))
        if not univ:
            continue
        bucket_universes[sleeve] = univ
        bucket_mirror_dfs[sleeve] = sub
        sleeve_targets[sleeve] = raw_budget[sleeve]

    tot_b = float(sum(sleeve_targets.values()))
    if tot_b <= 0:
        raise RuntimeError("Joint QCQP: no sleeve budgets with gross>0 rows.")
    sleeve_targets = {k: float(v) / tot_b for k, v in sleeve_targets.items()}

    unds = sorted({u for sl in bucket_universes.values() for _, u, _ in sl})
    etfs = sorted({e for sl in bucket_universes.values() for e, _, _ in sl})

    def _price_frame(syms: list[str]) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        mo = int(price_min_obs)
        for s in syms:
            ser = prices.get(s)
            if ser is None:
                continue
            ss = pd.to_numeric(ser, errors="coerce").astype(float)
            if ss.dropna().size >= mo + 5:
                cols[s] = ss
        return pd.DataFrame(cols)

    underlying_returns = _price_frame(unds)
    etf_returns = _price_frame(etfs)
    return {
        "bucket_universes": bucket_universes,
        "bucket_mirror_dfs": bucket_mirror_dfs,
        "sleeve_targets": sleeve_targets,
        "underlying_returns": underlying_returns,
        "etf_returns": etf_returns,
    }


def caps_from_book_limits(
    sleeve_targets: Mapping[str, float],
    book_max_pair: float,
    book_max_under: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Invert DCQ's ``cap * sleeve_budget`` so caps apply to the full book."""
    bmw: dict[str, float] = {}
    bmu: dict[str, float] = {}
    for s, sb in sleeve_targets.items():
        sb_f = float(sb)
        if sb_f > 1e-12:
            bmw[str(s)] = float(book_max_pair) / sb_f
            bmu[str(s)] = float(book_max_under) / sb_f
    return bmw, bmu


def merge_joint_to_pair_globals(
    *,
    pair_frac_book: Mapping[str, Mapping[tuple[str, str], float]],
    pair_weights_by_bucket: Mapping[str, Mapping[str, float]],
    sleeve_targets: Mapping[str, float],
    norm_sym: Callable[[Any], str],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    etf_acc: dict[str, float] = {}
    for s, pw in pair_weights_by_bucket.items():
        ts = float(sleeve_targets.get(s, 0.0))
        for e, w in pw.items():
            e2 = norm_sym(e)
            etf_acc[e2] = etf_acc.get(e2, 0.0) + ts * float(w)
    s_et = float(sum(etf_acc.values()))
    pair_weights_joint = {e: v / s_et for e, v in etf_acc.items()} if s_et > 0 else {}

    pair_frac_joint: dict[tuple[str, str], float] = {}
    for s, pfd in pair_frac_book.items():
        ts = float(sleeve_targets.get(s, 0.0))
        for k, v in pfd.items():
            fv = float(v)
            if fv <= 0:
                continue
            kk = (norm_sym(k[0]), norm_sym(k[1]))
            pair_frac_joint[kk] = pair_frac_joint.get(kk, 0.0) + ts * fv
    s_pf = float(sum(pair_frac_joint.values()))
    if s_pf > 0:
        pair_frac_joint = {k: v / s_pf for k, v in pair_frac_joint.items()}
    return pair_weights_joint, pair_frac_joint


def run_joint_qcqp_single(
    bundle: Mapping[str, Any],
    pair_weights_qcqp_joint: Callable[..., Any],
    *,
    norm_sym: Callable[[Any], str],
    book_max_pair_weight: float,
    book_max_underlying_weight: float,
    book_sigma_target_annual: float,
    weight_ridge_lambda: float,
    w_min_floor_frac: float,
    score_method: str = "robust_sharpe",
    kelly_fraction: float = 0.35,
    sigma_floor: float = 0.04,
    sigma_shrink_cohort: float = 0.5,
    edge_blend_p25: float = 0.5,
    cov_lookback: int = 252,
    cov_shrink_alpha: float = 0.30,
    cov_min_obs: int = 60,
    spread_sigma_floor_annual: float = 0.05,
    mu_shrink_intensity: float = 0.0,
    fallback_to_proportional: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float], dict[tuple[str, str], float]]:
    st = dict(bundle["sleeve_targets"])
    bmw, bmu = caps_from_book_limits(st, book_max_pair_weight, book_max_underlying_weight)
    pair_frac_book, pair_weights_by_bucket, diag_by_bucket, joint_meta = pair_weights_qcqp_joint(
        bundle["bucket_universes"],
        bundle["bucket_mirror_dfs"],
        st,
        bundle["underlying_returns"],
        etf_returns=bundle["etf_returns"] if not bundle["etf_returns"].empty else None,
        score_method=score_method,
        kelly_fraction=kelly_fraction,
        sigma_floor=sigma_floor,
        edge_floor=0.0,
        sigma_shrink_cohort=sigma_shrink_cohort,
        edge_blend_p25=edge_blend_p25,
        cov_lookback=cov_lookback,
        cov_shrink_alpha=cov_shrink_alpha,
        cov_min_obs=cov_min_obs,
        spread_sigma_floor_annual=spread_sigma_floor_annual,
        book_sigma_target_annual=book_sigma_target_annual,
        bucket_max_pair_weight=bmw,
        bucket_max_underlying_weight=bmu,
        w_min_floor_frac=w_min_floor_frac,
        weight_ridge_lambda=weight_ridge_lambda,
        mu_shrink_intensity=mu_shrink_intensity,
        fallback_to_proportional=fallback_to_proportional,
    )
    pw_j, pf_j = merge_joint_to_pair_globals(
        pair_frac_book=pair_frac_book,
        pair_weights_by_bucket=pair_weights_by_bucket,
        sleeve_targets=st,
        norm_sym=norm_sym,
    )
    return pair_frac_book, pair_weights_by_bucket, diag_by_bucket, joint_meta, pw_j, pf_j
