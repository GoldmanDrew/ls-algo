# -*- coding: utf-8 -*-
"""Shared helpers for ``notebooks/Buckets1-4Backtest.ipynb`` joint QCQP + grid runs.

Builds the same sleeve splits and book-wide cap inversion as the notebook joint
cell, runs ``dcq.sizing.sizing_v2.pair_weights_qcqp_joint``, and merges outputs
to ``PAIR_WEIGHTS`` / ``PAIR_FRAC_BY_KEY``-shaped dicts.

Public entry points used by the notebook:

* ``build_joint_bundle(..., min_decay_obs=, min_beta_obs=)`` — history-quality
  gate on the mirror frame before QCQP.
* ``run_joint_qcqp_single`` — threads DCQ sizing-v2 kwargs (confidence haircut,
  turnover anchor, EMA overrides, per-bucket pair-cap overrides); unknown kwargs
  are filtered against the installed ``pair_weights_qcqp_joint`` signature.
* ``ema_smooth_signal`` — optional EMA on ``mu_used`` / ``sigma_eff`` across
  rebalances.
* ``filter_callable_kwargs`` — shared helper for forward-compatible kwargs.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def filter_callable_kwargs(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments the callable does not accept (older DCQ builds).

    New notebook / script code can pass stability knobs such as
    ``confidence_haircut``; if the installed ``pair_weights_qcqp_joint`` predates
    those parameters, filtering avoids ``TypeError`` while still running the
    core QCQP.
    """
    try:
        names = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in names}


def load_notebook_cell_source(nb_path: Path | str, cell_id: str) -> str:
    """Return concatenated source for the first code cell with Jupyter ``id``."""
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for c in nb.get("cells", []):
        if c.get("id") == cell_id and c.get("cell_type") == "code":
            return "".join(c.get("source") or [])
    raise KeyError(f"code cell id={cell_id!r} not found in {nb_path}")


def _apply_history_quality_gate(
    mdf: pd.DataFrame,
    *,
    min_decay_obs: int = 0,
    min_beta_obs: int = 0,
    decay_col: str = "expected_gross_decay_dist_n_obs",
    beta_col: str = "Beta_n_obs",
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Drop rows whose bootstrap depth is below the configured thresholds.

    Returns the filtered frame and the set of dropped ``(ETF, Underlying)``
    keys (already upper-cased / dot-replaced by the caller).
    """
    if min_decay_obs <= 0 and min_beta_obs <= 0:
        return mdf, set()
    dec = pd.to_numeric(mdf.get(decay_col), errors="coerce").fillna(0.0)
    bet = pd.to_numeric(mdf.get(beta_col), errors="coerce").fillna(0.0)
    keep = pd.Series(True, index=mdf.index)
    if min_decay_obs > 0:
        keep = keep & (dec >= float(min_decay_obs))
    if min_beta_obs > 0:
        keep = keep & (bet >= float(min_beta_obs))
    dropped = mdf.loc[~keep, ["ETF", "Underlying"]].astype(str)
    drop_set = {(str(e), str(u)) for e, u in zip(dropped["ETF"].values, dropped["Underlying"].values)}
    return mdf.loc[keep].copy(), drop_set


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
    min_decay_obs: int = 0,
    min_beta_obs: int = 0,
) -> dict[str, Any]:
    """Return bucket_universes, mirror dfs, sleeve_targets, and price frames.

    ``min_decay_obs`` / ``min_beta_obs`` (>0) drop rows whose
    ``expected_gross_decay_dist_n_obs`` / ``Beta_n_obs`` columns are below the
    threshold. This is the *history-quality gate* used to keep thin-bootstrap
    pairs (e.g. simple-ITO fallback) from winning the QCQP cap plateaus.
    """
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
    mdf, dropped_pairs = _apply_history_quality_gate(
        mdf, min_decay_obs=int(min_decay_obs), min_beta_obs=int(min_beta_obs)
    )
    gross = pd.to_numeric(mdf.get("gross_target_usd"), errors="coerce").fillna(0.0)
    universe_set = {(norm_sym(e), norm_sym(u)) for e, u, _ in universe}
    universe_set -= dropped_pairs  # pairs dropped by the gate cannot enter

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
        "n_pairs_dropped_by_history": int(len(dropped_pairs)),
        "dropped_by_history_keys": dropped_pairs,
        "min_decay_obs": int(min_decay_obs),
        "min_beta_obs": int(min_beta_obs),
    }


def caps_from_book_limits(
    sleeve_targets: Mapping[str, float],
    book_max_pair: float,
    book_max_under: float,
    *,
    sleeve_cap_max_pair_frac: float | None = None,
    sleeve_cap_max_underlying_frac: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Invert DCQ's ``cap * sleeve_budget`` so caps apply to the full book.

    Optional ``sleeve_cap_*`` collars clip the inverted per-sleeve multipliers so
    no **bucket** (same keys as ``sleeve_targets`` / DCQ ``bucket``) can put more
    than that **fraction of that bucket’s own target** on a single (ETF, underlying)
    pair or on a single underlying. In ``pair_weights_qcqp_joint`` this is enforced
    as ``w_pair <= cap_pair * s_b`` and ``sum_{pairs on u} w <= cap_under * s_b``
    with ``s_b`` that bucket’s book-weight share.
    """
    bmw: dict[str, float] = {}
    bmu: dict[str, float] = {}
    for s, sb in sleeve_targets.items():
        sb_f = float(sb)
        if sb_f > 1e-12:
            bmw[str(s)] = float(book_max_pair) / sb_f
            bmu[str(s)] = float(book_max_under) / sb_f
    if sleeve_cap_max_pair_frac is not None and float(sleeve_cap_max_pair_frac) > 0:
        cap = float(sleeve_cap_max_pair_frac)
        bmw = {k: min(float(v), cap) for k, v in bmw.items()}
    if sleeve_cap_max_underlying_frac is not None and float(sleeve_cap_max_underlying_frac) > 0:
        cap_u = float(sleeve_cap_max_underlying_frac)
        bmu = {k: min(float(v), cap_u) for k, v in bmu.items()}
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


def ema_smooth_signal(
    *,
    bucket_mirror_dfs: Mapping[str, pd.DataFrame],
    kelly_pre_weights_from_net_edge: Callable[..., Any],
    norm_sym: Callable[[Any], str],
    prev_signal: pd.DataFrame | None,
    halflife_weeks: float | None,
    confidence_haircut: bool = False,
    conf_floor: float = 0.25,
    n_obs_full: int = 252,
    score_method: str = "robust_sharpe",
    kelly_fraction: float = 0.35,
    sigma_floor: float = 0.04,
    sigma_shrink_cohort: float = 0.5,
    edge_blend_p25: float = 0.5,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], pd.DataFrame]:
    """Build EMA-smoothed ``mu_used`` / ``sigma_eff`` overrides for the QCQP.

    Returns ``(mu_override, sigma_override, current_signal)`` where the override
    maps are keyed by ``(ETF, Underlying)`` and ``current_signal`` is the
    untouched per-pair signal frame for the current cycle (used as
    ``prev_signal`` on the next call).

    When ``halflife_weeks`` is None or ``prev_signal`` is None, the override
    maps are empty (no-op) and the function still returns the current signal so
    the caller can persist it as the next anchor.
    """
    cur_pieces: list[pd.DataFrame] = []
    for b_id, mdf in bucket_mirror_dfs.items():
        if mdf is None or mdf.empty:
            continue
        sig = kelly_pre_weights_from_net_edge(
            mdf,
            **filter_callable_kwargs(
                kelly_pre_weights_from_net_edge,
                {
                    "kelly_fraction": kelly_fraction,
                    "sigma_floor": sigma_floor,
                    "score_method": score_method,
                    "sigma_shrink_cohort": sigma_shrink_cohort,
                    "edge_blend_p25": edge_blend_p25,
                    "confidence_haircut": confidence_haircut,
                    "conf_floor": conf_floor,
                    "n_obs_full": n_obs_full,
                },
            ),
        )
        sig = sig.copy()
        sig["ETF"] = sig["ETF"].astype(str).map(norm_sym)
        sig["Underlying"] = sig["Underlying"].astype(str).map(norm_sym)
        sig["bucket"] = str(b_id)
        cur_pieces.append(sig)
    if not cur_pieces:
        return {}, {}, pd.DataFrame()
    cur = pd.concat(cur_pieces, ignore_index=True)
    cur = cur.sort_values("w_pre", ascending=False).drop_duplicates(
        subset=["ETF", "Underlying"], keep="first"
    ).reset_index(drop=True)
    if prev_signal is None or prev_signal.empty or halflife_weeks is None or float(halflife_weeks) <= 0:
        return {}, {}, cur

    alpha = 1.0 - 0.5 ** (1.0 / float(halflife_weeks))  # weekly EMA weight on new value
    alpha = float(min(max(alpha, 0.0), 1.0))
    prev = prev_signal.copy()
    prev["ETF"] = prev["ETF"].astype(str).map(norm_sym)
    prev["Underlying"] = prev["Underlying"].astype(str).map(norm_sym)
    prev_mu = dict(zip(zip(prev["ETF"], prev["Underlying"]), prev.get("mu_used", pd.Series(dtype=float)).astype(float).values))
    prev_s = dict(zip(zip(prev["ETF"], prev["Underlying"]), prev.get("sigma_eff", pd.Series(dtype=float)).astype(float).values))
    mu_ov: dict[tuple[str, str], float] = {}
    s_ov: dict[tuple[str, str], float] = {}
    for _, r in cur.iterrows():
        k = (str(r["ETF"]), str(r["Underlying"]))
        cur_mu = float(r["mu_used"])
        cur_s = float(r["sigma_eff"])
        pmu = prev_mu.get(k)
        ps = prev_s.get(k)
        if pmu is not None and pd.notna(pmu):
            mu_ov[k] = alpha * cur_mu + (1.0 - alpha) * float(pmu)
        if ps is not None and pd.notna(ps):
            s_ov[k] = alpha * cur_s + (1.0 - alpha) * float(ps)
    return mu_ov, s_ov, cur


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
    sleeve_cap_max_pair_frac: float | None = None,
    sleeve_cap_max_underlying_frac: float | None = None,
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
    # New stability knobs (all off by default).
    confidence_haircut: bool = False,
    conf_floor: float = 0.25,
    n_obs_full: int = 252,
    turnover_lambda: float = 0.0,
    turnover_l1_max: float | None = None,
    prev_pair_weights: Mapping[tuple[str, str], float] | None = None,
    mu_used_override: Mapping[tuple[str, str], float] | None = None,
    sigma_eff_override: Mapping[tuple[str, str], float] | None = None,
    bucket_max_pair_weight_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float], dict[tuple[str, str], float]]:
    """Run a single joint QCQP solve.

    History-quality gating is performed up-stream by ``build_joint_bundle``;
    this entry point just plumbs the v2 stability flags into
    ``pair_weights_qcqp_joint``.
    """
    st = dict(bundle["sleeve_targets"])
    bmw, bmu = caps_from_book_limits(
        st,
        book_max_pair_weight,
        book_max_underlying_weight,
        sleeve_cap_max_pair_frac=sleeve_cap_max_pair_frac,
        sleeve_cap_max_underlying_frac=sleeve_cap_max_underlying_frac,
    )
    # Per-pair caps override (G8 score-aware caps): the override dict is keyed
    # by bucket and contains either a scalar (legacy) or a dict-of-pair-caps.
    if bucket_max_pair_weight_override:
        for k, v in bucket_max_pair_weight_override.items():
            bmw[str(k)] = v  # type: ignore[assignment]
    _full_joint_kw: dict[str, Any] = {
        "bucket_universes": bundle["bucket_universes"],
        "bucket_mirror_dfs": bundle["bucket_mirror_dfs"],
        "sleeve_targets": st,
        "underlying_returns": bundle["underlying_returns"],
        "etf_returns": bundle["etf_returns"] if not bundle["etf_returns"].empty else None,
        "score_method": score_method,
        "kelly_fraction": kelly_fraction,
        "sigma_floor": sigma_floor,
        "edge_floor": 0.0,
        "sigma_shrink_cohort": sigma_shrink_cohort,
        "edge_blend_p25": edge_blend_p25,
        "cov_lookback": cov_lookback,
        "cov_shrink_alpha": cov_shrink_alpha,
        "cov_min_obs": cov_min_obs,
        "spread_sigma_floor_annual": spread_sigma_floor_annual,
        "book_sigma_target_annual": book_sigma_target_annual,
        "bucket_max_pair_weight": bmw,
        "bucket_max_underlying_weight": bmu,
        "w_min_floor_frac": w_min_floor_frac,
        "weight_ridge_lambda": weight_ridge_lambda,
        "mu_shrink_intensity": mu_shrink_intensity,
        "fallback_to_proportional": fallback_to_proportional,
        "confidence_haircut": confidence_haircut,
        "conf_floor": conf_floor,
        "n_obs_full": n_obs_full,
        "turnover_lambda": turnover_lambda,
        "turnover_l1_max": turnover_l1_max,
        "prev_pair_weights": dict(prev_pair_weights) if prev_pair_weights else None,
        "mu_used_override": dict(mu_used_override) if mu_used_override else None,
        "sigma_eff_override": dict(sigma_eff_override) if sigma_eff_override else None,
    }
    _joint_kw = filter_callable_kwargs(pair_weights_qcqp_joint, _full_joint_kw)
    _dropped = set(_full_joint_kw) - set(_joint_kw)
    if _dropped:
        import warnings

        warnings.warn(
            "Installed DCQ pair_weights_qcqp_joint dropped kwargs "
            f"{sorted(_dropped)} — pull latest Diamond-Creek-Quant for sizing-v2 stability.",
            stacklevel=2,
        )
    pair_frac_book, pair_weights_by_bucket, diag_by_bucket, joint_meta = pair_weights_qcqp_joint(
        **_joint_kw
    )
    # Surface bundle-level history-gate diagnostics into joint_meta so the
    # notebook can log how many pairs were dropped this cycle.
    if isinstance(joint_meta, dict):
        joint_meta.setdefault("n_pairs_dropped_by_history", int(bundle.get("n_pairs_dropped_by_history", 0)))
        joint_meta.setdefault("min_decay_obs", int(bundle.get("min_decay_obs", 0)))
        joint_meta.setdefault("min_beta_obs", int(bundle.get("min_beta_obs", 0)))
    pw_j, pf_j = merge_joint_to_pair_globals(
        pair_frac_book=pair_frac_book,
        pair_weights_by_bucket=pair_weights_by_bucket,
        sleeve_targets=st,
        norm_sym=norm_sym,
    )
    return pair_frac_book, pair_weights_by_bucket, diag_by_bucket, joint_meta, pw_j, pf_j
