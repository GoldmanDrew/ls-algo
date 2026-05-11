"""
Tail-risk-weighted Bucket 4 portfolio aggregation (matches ``Bucket_4_Backtest.ipynb``
cell 17 ``_run_weighted_portfolio`` logic for a single weight dict).

Each live pair runs ``run_bucket4_backtest_dynamic_h`` with
``initial_capital = weight * pf_initial`` using each pair's resolved ETF borrow rate,
then equities are summed on the union calendar from ``start_sim`` onward.

**Portfolio ``nav`` is the sum of sub-account book values.** When a new pair’s
history starts later on the union index, that pair’s book appears as a **level step**
in total NAV (capital allocated to that pair), not trading PnL. Dashboard charts
that use ``build_b4_attribution_from_pair_bts`` strip those inception steps from
attributed *trading* flows; the raw NAV series still shows funding jumps.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from scripts import bucket4_dynamic_bt as _b4_dyn

V6_OPT2_H_BASE = _b4_dyn.V6_OPT2_H_BASE


def _nonneg_borrow_annual(x: float) -> float:
    v = float(x)
    if not np.isfinite(v) or v < 0.0:
        return 0.0
    return float(v)


def _etf_borrow_annual_actual(
    etf_sym: str,
    kw0: dict[str, Any],
    *,
    uvix_borrow_annual_base: float | None,
    norm_sym: Callable[[str], str],
) -> float:
    """Mirror ``Bucket_4_Backtest.ipynb`` cell 17 ``_etf_borrow_annual_actual``."""
    if uvix_borrow_annual_base is not None and norm_sym(str(etf_sym)) == "UVIX":
        return float(uvix_borrow_annual_base)
    return _nonneg_borrow_annual(float(kw0.get("borrow_a_annual", 0.0)))


def aggregate_tail_risk_weighted_portfolio(
    pair_cache: dict[tuple[str, str], dict[str, Any]],
    v6_opt2_h_daily_map: dict[str, pd.Series],
    v6_opt2_rebal_index: pd.DatetimeIndex | None,
    weight_by_key: dict[tuple[str, str], float],
    *,
    start_sim: pd.Timestamp | str,
    pf_initial: float,
    opt2_h_base: float | None = None,
    get_ibkr_borrow_map: Callable[[list[str]], dict[str, float]] | None = None,
    norm_sym: Callable[[str], str] | None = None,
    use_ibkr_uvix_borrow: bool = True,
    drift_threshold_share_of_gross: float | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """
    Parameters
    ----------
    pf_initial
        Total capital across the B4 book (each pair gets ``weight * pf_initial``).
    get_ibkr_borrow_map
        When provided, used like the Bucket 4 notebook to resolve **UVIX** borrow for
        ``_etf_borrow_annual_actual``. Pass the same callable the notebook uses (e.g.
        ``get_ibkr_borrow_map`` / v6 backtest wrapper) for full parity.
    norm_sym
        Symbol normalizer (defaults to upper/strip/replace dot with dash).
    drift_threshold_share_of_gross
        When set (e.g. ``0.05``), forwarded to each per-pair ``run_bucket4_backtest_dynamic_h``
        so a scheduled rebalance only fires if the leg-share-of-gross has drifted from the
        target h-ratio split by more than the threshold (see that function's docstring).
        If the loaded engine predates that parameter (stale Jupyter import), the kwarg is
        omitted and a one-time warning is emitted.
    """
    _norm = norm_sym or (lambda x: str(x).strip().upper().replace(".", "-"))
    _run_engine = _b4_dyn.run_bucket4_backtest_dynamic_h
    _engine_accepts_drift = "drift_threshold_share_of_gross" in inspect.signature(_run_engine).parameters
    _warned_stale_engine = False

    uvix_borrow_annual_base: float | None = None
    if use_ibkr_uvix_borrow and get_ibkr_borrow_map is not None:
        try:
            m = get_ibkr_borrow_map(["UVIX"])
            v = m.get("UVIX") if isinstance(m, dict) else None
            if v is not None and np.isfinite(float(v)) and float(v) > 0:
                uvix_borrow_annual_base = _nonneg_borrow_annual(float(v))
        except Exception:
            uvix_borrow_annual_base = None

    hb = float(opt2_h_base if opt2_h_base is not None else V6_OPT2_H_BASE)
    _start_sim = pd.Timestamp(start_sim)

    if v6_opt2_rebal_index is None or len(v6_opt2_rebal_index) == 0:
        raise ValueError("v6_opt2_rebal_index is required for portfolio aggregation")

    rebal = pd.DatetimeIndex(v6_opt2_rebal_index)

    bt_by_pair: dict[tuple[str, str], pd.DataFrame] = {}
    for key, w in weight_by_key.items():
        if key not in pair_cache or "skip_reason" in pair_cache[key]:
            continue
        etf_sym, und_sym = key
        if und_sym not in v6_opt2_h_daily_map:
            continue
        w_pair = float(w)
        if w_pair <= 0.0:
            continue
        c = pair_cache[key]
        prices_full = c["prices"]
        prices_i = prices_full.loc[prices_full.index >= _start_sim]
        if prices_i.empty:
            continue
        kw = dict(c["kw"])
        base_borrow = _etf_borrow_annual_actual(
            etf_sym, kw, uvix_borrow_annual_base=uvix_borrow_annual_base, norm_sym=_norm
        )
        kw["initial_capital"] = w_pair * float(pf_initial)
        kw["borrow_a_annual"] = base_borrow

        h_d = v6_opt2_h_daily_map[und_sym].reindex(prices_i.index).ffill().fillna(hb)
        pair_rebal = rebal.intersection(prices_i.index)
        if len(pair_rebal) == 0:
            pair_rebal = pd.DatetimeIndex([prices_i.index[0]])

        # Notebook cell 17: ``run_bucket4_backtest_dynamic_h(prices_i, h_d, pair_rebal, **kw)``
        kw.pop("drift_threshold_share_of_gross", None)
        if drift_threshold_share_of_gross is not None and _engine_accepts_drift:
            kw["drift_threshold_share_of_gross"] = float(drift_threshold_share_of_gross)
        elif drift_threshold_share_of_gross is not None and not _engine_accepts_drift and not _warned_stale_engine:
            warnings.warn(
                "EXP['b4_drift_threshold_share_of_gross'] is set but the loaded "
                "run_bucket4_backtest_dynamic_h has no 'drift_threshold_share_of_gross' "
                "parameter (stale import). Run: import importlib; "
                "import scripts.bucket4_dynamic_bt as m; importlib.reload(m). "
                "Reverting to scheduling a rebalance on every calendar date.",
                UserWarning,
                stacklevel=2,
            )
            _warned_stale_engine = True
        bt_by_pair[key] = _run_engine(prices_i, h_d, pair_rebal, **kw)

    if not bt_by_pair:
        return pd.DataFrame(), {}

    all_idx = pd.DatetimeIndex(
        sorted({pd.Timestamp(d) for k in bt_by_pair for d in bt_by_pair[k].index if pd.Timestamp(d) >= _start_sim})
    )
    if len(all_idx) < 5:
        raise RuntimeError("Too few calendar days in union index for portfolio aggregation.")

    port_equity = pd.Series(0.0, index=all_idx, dtype=float)
    agg_borrow_pf = pd.Series(0.0, index=all_idx, dtype=float)
    for k, bt in bt_by_pair.items():
        first_p = bt.index[0]
        eq = bt["equity"].reindex(all_idx).where(all_idx >= first_p, 0.0).ffill().fillna(0.0).astype(float)
        port_equity = port_equity.add(eq, fill_value=0.0)
        br = bt["borrow_cost"].reindex(all_idx).fillna(0.0).where(all_idx >= first_p, 0.0).astype(float)
        agg_borrow_pf = agg_borrow_pf.add(br, fill_value=0.0)

    bt_pf = pd.DataFrame(
        {
            "equity": port_equity,
            "nav": port_equity,
            "ret": port_equity.pct_change().fillna(0.0),
            "drawdown": port_equity / port_equity.cummax() - 1.0,
            "beta_notional": 0.0,
            "borrow_cost": agg_borrow_pf,
            "rebalance": False,
            "short_proceeds_credit": 0.0,
            "financing_pnl": 0.0,
        }
    )
    return bt_pf, bt_by_pair
