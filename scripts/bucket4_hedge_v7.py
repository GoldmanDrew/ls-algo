"""
Bucket 4 dynamic_v7 hedge ratio: VCR-driven h with pair-specific bounds in [0.3, 0.8].

Higher variance contribution ratio (VCR) => higher hedge ratio h (more underlying short).
Built for ``Bucket4_PerPair_IBKR_Hedge_Grid.ipynb``; consumed as a daily ``pd.Series`` by
``run_bucket4_backtest_dynamic_h``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

V7_GLOBAL_H_MIN = 0.30
V7_GLOBAL_H_MAX = 0.80
V7_DEFAULT_H_MID = 0.55
V7_DEFAULT_K_VCR = 1.0
V7_DEFAULT_SMOOTH_ALPHA = 0.25


def _default_norm(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def resolve_pair_h_bounds(
    etf: str,
    und: str,
    *,
    pair_bounds: dict[tuple[str, str], tuple[float, float]] | None = None,
    global_h_min: float = V7_GLOBAL_H_MIN,
    global_h_max: float = V7_GLOBAL_H_MAX,
    default_bounds: tuple[float, float] = (V7_GLOBAL_H_MIN, V7_GLOBAL_H_MAX),
    norm_sym: Callable[[str], str] = _default_norm,
) -> tuple[float, float]:
    """Pair (h_min, h_max) clipped to the global envelope."""
    key = (norm_sym(etf), norm_sym(und))
    lo, hi = (pair_bounds or {}).get(key, default_bounds)
    lo = max(float(global_h_min), min(float(lo), float(global_h_max)))
    hi = max(float(global_h_min), min(float(hi), float(global_h_max)))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def vcr_to_h_star(
    vcr: float,
    vcr_med: float,
    *,
    h_mid: float = V7_DEFAULT_H_MID,
    k_vcr: float = V7_DEFAULT_K_VCR,
    use_sigma_scale: bool = False,
    vcr_sigma: float = 0.05,
) -> float:
    """Map VCR level vs baseline to raw hedge target before pair/global clip."""
    if not np.isfinite(vcr):
        return float(h_mid)
    baseline = float(vcr_med) if np.isfinite(vcr_med) else float(vcr)
    delta = float(vcr) - baseline
    if use_sigma_scale and vcr_sigma > 1e-9:
        z = delta / float(vcr_sigma)
        return float(h_mid + k_vcr * z)
    return float(h_mid + k_vcr * delta)


def build_h_series_v7(
    signal: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    *,
    h_min: float,
    h_max: float,
    h_mid: float = V7_DEFAULT_H_MID,
    k_vcr: float = V7_DEFAULT_K_VCR,
    global_h_min: float = V7_GLOBAL_H_MIN,
    global_h_max: float = V7_GLOBAL_H_MAX,
    smooth_alpha: float = V7_DEFAULT_SMOOTH_ALPHA,
    use_sigma_scale: bool = False,
    vcr_sigma: float = 0.05,
) -> pd.Series:
    """
    Daily hedge ratio on *price_index*, aligned to *signal* (tr/vcr/vcr_med), ffill + optional EWM smooth.

    Parameters
    ----------
    signal
        Output of ``get_pair_signal`` (columns ``vcr``, ``vcr_med``; already lookahead-shifted).
    h_min, h_max
        Pair-specific bounds (already intersected with global envelope by caller).
    smooth_alpha
        If in (0, 1], apply exponential smoothing on the raw clipped series after ffill.
        If 0 or None, use ffill only.
    """
    idx = pd.DatetimeIndex(price_index).sort_values()
    if len(idx) == 0:
        return pd.Series(dtype=float)

    vcr = signal.get("vcr") if signal is not None else None
    vcr_med = signal.get("vcr_med") if signal is not None else None
    if vcr is None:
        out = pd.Series(float(h_mid), index=idx, dtype=float)
        return out.clip(float(global_h_min), float(global_h_max))

    vcr_a = vcr.reindex(idx).ffill()
    med_a = (
        vcr_med.reindex(idx).ffill()
        if vcr_med is not None
        else vcr_a.expanding(min_periods=1).median()
    )

    h_star = pd.Series(index=idx, dtype=float)
    for d in idx:
        h_star.loc[d] = vcr_to_h_star(
            float(vcr_a.loc[d]) if pd.notna(vcr_a.loc[d]) else np.nan,
            float(med_a.loc[d]) if pd.notna(med_a.loc[d]) else np.nan,
            h_mid=float(h_mid),
            k_vcr=float(k_vcr),
            use_sigma_scale=use_sigma_scale,
            vcr_sigma=vcr_sigma,
        )

    h_clip = h_star.clip(float(h_min), float(h_max)).clip(float(global_h_min), float(global_h_max))
    h_clip = h_clip.fillna(float(h_mid)).clip(float(global_h_min), float(global_h_max))

    alpha = float(smooth_alpha) if smooth_alpha is not None else 0.0
    if 0.0 < alpha < 1.0:
        smoothed = h_clip.copy()
        prev = float(h_clip.iloc[0]) if len(h_clip) else float(h_mid)
        for i, d in enumerate(idx):
            target = float(h_clip.iloc[i])
            prev = (1.0 - alpha) * prev + alpha * target
            smoothed.iloc[i] = prev
        h_clip = smoothed.clip(float(h_min), float(h_max)).clip(float(global_h_min), float(global_h_max))

    return h_clip.astype(float)


def build_h_v7_by_pair(
    signals_by_key: dict[tuple[str, str], pd.DataFrame],
    pair_cache: dict[tuple[str, str], dict[str, Any]],
    *,
    start_sim: pd.Timestamp | str,
    pair_bounds: dict[tuple[str, str], tuple[float, float]] | None = None,
    h_mid: float = V7_DEFAULT_H_MID,
    k_vcr: float = V7_DEFAULT_K_VCR,
    global_h_min: float = V7_GLOBAL_H_MIN,
    global_h_max: float = V7_GLOBAL_H_MAX,
    smooth_alpha: float = V7_DEFAULT_SMOOTH_ALPHA,
    norm_sym: Callable[[str], str] = _default_norm,
) -> dict[tuple[str, str], pd.Series]:
    """Build ``{(etf, und): h_series}`` for all keys with signals and prices."""
    _start = pd.Timestamp(start_sim)
    out: dict[tuple[str, str], pd.Series] = {}
    for key, sig in signals_by_key.items():
        if key not in pair_cache or "skip_reason" in pair_cache[key]:
            continue
        prices = pair_cache[key].get("prices")
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            continue
        cal = prices.loc[prices.index >= _start].index
        if len(cal) == 0:
            continue
        etf, und = key
        lo, hi = resolve_pair_h_bounds(
            etf,
            und,
            pair_bounds=pair_bounds,
            global_h_min=global_h_min,
            global_h_max=global_h_max,
            norm_sym=norm_sym,
        )
        out[key] = build_h_series_v7(
            sig,
            cal,
            h_min=lo,
            h_max=hi,
            h_mid=h_mid,
            k_vcr=k_vcr,
            global_h_min=global_h_min,
            global_h_max=global_h_max,
            smooth_alpha=smooth_alpha,
        )
    return out


def default_v7_pair_bounds_from_screened(
    screened: pd.DataFrame,
    candidate_keys: list[tuple[str, str]],
    *,
    global_h_min: float = V7_GLOBAL_H_MIN,
    global_h_max: float = V7_GLOBAL_H_MAX,
    high_borrow_cap_h_max: float = 0.65,
    borrow_threshold: float = 0.50,
    norm_sym: Callable[[str], str] = _default_norm,
) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Heuristic pair bounds for v6 candidates: high borrow => lower h_max; else full [0.3, 0.8].
    """
    borrow_col = None
    for c in ("borrow_fee_annual", "borrow_avg_annual", "borrow_current"):
        if c in screened.columns:
            borrow_col = c
            break
    etf_to_borrow: dict[str, float] = {}
    if borrow_col:
        tmp = screened.copy()
        tmp["ETF_N"] = tmp["ETF"].astype(str).map(norm_sym)
        tmp["_b"] = pd.to_numeric(tmp[borrow_col], errors="coerce")
        for etf, grp in tmp.groupby("ETF_N"):
            v = grp["_b"].max()
            if np.isfinite(v):
                etf_to_borrow[etf] = float(v)

    out: dict[tuple[str, str], tuple[float, float]] = {}
    for etf, und in candidate_keys:
        e = norm_sym(etf)
        b = etf_to_borrow.get(e, 0.0)
        if b >= float(borrow_threshold):
            out[(e, norm_sym(und))] = (float(global_h_min), float(min(high_borrow_cap_h_max, global_h_max)))
        else:
            out[(e, norm_sym(und))] = (float(global_h_min), float(global_h_max))
    return out


__all__ = [
    "V7_GLOBAL_H_MIN",
    "V7_GLOBAL_H_MAX",
    "V7_DEFAULT_H_MID",
    "V7_DEFAULT_K_VCR",
    "V7_DEFAULT_SMOOTH_ALPHA",
    "resolve_pair_h_bounds",
    "vcr_to_h_star",
    "build_h_series_v7",
    "build_h_v7_by_pair",
    "default_v7_pair_bounds_from_screened",
]
