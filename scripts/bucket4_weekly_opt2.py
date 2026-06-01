"""
Bucket 4 weekly Opt-2 engine (research + production hooks).

Weekly scheduled rebalances, optional drift thresholds, dynamic hedge ratios,
and tail-risk / covariance weights aligned with ``scripts/v6_b4_pf_weights.py``.

**Notebook alignment (applied in-repo):**

- ``notebooks/Bucket_4_Backtest.ipynb`` cell **2** defines ``V6_OPT2_WEEKLY_REBAL_FREQ`` (default
  ``\"W-FRI\"``). Cell **14** imports ``weekly_rebalance_dates`` and builds the hedge panel on
  that weekly calendar instead of ``V6_BDAY_STEP`` business-day skips. Cell **17** starts with a
  pointer to :func:`run_bucket4_backtest` for drift + weekly aggregation outside the large inline
  research cell.

- ``notebooks/Buckets1-4_v2.ipynb`` cell **6** adds ``_maybe_resync_v6_opt2_rebal_from_weekly_module``:
  set ``EXP[\"b4_resync_v6_rebal_weekly\"] = True`` to overwrite a **pickled** 10-day
  ``v6_opt2_rebal_index`` with the same weekly calendar as cell 14 (uses ``closes_broad`` or the
  union of pair price indices). Optional EXP keys: ``b4_weekly_warmup_bdays``, ``b4_weekly_rebalance_freq``,
  ``b4_weights_use_ibkr_uvix_borrow`` (passed into ``compute_v6_b4_pf_weight_dict``).

- Re-run idempotently: ``python scripts/apply_b4_notebook_patches.py`` from the repo root.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scripts.bucket4_dynamic_bt import V6_OPT2_H_BASE, run_bucket4_backtest_dynamic_h
from scripts.bucket4_price_loading import (
    configure_price_cache_dirs,
    load_beta_values,
    load_pair_borrow_rates,
    load_prices,
    load_single_close,
    perf_stats,
)
from scripts.v6_b4_pf_weights import V6PfParams, compute_v6_b4_pf_weight_dict

def norm_sym_gtp(x: str) -> str:
    """Match ``generate_trade_plan._norm_sym`` (upper, strip, dot→dash)."""
    return str(x).strip().upper().replace(".", "-")


def norm_sym_nb(x: str) -> str:
    """Notebook / screener row convention (upper + strip)."""
    return str(x).strip().upper()


def robust_z_cross_sectional(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    m = v.median(skipna=True)
    mad = (v - m).abs().median(skipna=True)
    scale = 1.4826 * float(mad) if pd.notna(mad) and mad > 0 else float(v.std(skipna=True) or 1.0)
    z = (v - m) / scale if scale > 0 else v * 0.0
    return z.clip(lower=-3.0, upper=3.0)


def weekly_rebalance_dates(
    trading_index: pd.DatetimeIndex,
    weekly_rebalance_freq: str = "W-FRI",
    *,
    warmup_bdays: int = 0,
) -> pd.DatetimeIndex:
    """
    Last available trading stamp in each resample bucket (pandas ``freq`` string),
    after optional warmup skip of the first ``warmup_bdays`` rows of ``trading_index``.
    """
    ix = pd.DatetimeIndex(trading_index).sort_values().unique()
    if ix.tz is not None:
        ix = ix.tz_convert("UTC").tz_localize(None)
    if warmup_bdays > 0:
        ix = ix[int(warmup_bdays) :]
    if len(ix) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(1.0, index=ix)
    ends = s.resample(weekly_rebalance_freq, label="right", closed="right").last().index
    out: list[pd.Timestamp] = []
    for w_end in ends:
        elig = ix[ix <= pd.Timestamp(w_end)]
        if len(elig) > 0:
            out.append(pd.Timestamp(elig[-1]))
    return pd.DatetimeIndex(sorted(set(out)))


def _opt2_price_features(c: pd.Series) -> pd.DataFrame:
    c = c.astype(float)
    logret = np.log(c / c.shift(1))
    sigma5 = logret.rolling(5, min_periods=5).std(ddof=1)
    sigma63 = logret.rolling(63, min_periods=30).std(ddof=1)
    out = pd.DataFrame(index=c.index)
    out["r_10d"] = np.log(c / c.shift(10))
    out["range_expansion"] = sigma5 / sigma63
    return out


def _get_macro_series(
    symbol: str,
    master_index: pd.DatetimeIndex,
    *,
    macro: dict[str, pd.Series] | None,
    fetch_close: Callable[[str, str], pd.Series] | None,
) -> pd.Series | None:
    s = None
    if isinstance(macro, dict):
        s = macro.get(symbol)
    if s is None or len(getattr(s, "dropna", lambda: s)()) < 20:
        if fetch_close is None:
            return None
        try:
            s = fetch_close(symbol, "max")
        except Exception:
            s = None
    if s is None or len(s.dropna()) < 20:
        return None
    return s.reindex(master_index).ffill()


def _vix_feature_series(
    master_index: pd.DatetimeIndex,
    feature: str,
    vix_symbol: str,
    *,
    macro: dict[str, pd.Series] | None,
    fetch_close: Callable[[str, str], pd.Series] | None,
) -> tuple[pd.Series | None, str | None]:
    vx = _get_macro_series(vix_symbol, master_index, macro=macro, fetch_close=fetch_close)
    if vx is None:
        return None, None
    lv = np.log(vx.astype(float))
    if feature == "dlog5":
        return (lv - lv.shift(5)), "ge"
    if feature == "dlog21":
        return (lv - lv.shift(21)), "ge"
    if feature == "level_z":
        win = 252
        med = vx.rolling(win, min_periods=max(60, win // 4)).median()
        mad = (vx - med).abs().rolling(win, min_periods=max(60, win // 4)).median()
        scale = (1.4826 * mad).where(mad > 0, vx.rolling(win, min_periods=max(60, win // 4)).std(ddof=1))
        return (vx - med) / scale.replace(0.0, np.nan), "ge"
    if feature == "high_prox":
        hi = vx.rolling(252, min_periods=60).max()
        return (lv - np.log(hi)), "ge"
    if feature == "term_ratio":
        v3m = _get_macro_series("^VIX3M", master_index, macro=macro, fetch_close=fetch_close)
        if v3m is None:
            v3m = vx.rolling(63, min_periods=20).mean()
        return (v3m.astype(float) / vx.astype(float).replace(0.0, np.nan)), "le"
    if feature == "vvix_dlog5":
        vv = _get_macro_series("^VVIX", master_index, macro=macro, fetch_close=fetch_close)
        if vv is None:
            return None, None
        lvv = np.log(vv.astype(float))
        return (lvv - lvv.shift(5)), "ge"
    return (lv - lv.shift(5)), "ge"


def _underlying_vix_beta_panel(
    closes: pd.DataFrame,
    underlyings: list[str],
    master_index: pd.DatetimeIndex,
    vix_symbol: str,
    window: int,
) -> pd.DataFrame:
    vx = _get_macro_series(vix_symbol, master_index, macro=None, fetch_close=None)
    if vx is None and vix_symbol in closes.columns:
        vx = closes[vix_symbol] if vix_symbol in closes.columns else None
    if vx is None:
        return pd.DataFrame(index=master_index, columns=underlyings, dtype=float)
    dvix = np.log(vx.astype(float)).diff()
    out = pd.DataFrame(index=master_index, columns=underlyings, dtype=float)
    win = max(60, int(window))
    for u in underlyings:
        if u not in closes.columns:
            continue
        ru = np.log(closes[u].astype(float)).diff().reindex(master_index)
        dvix_r = dvix.reindex(master_index)
        corr = ru.rolling(win, min_periods=max(60, win // 4)).corr(dvix_r)
        out[u] = corr.clip(lower=-1.0, upper=1.0).abs()
    return out


def _underlying_drawdown_panel(closes: pd.DataFrame, underlyings: list[str], lookback: int) -> pd.DataFrame:
    cols = [u for u in underlyings if u in closes.columns]
    if not cols:
        return pd.DataFrame(index=closes.index)
    sub = closes[cols].astype(float)
    peak = sub.rolling(int(lookback), min_periods=max(10, int(lookback) // 4)).max()
    return sub / peak - 1.0


def _macro_trailing_dd(master_index: pd.DatetimeIndex, symbol: str, lookback: int, *, macro: Any) -> pd.Series | None:
    s = _get_macro_series(symbol, master_index, macro=macro, fetch_close=None)
    if s is None or len(s.dropna()) < 20:
        return None
    s = s.astype(float)
    lb = int(lookback)
    roll_max = s.rolling(lb, min_periods=max(10, lb // 4)).max()
    return s / roll_max - 1.0


def _macro_logret_n(master_index: pd.DatetimeIndex, symbol: str, n: int, *, macro: Any) -> pd.Series | None:
    s = _get_macro_series(symbol, master_index, macro=macro, fetch_close=None)
    if s is None or len(s.dropna()) < int(n) + 5:
        return None
    s = s.astype(float)
    return np.log(s / s.shift(int(n)))


def _macro_vol_ratio(master_index: pd.DatetimeIndex, symbol: str, *, macro: Any) -> pd.Series | None:
    s = _get_macro_series(symbol, master_index, macro=macro, fetch_close=None)
    if s is None or len(s.dropna()) < 65:
        return None
    logret = np.log(s.astype(float) / s.shift(1))
    sigma5 = logret.rolling(5, min_periods=5).std(ddof=1)
    sigma63 = logret.rolling(63, min_periods=30).std(ddof=1)
    return sigma5 / sigma63.replace(0.0, np.nan)


def panel_to_hedge_by_underlying(
    panel: pd.DataFrame | None,
    master_index: pd.DatetimeIndex,
    underlyings: list[str],
    hedge_base: float,
) -> dict[str, pd.Series]:
    if panel is None or panel.empty:
        return {u: pd.Series(float(hedge_base), index=master_index) for u in underlyings}
    pivot = panel.pivot_table(index="date", columns="underlying", values="h_applied", aggfunc="last")
    out: dict[str, pd.Series] = {}
    for u in underlyings:
        s = pivot[u].reindex(master_index).ffill() if u in pivot.columns else pd.Series(np.nan, index=master_index)
        out[u] = s.fillna(float(hedge_base))
    return out


def build_hedge_panel_opt2(
    closes_broad: pd.DataFrame,
    bucket4_pairs: Sequence[tuple[str, str]],
    *,
    weekly_rebalance_freq: str = "W-FRI",
    warmup_bdays: int = 65,
    hedge_base: float = V6_OPT2_H_BASE,
    opt2_k: float = 0.05,
    opt2_alpha: float = 0.25,
    h_min: float = 0.10,
    h_max: float = 1.10,
    min_xsec: int = 5,
    overlay: Mapping[str, Any] | None = None,
    macro: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    """Dynamic hedge panel; scheduled rows follow ``weekly_rebalance_freq`` (not 10 bday steps)."""
    p = dict(DEFAULT_OVERLAY_NOTEBOOK)
    if overlay:
        p.update(dict(overlay))
    all_dates = closes_broad.index.sort_values()
    b4_unds = sorted({u for _, u in bucket4_pairs} & set(closes_broad.columns))
    if not b4_unds:
        raise RuntimeError("No B4 underlyings intersect closes_broad.")
    feats = {u: _opt2_price_features(closes_broad[u]) for u in b4_unds}
    rebal = weekly_rebalance_dates(all_dates, weekly_rebalance_freq, warmup_bdays=warmup_bdays)
    h_prev = {u: float(hedge_base) for u in b4_unds}
    rows: list[dict[str, Any]] = []

    use_regime = bool(p["regime_overlay_enable"])
    vix_feature = str(p.get("vix_feature", "dlog5"))
    _vix_thr_by_feature = {
        "dlog5": float(p["vix_r5d_logret_min"]),
        "dlog21": float(p["vix_r5d_logret_min"]),
        "level_z": float(p["vix_r5d_logret_min"]),
        "high_prox": float(p.get("vix_high_prox_max", -0.05)),
        "term_ratio": float(p.get("vix_term_ratio_max", 1.00)),
        "vvix_dlog5": float(p.get("vix_vvix_dlog5_min", 0.15)),
    }
    vix_ser, vix_dir = (
        _vix_feature_series(all_dates, vix_feature, p["vix_symbol"], macro=macro, fetch_close=None)
        if use_regime and p["vix_shock_enable"]
        else (None, None)
    )
    vix_beta_panel = (
        _underlying_vix_beta_panel(closes_broad, b4_unds, all_dates, p["vix_symbol"], window=int(p.get("vix_beta_window", 252)))
        if use_regime and p["vix_shock_enable"] and float(p.get("vix_delta_weight", 0.0)) > 0.0
        else None
    )
    dd_panel = (
        _underlying_drawdown_panel(closes_broad, b4_unds, int(p["dd_lookback"]))
        if use_regime and p["dd_cut_enable"]
        else None
    )
    spy_dd_ser = (
        _macro_trailing_dd(all_dates, str(p.get("spy_dd_symbol", "SPY")), int(p.get("spy_dd_lookback", 63)), macro=macro)
        if use_regime and bool(p.get("spy_dd_stress_enable", False))
        else None
    )
    hyg_r21_ser = (
        _macro_logret_n(all_dates, str(p.get("hyg_symbol", "HYG")), int(p.get("hyg_logret_window", 21)), macro=macro)
        if use_regime and bool(p.get("hyg_21d_stress_enable", False))
        else None
    )
    spy_vol_ser = (
        _macro_vol_ratio(all_dates, str(p.get("spy_vol_symbol", "SPY")), macro=macro)
        if use_regime and bool(p.get("spy_index_vol_shock_enable", False))
        else None
    )

    for as_of in rebal:
        v10, vrx = {}, {}
        for u, ts in feats.items():
            if as_of not in ts.index:
                continue
            t10, trx = ts.at[as_of, "r_10d"], ts.at[as_of, "range_expansion"]
            if pd.notna(t10):
                v10[u] = float(t10)
            if pd.notna(trx):
                vrx[u] = float(trx)
        if len(v10) < min_xsec or len(vrx) < min_xsec:
            for u in b4_unds:
                row: dict[str, Any] = {"date": as_of, "underlying": u, "z_composite": np.nan, "h_applied": h_prev[u]}
                if use_regime:
                    row.update(
                        {
                            "regime_mult": np.nan,
                            "rally_cut": 0,
                            "vix_shock": 0,
                            "dd_cut": 0,
                            "spy_dd_stress": 0,
                            "hyg_stress": 0,
                            "spy_vol_shock": 0,
                        }
                    )
                rows.append(row)
            continue
        z10 = robust_z_cross_sectional(pd.Series(v10))
        zrx = robust_z_cross_sectional(pd.Series(vrx))
        r10_thr = (
            float(np.quantile(np.array(list(v10.values()), dtype=float), float(p["rally_r10d_quantile"])))
            if use_regime and p["rally_cut_enable"] and len(v10) >= min_xsec
            else np.nan
        )
        vix_shock_now = False
        if use_regime and p["vix_shock_enable"] and vix_ser is not None and as_of in vix_ser.index:
            vr = float(vix_ser.loc[as_of])
            thr = _vix_thr_by_feature.get(vix_feature, float(p["vix_r5d_logret_min"]))
            if np.isfinite(vr):
                vix_shock_now = (vr >= thr) if vix_dir == "ge" else (vr <= thr)
        spy_dd_now = False
        if spy_dd_ser is not None and as_of in spy_dd_ser.index:
            ddv = float(spy_dd_ser.loc[as_of])
            if np.isfinite(ddv) and ddv <= float(p.get("spy_dd_max", -0.05)):
                spy_dd_now = True
        hyg_stress_now = False
        if hyg_r21_ser is not None and as_of in hyg_r21_ser.index:
            hv = float(hyg_r21_ser.loc[as_of])
            if np.isfinite(hv) and hv <= float(p.get("hyg_logret21_max", -0.03)):
                hyg_stress_now = True
        spy_vol_shock_now = False
        if spy_vol_ser is not None and as_of in spy_vol_ser.index:
            sv = float(spy_vol_ser.loc[as_of])
            if np.isfinite(sv) and sv >= float(p.get("spy_vol_ratio_min", 1.35)):
                spy_vol_shock_now = True

        for u in b4_unds:
            a_raw = -1.0 * z10.get(u, np.nan)
            b_raw = +1.0 * zrx.get(u, np.nan)
            zc = np.nan
            regime_mult = np.nan
            rally_cut = 0
            vix_u = 1 if (use_regime and vix_shock_now) else 0
            spy_dd_u = 1 if (use_regime and spy_dd_now) else 0
            hyg_u = 1 if (use_regime and hyg_stress_now) else 0
            spy_vol_u = 1 if (use_regime and spy_vol_shock_now) else 0
            dd_cut = 0
            if pd.isna(a_raw) and pd.isna(b_raw):
                h_new = h_prev[u]
            else:
                a = 0.0 if pd.isna(a_raw) else float(a_raw)
                b = 0.0 if pd.isna(b_raw) else float(b_raw)
                zc = 0.5 * a + 0.5 * b
                h_star = float(np.clip(hedge_base - opt2_k * zc, h_min, h_max))
                if use_regime:
                    regime_mult = 1.0
                    if p["rally_cut_enable"] and np.isfinite(r10_thr):
                        ru = v10.get(u, np.nan)
                        if np.isfinite(ru) and ru >= r10_thr:
                            regime_mult *= float(p["rally_h_mult"])
                            rally_cut = 1
                    if p["vix_shock_enable"] and vix_shock_now:
                        vix_mult_universal = float(p["vix_h_mult"])
                        beta_w = float(p.get("vix_delta_weight", 0.0))
                        beta_u = 1.0
                        if vix_beta_panel is not None and u in vix_beta_panel.columns and as_of in vix_beta_panel.index:
                            b_val = vix_beta_panel.at[as_of, u]
                            if pd.notna(b_val):
                                beta_u = float(np.clip(b_val, 0.0, 1.0))
                        eff_mult = 1.0 + (vix_mult_universal - 1.0) * ((1.0 - beta_w) + beta_w * beta_u)
                        regime_mult *= eff_mult
                    if p["dd_cut_enable"] and dd_panel is not None and u in dd_panel.columns and as_of in dd_panel.index:
                        du = float(dd_panel.at[as_of, u])
                        if np.isfinite(du) and du <= float(p["dd_threshold"]):
                            regime_mult *= float(p["dd_h_mult"])
                            dd_cut = 1
                    if bool(p.get("spy_dd_stress_enable", False)) and spy_dd_now:
                        regime_mult *= float(p.get("spy_dd_h_mult", 1.08))
                    if bool(p.get("hyg_21d_stress_enable", False)) and hyg_stress_now:
                        regime_mult *= float(p.get("hyg_stress_h_mult", 1.06))
                    if bool(p.get("spy_index_vol_shock_enable", False)) and spy_vol_shock_now:
                        regime_mult *= float(p.get("spy_vol_shock_h_mult", 1.08))
                    h_star = float(np.clip(h_star * regime_mult, h_min, h_max))
                h_new = (1.0 - opt2_alpha) * h_prev[u] + opt2_alpha * h_star
                h_prev[u] = h_new
            row = {"date": as_of, "underlying": u, "z_composite": zc, "h_applied": h_new}
            if use_regime:
                row.update(
                    {
                        "regime_mult": regime_mult,
                        "rally_cut": rally_cut,
                        "vix_shock": vix_u,
                        "dd_cut": dd_cut,
                        "spy_dd_stress": spy_dd_u,
                        "hyg_stress": hyg_u,
                        "spy_vol_shock": spy_vol_u,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows), rebal, b4_unds


DEFAULT_OVERLAY_NOTEBOOK: dict[str, Any] = dict(
    regime_overlay_enable=True,
    rally_cut_enable=True,
    rally_r10d_quantile=0.90,
    rally_h_mult=0.72,
    vix_shock_enable=True,
    vix_symbol="^VIX",
    vix_feature="dlog5",
    vix_r5d_logret_min=0.12,
    vix_high_prox_max=-0.05,
    vix_term_ratio_max=1.00,
    vix_vvix_dlog5_min=0.15,
    vix_h_mult=1.12,
    vix_delta_weight=0.75,
    vix_beta_window=252,
    dd_cut_enable=True,
    dd_lookback=252,
    dd_threshold=-0.30,
    dd_h_mult=0.72,
    spy_dd_stress_enable=False,
    spy_dd_symbol="SPY",
    spy_dd_lookback=63,
    spy_dd_max=-0.05,
    spy_dd_h_mult=1.08,
    hyg_21d_stress_enable=False,
    hyg_symbol="HYG",
    hyg_logret_window=21,
    hyg_logret21_max=-0.03,
    hyg_stress_h_mult=1.06,
    spy_index_vol_shock_enable=False,
    spy_vol_symbol="SPY",
    spy_vol_ratio_min=1.35,
    spy_vol_shock_h_mult=1.08,
)


def load_bucket4_pairs_from_screened(
    screened_csv: str,
    *,
    min_underlying_vol: float = 0.60,
    min_net_decay: float = 0.20,
    require_bucket_tag: bool = True,
    excluded_inverse_etfs: frozenset[str] | None = None,
    norm_sym: Callable[[str], str] = norm_sym_nb,
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """
    Inverse ETF pairs from screener (no manual UVIX insertion; ``SCO`` excluded by default).
    """
    excl = excluded_inverse_etfs or frozenset({"SCO"})
    df = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in df.columns}
    etf_col = cols.get("etf")
    und_col = cols.get("underlying")
    delta_col = cols.get("delta") or cols.get("beta")
    bucket_col = cols.get("bucket")
    inverse_shortable_col = cols.get("inverse_shortable")
    if etf_col is None or und_col is None or delta_col is None:
        raise ValueError("screened_csv must include ETF/Underlying/Beta columns")
    vol_candidates = [
        "vol_underlying_annual",
        "underlying_vol_annual",
        "underlying_vol",
        "underlying_volatility_annual",
        "underlying_realized_vol_annual",
    ]
    decay_candidates = ["net_edge_p50_annual", "net_edge_annual", "net_decay_annual", "bucket4_net_edge_annual"]
    vol_col = next((cols[c] for c in vol_candidates if c in cols), None)
    decay_col = next((cols[c] for c in decay_candidates if c in cols), None)
    if vol_col is None:
        raise ValueError(f"Could not find an underlying vol column. Tried: {vol_candidates}")
    if decay_col is None:
        raise ValueError(f"Could not find a net decay column. Tried: {decay_candidates}")
    use_cols = [etf_col, und_col, delta_col, vol_col, decay_col] + ([bucket_col] if bucket_col else [])
    if inverse_shortable_col:
        use_cols.append(inverse_shortable_col)
    tmp = df[use_cols].copy()
    tmp[etf_col] = tmp[etf_col].astype(str).map(norm_sym)
    tmp[und_col] = tmp[und_col].astype(str).map(norm_sym)
    tmp[delta_col] = pd.to_numeric(tmp[delta_col], errors="coerce")
    tmp[vol_col] = pd.to_numeric(tmp[vol_col], errors="coerce")
    tmp[decay_col] = pd.to_numeric(tmp[decay_col], errors="coerce")
    mask = tmp[delta_col].notna() & (tmp[delta_col] < 0) & tmp[etf_col].ne("") & tmp[und_col].ne("")
    if require_bucket_tag and bucket_col:
        b = tmp[bucket_col].astype(str).str.lower()
        mask = mask & b.isin(["bucket_4", "bucket_3_inverse", "bucket_3"])
    if inverse_shortable_col:
        mask = mask & tmp[inverse_shortable_col].fillna(False).astype(bool)
    mask = mask & (tmp[vol_col] > min_underlying_vol) & (tmp[decay_col] > min_net_decay)
    mask = mask & ~tmp[etf_col].isin({norm_sym(x) for x in excl})
    out = tmp.loc[mask, [etf_col, und_col]].drop_duplicates()
    pairs = [(str(r[etf_col]), str(r[und_col])) for _, r in out.iterrows()]
    return pairs, out


def run_bucket4_pair_backtest_threshold(
    prices: pd.DataFrame,
    h_daily: pd.Series,
    scheduled_rebal: pd.DatetimeIndex,
    *,
    drift_threshold_long: float = math.inf,
    drift_threshold_short: float = math.inf,
    threshold_check_frequency: str = "B",
    small_epsilon: float = 100.0,
    opt2_h_base: float | None = None,
    **engine_kw: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same economics as ``run_bucket4_backtest_dynamic_h`` plus optional drift triggers.

    Drift definition (documented / diagnostics):
      - ``long`` leg = underlying short notional (``b`` leg, GTP ``long_usd`` side for B4).
      - ``short`` leg = inverse ETF short notional (``a`` leg).

    Trigger when ``abs(current-target)/max(abs(target), small_epsilon)`` exceeds the
    corresponding threshold. ``threshold_check_frequency`` ``\"B\"`` = every row in ``prices``.
    """
    hb = float(opt2_h_base if opt2_h_base is not None else V6_OPT2_H_BASE)
    bt = prices.copy()
    h_aligned = h_daily.reindex(bt.index).ffill().fillna(hb)
    sched_set = set(pd.DatetimeIndex(scheduled_rebal))
    fee_rate = float(engine_kw.get("fee_bps", 0.0)) / 10_000.0
    slip_rate = float(engine_kw.get("slippage_bps", 0.0)) / 10_000.0
    gross_multiplier = float(engine_kw.get("gross_multiplier", 1.0))
    beta_a = float(engine_kw.get("beta_a", -2.0))
    beta_b = float(engine_kw.get("beta_b", 1.0))
    borrow_a_annual = float(engine_kw.get("borrow_a_annual", 0.0))
    borrow_b_annual = float(engine_kw.get("borrow_b_annual", 0.0))
    short_proceeds_annual = float(engine_kw.get("short_proceeds_annual", 0.0))
    initial_capital = float(engine_kw.get("initial_capital", 100_000.0))
    borrow_a_daily = borrow_a_annual / 252.0
    borrow_b_daily = borrow_b_annual / 252.0
    short_proceeds_daily = short_proceeds_annual / 252.0
    beta_inv_abs = abs(beta_a)

    a_sh, b_sh = 0.0, 0.0
    cash = initial_capital
    rows: list[dict[str, Any]] = []
    rlog: list[dict[str, Any]] = []
    first = True
    prev_idx: pd.Timestamp | None = None

    for dt, row in bt.iterrows():
        ap = float(row["a_px"])
        bp = float(row["b_px"])
        h = float(h_aligned.loc[dt])
        a_pos_notional = a_sh * ap
        b_pos_notional = b_sh * bp
        borrow_cost = 0.0
        short_proceeds_credit = 0.0
        rebalance_fee = 0.0
        slippage_cost = 0.0
        rebalance_commission = 0.0
        if a_pos_notional < 0:
            borrow_cost += abs(a_pos_notional) * borrow_a_daily
            short_proceeds_credit += abs(a_pos_notional) * short_proceeds_daily
        if b_pos_notional < 0:
            borrow_cost += abs(b_pos_notional) * borrow_b_daily
            short_proceeds_credit += abs(b_pos_notional) * short_proceeds_daily
        financing_pnl = short_proceeds_credit - borrow_cost
        cash += financing_pnl
        equity = cash + a_pos_notional + b_pos_notional

        do_sched = dt in sched_set
        check_thr = threshold_check_frequency.upper() == "B"

        target_gross = max(0.0, gross_multiplier * equity)
        denom = 1.0 + h * beta_inv_abs
        n_a = target_gross / denom if denom > 1e-12 else 0.5 * target_gross
        n_b = max(0.0, target_gross - n_a)
        target_a_pos, target_b_pos = -n_a, -n_b

        long_d = abs(b_pos_notional - target_b_pos) / max(abs(target_b_pos), small_epsilon)
        short_d = abs(a_pos_notional - target_a_pos) / max(abs(target_a_pos), small_epsilon)
        trig_long = math.isfinite(drift_threshold_long) and long_d > drift_threshold_long
        trig_short = math.isfinite(drift_threshold_short) and short_d > drift_threshold_short

        do_thr = (not first) and bool(check_thr) and (trig_long or trig_short) and (not do_sched)
        do_rebal = bool(first or do_sched or do_thr)

        reason = ""
        if do_rebal:
            if first:
                reason = "initial"
            elif do_sched:
                reason = "scheduled"
            elif trig_long and trig_short:
                reason = "both_drift"
            elif trig_long:
                reason = "long_drift"
            else:
                reason = "short_drift"

        rebalance_flag = False
        if do_rebal:
            rebalance_flag = True
            a_prev, b_prev = float(a_pos_notional), float(b_pos_notional)
            delta_a, delta_b = target_a_pos - a_pos_notional, target_b_pos - b_pos_notional
            traded = abs(delta_a) + abs(delta_b)
            fee = traded * fee_rate
            slip = traded * slip_rate
            rebalance_commission = float(fee)
            rebalance_fee = float(fee + slip)
            slippage_cost = float(slip)
            cash -= delta_a + delta_b + fee + slip
            a_sh = target_a_pos / ap if ap > 0 else 0.0
            b_sh = target_b_pos / bp if bp > 0 else 0.0
            a_pos_notional, b_pos_notional = a_sh * ap, b_sh * bp
            equity = cash + a_pos_notional + b_pos_notional
            rlog.append(
                {
                    "date": dt,
                    "reason": reason,
                    "a_pos_before": a_prev,
                    "b_pos_before": b_prev,
                    "target_a": float(target_a_pos),
                    "target_b": float(target_b_pos),
                    "drift_long_pct": float(long_d),
                    "drift_short_pct": float(short_d),
                    "traded_notional": float(traded),
                    "is_scheduled": bool(do_sched),
                    "is_threshold": bool(do_thr),
                }
            )

        beta_notional = (-1.0) * beta_a * abs(a_pos_notional) + (-1.0) * beta_b * abs(b_pos_notional)
        rows.append(
            {
                "date": dt,
                "a_px": ap,
                "b_px": bp,
                "cash": cash,
                "a_shares": a_sh,
                "b_shares": b_sh,
                "equity": equity,
                "h_used": h,
                "rebalance": rebalance_flag,
                "rebalance_reason": reason if rebalance_flag else "",
                "beta_notional": beta_notional,
                "borrow_cost": borrow_cost,
                "short_proceeds_credit": short_proceeds_credit,
                "financing_pnl": financing_pnl,
                "rebalance_fee": rebalance_fee,
                "rebalance_commission": rebalance_commission,
                "slippage_cost": slippage_cost,
            }
        )
        first = False

    out = pd.DataFrame(rows).set_index("date")
    out["ret"] = out["equity"].pct_change().fillna(0.0)
    out["drawdown"] = out["equity"].div(out["equity"].cummax()).sub(1.0)
    out["beta_exposure_frac"] = np.where(out["equity"].abs() > 1e-9, out["beta_notional"] / out["equity"], np.nan)
    return out, pd.DataFrame(rlog)


@dataclass
class Bucket4WeeklyConfig:
    screened_csv: str
    start: str
    end: str | None = None
    weekly_rebalance_freq: str = "W-FRI"
    warmup_bdays: int = 65
    hedge_base: float = V6_OPT2_H_BASE
    borrow_globs: list[str] | str = field(
        default_factory=lambda: [
            "data/etf_screened_today.csv",
            "data/runs/*/etf_screened_today.csv",
        ]
    )
    use_borrow_from_screened: bool = True
    borrow_fallback_annual: float = 0.1
    use_delta_from_screened: bool = True
    leg_a_delta_fallback: float = -2.0
    leg_b_delta_fallback: float = 1.0
    fee_bps: float = 1.0
    slippage_bps: float = 20.0
    gross_multiplier: float = 1.0
    initial_capital_lab: float = 100_000.0
    underlying_internalized_borrow_annual: float = 0.0
    short_proceeds_annual: float = 0.0
    manual_borrow_override: dict[str, float] | None = None
    excluded_inverse_etfs: frozenset[str] = field(default_factory=lambda: frozenset({"SCO"}))
    min_underlying_vol: float = 0.60
    min_net_decay: float = 0.20
    drift_threshold_long: float = math.inf
    drift_threshold_short: float = math.inf
    threshold_check_frequency: str = "B"
    borrow_multiplier: float = 1.0
    overlay: dict[str, Any] | None = None
    pf_params: V6PfParams | None = None
    use_ibkr_uvix_borrow: bool = False


@dataclass
class Bucket4State:
    pair_cache: dict[tuple[str, str], dict[str, Any]]
    hedge_by_underlying: dict[str, pd.Series]
    hedge_panel: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex
    hedge_base: float
    screened_subset: pd.DataFrame
    pair_metadata: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    config: Bucket4WeeklyConfig
    closes_broad: pd.DataFrame
    bucket4_pairs: list[tuple[str, str]]


def build_closes_broad(
    bucket4_pairs: Sequence[tuple[str, str]],
    inverse_etf_universe: Sequence[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Daily closes for all bucket-4 underlyings plus inverse ETF tickers (for broad matrix)."""
    syms = sorted({u for _, u in bucket4_pairs} | {e for e, _ in bucket4_pairs} | set(inverse_etf_universe))
    cols: dict[str, pd.Series] = {}
    for sym in syms:
        try:
            cols[sym] = load_single_close(sym, start, end).rename(sym)
        except Exception:
            continue
    if len(cols) < 2:
        raise RuntimeError("closes_broad: insufficient price columns.")
    df = pd.DataFrame(cols).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _inverse_etfs_from_screened(screened_csv: str) -> list[str]:
    df = pd.read_csv(screened_csv)
    cl = {c.lower(): c for c in df.columns}
    ec, bc = cl.get("etf"), cl.get("delta") or cl.get("beta")
    if ec is None or bc is None:
        return []
    beta = pd.to_numeric(df[bc], errors="coerce")
    etfs = df.loc[beta < 0, ec].dropna().astype(str).str.upper().str.strip()
    return sorted({x for x in etfs if x and x.upper() != "NAN"})


def build_pair_cache(
    pairs: Sequence[tuple[str, str]],
    cfg: Bucket4WeeklyConfig,
    *,
    underlying_ibkr_map: dict[str, float] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for etf_sym, und_sym in pairs:
        try:
            prices_i = load_prices(etf_sym, und_sym, cfg.start, cfg.end)
        except Exception as e:
            cache[(etf_sym, und_sym)] = {"skip_reason": str(e)}
            continue
        if prices_i is None or prices_i.empty:
            cache[(etf_sym, und_sym)] = {"skip_reason": "empty prices"}
            continue
        beta_a_i, beta_b_i = load_beta_values(
            etf_sym,
            und_sym,
            cfg.screened_csv,
            cfg.use_delta_from_screened,
            cfg.leg_a_delta_fallback,
            cfg.leg_b_delta_fallback,
        )
        if beta_a_i >= 0:
            cache[(etf_sym, und_sym)] = {"skip_reason": "non-inverse beta_a"}
            continue
        borrow_a_i, _ = load_pair_borrow_rates(
            etf_sym,
            und_sym,
            cfg.borrow_globs,
            cfg.use_borrow_from_screened,
            cfg.borrow_fallback_annual,
            underlying_ibkr_map=underlying_ibkr_map,
            manual_override=cfg.manual_borrow_override,
        )
        borrow_b_i = float(cfg.underlying_internalized_borrow_annual)
        cache[(etf_sym, und_sym)] = {
            "prices": prices_i,
            "kw": dict(
                initial_capital=cfg.initial_capital_lab,
                gross_multiplier=cfg.gross_multiplier,
                beta_a=beta_a_i,
                beta_b=beta_b_i,
                borrow_a_annual=borrow_a_i,
                borrow_b_annual=borrow_b_i,
                short_proceeds_annual=cfg.short_proceeds_annual,
                fee_bps=cfg.fee_bps,
                slippage_bps=cfg.slippage_bps,
                opt2_h_base=cfg.hedge_base,
            ),
        }
    return cache


def build_bucket4_state(
    cfg: Bucket4WeeklyConfig,
    *,
    get_ibkr_borrow_map: Callable[[list[str]], dict[str, float]] | None = None,
    pair_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    closes_broad: pd.DataFrame | None = None,
    bucket4_pairs: Sequence[tuple[str, str]] | None = None,
    macro: dict[str, pd.Series] | None = None,
) -> Bucket4State:
    pairs, screened_sub = load_bucket4_pairs_from_screened(
        cfg.screened_csv,
        min_underlying_vol=cfg.min_underlying_vol,
        min_net_decay=cfg.min_net_decay,
        excluded_inverse_etfs=cfg.excluded_inverse_etfs,
    )
    if bucket4_pairs is not None:
        pairs = list(bucket4_pairs)
    if pair_cache is None:
        umap: dict[str, float] | None = None
        if get_ibkr_borrow_map is not None:
            try:
                syms = sorted({u for _, u in pairs})
                rawm = get_ibkr_borrow_map(syms)
                umap = {str(k).upper(): float(v) for k, v in rawm.items() if np.isfinite(float(v))}
            except Exception:
                umap = None
        pair_cache = build_pair_cache(pairs, cfg, underlying_ibkr_map=umap)
    if closes_broad is None:
        inv = _inverse_etfs_from_screened(cfg.screened_csv)
        closes_broad = build_closes_broad(pairs, inv, cfg.start, cfg.end)
    panel, rebal, _ = build_hedge_panel_opt2(
        closes_broad,
        pairs,
        weekly_rebalance_freq=cfg.weekly_rebalance_freq,
        warmup_bdays=cfg.warmup_bdays,
        hedge_base=cfg.hedge_base,
        overlay=cfg.overlay,
        macro=macro,
    )
    master = closes_broad.index.sort_values()
    b4_unds = sorted({u for _, u in pairs} & set(closes_broad.columns))
    hedge_map = panel_to_hedge_by_underlying(panel, master, b4_unds, cfg.hedge_base)
    meta = [{"etf": e, "underlying": u, "in_cache": (e, u) in pair_cache and "skip_reason" not in pair_cache[(e, u)]} for e, u in pairs]
    diag = {
        "n_pairs_screened": len(pairs),
        "n_pairs_cached": sum(1 for k, v in pair_cache.items() if "skip_reason" not in v),
        "weekly_rebalance_freq": cfg.weekly_rebalance_freq,
        "n_scheduled_rebalances": len(rebal),
    }
    return Bucket4State(
        pair_cache=pair_cache,
        hedge_by_underlying=hedge_map,
        hedge_panel=panel,
        rebalance_dates=rebal,
        hedge_base=cfg.hedge_base,
        screened_subset=screened_sub,
        pair_metadata=meta,
        diagnostics=diag,
        config=cfg,
        closes_broad=closes_broad,
        bucket4_pairs=pairs,
    )


def compute_bucket4_weights(
    state: Bucket4State,
    *,
    get_ibkr_borrow_map: Callable[[list[str]], dict[str, float]] | None = None,
    norm_sym: Callable[[str], str] = norm_sym_nb,
) -> tuple[dict[tuple[str, str], float], pd.DataFrame, dict[str, Any]]:
    """Tail-risk + covariance weights (``pair_weights`` sum to 1 over live pairs)."""
    p = state.config.pf_params or V6PfParams()
    _ibkr = get_ibkr_borrow_map or (lambda _symbols: {})
    w, wdf, meta = compute_v6_b4_pf_weight_dict(
        pair_cache=state.pair_cache,
        v6_opt2_h_daily_map=state.hedge_by_underlying,
        screened_csv=state.config.screened_csv,
        closes_broad=state.closes_broad,
        norm_sym=norm_sym,
        get_ibkr_borrow_map=_ibkr,
        opt2_h_base=state.hedge_base,
        params=p,
        use_ibkr_uvix_borrow=state.config.use_ibkr_uvix_borrow,
    )
    meta = dict(meta)
    meta["min_live_pairs"] = int(p.min_pairs)
    meta["weekly_rebalance_freq"] = state.config.weekly_rebalance_freq
    meta["use_ibkr_uvix_borrow"] = state.config.use_ibkr_uvix_borrow
    return w, wdf, meta


def compute_bucket4_targets(
    state: Bucket4State,
    pair_weights: Mapping[tuple[str, str], float],
    as_of: str | pd.Timestamp,
    sleeve_budget_usd: float,
    *,
    fee_bps: float | None = None,
    slippage_bps: float | None = None,
    partial_hedge_ratio: float = 1.0,
    delta_floor: float = 0.1,
    current_leg_notional_by_pair: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
    small_epsilon: float = 100.0,
    ratchet_enabled: bool = False,
    ratchet_floor_by_pair: Mapping[tuple[str, str], float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Production-style target notionals for one run date (weights × sleeve budget; dynamic hedge).

    ``current_leg_notional_by_pair`` is optional because trade-plan generation may not always
    have current positions. When supplied, values are expected as positive short notionals using
    keys ``inverse_etf_short_usd`` and ``underlying_short_usd`` (signed aliases
    ``etf_target_usd`` / ``underlying_target_usd`` are also accepted and converted with ``abs``).
    Threshold diagnostics then mirror the backtest rule:
    ``abs(current - target) / max(abs(target), small_epsilon)``.

    RATCHET (``ratchet_enabled``): the inverse-ETF short leg is *grow-only*. The solved
    delta-neutral inverse target is floored at ``max(solved, current_held, persisted_floor)``
    so we never propose buying back (covering) inverse-ETF inventory that is hard to relocate.
    The underlying short leg is then re-solved against the floored inverse leg
    (``und = h * beta_used * inv * partial_hedge_ratio``) so the hedge stays consistent — all
    delta reduction is expressed through the (bidirectional) underlying leg, never the inverse leg.
    Provenance columns (``inverse_short_solved_usd``, ``ratchet_floor_usd``, ``ratchet_binding``,
    ``ratchet_source``, ``ratchet_explain``) make every floored value reverse-engineerable.
    """
    as_of_ts = pd.Timestamp(as_of)
    rows: list[dict[str, Any]] = []
    sched_set = set(state.rebalance_dates)
    next_sched = None
    future = [d for d in state.rebalance_dates if pd.Timestamp(d) > as_of_ts]
    if future:
        next_sched = pd.Timestamp(future[0])

    is_sched = as_of_ts in sched_set
    f_bps = float(state.config.fee_bps if fee_bps is None else fee_bps)
    s_bps = float(state.config.slippage_bps if slippage_bps is None else slippage_bps)

    active_w = {k: float(v) for k, v in pair_weights.items() if float(v) > 0.0}
    wsum = sum(active_w.values()) or 1.0
    active_w = {k: v / wsum for k, v in active_w.items()}

    for (etf_sym, und_sym), w in active_w.items():
        if (etf_sym, und_sym) not in state.pair_cache or "skip_reason" in state.pair_cache[(etf_sym, und_sym)]:
            continue
        if und_sym not in state.hedge_by_underlying:
            continue
        kw = state.pair_cache[(etf_sym, und_sym)]["kw"].copy()
        kw["fee_bps"] = f_bps
        kw["slippage_bps"] = s_bps
        beta_a = float(kw.get("beta_a", -2.0))
        beta_b = float(kw.get("beta_b", 1.0))
        beta_used = max(delta_floor, abs(beta_a))
        h_ser = state.hedge_by_underlying[und_sym]
        h_hist = h_ser.loc[pd.DatetimeIndex(h_ser.index) <= as_of_ts].dropna()
        h = float(h_hist.iloc[-1]) if len(h_hist) else float(state.hedge_base)
        gross = float(w) * float(sleeve_budget_usd)
        denom = 1.0 + h * beta_used
        n_inv = gross / denom if denom > 1e-12 else 0.5 * gross
        n_und = max(0.0, gross - n_inv)
        inv_short_usd = n_inv
        und_short_usd = n_und * float(partial_hedge_ratio)
        cur = (current_leg_notional_by_pair or {}).get((etf_sym, und_sym), {})
        cur_inv = abs(float(cur.get("inverse_etf_short_usd", cur.get("etf_target_usd", inv_short_usd))))
        cur_und = abs(float(cur.get("underlying_short_usd", cur.get("underlying_target_usd", und_short_usd))))

        # ---- RATCHET: inverse-ETF short leg is grow-only --------------------
        inv_short_solved = inv_short_usd
        ratchet_floor = abs(float((ratchet_floor_by_pair or {}).get((etf_sym, und_sym), 0.0)))
        cur_inv_held = abs(float(cur.get("inverse_etf_short_usd", cur.get("etf_target_usd", 0.0)))) \
            if (etf_sym, und_sym) in (current_leg_notional_by_pair or {}) else 0.0
        ratchet_binding = False
        ratchet_source = "solve"
        if ratchet_enabled:
            floor_val = max(inv_short_solved, cur_inv_held, ratchet_floor)
            if floor_val > inv_short_solved + 1e-6:
                ratchet_binding = True
                ratchet_source = "held_position" if cur_inv_held >= ratchet_floor else "ratchet_state"
            inv_short_usd = floor_val
            # re-solve underlying leg against the floored inverse leg (keeps hedge h consistent)
            und_short_usd = h * beta_used * inv_short_usd * float(partial_hedge_ratio)
        if ratchet_binding:
            ratchet_explain = (
                f"inverse floored to {inv_short_usd:,.0f} (>solved {inv_short_solved:,.0f}) "
                f"from {ratchet_source} [held={cur_inv_held:,.0f}, state={ratchet_floor:,.0f}]; "
                f"underlying re-solved = h({h:.3f})*beta({beta_used:.3f})*inv*phr({partial_hedge_ratio:.2f}) "
                f"= {und_short_usd:,.0f}"
            )
        elif ratchet_enabled:
            ratchet_explain = f"no cover needed; solved inverse {inv_short_solved:,.0f} already >= floor"
        else:
            ratchet_explain = "ratchet disabled"
        drift_short = abs(cur_inv - inv_short_usd) / max(abs(inv_short_usd), float(small_epsilon))
        drift_long = abs(cur_und - und_short_usd) / max(abs(und_short_usd), float(small_epsilon))
        trig_long = (
            current_leg_notional_by_pair is not None
            and math.isfinite(state.config.drift_threshold_long)
            and drift_long > float(state.config.drift_threshold_long)
        )
        trig_short = (
            current_leg_notional_by_pair is not None
            and math.isfinite(state.config.drift_threshold_short)
            and drift_short > float(state.config.drift_threshold_short)
        )
        if is_sched:
            reason = "scheduled"
        elif trig_long and trig_short:
            reason = "both_drift"
        elif trig_long:
            reason = "long_drift"
        elif trig_short:
            reason = "short_drift"
        else:
            reason = "intraday_targets_only"
        rows.append(
            {
                "ETF": norm_sym_gtp(etf_sym),
                "Underlying": norm_sym_gtp(und_sym),
                "inverse_etf_short_usd": inv_short_usd,
                "underlying_short_usd": und_short_usd,
                "gross_target_usd": gross,
                "hedge_ratio": h,
                "inverse_beta": beta_a,
                "underlying_beta": beta_b,
                "pair_weight": w,
                "borrow_etf_annual": float(kw.get("borrow_a_annual", 0.0)),
                "borrow_multiplier": state.config.borrow_multiplier,
                "fee_bps": f_bps,
                "slippage_bps": s_bps,
                "is_scheduled_rebalance": bool(is_sched),
                "is_threshold_rebalance": bool((trig_long or trig_short) and not is_sched),
                "rebalance_reason": reason,
                "current_inverse_etf_short_usd": cur_inv,
                "current_underlying_short_usd": cur_und,
                "drift_short_pct": drift_short,
                "drift_long_pct": drift_long,
                "next_scheduled_rebalance_date": next_sched,
                "inverse_short_solved_usd": inv_short_solved,
                "ratchet_floor_usd": ratchet_floor,
                "ratchet_binding": bool(ratchet_binding),
                "ratchet_source": ratchet_source,
                "ratchet_explain": ratchet_explain,
            }
        )
    meta = {
        "as_of": str(as_of_ts.date()),
        "sleeve_budget_usd": sleeve_budget_usd,
        "fee_bps": f_bps,
        "slippage_bps": s_bps,
        "thresholds_evaluated": current_leg_notional_by_pair is not None,
    }
    return pd.DataFrame(rows), meta


def run_bucket4_backtest(
    state: Bucket4State,
    pair_weights: Mapping[tuple[str, str], float],
    *,
    initial_capital: float,
    get_ibkr_borrow_map: Callable[[list[str]], dict[str, float]] | None = None,
    use_thresholds: bool = True,
) -> dict[str, Any]:
    """
    Aggregate portfolio backtest (per-pair engine, no ``simulate_sized_pairs``).
    """
    cfg = state.config
    umap: dict[str, float] | None = None
    if get_ibkr_borrow_map is not None:
        try:
            syms = sorted({u for _, u in state.bucket4_pairs})
            rawm = get_ibkr_borrow_map(syms)
            umap = {str(k).upper(): float(v) for k, v in rawm.items() if np.isfinite(float(v))}
        except Exception:
            umap = None

    bt_by_pair: dict[tuple[str, str], pd.DataFrame] = {}
    logs: list[pd.DataFrame] = []
    for (etf_sym, und_sym), w in pair_weights.items():
        w_pair = float(w)
        if w_pair <= 0.0:
            continue
        c = state.pair_cache.get((etf_sym, und_sym))
        if c is None or "skip_reason" in c:
            continue
        und_sym = str(und_sym)
        if und_sym not in state.hedge_by_underlying:
            continue
        prices_i = c["prices"]
        kw = dict(c["kw"])
        kw["initial_capital"] = w_pair * float(initial_capital)
        kw["fee_bps"] = cfg.fee_bps
        kw["slippage_bps"] = cfg.slippage_bps
        base_borrow, _ = load_pair_borrow_rates(
            etf_sym,
            und_sym,
            cfg.borrow_globs,
            cfg.use_borrow_from_screened,
            cfg.borrow_fallback_annual,
            underlying_ibkr_map=umap,
            manual_override=cfg.manual_borrow_override,
        )
        if cfg.use_ibkr_uvix_borrow and get_ibkr_borrow_map is not None and norm_sym_nb(etf_sym) == "UVIX":
            try:
                v = get_ibkr_borrow_map(["UVIX"]).get("UVIX")
                if v is not None and np.isfinite(float(v)) and float(v) > 0:
                    base_borrow = float(v)
            except Exception:
                pass
        kw["borrow_a_annual"] = float(base_borrow) * float(cfg.borrow_multiplier)
        h_d = state.hedge_by_underlying[und_sym].reindex(prices_i.index).ffill().fillna(cfg.hedge_base)
        pair_sched = state.rebalance_dates.intersection(prices_i.index)
        if len(pair_sched) == 0:
            pair_sched = pd.DatetimeIndex([prices_i.index[0]])
        if use_thresholds and (
            np.isfinite(cfg.drift_threshold_long) or np.isfinite(cfg.drift_threshold_short)
        ):
            bt, lg = run_bucket4_pair_backtest_threshold(
                prices_i,
                h_d,
                pair_sched,
                drift_threshold_long=cfg.drift_threshold_long,
                drift_threshold_short=cfg.drift_threshold_short,
                threshold_check_frequency=cfg.threshold_check_frequency,
                opt2_h_base=cfg.hedge_base,
                **kw,
            )
            if len(lg):
                lg.insert(0, "pair", f"{etf_sym}/{und_sym}")
                logs.append(lg)
        else:
            bt = run_bucket4_backtest_dynamic_h(prices_i, h_d, pair_sched, **kw)
        bt_by_pair[(etf_sym, und_sym)] = bt

    if not bt_by_pair:
        return {
            "portfolio_curve": pd.DataFrame(),
            "pair_backtests": {},
            "attribution": pd.DataFrame(),
            "diagnostics": {},
            "rebalance_log": pd.DataFrame(),
        }

    all_idx = pd.DatetimeIndex(
        sorted({pd.Timestamp(d) for k in bt_by_pair for d in bt_by_pair[k].index})
    )
    port_equity = pd.Series(0.0, index=all_idx, dtype=float)
    agg_borrow = pd.Series(0.0, index=all_idx, dtype=float)
    agg_comm = pd.Series(0.0, index=all_idx, dtype=float)
    agg_slip = pd.Series(0.0, index=all_idx, dtype=float)
    for k, bt in bt_by_pair.items():
        first_p = bt.index[0]
        eq = bt["equity"].reindex(all_idx).where(all_idx >= first_p, 0.0).ffill().fillna(0.0).astype(float)
        port_equity = port_equity.add(eq, fill_value=0.0)
        br = bt["borrow_cost"].reindex(all_idx).fillna(0.0).where(all_idx >= first_p, 0.0).astype(float)
        agg_borrow = agg_borrow.add(br, fill_value=0.0)
        if "rebalance_commission" in bt.columns:
            c = bt["rebalance_commission"].reindex(all_idx).fillna(0.0).where(all_idx >= first_p, 0.0).astype(float)
            agg_comm = agg_comm.add(c, fill_value=0.0)
        if "slippage_cost" in bt.columns:
            sl = bt["slippage_cost"].reindex(all_idx).fillna(0.0).where(all_idx >= first_p, 0.0).astype(float)
            agg_slip = agg_slip.add(sl, fill_value=0.0)

    bt_pf = pd.DataFrame(
        {
            "equity": port_equity,
            "nav": port_equity,
            "ret": port_equity.pct_change().fillna(0.0),
            "drawdown": port_equity / port_equity.cummax() - 1.0,
            "borrow_cost": agg_borrow,
            "rebalance_commission": agg_comm,
            "slippage_cost": agg_slip,
        }
    )
    rlog = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    diag = {
        "nav_last": float(bt_pf["nav"].iloc[-1]) if len(bt_pf) else float("nan"),
        "sum_pair_last_equity": float(sum(float(bt["equity"].iloc[-1]) for bt in bt_by_pair.values())),
        "scheduled_rebalance_count": int(rlog["is_scheduled"].sum()) if len(rlog) and "is_scheduled" in rlog.columns else 0,
        "threshold_rebalance_count": int(rlog["is_threshold"].sum()) if len(rlog) and "is_threshold" in rlog.columns else 0,
    }
    attr_rows = []
    for k, bt in bt_by_pair.items():
        st = perf_stats(bt)
        attr_rows.append({"pair": f"{k[0]}/{k[1]}", **{str(x): float(st.get(x, np.nan)) for x in st.index}})
    attr_df = pd.DataFrame(attr_rows)
    return {
        "portfolio_curve": bt_pf,
        "pair_backtests": bt_by_pair,
        "attribution": attr_df,
        "diagnostics": diag,
        "rebalance_log": rlog,
    }


def state_config_dict(cfg: Bucket4WeeklyConfig) -> dict[str, Any]:
    return asdict(cfg)


__all__ = [
    "Bucket4State",
    "Bucket4WeeklyConfig",
    "build_bucket4_state",
    "build_hedge_panel_opt2",
    "compute_bucket4_targets",
    "compute_bucket4_weights",
    "configure_price_cache_dirs",
    "load_bucket4_pairs_from_screened",
    "norm_sym_gtp",
    "panel_to_hedge_by_underlying",
    "run_bucket4_backtest",
    "run_bucket4_pair_backtest_threshold",
    "weekly_rebalance_dates",
]
