#!/usr/bin/env python3
"""
generate_trade_plan.py

Production portfolio construction for YAML sleeves.

Implements:
- ``core_leveraged`` sleeve: high-delta bucket-1 positives only (**excludes** ``is_yieldboost``
  rows; those size in ``yieldboost``).
    * Post-B4 gross is split vs ``yieldboost`` using normalized ``target_weight`` on each sleeve.
- ``yieldboost`` sleeve: bucket‑2 / YieldBoost only (`is_yieldboost`), gated by
  ``portfolio.sleeves.yieldboost.rules.min_net_edge_annual`` on ``net_edge_p50_annual``.
  Stock sleeves do **not** apply a blanket ``net_decay_annual >= 0`` rule; core may still use
  ``min_net_decay_annual`` / hysteresis from YAML.

- Bucket 4: ``inverse_decay_bucket4`` unchanged; core inverse rows still require
  ``inverse_shortable``. The optional volatility-ETP bucket-5 sleeve does not (see
  ``_in_b4_volatility_etp_sleeve_mask``).

- Purgatory handling (from screened CSV; see daily_screener ``recompute_purgatory_by_bucket``):
    * OUTPUT purgatory rows with 0 targets so execution won’t auto-close.
    * Does NOT allocate new exposure to purgatory (borrow soft band OR net_edge_p50 in 0–5%).
    * Sizing set also requires ``net_edge_p50_annual >= 0`` when that column exists.

- inverse_decay_bucket4: optional ``enabled: false`` on the sleeve — disables all B4 targets; stock
  sleeve absorbs the full gross budget.

- ``core_leveraged`` bucket-1 path: optional ``min_net_decay_annual`` and ``net_decay_hysteresis`` in
  YAML — tighter net-decay selectivity for **high-delta core rows only**. If ``min_net_decay_annual``
  > 0 it is a **hard floor** (always enforced, including over hysteresis and missing-data paths).
  Hysteresis uses ``paths.core_leveraged_decay_state_json`` to reduce pairs bouncing in/out when decay oscillates.

Outputs:
- proposed_trades.csv (stock sleeves sized + purgatory keep-open rows)

Notes:
- This script assumes screened_csv includes: ETF, Underlying, purgatory, Beta
- Borrow caps require screened_csv to include a borrow column OR you pass a borrow_map externally.
  If you don’t have borrow in screened_csv, set caps high or skip borrow filtering here.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import argparse
import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Set, Tuple

import numpy as np
import pandas as pd

from strategy_config import load_config

TRADING_DAYS = 252
VOL_ETP_BUCKET5_SLEEVE = "volatility_etp_bucket5"


# -----------------------------
# Basic helpers
# -----------------------------
def today_str() -> str:
    return date.today().isoformat()


def run_dir(run_date: str) -> Path:
    return Path("data") / "runs" / run_date


def _b4_ratchet_state_path(b4_rules: dict) -> Path:
    rcfg = (b4_rules.get("ratchet") or {})
    return Path(str(rcfg.get("state_json") or "data/b4_inverse_ratchet_state.json"))


def _b4_load_ratchet_state(path: Path) -> dict:
    """Load persisted per-pair inverse short floors ({'ETF|UND': usd})."""
    try:
        if path.is_file():
            raw = json.loads(path.read_text())
            d = raw.get("inverse_short_usd_by_pair", raw) if isinstance(raw, dict) else {}
            return {str(k): float(v) for k, v in (d or {}).items()
                    if v is not None and np.isfinite(float(v))}
    except Exception:
        pass
    return {}


def _b4_write_ratchet_state(path: Path, state: dict, run_date: str) -> None:
    """Atomically persist the grow-only inverse short floors."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_date": str(run_date),
            "inverse_short_usd_by_pair": {k: round(float(v), 2) for k, v in state.items()},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
    except Exception as e:
        print(f"[WARN] could not persist b4 ratchet state ({e})")


def _plot_b4_cadence(cad_df: "pd.DataFrame", state, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    d = cad_df.dropna(subset=["interval_days"]).sort_values("interval_days")
    if not d.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
        ax.bar(d["Underlying"].astype(str), d["interval_days"].astype(float), color="#1f6f54")
        for i, (_, r) in enumerate(d.iterrows()):
            ax.text(i, float(r["interval_days"]) + 0.1, f"{int(r['interval_days'])}d", ha="center", fontsize=8)
        ax.set_ylabel("days to rebalance")
        ax.set_title("Bucket 4 - days to rebalance (per underlying)")
        ax.tick_params(axis="x", rotation=45)
        fig.savefig(out_dir / "b4_days_to_rebalance.png", dpi=130)
        plt.close(fig)
    hbu = getattr(state, "hedge_by_underlying", None) or {}
    if hbu:
        fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
        for u, ser in hbu.items():
            try:
                s = ser.dropna().tail(252)
            except Exception:
                continue
            if len(s):
                ax.plot(s.index, s.values, label=str(u), lw=1.2)
        ax.set_ylabel("hedge ratio h")
        ax.set_title("Bucket 4 - hedge ratio over time")
        ax.legend(fontsize=7, ncol=4)
        fig.savefig(out_dir / "b4_hedge_ratio_over_time.png", dpi=130)
        plt.close(fig)


def _emit_b4_cadence_outputs(state, tgt_df: "pd.DataFrame", run_date: str) -> None:
    """Write human-readable cadence/hedge explainability + plots + ratchet provenance.

    Answers, for each underlying: how many days until the next rebalance and why
    (inputs TR/VCR -> output N days), plus the hedge ratio now/over-time. Lets a
    human (or an AI) reverse-engineer every value from b4_cadence_explain.csv/.txt.
    """
    try:
        out_dir = run_dir(run_date) / "b4_hedge_cadence"
        out_dir.mkdir(parents=True, exist_ok=True)
        cad = (getattr(state, "diagnostics", {}) or {}).get("cadence_by_underlying", {}) or {}
        keys = ("interval_days", "hedge_ratio", "tr", "vcr", "vcr_med",
                "interval_explain", "h_explain", "reason")
        rows = [{"Underlying": u, **{k: c.get(k) for k in keys}} for u, c in cad.items()]
        cad_df = pd.DataFrame(rows)
        if not cad_df.empty:
            cad_df.to_csv(out_dir / "b4_cadence_explain.csv", index=False)
            lines = [f"Bucket 4 rebalance cadence + hedge ratio  (run {run_date})", "=" * 64]
            _srt = cad_df.copy()
            _srt["_ord"] = pd.to_numeric(_srt["interval_days"], errors="coerce")
            for _, r in _srt.sort_values("_ord", na_position="last").iterrows():
                lines.append(
                    f"{r['Underlying']}: rebalance every ~{r['interval_days']} trading day(s); "
                    f"h={r['hedge_ratio']}"
                )
                if isinstance(r.get("interval_explain"), str):
                    lines.append(f"   {r['interval_explain']}")
                if isinstance(r.get("h_explain"), str):
                    lines.append(f"   {r['h_explain']}")
            (out_dir / "b4_cadence_explain.txt").write_text("\n".join(lines))
            print(f"[INFO] bucket4 cadence explain -> {out_dir / 'b4_cadence_explain.csv'}")
            _plot_b4_cadence(cad_df, state, out_dir)
        rcols = [c for c in (
            "ETF", "Underlying", "inverse_etf_short_usd", "inverse_short_solved_usd",
            "underlying_short_usd", "hedge_ratio", "ratchet_binding", "ratchet_source",
            "ratchet_explain") if c in tgt_df.columns]
        if rcols:
            tgt_df[rcols].to_csv(out_dir / "b4_ratchet_targets.csv", index=False)
    except Exception as e:
        print(f"[WARN] b4 cadence outputs skipped ({e})")


def _norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _volatility_etp_rows_mask(df: pd.DataFrame) -> pd.Series:
    """True for screener rows classified as volatility ETPs (VIX complex, etc.)."""
    out = pd.Series(False, index=df.index)
    for col in ("Delta_product_class", "product_class"):
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()
            out |= s.eq("volatility_etp")
    try:
        from daily_screener import VOLATILITY_ETP_SYMBOLS as _VOL_SYMS
    except ImportError:
        _VOL_SYMS = frozenset()
    if _VOL_SYMS and "ETF" in df.columns:
        out |= df["ETF"].astype(str).map(_norm_sym).isin(_VOL_SYMS)
    if _VOL_SYMS and "Underlying" in df.columns:
        und = df["Underlying"].map(
            lambda x: _norm_sym(x) if pd.notna(x) and str(x).strip() else ""
        )
        out |= und.isin(_VOL_SYMS)
    return out


def _in_b4_volatility_etp_sleeve_mask(
    eligible: pd.DataFrame,
    *,
    b4_borrow_ok: pd.Series,
    b4_edge_ok: pd.Series,
    b4_vol_ok: pd.Series,
    b4_not_excluded: pd.Series,
    in_flow_program: pd.Series,
    in_b4_core: pd.Series,
) -> pd.Series:
    """Volatility-ETP bucket-5 slice (separate from core inverse ``in_b4``).

    Core ``in_b4`` still requires ``inverse_shortable``; VIX-complex ETPs are often flagged
    false on screened rows despite being the intended inverse-vol sleeve, so this path omits it.
    """
    is_volatility_etp = _volatility_etp_rows_mask(eligible)
    return (
        is_volatility_etp
        & b4_borrow_ok
        & b4_edge_ok
        & b4_vol_ok
        & b4_not_excluded
        & ~in_flow_program
        & ~in_b4_core
    )


def _normalize_two_nonnegative_weights(a: float, b: float) -> tuple[float, float]:
    """Return (frac_a, frac_b) that sum to 1; if both zero, (1, 0)."""
    aa = max(0.0, float(a))
    bb = max(0.0, float(b))
    s = aa + bb
    if s <= 1e-18:
        return 1.0, 0.0
    return aa / s, bb / s


def _b2_b4_universe_masks(
    eligible: pd.DataFrame,
    *,
    flow_program_etfs: set[str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the B2/B4 universe gating masks shared by ``main()`` and the sizing mirror.

    Returns
    -------
    is_yieldboost : pd.Series[bool]
        Per-row screener flag (``False`` when the column is absent).
    in_b2_universe : pd.Series[bool]
        ``is_yieldboost`` only (bucket‑2 stock candidates for ``generate_trade_plan``).
    in_flow_program : pd.Series[bool]
        Rows whose ``ETF`` ticker lives in ``flow_program.universe.shorts`` —
        excluded from both B2 and B4 because the weekly flow sleeve sizes them.
    """
    if "is_yieldboost" in eligible.columns:
        is_yieldboost = eligible["is_yieldboost"].fillna(False).astype(bool)
    else:
        is_yieldboost = pd.Series(False, index=eligible.index)
    in_b2_universe = is_yieldboost
    in_flow_program = (
        eligible["ETF"].isin(flow_program_etfs)
        if "ETF" in eligible.columns
        else pd.Series(False, index=eligible.index)
    )
    return is_yieldboost, in_b2_universe, in_flow_program


def _load_core_decay_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "by_etf": {}}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("by_etf"), dict):
            out = {"version": int(d.get("version", 1)), "by_etf": dict(d["by_etf"])}
            return out
    except Exception:
        pass
    return {"version": 1, "by_etf": {}}


def _save_core_decay_state(path: Path, by_etf: dict[str, Any], run_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "updated_run_date": run_date, "by_etf": by_etf}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _core_net_decay_gate_for_core(
    eligible: pd.DataFrame,
    *,
    core_pre_decay: pd.Series,
    core_neg_decay_reset: pd.Series,
    core_rules: dict,
    state_path: Path,
    run_date: str,
) -> pd.Series:
    """Extra core-only gate from net_decay_annual (bucket-1 selectivity).

    ``core_neg_decay_reset``: structural core shape (beta, borrow) but **negative** net decay —
    clears that ETF's sticky flag so a later recovery requires the enter threshold again.

    Only rows satisfying ``core_pre_decay`` (structural core candidate) can be gated off; all
    other rows return pass-through True so non-core rows do not corrupt sticky state.

    - If hysteresis disabled and min_net_decay_annual <= 0: all-True.
    - If hysteresis disabled and min_net_decay_annual > 0: require finite net_decay >= min on
      ``core_pre_decay`` rows.
    - If hysteresis enabled: sticky per ETF in ``state_path``; ``enter_above`` > ``exit_below``.
      Missing decay: if sticky was on, keep on (data-gap churn guard); otherwise off.
    - Whenever ``min_net_decay_annual`` > 0, that floor is **always** applied last: no NaN bypass,
      no hysteresis carry can admit core below the minimum (``sticky_in`` in state matches final).
    """
    nd = pd.to_numeric(eligible["net_decay_annual"], errors="coerce")
    min_nd = float(core_rules.get("min_net_decay_annual", 0.0) or 0.0)
    hyst_cfg = core_rules.get("net_decay_hysteresis") or {}
    hyst_on = bool(hyst_cfg.get("enabled", False))

    gate = pd.Series(True, index=eligible.index)
    idx_core = eligible.loc[core_pre_decay.fillna(False)].index

    if not hyst_on:
        if min_nd <= 0:
            return gate
        gate.loc[idx_core] = np.isfinite(nd.loc[idx_core]) & (nd.loc[idx_core] >= min_nd)
        return gate

    enter = float(hyst_cfg.get("enter_above", 0.0))
    exit_b = float(hyst_cfg.get("exit_below", -1.0))
    if enter <= exit_b:
        raise ValueError(
            "portfolio.sleeves.core_leveraged.rules.net_decay_hysteresis: "
            "enter_above must be strictly greater than exit_below"
        )

    doc = _load_core_decay_state(state_path)
    by_etf: dict[str, Any] = deepcopy(doc.get("by_etf", {}))

    for idx in eligible.loc[core_neg_decay_reset.fillna(False)].index:
        etf = _norm_sym(str(eligible.loc[idx, "ETF"]))
        by_etf[etf] = {"sticky_in": False}

    for idx in idx_core:
        etf = _norm_sym(str(eligible.loc[idx, "ETF"]))
        v_raw = nd.loc[idx]
        v = float(v_raw) if pd.notna(v_raw) else float("nan")
        prev_entry = by_etf.get(etf, {})
        prev = prev_entry.get("sticky_in") if isinstance(prev_entry, dict) else None
        prev_on = prev is True

        if not np.isfinite(v):
            raw = prev_on
        elif prev_on:
            raw = bool(v >= exit_b)
        else:
            raw = bool(v >= enter)

        if min_nd > 0.0:
            meets_min = bool(np.isfinite(v) and v >= min_nd)
        else:
            meets_min = True
        final_in = bool(raw) and meets_min
        gate.loc[idx] = final_in
        by_etf[etf] = {"sticky_in": final_in}

    _save_core_decay_state(state_path, by_etf, run_date)
    return gate



def _sizing_signal_column(weighting_cfg: dict, *, sleeve_name: str | None = None) -> tuple[str, str]:
    """
    Return (mode, column_name) for decay_score sizing.

    *mode* is ``blended_decay`` or ``net_edge`` (affects fallbacks only). The default primary
    signal for **all** sleeves is ``net_edge_p50_annual``. Override via ``sizing_edge_column``.
    """
    mode = str(weighting_cfg.get("sizing_signal", "blended_decay") or "blended_decay").lower().strip()
    if mode in ("blended", "blended_decay", "decay", "gross_decay"):
        mode = "blended_decay"
    elif mode in ("net_edge", "edge", "net"):
        mode = "net_edge"
    else:
        mode = "blended_decay"

    col = weighting_cfg.get("sizing_edge_column")
    if isinstance(col, str) and col.strip():
        return mode, col.strip()

    return mode, "net_edge_p50_annual"


def _b4_eligibility_edge_column(df: pd.DataFrame) -> str:
    """Screener column for B4 ``min_net_edge_annual`` gate (inverse_decay_bucket4)."""
    if "net_edge_p50_annual" in df.columns:
        return "net_edge_p50_annual"
    if "bucket4_net_edge_annual" in df.columns:
        return "bucket4_net_edge_annual"
    return "net_decay_annual"


def _trend_percentile_signal(df: pd.DataFrame, weighting_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return optional trend percentile + multiplier arrays for sizing.

    B1 and B4 production config use ``und_trend_ratio_fwd_60d`` with
    ``percentile_mode: cross_sectional``. The historical vol-shape percentile
    columns (for example ``und_trend_ratio_60d_pctile``) are diagnostics, not
    same-day sizing inputs.
    """
    cfg = weighting_cfg.get("trend_percentile_multiplier") or {}
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return np.full(len(df), np.nan, dtype=float), np.ones(len(df), dtype=float)

    col = str(cfg.get("column", "und_trend_ratio_60d") or "").strip()
    if not col:
        return np.full(len(df), np.nan, dtype=float), np.ones(len(df), dtype=float)

    alpha = float(cfg.get("alpha", 1.0) or 0.0)
    neutral = float(cfg.get("neutral_pctile", 0.5) or 0.5)
    floor = float(cfg.get("floor", 0.5) or 0.5)
    ceiling = float(cfg.get("ceiling", 1.5) or 1.5)
    if floor > ceiling:
        floor, ceiling = ceiling, floor

    percentile_mode = str(cfg.get("percentile_mode", "as_is") or "as_is").lower().strip()
    if col in df.columns:
        raw = pd.to_numeric(df[col], errors="coerce")
        if percentile_mode in ("cross_sectional", "xsec", "rank"):
            pctile = raw.rank(pct=True, method="average")
        else:
            pctile = raw.clip(lower=0.0, upper=1.0)
    else:
        pctile = pd.Series(np.nan, index=df.index)

    missing_mode = str(cfg.get("missing", "neutral") or "neutral").lower().strip()
    if missing_mode in ("floor", "min"):
        pctile = pctile.fillna(1.0)
    elif missing_mode in ("ceiling", "max"):
        pctile = pctile.fillna(0.0)
    else:
        pctile = pctile.fillna(neutral)

    mult = 1.0 + alpha * (neutral - pctile.to_numpy(dtype=float))
    return pctile.to_numpy(dtype=float), np.clip(mult, floor, ceiling)


def _trend_percentile_multiplier(df: pd.DataFrame, weighting_cfg: dict) -> np.ndarray:
    """Optional continuous trend tilt multiplier; see ``_trend_percentile_signal``."""
    _pctile, mult = _trend_percentile_signal(df, weighting_cfg)
    return mult


def _with_b1_trend_audit_columns(df: pd.DataFrame, weighting_cfg: dict) -> pd.DataFrame:
    """Attach B1 trend sizing audit columns to a sized core-leveraged frame."""
    out = df.copy()
    pctile, mult = _trend_percentile_signal(out, weighting_cfg)
    out["b1_trend_xsec_pctile"] = pctile
    out["b1_trend_multiplier"] = mult
    return out


def _decay_score_weights(
    df: pd.DataFrame,
    weighting_cfg: dict,
    delta_col: str = "delta_abs",
    sleeve_name: str | None = None,
) -> np.ndarray:
    """Compute portfolio weights from decay-score signal blended with equal weight.

    Parameters
    ----------
    df : DataFrame of eligible names. Must contain *delta_col* and borrow ``borrow_current``.
         Primary signal column defaults to ``net_edge_p50_annual`` for every sleeve (override
         with ``sizing_edge_column``). If that column is missing or all-NaN, falls back to
         ``blended_gross_decay``. Same borrow discount on the resolved signal in all modes.
    weighting_cfg : sleeve ``weighting`` dict from strategy_config.yml.
    delta_col : column name for absolute-value beta (default ``delta_abs``).
    sleeve_name : optional sleeve key (e.g. ``inverse_decay_bucket4``) for net-edge column defaults.

    Returns
    -------
    np.ndarray of weights summing to 1.0, aligned with *df* row order.
    """
    n = len(df)
    if n == 0:
        return np.array([])

    # --- config ----------------------------------------------------------
    borrow_aversion = float(weighting_cfg.get("borrow_aversion", 1.0))
    margin_power    = float(weighting_cfg.get("margin_efficiency_power", 0.0))
    eq_blend        = _clamp01(weighting_cfg.get("eq_blend", 0.0))
    max_w           = float(weighting_cfg.get("max_name_weight", 1.0))
    # New iteration knobs (default: identity / off so baseline behavior is unchanged).
    score_p         = float(weighting_cfg.get("score_concavity_p", 1.0) or 1.0)

    # --- raw sizing score (same borrow discount as blended decay path) ---
    _mode, sig_col = _sizing_signal_column(weighting_cfg, sleeve_name=sleeve_name)
    if sig_col in df.columns:
        signal = pd.to_numeric(df[sig_col], errors="coerce")
    else:
        signal = pd.Series(np.nan, index=df.index)
    if bool(signal.isna().all()) and "blended_gross_decay" in df.columns:
        signal = pd.to_numeric(df["blended_gross_decay"], errors="coerce")

    borrow = pd.to_numeric(df["borrow_current"], errors="coerce").fillna(0.0)
    raw_score = signal - borrow_aversion * borrow  # higher = better (annual units)

    # --- Candidate A: signed-power concavity on the raw score -------------
    # score = sign(raw) * |raw|^p; p in (0, 1] de-peaks the top tail without flipping signs.
    if abs(score_p - 1.0) > 1e-9:
        rs = pd.to_numeric(raw_score, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        raw_score = pd.Series(np.sign(rs) * np.power(np.abs(rs), float(score_p)), index=df.index)

    # --- margin efficiency adjustment ------------------------------------
    delta_abs = pd.to_numeric(df[delta_col], errors="coerce").clip(lower=0.1)
    margin_adj = np.power(1.0 / delta_abs, margin_power)  # favours 2x over 3x
    adjusted = (raw_score * margin_adj).fillna(0.0).clip(lower=0.0)

    # --- normalise signal weights ----------------------------------------
    sig_total = adjusted.sum()
    signal_w = adjusted.values / sig_total if sig_total > 0 else np.zeros(n)

    # --- blend with equal weight -----------------------------------------
    eq_w = np.ones(n) / n
    final_w = eq_blend * eq_w + (1.0 - eq_blend) * signal_w

    # --- optional trend-percentile tilt ----------------------------------
    _trend_pctile, trend_mult = _trend_percentile_signal(df, weighting_cfg)
    if not np.allclose(trend_mult, 1.0):
        final_w = final_w * trend_mult
        fw_sum = final_w.sum()
        final_w = final_w / fw_sum if fw_sum > 0 else eq_w

    # --- max-name-weight cap with redistribution -------------------------
    for _ in range(10):
        excess = np.maximum(final_w - max_w, 0.0)
        total_excess = excess.sum()
        if total_excess < 1e-12:
            break
        final_w = np.minimum(final_w, max_w)
        uncapped = final_w < max_w - 1e-12
        if uncapped.any():
            uc_total = final_w[uncapped].sum()
            if uc_total > 0:
                final_w[uncapped] += total_excess * (final_w[uncapped] / uc_total)
        else:
            break

    # --- safety normalisation --------------------------------------------
    s = final_w.sum()
    return final_w / s if s > 0 else eq_w


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_shares_outstanding_map(paths_cfg: dict) -> tuple[dict[str, float], Path | None]:
    """
    Load ETF shares outstanding map from etf-dashboard CSV.
    Returns (symbol->shares_outstanding, resolved_path).
    """
    configured = paths_cfg.get("etf_shares_outstanding_csv")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(str(configured)))
    candidates.extend(
        [
            Path("../etf-dashboard/data/etf_shares_outstanding.csv"),
            Path("../etf-dashboard/etf_shares_outstanding.csv"),
            Path("data/etf_shares_outstanding.csv"),
        ]
    )
    src = _first_existing_path(candidates)
    if src is None:
        return {}, None

    try:
        df = pd.read_csv(src)
    except Exception:
        return {}, src
    if df.empty:
        return {}, src

    symbol_col = None
    for c in ["ETF", "ticker", "symbol", "Ticker", "Symbol"]:
        if c in df.columns:
            symbol_col = c
            break
    shares_col = None
    for c in [
        "shares_outstanding",
        "sharesOutstanding",
        "SharesOutstanding",
        "total_shares_outstanding",
        "totalSharesOutstanding",
        "shares",
    ]:
        if c in df.columns:
            shares_col = c
            break
    if symbol_col is None or shares_col is None:
        return {}, src

    out: dict[str, float] = {}
    for _, r in df[[symbol_col, shares_col]].dropna().iterrows():
        sym = _norm_sym(r[symbol_col])
        sh = pd.to_numeric(r[shares_col], errors="coerce")
        if pd.notna(sh) and float(sh) > 0:
            out[sym] = float(sh)
    return out, src


def _apply_notional_caps_with_redistribution(
    desired: pd.Series,
    caps: pd.Series,
) -> pd.Series:
    """
    Cap desired notionals at per-row limits and redistribute excess to uncapped rows.
    """
    desired = pd.to_numeric(desired, errors="coerce").fillna(0.0).clip(lower=0.0)
    caps = pd.to_numeric(caps, errors="coerce")
    caps = np.where((~np.isfinite(caps)) | (caps <= 0), np.inf, caps)
    caps_s = pd.Series(caps, index=desired.index, dtype=float)

    alloc = desired.copy()
    for _ in range(20):
        over = (alloc - caps_s).clip(lower=0.0)
        excess = float(over.sum())
        if excess <= 1e-9:
            break
        alloc = np.minimum(alloc, caps_s)
        headroom_mask = alloc < (caps_s - 1e-9)
        if not headroom_mask.any():
            break
        w = desired[headroom_mask].copy()
        wsum = float(w.sum())
        if wsum <= 1e-12:
            w = pd.Series(1.0, index=w.index)
            wsum = float(w.sum())
        alloc.loc[headroom_mask] = alloc.loc[headroom_mask] + excess * (w / wsum)
    alloc = np.minimum(alloc, caps_s)
    return alloc


def _pick_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in cols:
            return cols[key]
    return None


def _short_leg_frac_array(
    delta_abs: np.ndarray,
    delta_floor: float,
    sleeve: np.ndarray,
) -> np.ndarray:
    """
    Fraction of pair gross that is ETF short USD (stock sleeves: hedge_ratio / (1+hr)).
    Bucket 4: gross is the inverse ETF short leg → fraction 1.0.
    """
    b = np.maximum(np.asarray(delta_abs, dtype=float), float(delta_floor))
    hr = 1.0 / b
    sf = hr / (1.0 + hr)
    s = np.asarray(sleeve).astype(str)
    is_b4 = (s == "inverse_decay_bucket4") | (s == VOL_ETP_BUCKET5_SLEEVE)
    sf = np.where(is_b4, 1.0, sf)
    return np.clip(sf, 1e-12, 1.0)


def _project_to_capped_simplex_numpy(desired: np.ndarray, caps: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Match backtest notebook: project nonnegative desired onto caps, sum to 1."""
    d = np.asarray(desired, dtype=float).copy()
    c = np.asarray(caps, dtype=float).copy()
    n = len(d)
    if n == 0:
        return d
    d = np.clip(d, 0.0, None)
    s = float(d.sum())
    if s <= 0:
        d[:] = 1.0 / n
    else:
        d /= s
    c = np.clip(c, 0.0, None)
    if float(c.sum()) < 1.0 - tol:
        c = np.maximum(c, d)
    w = np.zeros(n, dtype=float)
    free = np.ones(n, dtype=bool)
    rem = 1.0
    while bool(free.any()) and rem > tol:
        idx = np.where(free)[0]
        base = d[idx]
        bsum = float(base.sum())
        trial = (rem * base / bsum) if bsum > 0 else np.full(len(idx), rem / max(len(idx), 1))
        hit = trial > (c[idx] + tol)
        if not bool(hit.any()):
            w[idx] = trial
            rem = 0.0
            break
        hit_idx = idx[hit]
        w[hit_idx] = c[hit_idx]
        rem = 1.0 - float(w.sum())
        free[hit_idx] = False
    if rem > tol:
        idx = np.where(free)[0]
        if len(idx):
            room = np.clip(c[idx] - w[idx], 0.0, None)
            rs = float(room.sum())
            if rs > tol:
                w[idx] += rem * (room / rs)
    w = np.clip(w, 0.0, None)
    s2 = float(w.sum())
    if s2 > 0:
        w /= s2
    m = np.minimum(w, c)
    ms = float(m.sum())
    return m / max(ms, 1e-12)


def _project_pair_and_underlying_numpy(
    w: np.ndarray,
    pair_caps: np.ndarray,
    und_code: np.ndarray,
    und_cap: float,
    *,
    max_iter: int = 400,
    tol: float = 1e-11,
) -> np.ndarray:
    """Iterative projection (pair caps + per-underlying mass cap). Weights sum to ≤ 1."""
    w = np.asarray(w, dtype=float).copy()
    pc = np.asarray(pair_caps, dtype=float).copy()
    uc = np.asarray(und_code, dtype=int).copy()
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w[:] = 1.0 / max(len(w), 1)
    else:
        w /= w.sum()
    pc = np.clip(pc, 0.0, None)
    n = len(w)
    if n == 0:
        return w
    max_u = int(uc.max()) + 1 if n else 0
    for _ in range(int(max_iter)):
        prev = w.copy()
        w = np.minimum(w, pc)
        und_tot = np.bincount(uc, weights=w, minlength=max_u)
        for u in range(max_u):
            tot = float(und_tot[u])
            if tot > float(und_cap) + tol:
                mask = uc == u
                if tot > 0:
                    w[mask] *= float(und_cap) / tot
        rem = 1.0 - float(w.sum())
        if rem > tol:
            und_tot = np.bincount(uc, weights=w, minlength=max_u)
            und_head = np.clip(float(und_cap) - und_tot, 0.0, None)
            name_und_head = und_head[uc]
            pair_head = np.clip(pc - w, 0.0, None)
            head = np.minimum(pair_head, name_und_head)
            hs = float(head.sum())
            if hs > tol:
                w += rem * (head / hs)
        # Do not force renorm to 1 here: if constraints leave slack unplaceable, total weight < 1
        # (partial gross deployment vs target book).
        if float(np.max(np.abs(w - prev))) < 1e-10:
            break
    st = float(w.sum())
    if st > 1.0 + 1e-9:
        w /= st
    return w


_STRUCTURAL_CAP_KEYS = ("aum", "shares_outstanding")
_DAY_LIQUIDITY_CAP_KEYS = ("shares_available", "adv")
_ALL_LIQUIDITY_CAP_KEYS = _STRUCTURAL_CAP_KEYS + _DAY_LIQUIDITY_CAP_KEYS


def _resolve_cap_subset(cap_mode: str | None) -> tuple[str, ...]:
    """Map ``cap_mode`` to the liquidity inputs that participate in
    :func:`_liquidity_tight_book_weights`.

    Modes:
      - ``structural_only`` — AUM + shares_outstanding (used for the **optimal** target).
      - ``day_liquidity_only`` — shares_available + ADV (used for what today's market allows).
      - ``structural_plus_day_liquidity`` (default) — full legacy stack used to size the
        **executable** target. Behavior unchanged from prior versions when this is the mode.
    """
    raw = str(cap_mode or "structural_plus_day_liquidity").strip().lower()
    if raw in ("structural_only", "structural", "optimal"):
        return _STRUCTURAL_CAP_KEYS
    if raw in ("day_liquidity_only", "day", "liquidity_only", "executable_only"):
        return _DAY_LIQUIDITY_CAP_KEYS
    return _ALL_LIQUIDITY_CAP_KEYS


def _liquidity_tight_book_weights(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    delta_floor: float,
    caps: dict[str, Any],
    shares_out_map: dict[str, float],
    cap_subset: tuple[str, ...] | None = None,
    return_binding_label: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Per-row upper bound on **book** weight ``gross_i / sum(gross)`` implied by liquidity
    inputs only (AUM, shares available, shares outstanding, median volume vs short leg).

    ``cap_subset`` selects which inputs participate; ``None`` keeps the full legacy stack.
    With ``return_binding_label=True``, also returns an array of strings labeling which input
    bound each row (``aum`` / ``shares_outstanding`` / ``shares_available`` / ``adv`` /
    ``missing_shares``).
    """
    n = len(sized)
    if n == 0:
        empty = np.zeros(0, dtype=float)
        if return_binding_label:
            return empty, np.array([], dtype=object)
        return empty
    T = max(float(target_gross_usd), 1.0)
    subset = set(cap_subset) if cap_subset is not None else set(_ALL_LIQUIDITY_CAP_KEYS)
    missing = float(caps.get("missing_shares_cap", 0.02) or 0.02)
    aum_pct = float(caps.get("aum_use_pct", 0.0) or 0.0) if "aum" in subset else 0.0
    sh_av_pct = float(caps.get("short_avail_use_pct", 0.25) or 0.25) if "shares_available" in subset else 0.0
    sout_frac = float(caps.get("shares_outstanding_use_frac", 0.0) or 0.0) if "shares_outstanding" in subset else 0.0
    mv_pct = float(caps.get("median_daily_volume_use_pct", 0.0) or 0.0) if "adv" in subset else 0.0

    if "delta_abs" in sized.columns:
        ba = pd.to_numeric(sized["delta_abs"], errors="coerce").to_numpy(dtype=float)
    else:
        ba = pd.to_numeric(sized["Delta"], errors="coerce").abs().to_numpy(dtype=float)
    if "sleeve" in sized.columns:
        slv = sized["sleeve"].astype(str).to_numpy()
    else:
        slv = np.array(["core_leveraged"] * n, dtype=object)
    sf = _short_leg_frac_array(ba, delta_floor, slv)
    ref_short = T * sf

    if "borrow_price_ref" in sized.columns:
        px = pd.to_numeric(sized["borrow_price_ref"], errors="coerce").to_numpy(dtype=float)
    else:
        px = np.full(n, np.nan, dtype=float)

    if "shares_available" in sized.columns:
        sh_av = pd.to_numeric(sized["shares_available"], errors="coerce").to_numpy(dtype=float)
    else:
        sh_av = np.full(n, np.nan, dtype=float)
    aum_col = caps.get("aum_column")
    if isinstance(aum_col, str) and aum_col in sized.columns:
        aum = pd.to_numeric(sized[aum_col], errors="coerce").to_numpy(dtype=float)
    else:
        ac = _pick_first_column(
            sized,
            ["aum_usd", "etf_aum_usd", "AUM_USD", "aum", "etf_aum", "nav_aum_usd"],
        )
        aum = pd.to_numeric(sized[ac], errors="coerce").to_numpy(dtype=float) if ac else np.full(n, np.nan)

    mv_col = caps.get("median_volume_column")
    if isinstance(mv_col, str) and mv_col in sized.columns:
        med_vol = pd.to_numeric(sized[mv_col], errors="coerce").to_numpy(dtype=float)
    else:
        vc = _pick_first_column(
            sized,
            [
                "median_daily_volume_shares",
                "median_volume_shares_60d",
                "adv_median_shares",
                "adv_shares_median",
                "median_volume_shares",
            ],
        )
        med_vol = pd.to_numeric(sized[vc], errors="coerce").to_numpy(dtype=float) if vc else np.full(n, np.nan)

    etf_syms = sized["ETF"].astype(str).map(_norm_sym).to_numpy()
    sh_out = np.array([float(shares_out_map.get(str(e), float("nan"))) for e in etf_syms], dtype=float)

    def _w_cap(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den
        out = np.where(np.isfinite(out) & (out > 0), out, np.nan)
        return out

    pieces: list[np.ndarray] = []
    labels: list[str] = []
    if aum_pct > 0:
        num = aum_pct * np.where(np.isfinite(aum) & (aum > 0), aum, np.nan)
        pieces.append(_w_cap(num, ref_short)); labels.append("aum")
    if sh_av_pct > 0:
        num = sh_av_pct * np.where(np.isfinite(sh_av) & (sh_av > 0), sh_av, np.nan) * px
        pieces.append(_w_cap(num, ref_short)); labels.append("shares_available")
    if sout_frac > 0 and shares_out_map:
        num = sout_frac * np.where(np.isfinite(sh_out) & (sh_out > 0), sh_out, np.nan) * px
        pieces.append(_w_cap(num, ref_short)); labels.append("shares_outstanding")
    if mv_pct > 0:
        num = mv_pct * np.where(np.isfinite(med_vol) & (med_vol > 0), med_vol, np.nan) * px
        pieces.append(_w_cap(num, ref_short)); labels.append("adv")

    if pieces:
        mat = np.column_stack(pieces)
        row_min = np.nanmin(mat, axis=1)
        tight = np.where(np.isfinite(row_min), row_min, missing)
        if return_binding_label:
            arg = np.zeros(n, dtype=int)
            finite_mask = np.isfinite(mat)
            mat_for_argmin = np.where(finite_mask, mat, np.inf)
            arg = np.argmin(mat_for_argmin, axis=1)
            binding = np.array([labels[i] for i in arg], dtype=object)
            binding[~np.isfinite(row_min)] = "missing_shares"
    else:
        tight = np.full(n, missing, dtype=float)
        binding = np.full(n, "missing_shares", dtype=object) if return_binding_label else None
    out = np.clip(tight, 0.0, None).astype(float)
    if return_binding_label:
        return out, binding  # type: ignore[return-value]
    return out


def _compute_pair_weight_caps_array(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    delta_floor: float,
    caps: dict[str, Any],
    shares_out_map: dict[str, float],
    cap_subset: tuple[str, ...] | None = None,
) -> np.ndarray:
    """
    Per-row max weight (fraction of total book gross) from hard pair cap + liquidity.
    Liquidity caps mirror backtest notebooks (AUM, shares_available, float, median volume).
    ``cap_subset`` restricts which liquidity inputs participate (see :func:`_resolve_cap_subset`).
    """
    n = len(sized)
    if n == 0:
        return np.zeros(0, dtype=float)
    max_pair = float(caps.get("max_pair_weight_cap", 0.05) or 0.05)
    tight = _liquidity_tight_book_weights(
        sized,
        target_gross_usd=target_gross_usd,
        delta_floor=delta_floor,
        caps=caps,
        shares_out_map=shares_out_map,
        cap_subset=cap_subset,
    )
    return np.minimum(max_pair, tight).astype(float)


def _liquidity_book_anchor_usd(
    *,
    deployed_gross_sum: float,
    strategy_target_gross_usd: float,
    caps: dict[str, Any],
) -> float:
    """Scale factor ``T`` in :func:`_liquidity_tight_book_weights` ``ref_short = T * sf``."""
    raw = str(caps.get("liquidity_book_reference", "target_book") or "target_book").strip().lower()
    if raw in {"deployed_book", "deployed", "current_book", "deployed_scale", "sum"}:
        return max(float(deployed_gross_sum), 1.0)
    if raw in {"target_book", "target", "strategy", "strategy_target"}:
        return max(float(strategy_target_gross_usd), 1.0)
    return max(float(strategy_target_gross_usd), 1.0)


def rescale_gross_targets_to_sleeve_budget_weights(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    sleeve_budget_usd: Mapping[str, float],
    rescale_to: str = "absolute_budget",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Multiply ``gross_target_usd`` within each YAML budgeted sleeve to match the budget target.

    Two modes via ``rescale_to``:
    - ``absolute_budget`` (default): each sleeve's deployed sum becomes its YAML
      ``budget_sleeve_usd`` directly. The total deployed gross matches sum(budgets) =
      ``target_gross_usd`` if all sleeves are present. Per-row caps run downstream may then clip
      excess (sleeves with insufficient capacity will under-fill).
    - ``fraction_of_deployed``: legacy — each sleeve's sum becomes
      ``budget_s / target_gross_usd × current_total_gross``, preserving total deployed gross.

    Within each sleeve, relative weights across rows are preserved. Sleeves with zero deployed
    gross are reported in ``skipped_zero_deployed``; leftover budget share is **not** reallocated
    here.
    """
    diag: dict[str, Any] = {"applied": False}
    if sized is None or sized.empty:
        return sized, diag
    tg = float(target_gross_usd)
    if tg <= 1e-18 or not isinstance(sleeve_budget_usd, Mapping):
        return sized.copy(), diag
    if "sleeve" not in sized.columns:
        diag["reason"] = "no_sleeve_column"
        return sized.copy(), diag

    mode = str(rescale_to or "absolute_budget").strip().lower()
    if mode not in ("absolute_budget", "fraction_of_deployed"):
        mode = "absolute_budget"

    out = sized.copy()
    # ``to_numpy`` can return a read-only view; we mutate ``gross`` in-place below.
    gross = pd.to_numeric(out["gross_target_usd"], errors="coerce").fillna(0.0).to_numpy(
        dtype=float, copy=True
    )
    slv = out["sleeve"].astype(str).to_numpy()
    S_before = float(np.sum(gross))
    if S_before <= 1e-18:
        diag["reason"] = "zero_gross"
        return out, diag

    fracs_before: dict[str, float] = {}
    skipped: list[str] = []
    budgets_f: dict[str, float] = {str(k): float(v) for k, v in sleeve_budget_usd.items()}

    for sleeve_key in budgets_f:
        m = slv == str(sleeve_key)
        if not bool(np.any(m)):
            continue
        fracs_before[str(sleeve_key)] = float(np.sum(gross[m])) / S_before

    for sleeve_key, bud in budgets_f.items():
        if mode == "absolute_budget":
            desired = float(bud)
        else:
            f_tar = bud / tg
            if f_tar <= 1e-15:
                continue
            desired = f_tar * S_before
        if desired <= 1e-15:
            continue
        m = slv == str(sleeve_key)
        if not np.any(m):
            continue
        curr = float(np.sum(gross[m]))
        if curr <= 1e-15:
            if desired > 1e-9:
                skipped.append(str(sleeve_key))
            continue
        gross[m] *= desired / curr

    S_after = float(np.sum(gross))
    fracs_after: dict[str, float] = {}
    if S_after > 1e-18:
        for sleeve_key in budgets_f:
            m = slv == str(sleeve_key)
            if not bool(np.any(m)):
                continue
            fracs_after[str(sleeve_key)] = float(np.sum(gross[m])) / S_after

    out["gross_target_usd"] = gross
    diag.update(
        {
            "applied": True,
            "rescale_to": mode,
            "gross_sum_before": S_before,
            "gross_sum_after": S_after,
            "sleeve_fractions_before_rescale": fracs_before,
            "sleeve_fractions_after_rescale": fracs_after,
            "skipped_zero_deployed": skipped,
        }
    )
    return out, diag


def _enforce_max_pair_weight_within_deployed_sleeve_gross(
    gross: np.ndarray,
    sized: pd.DataFrame,
    sleeve_caps: dict[str, dict[str, float]],
) -> np.ndarray:
    """
    Re-scale **within each sleeve** so no pair holds more than ``max_pair_weight`` times that
    sleeve's **deployed** gross (sum of ``gross_target_usd`` over rows in the sleeve).

    Liquidity can leave most sleeve rows at ~0 while a few absorb the budget; normalizing by
    *deployed* mass avoids a single name showing 60%+ of placed gross while still being under
    the target-book pair cap. Mass is redistributed across rows that already have positive
    gross after liquidity (we do not revive names hard-zeroed by liquidity).

    Diversification **within a sleeve** requires at least two sized rows; single-row sleeves are
    unchanged (that pair is the sleeve). For multi-row sleeves with only one positive-gross row,
    gross is clipped to ``max_pair_weight * sleeve_deployed_sum`` when needed.
    """
    g = np.asarray(gross, dtype=float).copy()
    if "sleeve" not in sized.columns or not sleeve_caps:
        return g
    slv = sized["sleeve"].astype(str).to_numpy()
    for sleeve, meta_cap in sleeve_caps.items():
        cap = float((meta_cap or {}).get("max_pair_weight", 1.0))
        if cap >= 1.0 - 1e-12:
            continue
        # When pre-cap score haircut already imposes per-row ceilings tighter than ``cap`` for
        # under-scored rows, this function would re-fill them via proportional redistribution
        # and undo the haircut. Skip the deployed-sleeve enforcement for those sleeves.
        if bool((meta_cap or {}).get("_skip_deployed_enforce", False)):
            continue
        idx = np.where(slv == str(sleeve))[0]
        if len(idx) <= 1:
            # One pair defines the whole sleeve; ``max_pair_weight`` is for diversity across rows.
            continue
        s_sleeve = float(np.sum(g[idx]))
        if s_sleeve <= 1e-18:
            continue
        pos = idx[g[idx] > 1e-18]
        if len(pos) == 0:
            continue
        if len(pos) == 1:
            i = int(pos[0])
            max_g = cap * s_sleeve
            if g[i] > max_g + 1e-6:
                g[i] = max_g
            continue
        s_dep = float(g[pos].sum())
        if s_dep <= 1e-18:
            continue
        w = g[pos] / s_dep
        if float(np.max(w)) <= cap + 1e-8:
            continue
        n_p = len(pos)
        cap_eff = float(cap)
        # Capped simplex {w: sum w = 1, w_i <= cap} is non-empty iff n * cap >= 1.
        if n_p * cap_eff < 1.0 - 1e-10:
            cap_eff = 1.0 / float(n_p)
        w_new = _project_positive_weights_to_sum_one_hard_cap(w, cap_eff)
        g[pos] = w_new * s_dep
    return g


def _project_positive_weights_to_sum_one_hard_cap(w: np.ndarray, cap: float, *, tol: float = 1e-10) -> np.ndarray:
    """
    Find ``w_out`` with ``w_out.sum() == 1`` (or maximal feasible ``< 1`` if ``n * cap < 1``),
    ``0 <= w_out[i] <= cap``, preferring to stay close to proportional redistribution
    from an initial ``w``.

    Unlike :func:`_project_to_capped_simplex_numpy`, caps here are **hard** — they must not be
    relaxed toward an infeasibly concentrated desired vector.
    """
    w0 = np.asarray(w, dtype=float).copy()
    if len(w0) == 0:
        return w0
    s0 = float(np.sum(w0))
    if s0 <= 1e-18:
        n = len(w0)
        return np.full(n, 1.0 / max(n, 1), dtype=float)
    w0 /= s0
    cap_f = float(cap)
    w0 = np.clip(w0, 0.0, cap_f)
    n = len(w0)
    if n * cap_f < 1.0 - tol:
        return np.full(n, cap_f, dtype=float)
    for _ in range(max(4, n * 8)):
        rem = 1.0 - float(np.sum(w0))
        if abs(rem) <= tol:
            break
        if rem < 0:
            tot = float(np.sum(w0))
            if tot > 1e-18:
                w0 /= tot
            w0 = np.clip(w0, 0.0, cap_f)
            continue
        head = np.clip(cap_f - w0, 0.0, None)
        hs = float(np.sum(head))
        if hs <= 1e-18:
            break
        w0 += rem * (head / hs)
        w0 = np.clip(w0, 0.0, cap_f)
    s1 = float(np.sum(w0))
    if s1 >= 1.0 - tol:
        w0 /= max(s1, 1e-18)
    # else: infeasible to reach sum 1 under cap — keep sub‑unit mass (caller scales gross down)
    return np.clip(w0, 0.0, cap_f)


def _apply_gross_sizing_per_sleeve_book_caps(
    sized: pd.DataFrame,
    gross: np.ndarray,
    *,
    target_gross_usd: float,
    delta_floor: float,
    caps: dict[str, Any],
    shares_out_map: dict[str, float],
    cap_subset: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Diamond-Creek-style **per-sleeve** concentration: ``max_pair_weight`` and
    ``max_underlying_weight`` apply to weights **within each sleeve's allocated gross**,
    while liquidity rows still cap **book** weights (same construction as
    :func:`_liquidity_tight_book_weights`), passing ``target_gross_usd`` as the resolved **liquidity
    book anchor** ``T`` from ``strategy.gross_sizing_caps.liquidity_book_reference`` (deployed sum vs YAML).
    Otherwise preserves each sleeve's share of total gross through the capped simplex projection.

    ``cap_subset`` selects which liquidity inputs participate (structural vs day-liquidity).
    Returned meta exposes ``binding_per_row`` strings for downstream ``binding_cap`` columns.
    """
    meta: dict[str, Any] = {}
    n = len(sized)
    if n == 0:
        meta["binding_per_row"] = np.array([], dtype=object)
        return gross, meta
    gsum = float(np.sum(gross))
    if gsum <= 1e-18:
        meta["binding_per_row"] = np.full(n, "none", dtype=object)
        return gross, meta
    per_sleeve = caps.get("per_sleeve") or {}
    if not isinstance(per_sleeve, dict) or not per_sleeve:
        meta["binding_per_row"] = np.full(n, "none", dtype=object)
        return gross, meta
    if "sleeve" not in sized.columns:
        meta["binding_per_row"] = np.full(n, "none", dtype=object)
        return gross, meta

    liq_book, liq_binding = _liquidity_tight_book_weights(
        sized,
        target_gross_usd=target_gross_usd,
        delta_floor=delta_floor,
        caps=caps,
        shares_out_map=shares_out_map,
        cap_subset=cap_subset,
        return_binding_label=True,
    )
    default_pair = float(caps.get("max_pair_weight_cap", 0.05) or 0.05)
    default_und = float(caps.get("max_underlying_weight_cap", 0.11) or 0.11)
    default_haircut = float(caps.get("pre_cap_score_haircut_multiplier", 0.0) or 0.0)

    slv = sized["sleeve"].astype(str).to_numpy()
    w = gross / gsum
    w_out = np.zeros_like(w, dtype=float)
    binding_per_row = np.full(n, "none", dtype=object)

    sleeve_caps_out: dict[str, dict[str, float]] = {}
    haircut_meta: dict[str, dict[str, Any]] = {}
    for sleeve in sorted({str(x) for x in slv.tolist()}):
        idx = np.where(slv == sleeve)[0]
        s_b = float(w[idx].sum())
        if s_b <= 1e-18:
            continue
        rules = per_sleeve.get(sleeve)
        if isinstance(rules, dict):
            mp_raw = rules.get("max_pair_weight", default_pair)
            mu_raw = rules.get("max_underlying_weight", default_und)
            mp_f = float(mp_raw) if mp_raw is not None and float(mp_raw) > 0 else default_pair
            mu_f = float(mu_raw) if mu_raw is not None and float(mu_raw) > 0 else default_und
            ha_raw = rules.get("pre_cap_score_haircut_multiplier", default_haircut)
            ha_f = float(ha_raw) if ha_raw is not None else default_haircut
        else:
            mp_f, mu_f = default_pair, default_und
            ha_f = default_haircut
        sleeve_caps_out[str(sleeve)] = {"max_pair_weight": mp_f, "max_underlying_weight": mu_f}

        v = w[idx] / s_b
        liq_slice = liq_book[idx]
        if len(idx) == 1:
            # One active row holds the entire sleeve; ``max_pair_weight`` is a diversification
            # knob across pairs and does not apply below 100% of sleeve gross. Liquidity still can.
            cap0 = min(1.0, float(liq_slice[0]) / max(s_b, 1e-18))
            cap_v = np.array([max(cap0, 1e-18)], dtype=float)
            cap_pair = np.array([1.0], dtype=float)
            cap_liq_v = np.array([float(liq_slice[0]) / max(s_b, 1e-18)], dtype=float)
            cap_haircut_v = np.full(1, np.inf, dtype=float)
        else:
            cap_pair = np.full(len(idx), float(mp_f), dtype=float)
            cap_liq_v = liq_slice / max(s_b, 1e-18)
            cap_v = np.minimum(mp_f, liq_slice / max(s_b, 1e-18))
            cap_v = np.clip(cap_v, 1e-18, None)
            cap_haircut_v = np.full(len(idx), np.inf, dtype=float)
            # Pre-cap **score haircut**: limit how much an under-weighted (low decay/edge score)
            # row can grow when the simplex projector redistributes excess from over-cap names.
            # Without this, the projector reallocates proportionally to **headroom** (cap − w),
            # which can lift a weak row like a high-borrow YieldBoost name from ~6% → 20%.
            #
            # ``ha_f`` is the maximum multiple of the row's pre-cap sleeve-internal weight that
            # the projector may grow it to. The baseline is the **frozen** ``_pre_cap_score_weight``
            # column (sleeve-internal, sums to 1 within sleeve) when present — survives covariance
            # and post-rebalance cap re-runs. If the column is absent (legacy), fall back to the
            # current sleeve-internal share ``v``.
            if ha_f > 0:
                if "_pre_cap_score_weight" in sized.columns:
                    base_w = pd.to_numeric(
                        sized["_pre_cap_score_weight"].iloc[idx], errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=float)
                    bs = float(base_w.sum())
                    if bs > 1e-18:
                        base_w = base_w / bs
                    else:
                        base_w = v
                else:
                    base_w = v
                cap_haircut_v = np.maximum(ha_f * base_w, 1e-18)
                cap_v = np.minimum(cap_v, cap_haircut_v)
                cap_v = np.clip(cap_v, 1e-18, None)
                haircut_meta[str(sleeve)] = {
                    "pre_cap_score_haircut_multiplier": float(ha_f),
                    "n_rows_capped_by_haircut": int(np.sum(cap_v < mp_f - 1e-12)),
                }
                # ``_enforce_max_pair_weight_within_deployed_sleeve_gross`` would otherwise project
                # back to ``max_pair_weight × deployed_sleeve_sum``, redistributing mass from haircut
                # winners back into the under-scored rows we just clipped down — defeats the haircut.
                # Mark this sleeve so the deployed enforcement step skips it.
                sleeve_caps_out[str(sleeve)]["_skip_deployed_enforce"] = True

        und_sub = sized.iloc[idx]["Underlying"].astype(str).map(_norm_sym).to_numpy()
        und_code = pd.factorize(und_sub)[0]
        v1 = _project_to_capped_simplex_numpy(v, cap_v)
        # Same reasoning as ``cap_v`` for a lone pair: one underlying may carry 100% of sleeve gross.
        mu_eff = 1.0 if len(idx) == 1 else float(mu_f)
        v2 = _project_pair_and_underlying_numpy(v1, cap_v, und_code, mu_eff)
        w_out[idx] = v2 * s_b

        # Record which cap is binding per row using the lowest finite cap value seen here.
        # Liquidity bucket label comes from ``_liquidity_tight_book_weights``; pair / haircut
        # are sleeve-internal projector caps.
        if len(idx) > 0:
            cap_stack = np.stack([cap_pair, cap_liq_v, cap_haircut_v], axis=0)
            argmin_local = np.argmin(cap_stack, axis=0)
            for k_local, irow in enumerate(idx):
                src = int(argmin_local[k_local])
                if src == 0:
                    binding_per_row[irow] = "pair_cap"
                elif src == 1:
                    binding_per_row[irow] = str(liq_binding[irow]) if irow < len(liq_binding) else "liquidity"
                else:
                    binding_per_row[irow] = "haircut"

    meta["per_sleeve_caps"] = sleeve_caps_out
    meta["binding_per_row"] = binding_per_row
    if haircut_meta:
        meta["pre_cap_score_haircut"] = haircut_meta
    new_gross = w_out * gsum
    new_gross = _enforce_max_pair_weight_within_deployed_sleeve_gross(
        new_gross,
        sized,
        sleeve_caps_out,
    )
    return new_gross, meta


def _shrunk_covariance(returns: pd.DataFrame, *, shrink: float) -> pd.DataFrame:
    """Sample covariance shrunk towards its diagonal: (1-s)*Σ + s*diag(Σ)."""
    cov = returns.cov(min_periods=1)
    diag = np.diag(np.diag(cov.values))
    s = float(np.clip(shrink, 0.0, 1.0))
    out = (1.0 - s) * cov.values + s * diag
    return pd.DataFrame(out, index=cov.index, columns=cov.columns)


def _normalize_underlying_returns(
    returns_df: pd.DataFrame | None,
    *,
    lookback: int,
    min_obs: int,
    needed_underlyings: list[str],
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Coerce a wide returns frame (dates × underlyings) of EITHER prices OR returns into
    log returns covering the *needed_underlyings* list.

    Returns ``(frame, None)`` on success, or ``(None, detail)`` when coverage is thin —
    *detail* is suitable for operator logs.
    """
    if returns_df is None or returns_df.empty:
        return None, "no_returns_frame (missing csv or empty table; see paths.underlying_returns_csv)"
    need_set = {_norm_sym(str(u)) for u in needed_underlyings}
    # Subset columns **before** copy — avoids O(rows × full_universe) allocation when callers
    # pass the full Yahoo matrix (notebook grid / diagnostics).
    pick: list[Any] = []
    seen_nc: set[str] = set()
    for col in returns_df.columns:
        nc = _norm_sym(str(col))
        if nc not in need_set or nc in seen_nc:
            continue
        seen_nc.add(nc)
        pick.append(col)
    if len(pick) < 2:
        hit = need_set & {_norm_sym(str(c)) for c in returns_df.columns}
        return (
            None,
            (
                f"need ≥2 overlapping underlying columns vs sized book; "
                f"matched={sorted(hit)} sized_underlyings={sorted(need_set)}"
            ),
        )
    df = returns_df.loc[:, pick].copy()
    df.columns = [_norm_sym(str(c)) for c in pick]
    df = df.replace([np.inf, -np.inf], np.nan)
    looks_like_prices = bool((df.dropna(how="all") > 1.0).any().mean() > 0.5) if len(df) else False
    if looks_like_prices:
        df = np.log(df.clip(lower=1e-12)).diff()
    df = df.dropna(axis=0, how="all")
    if len(df) > int(lookback):
        df = df.iloc[-int(lookback) :]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        return None, "fewer than 2 return columns after dropping all-NaN columns"
    if len(df) < int(min_obs):
        return (
            None,
            f"history rows {len(df)} < covariance_balance.min_obs ({min_obs}) "
            f"after lookback={lookback}",
        )
    return df, None


def _run_book_cap_pipeline_quiet(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    delta_floor: float,
    strategy: dict[str, Any],
    paths: dict[str, Any] | None,
    shares_out_map: dict[str, float],
    returns_df: pd.DataFrame | None,
    sleeve_budget_usd: Mapping[str, float] | None,
    cap_mode: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-run the executable cap → covariance → sleeve-rebalance → cap stack with a chosen
    ``cap_mode``. Used to compute the **optimal** (structural-only) target alongside the
    **executable** target, without duplicating logging or pipeline branching in main().

    Returns the rebuilt sized frame and a diag dict with both cap-stack diagnostics and
    a final ``binding_per_row`` array (per row: ``pair_cap`` / ``und_cap`` / ``shares_outstanding``
    / ``aum`` / ``shares_available`` / ``adv`` / ``haircut`` / ``missing_shares`` / ``none``).
    """
    out = sized.copy()
    diag: dict[str, Any] = {"cap_mode": str(cap_mode or "structural_plus_day_liquidity")}

    out, init_diag = apply_gross_sizing_book_caps(
        out,
        target_gross_usd=float(target_gross_usd),
        delta_floor=float(delta_floor),
        strategy=strategy,
        shares_out_map=shares_out_map,
        cap_mode=cap_mode,
    )
    diag["initial_cap"] = init_diag
    binding = init_diag.get("binding_per_row") if isinstance(init_diag, dict) else None

    cov_cfg = strategy.get("covariance_balance") or {}
    if isinstance(cov_cfg, dict) and bool(cov_cfg.get("enabled", False)) and returns_df is not None:
        per_scopes_raw = cov_cfg.get("per_sleeve_scopes")
        per_scopes = (
            [str(s).strip() for s in per_scopes_raw if str(s).strip()]
            if isinstance(per_scopes_raw, list) else []
        )
        if per_scopes:
            for sc in per_scopes:
                if "sleeve" not in out.columns:
                    break
                if str(sc) not in set(out["sleeve"].astype(str).unique()):
                    continue
                out, _ = apply_covariance_balance(
                    out,
                    target_gross_usd=float(target_gross_usd),
                    delta_floor=float(delta_floor),
                    strategy=strategy,
                    paths=paths,
                    shares_out_map=shares_out_map,
                    returns_df=returns_df,
                    covariance_scope=[sc],
                    cap_mode=cap_mode,
                )
        else:
            out, _ = apply_covariance_balance(
                out,
                target_gross_usd=float(target_gross_usd),
                delta_floor=float(delta_floor),
                strategy=strategy,
                paths=paths,
                shares_out_map=shares_out_map,
                returns_df=returns_df,
                cap_mode=cap_mode,
            )

    gs_reb_raw = strategy.get("gross_sizing_caps") or {}
    if (
        isinstance(gs_reb_raw, dict)
        and bool(gs_reb_raw.get("enabled", False))
        and bool(gs_reb_raw.get("rebalance_sleeve_weights_to_budget", False))
        and sleeve_budget_usd is not None
    ):
        rescale_to = str(gs_reb_raw.get("rebalance_sleeve_target_mode", "absolute_budget")).strip().lower()
        out, _ = rescale_gross_targets_to_sleeve_budget_weights(
            out,
            target_gross_usd=float(target_gross_usd),
            sleeve_budget_usd=sleeve_budget_usd,
            rescale_to=rescale_to,
        )
        out, final_diag = apply_gross_sizing_book_caps(
            out,
            target_gross_usd=float(target_gross_usd),
            delta_floor=float(delta_floor),
            strategy=strategy,
            shares_out_map=shares_out_map,
            cap_mode=cap_mode,
        )
        diag["final_cap"] = final_diag
        if isinstance(final_diag, dict) and final_diag.get("binding_per_row") is not None:
            binding = final_diag.get("binding_per_row")

    if binding is None:
        binding = np.full(len(out), "none", dtype=object)
    diag["binding_per_row"] = binding
    return out, diag


def apply_covariance_balance(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    delta_floor: float,
    strategy: dict[str, Any],
    paths: dict[str, Any] | None = None,
    shares_out_map: dict[str, float] | None = None,
    returns_df: pd.DataFrame | None = None,
    covariance_scope: Collection[str] | None = None,
    cap_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    One-shot covariance penalty on per-pair ``gross_target_usd`` to attenuate exposure to
    high-correlation underlyings, then re-apply :func:`apply_gross_sizing_book_caps` so
    pair / underlying / liquidity caps still bind.

    Aggregates exposure per underlying as ``e_u = Σ_p w_p · |β_p|``, computes diagonally-shrunk
    covariance Σ̃ on log-returns, marginal contribution ``mrc_u = (Σ̃ ê)_u`` and risk contribution
    ``c_u = max(ê_u·mrc_u, 0)``. Multiplier ``m_u = 1 / (1 + λ · c_u / median(c>0))`` is applied to
    every pair sharing that underlying. Gross is preserved by renormalizing to the cap-respecting
    sum, then re-projected through the cap stack.

    Config: ``strategy.covariance_balance`` (enabled, lookback_days, min_obs, shrink,
    penalty_strength, max_relative_shift). Disabled → no-op.

    When ``covariance_scope`` lists ``sleeve`` names, exposure uses only those rows, multipliers
    apply only there, and sleeve gross is preserved across the penalty before caps; other rows are
    untouched until the global cap projection. When omitted or empty, all rows run together
    (legacy whole-book pass).
    """
    diag: dict[str, Any] = {"applied": False}
    if sized is None or sized.empty:
        return sized, diag
    raw = strategy.get("covariance_balance")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return sized, diag
    cfg = dict(raw)
    lookback = int(cfg.get("lookback_days", 252) or 252)
    min_obs = int(cfg.get("min_obs", 60) or 60)
    shrink = float(cfg.get("shrink", 0.35) or 0.35)
    lam = float(cfg.get("penalty_strength", 0.85) or 0.85)
    max_shift = float(cfg.get("max_relative_shift", 0.50) or 0.50)

    scope_set: set[str] | None = None
    if covariance_scope is not None:
        scope_set = {str(s).strip() for s in covariance_scope if str(s).strip()}
        if not scope_set:
            scope_set = None

    df = sized.copy()
    df["ETF"] = df["ETF"].astype(str).map(_norm_sym)
    df["Underlying"] = df["Underlying"].astype(str).map(_norm_sym)
    if "delta_abs" not in df.columns:
        df["delta_abs"] = pd.to_numeric(df["Delta"], errors="coerce").abs()

    g = pd.to_numeric(df["gross_target_usd"], errors="coerce").fillna(0.0).clip(lower=0.0)
    gsum_book = float(g.sum())
    if gsum_book <= 1e-18:
        return sized, diag

    if scope_set is not None:
        row_mask = df["sleeve"].astype(str).isin(scope_set)
    else:
        row_mask = pd.Series(True, index=df.index)
    if not bool(row_mask.any()):
        diag.update({"applied": False, "reason": "empty_covariance_scope"})
        return sized, diag

    needed = sorted({str(u) for u in df.loc[row_mask, "Underlying"].astype(str).tolist()})
    R, ret_detail = _normalize_underlying_returns(
        returns_df, lookback=lookback, min_obs=min_obs, needed_underlyings=needed,
    )
    if R is None:
        diag.update(
            {
                "applied": False,
                "reason": "insufficient_returns",
                "returns_skip_detail": ret_detail,
            }
        )
        return sized, diag

    syms = [c for c in R.columns]
    sym_idx = {s: i for i, s in enumerate(syms)}

    delta_abs = pd.to_numeric(df["delta_abs"], errors="coerce").fillna(1.0).clip(lower=float(delta_floor))
    exposure = np.zeros(len(syms), dtype=float)
    sub_und = df.loc[row_mask, "Underlying"].astype(str).tolist()
    sub_g = g.loc[row_mask].tolist()
    sub_b = delta_abs.loc[row_mask].tolist()
    for u, w_pair, b in zip(sub_und, sub_g, sub_b):
        i = sym_idx.get(str(u))
        if i is None:
            continue
        exposure[i] += float(w_pair) * float(b)
    e_sum = float(exposure.sum())
    if e_sum <= 1e-18:
        diag.update({"applied": False, "reason": "no_exposure_overlap"})
        return sized, diag
    e_hat = exposure / e_sum

    Sigma = _shrunk_covariance(R, shrink=shrink).values
    mrc = Sigma @ e_hat
    contrib = np.clip(e_hat * mrc, 0.0, None)
    pos = contrib[contrib > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    med = max(med, 1e-18)
    contrib_norm = contrib / med
    mult = 1.0 / (1.0 + lam * contrib_norm)
    if max_shift > 0:
        mult = np.clip(mult, 1.0 - max_shift, 1.0 + max_shift)
    mult_by_und = {syms[i]: float(mult[i]) for i in range(len(syms))}

    g_arr = g.to_numpy(dtype=float)
    mask_np = row_mask.to_numpy(dtype=bool)
    new_g = g_arr.copy()
    for i in range(len(df)):
        if not mask_np[i]:
            continue
        u = str(df["Underlying"].iloc[i])
        m = float(mult_by_und.get(u, 1.0))
        new_g[i] = float(g_arr[i]) * m
    scope_sum_before = float(np.sum(g_arr[mask_np]))
    scope_sum_after = float(np.sum(new_g[mask_np]))
    if scope_sum_after <= 1e-18:
        diag.update({"applied": False, "reason": "all_zero_after_penalty"})
        return sized, diag

    new_g[mask_np] *= scope_sum_before / scope_sum_after

    out = sized.copy()
    out["gross_target_usd"] = new_g

    out, _cap_diag = apply_gross_sizing_book_caps(
        out,
        target_gross_usd=float(target_gross_usd),
        delta_floor=float(delta_floor),
        strategy=strategy,
        shares_out_map=(shares_out_map or {}),
        cap_mode=cap_mode,
    )

    attenuated = sorted(
        ((u, float(m)) for u, m in mult_by_und.items() if m < 1.0 - 1e-9),
        key=lambda kv: kv[1],
    )[:5]
    book_after = float(pd.to_numeric(out["gross_target_usd"], errors="coerce").fillna(0).sum())
    diag.update(
        {
            "applied": True,
            "covariance_scope": sorted(scope_set) if scope_set else None,
            "n_rows_scope": int(np.sum(mask_np)),
            "n_underlyings": int(len(syms)),
            "obs_used": int(len(R)),
            "lookback_used": int(min(len(R), lookback)),
            "shrink": float(shrink),
            "penalty_strength": float(lam),
            "max_relative_shift": float(max_shift),
            "gross_sum_book_before": gsum_book,
            "gross_sum_scope_before": scope_sum_before,
            "gross_sum_scope_after_caps": float(
                pd.to_numeric(out.loc[row_mask, "gross_target_usd"], errors="coerce").fillna(0.0).sum()
            ),
            "gross_sum_before": gsum_book,
            "gross_sum_after": book_after,
            "median_risk_contrib": float(med),
            "top_attenuated_underlyings": attenuated,
            "post_cov_cap_diag": _cap_diag,
        }
    )
    return out, diag


def _apply_weight_hysteresis(
    sized: pd.DataFrame,
    new_gross: np.ndarray,
    *,
    prev_gross_by_pair: dict[tuple[str, str], float] | None,
    abs_threshold: float,
    rel_threshold: float,
) -> np.ndarray:
    """Stability #2: snap each pair's new gross to its prior value when the change is
    smaller than ``max(abs_threshold * sum_gross, rel_threshold * prev_gross)``.

    Operates in dollar space (gross_target_usd) so it composes with cap projections.
    Total book gross is rescaled back to the original ``new_gross.sum()`` after snapping
    so the projection stack downstream is unchanged.
    """
    if prev_gross_by_pair is None or not prev_gross_by_pair:
        return new_gross
    g = np.asarray(new_gross, dtype=float).copy()
    s_total = float(np.sum(g))
    if s_total <= 1e-18:
        return new_gross
    abs_t = float(abs_threshold) * s_total if abs_threshold and abs_threshold > 0 else 0.0
    rel_t = float(rel_threshold) if rel_threshold and rel_threshold > 0 else 0.0
    etfs = sized["ETF"].astype(str).map(_norm_sym).to_numpy()
    unds = sized["Underlying"].astype(str).map(_norm_sym).to_numpy()
    snapped = g.copy()
    for i, (e, u) in enumerate(zip(etfs, unds)):
        prev = float(prev_gross_by_pair.get((str(e), str(u)), float("nan")))
        if not np.isfinite(prev) or prev <= 0:
            continue
        delta = abs(g[i] - prev)
        if delta <= abs_t:
            snapped[i] = prev
            continue
        if rel_t > 0 and delta <= rel_t * prev:
            snapped[i] = prev
    s_new = float(np.sum(snapped))
    if s_new > 1e-18:
        snapped *= s_total / s_new
    return snapped


def apply_gross_sizing_book_caps(
    sized: pd.DataFrame,
    *,
    target_gross_usd: float,
    delta_floor: float,
    strategy: dict[str, Any],
    shares_out_map: dict[str, float] | None = None,
    prev_gross_by_pair: dict[tuple[str, str], float] | None = None,
    cap_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Enforce book-level max pair / max underlying weights and liquidity-style per-pair caps
    (AUM, shares available, shares outstanding, median daily volume). Normally preserves total
    allocated gross; if constraints cannot place all mass (e.g. a tight underlying cap on a
    single-name book), total gross is scaled down.

    Config: ``strategy.gross_sizing_caps`` in YAML. Omitted or ``enabled: false`` → no-op.

    Liquidity-style rows use ``ref_short = T * short_leg_frac`` inside
    :func:`_liquidity_tight_book_weights`. Pick ``T`` via ``liquidity_book_reference``:
    ``target_book`` (default) anchors to YAML ``target_gross_usd`` passed in; ``deployed_book``
    anchors to the current SUM(``gross_target_usd``) so ladders match placed scale after shrinks.

    ``cap_mode`` selects the liquidity inputs that participate (see :func:`_resolve_cap_subset`).
    Default ``None`` keeps the legacy ``structural_plus_day_liquidity`` (what's executable today).
    Pass ``structural_only`` to compute the **optimal** target ignoring today's IBKR
    ``shares_available`` / ADV. The returned ``diag`` includes ``binding_per_row`` and
    ``cap_mode`` for downstream column emission.
    """
    diag: dict[str, Any] = {"applied": False}
    if sized is None or sized.empty:
        return sized, diag
    raw = strategy.get("gross_sizing_caps")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return sized, diag
    caps = dict(raw)
    und_cap = float(caps.get("max_underlying_weight_cap", 0.11) or 0.11)
    som = shares_out_map if shares_out_map is not None else {}
    cap_subset = _resolve_cap_subset(cap_mode)

    gross = pd.to_numeric(sized["gross_target_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gsum = float(gross.sum())
    if gsum <= 1e-18:
        return sized, diag

    liquidity_anchor_usd = _liquidity_book_anchor_usd(
        deployed_gross_sum=gsum,
        strategy_target_gross_usd=float(target_gross_usd),
        caps=caps,
    )
    liq_ref_raw = str(caps.get("liquidity_book_reference", "target_book") or "target_book").strip()

    per_sleeve_cfg = caps.get("per_sleeve") or caps.get("per_bucket")
    use_per_sleeve = (
        isinstance(per_sleeve_cfg, dict)
        and bool(per_sleeve_cfg)
        and ("sleeve" in sized.columns)
    )

    abs_t = float(caps.get("weight_hysteresis_abs", 0.0) or 0.0)
    rel_t = float(caps.get("weight_hysteresis_rel", 0.0) or 0.0)
    apply_hyst = bool(prev_gross_by_pair) and (abs_t > 0 or rel_t > 0)

    if use_per_sleeve:
        new_gross, ps_meta = _apply_gross_sizing_per_sleeve_book_caps(
            sized,
            gross,
            target_gross_usd=float(liquidity_anchor_usd),
            delta_floor=float(delta_floor),
            caps=caps,
            shares_out_map=som,
            cap_subset=cap_subset,
        )
        if apply_hyst:
            new_gross = _apply_weight_hysteresis(
                sized,
                new_gross,
                prev_gross_by_pair=prev_gross_by_pair,
                abs_threshold=abs_t,
                rel_threshold=rel_t,
            )
        out = sized.copy()
        out["gross_target_usd"] = new_gross
        diag.update(
            {
                "applied": True,
                "gross_sum_before": gsum,
                "gross_sum_after": float(new_gross.sum()),
                "max_pair_weight_cap": float(caps.get("max_pair_weight_cap", 0.05) or 0.05),
                "max_underlying_weight_cap": float(und_cap),
                "per_sleeve_enforced": True,
                "per_sleeve_caps": ps_meta.get("per_sleeve_caps", {}),
                "pre_cap_score_haircut": ps_meta.get("pre_cap_score_haircut", {}),
                "hysteresis_applied": bool(apply_hyst),
                "liquidity_book_anchor_usd": float(liquidity_anchor_usd),
                "liquidity_book_reference_raw": liq_ref_raw,
                "cap_mode": str(cap_mode or "structural_plus_day_liquidity"),
                "cap_subset": list(cap_subset),
                "binding_per_row": ps_meta.get("binding_per_row", np.full(len(sized), "none", dtype=object)),
            }
        )
        return out, diag

    pair_caps = _compute_pair_weight_caps_array(
        sized,
        target_gross_usd=float(liquidity_anchor_usd),
        delta_floor=float(delta_floor),
        caps=caps,
        shares_out_map=som,
        cap_subset=cap_subset,
    )
    desired = gross / gsum
    w1 = _project_to_capped_simplex_numpy(desired, pair_caps)
    und_code = pd.factorize(sized["Underlying"].astype(str).map(_norm_sym).to_numpy())[0]
    w2 = _project_pair_and_underlying_numpy(w1, pair_caps, und_code, float(und_cap))
    new_gross = w2 * gsum
    if apply_hyst:
        new_gross = _apply_weight_hysteresis(
            sized,
            new_gross,
            prev_gross_by_pair=prev_gross_by_pair,
            abs_threshold=abs_t,
            rel_threshold=rel_t,
        )
    out = sized.copy()
    out["gross_target_usd"] = new_gross
    # Best-effort binding label for the non-per-sleeve branch using liquidity-only categories.
    _, liq_binding_flat = _liquidity_tight_book_weights(
        sized,
        target_gross_usd=float(liquidity_anchor_usd),
        delta_floor=float(delta_floor),
        caps=caps,
        shares_out_map=som,
        cap_subset=cap_subset,
        return_binding_label=True,
    )
    diag.update(
        {
            "applied": True,
            "gross_sum_before": gsum,
            "gross_sum_after": float(new_gross.sum()),
            "max_pair_weight_cap": float(caps.get("max_pair_weight_cap", 0.05) or 0.05),
            "max_underlying_weight_cap": float(und_cap),
            "per_sleeve_enforced": False,
            "hysteresis_applied": bool(apply_hyst),
            "liquidity_book_anchor_usd": float(liquidity_anchor_usd),
            "liquidity_book_reference_raw": liq_ref_raw,
            "cap_mode": str(cap_mode or "structural_plus_day_liquidity"),
            "cap_subset": list(cap_subset),
            "binding_per_row": liq_binding_flat,
        }
    )
    return out, diag


def load_blacklist(cfg: dict) -> Set[str]:
    raw = cfg.get("strategy", {}).get("blacklist", []) or []
    return {_norm_sym(sym) for sym in raw if str(sym).strip()}


def get_borrow_col(df: pd.DataFrame) -> str | None:
    """
    Try to infer an annual borrow column from screened_csv.
    Accepts common names; returns column name or None.
    """
    candidates = [
        "borrow_current", "Borrow_Current", "borrow_fee_annual", "Borrow_Fee_Annual",
        "borrow_annual", "Borrow_annual", "borrow", "Borrow",
        "borrow_rate", "borrowRate", "feerate", "fee_annual"
    ]
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def compute_borrow_annual_series(df: pd.DataFrame, borrow_col: str | None) -> pd.Series:
    """
    Return annual borrow as decimal. If missing, returns NaN series.
    If the borrow column looks like percent (e.g. 3.63), you should
    normalize upstream; here we assume decimal already.
    """
    if borrow_col is None:
        return pd.Series(index=df.index, data=np.nan, dtype=float)
    s = pd.to_numeric(df[borrow_col], errors="coerce")
    return s.astype(float)


# -----------------------------
# Portfolio sizing logic
# -----------------------------
def hedge_ratio_from_beta(beta: float, delta_floor: float) -> float:
    b = float(beta) if np.isfinite(beta) else 1.0
    b_abs = max(abs(b), float(delta_floor))
    return 1.0 / b_abs


def size_pair_long_short(gross_usd: float, beta: float, delta_floor: float) -> Tuple[float, float]:
    """
    gross = long + |short| where short = -hedge_ratio * long
    """
    hr = hedge_ratio_from_beta(beta, delta_floor)
    long_usd = float(gross_usd) / (1.0 + hr)
    short_usd = -(hr * long_usd)
    return long_usd, short_usd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=today_str(), help="YYYY-MM-DD for data/runs/<date>/ outputs")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    core_decay_state_path = Path(
        paths.get("core_leveraged_decay_state_json", "data/core_leveraged_decay_state.json")
    )
    strategy = cfg["strategy"]
    sleeves = cfg.get("portfolio", {}).get("sleeves", {})

    screened_csv = Path(paths["screened_csv"])
    proposed_latest_csv = Path(paths["proposed_trades_csv"])

    tag = str(strategy.get("tag", "")).strip() or "strategy"
    blacklist = load_blacklist(cfg)

    # Key sizing params
    capital_usd = float(strategy["capital_usd"])
    gross_leverage = float(strategy["gross_leverage"])
    target_gross_usd = capital_usd * gross_leverage
    delta_floor = float(strategy.get("delta_floor", 0.1))

    # Sleeves
    core = sleeves.get("core_leveraged", {})
    b4 = sleeves.get("inverse_decay_bucket4", {})
    flow = sleeves.get("flow_program", {})
    yb_sleeve = sleeves.get("yieldboost", {}) or {}

    b4_w = float(b4.get("target_weight", 0.0))
    b4_enabled = bool(b4.get("enabled", True))
    yb_enabled = bool(yb_sleeve.get("enabled", True))
    core_stock_tw = float(core.get("target_weight", 1.0))
    yb_stock_tw = float(yb_sleeve.get("target_weight", 0.0))
    core_stock_frac, yb_stock_frac = _normalize_two_nonnegative_weights(
        core_stock_tw,
        yb_stock_tw if yb_enabled else 0.0,
    )
    stock_nominal_w = (
        max(0.0, min(1.0, 1.0 - b4_w)) if b4_enabled else 1.0
    )

    core_rules = core.get("rules", {}) or {}
    core_delta_min = float(core_rules.get("min_delta_used", 1.5))
    yb_rules = yb_sleeve.get("rules", {}) or {}
    _yb_edge_yaml = yb_rules.get("min_net_edge_annual", None)
    if _yb_edge_yaml is None or (isinstance(_yb_edge_yaml, float) and not np.isfinite(_yb_edge_yaml)):
        yb_min_edge = float(core_rules.get("yieldboost_min_net_edge_annual", 0.0) or 0.0)
    else:
        yb_min_edge = float(_yb_edge_yaml or 0.0)
    b4_rules = b4.get("rules", {}) or {}
    b4_min_edge = float(b4_rules.get("min_net_edge_annual", 0.0))
    b4_partial_hedge_ratio = _clamp01(b4_rules.get("partial_hedge_ratio", 1.0))
    b4_max_shares_outstanding_frac = _clamp01(b4_rules.get("max_shares_outstanding_frac", 0.20))
    # Underlying-vol floors: NEW pairs must clear ``min_underlying_vol_entry``;
    # pairs already tracked by the lifecycle state (i.e. held) may stay down to
    # ``min_underlying_vol_keep``. Legacy single-floor ``min_underlying_vol``
    # remains the fallback for both.
    _b4_vol_floor_legacy = float(b4_rules.get("min_underlying_vol", 0.50))
    b4_min_underlying_vol_entry = float(b4_rules.get("min_underlying_vol_entry", _b4_vol_floor_legacy))
    b4_min_underlying_vol_keep = float(b4_rules.get("min_underlying_vol_keep", _b4_vol_floor_legacy))
    b4_excluded_etfs = {_norm_sym(x) for x in (b4_rules.get("excluded_etfs") or [])}
    # Pair lifecycle (Phase 2 demotion ladder): half / freeze / exit from monitor flags.
    from scripts.bucket4_pair_lifecycle import (
        LifecycleConfig as _B4LifecycleConfig,
        apply_lifecycle_to_b4 as _b4_apply_lifecycle,
        held_etfs as _b4_held_etfs,
        load_state as _b4_load_lifecycle_state,
    )
    b4_lifecycle_cfg = _B4LifecycleConfig.from_rules(b4_rules)
    _lc_path = b4_lifecycle_cfg.state_json
    if not _lc_path.is_absolute():
        _lc_path = Path(__file__).resolve().parent / _lc_path
    # WS4 concentration + WS5 cluster caps (scripts/bucket4_sizing.py)
    from scripts.bucket4_sizing import (
        apply_cluster_caps_to_b4 as _b4_apply_cluster_caps,
        apply_concentration_to_b4 as _b4_apply_concentration,
    )
    _b4_conc_cfg = b4_rules.get("concentration") or {}
    b4_conc_enabled = bool(_b4_conc_cfg.get("enabled", False))
    b4_conc_top_n = int(_b4_conc_cfg.get("top_n_pairs", 0) or 0)
    b4_cluster_caps = b4_rules.get("cluster_caps") or {}
    b4_lifecycle_state = _b4_load_lifecycle_state(_lc_path) if b4_lifecycle_cfg.enabled else {}
    if b4_lifecycle_cfg.enabled:
        _lc_counts: dict[str, int] = {}
        for _v in b4_lifecycle_state.values():
            _s = str(_v.get("status", "normal"))
            _lc_counts[_s] = _lc_counts.get(_s, 0) + 1
        print(f"[INFO] B4 pair lifecycle ON: state={_lc_path} counts={_lc_counts or '{}'}")

    # Borrow entry caps come from the same per-bucket bands that define
    # purgatory keep thresholds in ``daily_screener``.
    _per_bucket = (cfg.get("screener", {}) or {}).get("per_bucket", {}) or {}
    b1_entry_borrow_cap = float(((_per_bucket.get("bucket_1") or {}).get("entry_borrow_cap", 1.0)))
    b2_entry_borrow_cap = float(((_per_bucket.get("bucket_2") or {}).get("entry_borrow_cap", b1_entry_borrow_cap)))
    b4_entry_borrow_cap = float(((_per_bucket.get("bucket_4") or {}).get("entry_borrow_cap", np.inf)))
    flow_borrow_cap = float(flow.get("rules", {}).get("hard_borrow_cap", np.inf))

    def fmt_cap(x: float) -> str:
        return f"{x:.1%}" if np.isfinite(x) else "inf"

    # Weighting configs (full dicts, consumed by _decay_score_weights)
    core_weighting_cfg = core.get("weighting", {})
    b4_weighting_cfg = b4.get("weighting", {})
    yb_weighting_cfg = yb_sleeve.get("weighting", {}) or {}
    core_weight_method = str(core_weighting_cfg.get("method", "equal")).lower()
    b4_weight_method = str(b4_weighting_cfg.get("method", "decay_score")).lower()
    yb_weight_method = str(yb_weighting_cfg.get("method", "decay_score")).lower()

    # Flow config
    flow_shorts = [_norm_sym(x) for x in (flow.get("universe", {}).get("shorts", []) or [])]

    shares_out_map, shares_src = load_shares_outstanding_map(paths)
    print(
        f"[INFO] target_gross_usd=${target_gross_usd:,.0f} | "
        f"post_b4_stock(nominal)={stock_nominal_w:.0%} "
        f"(core_frac={core_stock_frac:.1%} yb_frac={yb_stock_frac:.1%} of that) "
        f"b4={b4_w:.0%} (enabled={b4_enabled}) "
        f"| delta_floor={delta_floor}"
    )
    print(
        f"[INFO] borrow entry caps: b1={fmt_cap(b1_entry_borrow_cap)} "
        f"b2={fmt_cap(b2_entry_borrow_cap)} "
        f"b4={fmt_cap(b4_entry_borrow_cap)} | "
        f"flow hard cap={fmt_cap(flow_borrow_cap)}"
    )
    print(
        f"[INFO] b4 universe filters: min_underlying_vol entry={b4_min_underlying_vol_entry:.0%} keep={b4_min_underlying_vol_keep:.0%} | "
        f"min_net_edge_annual={b4_min_edge:.0%}"
    )
    if b4_excluded_etfs:
        print(f"[INFO] b4 excluded_etfs ({len(b4_excluded_etfs)}): {', '.join(sorted(b4_excluded_etfs))}")
    _core_nd_min = float(core_rules.get("min_net_decay_annual", 0.0) or 0.0)
    _core_hyst = core_rules.get("net_decay_hysteresis") or {}
    if _core_nd_min > 0 or bool(_core_hyst.get("enabled", False)):
        print(
            f"[INFO] core (bucket-1) net-decay selectivity: min_net_decay_annual={_core_nd_min:.2%} | "
            f"hysteresis_enabled={bool(_core_hyst.get('enabled', False))} | "
            f"state_file={core_decay_state_path}"
        )
        if bool(_core_hyst.get("enabled", False)):
            print(
                f"[INFO]   hysteresis enter_above={float(_core_hyst.get('enter_above', 0)):.2%} "
                f"exit_below={float(_core_hyst.get('exit_below', 0)):.2%}"
            )
    if not b4_enabled:
        print("[INFO] inverse_decay_bucket4 disabled — no B4 allocation or targets in proposed trades")
    print(f"[INFO] flow shorts={len(flow_shorts)}")
    if yb_min_edge > 0:
        print(f"[INFO] yieldboost sleeve min net_edge_p50_annual floor={yb_min_edge:.0%}")
    print(f"[INFO] weighting: core={core_weight_method} yieldboost={yb_weight_method} b4={b4_weight_method}")
    if shares_out_map:
        print(
            f"[INFO] bucket4 shares-outstanding cap enabled: "
            f"{b4_max_shares_outstanding_frac:.0%} of float "
            f"(source={shares_src})"
        )
    else:
        print("[WARN] bucket4 shares-outstanding source not found/usable; cap enforcement skipped.")

    if not screened_csv.exists():
        raise FileNotFoundError(f"Screened CSV not found: {screened_csv}")

    screened = pd.read_csv(screened_csv)
    if "Delta" not in screened.columns and "Beta" in screened.columns:
        screened["Delta"] = screened["Beta"]
    if screened.empty:
        print("[WARN] Screened universe is empty.")
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_latest_csv, index=False)
        return

    required_cols = {"ETF", "Underlying", "purgatory", "Delta"}
    missing = required_cols - set(screened.columns)
    if missing:
        raise ValueError(f"Screened CSV missing required columns: {sorted(missing)}. Found: {list(screened.columns)}")

    # Normalize tickers
    screened["ETF"] = screened["ETF"].astype(str).map(_norm_sym)
    screened["Underlying"] = screened["Underlying"].astype(str).map(_norm_sym)

    # Blacklist filter
    screened = screened[(~screened["Underlying"].isin(blacklist)) & (~screened["ETF"].isin(blacklist))].copy()
    if screened.empty:
        print("[WARN] No eligible pairs after blacklist filtering.")
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_latest_csv, index=False)
        return

    screened["Delta"] = pd.to_numeric(screened["Delta"], errors="coerce")
    screened["delta_abs"] = screened["Delta"].abs()

    # Coerce decay columns for weighting (may be absent in older CSVs)
    for _col in ("blended_gross_decay", "borrow_current", "net_decay_annual", "net_edge_p50_annual"):
        if _col not in screened.columns:
            screened[_col] = np.nan
        else:
            screened[_col] = pd.to_numeric(screened[_col], errors="coerce")

    # Borrow if available
    borrow_col = get_borrow_col(screened)
    screened["borrow_annual"] = compute_borrow_annual_series(screened, borrow_col)
    # if borrow missing, borrow_annual is NaN, and caps won't filter (we treat NaN as "unknown -> allow")
    # if you prefer "unknown -> exclude", change the predicate below accordingly.

    # KEEP set: full screened rows. Final output is filtered to
    # non-zero targets plus purgatory rows.
    keep = screened.copy()

    # Initialize outputs on KEEP
    keep["strategy_tag"] = tag
    keep["long_usd"] = 0.0
    keep["short_usd"] = 0.0
    keep["underlying_target_usd"] = 0.0
    keep["etf_target_usd"] = 0.0
    keep["underlying_target_from_b12_usd"] = 0.0
    keep["underlying_target_from_b4_usd"] = 0.0
    keep["underlying_internalized_usd"] = 0.0
    keep["underlying_external_trade_usd"] = 0.0
    keep["sleeve"] = ""
    keep["b1_trend_xsec_pctile"] = np.nan
    keep["b1_trend_multiplier"] = np.nan

    # SIZING set: not purgatory (borrow + net-edge soft band in daily_screener) and
    # net_edge_p50_annual >= 0 when present (NaN treated as ineligible).
    if "net_edge_p50_annual" in screened.columns:
        screened["net_edge_p50_annual"] = pd.to_numeric(
            screened["net_edge_p50_annual"], errors="coerce"
        )
        _ne_ok = screened["net_edge_p50_annual"].fillna(-1.0).ge(0.0)
    else:
        _ne_ok = pd.Series(True, index=screened.index)
    eligible = screened.loc[(screened["purgatory"] != True) & _ne_ok].copy()  # noqa: E712
    if eligible.empty:
        print("[WARN] No eligible rows to size (all rows are purgatory).")
        proposed = keep.copy()
        nonzero_mask = (proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)
        proposed = proposed[nonzero_mask | (proposed["purgatory"] == True)]  # noqa: E712

        cols_to_drop = ["Leverage", "ExpectedLeverage", "cagr_positive", "delta_abs"]
        proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns], errors="ignore")

        dated_path = run_dir(args.run_date) / "proposed_trades.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(dated_path, index=False)
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(proposed_latest_csv, index=False)
        print(f"[OK] Wrote proposed trades -> {dated_path}  (n={len(proposed)})")
        print(f"[OK] Updated latest proposed trades -> {proposed_latest_csv}  (n={len(proposed)})")
    else:
        # -----------------------------
        # Build sleeve membership
        # -----------------------------
        # core_leveraged: high-delta bucket-1 path only (~not ``is_yieldboost``).
        # yieldboost: ``is_yieldboost`` rows that pass borrow + min_net_edge on net_edge_p50.
        flow_program_etfs = {_norm_sym(x) for x in (flow_shorts or [])}
        is_yieldboost, in_b2_universe, in_flow_program = _b2_b4_universe_masks(
            eligible, flow_program_etfs=flow_program_etfs
        )
        nd_annual = pd.to_numeric(eligible["net_decay_annual"], errors="coerce")
        neg_net_decay = nd_annual < 0

        b = eligible["borrow_annual"]
        core_borrow_ok = (~np.isfinite(b)) | (b <= b1_entry_borrow_cap)
        yb_borrow_ok = (~np.isfinite(b)) | (b <= b2_entry_borrow_cap)
        b4_borrow_ok = (~np.isfinite(b)) | (b <= b4_entry_borrow_cap)

        # Exclude inverse (β < 0) ETFs — they belong to Bucket 4 / flow, not the stock sleeve
        positive_beta = eligible["Delta"].gt(0)
        negative_beta = eligible["Delta"].lt(0)
        if "inverse_shortable" in eligible.columns:
            inverse_shortable = eligible["inverse_shortable"].fillna(False).astype(bool)
        else:
            inverse_shortable = negative_beta
        edge_col = _b4_eligibility_edge_column(eligible)
        b4_edge = pd.to_numeric(eligible.get(edge_col), errors="coerce")
        b4_edge_ok = (~np.isfinite(b4_edge)) | (b4_edge >= b4_min_edge)
        ne_p50 = pd.to_numeric(eligible.get("net_edge_p50_annual"), errors="coerce")
        yieldboost_edge_ok = (
            np.isfinite(ne_p50) & (ne_p50 >= yb_min_edge)
            if yb_min_edge > 0
            else pd.Series(True, index=eligible.index)
        )
        b4_und_vol = pd.to_numeric(eligible.get("vol_underlying_annual"), errors="coerce")
        # Require realized underlying vol above the configured floor. Missing vol is treated as
        # a rejection: we will not admit a pair whose underlying-vol we cannot measure.
        # Held pairs (present in the lifecycle state) use the lower keep-side floor so
        # they roll off gradually instead of being cleaned up the day vol dips.
        _b4_held_set = _b4_held_etfs(b4_lifecycle_state) if b4_lifecycle_cfg.enabled else set()
        _b4_is_held = eligible["ETF"].astype(str).map(_norm_sym).isin(_b4_held_set)
        _b4_vol_floor_row = np.where(_b4_is_held, b4_min_underlying_vol_keep, b4_min_underlying_vol_entry)
        b4_vol_ok = np.isfinite(b4_und_vol) & (b4_und_vol >= _b4_vol_floor_row)
        core_pre_decay = (
            positive_beta
            & eligible["delta_abs"].ge(core_delta_min)
            & core_borrow_ok
        )
        core_neg_decay_reset = (
            positive_beta
            & eligible["delta_abs"].ge(core_delta_min)
            & core_borrow_ok
            & neg_net_decay
        )
        try:
            core_decay_gate = _core_net_decay_gate_for_core(
                eligible,
                core_pre_decay=core_pre_decay,
                core_neg_decay_reset=core_neg_decay_reset,
                core_rules=core_rules,
                state_path=core_decay_state_path,
                run_date=args.run_date,
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
        eligible["in_core"] = core_pre_decay & core_decay_gate & ~in_b2_universe

        in_yieldboost_stock = (
            positive_beta
            & in_b2_universe
            & ~in_flow_program
            & yb_borrow_ok
            & yieldboost_edge_ok
        )

        n_b2_yb_rows = int(in_b2_universe.sum())
        n_b2_flow_excluded = int((positive_beta & in_b2_universe & in_flow_program).sum())
        b4_not_excluded = ~eligible["ETF"].isin(b4_excluded_etfs)
        n_b4_flow_excluded = int(
            (
                negative_beta
                & inverse_shortable
                & b4_borrow_ok
                & b4_edge_ok
                & b4_vol_ok
                & b4_not_excluded
                & in_flow_program
            ).sum()
        )
        if n_b2_flow_excluded or n_b4_flow_excluded:
            print(
                f"[INFO] B2 YieldBoost ticker rows≈{n_b2_yb_rows} | "
                f"B2↔flow overlap (excluded from stock sleeve)={n_b2_flow_excluded}, "
                f"B4↔flow overlap={n_b4_flow_excluded}"
            )
        is_volatility_etp = _volatility_etp_rows_mask(eligible)
        eligible["in_b4"] = (
            negative_beta
            & inverse_shortable
            & b4_borrow_ok
            & b4_edge_ok
            & b4_vol_ok
            & b4_not_excluded
            & ~in_flow_program
            & ~is_volatility_etp
        )
        in_b4_volatility_etp = _in_b4_volatility_etp_sleeve_mask(
            eligible,
            b4_borrow_ok=b4_borrow_ok,
            b4_edge_ok=b4_edge_ok,
            b4_vol_ok=b4_vol_ok,
            b4_not_excluded=b4_not_excluded,
            in_flow_program=in_flow_program,
            in_b4_core=eligible["in_b4"],
        )

        core_names = eligible.loc[eligible["in_core"]].copy()
        yb_names = eligible.loc[in_yieldboost_stock].copy()
        if not yb_enabled:
            yb_names = eligible.loc[[]].copy()
        b4_core_names = eligible.loc[eligible["in_b4"]].copy()
        b4_vol_names = eligible.loc[in_b4_volatility_etp].copy()
        if not b4_enabled:
            b4_core_names = eligible.loc[[]].copy()
            b4_vol_names = eligible.loc[[]].copy()
        n_core_decay_blocked = int((core_pre_decay & ~core_decay_gate).sum())
        if n_core_decay_blocked:
            print(
                f"[INFO] Excluded {n_core_decay_blocked} core (bucket-1) candidate row(s) "
                f"via min_net_decay_annual / hysteresis gate."
            )
        n_yb_edge_blocked = int((positive_beta & in_b2_universe & ~in_flow_program & ~yieldboost_edge_ok).sum())
        if n_yb_edge_blocked:
            print(
                f"[INFO] Excluded {n_yb_edge_blocked} yieldboost sleeve candidate row(s) "
                f"below min_net_edge_annual={yb_min_edge:.2%}."
            )

        # Budgeting:
        # - Bucket 4 gets ``target_weight`` of total gross when core inverse rows are active.
        # - Volatility ETPs are a separate bucket-5 sleeve with a fixed share of total gross.
        # - Bucket 5 is subtracted from the B4 allocation so the combined inverse-vol budget
        #   stays at the configured B4 target weight when both slices exist.
        vol_etp_b4_cfg = (
            b4_rules.get("volatility_etp_bucket5")
            or b4_rules.get("volatility_etp_bucket4")
            or {}
        )
        vol_etp_b4_enabled = bool(vol_etp_b4_cfg.get("enabled", False))
        if "target_weight" in vol_etp_b4_cfg:
            vol_etp_book_weight = _clamp01(float(vol_etp_b4_cfg.get("target_weight", 0.0) or 0.0))
        else:
            vol_etp_book_weight = _clamp01(
                b4_w * float(vol_etp_b4_cfg.get("share_of_b4_budget", 0.0) or 0.0)
            )

        b4_any = bool(
            b4_enabled
            and b4_w > 0
            and (not b4_core_names.empty or not b4_vol_names.empty)
        )
        b4_budget_total = min(target_gross_usd * b4_w, target_gross_usd) if b4_any else 0.0

        b4_vol_cash = 0.0
        if (
            b4_budget_total > 1e-12
            and vol_etp_b4_enabled
            and vol_etp_book_weight > 0.0
            and not b4_vol_names.empty
        ):
            b4_vol_cash = min(b4_budget_total, target_gross_usd * vol_etp_book_weight)
        b4_core_cash = max(0.0, b4_budget_total - b4_vol_cash)

        b4_reserved = 0.0
        if not b4_core_names.empty:
            b4_reserved += b4_core_cash
        if not b4_vol_names.empty:
            b4_reserved += b4_vol_cash

        b4_core_reserved = b4_core_cash if not b4_core_names.empty else 0.0
        b4_vol_reserved = b4_vol_cash if not b4_vol_names.empty else 0.0

        remainder_budget = max(0.0, target_gross_usd - b4_reserved)
        core_budget = remainder_budget * core_stock_frac
        yb_budget = remainder_budget * yb_stock_frac
        if core_budget > 1e-9 and core_names.empty:
            yb_budget += core_budget
            core_budget = 0.0
        if yb_budget > 1e-9 and yb_names.empty:
            core_budget += yb_budget
            yb_budget = 0.0

        core_names_fit = core_names.copy()
        yb_names_fit = yb_names.copy()

        # -----------------------------
        # Allocate CORE stock sleeve (`core_leveraged`)
        # -----------------------------
        if not core_names_fit.empty and core_budget > 0:
            if core_weight_method == "decay_score":
                w = _decay_score_weights(core_names_fit, core_weighting_cfg, sleeve_name="core_leveraged")
            else:   # "equal" or unrecognised → equal weight
                w = np.ones(len(core_names_fit)) / len(core_names_fit)
            core_names_fit["gross_target_usd"] = core_budget * w
            core_names_fit["sleeve"] = "core_leveraged"
        # Frozen pre-cap sleeve-internal weight (sums to 1) — used by the gross-cap stack as
        # the haircut baseline so covariance + sleeve rebalance can't widen the per-row cap.
            core_names_fit["_pre_cap_score_weight"] = w
            core_names_fit = _with_b1_trend_audit_columns(core_names_fit, core_weighting_cfg)
        elif not core_names_fit.empty:
            core_names_fit["gross_target_usd"] = 0.0
            core_names_fit["sleeve"] = "core_leveraged"
            core_names_fit["_pre_cap_score_weight"] = 0.0
            core_names_fit["b1_trend_xsec_pctile"] = np.nan
            core_names_fit["b1_trend_multiplier"] = np.nan

        # -----------------------------
        # Allocate YIELDBOOST sleeve (bucket‑2 candidates only)
        # -----------------------------
        if not yb_names_fit.empty and yb_budget > 0:
            if yb_weight_method == "decay_score":
                w = _decay_score_weights(yb_names_fit, yb_weighting_cfg, sleeve_name="yieldboost")
            else:
                w = np.ones(len(yb_names_fit)) / len(yb_names_fit)
            yb_names_fit["gross_target_usd"] = yb_budget * w
            yb_names_fit["sleeve"] = "yieldboost"
            yb_names_fit["_pre_cap_score_weight"] = w
        elif not yb_names_fit.empty:
            yb_names_fit["gross_target_usd"] = 0.0
            yb_names_fit["sleeve"] = "yieldboost"
            yb_names_fit["_pre_cap_score_weight"] = 0.0

        stock_frames = []
        if not core_names_fit.empty:
            stock_frames.append(core_names_fit)
        if not yb_names_fit.empty:
            stock_frames.append(yb_names_fit)
        stock_names = pd.concat(stock_frames, axis=0, ignore_index=False) if stock_frames else eligible.loc[[]].copy()

        # -----------------------------
        # Allocate BUCKET 4 (core inverse β<0 + optional volatility-ETP slice)
        # -----------------------------
        b4_parts: list[pd.DataFrame] = []

        if not b4_core_names.empty and b4_core_cash > 1e-9:
            b4c = b4_core_names.copy()
            b4c["_b4_slice"] = "core"
            if b4_weight_method == "equal":
                w = np.ones(len(b4c)) / len(b4c)
            else:
                w = _decay_score_weights(b4c, b4_weighting_cfg, sleeve_name="inverse_decay_bucket4")
            if b4_lifecycle_cfg.enabled:
                b4c, w, _lc_info = _b4_apply_lifecycle(b4c, w, b4_lifecycle_state, b4_lifecycle_cfg)
                if any(_lc_info.values()):
                    print(
                        f"[INFO] B4 lifecycle ladder: exit={_lc_info['n_exit']} "
                        f"freeze={_lc_info['n_freeze']} half={_lc_info['n_half']} "
                        f"({len(b4c)} pairs remain in core slice)"
                    )
            if b4_conc_enabled and b4_conc_top_n > 0:
                _held = _b4_held_etfs(b4_lifecycle_state) if b4_lifecycle_cfg.enabled else set()
                b4c, w, _conc_info = _b4_apply_concentration(b4c, w, top_n=b4_conc_top_n, held=_held)
                if any(_conc_info.values()):
                    print(
                        f"[INFO] B4 concentration top-{b4_conc_top_n}: "
                        f"dropped={_conc_info['n_dropped']} keep_open={_conc_info['n_keep_open']} "
                        f"({len(b4c)} pairs remain)"
                    )
            if b4_cluster_caps and len(b4c) > 0:
                w, _cl_info = _b4_apply_cluster_caps(b4c, w, b4_cluster_caps)
                for _cname, _cinfo in _cl_info.items():
                    if _cinfo.get("capped"):
                        print(
                            f"[INFO] B4 cluster cap '{_cname}': weight "
                            f"{_cinfo['weight']:.1%} -> capped at {_cinfo['cap']:.0%}"
                        )
            b4c["gross_target_usd"] = b4_core_cash * w
            # ``_pre_cap_score_weight`` here is the b4-core-slice-internal weight; combined-sleeve
            # baseline is normalized below in the ``b4_names`` concat once all slices are merged.
            b4c["_pre_cap_score_weight"] = w
            b4_opt2 = b4_rules.get("bucket4_weekly_opt2") or {}
            if bool(b4_opt2.get("enabled")):
                try:
                    from scripts.bucket4_weekly_opt2 import (
                        Bucket4WeeklyConfig,
                        build_bucket4_state,
                        compute_bucket4_targets,
                        compute_bucket4_weights,
                    )
                    from scripts.v6_b4_pf_weights import V6PfParams

                    pairs_subset = [
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"])))
                        for _, r in b4c[["ETF", "Underlying"]].iterrows()
                    ]
                    excl_inv = frozenset({"SCO"} | {_norm_sym(x) for x in (b4_opt2.get("excluded_inverse_etfs") or [])})
                    mp = int(b4_opt2.get("pf_min_pairs", 5))
                    mp = min(mp, max(1, len(pairs_subset)))
                    _hcp = b4_opt2.get("hedge_cadence_policy") or {}
                    _hedge_source = str(_hcp.get("source", "v6_panel"))
                    cfg_b4 = Bucket4WeeklyConfig(
                        screened_csv=str(screened_csv),
                        start=str(b4_opt2.get("history_start", "2018-01-01")),
                        end=str(args.run_date),
                        weekly_rebalance_freq=str(b4_opt2.get("weekly_rebalance_freq", "W-FRI")),
                        warmup_bdays=int(b4_opt2.get("warmup_bdays", 65)),
                        fee_bps=float(b4_opt2.get("fee_bps", 1.0)),
                        slippage_bps=float(b4_opt2.get("slippage_bps", 20.0)),
                        borrow_multiplier=float(b4_opt2.get("borrow_multiplier", 1.0)),
                        excluded_inverse_etfs=excl_inv,
                        min_underlying_vol=float(b4_opt2.get("min_underlying_vol", b4_min_underlying_vol_entry)),
                        min_net_decay=float(b4_opt2.get("min_net_decay", b4_min_edge)),
                        use_ibkr_uvix_borrow=bool(b4_opt2.get("use_ibkr_uvix_borrow", False)),
                        pf_params=V6PfParams(
                            min_pairs=mp,
                            vol_etp_weight_penalty=float(
                                b4_opt2.get("vol_etp_weight_penalty", 0.0)
                            ),
                        ),
                        hedge_source=_hedge_source,
                        hedge_cadence_policy=_hcp,
                    )
                    st_b4 = build_bucket4_state(cfg_b4, bucket4_pairs=pairs_subset)
                    pw, _, _ = compute_bucket4_weights(st_b4)
                    # Grow-only ratchet: floor inverse leg at persisted per-pair state.
                    _rcfg = b4_rules.get("ratchet") or {}
                    _ratchet_on = bool(_rcfg.get("enabled"))
                    _ratchet_path = _b4_ratchet_state_path(b4_rules)
                    _ratchet_state = _b4_load_ratchet_state(_ratchet_path) if _ratchet_on else {}
                    _ratchet_floor = {
                        (e, u): float(_ratchet_state[f"{e}|{u}"])
                        for (e, u) in pairs_subset if f"{e}|{u}" in _ratchet_state
                    }
                    tgt_df, _ = compute_bucket4_targets(
                        st_b4,
                        pw,
                        args.run_date,
                        float(b4_core_cash),
                        fee_bps=float(b4_opt2.get("fee_bps", 1.0)),
                        slippage_bps=float(b4_opt2.get("slippage_bps", 20.0)),
                        partial_hedge_ratio=b4_partial_hedge_ratio,
                        delta_floor=delta_floor,
                        ratchet_enabled=_ratchet_on,
                        ratchet_floor_by_pair=_ratchet_floor,
                    )
                    # Emit human-readable cadence + hedge-ratio explainability and plots.
                    _emit_b4_cadence_outputs(st_b4, tgt_df, args.run_date)
                    # Persist the grow-only floor (max of prior and this run's inverse target).
                    if _ratchet_on:
                        _new_state = dict(_ratchet_state)
                        for _, r in tgt_df.iterrows():
                            _k = f"{_norm_sym(str(r['ETF']))}|{_norm_sym(str(r['Underlying']))}"
                            _inv = float(r.get("inverse_etf_short_usd", 0.0) or 0.0)
                            _new_state[_k] = max(float(_new_state.get(_k, 0.0)), _inv)
                        _b4_write_ratchet_state(_ratchet_path, _new_state, args.run_date)
                    gross_by_key = {
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"]))): float(r["gross_target_usd"])
                        for _, r in tgt_df.iterrows()
                        if pd.notna(r.get("gross_target_usd"))
                    }
                    inv_short_by_key = {
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"]))): float(r["inverse_etf_short_usd"])
                        for _, r in tgt_df.iterrows()
                        if pd.notna(r.get("inverse_etf_short_usd"))
                    }
                    und_short_by_key = {
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"]))): float(r["underlying_short_usd"])
                        for _, r in tgt_df.iterrows()
                        if pd.notna(r.get("underlying_short_usd"))
                    }
                    hedge_by_key = {
                        (_norm_sym(str(r["ETF"])), _norm_sym(str(r["Underlying"]))): float(r["hedge_ratio"])
                        for _, r in tgt_df.iterrows()
                        if pd.notna(r.get("hedge_ratio"))
                    }
                    for idx in b4c.index:
                        k = (
                            _norm_sym(str(b4c.at[idx, "ETF"])),
                            _norm_sym(str(b4c.at[idx, "Underlying"])),
                        )
                        if k in gross_by_key:
                            b4c.at[idx, "gross_target_usd"] = gross_by_key[k]
                        if k in inv_short_by_key:
                            b4c.at[idx, "b4_opt2_inverse_etf_short_usd"] = inv_short_by_key[k]
                        if k in und_short_by_key:
                            b4c.at[idx, "b4_opt2_underlying_short_usd"] = und_short_by_key[k]
                        if k in hedge_by_key:
                            b4c.at[idx, "b4_opt2_hedge_ratio"] = hedge_by_key[k]
                    print(f"[INFO] bucket4_weekly_opt2: tail-risk weights + dynamic hedge targets (n={len(tgt_df)})")
                except Exception as e:
                    print(f"[WARN] bucket4_weekly_opt2 disabled for this run ({e}); using decay_score sizing")
            b4_parts.append(b4c)

        if not b4_vol_names.empty and b4_vol_cash > 1e-9:
            b4v = b4_vol_names.copy()
            b4v["_b4_slice"] = "vol_etp"
            if b4_weight_method == "equal":
                wv = np.ones(len(b4v)) / len(b4v)
            else:
                wv = _decay_score_weights(b4v, b4_weighting_cfg, sleeve_name="inverse_decay_bucket4")
            b4v["gross_target_usd"] = b4_vol_cash * wv
            b4v["_pre_cap_score_weight"] = wv
            b4_parts.append(b4v)

        if b4_parts:
            b4_names = pd.concat(b4_parts, axis=0, ignore_index=False)
            # Re-normalize pre-cap weights across the combined inverse-vol allocation so
            # the haircut baseline matches the merged allocation seen by the cap stack.
            if "_pre_cap_score_weight" in b4_names.columns:
                _b4_g = pd.to_numeric(b4_names["gross_target_usd"], errors="coerce").fillna(0.0)
                _b4_s = float(_b4_g.sum())
                if _b4_s > 1e-18:
                    b4_names["_pre_cap_score_weight"] = (_b4_g / _b4_s).to_numpy()
                else:
                    b4_names["_pre_cap_score_weight"] = 0.0
            # Cap bucket-4 ETF notionals by shares outstanding and reference price (combined sleeve).
            b4_names["shares_outstanding_total"] = b4_names["ETF"].map(shares_out_map)
            b4_names["price_ref"] = pd.to_numeric(
                b4_names.get("borrow_price_ref", np.nan), errors="coerce"
            )
            b4_names["gross_target_cap_usd"] = (
                b4_max_shares_outstanding_frac
                * pd.to_numeric(b4_names["shares_outstanding_total"], errors="coerce")
                * pd.to_numeric(b4_names["price_ref"], errors="coerce")
            )
            if shares_out_map:
                before_sum = float(pd.to_numeric(b4_names["gross_target_usd"], errors="coerce").fillna(0.0).sum())
                b4_names["gross_target_usd"] = _apply_notional_caps_with_redistribution(
                    b4_names["gross_target_usd"],
                    b4_names["gross_target_cap_usd"],
                )
                after_sum = float(pd.to_numeric(b4_names["gross_target_usd"], errors="coerce").fillna(0.0).sum())
                if after_sum + 1e-6 < before_sum:
                    print(
                        "[WARN] bucket4 shares-outstanding caps constrained allocated notional: "
                        f"requested=${before_sum:,.0f}, capped=${after_sum:,.0f}"
                    )
            b4_names["sleeve"] = np.where(
                b4_names["_b4_slice"].eq("vol_etp"),
                VOL_ETP_BUCKET5_SLEEVE,
                "inverse_decay_bucket4",
            )

            if b4_weight_method == "decay_score":
                b4c_w = b4_names.loc[b4_names["_b4_slice"].eq("core")]
                b4v_w = b4_names.loc[b4_names["_b4_slice"].eq("vol_etp")]
                if not b4c_w.empty and b4_core_cash > 1e-9:
                    cw_b4 = b4c_w["gross_target_usd"] / b4_core_cash
                    print(
                        f"[INFO] b4 (core inverse β<0) weights: max={cw_b4.max():.3f} min={cw_b4.min():.3f} "
                        f"nonzero={int((cw_b4 > 1e-9).sum())}/{len(b4c_w)}"
                    )
                if not b4v_w.empty and b4_vol_cash > 1e-9:
                    vw_b4 = b4v_w["gross_target_usd"] / b4_vol_cash
                    print(
                        f"[INFO] b4 (volatility ETP) weights: max={vw_b4.max():.3f} min={vw_b4.min():.3f} "
                        f"nonzero={int((vw_b4 > 1e-9).sum())}/{len(b4v_w)}"
                    )

            b4_names = b4_names.drop(columns=["_b4_slice"])
        else:
            b4_names = eligible.loc[[]].copy()

        sized = pd.concat([stock_names, b4_names], axis=0, ignore_index=False)
        sized = sized[~sized.index.duplicated(keep="first")].copy()

        # Snapshot the post-decay-score / pre-cap-stack frame so we can later run a parallel
        # **structural-only** pipeline for the optimal end-state target (independent of today's
        # IBKR shares_available + ADV constraints). See ``_run_book_cap_pipeline_quiet``.
        sized_pre_caps = sized.copy()

        sized, _cap_diag = apply_gross_sizing_book_caps(
            sized,
            target_gross_usd=float(target_gross_usd),
            delta_floor=float(delta_floor),
            strategy=strategy,
            shares_out_map=shares_out_map,
        )
        executable_binding = _cap_diag.get("binding_per_row") if isinstance(_cap_diag, dict) else None
        if _cap_diag.get("applied"):
            if _cap_diag.get("per_sleeve_enforced"):
                _ps = _cap_diag.get("per_sleeve_caps") or {}
                _hc = _cap_diag.get("pre_cap_score_haircut") or {}
                print(
                    "[INFO] gross_sizing_caps (per-sleeve, DCQ-style): "
                    f"sleeves={list(_ps.keys())} "
                    f"gross_before=${_cap_diag.get('gross_sum_before', 0):,.0f} "
                    f"gross_after=${_cap_diag.get('gross_sum_after', 0):,.0f}"
                )
                if _hc:
                    bits = ", ".join(
                        f"{k}: x{v.get('pre_cap_score_haircut_multiplier'):.2f}"
                        f" (rows_capped={v.get('n_rows_capped_by_haircut', 0)})"
                        for k, v in _hc.items()
                    )
                    print(f"[INFO] pre_cap_score_haircut active per sleeve: {bits}")
            else:
                print(
                    "[INFO] gross_sizing_caps: "
                    f"pair_cap={_cap_diag.get('max_pair_weight_cap'):.1%} "
                    f"under_cap={_cap_diag.get('max_underlying_weight_cap'):.1%} "
                    f"gross_before=${_cap_diag.get('gross_sum_before', 0):,.0f} "
                    f"gross_after=${_cap_diag.get('gross_sum_after', 0):,.0f}"
                )

        cov_cfg = strategy.get("covariance_balance") or {}
        if isinstance(cov_cfg, dict) and bool(cov_cfg.get("enabled", False)):
            ur_csv = paths.get("underlying_returns_csv") if isinstance(paths, dict) else None
            ur_df: pd.DataFrame | None = None
            ur_load_msg: str | None = None
            if not isinstance(ur_csv, str) or not str(ur_csv).strip():
                ur_load_msg = "paths.underlying_returns_csv is empty — covariance_balance needs a wide CSV (dates × underlyings)"
            else:
                ur_path = Path(ur_csv)
                if not ur_path.is_file():
                    ur_load_msg = f"underlying returns file not found: {ur_path}"
                else:
                    try:
                        ur_df = pd.read_csv(ur_path, index_col=0, parse_dates=True)
                        ur_df.index = pd.to_datetime(ur_df.index, utc=False)
                        if ur_df.empty:
                            ur_df = None
                            ur_load_msg = f"underlying returns file is empty: {ur_path}"
                    except Exception as ex:
                        ur_df = None
                        ur_load_msg = f"failed to read {ur_path}: {ex}"
            per_scopes_raw = cov_cfg.get("per_sleeve_scopes")
            per_scopes = (
                [str(s).strip() for s in per_scopes_raw if str(s).strip()]
                if isinstance(per_scopes_raw, list)
                else []
            )

            def _log_cov_applied(d: dict) -> None:
                top = ", ".join(f"{u}(x{m:.2f})" for u, m in (d.get("top_attenuated_underlyings") or []))
                sc_l = d.get("covariance_scope") or []
                sc_s = ",".join(sc_l) if sc_l else "book"
                print(
                    f"[INFO] covariance_balance[{sc_s}]: shrink={d.get('shrink'):.2f} "
                    f"lambda={d.get('penalty_strength'):.2f} "
                    f"n_und={d.get('n_underlyings')} obs={d.get('obs_used')} "
                    f"rows_scope={d.get('n_rows_scope')} "
                    f"gross_scope=${d.get('gross_sum_scope_before', 0):,.0f}->{d.get('gross_sum_scope_after_caps', 0):,.0f} "
                    f"attenuated=[{top}] book_gross_after=${d.get('gross_sum_after', 0):,.0f}"
                )

            if ur_df is None:
                if ur_load_msg:
                    print(f"[WARN] covariance_balance: {ur_load_msg}")
            elif per_scopes:
                cov_last_diag: dict[str, Any] = {}
                printed_rs_detail = False
                matched_any_scope = False
                for sc in per_scopes:
                    if str(sc) not in set(sized["sleeve"].astype(str).unique()):
                        continue
                    matched_any_scope = True
                    sized, cov_last_diag = apply_covariance_balance(
                        sized,
                        target_gross_usd=float(target_gross_usd),
                        delta_floor=float(delta_floor),
                        strategy=strategy,
                        paths=paths,
                        shares_out_map=shares_out_map,
                        returns_df=ur_df,
                        covariance_scope=[sc],
                    )
                    if cov_last_diag.get("applied"):
                        _log_cov_applied(cov_last_diag)
                    elif cov_last_diag.get("returns_skip_detail"):
                        if not printed_rs_detail:
                            print(
                                f"[WARN] covariance_balance: insufficient_returns — "
                                f"{cov_last_diag['returns_skip_detail']}"
                            )
                            printed_rs_detail = True
                    else:
                        print(
                            f"[INFO] covariance_balance[{sc}] skipped ({cov_last_diag.get('reason', '')})"
                        )
                if not matched_any_scope:
                    print("[INFO] covariance_balance: no per_sleeve_scopes matched sized rows — skipped")
            else:
                sized, _cov_diag = apply_covariance_balance(
                    sized,
                    target_gross_usd=float(target_gross_usd),
                    delta_floor=float(delta_floor),
                    strategy=strategy,
                    paths=paths,
                    shares_out_map=shares_out_map,
                    returns_df=ur_df,
                )
                if _cov_diag.get("applied"):
                    _log_cov_applied(_cov_diag)
                else:
                    if _cov_diag.get("returns_skip_detail"):
                        print(
                            f"[WARN] covariance_balance: insufficient_returns — "
                            f"{_cov_diag['returns_skip_detail']}"
                        )
                    else:
                        print(f"[INFO] covariance_balance: skipped ({_cov_diag.get('reason', 'disabled')})")

        gs_reb_raw = strategy.get("gross_sizing_caps") or {}
        if (
            isinstance(gs_reb_raw, dict)
            and bool(gs_reb_raw.get("enabled", False))
            and bool(gs_reb_raw.get("rebalance_sleeve_weights_to_budget", False))
        ):
            rescale_to = str(gs_reb_raw.get("rebalance_sleeve_target_mode", "absolute_budget")).strip().lower()
            sleeve_targets = {
                "core_leveraged": float(core_budget),
                "yieldboost": float(yb_budget),
                "inverse_decay_bucket4": float(b4_core_reserved),
                VOL_ETP_BUCKET5_SLEEVE: float(b4_vol_reserved),
            }
            sized, sleeve_reb_diag = rescale_gross_targets_to_sleeve_budget_weights(
                sized,
                target_gross_usd=float(target_gross_usd),
                sleeve_budget_usd=sleeve_targets,
                rescale_to=rescale_to,
            )
            if sleeve_reb_diag.get("applied"):
                print(
                    f"[INFO] sleeve_budget_rescale (mode={sleeve_reb_diag.get('rescale_to')}) "
                    f"vs YAML target_gross_usd: "
                    f"gross=${sleeve_reb_diag.get('gross_sum_before', 0):,.0f}→${sleeve_reb_diag.get('gross_sum_after', 0):,.0f}; "
                    f"fraction_before={sleeve_reb_diag.get('sleeve_fractions_before_rescale')} "
                    f"fraction_after={sleeve_reb_diag.get('sleeve_fractions_after_rescale')}"
                )
                sk = sleeve_reb_diag.get("skipped_zero_deployed") or []
                if sk:
                    print(
                        "[WARN] sleeve_budget_rescale: skipped sleeves with zero deployed gross "
                        f"(nonzero YAML budget fraction): {sk}"
                    )
            sized, _final_cap_diag = apply_gross_sizing_book_caps(
                sized,
                target_gross_usd=float(target_gross_usd),
                delta_floor=float(delta_floor),
                strategy=strategy,
                shares_out_map=shares_out_map,
            )
            if isinstance(_final_cap_diag, dict) and _final_cap_diag.get("binding_per_row") is not None:
                executable_binding = _final_cap_diag.get("binding_per_row")
            if _final_cap_diag.get("applied"):
                print(
                    "[INFO] gross_sizing_caps (final pass after sleeve rebalance): "
                    f"gross_before=${_final_cap_diag.get('gross_sum_before', 0):,.0f} "
                    f"gross_after=${_final_cap_diag.get('gross_sum_after', 0):,.0f}"
                )
                if "sleeve" in sized.columns:
                    g_post = pd.to_numeric(sized["gross_target_usd"], errors="coerce").fillna(0.0)
                    print(
                        "[INFO] sleeve deployed vs YAML budget after final cap pass: "
                        + ", ".join(
                            f"{name}: ${float(g_post[sized['sleeve']==name].sum()):,.0f}/"
                            f"${tgt:,.0f} "
                            f"({100.0*float(g_post[sized['sleeve']==name].sum())/max(tgt,1e-9):.0f}%)"
                            for name, tgt in sleeve_targets.items()
                            if tgt > 1e-9
                        )
                    )

        # ----------------------------------------------------------------- optimal target pass
        # Re-run the full cap stack against the **structural** caps only (drops today's IBKR
        # ``shares_available`` and median-volume liquidity). This is the steady-state target
        # we'd hold if borrow returned and ADV were unconstrained — used by harvest +
        # rebalancers to anchor drift detection independently of today's tradeable size.
        try:
            ur_for_opt = ur_df if ('ur_df' in locals() and isinstance(ur_df, pd.DataFrame) and not ur_df.empty) else None
        except Exception:
            ur_for_opt = None
        sleeve_targets_for_opt = {
            "core_leveraged": float(core_budget),
            "yieldboost": float(yb_budget),
            "inverse_decay_bucket4": float(b4_core_reserved),
            VOL_ETP_BUCKET5_SLEEVE: float(b4_vol_reserved),
        }
        try:
            optimal_sized, _optimal_diag = _run_book_cap_pipeline_quiet(
                sized_pre_caps,
                target_gross_usd=float(target_gross_usd),
                delta_floor=float(delta_floor),
                strategy=strategy,
                paths=paths,
                shares_out_map=shares_out_map,
                returns_df=ur_for_opt,
                sleeve_budget_usd=sleeve_targets_for_opt,
                cap_mode="structural_only",
            )
        except Exception as ex:
            print(f"[WARN] optimal-target pipeline failed ({ex}); optimal columns will mirror executable.")
            optimal_sized = sized.copy()
            _optimal_diag = {"binding_per_row": np.full(len(sized), "fallback_to_executable", dtype=object)}
        optimal_binding = _optimal_diag.get("binding_per_row")
        if "sleeve" in sized.columns:
            opt_summary = []
            for sname, sbud in sleeve_targets_for_opt.items():
                if sbud <= 1e-9:
                    continue
                opt_g = float(pd.to_numeric(optimal_sized.loc[optimal_sized["sleeve"].eq(sname), "gross_target_usd"], errors="coerce").fillna(0.0).sum())
                exe_g = float(pd.to_numeric(sized.loc[sized["sleeve"].eq(sname), "gross_target_usd"], errors="coerce").fillna(0.0).sum())
                opt_summary.append(
                    f"{sname}: opt=${opt_g:,.0f} exe=${exe_g:,.0f} ({100.0 * exe_g / max(opt_g, 1e-9):.0f}%)"
                )
            if opt_summary:
                print("[INFO] sleeve optimal vs executable: " + "; ".join(opt_summary))

        # Refresh sleeve subsets from sized so weight diagnostics reflect caps + covariance.
        if not core_names_fit.empty:
            core_names_fit["gross_target_usd"] = sized.loc[core_names_fit.index, "gross_target_usd"].to_numpy()
        if not yb_names_fit.empty:
            yb_names_fit["gross_target_usd"] = sized.loc[yb_names_fit.index, "gross_target_usd"].to_numpy()

        def _populate_long_short_columns(frame: pd.DataFrame) -> pd.DataFrame:
            """Apply the standard core/yieldboost stock split + bucket-4 inverse legs to a sized
            frame. Same math as the executable pipeline; kept as a helper so we can run it twice
            (once for executable ``gross_target_usd``, once for ``optimal_gross_target_usd``).
            """
            frame = frame.copy()
            frame["beta_used_abs"] = frame["delta_abs"].clip(lower=delta_floor).fillna(1.0)
            frame["hedge_ratio"] = 1.0 / frame["beta_used_abs"]
            b4m = frame["sleeve"].isin(["inverse_decay_bucket4", VOL_ETP_BUCKET5_SLEEVE])
            sm = ~b4m
            frame.loc[sm, "long_usd"] = frame.loc[sm, "gross_target_usd"] / (1.0 + frame.loc[sm, "hedge_ratio"])
            frame.loc[sm, "short_usd"] = -(frame.loc[sm, "hedge_ratio"] * frame.loc[sm, "long_usd"])

            opt2_cols = {"b4_opt2_inverse_etf_short_usd", "b4_opt2_underlying_short_usd"}
            if opt2_cols.issubset(set(frame.columns)):
                inv_leg = pd.to_numeric(frame["b4_opt2_inverse_etf_short_usd"], errors="coerce")
                und_leg = pd.to_numeric(frame["b4_opt2_underlying_short_usd"], errors="coerce")
                opt2_rows = inv_leg.notna() & und_leg.notna()
            else:
                opt2_rows = pd.Series(False, index=frame.index)
            o2m = b4m & opt2_rows
            b4dm = b4m & ~o2m
            frame.loc[b4dm, "short_usd"] = -frame.loc[b4dm, "gross_target_usd"]
            frame.loc[b4dm, "long_usd"] = -(
                b4_partial_hedge_ratio * frame.loc[b4dm, "beta_used_abs"] * frame.loc[b4dm, "gross_target_usd"]
            )
            if bool(o2m.any()):
                opt2_gross = (
                    pd.to_numeric(frame.loc[o2m, "b4_opt2_inverse_etf_short_usd"], errors="coerce").fillna(0.0)
                    + pd.to_numeric(frame.loc[o2m, "b4_opt2_underlying_short_usd"], errors="coerce").fillna(0.0)
                ).replace(0.0, np.nan)
                cap_scale = (
                    pd.to_numeric(frame.loc[o2m, "gross_target_usd"], errors="coerce").fillna(0.0) / opt2_gross
                ).fillna(0.0).clip(lower=0.0)
                frame.loc[o2m, "short_usd"] = -(
                    pd.to_numeric(frame.loc[o2m, "b4_opt2_inverse_etf_short_usd"], errors="coerce").fillna(0.0)
                    * cap_scale
                )
                frame.loc[o2m, "long_usd"] = -(
                    pd.to_numeric(frame.loc[o2m, "b4_opt2_underlying_short_usd"], errors="coerce").fillna(0.0)
                    * cap_scale
                )
                if "b4_opt2_hedge_ratio" in frame.columns:
                    frame.loc[o2m, "hedge_ratio"] = pd.to_numeric(
                        frame.loc[o2m, "b4_opt2_hedge_ratio"], errors="coerce"
                    ).fillna(frame.loc[o2m, "hedge_ratio"])
            frame["underlying_target_usd"] = frame["long_usd"]
            frame["etf_target_usd"] = frame["short_usd"]
            return frame

        sized = _populate_long_short_columns(sized)
        optimal_sized = _populate_long_short_columns(optimal_sized)

        # Write sized notionals back into KEEP (purgatory remains 0)
        if "gross_target_usd" not in keep.columns:
            keep["gross_target_usd"] = 0.0
        keep.loc[sized.index, "gross_target_usd"] = pd.to_numeric(
            sized["gross_target_usd"], errors="coerce"
        ).fillna(0.0)
        keep.loc[sized.index, "long_usd"] = sized["long_usd"]
        keep.loc[sized.index, "short_usd"] = sized["short_usd"]
        keep.loc[sized.index, "underlying_target_usd"] = sized["underlying_target_usd"]
        keep.loc[sized.index, "etf_target_usd"] = sized["etf_target_usd"]
        keep.loc[sized.index, "sleeve"] = sized["sleeve"]
        for col in ("b1_trend_xsec_pctile", "b1_trend_multiplier"):
            if col in sized.columns:
                keep.loc[sized.index, col] = pd.to_numeric(sized[col], errors="coerce")

        # ---- Optimal (structural-only) target columns: parallel set written next to executable.
        # ``optimal_*`` reflects "where the position should sit" if today's IBKR shares_available
        # and ADV were unconstrained. ``binding_cap`` / ``optimal_binding_cap`` label what bound
        # each row in the executable / structural pass respectively. ``liquidity_gap_usd`` and
        # ``executable_pct_of_optimal`` summarize the daily gap that harvest fills as borrow returns.
        for col in (
            "optimal_gross_target_usd",
            "optimal_long_usd",
            "optimal_short_usd",
            "optimal_underlying_target_usd",
            "optimal_etf_target_usd",
            "optimal_binding_cap",
            "binding_cap",
            "liquidity_gap_usd",
            "executable_pct_of_optimal",
        ):
            if col not in keep.columns:
                keep[col] = 0.0 if col not in ("optimal_binding_cap", "binding_cap") else "none"

        keep.loc[optimal_sized.index, "optimal_gross_target_usd"] = pd.to_numeric(
            optimal_sized["gross_target_usd"], errors="coerce"
        ).fillna(0.0)
        keep.loc[optimal_sized.index, "optimal_long_usd"] = optimal_sized["long_usd"]
        keep.loc[optimal_sized.index, "optimal_short_usd"] = optimal_sized["short_usd"]
        keep.loc[optimal_sized.index, "optimal_underlying_target_usd"] = optimal_sized["underlying_target_usd"]
        keep.loc[optimal_sized.index, "optimal_etf_target_usd"] = optimal_sized["etf_target_usd"]

        if executable_binding is not None and len(executable_binding) == len(sized):
            keep.loc[sized.index, "binding_cap"] = list(executable_binding)
        if optimal_binding is not None and len(optimal_binding) == len(optimal_sized):
            keep.loc[optimal_sized.index, "optimal_binding_cap"] = list(optimal_binding)

        opt_g_series = pd.to_numeric(keep["optimal_gross_target_usd"], errors="coerce").fillna(0.0)
        exe_g_series = pd.to_numeric(keep["gross_target_usd"], errors="coerce").fillna(0.0)
        keep["liquidity_gap_usd"] = (opt_g_series - exe_g_series).astype(float)
        keep["executable_pct_of_optimal"] = (
            exe_g_series / opt_g_series.where(opt_g_series.abs() > 1e-9, np.nan)
        ).astype(float).fillna(1.0).clip(lower=0.0, upper=2.0)

        print(
            f"[INFO] sized core={len(core_names_fit)} yb={len(yb_names_fit)} b4={len(b4_names)} | "
            f"budgets: core=${core_budget:,.0f} yb=${yb_budget:,.0f} (post-b4 ${remainder_budget:,.0f}) "
            f"b4=${b4_reserved:,.0f} (core=${b4_core_reserved:,.0f} vol_etp=${b4_vol_reserved:,.0f})"
        )

        # Weight diagnostics
        if core_weight_method == "decay_score" and not core_names_fit.empty and core_budget > 0:
            cw = core_names_fit["gross_target_usd"] / core_budget
            print(
                f"[INFO] core_leveraged weights: max={cw.max():.3f} min={cw.min():.3f} "
                f"nonzero={int((cw > 1e-9).sum())}/{len(core_names_fit)}"
            )
        if yb_weight_method == "decay_score" and not yb_names_fit.empty and yb_budget > 0:
            yw = yb_names_fit["gross_target_usd"] / yb_budget
            print(
                f"[INFO] yieldboost weights: max={yw.max():.3f} min={yw.min():.3f} "
                f"nonzero={int((yw > 1e-9).sum())}/{len(yb_names_fit)}"
            )
        if not keep.empty:
            stock_mask_plan = keep["sleeve"].astype(str).isin(["core_leveraged", "yieldboost"])
            b12_under = (
                keep.loc[stock_mask_plan]
                .groupby("Underlying", as_index=False)["long_usd"]
                .sum()
                .rename(columns={"long_usd": "underlying_target_from_b12_usd"})
            )
            b4_under = (
                keep[keep["sleeve"].eq("inverse_decay_bucket4")]
                .groupby("Underlying", as_index=False)["long_usd"]
                .sum()
                .rename(columns={"long_usd": "underlying_target_from_b4_usd"})
            )
            by_under = (
                keep[["Underlying"]]
                .drop_duplicates()
                .merge(b12_under, on="Underlying", how="left")
                .merge(b4_under, on="Underlying", how="left")
            )
            by_under["underlying_target_from_b12_usd"] = pd.to_numeric(
                by_under["underlying_target_from_b12_usd"], errors="coerce"
            ).fillna(0.0)
            by_under["underlying_target_from_b4_usd"] = pd.to_numeric(
                by_under["underlying_target_from_b4_usd"], errors="coerce"
            ).fillna(0.0)
            by_under["underlying_internalized_usd"] = np.minimum(
                by_under["underlying_target_from_b12_usd"].clip(lower=0.0),
                (-by_under["underlying_target_from_b4_usd"]).clip(lower=0.0),
            )
            by_under["underlying_external_trade_usd"] = (
                by_under["underlying_target_from_b12_usd"] + by_under["underlying_target_from_b4_usd"]
            )
            keep = keep.merge(by_under, on="Underlying", how="left", suffixes=("", "_calc"))
            for col in (
                "underlying_target_from_b12_usd",
                "underlying_target_from_b4_usd",
                "underlying_internalized_usd",
                "underlying_external_trade_usd",
            ):
                calc_col = f"{col}_calc"
                if calc_col in keep.columns:
                    keep[col] = pd.to_numeric(keep[calc_col], errors="coerce").fillna(0.0)
                    keep = keep.drop(columns=[calc_col])

        # Output proposed trades:
        proposed = keep.copy()
        nonzero_mask = (
            (proposed["long_usd"] != 0)
            | (proposed["short_usd"] != 0)
            | (proposed.get("optimal_long_usd", 0) != 0)
            | (proposed.get("optimal_short_usd", 0) != 0)
        )
        proposed = proposed[nonzero_mask | (proposed["purgatory"] == True)]  # noqa: E712

        cols_to_drop = ["Leverage", "ExpectedLeverage", "cagr_positive", "delta_abs"]
        proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns], errors="ignore")

        dated_path = run_dir(args.run_date) / "proposed_trades.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(dated_path, index=False)
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        proposed.to_csv(proposed_latest_csv, index=False)

        print(f"[OK] Wrote proposed trades -> {dated_path}  (n={len(proposed)})")
        print(f"[OK] Updated latest proposed trades -> {proposed_latest_csv}  (n={len(proposed)})")

        # ----- optimal_targets.csv: structural-only target slice for harvest + rebalance.
        # Same row-level granularity as proposed_trades.csv but only the optimal_* columns
        # plus identifiers. Downstream tools (harvest, rebalancer) can read this even when an
        # older proposed_trades.csv lacks the new columns.
        optimal_path = run_dir(args.run_date) / "optimal_targets.csv"
        opt_keep_cols = [
            c for c in (
                "ETF",
                "Underlying",
                "sleeve",
                "strategy_tag",
                "Delta",
                "borrow_current",
                "optimal_gross_target_usd",
                "optimal_long_usd",
                "optimal_short_usd",
                "optimal_underlying_target_usd",
                "optimal_etf_target_usd",
                "optimal_binding_cap",
                "binding_cap",
                "liquidity_gap_usd",
                "executable_pct_of_optimal",
                "gross_target_usd",
                "long_usd",
                "short_usd",
            ) if c in proposed.columns
        ]
        proposed[opt_keep_cols].to_csv(optimal_path, index=False)
        print(f"[OK] Wrote optimal targets -> {optimal_path}  (rows={len(proposed)})")

        # ----- liquidity_gap_state.json: sticky per-pair gap age tracker. Records how many
        # consecutive run-dates a pair's structural target has exceeded its executable target,
        # so harvest + rebalance scripts can prioritize the longest-standing gaps when borrow
        # returns. Existing state is loaded and merged so days_open carries across runs.
        try:
            import json
            gap_state_path = Path("data") / "liquidity_gap_state.json"
            run_date_str = str(args.run_date)
            prior: dict[str, Any] = {}
            if gap_state_path.exists():
                try:
                    prior = json.loads(gap_state_path.read_text(encoding="utf-8"))
                    if not isinstance(prior, dict):
                        prior = {}
                except Exception:
                    prior = {}
            new_state: dict[str, Any] = {}
            for _, row in proposed.iterrows():
                etf_s = str(row.get("ETF", "")).strip()
                und_s = str(row.get("Underlying", "")).strip()
                if not etf_s or not und_s:
                    continue
                key = f"{etf_s}|{und_s}"
                gap = float(row.get("liquidity_gap_usd", 0.0) or 0.0)
                opt = float(row.get("optimal_gross_target_usd", 0.0) or 0.0)
                exe = float(row.get("gross_target_usd", 0.0) or 0.0)
                prev = prior.get(key, {}) if isinstance(prior.get(key), dict) else {}
                if abs(gap) > 1.0 and opt > 1.0:
                    days_open = int(prev.get("days_open", 0) or 0) + (
                        0 if str(prev.get("last_seen_run_date", "")) == run_date_str else 1
                    )
                else:
                    days_open = 0
                new_state[key] = {
                    "ETF": etf_s,
                    "Underlying": und_s,
                    "optimal_target": opt,
                    "last_executable": exe,
                    "gap": gap,
                    "days_open": days_open,
                    "last_seen_run_date": run_date_str,
                }
            gap_state_path.parent.mkdir(parents=True, exist_ok=True)
            gap_state_path.write_text(json.dumps(new_state, indent=2, sort_keys=True), encoding="utf-8")
            n_open = sum(1 for v in new_state.values() if int(v.get("days_open", 0) or 0) > 0)
            print(f"[OK] Updated liquidity gap state -> {gap_state_path}  (rows={len(new_state)}, gaps_open={n_open})")
        except Exception as ex:
            print(f"[WARN] liquidity_gap_state.json update failed: {ex}")

    print("[OK] Bucket 1/2/4 position generation complete. Flow sleeve execution remains separate.")


if __name__ == "__main__":
    main()
