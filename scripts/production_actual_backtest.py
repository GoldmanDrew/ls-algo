"""Production-actual book backtest (May 2025 -> present).

Primary mode
------------
``prod`` (default)
    Full historical replay of today's ``generate_trade_plan`` stack on each
    archived ``etf_screened_today.csv``: B1/B2 + B4 opt2 → crash → smooth →
    ratchet, with isolated state carried day-to-day. Spot ``borrow_current`` /
    screener edge columns as production (no avg-borrow overlay). Does **not**
    consume archived ``proposed_trades.csv``. Plans execute via the unified
    share-hold ledger (next-close, weekly W-FRI Phase-2b, borrow/margin/slippage,
    3.8% short-proceeds credit).

Legacy modes (CLI only; notebook no longer runs them)
-----------------------------------------------------
``gtp`` — decay-score mirror (avg borrow + net edge; no opt2/crash/ratchet).
``frozen`` / ``replay`` / ``recompute`` — see git history or ``--help``.

Outputs under ``notebooks/output/production_actual_bt/`` (``prod`` writes at the
root; legacy modes still use subdirs).

Run
---
    python -m scripts.production_actual_backtest
    python -m scripts.production_actual_backtest --mode prod --start 2026-02-27
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Keys serialized into report.json["book"] from simulate meta.
BOOK_REPORT_KEYS = (
    "cagr",
    "vol",
    "sharpe",
    "maxdd",
    "start_usd",
    "end_usd",
    "n_plans_used",
    "first_plan",
    "cash_days",
    "same_run_churn_enabled",
    "avoided_round_trip_usd",
    "risk_override_turnover_usd",
    "one_terminal_target_per_symbol",
    "b4_membership_clock",
    "stock_rebalance_clock",
    "operator_check_days",
    "b4_apply_resize_bands",
    "b4_ratchet_execution_guard",
    "b4_empty_plan_policy",
    "net_shared_underlyings",
    "turnover_pace_enabled",
    "turnover_pace_mode",
    "turnover_pace_version",
    "confirmation_count",
    "entry_ramp_sessions",
    "reduction_ramp_sessions",
    "remaining_gap_rate",
    "stock_midweek_mode",
    "midweek_hedge_repair",
    "hedge_reserve_frac",
    "adv_participation_pct",
    "n_deferred_pace",
    "n_hedge_repairs",
    "n_growth_blocked_hedge_infeasible",
    "n_b4_cadence_rebals",
    "n_b4_membership_deferred",
    "n_b4_empty_plan_holds",
    "n_b4_ratchet_pins",
    "n_b4_cadence_pairs",
    "n_purgatory_reductions",
    "purgatory_model_zero_policy",
)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from risk_dashboard.metrics import compute_sleeve_target_weights  # noqa: E402
from strategy_config import load_config  # noqa: E402
from purgatory_policy import constrain_pair_targets  # noqa: E402
from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h  # noqa: E402
from scripts.bucket4_hedge_cadence import build_h_series, build_rebal_dates  # noqa: E402
from scripts.bucket4_vol_shape_signals import get_pair_signal  # noqa: E402
from scripts.sizing_tilt_cadence_bt import (  # noqa: E402
    knobs_from_yaml,
    load_price_panel,
    load_universe,
    make_knobs,
    pair_daily_returns,
    perf,
)
from scripts.pair_price_panel import (  # noqa: E402
    apply_delist_cutoff,
    apply_panel_leg_patches,
    frames_from_metrics,
)

TRADING_DAYS = 252
STOCK_SLEEVES = ("core_leveraged", "yieldboost")
B4_SLEEVE = "inverse_decay_bucket4"
B5_SLEEVE = "volatility_etp_bucket5"
B4_SLEEVES = (B4_SLEEVE,)  # B5 routed separately via carry engine
ALL_SLEEVES = STOCK_SLEEVES + (B4_SLEEVE, B5_SLEEVE)
RUNS_DIR = REPO / "data" / "runs"
MIN_B4_SESSIONS = 40  # admit newer listings; signals warm up on available history
MIN_B4_TRADE_DAYS = 20
# A production plan already owns eligibility/sizing. A newly listed B4 wrapper
# needs only enough prices to mark its position; it must not be excluded by the
# generic 40-session panel minimum.
MIN_B4_LISTING_PRICE_DAYS = 2

SLEEVE_ALIASES = {
    "whitelist_stock": "yieldboost",
    "yield_boost": "yieldboost",
    "bucket1": "core_leveraged",
    "bucket_1": "core_leveraged",
    "bucket2": "yieldboost",
    "bucket_2": "yieldboost",
    "bucket4": "inverse_decay_bucket4",
    "bucket_4": "inverse_decay_bucket4",
    "bucket5": "volatility_etp_bucket5",
    "bucket_5": "volatility_etp_bucket5",
}


def _norm(x) -> str:
    return str(x).strip().upper().replace(".", "-")


def _float_or(x: Any, default: float = 0.0) -> float:
    """Finite numeric coercion; unlike ``x or default``, NaN uses default."""
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) and np.isfinite(v) else float(default)


def sleeve_budgets_usd(cfg: dict) -> dict[str, float]:
    weights, book = compute_sleeve_target_weights(cfg)
    return {
        "core_leveraged": float(weights.get("bucket_1") or 0.0) * book,
        "yieldboost": float(weights.get("bucket_2") or 0.0) * book,
        "inverse_decay_bucket4": float(weights.get("bucket_4") or 0.0) * book,
        "volatility_etp_bucket5": float(weights.get("bucket_5") or 0.0) * book,
    }


def apply_blacklist_except(cfg: dict, except_symbols: list[str] | tuple[str, ...] | None) -> list[str]:
    """Remove tickers from ``cfg['strategy']['blacklist']`` (in place).

    Returns the symbols actually removed (uppercased). Notebook / one-off
    backtests use this to temporarily re-admit names like APLD / SMR / CBRS
    without editing live ``strategy_config.yml``.
    """
    if not except_symbols:
        return []
    want = {str(s).strip().upper() for s in except_symbols if str(s).strip()}
    if not want:
        return []
    strat = cfg.setdefault("strategy", {})
    raw = list(strat.get("blacklist") or [])
    kept: list[Any] = []
    removed: list[str] = []
    for item in raw:
        sym = str(item).strip().upper()
        if sym in want:
            removed.append(sym)
        else:
            kept.append(item)
    strat["blacklist"] = kept
    return sorted(set(removed))


def apply_notebook_b4_borrow_overrides(
    cfg: dict,
    *,
    entry_borrow_cap: float | None = None,
    keep_borrow_cap: float | None = None,
    borrow_ramp_lo: float | None = None,
    borrow_ramp_hi: float | None = None,
    shift_ramp_with_band: bool = True,
) -> dict[str, Any]:
    """Notebook-only B4 borrow band / opt2 ramp overrides (mutates ``cfg``).

    Live ``strategy_config.yml`` is not written. Returns a small audit dict of
    before→after values applied.
    """
    audit: dict[str, Any] = {}
    screener = cfg.setdefault("screener", {})
    per_bucket = screener.setdefault("per_bucket", {})
    b4 = dict(per_bucket.get("bucket_4") or {})
    old_entry = float(b4.get("entry_borrow_cap", 0.70) or 0.70)
    old_keep = float(b4.get("keep_borrow_cap", 0.90) or 0.90)
    new_entry = old_entry if entry_borrow_cap is None else float(entry_borrow_cap)
    new_keep = old_keep if keep_borrow_cap is None else float(keep_borrow_cap)
    if new_keep < new_entry:
        new_keep = new_entry
    b4["entry_borrow_cap"] = new_entry
    b4["keep_borrow_cap"] = new_keep
    per_bucket["bucket_4"] = b4
    audit["entry_borrow_cap"] = {"old": old_entry, "new": new_entry}
    audit["keep_borrow_cap"] = {"old": old_keep, "new": new_keep}

    opt2 = (
        ((cfg.get("portfolio") or {}).get("sleeves") or {})
        .get("inverse_decay_bucket4", {})
        .get("rules", {})
        .get("bucket4_weekly_opt2", {})
    )
    if not isinstance(opt2, dict):
        return audit
    old_lo = float(opt2.get("borrow_ramp_lo", 0.80) or 0.80)
    old_hi = float(opt2.get("borrow_ramp_hi", 1.20) or 1.20)
    if borrow_ramp_lo is not None or borrow_ramp_hi is not None:
        new_lo = old_lo if borrow_ramp_lo is None else float(borrow_ramp_lo)
        new_hi = old_hi if borrow_ramp_hi is None else float(borrow_ramp_hi)
    elif shift_ramp_with_band and (
        entry_borrow_cap is not None or keep_borrow_cap is not None
    ):
        # Shift ramp by the same delta as the entry band (70→60 ⇒ 80→70, 120→110).
        delta = new_entry - old_entry
        new_lo = old_lo + delta
        new_hi = old_hi + delta
    else:
        return audit
    if new_hi < new_lo:
        new_hi = new_lo
    opt2["borrow_ramp_lo"] = float(new_lo)
    opt2["borrow_ramp_hi"] = float(new_hi)
    audit["borrow_ramp_lo"] = {"old": old_lo, "new": float(new_lo)}
    audit["borrow_ramp_hi"] = {"old": old_hi, "new": float(new_hi)}
    return audit


def rebalance_knobs(cfg: dict) -> dict:
    reb = ((cfg.get("portfolio") or {}).get("rebalance") or {})
    resize = reb.get("resize") or {}
    return {
        "target_basis": reb.get("target_basis", "hybrid"),
        "purgatory_execution": str(reb.get("purgatory_execution", "reduce_only")),
        "same_run_churn_enabled": bool(
            (reb.get("same_run_churn") or {}).get("enabled", True)
        ),
        "enter_band_pct": float(resize.get("enter_band_pct", 0.12) or 0.12),
        "exit_band_pct": float(resize.get("exit_band_pct", 0.04) or 0.04),
        "min_trade_usd": float(reb.get("min_trade_usd", 250) or 250),
        "stock_rebalance": "W-FRI",  # legacy label; see stock_rebalance_clock
        "stock_rebalance_clock": str(
            ((cfg.get("production_actual_backtest") or {}).get(
                "stock_rebalance_clock", "operator_5d"
            ))
            or "operator_5d"
        ).strip().lower(),
        "slippage_bps": 20.0,
        # Diamond Creek v15 / Clear Street low-touch baseline.  Commission is
        # charged per leg in replay/recompute, including the opening trade.
        "commission_per_share": 0.0035,
        # Until an archived daily OBFR curve is part of the run artifact, use
        # the same explicit fallback economics as the Diamond Creek notebook:
        # 4.00% benchmark + 45 bp debit spread, Actual/360.  The notebook
        # includes a sensitivity panel so this approximation stays visible.
        "margin_rate_annual": 0.0445,
        "financing_daycount": 360.0,
        # IBKR short-sale proceeds credit (interest on cash from shorts).
        # Modelled as a flat 3.8% annual on absolute short notional / daycount.
        # Borrow fee is still charged separately from IBKR feerate / screened borrow.
        "short_proceeds_credit_annual": 0.038,
        # A plan written on day T is assumed known after that close and is
        # executed at the next available close.  Its P&L starts on T+2 close.
        "execution_lag_sessions": 1,
        # Preserve the plan's gross/equity multiple as NAV changes, matching
        # Diamond Creek's dynamic target-gross convention.
        "target_notional_mode": "equity_scaled",
        # Scale each sleeve's plan legs up/down so sleeve gross equals the YAML
        # sleeve budget (then apply equity_scaled NAV scaling).
        "scale_sleeves_to_budget": True,
        # GTP mode: size every screened day, but trade on the weekly clock so
        # daily score reshuffles do not pay 20 bp on the whole book.
        "retarget_on_plan_change": False,
        # B4 book execution: cadence (TR/VCR) vs legacy weekly_plan_legs.
        "b4_execution": str(
            ((cfg.get("production_actual_backtest") or {}).get("b4_execution", "cadence"))
            or "cadence"
        ).strip().lower(),
        "apply_delist_flatten": bool(
            (cfg.get("production_actual_backtest") or {}).get("apply_delist_flatten", True)
        ),
        "use_borrow_history": bool(
            (cfg.get("production_actual_backtest") or {}).get("use_borrow_history", True)
        ),
        # Purgatory with missing/zero model_* share-holds (live: executable 0 is
        # not a Phase-1 close). Set "exit" only for A/B vs the old flatten path.
        "purgatory_model_zero_policy": str(
            ((cfg.get("production_actual_backtest") or {}).get(
                "purgatory_model_zero_policy", "hold"
            ))
            or "hold"
        ).strip().lower(),
        # G1: membership add/drop clock (kills daily plan flicker).
        "b4_membership_clock": str(
            ((cfg.get("production_actual_backtest") or {}).get(
                "b4_membership_clock", "operator_5d"
            ))
            or "operator_5d"
        ).strip().lower(),
        "operator_check_days": int(
            ((cfg.get("production_actual_backtest") or {}).get("operator_check_days"))
            or (
                (
                    (
                        ((cfg.get("portfolio") or {}).get("sleeves") or {})
                        .get("inverse_decay_bucket4", {})
                        .get("rules", {})
                        .get("bucket4_weekly_opt2", {})
                        .get("hedge_cadence_policy", {})
                        or {}
                    ).get("operator_check_days")
                )
                or 5
            )
        ),
        # G2: apply Phase-2b bands on B4 cadence retargets.
        "b4_apply_resize_bands": bool(
            (cfg.get("production_actual_backtest") or {}).get("b4_apply_resize_bands", True)
        ),
        # G3: ledger-time inverse cover pin / trim cap (live Phase-2b guard).
        "b4_ratchet_execution_guard": bool(
            (cfg.get("production_actual_backtest") or {}).get(
                "b4_ratchet_execution_guard", True
            )
        ),
        "b4_allow_inverse_cover": bool(
            (
                (
                    ((cfg.get("portfolio") or {}).get("sleeves") or {})
                    .get("inverse_decay_bucket4", {})
                    .get("rules", {})
                    .get("ratchet", {})
                    or {}
                )
                .get("execution", {})
                or {}
            ).get("allow_inverse_cover", True)
        ),
        # When plan B4 executable gross is ~0 (archive gap / sizing fail), do not
        # true-drop the held sleeve into cash. hard_exit / delist still flatten.
        "b4_empty_plan_policy": str(
            ((cfg.get("production_actual_backtest") or {}).get(
                "b4_empty_plan_policy", "hold"
            ))
            or "hold"
        ).strip().lower(),
        # Net shared underlyings across sleeves for borrow / short-credit / margin.
        "net_shared_underlyings": bool(
            (cfg.get("production_actual_backtest") or {}).get(
                "net_shared_underlyings", True
            )
        ),
        # Sim-only gradual turnover (EMA sleeve gross + leg step + daily cap).
        **_turnover_pace_knobs(cfg),
    }


def _turnover_pace_knobs(cfg: dict) -> dict[str, Any]:
    """``production_actual_backtest.turnover_pace`` → flat rebalance_knobs keys."""
    pace = ((cfg.get("production_actual_backtest") or {}).get("turnover_pace") or {})
    hedge = ((((cfg.get("portfolio") or {}).get("rebalance") or {}).get("hedge")) or {})
    too_long = hedge.get("too_long") or {}
    too_short = hedge.get("too_short") or {}
    enabled = bool(pace.get("enabled", True))
    mode = str(pace.get("mode", "hedge_safe_v1") or "hedge_safe_v1").strip().lower()
    if not enabled:
        mode = "off"
    if mode not in {"hedge_safe_v1", "legacy", "off"}:
        mode = "hedge_safe_v1"
    return {
        "turnover_pace_enabled": mode != "off",
        "turnover_pace_mode": mode,
        # Gradual weekly-gross preset: confirm once, step remaining gap on the
        # operator clock only (entry/reduction multi-session ramps stay for
        # ramp_and_hedge / hedge_only midweek continuation).
        "confirmation_count": int(pace.get("confirmation_count", 1) or 1),
        "entry_ramp_sessions": int(pace.get("entry_ramp_sessions", 1) or 1),
        "reduction_ramp_sessions": int(pace.get("reduction_ramp_sessions", 1) or 1),
        "remaining_gap_rate": float(pace.get("remaining_gap_rate", 0.25) or 0.25),
        "target_blend_alpha": float(pace.get("target_blend_alpha", 0.25) or 0.25),
        # Off stock-rebal-clock trading under hedge_safe_v1:
        #   rebal_only — no structural gross midweek; Phase-3 only if
        #               midweek_hedge_repair is true
        #   hedge_only — incomplete entry/exit ramps + Phase-3 hedge repair
        #   ramp_and_hedge — legacy daily gap-chase
        "stock_midweek_mode": str(
            pace.get("stock_midweek_mode", "rebal_only") or "rebal_only"
        ).strip().lower(),
        # When true with rebal_only: restore delta hedge midweek without
        # changing structural gross (share-hold size, repair residual only).
        "midweek_hedge_repair": bool(pace.get("midweek_hedge_repair", False)),
        "hedge_reserve_frac": float(pace.get("hedge_reserve_frac", 0.15) or 0.15),
        "adv_participation_pct": float(pace.get("adv_participation_pct", 0.10) or 0.10),
        "sleeve_gross_ema_alpha": float(pace.get("sleeve_gross_ema_alpha", 0.35) or 0.35),
        "max_leg_step_pct": float(pace.get("max_leg_step_pct", 0.25) or 0.25),
        "pair_gross_ramp_pct": float(
            pace.get("pair_gross_ramp_pct", pace.get("max_leg_step_pct", 0.25)) or 0.25
        ),
        "max_daily_turnover_pct": float(pace.get("max_daily_turnover_pct", 0.10) or 0.10),
        "legacy_max_daily_turnover_pct": float(
            pace.get("legacy_max_daily_turnover_pct", 0.15) or 0.15
        ),
        "establish_budget_frac": float(pace.get("establish_budget_frac", 0.50) or 0.50),
        "resize_age_boost_days": int(pace.get("resize_age_boost_days", 5) or 5),
        "hedge_long_trigger_net_pct": float(too_long.get("trigger_net_pct", 0.04) or 0.04),
        "hedge_long_target_net_pct": float(too_long.get("target_net_pct", 0.01) or 0.01),
        "hedge_short_trigger_net_pct": float(too_short.get("trigger_net_pct", 0.01) or 0.01),
        "hedge_short_target_net_pct": float(too_short.get("target_net_pct", 0.00) or 0.00),
    }


def b4_leg_targets_from_gross(gross_usd: float, h: float, beta_abs: float) -> tuple[float, float]:
    """Opt2-convention B4 shorts: ``inv = G/(1+h*|β|)``, ``und = h*|β|*inv``."""
    g = abs(float(gross_usd))
    hh = max(float(h), 0.0)
    ba = max(abs(float(beta_abs)), 1e-6)
    if g <= 1e-9:
        return 0.0, 0.0
    inv = g / (1.0 + hh * ba)
    und = hh * ba * inv
    return -float(inv), -float(und)


def _b4_plan_executable_gross(plan: pd.DataFrame | None) -> float:
    """Sum of executable |legs| on inverse_decay_bucket4 rows (0 if absent)."""
    if plan is None or getattr(plan, "empty", True):
        return 0.0
    df = plan
    if "sleeve" in df.columns:
        m = df["sleeve"].astype(str).str.strip().str.lower().eq(B4_SLEEVE)
        df = df.loc[m]
    if df.empty:
        return 0.0
    if {"etf_usd", "underlying_usd"}.issubset(df.columns):
        a = pd.to_numeric(df["etf_usd"], errors="coerce").fillna(0.0).abs()
        b = pd.to_numeric(df["underlying_usd"], errors="coerce").fillna(0.0).abs()
        return float((a + b).sum())
    if {"long_usd", "short_usd"}.issubset(df.columns):
        a = pd.to_numeric(df["long_usd"], errors="coerce").fillna(0.0).abs()
        b = pd.to_numeric(df["short_usd"], errors="coerce").fillna(0.0).abs()
        return float((a + b).sum())
    for col in ("gross_target_usd", "gross_usd", "model_gross_target_usd"):
        if col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).abs().sum())
    return 0.0


def _underlying_net_by_symbol(
    cur: pd.DataFrame,
) -> dict[str, float]:
    """Signed underlying USD summed across all pairs sharing an Underlying."""
    if cur is None or cur.empty or "underlying_usd" not in cur.columns:
        return {}
    und_col = "Underlying" if "Underlying" in cur.columns else None
    out: dict[str, float] = {}
    for etf, row in cur.iterrows():
        und = _norm(row.get(und_col, "")) if und_col else ""
        if not und or und in {"NAN", "NONE", ""}:
            # Fall back to pair id so orphan legs still finance.
            und = f"__PAIR__{_norm(etf)}"
        out[und] = float(out.get(und, 0.0)) + float(row["underlying_usd"])
    return out


def _netted_book_notionals(cur: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return (long, short, gross, net) after netting shared underlyings.

    ETF legs stay per-pair. Underlying legs collapse by Underlying ticker so a
    B1 long and B4 short on the same name offset before gross/net are computed.
    """
    if cur is None or cur.empty:
        return 0.0, 0.0, 0.0, 0.0
    long_n = 0.0
    short_n = 0.0
    for _, row in cur.iterrows():
        etf_usd = float(row["etf_usd"])
        if etf_usd >= 0:
            long_n += etf_usd
        else:
            short_n += -etf_usd
    for net_und in _underlying_net_by_symbol(cur).values():
        if net_und >= 0:
            long_n += float(net_und)
        else:
            short_n += -float(net_und)
    gross = long_n + short_n
    net = long_n - short_n
    return float(long_n), float(short_n), float(gross), float(net)


def _sleeve_capital_bases(cur: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-sleeve (net_cap, gross_cap) from open pair legs (EOD-style, un-netted).

    Net = signed sum of ETF + underlying USD in the sleeve. Gross = sum of
    absolute legs. Cross-sleeve underlying internalization is *not* applied
    here so ROC/ROG match the EOD email bucket capital convention.
    """
    out: dict[str, tuple[float, float]] = {s: (0.0, 0.0) for s in ALL_SLEEVES}
    if cur is None or cur.empty:
        return out
    for _, row in cur.iterrows():
        sl = str(row.get("sleeve", "") or "")
        if sl not in out:
            continue
        etf_usd = float(row["etf_usd"])
        und_usd = float(row["underlying_usd"])
        net_c, gross_c = out[sl]
        out[sl] = (net_c + etf_usd + und_usd, gross_c + abs(etf_usd) + abs(und_usd))
    return out


def compute_sleeve_return_metrics(
    sleeve_daily: pd.DataFrame,
    *,
    sleeves: tuple[str, ...] | None = None,
    deployed_gross_floor_usd: float = 1.0,
) -> pd.DataFrame:
    """Return metrics on **deployed** capital (EOD email convention).

    Primary metrics (use these):
    - ``rog_deployed`` = PnL on deployed days / mean(gross | gross > floor)
    - ``rog_deployed_ann`` = ``rog_deployed * 252 / n_days`` (simple annualization)
    - ``roc_deployed`` = same with mean(|net| | deployed) when that base > 0

    Calendar averages ``rog`` / ``roc`` (mean over *all* days, including flat)
    are kept for backward compatibility but understate capital when a sleeve
    ramps slowly and inflate ROG via a smaller denominator.

    Sleeve capital bases use each sleeve's own MV. Book row uses book
    ``gross_notional`` / ``net_notional`` (netted underlyings when
    ``net_shared_underlyings`` was on — book ``roc`` is then not "capital at work").
    """
    sleeves = tuple(sleeves) if sleeves is not None else ALL_SLEEVES
    empty_cols = [
        "sleeve",
        "pnl_usd",
        "pnl_deployed_usd",
        "avg_net_cap",
        "avg_gross_cap",
        "avg_gross_deployed",
        "avg_abs_net_deployed",
        "roc",
        "rog",
        "roc_deployed",
        "rog_deployed",
        "rog_deployed_ann",
        "deployed_day_frac",
        "n_days",
        "n_deployed_days",
    ]
    if sleeve_daily is None or sleeve_daily.empty:
        return pd.DataFrame(columns=empty_cols)
    df = sleeve_daily.copy()
    rows: list[dict[str, Any]] = []
    n_days = int(len(df))
    floor = float(max(deployed_gross_floor_usd, 0.0))

    def _safe_ret(pnl: float, base: float) -> float:
        if not np.isfinite(base) or abs(base) <= 1e-9:
            return float("nan")
        return float(pnl) / float(base)

    def _ann(period_ret: float) -> float:
        if not np.isfinite(period_ret) or n_days <= 0:
            return float("nan")
        return float(period_ret) * (float(TRADING_DAYS) / float(n_days))

    def _row_from_series(
        *,
        name: str,
        pnl_all: pd.Series,
        net: pd.Series,
        gross: pd.Series,
    ) -> dict[str, Any]:
        pnl = float(pnl_all.sum())
        avg_net = float(net.mean()) if len(net) else float("nan")
        avg_gross = float(gross.mean()) if len(gross) else float("nan")
        deployed = gross > floor
        n_dep = int(deployed.sum())
        if n_dep > 0:
            pnl_dep = float(pnl_all.loc[deployed].sum())
            avg_g_dep = float(gross.loc[deployed].mean())
            avg_abs_net_dep = float(net.loc[deployed].abs().mean())
        else:
            pnl_dep = float("nan")
            avg_g_dep = float("nan")
            avg_abs_net_dep = float("nan")
        rog_dep = _safe_ret(pnl_dep, avg_g_dep)
        roc_dep = _safe_ret(pnl_dep, avg_abs_net_dep)
        return {
            "sleeve": name,
            "pnl_usd": pnl,
            "pnl_deployed_usd": pnl_dep if n_dep > 0 else 0.0,
            "avg_net_cap": avg_net,
            "avg_gross_cap": avg_gross,
            "avg_gross_deployed": avg_g_dep,
            "avg_abs_net_deployed": avg_abs_net_dep,
            "roc": _safe_ret(pnl, avg_net) if avg_net > 0 else float("nan"),
            "rog": _safe_ret(pnl, avg_gross),
            "roc_deployed": roc_dep,
            "rog_deployed": rog_dep,
            "rog_deployed_ann": _ann(rog_dep),
            "deployed_day_frac": float(n_dep) / float(n_days) if n_days else float("nan"),
            "n_days": n_days,
            "n_deployed_days": n_dep,
        }

    for s in sleeves:
        if s not in df.columns:
            continue
        pnl_s = pd.to_numeric(df[s], errors="coerce").fillna(0.0)
        net_col = f"{s}__net_cap"
        gross_col = f"{s}__gross_cap"
        net_s = (
            pd.to_numeric(df[net_col], errors="coerce").fillna(0.0)
            if net_col in df.columns
            else pd.Series(0.0, index=df.index)
        )
        gross_s = (
            pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0)
            if gross_col in df.columns
            else pd.Series(0.0, index=df.index)
        )
        rows.append(_row_from_series(name=s, pnl_all=pnl_s, net=net_s, gross=gross_s))

    if "daily_net_pnl" in df.columns and "gross_notional" in df.columns:
        book_pnl_s = pd.to_numeric(df["daily_net_pnl"], errors="coerce").fillna(0.0)
        book_gross = pd.to_numeric(df["gross_notional"], errors="coerce").fillna(0.0)
        book_net = (
            pd.to_numeric(df["net_notional"], errors="coerce").fillna(0.0)
            if "net_notional" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        rows.insert(
            0,
            _row_from_series(
                name="BOOK", pnl_all=book_pnl_s, net=book_net, gross=book_gross
            ),
        )
    return pd.DataFrame(rows)


def _membership_day_set(
    cal: pd.DatetimeIndex,
    *,
    mode: str,
    check_days: int,
) -> set[pd.Timestamp]:
    """Sessions when Phase-1-style add/drop is allowed."""
    mode_n = str(mode or "operator_5d").strip().lower()
    cal = pd.DatetimeIndex(cal).normalize()
    if mode_n in {"every_plan", "every", "all", "off"}:
        return set(cal)
    if mode_n in {"weekly_fri", "w-fri", "friday"}:
        return {pd.Timestamp(d) for d in cal if int(pd.Timestamp(d).weekday()) == 4}
    step = max(int(check_days or 5), 1)
    return {pd.Timestamp(d) for i, d in enumerate(cal) if i % step == 0}


def _truthy(x: object) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _apply_b4_inverse_floor(
    old_etf: float,
    raw_etf: float,
    raw_und: float,
    *,
    h: float,
    beta_abs: float,
    ratchet_released: bool,
) -> tuple[float, float]:
    """Grow-only inverse pin when ratchet not released; re-solve und from h."""
    if ratchet_released:
        return float(raw_etf), float(raw_und)
    # Both legs short (negative): larger short = more negative.
    if old_etf < -1e-9 and raw_etf <= 0:
        floored = min(float(raw_etf), float(old_etf))
    elif abs(old_etf) > 1e-9 and abs(raw_etf) + 1e-9 < abs(old_etf):
        floored = float(old_etf)
    else:
        floored = float(raw_etf)
    ba = max(abs(float(beta_abs)), 1e-6)
    hh = max(float(h), 0.0)
    und = -hh * ba * abs(floored) if floored <= 0 else float(raw_und)
    return float(floored), float(und)


def _apply_b4_ratchet_cover_guard(
    old_etf: float,
    new_etf: float,
    new_und: float,
    *,
    plan_row: pd.Series | Mapping[str, Any] | None,
    allow_inverse_cover: bool,
    h: float,
    beta_abs: float,
) -> tuple[float, float, str]:
    """Pin or cap inverse covers to match live Phase-2b ratchet guard.

    Returns ``(etf_usd, und_usd, reason)`` where reason is ``ok`` / ``pin`` / ``trim_cap``.
    """
    # Cover = reducing a short (buying back): new > old when old is negative.
    covering = float(old_etf) < -1e-9 and float(new_etf) > float(old_etf) + 1e-6
    if not covering:
        return float(new_etf), float(new_und), "ok"

    released = False
    trim_usd = 0.0
    if plan_row is not None:
        released = _truthy(plan_row.get("ratchet_released", False))
        trim_usd = float(
            pd.to_numeric(plan_row.get("ratchet_trim_usd", 0.0), errors="coerce") or 0.0
        )

    if not (allow_inverse_cover and released):
        # Pin inverse; keep und proposal (bands may still move hedge).
        return float(old_etf), float(new_und), "pin"

    cover_amt = float(new_etf) - float(old_etf)  # positive dollars bought back
    if trim_usd > 1e-9 and cover_amt > trim_usd + 1e-6:
        capped = float(old_etf) + float(trim_usd)
        ba = max(abs(float(beta_abs)), 1e-6)
        hh = max(float(h), 0.0)
        und = -hh * ba * abs(capped)
        return float(capped), float(und), "trim_cap"
    return float(new_etf), float(new_und), "ok"


def _model_legs_from_plan_row(row: pd.Series | Mapping[str, Any] | None) -> tuple[float, float, float]:
    """Return (etf_usd, underlying_usd, gross) from model_* / optimal_* / executable.

    Convention matches proposed_trades: short/etf = ETF leg, long/underlying = und leg.
    Missing model columns yield (nan, nan, nan) so callers can share-hold.
    """
    if row is None:
        return float("nan"), float("nan"), float("nan")

    def _get(*names: str) -> float:
        for n in names:
            if hasattr(row, "index") and n in row.index:
                v = pd.to_numeric(row.get(n), errors="coerce")
                if pd.notna(v):
                    return float(v)
            elif isinstance(row, Mapping) and n in row:
                v = pd.to_numeric(row.get(n), errors="coerce")
                if pd.notna(v):
                    return float(v)
        return float("nan")

    etf = _get("model_short_usd", "optimal_short_usd", "etf_target_usd")
    und = _get("model_long_usd", "optimal_long_usd", "underlying_target_usd")
    gross = _get("model_gross_target_usd", "optimal_gross_target_usd")
    if not np.isfinite(gross):
        parts = [x for x in (etf, und) if np.isfinite(x)]
        gross = float(sum(abs(x) for x in parts)) if parts else float("nan")
    if not np.isfinite(etf) and not np.isfinite(und) and not np.isfinite(gross):
        return float("nan"), float("nan"), float("nan")
    return (
        float(etf) if np.isfinite(etf) else 0.0,
        float(und) if np.isfinite(und) else 0.0,
        float(gross) if np.isfinite(gross) else 0.0,
    )


def _plan_row_for_etf(plan: pd.DataFrame | None, etf: str) -> pd.Series | None:
    if plan is None or plan.empty or "ETF" not in plan.columns:
        return None
    m = plan["ETF"].map(_norm) == _norm(etf)
    if not bool(m.any()):
        return None
    return plan.loc[m].iloc[-1]


def _prepare_b4_cadence_state(
    panel: dict[str, pd.DataFrame],
    cal: pd.DatetimeIndex,
    *,
    und_by_etf: dict[str, str] | None = None,
    b4_etfs: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Precompute per-ETF h series + rebal dates for B4 cadence execution."""
    blk = knobs_from_yaml()
    knobs = make_knobs(blk)
    und_map = {_norm(k): _norm(v) for k, v in (und_by_etf or {}).items()}
    want = {_norm(e) for e in (b4_etfs or panel.keys())}
    out: dict[str, dict[str, Any]] = {}
    for etf_raw, px in panel.items():
        etf = _norm(etf_raw)
        if etf not in want:
            continue
        if px is None or px.empty or "b_px" not in px.columns:
            continue
        try:
            cal_e = pd.DatetimeIndex(px.index).intersection(cal)
            if len(cal_e) < MIN_B4_TRADE_DAYS:
                continue
            warmup = min(60, max(0, len(cal_e) - MIN_B4_TRADE_DAYS))
            und = und_map.get(etf, "")
            sig = get_pair_signal(
                etf,
                und or etf,
                cal_e,
                history={},
                underlying_prices=px["b_px"],
                window=60,
                lookahead_shift=1,
            )
            h_daily = build_h_series(sig, cal_e, knobs=knobs)
            rb, _ = build_rebal_dates(sig, cal_e, knobs=knobs, warmup_bdays=warmup)
            out[etf] = {
                "h": h_daily,
                "rebal": set(pd.DatetimeIndex(rb)),
                "h_mid": float(getattr(knobs, "h_mid", 0.45) or 0.45),
            }
        except Exception:
            continue
    return out


def _load_borrow_history_map() -> dict[str, pd.Series]:
    """Optional day-by-day borrow from etf-dashboard ``borrow_history.json``."""
    candidates = [
        REPO.parent / "etf-dashboard" / "data" / "borrow_history.json",
        REPO / "data" / "borrow_history.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, pd.Series] = {}
    for sym, payload in raw.items():
        key = _norm(sym)
        try:
            if isinstance(payload, dict) and "history" in payload:
                payload = payload["history"]
            if isinstance(payload, list):
                dates, rates = [], []
                for row in payload:
                    if not isinstance(row, dict):
                        continue
                    dt = pd.to_datetime(row.get("date") or row.get("asof"), errors="coerce")
                    rt = pd.to_numeric(
                        row.get("rate") or row.get("borrow") or row.get("fee"), errors="coerce"
                    )
                    if pd.notna(dt) and pd.notna(rt):
                        dates.append(pd.Timestamp(dt).normalize())
                        rates.append(float(rt))
                if dates:
                    out[key] = pd.Series(rates, index=pd.DatetimeIndex(dates)).sort_index()
            elif isinstance(payload, dict):
                s = pd.Series({pd.Timestamp(k).normalize(): float(v) for k, v in payload.items()})
                out[key] = s.sort_index()
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Archived plan I/O (Phase A)
# ---------------------------------------------------------------------------
def list_archived_plan_dates() -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    if not RUNS_DIR.exists():
        return out
    for p in RUNS_DIR.iterdir():
        if not p.is_dir() or not (p / "proposed_trades.csv").is_file():
            continue
        try:
            out.append(pd.Timestamp(p.name).normalize())
        except Exception:
            continue
    return sorted(out)


def list_archived_screened_dates() -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    if not RUNS_DIR.exists():
        return out
    for p in RUNS_DIR.iterdir():
        if not p.is_dir() or not (p / "etf_screened_today.csv").is_file():
            continue
        try:
            out.append(pd.Timestamp(p.name).normalize())
        except Exception:
            continue
    return sorted(out)


def _sleeve_evidence(row: pd.Series) -> str | None:
    """Best sleeve guess from bucket / product_class / is_yieldboost (ignores sleeve)."""
    if "bucket" in row.index and pd.notna(row.get("bucket")):
        b = str(row["bucket"]).strip().lower()
        if b and b not in {"nan", "none", "null"}:
            mapped = SLEEVE_ALIASES.get(b)
            if mapped:
                return mapped

    for col in ("product_class", "Delta_product_class"):
        if col not in row.index or not pd.notna(row.get(col)):
            continue
        pc = str(row[col]).strip().lower()
        if "yieldboost" in pc or "yield_boost" in pc or pc in {"whitelist_stock", "yieldboost"}:
            return "yieldboost"
        if "volatility" in pc or "vol_etp" in pc:
            return B5_SLEEVE
        if "inverse" in pc:
            return B4_SLEEVE

    yb = row.get("is_yieldboost")
    if yb is True or (isinstance(yb, str) and yb.strip().lower() in {"1", "true", "yes"}):
        return "yieldboost"
    if pd.notna(yb) and not isinstance(yb, str):
        try:
            if bool(yb) and float(yb) != 0.0:
                return "yieldboost"
        except (TypeError, ValueError):
            pass
    return None


def _infer_sleeve(row: pd.Series) -> str:
    """Map a plan row to a canonical sleeve name.

    Archived / purgatory rows often leave ``sleeve`` blank (or wrongly defaulted
    to ``core_leveraged``) while still carrying ``bucket`` / ``product_class`` /
    ``is_yieldboost``. Prefer that evidence over a conflicting core stamp so
    AMYY/MUYY/TMYY stay in yieldboost.
    """
    evidence = _sleeve_evidence(row)

    explicit: str | None = None
    if "sleeve" in row.index and pd.notna(row.get("sleeve")):
        s = str(row["sleeve"]).strip().lower()
        if s and s not in {"nan", "none", "null"}:
            explicit = SLEEVE_ALIASES.get(s, s)

    # Never let a bare core default override clear yieldboost / B4 / B5 evidence.
    if evidence in {B4_SLEEVE, B5_SLEEVE, "yieldboost"}:
        if explicit in (None, "core_leveraged"):
            return evidence
        if evidence == "yieldboost" and explicit == "core_leveraged":
            return "yieldboost"

    if explicit:
        return explicit
    if evidence:
        return evidence

    delta = pd.to_numeric(row.get("Delta", row.get("Beta", np.nan)), errors="coerce")
    if np.isfinite(delta) and delta < 0:
        return "inverse_decay_bucket4"
    return "core_leveraged"


def normalize_plan(df: pd.DataFrame, *, source_date: str | None = None) -> pd.DataFrame:
    """Normalize heterogeneous archived proposed_trades schemas."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "ETF" not in out.columns or "Underlying" not in out.columns:
        return pd.DataFrame()
    out["ETF"] = out["ETF"].map(_norm)
    out["Underlying"] = out["Underlying"].map(_norm)
    if "Delta" not in out.columns and "Beta" in out.columns:
        out["Delta"] = out["Beta"]
    out["Delta"] = pd.to_numeric(out.get("Delta"), errors="coerce")
    out["long_usd"] = pd.to_numeric(out.get("long_usd"), errors="coerce").fillna(0.0)
    out["short_usd"] = pd.to_numeric(out.get("short_usd"), errors="coerce").fillna(0.0)
    for col in ("underlying_target_usd", "etf_target_usd"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "gross_target_usd" in out.columns:
        out["gross_target_usd"] = pd.to_numeric(out["gross_target_usd"], errors="coerce").fillna(0.0)
    else:
        out["gross_target_usd"] = out["long_usd"].abs() + out["short_usd"].abs()
    out["sleeve"] = out.apply(_infer_sleeve, axis=1)
    if "borrow_current" in out.columns:
        out["borrow_current"] = pd.to_numeric(out["borrow_current"], errors="coerce").fillna(0.0)
    else:
        out["borrow_current"] = 0.0
    for col in ("borrow_underlying", "borrow_b", "underlying_borrow_annual"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "borrow_underlying" not in out.columns:
        if "underlying_borrow_annual" in out.columns:
            out["borrow_underlying"] = out["underlying_borrow_annual"]
        elif "borrow_b" in out.columns:
            out["borrow_underlying"] = out["borrow_b"]
    # Preserve purgatory rows for holdings-aware reduce-only execution.
    if "purgatory" in out.columns:
        purg = out["purgatory"]
        if purg.dtype == object:
            purg = purg.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
        else:
            purg = purg.fillna(False).astype(bool)
    else:
        purg = pd.Series(False, index=out.index)
    if "execution_policy" not in out.columns:
        try:
            purgatory_mode = str(
                (((load_config().get("portfolio") or {}).get("rebalance") or {}).get(
                    "purgatory_execution", "reduce_only"
                ))
            ).strip().lower()
        except Exception:
            purgatory_mode = "reduce_only"
        if purgatory_mode not in {"reduce_only", "hold"}:
            purgatory_mode = "reduce_only"
        out["execution_policy"] = np.where(purg, purgatory_mode, "normal")
    policy = out["execution_policy"].fillna("normal").astype(str).str.lower()
    out["execution_policy"] = policy
    out["reduce_only"] = purg & policy.eq("reduce_only")
    out["keep_open"] = purg & policy.eq("hold")
    out["hard_exit"] = policy.eq("hard_exit")

    # Persist / backfill model_* for reduce-only execution. Prefer optimal_*;
    # do NOT copy executable 0 onto model for purgatory (that falsely signals exit).
    def _series(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    opt_g = _series("optimal_gross_target_usd")
    opt_l = _series("optimal_long_usd")
    opt_s = _series("optimal_short_usd")
    mod_g = _series("model_gross_target_usd")
    mod_l = _series("model_long_usd")
    mod_s = _series("model_short_usd")
    # Fill holes from optimal_* when model_* missing.
    mod_g = mod_g.where(np.isfinite(mod_g), opt_g)
    mod_l = mod_l.where(np.isfinite(mod_l), opt_l)
    mod_s = mod_s.where(np.isfinite(mod_s), opt_s)
    # Non-purgatory: if still missing, fall back to executable legs.
    exe_g = pd.to_numeric(out["gross_target_usd"], errors="coerce").fillna(0.0)
    exe_l = pd.to_numeric(out["long_usd"], errors="coerce").fillna(0.0)
    exe_s = pd.to_numeric(out["short_usd"], errors="coerce").fillna(0.0)
    non_purg = ~purg
    mod_g = mod_g.where(purg | np.isfinite(mod_g), exe_g)
    mod_l = mod_l.where(purg | np.isfinite(mod_l), exe_l)
    mod_s = mod_s.where(purg | np.isfinite(mod_s), exe_s)
    out["model_gross_target_usd"] = mod_g
    out["model_long_usd"] = mod_l
    out["model_short_usd"] = mod_s

    # Ratchet execution flags (missing → conservative grow-only).
    if "ratchet_released" not in out.columns:
        out["ratchet_released"] = False
    else:
        out["ratchet_released"] = out["ratchet_released"].map(_truthy)
    if "ratchet_trim_usd" not in out.columns:
        out["ratchet_trim_usd"] = 0.0
    else:
        out["ratchet_trim_usd"] = pd.to_numeric(
            out["ratchet_trim_usd"], errors="coerce"
        ).fillna(0.0)

    model_gross = pd.to_numeric(out["model_gross_target_usd"], errors="coerce").fillna(0.0)
    out = out[
        (out["gross_target_usd"] > 0)
        | (model_gross > 0)
        | out["reduce_only"]
        | out["keep_open"]
        | out["hard_exit"]
    ].copy()
    if source_date is not None:
        out["_plan_date"] = source_date
    return out.reset_index(drop=True)


def load_plan_file(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return normalize_plan(raw, source_date=path.parent.name)


def proposed_trades_full_asof(as_of: pd.Timestamp | str) -> pd.DataFrame | None:
    """Nearest archived full-book plan on/before ``as_of`` (all sleeves)."""
    as_of = pd.Timestamp(as_of).normalize()
    best: tuple[pd.Timestamp, Path] | None = None
    for d in list_archived_plan_dates():
        if d <= as_of:
            best = (d, RUNS_DIR / d.strftime("%Y-%m-%d") / "proposed_trades.csv")
    if best is None:
        return None
    plan = load_plan_file(best[1])
    return plan if not plan.empty else None


def load_plan_timeline(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """All normalized archived plans with date in [start, end]."""
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for d in list_archived_plan_dates():
        if d < start:
            continue
        if end is not None and d > end:
            continue
        plan = load_plan_file(RUNS_DIR / d.strftime("%Y-%m-%d") / "proposed_trades.csv")
        if not plan.empty:
            out[d] = plan
    return out


def b4_timeline_universe(
    timeline: Mapping[pd.Timestamp, pd.DataFrame],
) -> tuple[dict[str, str], dict[str, pd.Timestamp]]:
    """Return every B4 wrapper/underlying ever present in the PIT plan timeline.

    This deliberately includes purgatory and zero-executable rows: an
    incumbent can still need to be marked or trimmed even when it is not a
    fresh executable entry.
    """
    underlyings: dict[str, str] = {}
    first_seen: dict[str, pd.Timestamp] = {}
    for plan_date, plan in sorted(timeline.items()):
        if plan is None or plan.empty:
            continue
        rows = plan.loc[plan.get("sleeve", pd.Series("", index=plan.index)).astype(str).eq(B4_SLEEVE)]
        for _, row in rows.iterrows():
            etf = _norm(row.get("ETF", ""))
            und = _norm(row.get("Underlying", ""))
            if not etf or not und:
                continue
            underlyings[etf] = und
            first_seen.setdefault(etf, pd.Timestamp(plan_date).normalize())
    return underlyings, first_seen


def load_timeline_price_panel(
    timeline: Mapping[pd.Timestamp, pd.DataFrame],
    *,
    run_date: str,
    min_days: int = 40,
) -> dict[str, pd.DataFrame]:
    """Load a listing-aware panel for the full point-in-time B4 universe."""
    underlyings, first_seen = b4_timeline_universe(timeline)
    if not underlyings:
        return load_price_panel(run_date, min_days=min_days)

    source_dates = {pd.Timestamp(run_date).normalize(), *first_seen.values()}
    frames: list[pd.DataFrame] = []
    cols = ["date", "ticker", "etf_adj_close", "underlying_adj_close"]
    for as_of in sorted(source_dates):
        path = RUNS_DIR / as_of.strftime("%Y-%m-%d") / "model_inputs" / "etf_metrics_daily.parquet"
        if not path.is_file():
            continue
        try:
            frame = pd.read_parquet(path, columns=cols)
        except Exception:
            continue
        frame["_panel_source_date"] = as_of
        frames.append(frame)
    if not frames:
        return load_price_panel(run_date, min_days=min_days, underlying_by_etf=underlyings)

    metrics = pd.concat(frames, ignore_index=True)
    metrics["date"] = pd.to_datetime(metrics["date"], errors="coerce").dt.normalize()
    metrics["ticker"] = metrics["ticker"].map(_norm)
    metrics = metrics.dropna(subset=["date", "ticker"]).sort_values(
        ["date", "ticker", "_panel_source_date"]
    ).drop_duplicates(["date", "ticker"], keep="last")
    panel = frames_from_metrics(
        metrics,
        min_days=min_days,
        underlying_by_etf=underlyings,
        min_days_by_etf={etf: MIN_B4_LISTING_PRICE_DAYS for etf in underlyings},
        repo=REPO,
    )
    # Point-in-time run snapshots are the production source of truth here.
    # Do not Yahoo-extend every historical wrapper to the terminal date: a
    # delisted wrapper can otherwise trigger repeated external retries and turn
    # a historical mark into a synthetic post-life series.
    panel = apply_panel_leg_patches(panel, underlying_by_etf=underlyings, repo=REPO)
    return apply_delist_cutoff(panel, repo=REPO)


def archive_coverage_summary(start: str = "2025-05-01") -> pd.DataFrame:
    start_ts = pd.Timestamp(start).normalize()
    plans = list_archived_plan_dates()
    screened = list_archived_screened_dates()
    first_plan = plans[0] if plans else None
    rows = [
        {
            "artifact": "proposed_trades.csv",
            "n_dates": len(plans),
            "first": str(plans[0].date()) if plans else None,
            "last": str(plans[-1].date()) if plans else None,
            "n_on_or_after_start": sum(1 for d in plans if d >= start_ts),
            "gap_before_first_plan_days": (
                int((first_plan - start_ts).days) if first_plan is not None and first_plan > start_ts else 0
            ),
        },
        {
            "artifact": "etf_screened_today.csv",
            "n_dates": len(screened),
            "first": str(screened[0].date()) if screened else None,
            "last": str(screened[-1].date()) if screened else None,
            "n_on_or_after_start": sum(1 for d in screened if d >= start_ts),
            "gap_before_first_plan_days": (
                int((screened[0] - start_ts).days) if screened and screened[0] > start_ts else 0
            ),
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Frozen-mode sleeve engines (unchanged semantics)
# ---------------------------------------------------------------------------
def _pair_stats_from_navs(
    pair_navs: dict[str, pd.Series],
    *,
    sleeve: str,
    und_by_etf: dict[str, str] | None = None,
    start_usd_by_etf: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-pair contribution table from equity series (first live → end)."""
    rows = []
    und_by_etf = und_by_etf or {}
    start_usd_by_etf = start_usd_by_etf or {}
    for etf, nav in pair_navs.items():
        s = nav.dropna().astype(float)
        # Drop leading zeros/NaNs from staggered listings (do not treat as wipeouts).
        live = s[s > 1e-9]
        if len(live) < 2:
            continue
        start_usd = float(start_usd_by_etf.get(etf, live.iloc[0]))
        if start_usd <= 0:
            start_usd = float(live.iloc[0])
        end_usd = float(live.iloc[-1])
        scale = start_usd / float(live.iloc[0]) if float(live.iloc[0]) > 0 else 1.0
        end_usd = end_usd * scale
        m = perf(live)
        ret = (end_usd / start_usd - 1.0) if start_usd > 0 else np.nan
        cagr = m.get("cagr")
        corrupt = bool(
            np.isfinite(ret)
            and ret <= -0.999
            and cagr is not None
            and np.isfinite(cagr)
            and float(cagr) > 0
        )
        rows.append(
            {
                "sleeve": sleeve,
                "ETF": etf,
                "Underlying": und_by_etf.get(etf, ""),
                "start_usd": start_usd,
                "end_usd": end_usd,
                "pnl_usd": end_usd - start_usd,
                "ret": ret,
                "cagr": cagr,
                "maxdd": m.get("maxdd"),
                "vol": m.get("vol"),
                "sharpe": m.get("sharpe"),
                "n_days": int(len(live)),
                "first_trade_date": str(live.index[0].date()),
                "last_trade_date": str(live.index[-1].date()),
                "stats_corrupt": corrupt,
            }
        )
    return pd.DataFrame(rows)


def _stock_sleeve_nav(
    uni: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    *,
    sleeve: str,
    start: pd.Timestamp,
    budget_usd: float,
    enter_band_pct: float,
    slippage_bps: float,
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """B1/B2: production gross weights; weekly retarget to plan weights.

    Inactive names (no price yet / missing Friday) keep prior pair equity and
    contribute cash (0 return) to the sleeve until they are live. Pair stats
    compound from each name's first valid return date — never zeroed on rebal.
    """
    del enter_band_pct
    empty_stats = pd.DataFrame()
    df = uni[uni["sleeve"] == sleeve].copy()
    missing_panel = [e for e in df["ETF"].astype(str) if e not in panel]
    df = df[df["ETF"].isin(panel)].reset_index(drop=True)
    if df.empty or budget_usd <= 0:
        return (
            pd.Series(dtype=float),
            {
                "sleeve": sleeve,
                "n_pairs": 0,
                "skipped": True,
                "skip_reasons": "; ".join(f"{e}:not_in_panel" for e in missing_panel[:12]),
            },
            empty_stats,
        )

    daily_ret: dict[str, pd.Series] = {}
    und_by_etf: dict[str, str] = {}
    skipped: list[str] = [f"{e}:not_in_panel" for e in missing_panel]
    for _, row in df.iterrows():
        etf = row["ETF"]
        gross = abs(float(row["long_usd"])) + abs(float(row["short_usd"]))
        if gross <= 0:
            skipped.append(f"{etf}:zero_gross")
            continue
        # Price panel a=ETF, b=underlying.  proposed_trades stores ETF in
        # short_usd/etf_target_usd and underlying in
        # long_usd/underlying_target_usd.
        w_a = _float_or(row.get("etf_target_usd"), _float_or(row.get("short_usd"))) / gross
        w_b = _float_or(row.get("underlying_target_usd"), _float_or(row.get("long_usd"))) / gross
        px = panel[etf]
        ret = w_a * px["a_px"].pct_change() + w_b * px["b_px"].pct_change()
        borrow = _float_or(row.get("borrow_current"))
        borrow_b = _float_or(row.get("borrow_underlying"), _float_or(row.get("underlying_borrow_annual")))
        if w_a < 0 and borrow > 0:
            ret = ret - borrow * abs(w_a) / TRADING_DAYS
        if w_b < 0 and borrow_b > 0:
            ret = ret - borrow_b * abs(w_b) / TRADING_DAYS
        if len(ret.dropna()) < 20:
            skipped.append(f"{etf}:short_ret={len(ret.dropna())}")
            continue
        daily_ret[etf] = ret
        und_by_etf[etf] = str(row.get("Underlying", "") or "")
    df = df[df["ETF"].isin(daily_ret)].reset_index(drop=True)
    if df.empty:
        return (
            pd.Series(dtype=float),
            {"sleeve": sleeve, "n_pairs": 0, "skipped": True, "skip_reasons": "; ".join(skipped[:12])},
            empty_stats,
        )

    gross = pd.to_numeric(df["gross_target_usd"], errors="coerce").fillna(0.0).to_numpy()
    plan_deployed = float(gross.sum())
    sleeve_capital = min(float(budget_usd), plan_deployed) if plan_deployed > 0 else float(budget_usd)
    if plan_deployed <= 0:
        w0 = np.ones(len(df)) / len(df)
    else:
        w0 = gross / plan_deployed
    target_w = pd.Series(w0, index=df["ETF"].to_numpy())
    start_usd_by_etf = {e: float(target_w.loc[e]) * sleeve_capital for e in target_w.index}

    all_idx = sorted(set().union(*[set(s.dropna().index) for s in daily_ret.values()]))
    cal = pd.DatetimeIndex([d for d in all_idx if d >= start])
    if len(cal) < 40:
        return (
            pd.Series(dtype=float),
            {"sleeve": sleeve, "n_pairs": 0, "skipped": True, "reason": "short_calendar"},
            empty_stats,
        )

    check_days = set(
        pd.DatetimeIndex(pd.Series(1, index=cal).resample("W-FRI").last().index).intersection(cal)
    )

    R = pd.DataFrame({e: daily_ret[e].reindex(cal) for e in target_w.index})
    nav = pd.Series(index=cal, dtype=float)
    equity = float(sleeve_capital)
    cur_w = pd.Series(0.0, index=target_w.index)
    pair_eq = {e: 0.0 for e in target_w.index}
    pair_started = {e: False for e in target_w.index}
    pair_nav_paths: dict[str, pd.Series] = {e: pd.Series(index=cal, dtype=float) for e in target_w.index}
    n_rebal = 0
    turnover = 0.0
    slip = slippage_bps / 1e4
    for i, d in enumerate(cal):
        if i == 0 or d in check_days:
            active = [e for e in target_w.index if np.isfinite(R.at[d, e])]
            if active:
                # Keep absolute plan targets among live names; residual stays cash.
                new_w = pd.Series(0.0, index=target_w.index)
                new_w.loc[active] = target_w.reindex(active).fillna(0.0).to_numpy()
                # Cap sum at 1.0 (numerical)
                if float(new_w.sum()) > 1.0:
                    new_w = new_w / float(new_w.sum())
                turnover += float(np.abs(new_w - cur_w).sum())
                if i > 0 and cur_w.sum() > 0:
                    equity *= 1.0 - slip * float(np.abs(new_w - cur_w).sum()) / 2.0
                cur_w = new_w
                n_rebal += 1
                # Seed pair equity at inception (do NOT redistribute inactive → 0).
                for e in active:
                    if not pair_started[e]:
                        pair_eq[e] = float(start_usd_by_etf[e])
                        pair_started[e] = True

        for e in cur_w.index:
            r_e = R.at[d, e]
            if pair_started[e] and np.isfinite(r_e):
                pair_eq[e] *= 1.0 + float(r_e)
            pair_nav_paths[e].loc[d] = pair_eq[e] if pair_started[e] else np.nan

        r = float(np.nansum(cur_w.to_numpy() * np.nan_to_num(R.loc[d].to_numpy())))
        equity *= 1.0 + r
        nav.iloc[i] = equity

    stats = _pair_stats_from_navs(
        pair_nav_paths,
        sleeve=sleeve,
        und_by_etf=und_by_etf,
        start_usd_by_etf=start_usd_by_etf,
    )
    # Overlay end_usd from live pair_eq for reconciliation with sleeve path
    if not stats.empty:
        for e, pe in pair_eq.items():
            if pair_started[e] and e in set(stats["ETF"]):
                stats.loc[stats["ETF"] == e, "end_usd"] = float(pe)
                su = float(start_usd_by_etf[e])
                stats.loc[stats["ETF"] == e, "pnl_usd"] = float(pe) - su
                stats.loc[stats["ETF"] == e, "ret"] = (float(pe) / su - 1.0) if su > 0 else np.nan

    meta = {
        "sleeve": sleeve,
        "n_pairs": int(len(df)),
        "n_skipped": len(skipped),
        "skip_reasons": "; ".join(skipped[:20]),
        "n_rebal": n_rebal,
        "turnover_l1": turnover,
        "start_usd": sleeve_capital,
        "yaml_budget_usd": float(budget_usd),
        "plan_deployed_usd": plan_deployed,
        "end_usd": float(nav.iloc[-1]),
        "engine": "pair_daily_returns + weekly retarget (cash for inactive; no pair_eq wipe)",
        **perf(nav),
    }
    return nav, meta, stats


def _b4_family_nav(
    uni: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    *,
    sleeve: str,
    start: pd.Timestamp,
    budget_usd: float,
    slippage_bps: float,
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """B4: production v7 hedge + TR/VCR cadence; weights = plan gross.

    Skipped names' gross stays cash (does not dilute survivors). Pair stats
    use each pair's own first live equity date.
    """
    empty_stats = pd.DataFrame()
    df_all = uni[uni["sleeve"] == sleeve].copy()
    skipped: list[str] = []
    for etf in df_all["ETF"].astype(str):
        if etf not in panel:
            skipped.append(f"{etf}:not_in_panel")
    df = df_all[df_all["ETF"].isin(panel)].reset_index(drop=True)
    if df.empty or budget_usd <= 0:
        return (
            pd.Series(dtype=float),
            {"sleeve": sleeve, "n_pairs": 0, "skipped": True, "skip_reasons": "; ".join(skipped[:20])},
            empty_stats,
        )

    blk = knobs_from_yaml()
    knobs = make_knobs(blk)

    # Pass 1: admit pairs
    admitted_rows: list[tuple[pd.Series, pd.DatetimeIndex]] = []
    for _, row in df.iterrows():
        etf, und = row["ETF"], row["Underlying"]
        px = panel.get(etf)
        if px is None:
            skipped.append(f"{etf}:no_panel")
            continue
        if len(px) < MIN_B4_SESSIONS:
            skipped.append(f"{etf}:short_hist_total={len(px)}")
            continue
        sim_start = max(pd.Timestamp(start), pd.Timestamp(px.index.min()))
        cal = pd.DatetimeIndex([d for d in px.index if d >= sim_start])
        if len(cal) < MIN_B4_TRADE_DAYS:
            skipped.append(f"{etf}:short_trade_days={len(cal)}")
            continue
        admitted_rows.append((row, cal))

    if not admitted_rows:
        return (
            pd.Series(dtype=float),
            {"sleeve": sleeve, "n_pairs": 0, "skipped": True, "skip_reasons": "; ".join(skipped[:20])},
            empty_stats,
        )

    # Pass 2: allocate notionals only among admitted pairs (skipped gross = cash)
    adm_df = pd.DataFrame([r for r, _ in admitted_rows])
    gross = pd.to_numeric(adm_df["gross_target_usd"], errors="coerce").fillna(0.0).to_numpy()
    plan_deployed_all = float(pd.to_numeric(df_all["gross_target_usd"], errors="coerce").fillna(0.0).sum())
    admitted_gross = float(gross.sum())
    sleeve_capital = min(float(budget_usd), admitted_gross) if admitted_gross > 0 else 0.0
    if admitted_gross <= 0:
        notionals = np.zeros(len(adm_df))
    else:
        notionals = gross * (sleeve_capital / admitted_gross)

    pair_navs: list[pd.Series] = []
    und_by_etf: dict[str, str] = {}
    start_usd_by_etf: dict[str, float] = {}
    for (row, cal), start_notional in zip(admitted_rows, notionals):
        etf, und = row["ETF"], row["Underlying"]
        px = panel[etf]
        start_notional = float(start_notional)
        if start_notional <= 0:
            skipped.append(f"{etf}:zero_notional")
            continue
        try:
            warmup = min(60, max(0, len(cal) - MIN_B4_TRADE_DAYS))
            sig = get_pair_signal(
                etf,
                und,
                cal,
                history={},
                underlying_prices=px["b_px"],
                window=60,
                lookahead_shift=1,
            )
            h_daily = build_h_series(sig, cal, knobs=knobs)
            rb, _ = build_rebal_dates(sig, cal, knobs=knobs, warmup_bdays=warmup)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{etf}:{type(exc).__name__}")
            continue
        borrow = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
        beta = float(pd.to_numeric(row.get("Delta"), errors="coerce") or -1.0)
        beta_a = -abs(beta) if abs(beta) > 1e-6 else -1.0
        bt = run_bucket4_backtest_dynamic_h(
            px.reindex(cal),
            h_daily,
            rb,
            initial_capital=start_notional,
            beta_a=beta_a,
            beta_b=1.0,
            borrow_a_annual=borrow,
            slippage_bps=slippage_bps,
        )
        eq = bt["equity"].astype(float)
        eq = eq.dropna()
        if eq.empty or float(eq.iloc[0]) <= 0:
            skipped.append(f"{etf}:empty_equity")
            continue
        # Floor at 0 for path stability but keep pre-clip diagnostic in meta later
        eq = eq.clip(lower=0.0)
        if float(eq.iloc[0]) <= 0:
            skipped.append(f"{etf}:wiped_at_start")
            continue
        eq = eq * (start_notional / float(eq.iloc[0]))
        pair_navs.append(eq.rename(etf))
        und_by_etf[etf] = str(und)
        start_usd_by_etf[etf] = start_notional

    if not pair_navs:
        return pd.Series(dtype=float), {
            "sleeve": sleeve,
            "n_pairs": 0,
            "skipped": True,
            "skip_reasons": "; ".join(skipped[:20]),
        }, empty_stats

    # Sleeve NAV: sum pair paths (absent before listing = 0 contribution that day)
    wide = pd.concat(pair_navs, axis=1, sort=True)
    port = wide.fillna(0.0).sum(axis=1)
    # Hold undeployed (skipped) capital as flat cash so start matches admitted capital
    # (already sized to admitted only). Optionally pad to YAML with cash:
    cash_pad = max(0.0, min(float(budget_usd), plan_deployed_all) - sleeve_capital)
    if cash_pad > 0:
        port = port + cash_pad

    stats = _pair_stats_from_navs(
        {s.name: s for s in pair_navs},
        sleeve=sleeve,
        und_by_etf=und_by_etf,
        start_usd_by_etf=start_usd_by_etf,
    )

    meta = {
        "sleeve": sleeve,
        "n_pairs": int(len(pair_navs)),
        "n_skipped": len(skipped),
        "skip_reasons": "; ".join(skipped[:20]),
        "start_usd": float(port.iloc[0]) if len(port) else sleeve_capital,
        "yaml_budget_usd": float(budget_usd),
        "plan_deployed_usd": plan_deployed_all,
        "admitted_gross_usd": admitted_gross,
        "end_usd": float(port.iloc[-1]) if len(port) else np.nan,
        "engine": "bucket4_dynamic_bt + production cadence/v7 + admitted-only notional",
        "cadence_base_days": knobs.base_days,
        "cadence_k_tr": knobs.k_tr,
        **perf(port),
    }
    return port, meta, stats


def _b5_sleeve_nav(
    uni: pd.DataFrame,
    *,
    start: pd.Timestamp,
    budget_usd: float,
    slippage_bps: float,
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """B5: short UVIX / short SVIX carry via bucket5_carry_bt (not B4 dynamic-h)."""
    empty_stats = pd.DataFrame()
    df = uni[uni["sleeve"] == B5_SLEEVE].copy()
    if df.empty or budget_usd <= 0:
        return (
            pd.Series(dtype=float),
            {"sleeve": B5_SLEEVE, "n_pairs": 0, "skipped": True, "reason": "empty"},
            empty_stats,
        )

    from scripts.bucket5_carry_bt import run_carry_backtest, static_rho
    from scripts.bucket5_data import (
        DEFAULT_BORROW_SVIX,
        DEFAULT_BORROW_UVIX,
        load_vol_panel,
        rebalance_dates,
    )

    row = df.iloc[0]
    etf = str(row.get("ETF", "UVIX"))
    und = str(row.get("Underlying", "SVIX"))
    underlying_usd = abs(_float_or(row.get("underlying_target_usd"), _float_or(row.get("long_usd"))))
    etf_usd = abs(_float_or(row.get("etf_target_usd"), _float_or(row.get("short_usd"))))
    gross = _float_or(row.get("gross_target_usd"), underlying_usd + etf_usd)
    # Plan convention: ETF=UVIX, Underlying=SVIX; rho = SVIX / UVIX.
    uvix_usd = etf_usd if etf.upper() == "UVIX" else underlying_usd
    svix_usd = underlying_usd if und.upper() == "SVIX" else etf_usd
    if uvix_usd <= 0:
        uvix_usd = gross / 2.0
    rho = float(svix_usd / uvix_usd) if uvix_usd > 0 else 0.5

    borrow_u = _float_or(row.get("borrow_current"), DEFAULT_BORROW_UVIX)
    borrow_s = _float_or(
        row.get("borrow_underlying"),
        _float_or(row.get("borrow_b"), _float_or(row.get("underlying_borrow_annual"), DEFAULT_BORROW_SVIX)),
    )

    sleeve_capital = min(float(budget_usd), gross) if gross > 0 else float(budget_usd)
    try:
        vol = load_vol_panel(start=str(pd.Timestamp(start).date()), use_synthetic=True)
    except Exception as exc:  # noqa: BLE001
        return (
            pd.Series(dtype=float),
            {"sleeve": B5_SLEEVE, "n_pairs": 0, "skipped": True, "reason": f"vol_panel:{type(exc).__name__}"},
            empty_stats,
        )
    vol = vol.loc[vol.index >= pd.Timestamp(start)].dropna(subset=["uvix", "svix"])
    if len(vol) < 40:
        return (
            pd.Series(dtype=float),
            {"sleeve": B5_SLEEVE, "n_pairs": 0, "skipped": True, "reason": f"short_vol_panel={len(vol)}"},
            empty_stats,
        )

    rb = rebalance_dates(vol.index, freq="W-FRI")
    bt = run_carry_backtest(
        vol,
        static_rho(vol.index, rho),
        rb,
        initial_capital=sleeve_capital,
        gross_multiplier=1.0,
        borrow_uvix_annual=borrow_u if borrow_u > 0 else DEFAULT_BORROW_UVIX,
        borrow_svix_annual=borrow_s if borrow_s > 0 else DEFAULT_BORROW_SVIX,
        slippage_bps=slippage_bps,
    )
    eq = bt["equity"].astype(float)
    min_eq = float(eq.min()) if len(eq) else np.nan
    # Report true path; only floor display NAV at 0 for book combine stability
    eq_floor = eq.clip(lower=0.0)
    port = eq_floor.rename(B5_SLEEVE)
    if float(port.iloc[0]) > 0:
        port = port * (sleeve_capital / float(port.iloc[0]))

    stats = _pair_stats_from_navs(
        {etf: port},
        sleeve=B5_SLEEVE,
        und_by_etf={etf: und},
        start_usd_by_etf={etf: sleeve_capital},
    )
    meta = {
        "sleeve": B5_SLEEVE,
        "n_pairs": 1,
        "n_skipped": 0,
        "start_usd": sleeve_capital,
        "yaml_budget_usd": float(budget_usd),
        "plan_deployed_usd": gross,
        "end_usd": float(port.iloc[-1]) if len(port) else np.nan,
        "min_equity_pre_floor": min_eq,
        "rho": rho,
        "engine": "bucket5_carry_bt short-UVIX/short-SVIX (plan rho)",
        **perf(port),
    }
    return port, meta, stats


def combine_book_nav(
    sleeve_navs: dict[str, pd.Series],
    *,
    capital_usd: float,
    budgets: dict[str, float],
) -> pd.Series:
    """Sum sleeve NAVs and rescale the book to ``capital_usd``.

    Each sleeve keeps its own starting notional (plan-deployed gross, which may
    be below the YAML budget while entry-ramp / soft-exit cash is outstanding).
    We do **not** re-inflate under-deployed sleeves to their YAML budget.
    """
    if not sleeve_navs:
        return pd.Series(dtype=float)
    idx = None
    for s in sleeve_navs.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = pd.DatetimeIndex(sorted(idx))
    cols = {}
    for name, s in sleeve_navs.items():
        # Fallback fill uses the sleeve's own first value (or YAML budget if empty)
        b = float(budgets.get(name, 0.0) or 0.0)
        ser = s.reindex(idx)
        first = ser.first_valid_index()
        if first is None:
            cols[name] = pd.Series(0.0, index=idx)
            continue
        start_v = float(ser.loc[first])
        fill = start_v if start_v > 0 else b
        ser = ser.copy()
        ser.loc[:first] = ser.loc[:first].fillna(fill)
        ser = ser.ffill().fillna(fill)
        cols[name] = ser
    wide = pd.DataFrame(cols, index=idx)
    total = wide.sum(axis=1)
    if total.empty or float(total.iloc[0]) <= 0:
        return pd.Series(dtype=float)
    return capital_usd * (total / float(total.iloc[0]))


# ---------------------------------------------------------------------------
# Phase A/B: plan-timeline book simulator
# ---------------------------------------------------------------------------
def _plan_asof_from_timeline(
    timeline: dict[pd.Timestamp, pd.DataFrame],
    day: pd.Timestamp,
) -> pd.DataFrame | None:
    keys = [d for d in timeline if d <= day]
    if not keys:
        return None
    return timeline[max(keys)]


def _effective_plan_timeline(
    timeline: dict[pd.Timestamp, pd.DataFrame],
    cal: pd.DatetimeIndex,
    *,
    execution_lag_sessions: int,
) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]]:
    """Map execution close -> (source plan date, plan), without look-ahead.

    ``execution_lag_sessions=1`` means a plan stamped on T is first executable
    at the next available close.  If several weekend/run artifacts map to the
    same close, the latest source artifact wins.
    """
    lag = int(execution_lag_sessions)
    if lag < 0:
        raise ValueError("execution_lag_sessions must be >= 0")
    out: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]] = {}
    for plan_date in sorted(timeline):
        eligible = cal[cal >= plan_date] if lag == 0 else cal[cal > plan_date]
        offset = 0 if lag <= 1 else lag - 1
        if len(eligible) <= offset:
            continue
        out[pd.Timestamp(eligible[offset])] = (plan_date, timeline[plan_date])
    return out


def _resize_band_target(
    current_usd: float,
    target_usd: float,
    *,
    enter_band_pct: float,
    exit_band_pct: float,
    min_trade_usd: float,
) -> float:
    """Phase-2b hysteresis target for one signed leg.

    New entries establish at target and exits close fully.  Existing same-sign
    legs trade only after crossing the enter band, then move back to the exit
    band instead of all the way to target.  This mirrors ``phase2b_resize`` and
    prevents the replay from paying for a fictitious full weekly retarget.
    """
    cur = float(current_usd)
    tgt = float(target_usd)
    floor = max(float(min_trade_usd), 0.0)
    if abs(cur) <= 1e-9:
        return tgt if abs(tgt) >= floor else 0.0
    if abs(tgt) <= 1e-9:
        return 0.0
    if np.sign(cur) != np.sign(tgt):
        return tgt
    abs_cur = abs(cur)
    abs_tgt = abs(tgt)
    abs_drift = abs_cur - abs_tgt
    enter_thr = max(max(float(enter_band_pct), 0.0) * abs_tgt, floor)
    exit_thr = max(max(float(exit_band_pct), 0.0) * abs_tgt, floor)
    if abs(abs_drift) <= enter_thr:
        return cur
    trade_abs = max(0.0, abs(abs_drift) - exit_thr)
    if trade_abs < floor:
        return cur
    new_abs = abs_cur - trade_abs if abs_drift > 0 else abs_cur + trade_abs
    return float(np.sign(tgt) * max(new_abs, 0.0))


def _pace_leg(
    current_usd: float,
    desired_usd: float,
    *,
    max_leg_step_pct: float,
    min_trade_usd: float,
) -> float:
    """Move at most ``max_leg_step_pct`` of max(|cur|, |desired|) toward desired."""
    cur = float(current_usd)
    des = float(desired_usd)
    step = des - cur
    if abs(step) <= 1e-9:
        return cur
    pct = max(float(max_leg_step_pct), 0.0)
    if pct <= 0:
        return cur
    max_abs = pct * max(abs(cur), abs(des), max(float(min_trade_usd), 0.0))
    if abs(step) <= max_abs + 1e-12:
        return des
    return float(cur + np.sign(step) * max_abs)


def _apply_sleeve_gross_ema(
    target: pd.DataFrame,
    sleeve_ema: dict[str, float],
    *,
    alpha: float,
    sleeves: tuple[str, ...] = STOCK_SLEEVES,
) -> pd.DataFrame:
    """EMA planned sleeve gross and rescale legs so sleeve gross equals EMA.

    Updates ``sleeve_ema`` in place. Sleeves with zero planned gross are skipped.
    """
    if target is None or target.empty:
        return target
    a = float(np.clip(alpha, 0.0, 1.0))
    out = target.copy()
    for sleeve in sleeves:
        mask = out["sleeve"].astype(str) == sleeve
        if not mask.any():
            continue
        legs = out.loc[mask, ["etf_usd", "underlying_usd"]].astype(float)
        planned = float(legs.abs().sum().sum())
        if planned <= 1e-9:
            continue
        prev = sleeve_ema.get(sleeve)
        if prev is None or not np.isfinite(prev) or prev <= 1e-9:
            ema = planned
        else:
            ema = a * planned + (1.0 - a) * float(prev)
        sleeve_ema[sleeve] = float(ema)
        if abs(ema - planned) <= 1e-6:
            continue
        scale = float(ema) / planned
        out.loc[mask, "etf_usd"] = out.loc[mask, "etf_usd"].astype(float) * scale
        out.loc[mask, "underlying_usd"] = (
            out.loc[mask, "underlying_usd"].astype(float) * scale
        )
        out.loc[mask, "gross_usd"] = (
            out.loc[mask, ["etf_usd", "underlying_usd"]].astype(float).abs().sum(axis=1)
        )
    return out


def _allocate_turnover_budget(
    fills: list[dict[str, Any]],
    *,
    budget_usd: float,
    establish_budget_frac: float = 0.50,
) -> tuple[list[dict[str, Any]], int]:
    """Accept fills under a daily |Δlegs| budget; priority 0 exits always full.

    Priority: 0=exit/delist/hard_exit, 1=establish, 2=resize.
    Establishes share at most ``establish_budget_frac`` of the budget remaining
    after priority-0 fills. Returns (accepted fills, n_deferred).
    """
    if not fills:
        return [], 0
    budget = float(budget_usd)
    if not np.isfinite(budget) or budget < 0:
        return list(fills), 0
    if budget <= 1e-9:
        # Still allow hard exits.
        accepted = [f for f in fills if int(f.get("priority", 2)) == 0]
        return accepted, max(0, len(fills) - len(accepted))

    def _turn(f: dict[str, Any]) -> float:
        return abs(float(f["new_a"]) - float(f["old_a"])) + abs(
            float(f["new_b"]) - float(f["old_b"])
        )

    exits = [f for f in fills if int(f.get("priority", 2)) == 0]
    establishes = sorted(
        [f for f in fills if int(f.get("priority", 2)) == 1],
        key=_turn,
        reverse=True,
    )
    resizes = sorted(
        [f for f in fills if int(f.get("priority", 2)) >= 2],
        key=_turn,
        reverse=True,
    )

    accepted: list[dict[str, Any]] = []
    used = 0.0
    deferred = 0

    for f in exits:
        t = _turn(f)
        accepted.append(f)
        used += t  # exits always full even if over budget

    remaining = max(0.0, budget - used)
    est_cap = remaining * max(0.0, min(float(establish_budget_frac), 1.0))
    est_used = 0.0

    def _take(f: dict[str, Any], rem: float) -> tuple[dict[str, Any] | None, float]:
        t = _turn(f)
        if t <= 1e-9:
            return f, rem
        if rem <= 1e-9:
            return None, rem
        if t <= rem + 1e-9:
            return f, rem - t
        scale = rem / t
        out = dict(f)
        oa, ob = float(f["old_a"]), float(f["old_b"])
        out["new_a"] = oa + (float(f["new_a"]) - oa) * scale
        out["new_b"] = ob + (float(f["new_b"]) - ob) * scale
        out["scaled_by_turnover_budget"] = True
        return out, 0.0

    for f in establishes:
        room = max(0.0, est_cap - est_used)
        room = min(room, max(0.0, budget - used))
        taken, _ = _take(f, room)
        if taken is None:
            deferred += 1
            continue
        t = _turn(taken)
        accepted.append(taken)
        used += t
        est_used += t

    for f in resizes:
        room = max(0.0, budget - used)
        taken, _ = _take(f, room)
        if taken is None:
            deferred += 1
            continue
        accepted.append(taken)
        used += _turn(taken)

    return accepted, deferred


def _pace_pair_atomic(
    old_a: float,
    old_b: float,
    desired_a: float,
    desired_b: float,
    *,
    pair_gross_ramp_pct: float,
    min_trade_usd: float,
) -> tuple[float, float, float]:
    """Scale both requested leg changes by one completion fraction."""
    oa, ob = float(old_a), float(old_b)
    da, db = float(desired_a), float(desired_b)
    turn = abs(da - oa) + abs(db - ob)
    if turn <= 1e-9:
        return oa, ob, 1.0
    pct = max(float(pair_gross_ramp_pct), 0.0)
    if pct <= 0.0:
        return oa, ob, 0.0
    base = max(
        abs(oa) + abs(ob),
        abs(da) + abs(db),
        max(float(min_trade_usd), 0.0),
    )
    frac = min(1.0, pct * base / turn)
    return (
        float(oa + (da - oa) * frac),
        float(ob + (db - ob) * frac),
        float(frac),
    )


def _advance_pair_atomic(
    old_a: float,
    old_b: float,
    desired_a: float,
    desired_b: float,
    *,
    completion_fraction: float,
) -> tuple[float, float]:
    """Advance both legs by exactly one shared fraction of remaining gap."""
    f = float(np.clip(completion_fraction, 0.0, 1.0))
    return (
        float(old_a + (desired_a - old_a) * f),
        float(old_b + (desired_b - old_b) * f),
    )


def _liquidity_value(row: pd.Series | dict[str, Any], names: tuple[str, ...]) -> float:
    for name in names:
        v = pd.to_numeric(row.get(name), errors="coerce")
        if pd.notna(v) and np.isfinite(float(v)) and float(v) > 0:
            return float(v)
    return np.nan


def _apply_adv_participation_cap(
    fill: dict[str, Any],
    *,
    etf_price: float,
    underlying_price: float,
    adv_participation_pct: float,
) -> tuple[dict[str, Any], str | None]:
    """Atomically cap a pair fill by available per-leg ADV; missing ADV is no-op."""
    pct = max(float(adv_participation_pct), 0.0)
    if pct <= 0:
        return dict(fill), None
    ref = fill.get("new") if fill.get("new") is not None else fill.get("old")
    if ref is None:
        return dict(fill), None
    etf_adv = _liquidity_value(ref, ("etf_adv_usd", "adv_usd"))
    und_adv = _liquidity_value(ref, ("underlying_adv_usd",))
    if not np.isfinite(etf_adv):
        sh = _liquidity_value(
            ref, ("etf_adv_shares", "median_daily_volume_shares", "adv_median_shares")
        )
        if np.isfinite(sh) and etf_price > 0:
            etf_adv = sh * float(etf_price)
    if not np.isfinite(und_adv):
        sh = _liquidity_value(ref, ("underlying_adv_shares",))
        if np.isfinite(sh) and underlying_price > 0:
            und_adv = sh * float(underlying_price)
    da = abs(float(fill["new_a"]) - float(fill["old_a"]))
    db = abs(float(fill["new_b"]) - float(fill["old_b"]))
    fractions = [1.0]
    if da > 1e-9 and np.isfinite(etf_adv):
        fractions.append(pct * etf_adv / da)
    if db > 1e-9 and np.isfinite(und_adv):
        fractions.append(pct * und_adv / db)
    frac = float(np.clip(min(fractions), 0.0, 1.0))
    if frac >= 1.0 - 1e-12:
        return dict(fill), None
    out = dict(fill)
    out["new_a"] = float(fill["old_a"]) + (
        float(fill["new_a"]) - float(fill["old_a"])
    ) * frac
    out["new_b"] = float(fill["old_b"]) + (
        float(fill["new_b"]) - float(fill["old_b"])
    ) * frac
    out["adv_fill_fraction"] = frac
    return out, "adv_cap"


def _refresh_stock_target_metadata(
    frozen: pd.Series,
    fresh: pd.Series,
) -> pd.Series:
    """Refresh hedge/liquidity metadata while preserving structural pair gross."""
    out = frozen.copy()
    frozen_gross = abs(float(frozen.get("etf_usd", 0.0))) + abs(
        float(frozen.get("underlying_usd", 0.0))
    )
    fresh_a = float(fresh.get("etf_usd", 0.0))
    fresh_b = float(fresh.get("underlying_usd", 0.0))
    fresh_gross = abs(fresh_a) + abs(fresh_b)
    if frozen_gross > 1e-9 and fresh_gross > 1e-9:
        out["etf_usd"] = frozen_gross * fresh_a / fresh_gross
        out["underlying_usd"] = frozen_gross * fresh_b / fresh_gross
        out["gross_usd"] = frozen_gross
    structural = {"etf_usd", "underlying_usd", "gross_usd", "sleeve", "Underlying"}
    for key in fresh.index:
        if key not in structural:
            out[key] = fresh.get(key)
    return out


def _scale_pair_target_to_gross(row: pd.Series, gross_usd: float) -> pd.Series:
    """Scale both target legs atomically to a requested pair gross."""
    out = row.copy()
    gross = max(0.0, float(gross_usd))
    a = float(row.get("etf_usd", 0.0))
    b = float(row.get("underlying_usd", 0.0))
    source_gross = abs(a) + abs(b)
    if gross <= 1e-9 or source_gross <= 1e-9:
        out["etf_usd"] = 0.0
        out["underlying_usd"] = 0.0
        out["gross_usd"] = 0.0
    else:
        out["etf_usd"] = gross * a / source_gross
        out["underlying_usd"] = gross * b / source_gross
        out["gross_usd"] = gross
    return out


def _blend_stock_structural_targets(
    prior: pd.DataFrame,
    raw_plan: pd.DataFrame,
    *,
    confirmed_members: set[str],
    hard_exit_members: set[str] | None = None,
    alpha: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Convexly blend weekly B1/B2 pair gross without renormalizing sleeves."""
    blend_alpha = float(np.clip(alpha, 0.0, 1.0))
    hard = {_norm(x) for x in (hard_exit_members or set())}
    confirmed = {_norm(x) for x in confirmed_members}
    prior_by_norm = {_norm(idx): (idx, row) for idx, row in prior.iterrows()}
    raw_by_norm = {_norm(idx): (idx, row) for idx, row in raw_plan.iterrows()}
    eligible_raw = {key for key in raw_by_norm if key in confirmed}
    keys = set(prior_by_norm) | eligible_raw
    rows: list[pd.Series] = []
    audit: list[dict[str, Any]] = []

    for key in sorted(keys):
        prior_item = prior_by_norm.get(key)
        raw_item = raw_by_norm.get(key)
        prior_row = prior_item[1] if prior_item is not None else None
        raw_row = raw_item[1] if raw_item is not None else None
        output_idx = raw_item[0] if raw_item is not None else prior_item[0]
        prior_gross = (
            abs(float(prior_row.get("etf_usd", 0.0)))
            + abs(float(prior_row.get("underlying_usd", 0.0)))
            if prior_row is not None
            else 0.0
        )
        raw_gross = (
            abs(float(raw_row.get("etf_usd", 0.0)))
            + abs(float(raw_row.get("underlying_usd", 0.0)))
            if raw_row is not None
            else 0.0
        )
        if key in hard:
            effective_plan_gross = 0.0
            blended_gross = 0.0
            status = "hard_exit"
        elif raw_row is not None and key in confirmed:
            effective_plan_gross = raw_gross
            blended_gross = (
                (1.0 - blend_alpha) * prior_gross
                + blend_alpha * effective_plan_gross
            )
            status = "retained" if prior_row is not None else "new"
        elif prior_row is not None and key in confirmed:
            # An ordinary absence has not completed drop confirmation.
            effective_plan_gross = prior_gross
            blended_gross = prior_gross
            status = "drop_unconfirmed"
        else:
            effective_plan_gross = 0.0
            blended_gross = (1.0 - blend_alpha) * prior_gross
            status = "drop_decay"

        reference_row = (
            raw_row
            if raw_row is not None and key in confirmed
            else prior_row
        )
        if reference_row is not None and blended_gross > 1e-6:
            rows.append(
                _scale_pair_target_to_gross(
                    reference_row, blended_gross
                ).rename(output_idx)
            )
        audit.append(
            {
                "ETF": output_idx,
                "sleeve": str(
                    reference_row.get("sleeve", "")
                    if reference_row is not None
                    else ""
                ),
                "prior_structural_gross_usd": prior_gross,
                "raw_plan_gross_usd": raw_gross,
                "effective_plan_gross_usd": effective_plan_gross,
                "blended_structural_gross_usd": blended_gross,
                "target_blend_alpha": blend_alpha,
                "blend_status": status,
            }
        )

    out = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=raw_plan.columns.union(prior.columns))
    )
    out.index.name = raw_plan.index.name or prior.index.name
    return out, audit


def _turnover_budget_reference_gross(
    current_pair_gross: float,
    confirmed_desired_gross: float,
    *,
    hedge_safe: bool,
) -> float:
    """Use persistent risk destination for hedge-safe, preserving legacy basis."""
    current = max(0.0, float(current_pair_gross))
    if not hedge_safe:
        return current
    return max(current, max(0.0, float(confirmed_desired_gross)))


def _phase3_stock_residual_book(book: pd.DataFrame) -> pd.DataFrame:
    """Return only B1/B2 exposure governed by the live Phase-3 hedge pass."""
    if book is None or book.empty or "sleeve" not in book.columns:
        return pd.DataFrame(columns=book.columns if book is not None else None)
    return book[book["sleeve"].astype(str).isin(STOCK_SLEEVES)].copy()


def _fill_turnover(fill: dict[str, Any]) -> float:
    return abs(float(fill["new_a"]) - float(fill["old_a"])) + abs(
        float(fill["new_b"]) - float(fill["old_b"])
    )


def _allocate_hedge_safe_budget(
    fills: list[dict[str, Any]],
    *,
    budget_usd: float,
    establish_budget_frac: float = 0.50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Allocate hard exits, hedges, reductions, aged resizes, then growth."""
    if not fills:
        return [], []
    budget = float(budget_usd)
    if not np.isfinite(budget) or budget < 0:
        budget = float("inf")
    rank = {
        "hard_exit": 0,
        "hedge": 1,
        "gross_reduction": 2,
        "resize": 3,
        "growth": 4,
    }
    ordered = sorted(
        fills,
        key=lambda f: (
            rank.get(str(f.get("risk_class", "resize")), 3),
            -int(f.get("age", 0) or 0),
            -_fill_turnover(f),
            str(f.get("etf", "")),
        ),
    )
    accepted: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    used = 0.0
    growth_cap = (
        max(0.0, budget) * max(0.0, min(float(establish_budget_frac), 1.0))
        if np.isfinite(budget)
        else float("inf")
    )
    growth_used = 0.0
    for f in ordered:
        cls = str(f.get("risk_class", "resize"))
        turn = _fill_turnover(f)
        if cls in {"hard_exit", "hedge"}:
            accepted.append(f)
            continue
        room = max(0.0, budget - used)
        if cls == "growth":
            room = min(room, max(0.0, growth_cap - growth_used))
        if room <= 1e-9:
            deferred.append(f)
            continue
        if turn <= room + 1e-9:
            taken = f
        else:
            frac = room / max(turn, 1e-9)
            taken = dict(f)
            taken["new_a"] = float(f["old_a"]) + (
                float(f["new_a"]) - float(f["old_a"])
            ) * frac
            taken["new_b"] = float(f["old_b"]) + (
                float(f["new_b"]) - float(f["old_b"])
            ) * frac
            taken["scaled_by_turnover_budget"] = True
            taken["atomic_fill_fraction"] = float(frac)
            deferred.append(f)
        accepted.append(taken)
        used_now = _fill_turnover(taken)
        used += used_now
        if cls == "growth":
            growth_used += used_now
    return accepted, deferred


def _delta_adjusted_pair_exposure(
    etf_usd: float, underlying_usd: float, delta: Any
) -> tuple[float, float] | None:
    d = pd.to_numeric(delta, errors="coerce")
    if pd.isna(d) or not np.isfinite(float(d)):
        return None
    etf_delta = float(etf_usd) * float(d)
    und = float(underlying_usd)
    return float(und + etf_delta), float(abs(und) + abs(etf_delta))


def _hedge_correction_usd(
    *,
    net_notional: float,
    reference_gross: float,
    long_trigger_net_pct: float,
    long_target_net_pct: float,
    short_trigger_net_pct: float,
    short_target_net_pct: float,
) -> float:
    """Return underlying-notional correction using the live asymmetric rule."""
    gross = max(float(reference_gross), 0.0)
    net = float(net_notional)
    if gross <= 1e-9:
        return 0.0
    if net > float(long_trigger_net_pct) * gross:
        return float(long_target_net_pct) * gross - net
    if net < -float(short_trigger_net_pct) * gross:
        return -float(short_target_net_pct) * gross - net
    return 0.0


def _select_live_semantic_hedge_repair(
    group: pd.DataFrame,
    *,
    correction_usd: float,
    etf_prices: dict[str, float] | None = None,
) -> tuple[str | None, str | None, float, str | None]:
    """Choose the live-style repair leg without mutating structural B4 pairs.

    Returns ``(ETF row key, leg, notional change, block/fallback reason)``.
    Too-long exposure first adds a positive-delta B1/B2 ETF short. If locate is
    unavailable it may reduce that pair's long underlying. Too-short exposure
    buys the underlying. B4/B5 rows are never selected for the repair.
    """
    if group is None or group.empty or abs(float(correction_usd)) <= 1e-9:
        return None, None, 0.0, "no_repair"
    normal = group[group["sleeve"].astype(str).isin(STOCK_SLEEVES)].copy()
    if normal.empty:
        return None, None, 0.0, "no_normal_pair"
    normal = normal.sort_index()
    corr = float(correction_usd)
    if corr > 0:
        etf = str(normal.index[0])
        return etf, "underlying", corr, None

    # Too long: add ETF short by correction / Delta when executable.
    for etf, row in normal.iterrows():
        delta = pd.to_numeric(row.get("Delta"), errors="coerce")
        if pd.isna(delta) or float(delta) <= 0:
            continue
        etf_change = corr / float(delta)
        shares_available = pd.to_numeric(row.get("shares_available"), errors="coerce")
        locate_blocked = (
            _truthy(row.get("exclude_no_shares", False))
            or _truthy(row.get("borrow_missing_from_ftp", False))
            or (
                pd.notna(shares_available)
                and float(shares_available) <= 0
            )
        )
        if not locate_blocked and pd.notna(shares_available):
            px = float((etf_prices or {}).get(str(etf), 0.0) or 0.0)
            if px > 0 and abs(etf_change) > float(shares_available) * px + 1e-9:
                locate_blocked = True
        if not locate_blocked:
            return str(etf), "etf", float(etf_change), None

    # Locate/short growth infeasible: reduce an existing positive long only.
    for etf, row in normal.iterrows():
        long_now = float(row.get("underlying_usd", 0.0) or 0.0)
        if long_now > 1e-9:
            change = max(corr, -long_now)
            if change < -1e-9:
                return str(etf), "underlying", float(change), "fallback_reduce_long"
    return None, None, 0.0, "short_growth_infeasible"


def _targets_from_plan(
    plan: pd.DataFrame,
    *,
    budgets: dict[str, float],
    panel: dict[str, pd.DataFrame],
    equity: float,
    capital_usd: float,
    target_notional_mode: str,
    scale_sleeves_to_budget: bool = True,
) -> pd.DataFrame:
    """Build signed ETF/underlying close targets from one archived plan.

    Missing price panels remain undeployed.  They are deliberately *not*
    redistributed to surviving names.  ``equity_scaled`` preserves each
    archived plan's gross/equity multiple as NAV changes; ``fixed_plan_usd``
    replays the literal archived dollars.

    When ``scale_sleeves_to_budget`` is True (default), each sleeve with positive
    planned gross is scaled so sleeve gross equals the YAML sleeve budget
    (then × nav_scale). When False, uses ``min(budget, plan_gross)`` as before.
    """
    cols = [
        "ETF", "Underlying", "sleeve", "gross_usd", "etf_usd", "underlying_usd",
        "borrow_current", "borrow_underlying", "Delta", "execution_policy",
        "etf_adv_usd", "underlying_adv_usd", "etf_adv_shares",
        "underlying_adv_shares", "shares_available", "exclude_no_shares",
        "borrow_missing_from_ftp",
    ]
    if plan is None or plan.empty or capital_usd <= 0:
        return pd.DataFrame(columns=cols).set_index("ETF")
    mode = str(target_notional_mode or "equity_scaled").strip().lower()
    if mode not in {"equity_scaled", "fixed_plan_usd"}:
        raise ValueError(f"unknown target_notional_mode={target_notional_mode!r}")
    nav_scale = float(equity) / float(capital_usd) if mode == "equity_scaled" else 1.0
    nav_scale = max(nav_scale, 0.0)

    rows: list[dict[str, Any]] = []
    planned_gross = 0.0
    for sleeve, budget in budgets.items():
        if budget <= 0:
            continue
        sub_all = plan[plan["sleeve"] == sleeve].copy()
        if sub_all.empty:
            continue
        reduce_only = sub_all.get(
            "execution_policy", pd.Series("normal", index=sub_all.index)
        ).astype(str).str.lower().eq("reduce_only")
        executable_gross = pd.to_numeric(
            sub_all["gross_target_usd"], errors="coerce"
        ).fillna(0.0)
        model_gross = pd.to_numeric(
            sub_all.get("model_gross_target_usd", executable_gross), errors="coerce"
        ).fillna(0.0)
        # Reduce-only rows size off model gross when executable is 0 (live contract).
        g_all = executable_gross.where(~reduce_only, model_gross).clip(lower=0.0)
        g_sum = float(g_all.sum())
        if g_sum <= 0:
            continue
        if scale_sleeves_to_budget:
            sleeve_cap = float(budget) * nav_scale
        else:
            sleeve_cap = min(float(budget), g_sum) * nav_scale
        planned_gross += sleeve_cap
        for (_, r), raw_gross in zip(sub_all.iterrows(), g_all.to_numpy()):
            etf = str(r["ETF"])
            row_reduce_only = str(r.get("execution_policy", "")).strip().lower() == "reduce_only"
            # Executable-zero reduce-only with no positive model: keep out of
            # target (sim share-holds via reduce_only_etfs set).
            if etf not in panel or raw_gross <= 0:
                continue
            gross_usd = float(raw_gross / g_sum * sleeve_cap)
            # proposed_trades contract: long_usd is the UNDERLYING target and
            # short_usd is the ETF target.  Prefer the explicit columns when
            # present so this cannot silently regress to the old reversed map.
            if row_reduce_only:
                leg_a = _float_or(r.get("model_short_usd"), _float_or(r.get("optimal_short_usd")))
                leg_b = _float_or(r.get("model_long_usd"), _float_or(r.get("optimal_long_usd")))
            else:
                leg_a = _float_or(r.get("etf_target_usd"), _float_or(r.get("short_usd")))
                leg_b = _float_or(r.get("underlying_target_usd"), _float_or(r.get("long_usd")))
            leg_gross = abs(leg_a) + abs(leg_b)
            if leg_gross <= 0:
                continue
            borrow_b = _float_or(
                r.get("borrow_underlying"),
                _float_or(r.get("borrow_b"), _float_or(r.get("underlying_borrow_annual"))),
            )
            rows.append(
                {
                    "ETF": etf,
                    "Underlying": str(r.get("Underlying", "") or ""),
                    "sleeve": sleeve,
                    "gross_usd": gross_usd,
                    "etf_usd": gross_usd * leg_a / leg_gross,
                    "underlying_usd": gross_usd * leg_b / leg_gross,
                    "borrow_current": _float_or(r.get("borrow_current")),
                    "borrow_underlying": borrow_b,
                    "Delta": float(pd.to_numeric(r.get("Delta"), errors="coerce") or np.nan),
                    "execution_policy": str(r.get("execution_policy", "normal") or "normal"),
                    "etf_adv_usd": _liquidity_value(
                        r,
                        (
                            "etf_adv_usd", "adv_usd", "median_daily_dollar_volume",
                            "median_daily_volume_usd",
                        ),
                    ),
                    "underlying_adv_usd": _liquidity_value(
                        r,
                        ("underlying_adv_usd", "underlying_median_daily_dollar_volume"),
                    ),
                    "etf_adv_shares": _liquidity_value(
                        r,
                        ("etf_adv_shares", "median_daily_volume_shares", "adv_median_shares"),
                    ),
                    "underlying_adv_shares": _liquidity_value(
                        r, ("underlying_adv_shares",)
                    ),
                    "shares_available": pd.to_numeric(
                        r.get("shares_available"), errors="coerce"
                    ),
                    "exclude_no_shares": _truthy(r.get("exclude_no_shares", False)),
                    "borrow_missing_from_ftp": _truthy(
                        r.get("borrow_missing_from_ftp", False)
                    ),
                }
            )
    if not rows:
        out = pd.DataFrame(columns=cols).set_index("ETF")
        out.attrs["planned_gross_usd"] = planned_gross
        out.attrs["tradeable_gross_usd"] = 0.0
        return out

    raw = pd.DataFrame(rows)
    if raw["ETF"].duplicated().any():
        # Defensive aggregation for schema eras that emitted duplicate rows.
        out = (
            raw.groupby("ETF", as_index=False)
            .agg(
                Underlying=("Underlying", "last"), sleeve=("sleeve", "last"),
                gross_usd=("gross_usd", "sum"), etf_usd=("etf_usd", "sum"),
                underlying_usd=("underlying_usd", "sum"),
                borrow_current=("borrow_current", "last"),
                borrow_underlying=("borrow_underlying", "last"), Delta=("Delta", "last"),
                execution_policy=("execution_policy", "last"),
                etf_adv_usd=("etf_adv_usd", "last"),
                underlying_adv_usd=("underlying_adv_usd", "last"),
                etf_adv_shares=("etf_adv_shares", "last"),
                underlying_adv_shares=("underlying_adv_shares", "last"),
                shares_available=("shares_available", "last"),
                exclude_no_shares=("exclude_no_shares", "last"),
                borrow_missing_from_ftp=("borrow_missing_from_ftp", "last"),
            )
            .set_index("ETF")
        )
    else:
        out = raw.set_index("ETF")
    out.attrs["planned_gross_usd"] = planned_gross
    out.attrs["tradeable_gross_usd"] = float(out[["etf_usd", "underlying_usd"]].abs().sum().sum())
    return out


def _weights_from_plan(
    plan: pd.DataFrame,
    *,
    budgets: dict[str, float],
    panel: dict[str, pd.DataFrame],
    day: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compatibility view: gross/equity weights, never renormalized to 1x."""
    del day
    target = _targets_from_plan(
        plan, budgets=budgets, panel=panel, equity=1.0, capital_usd=1.0,
        target_notional_mode="fixed_plan_usd",
        scale_sleeves_to_budget=True,
    )
    if target.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    w = target[["etf_usd", "underlying_usd"]].abs().sum(axis=1)
    return w, target.reset_index()


def _build_return_cache(
    plans: list[pd.DataFrame],
    panel: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Per-leg close returns only; plan fractions/borrow stay point-in-time."""
    etfs = {str(row["ETF"]) for plan in plans for _, row in plan.iterrows()}
    out: dict[str, pd.DataFrame] = {}
    for etf in sorted(etfs):
        if etf not in panel:
            continue
        px = panel[etf].sort_index()
        if len(px) < 2:
            continue
        out[etf] = pd.DataFrame(
            {"r_etf": px["a_px"].pct_change(fill_method=None),
             "r_underlying": px["b_px"].pct_change(fill_method=None)},
            index=px.index,
        )
    return out


def simulate_book_from_plan_timeline(
    timeline: dict[pd.Timestamp, pd.DataFrame],
    panel: dict[str, pd.DataFrame],
    *,
    budgets: dict[str, float],
    capital_usd: float,
    start: pd.Timestamp,
    slippage_bps: float = 20.0,
    commission_per_share: float = 0.0035,
    margin_rate_annual: float = 0.0445,
    financing_daycount: float = 360.0,
    short_proceeds_credit_annual: float = 0.038,
    execution_lag_sessions: int = 1,
    target_notional_mode: str = "equity_scaled",
    scale_sleeves_to_budget: bool = True,
    enter_band_pct: float = 0.12,
    exit_band_pct: float = 0.04,
    min_trade_usd: float = 250.0,
    use_resize_bands: bool = True,
    check_freq: str = "W-FRI",
    stock_rebalance_clock: str | None = None,
    retarget_on_plan_change: bool = True,
    pre_archive_policy: str = "cash",
    b4_execution: str = "cadence",
    apply_delist_flatten: bool = True,
    use_borrow_history: bool = True,
    same_run_churn_enabled: bool = True,
    purgatory_model_zero_policy: str = "hold",
    b4_membership_clock: str = "operator_5d",
    operator_check_days: int = 5,
    b4_apply_resize_bands: bool = True,
    b4_ratchet_execution_guard: bool = True,
    b4_allow_inverse_cover: bool = True,
    b4_empty_plan_policy: str = "hold",
    net_shared_underlyings: bool = True,
    turnover_pace_enabled: bool = True,
    turnover_pace_mode: str | None = None,
    confirmation_count: int = 2,
    entry_ramp_sessions: int = 5,
    reduction_ramp_sessions: int = 3,
    remaining_gap_rate: float = 0.25,
    target_blend_alpha: float = 0.25,
    stock_midweek_mode: str = "ramp_and_hedge",
    midweek_hedge_repair: bool = False,
    hedge_reserve_frac: float = 0.20,
    adv_participation_pct: float = 0.10,
    sleeve_gross_ema_alpha: float = 0.35,
    max_leg_step_pct: float = 0.25,
    pair_gross_ramp_pct: float = 0.25,
    max_daily_turnover_pct: float = 0.15,
    legacy_max_daily_turnover_pct: float | None = None,
    establish_budget_frac: float = 0.50,
    resize_age_boost_days: int = 5,
    hedge_long_trigger_net_pct: float = 0.04,
    hedge_long_target_net_pct: float = 0.01,
    hedge_short_trigger_net_pct: float = 0.01,
    hedge_short_target_net_pct: float = 0.00,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Close-to-close, share-hold simulation with point-in-time plans.

    Returns ``(nav, audit, meta, pair_stats, sleeve_daily)``.

    ``meta["pair_daily"]`` is a long DataFrame (one row per pair×day with activity
    or open exposure) used for top/bottom time-series plots.

    ``pre_archive_policy``
        ``cash`` — hold capital until first plan (default).
        ``skip`` — start calendar at first plan date.

    ``b4_execution``
        ``cadence`` — B4 pairs retarget on TR/VCR rebal dates with dynamic ``h``
        (production default). ``weekly_plan_legs`` — legacy W-FRI plan-leg path.

    ``purgatory_model_zero_policy``
        ``hold`` (default) — reduce-only with missing/zero ``model_*`` share-holds
        (live: executable 0 is not a close). ``exit`` — legacy flatten on model 0.

    ``b4_empty_plan_policy``
        ``hold`` (default) — if active plan B4 executable gross is ~0, do not
        true-drop held B4 names (archive-gap / sizing-fail anti-glitch).
        ``exit`` — legacy: membership drops even into an empty B4 plan.

    ``net_shared_underlyings``
        When True (default), borrow / short-credit / margin use netted underlying
        exposure across sleeves (B1/B2 long vs B4 short). Price PnL still marks
        each pair leg so sleeve attribution stays readable; book price PnL nets.

    ``turnover_pace_mode`` (sim only)
        ``hedge_safe_v1`` persists confirmed pair targets and advances them every
        session with atomic two-leg fills, daily delta hedge repair, and a
        risk-first allocator. ``legacy`` retains EMA/leg/budget pacing; ``off``
        fully chases each eligible target.

    ``b4_membership_clock``
        ``operator_5d`` (default) — B4 add/true-drop only every N sessions;
        ``weekly_fri`` — Fridays only; ``every_plan`` — legacy flicker path.

    ``b4_apply_resize_bands`` / ``b4_ratchet_execution_guard``
        Phase-2b hysteresis + inverse cover pin/cap on B4 cadence retargets.

    The simulator applies one terminal target per pair/day rather than executing
    production's intermediate phases. Consequently predictable same-run
    reversals are structurally netted to zero; the policy flag and zero
    override/round-trip counters are emitted for production parity auditing.
    """
    empty = pd.DataFrame()
    purg_zero_pol = str(purgatory_model_zero_policy or "hold").strip().lower()
    if purg_zero_pol not in {"hold", "exit"}:
        purg_zero_pol = "hold"
    membership_mode = str(b4_membership_clock or "operator_5d").strip().lower()
    op_check_days = max(int(operator_check_days or 5), 1)
    apply_b4_bands = bool(b4_apply_resize_bands)
    ratchet_guard = bool(b4_ratchet_execution_guard)
    allow_inv_cover = bool(b4_allow_inverse_cover)
    empty_plan_pol = str(b4_empty_plan_policy or "hold").strip().lower()
    if empty_plan_pol not in {"hold", "exit"}:
        empty_plan_pol = "hold"
    net_underlyings = bool(net_shared_underlyings)
    requested_pace_mode = (
        str(turnover_pace_mode).strip().lower()
        if turnover_pace_mode is not None
        else ("legacy" if bool(turnover_pace_enabled) else "off")
    )
    if requested_pace_mode not in {"hedge_safe_v1", "legacy", "off"}:
        requested_pace_mode = "hedge_safe_v1"
    pace_mode = requested_pace_mode
    pace_on = pace_mode != "off"
    hedge_safe = pace_mode == "hedge_safe_v1"
    legacy_pace = pace_mode == "legacy"
    confirm_n = max(int(confirmation_count or 1), 1)
    entry_sessions = max(int(entry_ramp_sessions or 1), 1)
    reduction_sessions = max(int(reduction_ramp_sessions or 1), 1)
    resize_gap_rate = float(np.clip(remaining_gap_rate, 0.0, 1.0))
    structural_blend_alpha = float(np.clip(target_blend_alpha, 0.0, 1.0))
    midweek_mode = str(stock_midweek_mode or "ramp_and_hedge").strip().lower()
    if midweek_mode in {"full", "ramp", "all"}:
        midweek_mode = "ramp_and_hedge"
    if midweek_mode in {"none", "off", "strict", "share_hold"}:
        midweek_mode = "rebal_only"
    if midweek_mode not in {"rebal_only", "hedge_only", "ramp_and_hedge"}:
        midweek_mode = "ramp_and_hedge"
    allow_midweek_hedge_repair = bool(midweek_hedge_repair)
    stock_clock = str(
        stock_rebalance_clock
        if stock_rebalance_clock is not None
        else (
            "weekly_fri"
            if str(check_freq or "").strip().upper() in {"W-FRI", "W-FRI "}
            else "operator_5d"
        )
    ).strip().lower()
    if stock_clock in {"w-fri", "friday", "weekly"}:
        stock_clock = "weekly_fri"
    if stock_clock not in {"operator_5d", "weekly_fri", "every_plan"}:
        stock_clock = "operator_5d"
    hedge_reserve = float(np.clip(hedge_reserve_frac, 0.0, 1.0))
    adv_pct = max(float(adv_participation_pct), 0.0)
    pace_alpha = float(sleeve_gross_ema_alpha)
    pace_leg_pct = float(max_leg_step_pct)
    pair_ramp_pct = float(pair_gross_ramp_pct)
    pace_turn_pct = float(
        legacy_max_daily_turnover_pct
        if legacy_pace and legacy_max_daily_turnover_pct is not None
        else max_daily_turnover_pct
    )
    pace_est_frac = float(establish_budget_frac)
    age_boost_days = max(int(resize_age_boost_days or 1), 1)
    if not timeline:
        return (
            pd.Series(dtype=float),
            empty,
            {"error": "empty_timeline", "pair_daily": empty},
            empty,
            empty,
        )

    first_plan = min(timeline)
    sim_start = start
    if pre_archive_policy == "skip" and first_plan > start:
        sim_start = first_plan

    ret_cache = _build_return_cache(list(timeline.values()), panel)
    if not ret_cache:
        return pd.Series(dtype=float), empty, {"error": "no_returns", "pair_daily": empty}, empty, empty

    all_idx = sorted(set().union(*[set(s.index) for s in ret_cache.values()]))
    cal = pd.DatetimeIndex([d for d in all_idx if d >= sim_start])
    if len(cal) < 20:
        return (
            pd.Series(dtype=float),
            empty,
            {"error": "short_calendar", "pair_daily": empty},
            empty,
            empty,
        )

    # B1/B2 structural clock. Default matches live operator_5d (every N bdays).
    # Legacy W-FRI via stock_rebalance_clock=weekly_fri (or check_freq when clock unset).
    check_days = _membership_day_set(
        cal, mode=stock_clock, check_days=op_check_days
    )
    membership_days = _membership_day_set(
        cal, mode=membership_mode, check_days=op_check_days
    )
    effective = _effective_plan_timeline(
        timeline, cal, execution_lag_sessions=execution_lag_sessions
    )
    slip = slippage_bps / 1e4
    equity = float(capital_usd)
    pos_cols = [
        "Underlying", "sleeve", "gross_usd", "etf_usd", "underlying_usd",
        "borrow_current", "borrow_underlying", "Delta", "execution_policy", "plan_date",
        "etf_adv_usd", "underlying_adv_usd", "etf_adv_shares",
        "underlying_adv_shares", "shares_available", "exclude_no_shares",
        "borrow_missing_from_ftp",
    ]
    cur = pd.DataFrame(columns=pos_cols)
    cur.index.name = "ETF"
    nav = pd.Series(index=cal, dtype=float)
    audit_rows: list[dict] = []
    n_rebal = 0
    turnover_l1 = 0.0
    turnover_total = 0.0
    active_plan: pd.DataFrame | None = None
    active_plan_date: pd.Timestamp | None = None
    plans_used: set[pd.Timestamp] = set()
    cash_days = 0
    contrib: dict[str, dict[str, Any]] = {}
    sleeve_daily_rows: list[dict] = []
    pair_daily_rows: list[dict] = []
    running_peak = float(capital_usd)
    b4_mode = str(b4_execution or "cadence").strip().lower()
    use_b4_cadence = b4_mode in {"cadence", "tr_vcr", "dynamic_h"}

    # Und map + cadence state for B4
    und_by_etf: dict[str, str] = {}
    b4_etfs: set[str] = set()
    for plan in timeline.values():
        if plan is None or plan.empty:
            continue
        for _, r in plan.iterrows():
            etf, und = _norm(r.get("ETF")), _norm(r.get("Underlying"))
            if etf and und:
                und_by_etf[etf] = und
            sleeve = str(r.get("sleeve", "") or "").strip().lower()
            delta = pd.to_numeric(r.get("Delta"), errors="coerce")
            if sleeve == B4_SLEEVE or (pd.notna(delta) and float(delta) < 0 and sleeve in ("", B4_SLEEVE)):
                if etf:
                    b4_etfs.add(etf)
    b4_cadence = (
        _prepare_b4_cadence_state(panel, cal, und_by_etf=und_by_etf, b4_etfs=b4_etfs)
        if use_b4_cadence
        else {}
    )
    try:
        from scripts.pair_price_panel import load_delistings

        delist_map = load_delistings() if apply_delist_flatten else {}
    except Exception:
        delist_map = {}
    borrow_hist = _load_borrow_history_map() if use_borrow_history else {}
    n_b4_cadence_rebals = 0
    n_b4_membership_deferred = 0
    n_b4_empty_plan_holds = 0
    n_b4_ratchet_pins = 0
    n_delist_flat = 0
    n_purgatory_reductions = 0
    purgatory_blocked_add_usd = 0.0
    n_deferred_pace = 0
    sleeve_gross_ema: dict[str, float] = {}
    persistent_pair_targets = pd.DataFrame(columns=pos_cols)
    pair_request_age: dict[str, int] = {}
    confirmed_stock_members: set[str] = set()
    stock_presence_streak: dict[str, int] = {}
    stock_absence_streak: dict[str, int] = {}
    stock_seen: set[str] = set()
    latest_present_stock: set[str] = set()
    latest_hard_stock: set[str] = set()
    pending_target_rows: list[dict[str, Any]] = []
    n_hedge_repairs = 0
    hedge_repair_turnover = 0.0
    n_growth_blocked_hedge_infeasible = 0
    turnover_budget_day = 0.0
    turnover_used_pace_day = 0.0
    confirmed_desired_gross_ref = 0.0

    def _ensure_contrib(etf: str, row: pd.Series | None = None) -> dict[str, Any]:
        if etf not in contrib:
            contrib[etf] = {
                "ETF": etf,
                "Underlying": str(row.get("Underlying", "") if row is not None else ""),
                "sleeve": str(row.get("sleeve", "") if row is not None else ""),
                "price_pnl_usd": 0.0,
                "borrow_cost_usd": 0.0,
                "short_credit_usd": 0.0,
                "margin_cost_usd": 0.0,
                "txn_cost_usd": 0.0,
                "rebalance_dates": [],
                "end_etf_usd": 0.0,
                "end_underlying_usd": 0.0,
                "last_target_etf_usd": 0.0,
                "last_target_underlying_usd": 0.0,
                "Delta": np.nan,
            }
        elif row is not None:
            contrib[etf]["Underlying"] = str(row.get("Underlying", contrib[etf]["Underlying"]))
            contrib[etf]["sleeve"] = str(row.get("sleeve", contrib[etf]["sleeve"]))
            if "Delta" in row.index and pd.notna(row.get("Delta")):
                contrib[etf]["Delta"] = float(pd.to_numeric(row.get("Delta"), errors="coerce") or np.nan)
        return contrib[etf]

    for i, d in enumerate(cal):
        equity_start = float(equity)
        gross_start = float(cur[["etf_usd", "underlying_usd"]].abs().sum().sum()) if not cur.empty else 0.0
        if gross_start <= 1e-9:
            cash_days += 1
        sleeve_comp = {
            s: {"price": 0.0, "borrow": 0.0, "credit": 0.0, "margin": 0.0, "txn": 0.0}
            for s in ALL_SLEEVES
        }
        day_pair: dict[str, dict[str, Any]] = {}

        def _day_pair(etf: str, row: pd.Series | None = None) -> dict[str, Any]:
            if etf not in day_pair:
                day_pair[etf] = {
                    "ETF": etf,
                    "Underlying": str(row.get("Underlying", "") if row is not None else ""),
                    "sleeve": str(row.get("sleeve", "") if row is not None else ""),
                    "Delta": (
                        pd.to_numeric(row.get("Delta", np.nan), errors="coerce")
                        if row is not None
                        else np.nan
                    ),
                    "price_pnl": 0.0,
                    "borrow_cost": 0.0,
                    "short_credit": 0.0,
                    "margin_cost": 0.0,
                    "txn_cost": 0.0,
                }
            elif row is not None:
                day_pair[etf]["Underlying"] = str(
                    row.get("Underlying", day_pair[etf]["Underlying"])
                )
                day_pair[etf]["sleeve"] = str(row.get("sleeve", day_pair[etf]["sleeve"]))
                row_delta = pd.to_numeric(row.get("Delta", np.nan), errors="coerce")
                if pd.notna(row_delta):
                    day_pair[etf]["Delta"] = float(row_delta)
            return day_pair[etf]

        price_pnl = 0.0
        borrow_cost = 0.0
        short_credit = 0.0
        margin_cost = 0.0
        txn_cost = 0.0
        turnover_day = 0.0
        turnover_reference_gross_day = 0.0
        deployed_desired_ratio_day = np.nan
        raw_plan_stock_gross_day = np.nan
        blended_stock_structural_gross_day = np.nan
        stale_etf = 0
        stale_underlying = 0
        day_did_rebal = False

        # 1) Mark yesterday's shares to today's close.  Missing bars are stale
        # marks (zero return), never forced closes or zero-valued positions.
        if not cur.empty:
            for etf in list(cur.index):
                row = cur.loc[etf]
                rr = ret_cache.get(etf)
                if rr is None or d not in rr.index:
                    r_a = r_b = 0.0
                    stale_etf += 1
                    stale_underlying += 1
                else:
                    raw_a = rr.at[d, "r_etf"]
                    raw_b = rr.at[d, "r_underlying"]
                    r_a = float(raw_a) if np.isfinite(raw_a) else 0.0
                    r_b = float(raw_b) if np.isfinite(raw_b) else 0.0
                    stale_etf += int(not np.isfinite(raw_a) and i > 0)
                    stale_underlying += int(not np.isfinite(raw_b) and i > 0)
                pnl_e = float(row["etf_usd"]) * r_a + float(row["underlying_usd"]) * r_b
                cur.at[etf, "etf_usd"] = float(row["etf_usd"]) * (1.0 + r_a)
                cur.at[etf, "underlying_usd"] = float(row["underlying_usd"]) * (1.0 + r_b)
                cur.at[etf, "gross_usd"] = abs(float(cur.at[etf, "etf_usd"])) + abs(float(cur.at[etf, "underlying_usd"]))
                price_pnl += pnl_e
                c = _ensure_contrib(etf, cur.loc[etf])
                c["price_pnl_usd"] += pnl_e
                _day_pair(etf, cur.loc[etf])["price_pnl"] += pnl_e
                sl = str(row.get("sleeve", ""))
                if sl in sleeve_comp:
                    sleeve_comp[sl]["price"] += pnl_e

            # 2) Borrow fee + short-sale proceeds credit.
            # ETF shorts always charge per leg. Underlying shorts optionally net
            # across sleeves (B1/B2 long vs B4 short) so internalized notional
            # does not pay borrow / earn short credit twice.
            daycount = max(float(financing_daycount), 1.0)
            credit_rate = max(float(short_proceeds_credit_annual), 0.0)
            und_net = _underlying_net_by_symbol(cur) if net_underlyings else {}
            # Per-underlying short legs for pro-rata allocation of net short cost.
            und_short_legs: dict[str, list[tuple[str, float, float]]] = {}
            if net_underlyings:
                for etf, row in cur.iterrows():
                    und_usd = float(row["underlying_usd"])
                    if und_usd >= -1e-9:
                        continue
                    und = _norm(row.get("Underlying", "")) or f"__PAIR__{_norm(etf)}"
                    rate_b = float(
                        pd.to_numeric(row.get("borrow_underlying"), errors="coerce") or 0.0
                    )
                    und_short_legs.setdefault(und, []).append(
                        (str(etf), abs(und_usd), max(rate_b, 0.0))
                    )

            for etf, row in cur.iterrows():
                sl = str(row.get("sleeve", ""))
                rate_a = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
                rate_b = float(pd.to_numeric(row.get("borrow_underlying"), errors="coerce") or 0.0)
                if borrow_hist:
                    bh = borrow_hist.get(_norm(etf))
                    if bh is not None and len(bh):
                        hist = bh.loc[bh.index <= d]
                        if len(hist):
                            rate_a = float(hist.iloc[-1])
                cost = 0.0
                credit = 0.0
                etf_usd = float(row["etf_usd"])
                und_usd = float(row["underlying_usd"])
                if etf_usd < 0:
                    short_abs = abs(etf_usd)
                    cost += short_abs * max(rate_a, 0.0) / TRADING_DAYS
                    credit += short_abs * credit_rate / daycount
                if und_usd < 0 and not net_underlyings:
                    short_abs = abs(und_usd)
                    cost += short_abs * max(rate_b, 0.0) / TRADING_DAYS
                    credit += short_abs * credit_rate / daycount
                borrow_cost += cost
                short_credit += credit
                c = _ensure_contrib(etf, row)
                c["borrow_cost_usd"] += cost
                c["short_credit_usd"] += credit
                dp = _day_pair(etf, row)
                dp["borrow_cost"] += cost
                dp["short_credit"] += credit
                if sl in sleeve_comp:
                    sleeve_comp[sl]["borrow"] += cost
                    sleeve_comp[sl]["credit"] += credit

            if net_underlyings:
                for und, net_usd in und_net.items():
                    if net_usd >= -1e-9:
                        continue  # net flat or long → no und borrow/credit
                    legs = und_short_legs.get(und) or []
                    if not legs:
                        continue
                    gross_short = float(sum(a for _, a, _ in legs))
                    if gross_short <= 1e-9:
                        continue
                    net_short = abs(float(net_usd))
                    w_rate = float(sum(a * r for _, a, r in legs)) / gross_short
                    und_cost = net_short * w_rate / TRADING_DAYS
                    und_credit = net_short * credit_rate / daycount
                    for etf, abs_und, _r in legs:
                        share = abs_und / gross_short
                        c_cost = und_cost * share
                        c_cred = und_credit * share
                        borrow_cost += c_cost
                        short_credit += c_cred
                        row = cur.loc[etf]
                        sl = str(row.get("sleeve", ""))
                        _ensure_contrib(etf, row)["borrow_cost_usd"] += c_cost
                        _ensure_contrib(etf, row)["short_credit_usd"] += c_cred
                        dp = _day_pair(etf, row)
                        dp["borrow_cost"] += c_cost
                        dp["short_credit"] += c_cred
                        if sl in sleeve_comp:
                            sleeve_comp[sl]["borrow"] += c_cost
                            sleeve_comp[sl]["credit"] += c_cred

            # Margin debit on netted long market value less NAV.
            if net_underlyings:
                long_n, _short_n, _g, _net = _netted_book_notionals(cur)
                long_total = float(long_n)
                und_net_map = und_net
                # Precompute positive und sum per symbol.
                pos_und_by_sym: dict[str, float] = {}
                for etf2, row2 in cur.iterrows():
                    und2 = _norm(row2.get("Underlying", "")) or f"__PAIR__{_norm(etf2)}"
                    u2 = float(row2["underlying_usd"])
                    if u2 > 0:
                        pos_und_by_sym[und2] = float(pos_und_by_sym.get(und2, 0.0)) + u2
                long_by_pair = {}
                for etf, row in cur.iterrows():
                    basis = max(float(row["etf_usd"]), 0.0)
                    und = _norm(row.get("Underlying", "")) or f"__PAIR__{_norm(etf)}"
                    und_usd = float(row["underlying_usd"])
                    net_u = float(und_net_map.get(und, 0.0))
                    pos_sum = float(pos_und_by_sym.get(und, 0.0))
                    if und_usd > 0 and net_u > 0 and pos_sum > 1e-9:
                        basis += net_u * (und_usd / pos_sum)
                    if basis > 0:
                        long_by_pair[str(etf)] = basis
            else:
                long_by_pair = {
                    etf: max(float(row["etf_usd"]), 0.0) + max(float(row["underlying_usd"]), 0.0)
                    for etf, row in cur.iterrows()
                }
                long_total = float(sum(long_by_pair.values()))
            debit_base = max(0.0, long_total - (equity + price_pnl - borrow_cost + short_credit))
            margin_cost = debit_base * max(float(margin_rate_annual), 0.0) / daycount
            if margin_cost > 0 and long_total > 0:
                for etf, basis in long_by_pair.items():
                    alloc = margin_cost * basis / long_total
                    row = cur.loc[etf]
                    _ensure_contrib(etf, row)["margin_cost_usd"] += alloc
                    _day_pair(etf, row)["margin_cost"] += alloc
                    sl = str(row.get("sleeve", ""))
                    if sl in sleeve_comp:
                        sleeve_comp[sl]["margin"] += alloc

        equity += price_pnl - borrow_cost - margin_cost + short_credit

        # 3) At the close, activate any plan whose information lag has elapsed,
        # then retarget: B1/B2 on weekly W-FRI; B4 on TR/VCR cadence (default).
        plan_changed = d in effective
        if plan_changed:
            active_plan_date, active_plan = effective[d]
            plans_used.add(active_plan_date)
            if hedge_safe and active_plan is not None:
                ap = active_plan.copy()
                sleeves = ap.get("sleeve", pd.Series("", index=ap.index)).astype(str)
                policies = ap.get(
                    "execution_policy", pd.Series("normal", index=ap.index)
                ).astype(str).str.lower()
                hard_mask = policies.eq("hard_exit")
                if "hard_exit" in ap.columns:
                    hard_mask |= ap["hard_exit"].map(_truthy)
                stock_mask = sleeves.isin(STOCK_SLEEVES)
                present_stock = set(ap.loc[stock_mask & ~hard_mask, "ETF"].map(_norm))
                hard_stock = set(ap.loc[stock_mask & hard_mask, "ETF"].map(_norm))
                latest_present_stock = present_stock
                latest_hard_stock = hard_stock
                stock_seen |= present_stock | hard_stock
                for etf in stock_seen | confirmed_stock_members:
                    if etf in present_stock:
                        stock_presence_streak[etf] = stock_presence_streak.get(etf, 0) + 1
                        stock_absence_streak[etf] = 0
                    else:
                        stock_presence_streak[etf] = 0
                        if etf in confirmed_stock_members:
                            stock_absence_streak[etf] = stock_absence_streak.get(etf, 0) + 1
                for etf in hard_stock:
                    confirmed_stock_members.discard(etf)
                    stock_presence_streak[etf] = 0
                    stock_absence_streak[etf] = confirm_n
        stock_rebal_day = active_plan is not None and (
            d in check_days
            or (plan_changed and bool(retarget_on_plan_change))
            or (plan_changed and cur.empty)
        )
        if hedge_safe and stock_rebal_day:
            # Streaks update on every effective plan, but ordinary membership is
            # only admitted/removed on the stock structural decision clock.
            for etf in stock_seen | confirmed_stock_members:
                if (
                    etf in latest_present_stock
                    and stock_presence_streak.get(etf, 0) >= confirm_n
                ):
                    confirmed_stock_members.add(etf)
                elif (
                    etf in confirmed_stock_members
                    and stock_absence_streak.get(etf, 0) >= confirm_n
                ):
                    confirmed_stock_members.discard(etf)
            confirmed_stock_members -= latest_hard_stock
        b4_cadence_today = bool(
            use_b4_cadence
            and any(d in (cad.get("rebal") or ()) for cad in b4_cadence.values())
        )
        delist_today = bool(
            apply_delist_flatten
            and delist_map
            and not cur.empty
            and any(pd.Timestamp(d) >= pd.Timestamp(delist_map.get(_norm(e), "2100-01-01")) for e in cur.index)
        )
        # Cadence mode also needs targets on plan-change days for entries/exits,
        # and on operator membership days even when the plan is unchanged.
        membership_today = bool(use_b4_cadence and (d in membership_days))
        target_confirmed_today = active_plan is not None and (
            stock_rebal_day
            or b4_cadence_today
            or plan_changed
            or delist_today
            or membership_today
        )
        controller_confirmed_today = bool(
            active_plan is not None
            and (stock_rebal_day or b4_cadence_today or delist_today or membership_today)
        )
        need_target = bool(
            target_confirmed_today
            or (hedge_safe and active_plan is not None and not persistent_pair_targets.empty)
        )
        target = pd.DataFrame(columns=pos_cols)
        target_planned_gross = 0.0
        target_tradeable_gross = 0.0
        blocked_pairs = 0
        if need_target and active_plan is not None:
            if target_confirmed_today:
                target = _targets_from_plan(
                    active_plan,
                    budgets=budgets,
                    panel=panel,
                    equity=equity,
                    capital_usd=capital_usd,
                    target_notional_mode=target_notional_mode,
                    scale_sleeves_to_budget=scale_sleeves_to_budget,
                )
                target_planned_gross = float(target.attrs.get("planned_gross_usd", 0.0))
                target_tradeable_gross = float(target.attrs.get("tradeable_gross_usd", 0.0))
                target = target.copy()
            else:
                target = persistent_pair_targets.copy()
                target_planned_gross = float(
                    target[["etf_usd", "underlying_usd"]].abs().sum().sum()
                )
                target_tradeable_gross = target_planned_gross
            # Gradual sleeve budget: EMA planned stock-sleeve gross on weekly retargets.
            if legacy_pace and stock_rebal_day and not target.empty:
                target = _apply_sleeve_gross_ema(
                    target,
                    sleeve_gross_ema,
                    alpha=pace_alpha,
                    sleeves=STOCK_SLEEVES,
                )
                target_tradeable_gross = float(
                    target[["etf_usd", "underlying_usd"]].abs().sum().sum()
                ) if not target.empty else 0.0
            target["plan_date"] = str(active_plan_date.date()) if active_plan_date is not None else ""
            if hedge_safe and target_confirmed_today:
                # Daily plans refresh stock hedge/liquidity metadata, but only a
                # stock structural decision may replace pair-gross destinations.
                non_stock = target[
                    ~target["sleeve"].astype(str).isin(STOCK_SLEEVES)
                ]
                raw_stock = target[
                    target["sleeve"].astype(str).isin(STOCK_SLEEVES)
                ]
                old_stock = (
                    persistent_pair_targets[
                        persistent_pair_targets["sleeve"].astype(str).isin(STOCK_SLEEVES)
                    ]
                    if not persistent_pair_targets.empty
                    else pd.DataFrame(columns=target.columns)
                )

                selected_rows: list[pd.Series] = []
                if stock_rebal_day:
                    # Smooth weekly pair switches at target formation. This is
                    # a convex pair-gross blend, never a sleeve renormalization.
                    selected_stock, blend_audit = _blend_stock_structural_targets(
                        old_stock,
                        raw_stock,
                        confirmed_members=confirmed_stock_members,
                        hard_exit_members=latest_hard_stock,
                        alpha=structural_blend_alpha,
                    )
                    raw_plan_stock_gross_day = float(
                        raw_stock[["etf_usd", "underlying_usd"]]
                        .abs()
                        .sum()
                        .sum()
                    ) if not raw_stock.empty else 0.0
                    blended_stock_structural_gross_day = float(
                        selected_stock[["etf_usd", "underlying_usd"]]
                        .abs()
                        .sum()
                        .sum()
                    ) if not selected_stock.empty else 0.0
                    for blend_row in blend_audit:
                        etf = str(blend_row["ETF"])
                        cur_g = (
                            abs(float(cur.at[etf, "etf_usd"]))
                            + abs(float(cur.at[etf, "underlying_usd"]))
                            if etf in cur.index
                            else 0.0
                        )
                        pending_target_rows.append(
                            {
                                "date": d,
                                "plan_date": str(active_plan_date.date()),
                                "ETF": etf,
                                "Underlying": str(
                                    selected_stock.at[etf, "Underlying"]
                                    if etf in selected_stock.index
                                    else old_stock.at[etf, "Underlying"]
                                    if etf in old_stock.index
                                    else ""
                                ),
                                "sleeve": str(blend_row["sleeve"]),
                                "current_gross_usd": cur_g,
                                "desired_gross_usd": float(
                                    blend_row["blended_structural_gross_usd"]
                                ),
                                "next_gross_usd": cur_g,
                                "hedge_net_pct_before": np.nan,
                                "hedge_net_pct_after": np.nan,
                                "target_age": int(pair_request_age.get(etf, 0)),
                                "requested_turnover_usd": abs(
                                    float(blend_row["blended_structural_gross_usd"])
                                    - float(blend_row["prior_structural_gross_usd"])
                                ),
                                "allocated_turnover_usd": 0.0,
                                "deferred_turnover_usd": 0.0,
                                "block_reason": None,
                                "priority": "target_formation",
                                "hedge_repair_leg": None,
                                "hedge_reserve_usd": np.nan,
                                "reserve_committed_usd": np.nan,
                                "tracking_budget_usd": np.nan,
                                **blend_row,
                            }
                        )
                else:
                    # Freeze gross between decisions while allowing current
                    # Delta, leg mix, borrow, ADV, and locate metadata to refresh.
                    for etf, row in old_stock.iterrows():
                        if _norm(etf) in latest_hard_stock:
                            continue
                        if (
                            etf in raw_stock.index
                            and _norm(etf) in confirmed_stock_members
                        ):
                            row = _refresh_stock_target_metadata(
                                row, raw_stock.loc[etf]
                            )
                        selected_rows.append(row.rename(etf))
                    selected_stock = (
                        pd.DataFrame(selected_rows)
                        if selected_rows
                        else pd.DataFrame(columns=target.columns)
                    )
                    selected_stock.index.name = target.index.name
                    raw_plan_stock_gross_day = float(
                        raw_stock[["etf_usd", "underlying_usd"]]
                        .abs()
                        .sum()
                        .sum()
                    ) if not raw_stock.empty else 0.0
                    blended_stock_structural_gross_day = float(
                        selected_stock[["etf_usd", "underlying_usd"]]
                        .abs()
                        .sum()
                        .sum()
                    ) if not selected_stock.empty else 0.0

                confirmed_raw_mask = np.asarray(
                    [
                        _norm(e) in confirmed_stock_members
                        for e in raw_stock.index
                    ],
                    dtype=bool,
                )
                unconfirmed = raw_stock.loc[~confirmed_raw_mask]
                for etf, row in unconfirmed.iterrows():
                    cur_g = (
                        abs(float(cur.at[etf, "etf_usd"]))
                        + abs(float(cur.at[etf, "underlying_usd"]))
                        if etf in cur.index
                        else 0.0
                    )
                    des_g = abs(float(row["etf_usd"])) + abs(
                        float(row["underlying_usd"])
                    )
                    pending_target_rows.append(
                        {
                            "date": d, "plan_date": str(active_plan_date.date()),
                            "ETF": etf, "Underlying": str(row.get("Underlying", "")),
                            "sleeve": str(row.get("sleeve", "")),
                            "current_gross_usd": cur_g, "desired_gross_usd": des_g,
                            "next_gross_usd": cur_g, "hedge_net_pct_before": np.nan,
                            "hedge_net_pct_after": np.nan,
                            "target_age": int(stock_presence_streak.get(_norm(etf), 0)),
                            "requested_turnover_usd": abs(des_g - cur_g),
                            "allocated_turnover_usd": 0.0,
                            "deferred_turnover_usd": abs(des_g - cur_g),
                            "block_reason": "entry_confirmation",
                            "priority": "confirmation", "hedge_repair_leg": None,
                            "hedge_reserve_usd": np.nan,
                            "reserve_committed_usd": np.nan,
                            "tracking_budget_usd": np.nan,
                        }
                    )
                raw_names = set(raw_stock.index)
                for etf, row in old_stock.iterrows():
                    if (
                        _norm(etf) not in confirmed_stock_members
                        or etf in raw_names
                        or stock_absence_streak.get(_norm(etf), 0) <= 0
                    ):
                        continue
                    cur_g = (
                        abs(float(cur.at[etf, "etf_usd"]))
                        + abs(float(cur.at[etf, "underlying_usd"]))
                        if etf in cur.index
                        else 0.0
                    )
                    pending_target_rows.append(
                        {
                            "date": d, "plan_date": str(active_plan_date.date()),
                            "ETF": etf, "Underlying": str(row.get("Underlying", "")),
                            "sleeve": str(row.get("sleeve", "")),
                            "current_gross_usd": cur_g, "desired_gross_usd": 0.0,
                            "next_gross_usd": cur_g,
                            "hedge_net_pct_before": np.nan,
                            "hedge_net_pct_after": np.nan,
                            "target_age": int(stock_absence_streak.get(_norm(etf), 0)),
                            "requested_turnover_usd": cur_g,
                            "allocated_turnover_usd": 0.0,
                            "deferred_turnover_usd": cur_g,
                            "block_reason": "drop_confirmation",
                            "priority": "confirmation", "hedge_repair_leg": None,
                            "hedge_reserve_usd": np.nan,
                            "reserve_committed_usd": np.nan,
                            "tracking_budget_usd": np.nan,
                        }
                    )

                target = pd.concat([non_stock, selected_stock], axis=0)
                old_persistent = persistent_pair_targets
                for etf in set(target.index) | set(old_persistent.index):
                    old_gross = (
                        abs(float(old_persistent.at[etf, "etf_usd"]))
                        + abs(float(old_persistent.at[etf, "underlying_usd"]))
                        if etf in old_persistent.index
                        else None
                    )
                    new_gross = (
                        abs(float(target.at[etf, "etf_usd"]))
                        + abs(float(target.at[etf, "underlying_usd"]))
                        if etf in target.index
                        else 0.0
                    )
                    if old_gross is None or abs(old_gross - new_gross) > 1e-6:
                        pair_request_age[str(etf)] = 0
                persistent_pair_targets = target.copy()
            old_names = set(cur.index)
            new_names = set(target.index)
            n_added = len(new_names - old_names)
            n_exited = len(old_names - new_names)
            n_resized = 0
            for etf in old_names & new_names:
                old_g = abs(float(cur.at[etf, "etf_usd"])) + abs(float(cur.at[etf, "underlying_usd"]))
                new_g = abs(float(target.at[etf, "etf_usd"])) + abs(float(target.at[etf, "underlying_usd"]))
                if abs(new_g - old_g) > 250.0:
                    n_resized += 1
            n_resized_executed = 0
            keep_open_etfs: set[str] = set()
            if active_plan is not None and not active_plan.empty and "keep_open" in active_plan.columns:
                ko = active_plan.loc[active_plan["keep_open"].astype(bool), "ETF"].astype(str)
                keep_open_etfs = set(ko.map(_norm).tolist())
            reduce_only_etfs: set[str] = set()
            hard_exit_etfs: set[str] = set()
            if active_plan is not None and not active_plan.empty:
                policy = active_plan.get(
                    "execution_policy", pd.Series("normal", index=active_plan.index)
                ).astype(str).str.lower()
                reduce_only_etfs = set(
                    active_plan.loc[policy.eq("reduce_only"), "ETF"].map(_norm).tolist()
                )
                hard_exit_etfs = set(
                    active_plan.loc[policy.eq("hard_exit"), "ETF"].map(_norm).tolist()
                )
                if "hard_exit" in active_plan.columns:
                    hard_exit_etfs |= set(
                        active_plan.loc[active_plan["hard_exit"].astype(bool), "ETF"]
                        .map(_norm)
                        .tolist()
                    )
            union = sorted(
                set(cur.index) | set(target.index) | keep_open_etfs | reduce_only_etfs | hard_exit_etfs
            )
            plan_present_etfs: set[str] = set()
            if active_plan is not None and not active_plan.empty and "ETF" in active_plan.columns:
                plan_present_etfs = set(active_plan["ETF"].map(_norm).tolist())
            b4_plan_exec_gross = _b4_plan_executable_gross(active_plan)
            b4_plan_empty = b4_plan_exec_gross <= 1e-6
            membership_day = bool(d in membership_days)
            any_trade = False
            pending_fills: list[dict[str, Any]] = []
            n_hedge_repairs_today = 0
            hedge_repair_turnover_today = 0.0
            if hedge_safe:
                confirmed_desired_gross_ref = (
                    float(
                        persistent_pair_targets[
                            ["etf_usd", "underlying_usd"]
                        ].abs().sum().sum()
                    )
                    if not persistent_pair_targets.empty
                    else 0.0
                )
            elif target_confirmed_today:
                confirmed_desired_gross_ref = (
                    float(target[["etf_usd", "underlying_usd"]].abs().sum().sum())
                    if not target.empty
                    else 0.0
                )
            pair_gross_start = (
                float(cur[["etf_usd", "underlying_usd"]].abs().sum().sum())
                if not cur.empty
                else 0.0
            )
            turnover_reference_gross_day = _turnover_budget_reference_gross(
                pair_gross_start,
                confirmed_desired_gross_ref,
                hedge_safe=hedge_safe,
            )
            # Empty book: do not cap the opening establish (no gross base yet).
            if pace_on and pair_gross_start > 1e-9:
                day_turn_budget = (
                    max(0.0, pace_turn_pct) * turnover_reference_gross_day
                )
            else:
                day_turn_budget = float("inf")
            turnover_budget_day = (
                float(day_turn_budget) if pace_on and np.isfinite(day_turn_budget) else 0.0
            )
            hedge_reserve_usd = (
                hedge_reserve * float(day_turn_budget)
                if hedge_safe and np.isfinite(day_turn_budget)
                else 0.0
            )
            anticipated_hedge_turnover = 0.0
            # rebal_only freezes structural gross off-clock. Phase-3 residual
            # repair midweek only when midweek_hedge_repair is on.
            allow_stock_hedge_repair = bool(
                hedge_safe
                and (
                    midweek_mode != "rebal_only"
                    or stock_rebal_day
                    or allow_midweek_hedge_repair
                )
            )
            if allow_stock_hedge_repair and not cur.empty:
                phase3_cur = _phase3_stock_residual_book(cur)
                for _, grp0 in phase3_cur.groupby(
                    phase3_cur["Underlying"].map(_norm)
                ):
                    exps = [
                        _delta_adjusted_pair_exposure(
                            r.get("etf_usd", 0.0),
                            r.get("underlying_usd", 0.0),
                            r.get("Delta"),
                        )
                        for _, r in grp0.iterrows()
                    ]
                    if any(x is None for x in exps):
                        continue
                    net0 = float(sum(x[0] for x in exps if x is not None))
                    gross0 = float(sum(x[1] for x in exps if x is not None))
                    corr0 = _hedge_correction_usd(
                        net_notional=net0,
                        reference_gross=gross0,
                        long_trigger_net_pct=hedge_long_trigger_net_pct,
                        long_target_net_pct=hedge_long_target_net_pct,
                        short_trigger_net_pct=hedge_short_trigger_net_pct,
                        short_target_net_pct=hedge_short_target_net_pct,
                    )
                    _, _, repair0, _ = _select_live_semantic_hedge_repair(
                        grp0, correction_usd=corr0
                    )
                    anticipated_hedge_turnover += abs(repair0)
            reserve_committed_usd = min(
                hedge_reserve_usd, anticipated_hedge_turnover
            )
            tracking_budget_usd = (
                max(0.0, float(day_turn_budget) - reserve_committed_usd)
                if np.isfinite(day_turn_budget)
                else float("inf")
            )
            n_deferred_today = 0
            for etf in union:
                # No close bar -> cannot fill or liquidate; carry the old mark.
                if etf not in panel or d not in panel[etf].index:
                    blocked_pairs += 1
                    continue
                old = cur.loc[etf] if etf in cur.index else None
                new = target.loc[etf] if etf in target.index else None
                old_a = float(old["etf_usd"]) if old is not None else 0.0
                old_b = float(old["underlying_usd"]) if old is not None else 0.0
                new_a = float(new["etf_usd"]) if new is not None else 0.0
                new_b = float(new["underlying_usd"]) if new is not None else 0.0
                constraint = None
                sleeve = ""
                if new is not None and "sleeve" in new.index:
                    sleeve = str(new.get("sleeve", "") or "")
                elif old is not None:
                    sleeve = str(old.get("sleeve", "") or "")
                is_b4 = sleeve == B4_SLEEVE or (
                    new is not None
                    and float(pd.to_numeric(new.get("Delta"), errors="coerce") or 0.0) < 0
                    and sleeve in ("", B4_SLEEVE)
                )
                is_b5 = sleeve == B5_SLEEVE
                is_ratchet_sleeve = is_b4 or is_b5

                # Delist flatten: force exit on/after last_trade_date.
                force_exit = False
                last_tr = delist_map.get(_norm(etf))
                if (
                    apply_delist_flatten
                    and last_tr is not None
                    and pd.Timestamp(d) >= pd.Timestamp(last_tr)
                    and (abs(old_a) + abs(old_b) > 1e-9)
                ):
                    new_a, new_b = 0.0, 0.0
                    force_exit = True
                    if new is None and old is not None:
                        new = old.copy()
                        new["etf_usd"] = 0.0
                        new["underlying_usd"] = 0.0
                        new["gross_usd"] = 0.0
                    n_delist_flat += 1
                else:
                    # Hard exit: force flatten (borrow hard-exit path).
                    if etf in hard_exit_etfs:
                        if old is None:
                            continue
                        new_a, new_b = 0.0, 0.0
                        force_exit = True
                        if new is None:
                            new = old.copy()
                        new["etf_usd"] = 0.0
                        new["underlying_usd"] = 0.0
                        new["gross_usd"] = 0.0
                        new["execution_policy"] = "hard_exit"
                    # Purgatory reduce-only: trim toward model_*; never increase
                    # gross. Missing/zero model → share-hold by default.
                    elif etf in reduce_only_etfs:
                        if old is None:
                            continue
                        plan_row = _plan_row_for_etf(active_plan, etf)
                        des_a, des_b, des_g = _model_legs_from_plan_row(plan_row)
                        model_missing = not np.isfinite(des_g)
                        model_flat = (not model_missing) and float(des_g) <= 1e-9
                        if model_missing or (model_flat and purg_zero_pol == "hold"):
                            continue
                        if model_flat and purg_zero_pol == "exit":
                            des_a, des_b = 0.0, 0.0
                        constraint = constrain_pair_targets(
                            desired_underlying_usd=des_b if np.isfinite(des_b) else 0.0,
                            desired_etf_usd=des_a if np.isfinite(des_a) else 0.0,
                            current_underlying_usd=old_b,
                            current_etf_usd=old_a,
                        )
                        new_a = float(constraint.constrained_etf_usd)
                        new_b = float(constraint.constrained_underlying_usd)
                        if new is None:
                            new = old.copy()
                        new["etf_usd"] = new_a
                        new["underlying_usd"] = new_b
                        new["gross_usd"] = abs(new_a) + abs(new_b)
                        new["execution_policy"] = "reduce_only"

                    # Purgatory keep-open: hold current shares (do not liquidate).
                    if etf in keep_open_etfs and (new is None or abs(new_a) + abs(new_b) <= 1e-9):
                        if old is None:
                            continue
                        new_a, new_b = old_a, old_b
                        if new is None:
                            new = old.copy()
                            new["etf_usd"] = old_a
                            new["underlying_usd"] = old_b
                            new["gross_usd"] = abs(old_a) + abs(old_b)

                    # B4 cadence + membership clock + bands + ratchet.
                    if (
                        use_b4_cadence
                        and is_b4
                        and etf not in keep_open_etfs
                        and etf not in reduce_only_etfs
                        and etf not in hard_exit_etfs
                    ):
                        plan_row = _plan_row_for_etf(active_plan, etf)
                        present = _norm(etf) in plan_present_etfs
                        exec_gross = abs(new_a) + abs(new_b)
                        entering = old is None and exec_gross > 1e-9
                        # True drop = gone from plan entirely (not purgatory row).
                        exiting = (
                            old is not None
                            and abs(old_a) + abs(old_b) > 1e-9
                            and not present
                        )
                        cad = b4_cadence.get(_norm(etf), {})
                        on_cadence = bool(cad.get("rebal")) and d in cad["rebal"]

                        if entering or exiting:
                            if (
                                exiting
                                and b4_plan_empty
                                and empty_plan_pol == "hold"
                            ):
                                # Empty/zero B4 plan (archive gap, sizing fail):
                                # do not liquidate the held sleeve into cash.
                                n_b4_empty_plan_holds += 1
                                continue
                            if membership_mode not in {"every_plan", "every", "all", "off"} and (
                                not membership_day
                            ):
                                n_b4_membership_deferred += 1
                                continue
                            if exiting:
                                new_a, new_b = 0.0, 0.0
                                force_exit = True
                                if new is None and old is not None:
                                    new = old.copy()
                                if new is not None:
                                    new["etf_usd"] = 0.0
                                    new["underlying_usd"] = 0.0
                                    new["gross_usd"] = 0.0
                            # entering: keep plan target legs as-is
                        elif on_cadence:
                            gross = abs(new_a) + abs(new_b)
                            if gross <= 1e-9 and old is not None:
                                gross = abs(old_a) + abs(old_b)
                            beta = abs(
                                float(
                                    pd.to_numeric(
                                        new.get("Delta") if new is not None else np.nan,
                                        errors="coerce",
                                    )
                                    or (
                                        pd.to_numeric(old.get("Delta"), errors="coerce")
                                        if old is not None
                                        else np.nan
                                    )
                                    or 2.0
                                )
                            )
                            h_ser = cad.get("h")
                            h_mid = float(cad.get("h_mid", 0.45) or 0.45)
                            if h_ser is not None and d in h_ser.index and np.isfinite(h_ser.loc[d]):
                                h = float(h_ser.loc[d])
                            elif h_ser is not None and len(h_ser.dropna()):
                                h = float(h_ser.reindex(pd.DatetimeIndex([d])).ffill().iloc[0])
                                if not np.isfinite(h):
                                    h = h_mid
                            else:
                                h = h_mid
                            raw_a, raw_b = b4_leg_targets_from_gross(gross, h, beta)
                            released = (
                                _truthy(plan_row.get("ratchet_released", False))
                                if plan_row is not None
                                else False
                            )
                            if ratchet_guard and old is not None:
                                raw_a, raw_b = _apply_b4_inverse_floor(
                                    old_a,
                                    raw_a,
                                    raw_b,
                                    h=h,
                                    beta_abs=beta,
                                    ratchet_released=released,
                                )
                            new_a, new_b = raw_a, raw_b
                            if new is None and old is not None:
                                new = old.copy()
                            if new is not None:
                                new["etf_usd"] = new_a
                                new["underlying_usd"] = new_b
                                new["gross_usd"] = abs(new_a) + abs(new_b)
                            if (
                                apply_b4_bands
                                and use_resize_bands
                                and old is not None
                            ):
                                new_a = _resize_band_target(
                                    old_a, new_a, enter_band_pct=enter_band_pct,
                                    exit_band_pct=exit_band_pct, min_trade_usd=min_trade_usd,
                                )
                                new_b = _resize_band_target(
                                    old_b, new_b, enter_band_pct=enter_band_pct,
                                    exit_band_pct=exit_band_pct, min_trade_usd=min_trade_usd,
                                )
                            if ratchet_guard and old is not None:
                                new_a, new_b, rsn = _apply_b4_ratchet_cover_guard(
                                    old_a,
                                    new_a,
                                    new_b,
                                    plan_row=plan_row,
                                    allow_inverse_cover=allow_inv_cover,
                                    h=h,
                                    beta_abs=beta,
                                )
                                if rsn == "pin":
                                    n_b4_ratchet_pins += 1
                            if new is not None:
                                new["etf_usd"] = new_a
                                new["underlying_usd"] = new_b
                                new["gross_usd"] = abs(new_a) + abs(new_b)
                            n_b4_cadence_rebals += 1
                        else:
                            # Share-hold; skip Friday plan-leg chase for B4.
                            continue
                    elif (
                        not stock_rebal_day
                        and not (
                            hedge_safe
                            and not is_b4
                        )
                        and not (is_b4 and etf in reduce_only_etfs)
                        and etf not in hard_exit_etfs
                    ):
                        # Non-B4 (or legacy B4): only trade on weekly clock.
                        continue

                    # Off-clock B1/B2: rebal_only share-holds entirely between
                    # operator days; hedge_only allows incomplete entry/exit
                    # ramps only (no mark-chase resize). Phase-3 hedge repair
                    # is gated separately below.
                    if (
                        hedge_safe
                        and midweek_mode in {"rebal_only", "hedge_only"}
                        and not stock_rebal_day
                        and not is_b4
                        and not force_exit
                        and etf not in hard_exit_etfs
                        and etf not in reduce_only_etfs
                    ):
                        if midweek_mode == "rebal_only":
                            continue
                        age_mw = int(pair_request_age.get(etf, 0))
                        old_g_mw = abs(old_a) + abs(old_b)
                        des_g_mw = abs(new_a) + abs(new_b)
                        incomplete_entry = (
                            des_g_mw > old_g_mw + 1e-9 and age_mw < entry_sessions
                        )
                        incomplete_reduction = (
                            des_g_mw < old_g_mw - 1e-9
                            and age_mw < reduction_sessions
                        )
                        if not (incomplete_entry or incomplete_reduction):
                            continue

                    if (
                        use_resize_bands
                        and old is not None
                        and new is not None
                        and etf not in keep_open_etfs
                        and etf not in reduce_only_etfs
                        and etf not in hard_exit_etfs
                        and not (use_b4_cadence and is_b4)
                    ):
                        new_a = _resize_band_target(
                            old_a, new_a, enter_band_pct=enter_band_pct,
                            exit_band_pct=exit_band_pct, min_trade_usd=min_trade_usd,
                        )
                        new_b = _resize_band_target(
                            old_b, new_b, enter_band_pct=enter_band_pct,
                            exit_band_pct=exit_band_pct, min_trade_usd=min_trade_usd,
                        )
                        # B5 (and any non-cadence B4) still get ratchet cover guard.
                        if ratchet_guard and is_ratchet_sleeve:
                            plan_row = _plan_row_for_etf(active_plan, etf)
                            beta = abs(
                                float(
                                    pd.to_numeric(
                                        new.get("Delta") if new is not None else np.nan,
                                        errors="coerce",
                                    )
                                    or 2.0
                                )
                            )
                            h_guess = (
                                abs(new_b) / (abs(new_a) * beta)
                                if abs(new_a) > 1e-9 and beta > 1e-9
                                else 0.45
                            )
                            new_a, new_b, rsn = _apply_b4_ratchet_cover_guard(
                                old_a,
                                new_a,
                                new_b,
                                plan_row=plan_row,
                                allow_inverse_cover=allow_inv_cover,
                                h=h_guess,
                                beta_abs=beta,
                            )
                            if rsn == "pin":
                                n_b4_ratchet_pins += 1
                            new["etf_usd"] = new_a
                            new["underlying_usd"] = new_b
                            new["gross_usd"] = abs(new_a) + abs(new_b)

                if hedge_safe and new is None and old is not None and not force_exit:
                    # Preserve row metadata while an ordinary confirmed drop
                    # ramps toward zero over multiple sessions.
                    new = old.copy()

                # Controller ramp. Only hard exits/delists bypass it; ordinary
                # drops use the reduction schedule. Both legs always share one
                # completion fraction.
                desired_a, desired_b = float(new_a), float(new_b)
                is_full_exit = (
                    abs(desired_a) + abs(desired_b) <= 1e-9
                    and abs(old_a) + abs(old_b) > 1e-9
                )
                if (
                    hedge_safe
                    and not force_exit
                ):
                    old_gross_for_ramp = abs(old_a) + abs(old_b)
                    desired_gross_for_ramp = abs(desired_a) + abs(desired_b)
                    age = int(pair_request_age.get(etf, 0))
                    if midweek_mode == "rebal_only":
                        # Operator-day only path: close a fraction of the
                        # remaining gap atomically (both legs). Rate < 1
                        # converges over multiple 5d sessions without a
                        # one-day full chase.
                        ramp_fraction = resize_gap_rate
                    elif midweek_mode == "hedge_only" and stock_rebal_day:
                        # Same gradual gross step on the weekly clock;
                        # midweek incomplete ramps (if sessions > 1) use
                        # the age-based branch below.
                        ramp_fraction = resize_gap_rate
                    elif old_gross_for_ramp <= 1e-9:
                        remaining_sessions = max(entry_sessions - age, 1)
                        ramp_fraction = 1.0 / remaining_sessions
                    elif desired_gross_for_ramp < old_gross_for_ramp - 1e-9:
                        remaining_sessions = max(reduction_sessions - age, 1)
                        ramp_fraction = 1.0 / remaining_sessions
                    else:
                        ramp_fraction = resize_gap_rate
                    new_a, new_b = _advance_pair_atomic(
                        old_a,
                        old_b,
                        desired_a,
                        desired_b,
                        completion_fraction=ramp_fraction,
                    )
                    if new is not None:
                        new = new.copy()
                        new["etf_usd"] = new_a
                        new["underlying_usd"] = new_b
                        new["gross_usd"] = abs(new_a) + abs(new_b)
                elif (
                    legacy_pace
                    and not force_exit
                    and not is_full_exit
                    and old is not None
                ):
                    new_a = _pace_leg(
                        old_a,
                        new_a,
                        max_leg_step_pct=pace_leg_pct,
                        min_trade_usd=min_trade_usd,
                    )
                    new_b = _pace_leg(
                        old_b,
                        new_b,
                        max_leg_step_pct=pace_leg_pct,
                        min_trade_usd=min_trade_usd,
                    )
                    if new is not None:
                        new = new.copy()
                        new["etf_usd"] = new_a
                        new["underlying_usd"] = new_b
                        new["gross_usd"] = abs(new_a) + abs(new_b)

                turn_a = abs(new_a - old_a)
                turn_b = abs(new_b - old_b)
                pair_turn = turn_a + turn_b
                if pair_turn <= 1e-9 and not (
                    new is None or (abs(new_a) + abs(new_b) <= 1e-9 and abs(old_a) + abs(old_b) > 1e-9)
                ):
                    continue
                if force_exit or (is_full_exit and not hedge_safe):
                    priority = 0
                elif old is None:
                    priority = 1
                else:
                    priority = 2
                old_gross = abs(old_a) + abs(old_b)
                desired_gross = abs(desired_a) + abs(desired_b)
                if force_exit or (is_full_exit and not hedge_safe):
                    risk_class = "hard_exit"
                elif desired_gross < old_gross - 1e-9:
                    risk_class = "gross_reduction"
                elif old_gross <= 1e-9 and desired_gross > 1e-9:
                    risk_class = "growth"
                else:
                    risk_class = "resize"
                pending_fills.append(
                    {
                        "etf": etf,
                        "old_a": old_a,
                        "old_b": old_b,
                        "new_a": new_a,
                        "new_b": new_b,
                        "desired_a": desired_a,
                        "desired_b": desired_b,
                        "new": new,
                        "old": old,
                        "priority": priority,
                        "risk_class": risk_class,
                        "age": int(pair_request_age.get(etf, 0)),
                        "constraint": constraint,
                        "in_reduce_only": etf in reduce_only_etfs,
                    }
                )

            controller_candidates = [dict(f) for f in pending_fills]
            block_reason_by_etf: dict[str, str] = {}
            adv_limited_etfs: set[str] = set()

            # If delta is unavailable, hedge feasibility cannot be established.
            # hedge_safe_v1 permits reductions/exits but suppresses gross growth.
            if hedge_safe and pending_fills:
                safe_candidates: list[dict[str, Any]] = []
                for fill in pending_fills:
                    etf_key = str(fill["etf"])
                    ref = fill.get("new") if fill.get("new") is not None else fill.get("old")
                    exposure = _delta_adjusted_pair_exposure(
                        float(fill["new_a"]),
                        float(fill["new_b"]),
                        ref.get("Delta") if ref is not None else np.nan,
                    )
                    if exposure is None and fill.get("risk_class") == "growth":
                        n_growth_blocked_hedge_infeasible += 1
                        block_reason_by_etf[etf_key] = "hedge_infeasible_missing_delta"
                        continue
                    px_row = panel[etf_key].loc[d]
                    capped, adv_reason = _apply_adv_participation_cap(
                        fill,
                        etf_price=float(px_row["a_px"]),
                        underlying_price=float(px_row["b_px"]),
                        adv_participation_pct=adv_pct,
                    )
                    if adv_reason:
                        adv_limited_etfs.add(etf_key)
                        block_reason_by_etf[etf_key] = adv_reason
                    if _fill_turnover(capped) > 1e-9:
                        safe_candidates.append(capped)
                pending_fills = safe_candidates

            # Daily turnover allocator.  The versioned controller uses risk
            # classes and ages; legacy preserves the old exit/establish/resize
            # ordering and off performs full chase.
            deferred_fills: list[dict[str, Any]] = []
            if hedge_safe and pending_fills:
                pending_fills, deferred_fills = _allocate_hedge_safe_budget(
                    pending_fills,
                    budget_usd=tracking_budget_usd,
                    establish_budget_frac=pace_est_frac,
                )
                n_deferred_today = len(deferred_fills)
                for fill in deferred_fills:
                    block_reason_by_etf.setdefault(str(fill["etf"]), "turnover_budget")
                n_deferred_today += len(adv_limited_etfs)
                n_deferred_pace += n_deferred_today
            elif legacy_pace and pending_fills:
                pending_fills, n_deferred_today = _allocate_turnover_budget(
                    pending_fills,
                    budget_usd=day_turn_budget,
                    establish_budget_frac=pace_est_frac,
                )
                n_deferred_pace += int(n_deferred_today)

            if hedge_safe:
                deferred_etfs = {str(f["etf"]) for f in deferred_fills}
                accepted_etfs = {str(f["etf"]) for f in pending_fills}
                candidate_by_etf = {str(f["etf"]): f for f in controller_candidates}
                for etf, original in candidate_by_etf.items():
                    accepted = next(
                        (f for f in pending_fills if str(f["etf"]) == etf), None
                    )
                    complete = (
                        accepted is not None
                        and abs(float(accepted["new_a"]) - float(original["desired_a"])) <= 1e-6
                        and abs(float(accepted["new_b"]) - float(original["desired_b"])) <= 1e-6
                    )
                    pair_request_age[etf] = (
                        0 if complete else int(pair_request_age.get(etf, 0)) + 1
                    )

                # Phase-3-equivalent hedge check on the post-budget projected
                # book. Repairs alter one underlying leg and bypass the normal
                # turnover budget. Under rebal_only without midweek_hedge_repair,
                # skip off operator days.
                projected = cur.copy()
                fill_by_etf = {str(f["etf"]): f for f in pending_fills}
                for etf, fill in fill_by_etf.items():
                    if fill.get("new") is None or (
                        abs(float(fill["new_a"])) + abs(float(fill["new_b"])) <= 1e-9
                    ):
                        if etf in projected.index:
                            projected = projected.drop(index=etf)
                        continue
                    row = fill["new"].copy()
                    row["etf_usd"] = float(fill["new_a"])
                    row["underlying_usd"] = float(fill["new_b"])
                    row["gross_usd"] = abs(float(fill["new_a"])) + abs(float(fill["new_b"]))
                    projected.loc[etf, pos_cols] = [row.get(cn, np.nan) for cn in pos_cols]

                if allow_stock_hedge_repair:
                  phase3_projected = _phase3_stock_residual_book(projected)
                  for under, grp in phase3_projected.groupby(
                    phase3_projected["Underlying"].map(_norm)
                  ):
                    net = 0.0
                    gross = 0.0
                    feasible = True
                    for _, row in grp.iterrows():
                        exp = _delta_adjusted_pair_exposure(
                            row.get("etf_usd", 0.0),
                            row.get("underlying_usd", 0.0),
                            row.get("Delta"),
                        )
                        if exp is None:
                            feasible = False
                            break
                        net += exp[0]
                        gross += exp[1]
                    if not feasible or gross <= 1e-9:
                        continue
                    correction = _hedge_correction_usd(
                        net_notional=net,
                        reference_gross=gross,
                        long_trigger_net_pct=hedge_long_trigger_net_pct,
                        long_target_net_pct=hedge_long_target_net_pct,
                        short_trigger_net_pct=hedge_short_trigger_net_pct,
                        short_target_net_pct=hedge_short_target_net_pct,
                    )
                    if abs(correction) <= 1e-9:
                        continue
                    repair_grp = grp.loc[
                        [
                            e
                            for e in grp.index
                            if str(e) in panel and d in panel[str(e)].index
                        ]
                    ]
                    etf_prices = {
                        str(e): float(panel[str(e)].loc[d, "a_px"])
                        for e in repair_grp.index
                        if str(e) in panel and d in panel[str(e)].index
                    }
                    etf, repair_leg, repair_change, repair_reason = (
                        _select_live_semantic_hedge_repair(
                            repair_grp,
                            correction_usd=correction,
                            etf_prices=etf_prices,
                        )
                    )
                    if etf is None or repair_leg is None:
                        for e in grp.index:
                            block_reason_by_etf.setdefault(
                                str(e), repair_reason or "hedge_infeasible"
                            )
                        continue
                    selected = grp.loc[etf]
                    # Hedge overrides bypass turnover budget but still respect
                    # observable ADV. Missing ADV remains a safe no-op.
                    adv_col = (
                        "etf_adv_usd" if repair_leg == "etf" else "underlying_adv_usd"
                    )
                    adv_usd = pd.to_numeric(selected.get(adv_col), errors="coerce")
                    if pd.notna(adv_usd) and float(adv_usd) > 0 and adv_pct > 0:
                        cap = adv_pct * float(adv_usd)
                        repair_change = float(
                            np.sign(repair_change) * min(abs(repair_change), cap)
                        )
                        if abs(repair_change) + 1e-9 < abs(correction):
                            block_reason_by_etf[etf] = "hedge_adv_cap"
                    if etf in fill_by_etf:
                        fill = fill_by_etf[etf]
                        key = "new_a" if repair_leg == "etf" else "new_b"
                        fill[key] = float(fill[key]) + repair_change
                        fill["risk_class"] = "hedge"
                        fill["hedge_repair_usd"] = repair_change
                        fill["hedge_repair_leg"] = repair_leg
                        fill["hedge_repair_reason"] = repair_reason
                    else:
                        old = cur.loc[etf]
                        new = old.copy()
                        fill = {
                            "etf": etf,
                            "old_a": float(old["etf_usd"]),
                            "old_b": float(old["underlying_usd"]),
                            "new_a": float(old["etf_usd"])
                            + (repair_change if repair_leg == "etf" else 0.0),
                            "new_b": float(old["underlying_usd"])
                            + (repair_change if repair_leg == "underlying" else 0.0),
                            "desired_a": float(old["etf_usd"]),
                            "desired_b": float(old["underlying_usd"]),
                            "new": new,
                            "old": old,
                            "priority": 0,
                            "risk_class": "hedge",
                            "age": 0,
                            "constraint": None,
                            "in_reduce_only": False,
                            "hedge_repair_usd": repair_change,
                            "hedge_repair_leg": repair_leg,
                            "hedge_repair_reason": repair_reason,
                        }
                        pending_fills.append(fill)
                        fill_by_etf[etf] = fill
                    n_hedge_repairs += 1
                    hedge_repair_turnover += abs(repair_change)
                    n_hedge_repairs_today += 1
                    hedge_repair_turnover_today += abs(repair_change)

            if hedge_safe:
                requested_map = {str(f["etf"]): f for f in controller_candidates}
                accepted_map = {str(f["etf"]): f for f in pending_fills}
                for etf in sorted(set(requested_map) | set(accepted_map)):
                    req = requested_map.get(etf, accepted_map.get(etf))
                    acc = accepted_map.get(etf)
                    if req is None:
                        continue
                    oa, ob = float(req["old_a"]), float(req["old_b"])
                    da = float(req.get("desired_a", req.get("new_a", oa)))
                    db = float(req.get("desired_b", req.get("new_b", ob)))
                    na = float(acc["new_a"]) if acc is not None else oa
                    nb = float(acc["new_b"]) if acc is not None else ob
                    ref = (
                        req.get("new")
                        if req.get("new") is not None
                        else req.get("old")
                    )
                    delta = ref.get("Delta") if ref is not None else np.nan
                    before_exp = _delta_adjusted_pair_exposure(oa, ob, delta)
                    after_exp = _delta_adjusted_pair_exposure(na, nb, delta)
                    before_pct = (
                        before_exp[0] / before_exp[1]
                        if before_exp is not None and before_exp[1] > 1e-9
                        else np.nan
                    )
                    after_pct = (
                        after_exp[0] / after_exp[1]
                        if after_exp is not None and after_exp[1] > 1e-9
                        else np.nan
                    )
                    requested_turn = abs(da - oa) + abs(db - ob)
                    allocated_turn = (
                        abs(na - oa) + abs(nb - ob) if acc is not None else 0.0
                    )
                    pending_target_rows.append(
                        {
                            "date": d,
                            "plan_date": (
                                str(active_plan_date.date())
                                if active_plan_date is not None
                                else None
                            ),
                            "ETF": etf,
                            "Underlying": (
                                str(ref.get("Underlying", "")) if ref is not None else ""
                            ),
                            "sleeve": (
                                str(ref.get("sleeve", "")) if ref is not None else ""
                            ),
                            "current_gross_usd": abs(oa) + abs(ob),
                            "desired_gross_usd": abs(da) + abs(db),
                            "next_gross_usd": abs(na) + abs(nb),
                            "hedge_net_pct_before": before_pct,
                            "hedge_net_pct_after": after_pct,
                            "target_age": int(req.get("age", 0) or 0),
                            "requested_turnover_usd": requested_turn,
                            "allocated_turnover_usd": allocated_turn,
                            "deferred_turnover_usd": max(
                                0.0, requested_turn - allocated_turn
                            ),
                            "block_reason": block_reason_by_etf.get(etf),
                            "priority": (
                                str(acc.get("risk_class"))
                                if acc is not None
                                else str(req.get("risk_class", ""))
                            ),
                            "hedge_repair_leg": (
                                acc.get("hedge_repair_leg") if acc is not None else None
                            ),
                            "hedge_reserve_usd": hedge_reserve_usd,
                            "reserve_committed_usd": reserve_committed_usd,
                            "tracking_budget_usd": (
                                tracking_budget_usd
                                if np.isfinite(tracking_budget_usd)
                                else np.nan
                            ),
                        }
                    )

            for fill in pending_fills:
                etf = str(fill["etf"])
                old_a = float(fill["old_a"])
                old_b = float(fill["old_b"])
                new_a = float(fill["new_a"])
                new_b = float(fill["new_b"])
                new = fill.get("new")
                old = fill.get("old")
                constraint = fill.get("constraint")
                turn_a = abs(new_a - old_a)
                turn_b = abs(new_b - old_b)
                pair_turn = turn_a + turn_b
                any_trade = True
                if fill.get("in_reduce_only"):
                    n_purgatory_reductions += 1
                    if constraint is not None:
                        purgatory_blocked_add_usd += float(constraint.blocked_add_usd)
                if old is not None and new is not None and pair_turn > 1e-9:
                    n_resized_executed += 1
                px_row = panel[etf].loc[d]
                p_a = float(px_row["a_px"])
                p_b = float(px_row["b_px"])
                comm = (
                    (turn_a / p_a if p_a > 0 else 0.0)
                    + (turn_b / p_b if p_b > 0 else 0.0)
                ) * max(float(commission_per_share), 0.0)
                pair_txn = pair_turn * slip + comm
                turnover_day += pair_turn
                txn_cost += pair_txn
                ref = new if new is not None else old
                c = _ensure_contrib(etf, ref)
                c["txn_cost_usd"] += pair_txn
                dp = _day_pair(etf, ref)
                dp["txn_cost"] += pair_txn
                if pair_turn > 1e-9:
                    c["rebalance_dates"].append(str(pd.Timestamp(d).date()))
                if new is not None:
                    c["last_target_etf_usd"] = new_a
                    c["last_target_underlying_usd"] = new_b
                    if pd.notna(new.get("Delta")):
                        c["Delta"] = float(pd.to_numeric(new.get("Delta"), errors="coerce") or np.nan)
                sl = str(ref.get("sleeve", "")) if ref is not None else ""
                if sl in sleeve_comp:
                    sleeve_comp[sl]["txn"] += pair_txn
                if new is None or (abs(new_a) + abs(new_b) <= 1e-9):
                    if etf in cur.index:
                        cur = cur.drop(index=etf)
                    c["end_etf_usd"] = 0.0
                    c["end_underlying_usd"] = 0.0
                else:
                    exec_row = new.copy()
                    exec_row["etf_usd"] = new_a
                    exec_row["underlying_usd"] = new_b
                    exec_row["gross_usd"] = abs(new_a) + abs(new_b)
                    cur.loc[etf, pos_cols] = [exec_row.get(cn, np.nan) for cn in pos_cols]
                    c["end_etf_usd"] = new_a
                    c["end_underlying_usd"] = new_b

            turnover_used_pace_day = float(turnover_day)
            if any_trade or stock_rebal_day:
                day_did_rebal = True
                equity -= txn_cost
                n_rebal += 1
                turnover_total += turnover_day
                step_l1 = turnover_day / max(equity_start, 1e-9)
                turnover_l1 += step_l1
                deployed = float(cur[["etf_usd", "underlying_usd"]].abs().sum().sum()) if not cur.empty else 0.0
                deployed_desired_ratio_day = (
                    deployed / confirmed_desired_gross_ref
                    if confirmed_desired_gross_ref > 1e-9
                    else np.nan
                )
                audit_rows.append(
                    {
                        "date": d,
                        "plan_date": str(active_plan_date.date()) if active_plan_date is not None else None,
                        "n_pairs": int(len(cur)),
                        "turnover_step": step_l1,
                        "turnover_usd": turnover_day,
                        "txn_cost_usd": txn_cost,
                        "target_planned_gross_usd": target_planned_gross,
                        "target_tradeable_gross_usd": target_tradeable_gross,
                        "raw_plan_stock_gross_usd": raw_plan_stock_gross_day,
                        "blended_stock_structural_gross_usd": (
                            blended_stock_structural_gross_day
                        ),
                        "target_blend_alpha": (
                            structural_blend_alpha if hedge_safe else np.nan
                        ),
                        "deployed_gross_usd": deployed,
                        "confirmed_desired_gross_usd": float(
                            confirmed_desired_gross_ref
                        ),
                        "turnover_reference_gross_usd": float(
                            turnover_reference_gross_day
                        ),
                        "deployed_desired_gross_ratio": float(
                            deployed_desired_ratio_day
                        )
                        if pd.notna(deployed_desired_ratio_day)
                        else np.nan,
                        "untradeable_plan_gross_usd": max(0.0, target_planned_gross - target_tradeable_gross),
                        "blocked_pairs": blocked_pairs,
                        "n_added": n_added,
                        "n_exited": n_exited,
                        "n_resize_candidates": n_resized,
                        "n_resized": n_resized_executed,
                        "turnover_budget_usd": turnover_budget_day if pace_on else np.nan,
                        "turnover_used_usd": turnover_used_pace_day,
                        "n_deferred_pace": int(n_deferred_today) if pace_on else 0,
                        "turnover_pace_mode": pace_mode,
                        "target_confirmed_today": bool(controller_confirmed_today),
                        "persistent_target_pairs": int(len(persistent_pair_targets))
                        if hedge_safe
                        else 0,
                        "n_hedge_repairs": int(n_hedge_repairs_today),
                        "hedge_repair_turnover_usd": float(hedge_repair_turnover_today),
                        "hedge_reserve_usd": float(hedge_reserve_usd),
                        "reserve_committed_usd": float(reserve_committed_usd),
                        "unused_reserve_to_tracking_usd": float(
                            max(0.0, hedge_reserve_usd - reserve_committed_usd)
                        ),
                        "tracking_budget_usd": (
                            float(tracking_budget_usd)
                            if np.isfinite(tracking_budget_usd)
                            else np.nan
                        ),
                        "max_request_age": int(max(pair_request_age.values(), default=0))
                        if hedge_safe
                        else 0,
                        "equity": equity,
                    }
                )

        # Persist open-book pairs even on flat PnL days so time series stay continuous.
        if not cur.empty:
            for etf, row in cur.iterrows():
                _day_pair(etf, row)

        for etf, dp in day_pair.items():
            etf_usd = float(cur.at[etf, "etf_usd"]) if etf in cur.index else 0.0
            und_usd = float(cur.at[etf, "underlying_usd"]) if etf in cur.index else 0.0
            daily_net = (
                float(dp["price_pnl"])
                - float(dp["borrow_cost"])
                + float(dp["short_credit"])
                - float(dp["margin_cost"])
                - float(dp["txn_cost"])
            )
            pair_daily_rows.append(
                {
                    "date": d,
                    "ETF": etf,
                    "Underlying": dp["Underlying"],
                    "sleeve": dp["sleeve"],
                    "etf_usd": etf_usd,
                    "underlying_usd": und_usd,
                    "long_usd": und_usd,
                    "short_usd": etf_usd,
                    "Delta": float(pd.to_numeric(dp.get("Delta"), errors="coerce")),
                    "hedge_ratio": (
                        abs(und_usd) / abs(etf_usd) if abs(etf_usd) > 1e-9 else np.nan
                    ),
                    "price_pnl": float(dp["price_pnl"]),
                    "borrow_cost": float(dp["borrow_cost"]),
                    "short_credit": float(dp["short_credit"]),
                    "margin_cost": float(dp["margin_cost"]),
                    "txn_cost": float(dp["txn_cost"]),
                    "daily_pnl": daily_net,
                    "is_rebalance": int(day_did_rebal),
                    "active_plan_date": (
                        str(active_plan_date.date()) if active_plan_date is not None else None
                    ),
                }
            )

        nav.iloc[i] = equity
        if net_underlyings and not cur.empty:
            long_notional, short_notional, gross_notional, net_notional = _netted_book_notionals(cur)
        else:
            long_notional = 0.0
            short_notional = 0.0
            net_notional = 0.0
            if not cur.empty:
                vals = cur[["etf_usd", "underlying_usd"]].astype(float)
                long_notional = float(vals.clip(lower=0.0).sum().sum())
                short_notional = float(-vals.clip(upper=0.0).sum().sum())
                net_notional = float(vals.sum().sum())
            gross_notional = long_notional + short_notional
        # Pair-level HHI still uses un-netted pair gross (concentration of names).
        largest_pair_share = np.nan
        top5_gross_share = np.nan
        gross_hhi = np.nan
        pair_gross_total = 0.0
        if not cur.empty:
            pair_gross = cur[["etf_usd", "underlying_usd"]].abs().sum(axis=1).sort_values(ascending=False)
            pair_gross_total = float(pair_gross.sum())
            if pair_gross_total > 0:
                shares = pair_gross / pair_gross_total
                largest_pair_share = float(shares.iloc[0])
                top5_gross_share = float(shares.head(5).sum())
                gross_hhi = float((shares**2).sum())
        internalized = 0.0
        if net_underlyings and not cur.empty:
            # Internalized = 0.5 * (sum|und| - sum|net_und|) across symbols.
            abs_sum = float(cur["underlying_usd"].abs().sum())
            net_abs = float(sum(abs(v) for v in _underlying_net_by_symbol(cur).values()))
            internalized = max(0.0, 0.5 * (abs_sum - net_abs))
        net_pnl = float(equity - equity_start)
        recon = net_pnl - (price_pnl - borrow_cost - margin_cost - txn_cost + short_credit)
        running_peak = max(running_peak, float(equity))
        row_out: dict[str, Any] = {
            "date": d,
            "book_equity_start": equity_start,
            "book_equity": equity,
            "daily_price_pnl": price_pnl,
            "daily_borrow_cost": borrow_cost,
            "daily_short_credit": short_credit,
            "daily_margin_cost": margin_cost,
            "daily_txn_cost": txn_cost,
            "daily_net_pnl": net_pnl,
            "pnl_recon_residual": recon,
            "long_notional": long_notional,
            "short_notional": short_notional,
            "gross_notional": gross_notional,
            "net_notional": net_notional,
            "pair_gross_notional": pair_gross_total,
            "raw_plan_stock_gross_usd": raw_plan_stock_gross_day,
            "blended_stock_structural_gross_usd": (
                blended_stock_structural_gross_day
            ),
            "target_blend_alpha": (
                structural_blend_alpha if hedge_safe else np.nan
            ),
            "confirmed_desired_gross_usd": float(confirmed_desired_gross_ref),
            "turnover_reference_gross_usd": float(
                _turnover_budget_reference_gross(
                    pair_gross_total,
                    confirmed_desired_gross_ref,
                    hedge_safe=hedge_safe,
                )
            ),
            "deployed_desired_gross_ratio": (
                pair_gross_total / confirmed_desired_gross_ref
                if confirmed_desired_gross_ref > 1e-9
                else np.nan
            ),
            "underlying_internalized_usd": internalized,
            "gross_leverage": gross_notional / equity if equity > 0 else np.nan,
            "net_exposure_pct": net_notional / equity if equity > 0 else np.nan,
            "turnover_usd": turnover_day,
            "n_positions": int(len(cur)),
            "largest_pair_gross_share": largest_pair_share,
            "top5_gross_share": top5_gross_share,
            "gross_hhi": gross_hhi,
            "n_stale_etf": stale_etf,
            "n_stale_underlying": stale_underlying,
            "is_rebalance": int(day_did_rebal),
            "active_plan_date": str(active_plan_date.date()) if active_plan_date is not None else None,
            "drawdown": equity / running_peak - 1.0 if running_peak > 0 else np.nan,
        }
        sleeve_caps = _sleeve_capital_bases(cur)
        for s in ALL_SLEEVES:
            sc = sleeve_comp[s]
            row_out[s] = sc["price"] - sc["borrow"] - sc["margin"] - sc["txn"] + sc["credit"]
            row_out[f"{s}__price_pnl"] = sc["price"]
            row_out[f"{s}__borrow_cost"] = sc["borrow"]
            row_out[f"{s}__short_credit"] = sc["credit"]
            row_out[f"{s}__margin_cost"] = sc["margin"]
            row_out[f"{s}__txn_cost"] = sc["txn"]
            net_c, gross_c = sleeve_caps.get(s, (0.0, 0.0))
            row_out[f"{s}__net_cap"] = float(net_c)
            row_out[f"{s}__gross_cap"] = float(gross_c)
        sleeve_daily_rows.append(row_out)

    audit = pd.DataFrame(audit_rows)
    pair_rows = []
    for e, c in contrib.items():
        gross_end = (
            abs(float(cur.at[e, "etf_usd"])) + abs(float(cur.at[e, "underlying_usd"]))
            if e in cur.index else 0.0
        )
        if e in cur.index:
            end_etf = float(cur.at[e, "etf_usd"])
            end_und = float(cur.at[e, "underlying_usd"])
            c["end_etf_usd"] = end_etf
            c["end_underlying_usd"] = end_und
        else:
            end_etf = float(c.get("end_etf_usd", 0.0) or 0.0)
            end_und = float(c.get("end_underlying_usd", 0.0) or 0.0)
        # Prefer last target legs for exposure / hedge (stable vs drifted marks).
        tgt_etf = float(c.get("last_target_etf_usd", end_etf) or end_etf)
        tgt_und = float(c.get("last_target_underlying_usd", end_und) or end_und)
        # Convention: long_usd = underlying target; short_usd = ETF target.
        long_usd = tgt_und
        short_usd = tgt_etf
        hedge_ratio = abs(tgt_und) / abs(tgt_etf) if abs(tgt_etf) > 1e-9 else np.nan
        reb_dates = c.get("rebalance_dates") or []
        net_pair = (
            float(c["price_pnl_usd"])
            - float(c["borrow_cost_usd"])
            + float(c.get("short_credit_usd", 0.0))
            - float(c["margin_cost_usd"])
            - float(c["txn_cost_usd"])
        )
        pair_rows.append(
            {
                **{k: v for k, v in c.items() if k != "rebalance_dates"},
                "pnl_usd": net_pair,
                "end_weight": gross_end / equity if equity > 0 else np.nan,
                "long_usd": long_usd,
                "short_usd": short_usd,
                "end_etf_usd": end_etf,
                "end_underlying_usd": end_und,
                "hedge_ratio": hedge_ratio,
                "n_rebals": int(len(reb_dates)),
                "rebalance_dates": ";".join(reb_dates),
            }
        )
    pair_stats = pd.DataFrame(pair_rows)
    if not pair_stats.empty:
        pair_stats = pair_stats.sort_values("pnl_usd")
    sleeve_daily = pd.DataFrame(sleeve_daily_rows)
    if not sleeve_daily.empty:
        for s in ALL_SLEEVES:
            if s in sleeve_daily.columns:
                sleeve_daily[f"{s}_cum_pnl"] = sleeve_daily[s].cumsum()

    pair_daily = pd.DataFrame(pair_daily_rows)
    if not pair_daily.empty:
        pair_daily = pair_daily.sort_values(["sleeve", "ETF", "date"]).reset_index(drop=True)
        pair_daily["cum_pnl"] = pair_daily.groupby(["sleeve", "ETF"], sort=False)[
            "daily_pnl"
        ].cumsum()

    meta = {
        "n_rebal": n_rebal,
        "turnover_l1": turnover_l1,
        "turnover_usd": turnover_total,
        "same_run_churn_enabled": bool(same_run_churn_enabled),
        "avoided_round_trip_usd": 0.0,
        "risk_override_turnover_usd": float(hedge_repair_turnover),
        "one_terminal_target_per_symbol": True,
        "cash_days": cash_days,
        "first_plan": str(first_plan.date()),
        "n_plans_used": len(plans_used),
        "start_usd": capital_usd,
        "end_usd": float(nav.iloc[-1]) if len(nav) else np.nan,
        "execution_lag_sessions": int(execution_lag_sessions),
        "target_notional_mode": target_notional_mode,
        "scale_sleeves_to_budget": bool(scale_sleeves_to_budget),
        "commission_per_share": float(commission_per_share),
        "margin_rate_annual": float(margin_rate_annual),
        "financing_daycount": float(financing_daycount),
        "short_proceeds_credit_annual": float(short_proceeds_credit_annual),
        "retarget_on_plan_change": bool(retarget_on_plan_change),
        "use_resize_bands": bool(use_resize_bands),
        "enter_band_pct": float(enter_band_pct),
        "exit_band_pct": float(exit_band_pct),
        "min_trade_usd": float(min_trade_usd),
        "b4_execution": str(b4_execution),
        "purgatory_model_zero_policy": str(purg_zero_pol),
        "b4_membership_clock": str(membership_mode),
        "stock_rebalance_clock": str(stock_clock),
        "operator_check_days": int(op_check_days),
        "b4_apply_resize_bands": bool(apply_b4_bands),
        "b4_ratchet_execution_guard": bool(ratchet_guard),
        "b4_allow_inverse_cover": bool(allow_inv_cover),
        "b4_empty_plan_policy": str(empty_plan_pol),
        "net_shared_underlyings": bool(net_underlyings),
        "turnover_pace_enabled": bool(pace_on),
        "turnover_pace_mode": str(pace_mode),
        "turnover_pace_version": "1" if hedge_safe else ("legacy" if legacy_pace else "off"),
        "confirmation_count": int(confirm_n),
        "entry_ramp_sessions": int(entry_sessions),
        "reduction_ramp_sessions": int(reduction_sessions),
        "remaining_gap_rate": float(resize_gap_rate),
        "stock_midweek_mode": str(midweek_mode),
        "midweek_hedge_repair": bool(allow_midweek_hedge_repair),
        "hedge_reserve_frac": float(hedge_reserve),
        "adv_participation_pct": float(adv_pct),
        "sleeve_gross_ema_alpha": float(pace_alpha),
        "max_leg_step_pct": float(pace_leg_pct),
        "pair_gross_ramp_pct": float(pair_ramp_pct),
        "max_daily_turnover_pct": float(pace_turn_pct),
        "target_blend_alpha": float(structural_blend_alpha),
        "establish_budget_frac": float(pace_est_frac),
        "resize_age_boost_days": int(age_boost_days),
        "n_deferred_pace": int(n_deferred_pace),
        "n_hedge_repairs": int(n_hedge_repairs),
        "n_growth_blocked_hedge_infeasible": int(
            n_growth_blocked_hedge_infeasible
        ),
        "n_b4_cadence_rebals": int(n_b4_cadence_rebals),
        "n_b4_membership_deferred": int(n_b4_membership_deferred),
        "n_b4_empty_plan_holds": int(n_b4_empty_plan_holds),
        "n_b4_ratchet_pins": int(n_b4_ratchet_pins),
        "n_delist_flat": int(n_delist_flat),
        "n_b4_cadence_pairs": int(len(b4_cadence)),
        "n_purgatory_reductions": int(n_purgatory_reductions),
        "purgatory_blocked_add_usd": float(purgatory_blocked_add_usd),
        **perf(nav),
    }
    meta["pair_daily"] = pair_daily
    meta["pending_target_audit"] = pd.DataFrame(pending_target_rows)
    return nav, audit, meta, pair_stats, sleeve_daily


def _take_pair_daily(meta: dict[str, Any]) -> pd.DataFrame:
    """Remove pair_daily from sim meta so it is not serialized into report.json."""
    raw = meta.pop("pair_daily", None)
    return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()


def _take_pending_target_audit(meta: dict[str, Any]) -> pd.DataFrame:
    """Remove the detailed controller ledger before JSON serialization."""
    raw = meta.pop("pending_target_audit", None)
    return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()


def build_b4_membership_manifest(
    timeline: Mapping[pd.Timestamp, pd.DataFrame],
    pair_daily: pd.DataFrame,
    panel: Mapping[str, pd.DataFrame],
    *,
    run_end: str | None,
) -> pd.DataFrame:
    """Publish every B4 lifecycle, including members without accounting rows."""
    members: dict[str, dict[str, Any]] = {}
    for plan_date, plan in sorted(timeline.items()):
        if plan is None or plan.empty:
            continue
        b4 = plan.loc[plan.get("sleeve", pd.Series("", index=plan.index)).astype(str).eq(B4_SLEEVE)]
        for _, row in b4.iterrows():
            etf = _norm(row.get("ETF", ""))
            if not etf:
                continue
            item = members.setdefault(etf, {
                "ETF": etf,
                "Underlying": _norm(row.get("Underlying", "")),
                "first_plan_date": str(pd.Timestamp(plan_date).date()),
                "last_plan_date": str(pd.Timestamp(plan_date).date()),
                "plan_observations": 0,
            })
            item["last_plan_date"] = str(pd.Timestamp(plan_date).date())
            item["plan_observations"] += 1
            item["latest_purgatory"] = _truthy(row.get("purgatory", False))
            item["latest_reduce_only"] = _truthy(row.get("reduce_only", False))
            item["latest_keep_open"] = _truthy(row.get("keep_open", False))
            item["latest_hard_exit"] = _truthy(row.get("hard_exit", False))
            item["latest_execution_policy"] = str(row.get("execution_policy", "") or "")
            item["latest_model_gross_usd"] = _float_or(row.get("model_gross_target_usd"))
            item["latest_executable_gross_usd"] = _float_or(row.get("gross_target_usd"))

    actual = pair_daily.loc[pair_daily.get("sleeve", pd.Series("", index=pair_daily.index)).astype(str).eq(B4_SLEEVE)].copy()
    if not actual.empty:
        actual["ETF"] = actual["ETF"].map(_norm)
        actual["date"] = actual["date"].astype(str)
    for etf, item in members.items():
        ledger = actual.loc[actual["ETF"].eq(etf)] if not actual.empty else pd.DataFrame()
        item["has_ledger"] = not ledger.empty
        item["first_ledger_date"] = str(ledger["date"].min()) if not ledger.empty else ""
        item["last_ledger_date"] = str(ledger["date"].max()) if not ledger.empty else ""
        item["in_price_panel"] = etf in panel
        if not ledger.empty:
            last = ledger.sort_values("date").iloc[-1]
            gross = abs(_float_or(last.get("etf_usd"))) + abs(_float_or(last.get("underlying_usd")))
            item["latest_ledger_gross_usd"] = gross
            item["lifecycle_state"] = "open" if gross > 1e-6 else "closed"
            item["block_reason"] = ""
        elif not item["in_price_panel"]:
            item["lifecycle_state"] = "blocked"
            item["block_reason"] = "missing_price_panel"
        elif item.get("latest_purgatory"):
            item["lifecycle_state"] = "purgatory_not_incumbent"
            item["block_reason"] = "purgatory_without_prior_fill"
        elif item.get("latest_hard_exit"):
            item["lifecycle_state"] = "hard_exit_without_ledger"
            item["block_reason"] = "hard_exit"
        else:
            item["lifecycle_state"] = "pending_entry"
            item["block_reason"] = "awaiting_operator_or_execution"
        item["run_end"] = str(run_end or "")
    return pd.DataFrame(list(members.values())).sort_values("ETF").reset_index(drop=True) if members else pd.DataFrame()


# ---------------------------------------------------------------------------
# GTP-approx daily sizing (legacy) + full prod replay (primary)
# ---------------------------------------------------------------------------
def prod_replay_plan_timeline(
    *,
    cfg: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    state_root: Path | str | None = None,
    keep_state: bool = False,
    fallback_to_archived_plans: bool = True,
    plans_dir: Path | str | None = None,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Daily plans from full production GTP sizing on screened archives.

    Carries isolated crash / EMA / ratchet / decay state forward under
    ``state_root``. Ratchet floors use the prior day's sized plan (simulated
    holdings), not live Flex. Does not apply the avg-borrow overlay.

    When sizing fails (or returns an empty book) and
    ``fallback_to_archived_plans`` is True, insert that date's archived
    ``proposed_trades.csv`` if present so pre-schema-gap windows can still
    trade from live plans (e.g. 2026-03-24, 2026-04-01).

    If ``plans_dir`` is set, each accepted plan is also written to
    ``{plans_dir}/{YYYY-MM-DD}.csv`` so simulation can be re-run without
    re-sizing.
    """
    from scripts.gtp_prod_sizing import held_from_plan, size_book_from_screened

    screened_dates = [d for d in list_archived_screened_dates() if d >= start]
    if end is not None:
        screened_dates = [d for d in screened_dates if d <= end]

    # Also consider archived proposed_trades on dates without a screened CSV
    # (common in mid-April) when fallback is enabled.
    plan_only_dates: list[pd.Timestamp] = []
    if fallback_to_archived_plans:
        screened_set = set(screened_dates)
        for d in list_archived_plan_dates():
            if d < start:
                continue
            if end is not None and d > end:
                continue
            if d not in screened_set:
                plan_only_dates.append(d)

    own_tmp = state_root is None
    root = Path(state_root) if state_root is not None else Path(tempfile.mkdtemp(prefix="prod_replay_state_"))
    if not root.is_absolute():
        root = (REPO / root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    dump_dir: Path | None = None
    if plans_dir is not None:
        dump_dir = Path(plans_dir)
        if not dump_dir.is_absolute():
            dump_dir = (REPO / dump_dir).resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)

    timeline: dict[pd.Timestamp, pd.DataFrame] = {}
    diag_rows: list[dict] = []
    held: dict[tuple[str, str], dict[str, float]] = {}

    def _store_plan(d: pd.Timestamp, plan: pd.DataFrame) -> None:
        timeline[d] = plan
        if dump_dir is not None:
            plan.to_csv(dump_dir / f"{d.strftime('%Y-%m-%d')}.csv", index=False)

    def _append_archived(d: pd.Timestamp, *, reason: str) -> bool:
        nonlocal held
        arch = load_plan_file(RUNS_DIR / d.strftime("%Y-%m-%d") / "proposed_trades.csv")
        sized_arch = (
            arch.loc[arch["gross_target_usd"].abs() > 1e-6].copy()
            if not arch.empty
            else pd.DataFrame()
        )
        if sized_arch.empty:
            return False
        _store_plan(d, arch)
        held = held_from_plan(sized_arch)
        b4 = sized_arch[
            sized_arch["sleeve"].astype(str).str.lower().isin(
                {"inverse_decay_bucket4", "bucket4", "bucket_4"}
            )
        ]
        diag_rows.append(
            {
                "date": d,
                "source": "archived_proposed_fallback",
                "n_pairs": len(sized_arch),
                "n_keep_open": int(arch["keep_open"].sum()) if "keep_open" in arch.columns else 0,
                "n_reduce_only": int(arch["reduce_only"].sum()) if "reduce_only" in arch.columns else 0,
                "gross_sum": float(sized_arch["gross_target_usd"].sum()),
                "gross_b4": float(b4["gross_target_usd"].sum()) if len(b4) else 0.0,
                "n_b4": int(len(b4)),
                "n_held_in": int(len(held)),
                "n_screened": 0,
                "edge_source": "archived_proposed_trades",
                "n_edge_fallback": 0,
                "ok": True,
                "error": reason,
            }
        )
        return True

    try:
        for d in screened_dates:
            path = RUNS_DIR / d.strftime("%Y-%m-%d") / "etf_screened_today.csv"
            try:
                screened = pd.read_csv(path)
                screened, edge_diag = prepare_screened_for_prod_replay(screened)
                sized, sdiag = size_book_from_screened(
                    screened,
                    d.strftime("%Y-%m-%d"),
                    cfg,
                    state_root=root,
                    held_inverse_short_by_pair=held,
                    quiet=True,
                )
                plan = normalize_plan(sized, source_date=d.strftime("%Y-%m-%d"))
                sized_plan = plan.loc[plan["gross_target_usd"].abs() > 1e-6].copy()
                if sized_plan.empty:
                    raise ValueError("prod GTP returned empty plan")
                _store_plan(d, plan)  # includes reduce-only / rollback-hold purgatory rows
                held = held_from_plan(sized_plan)
                b4 = sized_plan[
                    sized_plan["sleeve"].astype(str).str.lower().isin(
                        {"inverse_decay_bucket4", "bucket4", "bucket_4"}
                    )
                ]
                diag_rows.append(
                    {
                        "date": d,
                        "source": "prod_replay",
                        "n_pairs": len(sized_plan),
                        "n_keep_open": int(plan["keep_open"].sum()) if "keep_open" in plan.columns else 0,
                        "n_reduce_only": int(plan["reduce_only"].sum()) if "reduce_only" in plan.columns else 0,
                        "gross_sum": float(sized_plan["gross_target_usd"].sum()),
                        "gross_b4": float(b4["gross_target_usd"].sum()) if len(b4) else 0.0,
                        "n_b4": int(len(b4)),
                        "n_held_in": int(sdiag.get("n_held_pairs", 0) or 0),
                        "n_screened": int(len(screened)),
                        "edge_source": edge_diag.get("edge_source", ""),
                        "n_edge_fallback": int(edge_diag.get("n_edge_fallback", 0) or 0),
                        "ok": True,
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                if fallback_to_archived_plans and _append_archived(
                    d, reason=f"fallback after {type(exc).__name__}: {exc}"
                ):
                    continue
                diag_rows.append(
                    {
                        "date": d,
                        "source": "prod_replay_failed",
                        "n_pairs": 0,
                        "gross_sum": 0.0,
                        "gross_b4": 0.0,
                        "n_b4": 0,
                        "n_held_in": int(len(held)),
                        "n_screened": 0,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        for d in plan_only_dates:
            _append_archived(d, reason="plan-only archive (no screened CSV)")
    finally:
        if own_tmp and not keep_state:
            shutil.rmtree(root, ignore_errors=True)

    return timeline, pd.DataFrame(diag_rows).sort_values("date") if diag_rows else pd.DataFrame()


def load_cached_plan_timeline(plans_dir: Path | str) -> dict[pd.Timestamp, pd.DataFrame]:
    """Load plans previously dumped by ``prod_replay_plan_timeline(..., plans_dir=...)``."""
    root = Path(plans_dir)
    if not root.is_absolute():
        root = (REPO / root).resolve()
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.csv")):
        try:
            d = pd.Timestamp(path.stem)
        except Exception:
            continue
        plan = normalize_plan(pd.read_csv(path), source_date=path.stem)
        if not plan.empty:
            out[d] = plan
    return out


def prepare_screened_for_prod_replay(screened: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backtest-only shim: fill missing ``net_edge_p50_annual`` for older archives.

    Live GTP treats all-NaN ``net_edge_p50_annual`` as ineligible, which zeros the
    book on pre-2026-04-25 screened CSVs. Map from historical proxies without
    changing live ``generate_trade_plan`` economics.
    """
    out = screened.copy()
    if "Delta" not in out.columns and "Beta" in out.columns:
        out["Delta"] = out["Beta"]

    diag: dict[str, Any] = {
        "n_rows": int(len(out)),
        "edge_source": "net_edge_p50_annual",
        "n_edge_fallback": 0,
    }
    edge = (
        pd.to_numeric(out["net_edge_p50_annual"], errors="coerce")
        if "net_edge_p50_annual" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    if int(np.isfinite(edge.to_numpy(dtype=float)).sum()) > 0:
        out["net_edge_p50_annual"] = edge
        diag["edge_source"] = "net_edge_p50_annual"
        return out, diag

    fallback_cols = (
        "net_decay_annual",
        "blended_gross_decay",
        "gross_decay_annual",
    )
    filled = pd.Series(np.nan, index=out.index, dtype=float)
    used = "none"
    for col in fallback_cols:
        if col not in out.columns:
            continue
        cand = pd.to_numeric(out[col], errors="coerce")
        if int(np.isfinite(cand.to_numpy(dtype=float)).sum()) == 0:
            continue
        # Prefer first proxy that has any finite values; fill remaining holes
        # from later proxies.
        if used == "none":
            filled = cand.copy()
            used = col
        else:
            filled = filled.where(np.isfinite(filled), cand)
    out["net_edge_p50_annual"] = filled
    diag["edge_source"] = used
    diag["n_edge_fallback"] = int(np.isfinite(filled.to_numpy(dtype=float)).sum())
    out["edge_used_for_prod_replay"] = used
    return out, diag


def prepare_screened_for_gtp_approx(screened: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time GTP inputs: average borrow + current net-edge scores.

    Overwrites ``borrow_current`` with ``borrow_avg_annual`` where the average is
    finite so ``mirror_generate_trade_plan_sizing`` / ``_decay_score_weights``
    use the same borrow figure for eligibility caps and the
    ``net_edge − aversion × borrow`` score. Leaves ``net_edge_p50_annual`` (and
    siblings) untouched — those are already the as-of screener edge scores.
    """
    out = screened.copy()
    if "borrow_avg_annual" in out.columns:
        avg = pd.to_numeric(out["borrow_avg_annual"], errors="coerce")
        if "borrow_current" not in out.columns:
            out["borrow_current"] = np.nan
        spot = pd.to_numeric(out["borrow_current"], errors="coerce")
        out["borrow_current"] = avg.where(np.isfinite(avg), spot)
        out["borrow_used_for_sizing"] = np.where(
            np.isfinite(avg),
            "borrow_avg_annual",
            "borrow_current",
        )
    elif "borrow_used_for_sizing" not in out.columns:
        out["borrow_used_for_sizing"] = "borrow_current"
    return out


def gtp_approx_plan_timeline(
    *,
    cfg: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Daily plans from screened archives only (no proposed_trades).

    Sizes each ``etf_screened_today.csv`` with average-borrow + net-edge GTP
    mirror. B1 hysteresis is carried in an isolated temp state file.
    """
    from scripts.gtp_sizing_mirror import mirror_generate_trade_plan_sizing

    screened_dates = [d for d in list_archived_screened_dates() if d >= start]
    if end is not None:
        screened_dates = [d for d in screened_dates if d <= end]

    paths = cfg.get("paths") or {}
    real_decay = Path(
        str(paths.get("core_leveraged_decay_state_json", REPO / "data/core_leveraged_decay_state.json"))
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="gtp_approx_state_"))
    tmp_decay = tmp_dir / "core_leveraged_decay_state.json"
    if real_decay.exists():
        shutil.copy2(real_decay, tmp_decay)
    else:
        tmp_decay.write_text("{}", encoding="utf-8")
    cfg_iso = dict(cfg)
    cfg_iso_paths = dict(paths)
    cfg_iso_paths["core_leveraged_decay_state_json"] = str(tmp_decay)
    cfg_iso["paths"] = cfg_iso_paths

    timeline: dict[pd.Timestamp, pd.DataFrame] = {}
    diag_rows: list[dict] = []

    try:
        for d in screened_dates:
            path = RUNS_DIR / d.strftime("%Y-%m-%d") / "etf_screened_today.csv"
            try:
                screened = prepare_screened_for_gtp_approx(pd.read_csv(path))
                n_avg = (
                    int((screened.get("borrow_used_for_sizing") == "borrow_avg_annual").sum())
                    if "borrow_used_for_sizing" in screened.columns
                    else 0
                )
                sized, _sdiag = mirror_generate_trade_plan_sizing(
                    screened,
                    cfg_iso,
                    run_date=d.strftime("%Y-%m-%d"),
                    paths=cfg_iso_paths,
                    hysteresis_touch_disk=True,
                )
                plan = normalize_plan(sized, source_date=d.strftime("%Y-%m-%d"))
                plan = plan.loc[plan["gross_target_usd"].abs() > 1e-6].copy()
                if plan.empty:
                    raise ValueError("mirror returned empty plan")
                timeline[d] = plan
                diag_rows.append(
                    {
                        "date": d,
                        "source": "gtp_approx",
                        "n_pairs": len(plan),
                        "gross_sum": float(plan["gross_target_usd"].sum()),
                        "n_borrow_avg": n_avg,
                        "n_screened": int(len(screened)),
                        "ok": True,
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                diag_rows.append(
                    {
                        "date": d,
                        "source": "gtp_approx_failed",
                        "n_pairs": 0,
                        "gross_sum": 0.0,
                        "n_borrow_avg": 0,
                        "n_screened": 0,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return timeline, pd.DataFrame(diag_rows).sort_values("date") if diag_rows else pd.DataFrame()


def recompute_plan_timeline(
    *,
    cfg: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    fallback_to_archived: bool = True,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Legacy: mirror on screened dates, optionally fall back to archived proposed."""
    from scripts.gtp_sizing_mirror import mirror_generate_trade_plan_sizing

    screened_dates = [d for d in list_archived_screened_dates() if d >= start]
    if end is not None:
        screened_dates = [d for d in screened_dates if d <= end]

    # Persist hysteresis across the loop in an isolated temp state file.
    paths = cfg.get("paths") or {}
    real_decay = Path(str(paths.get("core_leveraged_decay_state_json", REPO / "data/core_leveraged_decay_state.json")))
    tmp_dir = Path(tempfile.mkdtemp(prefix="gtp_replay_state_"))
    tmp_decay = tmp_dir / "core_leveraged_decay_state.json"
    if real_decay.exists():
        shutil.copy2(real_decay, tmp_decay)
    cfg_iso = dict(cfg)
    cfg_iso_paths = dict(paths)
    cfg_iso_paths["core_leveraged_decay_state_json"] = str(tmp_decay)
    cfg_iso["paths"] = cfg_iso_paths

    timeline: dict[pd.Timestamp, pd.DataFrame] = {}
    diag_rows: list[dict] = []

    try:
        for d in screened_dates:
            path = RUNS_DIR / d.strftime("%Y-%m-%d") / "etf_screened_today.csv"
            try:
                screened = prepare_screened_for_gtp_approx(pd.read_csv(path))
                sized, sdiag = mirror_generate_trade_plan_sizing(
                    screened,
                    cfg_iso,
                    run_date=d.strftime("%Y-%m-%d"),
                    paths=cfg_iso_paths,
                    hysteresis_touch_disk=True,  # write to isolated tmp only
                )
                plan = normalize_plan(sized, source_date=d.strftime("%Y-%m-%d"))
                if plan.empty:
                    raise ValueError("mirror returned empty plan")
                timeline[d] = plan
                diag_rows.append(
                    {
                        "date": d,
                        "source": "recompute_mirror",
                        "n_pairs": len(plan),
                        "gross_sum": float(plan["gross_target_usd"].sum()),
                        "ok": True,
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                diag_rows.append(
                    {
                        "date": d,
                        "source": "recompute_failed",
                        "n_pairs": 0,
                        "gross_sum": 0.0,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if fallback_to_archived:
                    arch = proposed_trades_full_asof(d)
                    if arch is not None and not arch.empty:
                        timeline[d] = arch
                        diag_rows[-1]["source"] = "fallback_archived"
                        diag_rows[-1]["n_pairs"] = len(arch)
                        diag_rows[-1]["gross_sum"] = float(arch["gross_target_usd"].sum())
                        diag_rows[-1]["ok"] = True

        if fallback_to_archived:
            # Fill gaps: archived plan dates without a recompute entry
            for d, plan in load_plan_timeline(start=start, end=end).items():
                if d not in timeline:
                    timeline[d] = plan
                    diag_rows.append(
                        {
                            "date": d,
                            "source": "archived_only",
                            "n_pairs": len(plan),
                            "gross_sum": float(plan["gross_target_usd"].sum()),
                            "ok": True,
                            "error": "",
                        }
                    )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return timeline, pd.DataFrame(diag_rows).sort_values("date") if diag_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _write_outputs(
    *,
    outdir: Path,
    mode: str,
    report: dict,
    summary: pd.DataFrame,
    series: pd.DataFrame,
    extra_csvs: dict[str, pd.DataFrame] | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if "skip_reasons" in summary.columns:
        summary = summary.copy()
        summary["skip_reasons"] = summary["skip_reasons"].astype(str)
    summary.to_csv(outdir / "sleeve_summary.csv", index=False)
    series.to_csv(outdir / "daily_nav.csv")
    (outdir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if extra_csvs:
        for name, df in extra_csvs.items():
            df.to_csv(outdir / name, index=False)

    book = report.get("book") or {}
    md_lines = [
        f"# Production actual backtest ({mode}) — {report.get('start')} → {report.get('end')}",
        "",
        f"Mode: **{mode}**",
        f"Capital: ${report.get('capital_usd', 0):,.0f} × {report.get('gross_leverage')}x",
        "",
        "## Sleeve / book summary",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Book",
        f"- CAGR: {book.get('cagr')}",
        f"- Vol: {book.get('vol')}",
        f"- Sharpe: {book.get('sharpe')}",
        f"- MaxDD: {book.get('maxdd')}",
        f"- End NAV: ${book.get('end_usd')}",
        "",
        "## Limitations",
        *[f"- {x}" for x in report.get("limitations", [])],
        "",
    ]
    (outdir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")


def run_frozen_backtest(
    *,
    run_date: str,
    start: str,
    outdir: Path,
) -> dict:
    cfg = load_config(REPO / "config" / "strategy_config.yml")
    capital = float(cfg["strategy"]["capital_usd"])
    lev = float(cfg["strategy"]["gross_leverage"])
    budgets = sleeve_budgets_usd(cfg)
    rknobs = rebalance_knobs(cfg)
    start_ts = pd.Timestamp(start)

    print(f"[prod-bt:frozen] capital=${capital:,.0f} lev={lev:g}x book=${capital * lev:,.0f}")
    print(f"[prod-bt:frozen] budgets: " + ", ".join(f"{k}=${v:,.0f}" for k, v in budgets.items()))
    print(f"[prod-bt:frozen] universe from data/runs/{run_date}/proposed_trades.csv  start={start}")

    uni = load_universe(run_date)
    # Normalize sleeve aliases if present
    if "sleeve" in uni.columns:
        uni = uni.copy()
        uni["sleeve"] = uni["sleeve"].astype(str).str.strip().str.lower().map(lambda s: SLEEVE_ALIASES.get(s, s))
    panel = load_price_panel(run_date)
    print(f"[prod-bt:frozen] plan rows with gross>0: {len(uni)} | price panels: {len(panel)}")

    sleeve_navs: dict[str, pd.Series] = {}
    metas: list[dict] = []
    pair_stats_parts: list[pd.DataFrame] = []

    for sleeve in STOCK_SLEEVES:
        print(f"[prod-bt:frozen] simulating {sleeve} ...")
        nav, meta, pstats = _stock_sleeve_nav(
            uni,
            panel,
            sleeve=sleeve,
            start=start_ts,
            budget_usd=budgets[sleeve],
            enter_band_pct=rknobs["enter_band_pct"],
            slippage_bps=rknobs["slippage_bps"],
        )
        if len(nav):
            sleeve_navs[sleeve] = nav
        metas.append(meta)
        if pstats is not None and not pstats.empty:
            pair_stats_parts.append(pstats)
        print(f"  -> n={meta.get('n_pairs')} CAGR={meta.get('cagr')} MaxDD={meta.get('maxdd')}")

    for sleeve in B4_SLEEVES:
        print(f"[prod-bt:frozen] simulating {sleeve} (production cadence/v7) ...")
        nav, meta, pstats = _b4_family_nav(
            uni,
            panel,
            sleeve=sleeve,
            start=start_ts,
            budget_usd=budgets[sleeve],
            slippage_bps=rknobs["slippage_bps"],
        )
        if len(nav):
            sleeve_navs[sleeve] = nav
        metas.append(meta)
        if pstats is not None and not pstats.empty:
            pair_stats_parts.append(pstats)
        print(f"  -> n={meta.get('n_pairs')} skipped={meta.get('n_skipped')} CAGR={meta.get('cagr')} MaxDD={meta.get('maxdd')}")
        if meta.get("skip_reasons"):
            print(f"     skips: {meta.get('skip_reasons')}")

    print("[prod-bt:frozen] simulating volatility_etp_bucket5 (bucket5 carry) ...")
    nav, meta, pstats = _b5_sleeve_nav(
        uni,
        start=start_ts,
        budget_usd=budgets[B5_SLEEVE],
        slippage_bps=rknobs["slippage_bps"],
    )
    if len(nav):
        sleeve_navs[B5_SLEEVE] = nav
    metas.append(meta)
    if pstats is not None and not pstats.empty:
        pair_stats_parts.append(pstats)
    print(f"  -> n={meta.get('n_pairs')} CAGR={meta.get('cagr')} MaxDD={meta.get('maxdd')} rho={meta.get('rho')}")

    book = combine_book_nav(sleeve_navs, capital_usd=capital, budgets=budgets)
    book_meta = {"sleeve": "BOOK", "n_pairs": sum(int(m.get("n_pairs") or 0) for m in metas), **perf(book)}
    book_meta["start_usd"] = capital
    book_meta["end_usd"] = float(book.iloc[-1]) if len(book) else np.nan
    book_meta["engine"] = "sum sleeve budgets, rescale to capital_usd"
    metas.append(book_meta)

    summary = pd.DataFrame(metas)
    series = pd.DataFrame({k: v for k, v in sleeve_navs.items()})
    if len(book):
        series["BOOK_NAV"] = book.reindex(series.index).ffill()

    pair_stats = (
        pd.concat(pair_stats_parts, ignore_index=True).sort_values("pnl_usd")
        if pair_stats_parts
        else pd.DataFrame()
    )
    # Sleeve rollup from pair stats (exact sum of pair PnL within each sleeve)
    sleeve_pnl = pd.DataFrame()
    if not pair_stats.empty:
        sleeve_pnl = (
            pair_stats.groupby("sleeve", as_index=False)
            .agg(n_pairs=("ETF", "count"), start_usd=("start_usd", "sum"),
                 end_usd=("end_usd", "sum"), pnl_usd=("pnl_usd", "sum"))
            .sort_values("pnl_usd")
        )

    flow = ((cfg.get("portfolio") or {}).get("sleeves") or {}).get("flow_program") or {}
    crash_budget_path = RUNS_DIR / run_date / "b4_crash_budget.csv"
    report = {
        "mode": "frozen",
        "run_date": run_date,
        "start": start,
        "end": str(book.index[-1].date()) if len(book) else None,
        "capital_usd": capital,
        "gross_leverage": lev,
        "budgets_usd": budgets,
        "rebalance_knobs": rknobs,
        "archive_coverage": archive_coverage_summary(start).to_dict(orient="records"),
        "book": {k: book_meta[k] for k in ("cagr", "vol", "sharpe", "maxdd", "start_usd", "end_usd") if k in book_meta},
        "crash_budget_csv": str(crash_budget_path) if crash_budget_path.is_file() else None,
        "flow_program_note": {
            "included_in_nav": False,
            "fixed_usd_per_week": flow.get("fixed_usd_per_week"),
            "reason": "B3 is a parallel weekly deployer; not in proposed_trades gross path",
        },
        "limitations": [
            "Universe/weights frozen to run_date proposed_trades (full B4 stack when GTP wrote them).",
            "B1/B2: plan leg fractions + weekly retarget; inactive names held as cash (no pair_eq wipe on NaN Fridays).",
            "Prices split-adjusted via data/splits_from_flex.csv before returns.",
            "B4: production cadence + v7 dynamic hedge; per-pair sim_start; skipped gross stays cash.",
            "B4 sizing: opt2 → crash-cap + scale_to_budget → post-cap dilution-aware smooth → legs/ratchet.",
            "B5: bucket5_carry_bt short-UVIX/short-SVIX at plan rho (not B4 dynamic-h).",
            "B1/B2 ETF shorts pay borrow_current; explicit underlying borrow is charged when that leg is short.",
            "B3 flow ($1,300/wk) excluded from NAV.",
            "Frozen is counterfactual (today's book from --start); prefer --mode replay for PIT history.",
        ],
    }
    extra = {}
    if not pair_stats.empty:
        extra["pair_stats.csv"] = pair_stats
    if not sleeve_pnl.empty:
        extra["sleeve_pnl.csv"] = sleeve_pnl
    # Skip audit from sleeve metas
    skip_rows = []
    for m in metas:
        reasons = str(m.get("skip_reasons") or "")
        if not reasons:
            continue
        for part in reasons.split(";"):
            part = part.strip()
            if not part:
                continue
            etf, _, reason = part.partition(":")
            skip_rows.append({"sleeve": m.get("sleeve"), "ETF": etf, "reason": reason or part})
    if skip_rows:
        extra["pair_skip_audit.csv"] = pd.DataFrame(skip_rows)
    _write_outputs(
        outdir=outdir, mode="frozen", report=report, summary=summary, series=series,
        extra_csvs=extra or None,
    )
    print(f"[prod-bt:frozen] wrote {outdir}")
    print(
        f"[prod-bt:frozen] BOOK CAGR={book_meta.get('cagr')} MaxDD={book_meta.get('maxdd')} "
        f"end=${book_meta.get('end_usd'):,.0f}"
    )
    return report


def run_replay_backtest(
    *,
    run_date: str,
    start: str,
    outdir: Path,
    pre_archive_policy: str = "cash",
) -> dict:
    """Phase A: day-by-day weights from archived proposed_trades."""
    cfg = load_config(REPO / "config" / "strategy_config.yml")
    capital = float(cfg["strategy"]["capital_usd"])
    lev = float(cfg["strategy"]["gross_leverage"])
    budgets = sleeve_budgets_usd(cfg)
    rknobs = rebalance_knobs(cfg)
    start_ts = pd.Timestamp(start)

    cov = archive_coverage_summary(start)
    print("[prod-bt:replay] archive coverage:")
    print(cov.to_string(index=False))

    timeline = load_plan_timeline(start=pd.Timestamp("2000-01-01"))  # keep plans before start for asof
    # Keep only plans that can affect the window (on/before end); include pre-start for asof
    if not timeline:
        raise FileNotFoundError("No archived proposed_trades.csv under data/runs/")

    panel = load_price_panel(run_date)
    print(f"[prod-bt:replay] {len(timeline)} archived plans | price panels={len(panel)} | start={start}")

    # Restrict active timeline keys used for updates to those >= start (asof still sees earlier)
    active = {d: p for d, p in timeline.items() if d >= start_ts}
    # Ensure we can asof from before start: keep last plan before start if any
    pre = [d for d in timeline if d < start_ts]
    if pre:
        last_pre = max(pre)
        active = {last_pre: timeline[last_pre], **active}

    nav, audit, meta, pair_stats, sleeve_daily = simulate_book_from_plan_timeline(
        active,
        panel,
        budgets=budgets,
        capital_usd=capital,
        start=start_ts,
        slippage_bps=rknobs["slippage_bps"],
        commission_per_share=rknobs["commission_per_share"],
        margin_rate_annual=rknobs["margin_rate_annual"],
        financing_daycount=rknobs["financing_daycount"],
        short_proceeds_credit_annual=rknobs["short_proceeds_credit_annual"],
        execution_lag_sessions=rknobs["execution_lag_sessions"],
        target_notional_mode=rknobs["target_notional_mode"],
        scale_sleeves_to_budget=bool(rknobs.get("scale_sleeves_to_budget", True)),
        enter_band_pct=rknobs["enter_band_pct"],
        exit_band_pct=rknobs["exit_band_pct"],
        min_trade_usd=rknobs["min_trade_usd"],
        check_freq="W-FRI",
        retarget_on_plan_change=True,  # archived plans already sparse; apply on arrival
        pre_archive_policy=pre_archive_policy,
        b4_execution=str(rknobs.get('b4_execution', 'cadence')),
        apply_delist_flatten=bool(rknobs.get('apply_delist_flatten', True)),
        use_borrow_history=bool(rknobs.get('use_borrow_history', True)),
        same_run_churn_enabled=bool(rknobs.get('same_run_churn_enabled', True)),
        purgatory_model_zero_policy=str(rknobs.get('purgatory_model_zero_policy', 'hold')),
        b4_membership_clock=str(rknobs.get('b4_membership_clock', 'operator_5d')),
        stock_rebalance_clock=str(rknobs.get('stock_rebalance_clock', 'operator_5d')),
        operator_check_days=int(rknobs.get('operator_check_days', 5) or 5),
        b4_apply_resize_bands=bool(rknobs.get('b4_apply_resize_bands', True)),
        b4_ratchet_execution_guard=bool(rknobs.get('b4_ratchet_execution_guard', True)),
        b4_allow_inverse_cover=bool(rknobs.get('b4_allow_inverse_cover', True)),
        b4_empty_plan_policy=str(rknobs.get('b4_empty_plan_policy', 'hold')),
        net_shared_underlyings=bool(rknobs.get('net_shared_underlyings', True)),
        turnover_pace_enabled=bool(rknobs.get('turnover_pace_enabled', True)),
        turnover_pace_mode=str(rknobs.get('turnover_pace_mode', 'hedge_safe_v1')),
        confirmation_count=int(rknobs.get('confirmation_count', 2)),
        entry_ramp_sessions=int(rknobs.get('entry_ramp_sessions', 5)),
        reduction_ramp_sessions=int(rknobs.get('reduction_ramp_sessions', 3)),
        remaining_gap_rate=float(rknobs.get('remaining_gap_rate', 0.25)),
        target_blend_alpha=float(rknobs.get('target_blend_alpha', 0.25)),
        stock_midweek_mode=str(rknobs.get('stock_midweek_mode', 'rebal_only')),
        midweek_hedge_repair=bool(rknobs.get('midweek_hedge_repair', False)),
        hedge_reserve_frac=float(rknobs.get('hedge_reserve_frac', 0.20)),
        adv_participation_pct=float(rknobs.get('adv_participation_pct', 0.10)),
        sleeve_gross_ema_alpha=float(rknobs.get('sleeve_gross_ema_alpha', 0.35)),
        max_leg_step_pct=float(rknobs.get('max_leg_step_pct', 0.25)),
        pair_gross_ramp_pct=float(rknobs.get('pair_gross_ramp_pct', 0.25)),
        max_daily_turnover_pct=float(rknobs.get('max_daily_turnover_pct', 0.15)),
        legacy_max_daily_turnover_pct=float(rknobs.get('legacy_max_daily_turnover_pct', 0.15)),
        establish_budget_frac=float(rknobs.get('establish_budget_frac', 0.50)),
        resize_age_boost_days=int(rknobs.get('resize_age_boost_days', 5)),
        hedge_long_trigger_net_pct=float(rknobs.get('hedge_long_trigger_net_pct', 0.04)),
        hedge_long_target_net_pct=float(rknobs.get('hedge_long_target_net_pct', 0.01)),
        hedge_short_trigger_net_pct=float(rknobs.get('hedge_short_trigger_net_pct', 0.01)),
        hedge_short_target_net_pct=float(rknobs.get('hedge_short_target_net_pct', 0.00)),
    )

    pair_daily = _take_pair_daily(meta)
    pending_target_audit = _take_pending_target_audit(meta)
    book_meta = {"sleeve": "BOOK", "mode": "replay", **meta}
    summary = pd.DataFrame([book_meta])
    series = pd.DataFrame({"BOOK_NAV": nav})
    if not sleeve_daily.empty:
        for s in ALL_SLEEVES:
            col = f"{s}_cum_pnl"
            if col in sleeve_daily.columns:
                series[s] = (
                    capital + sleeve_daily.set_index("date")[col]
                ).reindex(series.index).ffill()

    report = {
        "mode": "replay",
        "run_date": run_date,
        "start": start,
        "end": str(nav.index[-1].date()) if len(nav) else None,
        "capital_usd": capital,
        "gross_leverage": lev,
        "budgets_usd": budgets,
        "rebalance_knobs": rknobs,
        "archive_coverage": cov.to_dict(orient="records"),
        "book": {k: book_meta.get(k) for k in BOOK_REPORT_KEYS},
        "limitations": [
            "Phase A: replays archived proposed_trades (exact GTP output when archived).",
            f"Archives begin {cov.loc[0, 'first']}; before that policy={pre_archive_policy}.",
            "Schema normalized across eras (gross from |long|+|short|; whitelist_stock→yieldboost).",
            "Point-in-time plan legs are held as signed close notionals between weekly rebalances; no latest-plan look-ahead.",
            "Plans are known after their run-date close, execute at the next available close, and earn P&L from the following session.",
            "Gross targets preserve the archived plan gross/equity multiple as NAV changes; missing panels stay undeployed.",
            "Existing legs use production Phase-2b enter/exit hysteresis and the configured minimum trade; new pairs establish and exits close.",
            "Costs include 20 bp slippage per traded dollar, $0.0035/share commissions, archived borrow, and 4.45% margin debit / Actual-360.",
            "The 4.45% financing input is the Diamond Creek fallback (4.00% benchmark + 45 bp), not a point-in-time OBFR curve.",
            "Replay mirrors Phase-2b resize math but not broker execution sequencing or the B4 intra-pair cadence engine.",
            "B4 full stack (opt2 → crash+scale → post-cap smooth → legs) appears only on dates whose archived plan was generated after those features shipped.",
            "B3 flow excluded. Mirror recompute still skips live B4 opt2/crash/smooth/ratchet.",
        ],
    }
    extra: dict[str, pd.DataFrame] = {}
    if not audit.empty:
        extra["rebalance_audit.csv"] = audit
    if not pair_stats.empty:
        extra["pair_stats.csv"] = pair_stats
        sleeve_pnl = (
            pair_stats.groupby("sleeve", as_index=False)
            .agg(n_pairs=("ETF", "count"), pnl_usd=("pnl_usd", "sum"))
            .sort_values("pnl_usd")
        )
        extra["sleeve_pnl.csv"] = sleeve_pnl
    if not sleeve_daily.empty:
        extra["sleeve_daily_pnl.csv"] = sleeve_daily
        extra["daily_diagnostics.csv"] = sleeve_daily
        extra["sleeve_return_metrics.csv"] = compute_sleeve_return_metrics(sleeve_daily)
    if not pair_daily.empty:
        extra["pair_daily_pnl.csv"] = pair_daily
    if not pending_target_audit.empty:
        extra["pending_target_audit.csv"] = pending_target_audit
    _write_outputs(
        outdir=outdir,
        mode="replay",
        report=report,
        summary=summary,
        series=series,
        extra_csvs=extra or None,
    )
    print(f"[prod-bt:replay] wrote {outdir}")
    print(
        f"[prod-bt:replay] BOOK CAGR={book_meta.get('cagr')} MaxDD={book_meta.get('maxdd')} "
        f"end=${book_meta.get('end_usd'):,.0f} plans={book_meta.get('n_plans_used')} cash_days={book_meta.get('cash_days')}"
    )
    return report


def run_prod_replay_backtest(
    *,
    run_date: str,
    start: str,
    outdir: Path,
    pre_archive_policy: str = "cash",
    state_root: Path | str | None = None,
    reuse_plans: bool = False,
    blacklist_except: list[str] | tuple[str, ...] | None = None,
    price_panel_min_days: int | None = None,
    notebook_b4_borrow: Mapping[str, Any] | None = None,
) -> dict:
    """Primary path: full GTP stack (opt2→crash→smooth→ratchet) on screened archives."""
    cfg = load_config(REPO / "config" / "strategy_config.yml")
    removed_bl = apply_blacklist_except(cfg, blacklist_except)
    if removed_bl:
        print(f"[prod-bt:prod] blacklist exception (removed for this run): {removed_bl}")
    b4_borrow_audit: dict[str, Any] = {}
    if notebook_b4_borrow:
        b4_borrow_audit = apply_notebook_b4_borrow_overrides(
            cfg,
            entry_borrow_cap=notebook_b4_borrow.get("entry_borrow_cap"),
            keep_borrow_cap=notebook_b4_borrow.get("keep_borrow_cap"),
            borrow_ramp_lo=notebook_b4_borrow.get("borrow_ramp_lo"),
            borrow_ramp_hi=notebook_b4_borrow.get("borrow_ramp_hi"),
            shift_ramp_with_band=bool(
                notebook_b4_borrow.get("shift_ramp_with_band", True)
            ),
        )
        print(f"[prod-bt:prod] notebook B4 borrow overrides (cfg only): {b4_borrow_audit}")
        if reuse_plans:
            print(
                "[prod-bt:prod] WARN: --reuse-plans ignores B4 borrow overrides for sizing; "
                "set reuse_plans=False to rebuild plans under 60/80"
            )
    capital = float(cfg["strategy"]["capital_usd"])
    lev = float(cfg["strategy"]["gross_leverage"])
    budgets = sleeve_budgets_usd(cfg)
    rknobs = rebalance_knobs(cfg)
    start_ts = pd.Timestamp(start)

    cov = archive_coverage_summary(start)
    print("[prod-bt:prod] archive coverage:")
    print(cov.to_string(index=False))

    plans_dir = outdir / "plans"
    diag = pd.DataFrame()
    if reuse_plans:
        print(f"[prod-bt:prod] reusing cached plans from {plans_dir} ...")
        if removed_bl:
            print(
                "[prod-bt:prod] WARN: --reuse-plans ignores blacklist_except for sizing; "
                "re-run without reuse to rebuild plans"
            )
        timeline = load_cached_plan_timeline(plans_dir)
        if not timeline:
            raise RuntimeError(
                f"--reuse-plans set but no plan CSVs found under {plans_dir}"
            )
        n_ok = len(timeline)
        n_fail = 0
        n_fallback = 0
        print(f"[prod-bt:prod] timeline={len(timeline)} from cache (skip GTP sizing)")
    else:
        print("[prod-bt:prod] building plan timeline via full GTP sizing replay ...")
        timeline, diag = prod_replay_plan_timeline(
            cfg=cfg,
            start=start_ts,
            state_root=state_root,
            keep_state=state_root is not None,
            plans_dir=plans_dir,
        )
        if not timeline:
            raise RuntimeError(
                "prod replay produced empty timeline — need archived etf_screened_today.csv"
            )

        n_ok = int(diag["ok"].astype(bool).sum()) if not diag.empty and "ok" in diag.columns else 0
        n_fail = int((~diag["ok"].astype(bool)).sum()) if not diag.empty and "ok" in diag.columns else 0
        n_fallback = (
            int((diag["source"] == "archived_proposed_fallback").sum()) if not diag.empty else 0
        )
        print(
            f"[prod-bt:prod] timeline={len(timeline)} sized_ok={n_ok} "
            f"sized_fail={n_fail} archived_fallback={n_fallback}"
        )
        if not diag.empty and "gross_b4" in diag.columns and n_ok:
            print(
                f"[prod-bt:prod] B4 gross median=${float(diag.loc[diag['ok'], 'gross_b4'].median()):,.0f} "
                f"max=${float(diag.loc[diag['ok'], 'gross_b4'].max()):,.0f}"
            )

    panel_kwargs: dict[str, Any] = {}
    if price_panel_min_days is not None:
        panel_kwargs["min_days"] = int(price_panel_min_days)
        print(f"[prod-bt:prod] price panel min_days={int(price_panel_min_days)}")
    panel = load_timeline_price_panel(
        timeline,
        run_date=run_date,
        min_days=int(panel_kwargs.get("min_days", 40)),
    )
    nav, audit, meta, pair_stats, sleeve_daily = simulate_book_from_plan_timeline(
        timeline,
        panel,
        budgets=budgets,
        capital_usd=capital,
        start=start_ts,
        slippage_bps=rknobs["slippage_bps"],
        commission_per_share=rknobs["commission_per_share"],
        margin_rate_annual=rknobs["margin_rate_annual"],
        financing_daycount=rknobs["financing_daycount"],
        short_proceeds_credit_annual=rknobs["short_proceeds_credit_annual"],
        execution_lag_sessions=rknobs["execution_lag_sessions"],
        target_notional_mode=rknobs["target_notional_mode"],
        scale_sleeves_to_budget=bool(rknobs.get("scale_sleeves_to_budget", True)),
        enter_band_pct=rknobs["enter_band_pct"],
        exit_band_pct=rknobs["exit_band_pct"],
        min_trade_usd=rknobs["min_trade_usd"],
        check_freq="W-FRI",
        retarget_on_plan_change=bool(rknobs.get("retarget_on_plan_change", False)),
        pre_archive_policy=pre_archive_policy,
        b4_execution=str(rknobs.get('b4_execution', 'cadence')),
        apply_delist_flatten=bool(rknobs.get('apply_delist_flatten', True)),
        use_borrow_history=bool(rknobs.get('use_borrow_history', True)),
        same_run_churn_enabled=bool(rknobs.get('same_run_churn_enabled', True)),
        purgatory_model_zero_policy=str(rknobs.get('purgatory_model_zero_policy', 'hold')),
        b4_membership_clock=str(rknobs.get('b4_membership_clock', 'operator_5d')),
        stock_rebalance_clock=str(rknobs.get('stock_rebalance_clock', 'operator_5d')),
        operator_check_days=int(rknobs.get('operator_check_days', 5) or 5),
        b4_apply_resize_bands=bool(rknobs.get('b4_apply_resize_bands', True)),
        b4_ratchet_execution_guard=bool(rknobs.get('b4_ratchet_execution_guard', True)),
        b4_allow_inverse_cover=bool(rknobs.get('b4_allow_inverse_cover', True)),
        b4_empty_plan_policy=str(rknobs.get('b4_empty_plan_policy', 'hold')),
        net_shared_underlyings=bool(rknobs.get('net_shared_underlyings', True)),
        turnover_pace_enabled=bool(rknobs.get('turnover_pace_enabled', True)),
        turnover_pace_mode=str(rknobs.get('turnover_pace_mode', 'hedge_safe_v1')),
        confirmation_count=int(rknobs.get('confirmation_count', 2)),
        entry_ramp_sessions=int(rknobs.get('entry_ramp_sessions', 5)),
        reduction_ramp_sessions=int(rknobs.get('reduction_ramp_sessions', 3)),
        remaining_gap_rate=float(rknobs.get('remaining_gap_rate', 0.25)),
        target_blend_alpha=float(rknobs.get('target_blend_alpha', 0.25)),
        stock_midweek_mode=str(rknobs.get('stock_midweek_mode', 'rebal_only')),
        midweek_hedge_repair=bool(rknobs.get('midweek_hedge_repair', False)),
        hedge_reserve_frac=float(rknobs.get('hedge_reserve_frac', 0.20)),
        adv_participation_pct=float(rknobs.get('adv_participation_pct', 0.10)),
        sleeve_gross_ema_alpha=float(rknobs.get('sleeve_gross_ema_alpha', 0.35)),
        max_leg_step_pct=float(rknobs.get('max_leg_step_pct', 0.25)),
        pair_gross_ramp_pct=float(rknobs.get('pair_gross_ramp_pct', 0.25)),
        max_daily_turnover_pct=float(rknobs.get('max_daily_turnover_pct', 0.15)),
        legacy_max_daily_turnover_pct=float(rknobs.get('legacy_max_daily_turnover_pct', 0.15)),
        establish_budget_frac=float(rknobs.get('establish_budget_frac', 0.50)),
        resize_age_boost_days=int(rknobs.get('resize_age_boost_days', 5)),
        hedge_long_trigger_net_pct=float(rknobs.get('hedge_long_trigger_net_pct', 0.04)),
        hedge_long_target_net_pct=float(rknobs.get('hedge_long_target_net_pct', 0.01)),
        hedge_short_trigger_net_pct=float(rknobs.get('hedge_short_trigger_net_pct', 0.01)),
        hedge_short_target_net_pct=float(rknobs.get('hedge_short_target_net_pct', 0.00)),
    )

    pair_daily = _take_pair_daily(meta)
    pending_target_audit = _take_pending_target_audit(meta)
    b4_membership = build_b4_membership_manifest(
        timeline,
        pair_daily,
        panel,
        run_end=str(nav.index[-1].date()) if len(nav) else None,
    )
    book_meta = {"sleeve": "BOOK", "mode": "prod", **meta}
    summary = pd.DataFrame([book_meta])
    series = pd.DataFrame({"BOOK_NAV": nav})
    if not sleeve_daily.empty:
        for s in ALL_SLEEVES:
            col = f"{s}_cum_pnl"
            if col in sleeve_daily.columns:
                series[s] = (
                    capital + sleeve_daily.set_index("date")[col]
                ).reindex(series.index).ffill()

    b4_budget = float(budgets.get("inverse_decay_bucket4") or 0.0)
    report = {
        "mode": "prod",
        "run_date": run_date,
        "start": start,
        "end": str(nav.index[-1].date()) if len(nav) else None,
        "capital_usd": capital,
        "gross_leverage": lev,
        "budgets_usd": budgets,
        "rebalance_knobs": rknobs,
        "archive_coverage": cov.to_dict(orient="records"),
        "prod_stats": {
            "n_timeline": len(timeline),
            "n_sized_ok": n_ok,
            "n_sized_fail": n_fail,
            "b4_budget_usd": b4_budget,
            "blacklist_except": list(removed_bl),
            "notebook_b4_borrow": dict(b4_borrow_audit) if b4_borrow_audit else None,
            "price_panel_min_days": (
                int(price_panel_min_days) if price_panel_min_days is not None else None
            ),
            "b4_gross_median": float(diag.loc[diag["ok"], "gross_b4"].median())
            if not diag.empty and n_ok and "gross_b4" in diag.columns
            else None,
            "b4_gross_max": float(diag.loc[diag["ok"], "gross_b4"].max())
            if not diag.empty and n_ok and "gross_b4" in diag.columns
            else None,
            "b4_membership_count": int(len(b4_membership)),
            "b4_membership_with_ledger": int(b4_membership.get("has_ledger", pd.Series(dtype=bool)).sum()),
            "b4_membership_blocked": int(b4_membership.get("lifecycle_state", pd.Series(dtype=str)).eq("blocked").sum()),
        },
        "book": {k: book_meta.get(k) for k in BOOK_REPORT_KEYS},
        "limitations": [
            "Daily targets from full generate_trade_plan on archived etf_screened_today.csv "
            "(opt2 → crash → smooth → ratchet) with isolated state carried forward.",
            "Borrow/edge inputs: screened spot borrow_current + production edge/opt2 path "
            "(no avg-borrow overlay).",
            "Ratchet floors from prior-day sized plan (simulated), not live Flex holdings.",
            "Does not prefer archived proposed_trades.csv, but falls back to them when "
            "prod sizing fails or on plan-only archive dates.",
            "Archive gap ~Dec 2025 / sparse screened: pre-2026-04-25 archives lack "
            "net_edge_p50_annual — prod replay shims from net_decay_annual (backtest-only).",
            "B5 included only when GTP sizes it; no live locates / execution rejects.",
            "B1/B2 retarget on stock_rebalance_clock (default operator_5d) with Phase-2b hysteresis; purgatory is "
            "reduce-only toward model_* (trim, never increase gross); missing/zero "
            "model_* share-holds (purgatory_model_zero_policy=hold) — executable 0 "
            "is not a flatten.",
            "B4: TR/VCR cadence + Phase-2b bands on resize; membership add/drop gated "
            "by b4_membership_clock (default operator_5d); inverse ratchet pin/trim-cap "
            "on covers (b4_ratchet_execution_guard); empty B4 plan (exec gross~0) "
            "share-holds the open sleeve (b4_empty_plan_policy=hold).",
            "Shared underlyings net for financing when net_shared_underlyings=true: "
            "borrow / short-credit / margin use residual net short/long only "
            "(B1/B2 long vs B4 short internalization). Price PnL still marks each "
            "pair leg; book gross/net notionals are netted.",
            "Sim-only turnover pacing (turnover_pace): EMA stock-sleeve gross, "
            "per-leg max step toward plan, soft daily book turnover budget "
            "(exits first, then establishes, then resizes).",
            "Price panel: flex splits + overrides + price_patches + Yahoo referee; "
            "Yahoo tail extend; delist cutoff from data/delistings.csv.",
            "Sleeve legs scaled to YAML sleeve budgets (scale_sleeves_to_budget) then equity-scaled with NAV.",
            "IBKR short-sale proceeds credit modelled at 3.8% annual on short notional (Actual/360); "
            "borrow fee from screened rates with optional borrow_history overlay.",
            "Point-in-time legs use next-close execution, share-hold marking, 20 bp slippage, "
            "$0.0035/share commissions, and 4.45% margin debit / Actual-360.",
            "B3 flow excluded.",
            "Screened archives begin 2026-02-27 for this run window (sparse thereafter).",
            "Ratchet floors from prior-day sized plan (simulated), not live Flex holdings.",
        ]
        + (
            [
                f"Notebook blacklist exception (not live YAML): re-admitted {', '.join(removed_bl)}."
            ]
            if removed_bl
            else []
        )
        + (
            [
                f"Price panel min_days overridden to {int(price_panel_min_days)} "
                "(default 40) so short-history names like CBRZ can mark."
            ]
            if price_panel_min_days is not None
            else []
        ),
    }
    extra: dict[str, pd.DataFrame] = {}
    if not audit.empty:
        extra["rebalance_audit.csv"] = audit
    if not diag.empty:
        extra["prod_sizing_diag.csv"] = diag
    if not pair_stats.empty:
        extra["pair_stats.csv"] = pair_stats
        extra["sleeve_pnl.csv"] = (
            pair_stats.groupby("sleeve", as_index=False)
            .agg(n_pairs=("ETF", "count"), pnl_usd=("pnl_usd", "sum"))
            .sort_values("pnl_usd")
        )
    if not sleeve_daily.empty:
        extra["sleeve_daily_pnl.csv"] = sleeve_daily
        extra["daily_diagnostics.csv"] = sleeve_daily
        extra["sleeve_return_metrics.csv"] = compute_sleeve_return_metrics(sleeve_daily)
    # Always write (even empty) so a stale file from a killed run cannot linger.
    extra["pair_daily_pnl.csv"] = pair_daily
    extra["pending_target_audit.csv"] = pending_target_audit
    extra["b4_membership_manifest.csv"] = b4_membership
    _write_outputs(
        outdir=outdir,
        mode="prod",
        report=report,
        summary=summary,
        series=series,
        extra_csvs=extra or None,
    )
    print(f"[prod-bt:prod] wrote {outdir}")
    print(
        f"[prod-bt:prod] BOOK CAGR={book_meta.get('cagr')} MaxDD={book_meta.get('maxdd')} "
        f"end=${book_meta.get('end_usd'):,.0f}"
    )
    return report


def run_gtp_approx_backtest(
    *,
    run_date: str,
    start: str,
    outdir: Path,
    pre_archive_policy: str = "cash",
) -> dict:
    """Primary path: daily GTP-approx sizing from screened (avg borrow + net edge)."""
    cfg = load_config(REPO / "config" / "strategy_config.yml")
    capital = float(cfg["strategy"]["capital_usd"])
    lev = float(cfg["strategy"]["gross_leverage"])
    budgets = sleeve_budgets_usd(cfg)
    rknobs = rebalance_knobs(cfg)
    start_ts = pd.Timestamp(start)

    cov = archive_coverage_summary(start)
    print("[prod-bt:gtp] archive coverage:")
    print(cov.to_string(index=False))

    print("[prod-bt:gtp] building plan timeline via GTP approx (avg borrow + net edge) ...")
    timeline, diag = gtp_approx_plan_timeline(cfg=cfg, start=start_ts)
    if not timeline:
        raise RuntimeError(
            "gtp approx produced empty timeline — need archived etf_screened_today.csv "
            "(no proposed_trades fallback)"
        )

    n_ok = int((diag["source"] == "gtp_approx").sum()) if not diag.empty else 0
    n_fail = int((diag["source"] == "gtp_approx_failed").sum()) if not diag.empty else 0
    print(f"[prod-bt:gtp] timeline={len(timeline)} sized_ok={n_ok} sized_fail={n_fail}")

    panel = load_price_panel(run_date)
    nav, audit, meta, pair_stats, sleeve_daily = simulate_book_from_plan_timeline(
        timeline,
        panel,
        budgets=budgets,
        capital_usd=capital,
        start=start_ts,
        slippage_bps=rknobs["slippage_bps"],
        commission_per_share=rknobs["commission_per_share"],
        margin_rate_annual=rknobs["margin_rate_annual"],
        financing_daycount=rknobs["financing_daycount"],
        short_proceeds_credit_annual=rknobs["short_proceeds_credit_annual"],
        execution_lag_sessions=rknobs["execution_lag_sessions"],
        target_notional_mode=rknobs["target_notional_mode"],
        scale_sleeves_to_budget=bool(rknobs.get("scale_sleeves_to_budget", True)),
        enter_band_pct=rknobs["enter_band_pct"],
        exit_band_pct=rknobs["exit_band_pct"],
        min_trade_usd=rknobs["min_trade_usd"],
        check_freq="W-FRI",
        # Size daily from screened; trade on Friday with the latest plan so
        # daily score reshuffles do not churn the whole book at 20 bp.
        retarget_on_plan_change=bool(rknobs.get("retarget_on_plan_change", False)),
        pre_archive_policy=pre_archive_policy,
        b4_execution=str(rknobs.get('b4_execution', 'cadence')),
        apply_delist_flatten=bool(rknobs.get('apply_delist_flatten', True)),
        use_borrow_history=bool(rknobs.get('use_borrow_history', True)),
        same_run_churn_enabled=bool(rknobs.get('same_run_churn_enabled', True)),
        purgatory_model_zero_policy=str(rknobs.get('purgatory_model_zero_policy', 'hold')),
        b4_membership_clock=str(rknobs.get('b4_membership_clock', 'operator_5d')),
        stock_rebalance_clock=str(rknobs.get('stock_rebalance_clock', 'operator_5d')),
        operator_check_days=int(rknobs.get('operator_check_days', 5) or 5),
        b4_apply_resize_bands=bool(rknobs.get('b4_apply_resize_bands', True)),
        b4_ratchet_execution_guard=bool(rknobs.get('b4_ratchet_execution_guard', True)),
        b4_allow_inverse_cover=bool(rknobs.get('b4_allow_inverse_cover', True)),
        b4_empty_plan_policy=str(rknobs.get('b4_empty_plan_policy', 'hold')),
        net_shared_underlyings=bool(rknobs.get('net_shared_underlyings', True)),
        turnover_pace_enabled=bool(rknobs.get('turnover_pace_enabled', True)),
        turnover_pace_mode=str(rknobs.get('turnover_pace_mode', 'hedge_safe_v1')),
        confirmation_count=int(rknobs.get('confirmation_count', 2)),
        entry_ramp_sessions=int(rknobs.get('entry_ramp_sessions', 5)),
        reduction_ramp_sessions=int(rknobs.get('reduction_ramp_sessions', 3)),
        remaining_gap_rate=float(rknobs.get('remaining_gap_rate', 0.25)),
        target_blend_alpha=float(rknobs.get('target_blend_alpha', 0.25)),
        stock_midweek_mode=str(rknobs.get('stock_midweek_mode', 'rebal_only')),
        midweek_hedge_repair=bool(rknobs.get('midweek_hedge_repair', False)),
        hedge_reserve_frac=float(rknobs.get('hedge_reserve_frac', 0.20)),
        adv_participation_pct=float(rknobs.get('adv_participation_pct', 0.10)),
        sleeve_gross_ema_alpha=float(rknobs.get('sleeve_gross_ema_alpha', 0.35)),
        max_leg_step_pct=float(rknobs.get('max_leg_step_pct', 0.25)),
        pair_gross_ramp_pct=float(rknobs.get('pair_gross_ramp_pct', 0.25)),
        max_daily_turnover_pct=float(rknobs.get('max_daily_turnover_pct', 0.15)),
        legacy_max_daily_turnover_pct=float(rknobs.get('legacy_max_daily_turnover_pct', 0.15)),
        establish_budget_frac=float(rknobs.get('establish_budget_frac', 0.50)),
        resize_age_boost_days=int(rknobs.get('resize_age_boost_days', 5)),
        hedge_long_trigger_net_pct=float(rknobs.get('hedge_long_trigger_net_pct', 0.04)),
        hedge_long_target_net_pct=float(rknobs.get('hedge_long_target_net_pct', 0.01)),
        hedge_short_trigger_net_pct=float(rknobs.get('hedge_short_trigger_net_pct', 0.01)),
        hedge_short_target_net_pct=float(rknobs.get('hedge_short_target_net_pct', 0.00)),
    )

    pair_daily = _take_pair_daily(meta)
    pending_target_audit = _take_pending_target_audit(meta)
    book_meta = {"sleeve": "BOOK", "mode": "gtp", **meta}
    summary = pd.DataFrame([book_meta])
    series = pd.DataFrame({"BOOK_NAV": nav})
    if not sleeve_daily.empty:
        for s in ALL_SLEEVES:
            col = f"{s}_cum_pnl"
            if col in sleeve_daily.columns:
                series[s] = (
                    capital + sleeve_daily.set_index("date")[col]
                ).reindex(series.index).ffill()

    report = {
        "mode": "gtp",
        "run_date": run_date,
        "start": start,
        "end": str(nav.index[-1].date()) if len(nav) else None,
        "capital_usd": capital,
        "gross_leverage": lev,
        "budgets_usd": budgets,
        "rebalance_knobs": rknobs,
        "archive_coverage": cov.to_dict(orient="records"),
        "gtp_stats": {
            "n_timeline": len(timeline),
            "n_sized_ok": n_ok,
            "n_sized_fail": n_fail,
            "n_borrow_avg_rows": int(diag["n_borrow_avg"].sum()) if not diag.empty and "n_borrow_avg" in diag.columns else 0,
        },
        "book": {k: book_meta.get(k) for k in BOOK_REPORT_KEYS},
        "limitations": [
            "Daily targets from mirror_generate_trade_plan_sizing on archived etf_screened_today.csv.",
            "Borrow input: borrow_avg_annual when finite, else borrow_current (spot / IBKR feerate).",
            "Sizing signal: net_edge_p50_annual (YAML sizing_edge_column) minus borrow_aversion × borrow.",
            "Does not consume archived proposed_trades.csv; B5 is not sized by the mirror.",
            "Average borrow is often higher than spot → tighter entry caps → fewer pairs than replay.",
            "Plans sized every screened day; B1/B2 retarget on stock_rebalance_clock (default operator_5d).",
            "IBKR short-sale proceeds credit modelled at 3.8% annual on short notional (Actual/360); "
            "borrow fee still charged from screened/IBKR rates.",
            "Mirror decay-score path only — no live B4 opt2 / crash-budget / post-cap smooth / ratchet.",
            "B1 hysteresis carried across days in an isolated temp file (prod state untouched).",
            "Point-in-time legs use next-close execution, share-hold marking, 20 bp slippage, "
            "$0.0035/share commissions, and 4.45% margin debit / Actual-360.",
            "B3 flow excluded.",
        ],
    }
    extra: dict[str, pd.DataFrame] = {}
    if not audit.empty:
        extra["rebalance_audit.csv"] = audit
    if not diag.empty:
        extra["gtp_sizing_diag.csv"] = diag
    if not pair_stats.empty:
        extra["pair_stats.csv"] = pair_stats
        extra["sleeve_pnl.csv"] = (
            pair_stats.groupby("sleeve", as_index=False)
            .agg(n_pairs=("ETF", "count"), pnl_usd=("pnl_usd", "sum"))
            .sort_values("pnl_usd")
        )
    if not sleeve_daily.empty:
        extra["sleeve_daily_pnl.csv"] = sleeve_daily
        extra["daily_diagnostics.csv"] = sleeve_daily
        extra["sleeve_return_metrics.csv"] = compute_sleeve_return_metrics(sleeve_daily)
    if not pair_daily.empty:
        extra["pair_daily_pnl.csv"] = pair_daily
    if not pending_target_audit.empty:
        extra["pending_target_audit.csv"] = pending_target_audit
    _write_outputs(
        outdir=outdir,
        mode="gtp",
        report=report,
        summary=summary,
        series=series,
        extra_csvs=extra or None,
    )
    print(f"[prod-bt:gtp] wrote {outdir}")
    print(
        f"[prod-bt:gtp] BOOK CAGR={book_meta.get('cagr')} MaxDD={book_meta.get('maxdd')} "
        f"end=${book_meta.get('end_usd'):,.0f}"
    )
    return report


def run_recompute_backtest(
    *,
    run_date: str,
    start: str,
    outdir: Path,
    pre_archive_policy: str = "cash",
) -> dict:
    """Phase B: mirror GTP on screened dates; fall back to archived plans."""
    cfg = load_config(REPO / "config" / "strategy_config.yml")
    capital = float(cfg["strategy"]["capital_usd"])
    lev = float(cfg["strategy"]["gross_leverage"])
    budgets = sleeve_budgets_usd(cfg)
    rknobs = rebalance_knobs(cfg)
    start_ts = pd.Timestamp(start)

    cov = archive_coverage_summary(start)
    print("[prod-bt:recompute] archive coverage:")
    print(cov.to_string(index=False))

    print("[prod-bt:recompute] building plan timeline via mirror + archived fallback ...")
    timeline, diag = recompute_plan_timeline(cfg=cfg, start=start_ts, fallback_to_archived=True)
    if not timeline:
        raise RuntimeError("recompute produced empty timeline — need screened or archived plans")

    n_mirror = int((diag["source"] == "recompute_mirror").sum()) if not diag.empty else 0
    n_fail = int((diag["source"] == "recompute_failed").sum()) if not diag.empty else 0
    print(f"[prod-bt:recompute] timeline={len(timeline)} mirror_ok={n_mirror} mirror_fail={n_fail}")

    panel = load_price_panel(run_date)
    nav, audit, meta, pair_stats, sleeve_daily = simulate_book_from_plan_timeline(
        timeline,
        panel,
        budgets=budgets,
        capital_usd=capital,
        start=start_ts,
        slippage_bps=rknobs["slippage_bps"],
        commission_per_share=rknobs["commission_per_share"],
        margin_rate_annual=rknobs["margin_rate_annual"],
        financing_daycount=rknobs["financing_daycount"],
        short_proceeds_credit_annual=rknobs["short_proceeds_credit_annual"],
        execution_lag_sessions=rknobs["execution_lag_sessions"],
        target_notional_mode=rknobs["target_notional_mode"],
        scale_sleeves_to_budget=bool(rknobs.get("scale_sleeves_to_budget", True)),
        enter_band_pct=rknobs["enter_band_pct"],
        exit_band_pct=rknobs["exit_band_pct"],
        min_trade_usd=rknobs["min_trade_usd"],
        check_freq="W-FRI",
        retarget_on_plan_change=True,
        pre_archive_policy=pre_archive_policy,
        b4_execution=str(rknobs.get('b4_execution', 'cadence')),
        apply_delist_flatten=bool(rknobs.get('apply_delist_flatten', True)),
        use_borrow_history=bool(rknobs.get('use_borrow_history', True)),
        same_run_churn_enabled=bool(rknobs.get('same_run_churn_enabled', True)),
        purgatory_model_zero_policy=str(rknobs.get('purgatory_model_zero_policy', 'hold')),
        b4_membership_clock=str(rknobs.get('b4_membership_clock', 'operator_5d')),
        stock_rebalance_clock=str(rknobs.get('stock_rebalance_clock', 'operator_5d')),
        operator_check_days=int(rknobs.get('operator_check_days', 5) or 5),
        b4_apply_resize_bands=bool(rknobs.get('b4_apply_resize_bands', True)),
        b4_ratchet_execution_guard=bool(rknobs.get('b4_ratchet_execution_guard', True)),
        b4_allow_inverse_cover=bool(rknobs.get('b4_allow_inverse_cover', True)),
        b4_empty_plan_policy=str(rknobs.get('b4_empty_plan_policy', 'hold')),
        net_shared_underlyings=bool(rknobs.get('net_shared_underlyings', True)),
        turnover_pace_enabled=bool(rknobs.get('turnover_pace_enabled', True)),
        turnover_pace_mode=str(rknobs.get('turnover_pace_mode', 'hedge_safe_v1')),
        confirmation_count=int(rknobs.get('confirmation_count', 2)),
        entry_ramp_sessions=int(rknobs.get('entry_ramp_sessions', 5)),
        reduction_ramp_sessions=int(rknobs.get('reduction_ramp_sessions', 3)),
        remaining_gap_rate=float(rknobs.get('remaining_gap_rate', 0.25)),
        target_blend_alpha=float(rknobs.get('target_blend_alpha', 0.25)),
        stock_midweek_mode=str(rknobs.get('stock_midweek_mode', 'rebal_only')),
        midweek_hedge_repair=bool(rknobs.get('midweek_hedge_repair', False)),
        hedge_reserve_frac=float(rknobs.get('hedge_reserve_frac', 0.20)),
        adv_participation_pct=float(rknobs.get('adv_participation_pct', 0.10)),
        sleeve_gross_ema_alpha=float(rknobs.get('sleeve_gross_ema_alpha', 0.35)),
        max_leg_step_pct=float(rknobs.get('max_leg_step_pct', 0.25)),
        pair_gross_ramp_pct=float(rknobs.get('pair_gross_ramp_pct', 0.25)),
        max_daily_turnover_pct=float(rknobs.get('max_daily_turnover_pct', 0.15)),
        legacy_max_daily_turnover_pct=float(rknobs.get('legacy_max_daily_turnover_pct', 0.15)),
        establish_budget_frac=float(rknobs.get('establish_budget_frac', 0.50)),
        resize_age_boost_days=int(rknobs.get('resize_age_boost_days', 5)),
        hedge_long_trigger_net_pct=float(rknobs.get('hedge_long_trigger_net_pct', 0.04)),
        hedge_long_target_net_pct=float(rknobs.get('hedge_long_target_net_pct', 0.01)),
        hedge_short_trigger_net_pct=float(rknobs.get('hedge_short_trigger_net_pct', 0.01)),
        hedge_short_target_net_pct=float(rknobs.get('hedge_short_target_net_pct', 0.00)),
    )

    pair_daily = _take_pair_daily(meta)
    pending_target_audit = _take_pending_target_audit(meta)
    book_meta = {"sleeve": "BOOK", "mode": "recompute", **meta}
    summary = pd.DataFrame([book_meta])
    series = pd.DataFrame({"BOOK_NAV": nav})
    if not sleeve_daily.empty:
        for s in ALL_SLEEVES:
            col = f"{s}_cum_pnl"
            if col in sleeve_daily.columns:
                series[s] = (
                    capital + sleeve_daily.set_index("date")[col]
                ).reindex(series.index).ffill()

    report = {
        "mode": "recompute",
        "run_date": run_date,
        "start": start,
        "end": str(nav.index[-1].date()) if len(nav) else None,
        "capital_usd": capital,
        "gross_leverage": lev,
        "budgets_usd": budgets,
        "rebalance_knobs": rknobs,
        "archive_coverage": cov.to_dict(orient="records"),
        "recompute_stats": {
            "n_timeline": len(timeline),
            "n_mirror_ok": n_mirror,
            "n_mirror_fail": n_fail,
            "n_archived_only": int((diag["source"] == "archived_only").sum()) if not diag.empty else 0,
            "n_fallback_archived": int((diag["source"] == "fallback_archived").sum()) if not diag.empty else 0,
        },
        "book": {k: book_meta.get(k) for k in BOOK_REPORT_KEYS},
        "limitations": [
            "Phase B: mirror_generate_trade_plan_sizing on archived screened CSVs (decay-score path).",
            "Mirror does NOT run live B4 opt2 / crash-budget / scale_to_budget / post-cap smooth / ratchet — B4 hedge is plan long/short fractions.",
            "B1 hysteresis state is carried across recompute days in an isolated temp file (prod state untouched).",
            "Days without screened fall back to archived proposed_trades.",
            "Existing legs use production Phase-2b enter/exit hysteresis and the configured minimum trade.",
            "Point-in-time legs use next-close execution, share-hold marking, 20 bp slippage, $0.0035/share commissions, borrow, and 4.45% margin debit / Actual-360.",
            "The 4.45% financing input is a visible fallback, not a point-in-time OBFR curve.",
            "Pre-archive window uses scripts/backfill_screened_history.py (PIT prices; "
            "borrow carry-first-known/default; shares stubbed; no live FTP locates).",
            "B3 flow excluded.",
        ],
    }
    extra: dict[str, pd.DataFrame] = {}
    if not audit.empty:
        extra["rebalance_audit.csv"] = audit
    if not diag.empty:
        extra["recompute_diag.csv"] = diag
    if not pair_stats.empty:
        extra["pair_stats.csv"] = pair_stats
        extra["sleeve_pnl.csv"] = (
            pair_stats.groupby("sleeve", as_index=False)
            .agg(n_pairs=("ETF", "count"), pnl_usd=("pnl_usd", "sum"))
            .sort_values("pnl_usd")
        )
    if not sleeve_daily.empty:
        extra["sleeve_daily_pnl.csv"] = sleeve_daily
        extra["daily_diagnostics.csv"] = sleeve_daily
        extra["sleeve_return_metrics.csv"] = compute_sleeve_return_metrics(sleeve_daily)
    if not pair_daily.empty:
        extra["pair_daily_pnl.csv"] = pair_daily
    if not pending_target_audit.empty:
        extra["pending_target_audit.csv"] = pending_target_audit
    _write_outputs(
        outdir=outdir,
        mode="recompute",
        report=report,
        summary=summary,
        series=series,
        extra_csvs=extra or None,
    )
    print(f"[prod-bt:recompute] wrote {outdir}")
    print(
        f"[prod-bt:recompute] BOOK CAGR={book_meta.get('cagr')} MaxDD={book_meta.get('maxdd')} "
        f"end=${book_meta.get('end_usd'):,.0f}"
    )
    return report


def run_production_actual_backtest(
    *,
    run_date: str,
    start: str,
    outdir: Path,
    mode: str = "prod",
    pre_archive_policy: str = "cash",
    reuse_plans: bool = False,
    blacklist_except: list[str] | tuple[str, ...] | None = None,
    price_panel_min_days: int | None = None,
    notebook_b4_borrow: Mapping[str, Any] | None = None,
) -> dict:
    mode = str(mode or "prod").strip().lower()
    if mode in ("prod", "prod_replay"):
        return run_prod_replay_backtest(
            run_date=run_date,
            start=start,
            outdir=outdir,
            pre_archive_policy=pre_archive_policy,
            reuse_plans=reuse_plans,
            blacklist_except=blacklist_except,
            price_panel_min_days=price_panel_min_days,
            notebook_b4_borrow=notebook_b4_borrow,
        )
    if mode == "gtp":
        return run_gtp_approx_backtest(
            run_date=run_date,
            start=start,
            outdir=outdir,
            pre_archive_policy=pre_archive_policy,
        )
    if mode == "frozen":
        return run_frozen_backtest(run_date=run_date, start=start, outdir=outdir)
    if mode == "replay":
        return run_replay_backtest(
            run_date=run_date,
            start=start,
            outdir=outdir,
            pre_archive_policy=pre_archive_policy,
        )
    if mode == "recompute":
        return run_recompute_backtest(
            run_date=run_date,
            start=start,
            outdir=outdir,
            pre_archive_policy=pre_archive_policy,
        )
    raise ValueError(f"Unknown mode={mode!r}; use prod|gtp|frozen|replay|recompute")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-date", default="2026-07-10", help="Price-panel run date")
    ap.add_argument("--start", default="2026-02-27")
    ap.add_argument(
        "--mode",
        default="prod",
        choices=("prod", "gtp", "frozen", "replay", "recompute"),
        help="prod=full GTP historical replay (default); gtp=legacy mirror approx",
    )
    ap.add_argument(
        "--pre-archive-policy",
        default="cash",
        choices=("cash", "skip"),
        help="Before first sized plan: hold cash or start calendar at first plan",
    )
    ap.add_argument(
        "--reuse-plans",
        action="store_true",
        help="Skip GTP sizing; resimulate from outdir/plans/*.csv (split/panel fixes)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output dir (default: notebooks/output/production_actual_bt[/mode])",
    )
    args = ap.parse_args(argv)
    outdir = args.outdir
    if outdir is None:
        base = REPO / "notebooks" / "output" / "production_actual_bt"
        outdir = base if args.mode in ("prod", "gtp", "frozen") else base / args.mode
    run_production_actual_backtest(
        run_date=args.run_date,
        start=args.start,
        outdir=outdir,
        mode=args.mode,
        pre_archive_policy=args.pre_archive_policy,
        reuse_plans=bool(args.reuse_plans),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
