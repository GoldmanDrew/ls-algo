"""Production-actual book backtest (May 2025 -> present).

Modes
-----
``frozen``
    Single anchor ``proposed_trades.csv`` (legacy). Answers: given *today's*
    book (full B4 stack when GTP wrote it), what would NAV have done from
    ``--start``?

``replay`` (Phase A)
    Point-in-time signed legs from archived ``data/runs/*/proposed_trades.csv``.
    Plans execute at the next close; shares are held between weekly / plan
    events; Phase-2b resize bands, borrow, margin, commissions, and slippage are
    explicit. Archives start ~2025-12-28; before that the book holds cash.

``recompute`` (Phase B)
    Where ``etf_screened_today.csv`` exists, re-size via
    ``mirror_generate_trade_plan_sizing`` (B1/B2/B4 decay-score path; no live
    B4 opt2 / crash / smooth / ratchet). Falls back to archived proposed_trades
    when screened is missing or mirror fails. Carries B1 hysteresis state
    across days.

Outputs under ``notebooks/output/production_actual_bt/`` (or ``*_replay`` /
``*_recompute`` subdirs when those modes are selected).

Run
---
    python -m scripts.production_actual_backtest
    python -m scripts.production_actual_backtest --mode replay --start 2025-05-01
    python -m scripts.production_actual_backtest --mode recompute --start 2025-12-28
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from risk_dashboard.metrics import compute_sleeve_target_weights  # noqa: E402
from strategy_config import load_config  # noqa: E402
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

TRADING_DAYS = 252
STOCK_SLEEVES = ("core_leveraged", "yieldboost")
B4_SLEEVE = "inverse_decay_bucket4"
B5_SLEEVE = "volatility_etp_bucket5"
B4_SLEEVES = (B4_SLEEVE,)  # B5 routed separately via carry engine
ALL_SLEEVES = STOCK_SLEEVES + (B4_SLEEVE, B5_SLEEVE)
RUNS_DIR = REPO / "data" / "runs"
MIN_B4_SESSIONS = 40  # admit newer listings; signals warm up on available history
MIN_B4_TRADE_DAYS = 20

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


def rebalance_knobs(cfg: dict) -> dict:
    reb = ((cfg.get("portfolio") or {}).get("rebalance") or {})
    resize = reb.get("resize") or {}
    return {
        "target_basis": reb.get("target_basis", "hybrid"),
        "enter_band_pct": float(resize.get("enter_band_pct", 0.12) or 0.12),
        "exit_band_pct": float(resize.get("exit_band_pct", 0.04) or 0.04),
        "min_trade_usd": float(reb.get("min_trade_usd", 250) or 250),
        "stock_rebalance": "W-FRI",
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
        # A plan written on day T is assumed known after that close and is
        # executed at the next available close.  Its P&L starts on T+2 close.
        "execution_lag_sessions": 1,
        # Preserve the plan's gross/equity multiple as NAV changes, matching
        # Diamond Creek's dynamic target-gross convention.
        "target_notional_mode": "equity_scaled",
    }


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


def _infer_sleeve(row: pd.Series) -> str:
    if "sleeve" in row.index and pd.notna(row.get("sleeve")) and str(row.get("sleeve")).strip():
        s = str(row["sleeve"]).strip().lower()
        return SLEEVE_ALIASES.get(s, s)
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
    out = out[out["gross_target_usd"] > 0].copy()
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


def _targets_from_plan(
    plan: pd.DataFrame,
    *,
    budgets: dict[str, float],
    panel: dict[str, pd.DataFrame],
    equity: float,
    capital_usd: float,
    target_notional_mode: str,
) -> pd.DataFrame:
    """Build signed ETF/underlying close targets from one archived plan.

    Missing price panels remain undeployed.  They are deliberately *not*
    redistributed to surviving names.  ``equity_scaled`` preserves each
    archived plan's gross/equity multiple as NAV changes; ``fixed_plan_usd``
    replays the literal archived dollars.
    """
    cols = [
        "ETF", "Underlying", "sleeve", "gross_usd", "etf_usd", "underlying_usd",
        "borrow_current", "borrow_underlying", "Delta",
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
        g_all = pd.to_numeric(sub_all["gross_target_usd"], errors="coerce").fillna(0.0).clip(lower=0.0)
        g_sum = float(g_all.sum())
        if g_sum <= 0:
            continue
        sleeve_cap = min(float(budget), g_sum) * nav_scale
        planned_gross += sleeve_cap
        for (_, r), raw_gross in zip(sub_all.iterrows(), g_all.to_numpy()):
            etf = str(r["ETF"])
            if etf not in panel or raw_gross <= 0:
                continue
            gross_usd = float(raw_gross / g_sum * sleeve_cap)
            # proposed_trades contract: long_usd is the UNDERLYING target and
            # short_usd is the ETF target.  Prefer the explicit columns when
            # present so this cannot silently regress to the old reversed map.
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
    execution_lag_sessions: int = 1,
    target_notional_mode: str = "equity_scaled",
    enter_band_pct: float = 0.12,
    exit_band_pct: float = 0.04,
    min_trade_usd: float = 250.0,
    use_resize_bands: bool = True,
    check_freq: str = "W-FRI",
    pre_archive_policy: str = "cash",
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Close-to-close, share-hold simulation with point-in-time plans.

    Returns ``(nav, audit, meta, pair_stats, sleeve_daily)``.

    ``pre_archive_policy``
        ``cash`` — hold capital until first plan (default).
        ``skip`` — start calendar at first plan date.
    """
    empty = pd.DataFrame()
    if not timeline:
        return pd.Series(dtype=float), empty, {"error": "empty_timeline"}, empty, empty

    first_plan = min(timeline)
    sim_start = start
    if pre_archive_policy == "skip" and first_plan > start:
        sim_start = first_plan

    ret_cache = _build_return_cache(list(timeline.values()), panel)
    if not ret_cache:
        return pd.Series(dtype=float), empty, {"error": "no_returns"}, empty, empty

    all_idx = sorted(set().union(*[set(s.index) for s in ret_cache.values()]))
    cal = pd.DatetimeIndex([d for d in all_idx if d >= sim_start])
    if len(cal) < 20:
        return pd.Series(dtype=float), empty, {"error": "short_calendar"}, empty, empty

    check_days = set(
        pd.DatetimeIndex(pd.Series(1, index=cal).resample(check_freq).last().index).intersection(cal)
    )
    effective = _effective_plan_timeline(
        timeline, cal, execution_lag_sessions=execution_lag_sessions
    )
    slip = slippage_bps / 1e4
    equity = float(capital_usd)
    pos_cols = [
        "Underlying", "sleeve", "gross_usd", "etf_usd", "underlying_usd",
        "borrow_current", "borrow_underlying", "Delta", "plan_date",
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
    running_peak = float(capital_usd)

    def _ensure_contrib(etf: str, row: pd.Series | None = None) -> dict[str, Any]:
        if etf not in contrib:
            contrib[etf] = {
                "ETF": etf,
                "Underlying": str(row.get("Underlying", "") if row is not None else ""),
                "sleeve": str(row.get("sleeve", "") if row is not None else ""),
                "price_pnl_usd": 0.0,
                "borrow_cost_usd": 0.0,
                "margin_cost_usd": 0.0,
                "txn_cost_usd": 0.0,
            }
        elif row is not None:
            contrib[etf]["Underlying"] = str(row.get("Underlying", contrib[etf]["Underlying"]))
            contrib[etf]["sleeve"] = str(row.get("sleeve", contrib[etf]["sleeve"]))
        return contrib[etf]

    for i, d in enumerate(cal):
        equity_start = float(equity)
        gross_start = float(cur[["etf_usd", "underlying_usd"]].abs().sum().sum()) if not cur.empty else 0.0
        if gross_start <= 1e-9:
            cash_days += 1
        sleeve_comp = {
            s: {"price": 0.0, "borrow": 0.0, "margin": 0.0, "txn": 0.0}
            for s in ALL_SLEEVES
        }
        price_pnl = 0.0
        borrow_cost = 0.0
        margin_cost = 0.0
        txn_cost = 0.0
        turnover_day = 0.0
        stale_etf = 0
        stale_underlying = 0

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
                sl = str(row.get("sleeve", ""))
                if sl in sleeve_comp:
                    sleeve_comp[sl]["price"] += pnl_e

            # 2) Borrow on marked short notionals, using the rate attached to
            # the active archived plan. borrow_current belongs to the ETF;
            # borrow_underlying / underlying_borrow_annual belongs to the
            # underlying leg.
            for etf, row in cur.iterrows():
                sl = str(row.get("sleeve", ""))
                rate_a = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
                rate_b = float(pd.to_numeric(row.get("borrow_underlying"), errors="coerce") or 0.0)
                cost = 0.0
                if float(row["etf_usd"]) < 0:
                    cost += abs(float(row["etf_usd"])) * max(rate_a, 0.0) / TRADING_DAYS
                if float(row["underlying_usd"]) < 0:
                    cost += abs(float(row["underlying_usd"])) * max(rate_b, 0.0) / TRADING_DAYS
                borrow_cost += cost
                _ensure_contrib(etf, row)["borrow_cost_usd"] += cost
                if sl in sleeve_comp:
                    sleeve_comp[sl]["borrow"] += cost

            # Diamond Creek v15 convention: debit on positive long market
            # value less NAV; short proceeds do not offset the base.
            long_by_pair = {
                etf: max(float(row["etf_usd"]), 0.0) + max(float(row["underlying_usd"]), 0.0)
                for etf, row in cur.iterrows()
            }
            long_total = float(sum(long_by_pair.values()))
            debit_base = max(0.0, long_total - (equity + price_pnl - borrow_cost))
            margin_cost = debit_base * max(float(margin_rate_annual), 0.0) / max(float(financing_daycount), 1.0)
            if margin_cost > 0 and long_total > 0:
                for etf, basis in long_by_pair.items():
                    alloc = margin_cost * basis / long_total
                    row = cur.loc[etf]
                    _ensure_contrib(etf, row)["margin_cost_usd"] += alloc
                    sl = str(row.get("sleeve", ""))
                    if sl in sleeve_comp:
                        sleeve_comp[sl]["margin"] += alloc

        equity += price_pnl - borrow_cost - margin_cost

        # 3) At the close, activate any plan whose information lag has elapsed,
        # then retarget weekly.  New positions earn returns starting tomorrow.
        plan_changed = d in effective
        if plan_changed:
            active_plan_date, active_plan = effective[d]
            plans_used.add(active_plan_date)
        need_rebal = active_plan is not None and (plan_changed or d in check_days)
        target = pd.DataFrame(columns=pos_cols[:-1])
        target_planned_gross = 0.0
        target_tradeable_gross = 0.0
        blocked_pairs = 0
        if need_rebal and active_plan is not None:
            target = _targets_from_plan(
                active_plan,
                budgets=budgets,
                panel=panel,
                equity=equity,
                capital_usd=capital_usd,
                target_notional_mode=target_notional_mode,
            )
            target_planned_gross = float(target.attrs.get("planned_gross_usd", 0.0))
            target_tradeable_gross = float(target.attrs.get("tradeable_gross_usd", 0.0))
            target = target.copy()
            target["plan_date"] = str(active_plan_date.date()) if active_plan_date is not None else ""
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
            union = sorted(set(cur.index) | set(target.index))
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
                if use_resize_bands and old is not None and new is not None:
                    new_a = _resize_band_target(
                        old_a, new_a, enter_band_pct=enter_band_pct,
                        exit_band_pct=exit_band_pct, min_trade_usd=min_trade_usd,
                    )
                    new_b = _resize_band_target(
                        old_b, new_b, enter_band_pct=enter_band_pct,
                        exit_band_pct=exit_band_pct, min_trade_usd=min_trade_usd,
                    )
                turn_a = abs(new_a - old_a)
                turn_b = abs(new_b - old_b)
                pair_turn = turn_a + turn_b
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
                sl = str(ref.get("sleeve", "")) if ref is not None else ""
                if sl in sleeve_comp:
                    sleeve_comp[sl]["txn"] += pair_txn
                if new is None or (abs(new_a) + abs(new_b) <= 1e-9):
                    if etf in cur.index:
                        cur = cur.drop(index=etf)
                else:
                    exec_row = new.copy()
                    exec_row["etf_usd"] = new_a
                    exec_row["underlying_usd"] = new_b
                    exec_row["gross_usd"] = abs(new_a) + abs(new_b)
                    cur.loc[etf, pos_cols] = [exec_row.get(cn, np.nan) for cn in pos_cols]

            equity -= txn_cost
            n_rebal += 1
            turnover_total += turnover_day
            step_l1 = turnover_day / max(equity_start, 1e-9)
            turnover_l1 += step_l1
            deployed = float(cur[["etf_usd", "underlying_usd"]].abs().sum().sum()) if not cur.empty else 0.0
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
                    "deployed_gross_usd": deployed,
                    "untradeable_plan_gross_usd": max(0.0, target_planned_gross - target_tradeable_gross),
                    "blocked_pairs": blocked_pairs,
                    "n_added": n_added,
                    "n_exited": n_exited,
                    "n_resize_candidates": n_resized,
                    "n_resized": n_resized_executed,
                    "equity": equity,
                }
            )

        nav.iloc[i] = equity
        long_notional = 0.0
        short_notional = 0.0
        net_notional = 0.0
        if not cur.empty:
            vals = cur[["etf_usd", "underlying_usd"]].astype(float)
            long_notional = float(vals.clip(lower=0.0).sum().sum())
            short_notional = float(-vals.clip(upper=0.0).sum().sum())
            net_notional = float(vals.sum().sum())
        gross_notional = long_notional + short_notional
        largest_pair_share = np.nan
        top5_gross_share = np.nan
        gross_hhi = np.nan
        if gross_notional > 0 and not cur.empty:
            pair_gross = cur[["etf_usd", "underlying_usd"]].abs().sum(axis=1).sort_values(ascending=False)
            shares = pair_gross / gross_notional
            largest_pair_share = float(shares.iloc[0])
            top5_gross_share = float(shares.head(5).sum())
            gross_hhi = float((shares**2).sum())
        net_pnl = float(equity - equity_start)
        recon = net_pnl - (price_pnl - borrow_cost - margin_cost - txn_cost)
        running_peak = max(running_peak, float(equity))
        row_out: dict[str, Any] = {
            "date": d,
            "book_equity_start": equity_start,
            "book_equity": equity,
            "daily_price_pnl": price_pnl,
            "daily_borrow_cost": borrow_cost,
            "daily_margin_cost": margin_cost,
            "daily_txn_cost": txn_cost,
            "daily_net_pnl": net_pnl,
            "pnl_recon_residual": recon,
            "long_notional": long_notional,
            "short_notional": short_notional,
            "gross_notional": gross_notional,
            "net_notional": net_notional,
            "gross_leverage": gross_notional / equity if equity > 0 else np.nan,
            "net_exposure_pct": net_notional / equity if equity > 0 else np.nan,
            "turnover_usd": turnover_day,
            "n_positions": int(len(cur)),
            "largest_pair_gross_share": largest_pair_share,
            "top5_gross_share": top5_gross_share,
            "gross_hhi": gross_hhi,
            "n_stale_etf": stale_etf,
            "n_stale_underlying": stale_underlying,
            "is_rebalance": int(need_rebal),
            "active_plan_date": str(active_plan_date.date()) if active_plan_date is not None else None,
            "drawdown": equity / running_peak - 1.0 if running_peak > 0 else np.nan,
        }
        for s in ALL_SLEEVES:
            sc = sleeve_comp[s]
            row_out[s] = sc["price"] - sc["borrow"] - sc["margin"] - sc["txn"]
            row_out[f"{s}__price_pnl"] = sc["price"]
            row_out[f"{s}__borrow_cost"] = sc["borrow"]
            row_out[f"{s}__margin_cost"] = sc["margin"]
            row_out[f"{s}__txn_cost"] = sc["txn"]
        sleeve_daily_rows.append(row_out)

    audit = pd.DataFrame(audit_rows)
    pair_rows = []
    for e, c in contrib.items():
        gross_end = (
            abs(float(cur.at[e, "etf_usd"])) + abs(float(cur.at[e, "underlying_usd"]))
            if e in cur.index else 0.0
        )
        net_pair = (
            float(c["price_pnl_usd"]) - float(c["borrow_cost_usd"])
            - float(c["margin_cost_usd"]) - float(c["txn_cost_usd"])
        )
        pair_rows.append({**c, "pnl_usd": net_pair, "end_weight": gross_end / equity if equity > 0 else np.nan})
    pair_stats = pd.DataFrame(pair_rows)
    if not pair_stats.empty:
        pair_stats = pair_stats.sort_values("pnl_usd")
    sleeve_daily = pd.DataFrame(sleeve_daily_rows)
    if not sleeve_daily.empty:
        for s in ALL_SLEEVES:
            if s in sleeve_daily.columns:
                sleeve_daily[f"{s}_cum_pnl"] = sleeve_daily[s].cumsum()

    meta = {
        "n_rebal": n_rebal,
        "turnover_l1": turnover_l1,
        "turnover_usd": turnover_total,
        "cash_days": cash_days,
        "first_plan": str(first_plan.date()),
        "n_plans_used": len(plans_used),
        "start_usd": capital_usd,
        "end_usd": float(nav.iloc[-1]) if len(nav) else np.nan,
        "execution_lag_sessions": int(execution_lag_sessions),
        "target_notional_mode": target_notional_mode,
        "commission_per_share": float(commission_per_share),
        "margin_rate_annual": float(margin_rate_annual),
        "financing_daycount": float(financing_daycount),
        "use_resize_bands": bool(use_resize_bands),
        "enter_band_pct": float(enter_band_pct),
        "exit_band_pct": float(exit_band_pct),
        "min_trade_usd": float(min_trade_usd),
        **perf(nav),
    }
    return nav, audit, meta, pair_stats, sleeve_daily


# ---------------------------------------------------------------------------
# Phase B: recompute plans via mirror
# ---------------------------------------------------------------------------
def recompute_plan_timeline(
    *,
    cfg: dict,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    fallback_to_archived: bool = True,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    """Build plan timeline: mirror on screened dates, else archived proposed."""
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
                screened = pd.read_csv(path)
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
        execution_lag_sessions=rknobs["execution_lag_sessions"],
        target_notional_mode=rknobs["target_notional_mode"],
        enter_band_pct=rknobs["enter_band_pct"],
        exit_band_pct=rknobs["exit_band_pct"],
        min_trade_usd=rknobs["min_trade_usd"],
        check_freq="W-FRI",
        pre_archive_policy=pre_archive_policy,
    )

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
        "book": {k: book_meta.get(k) for k in ("cagr", "vol", "sharpe", "maxdd", "start_usd", "end_usd", "n_plans_used", "first_plan", "cash_days")},
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
        execution_lag_sessions=rknobs["execution_lag_sessions"],
        target_notional_mode=rknobs["target_notional_mode"],
        enter_band_pct=rknobs["enter_band_pct"],
        exit_band_pct=rknobs["exit_band_pct"],
        min_trade_usd=rknobs["min_trade_usd"],
        check_freq="W-FRI",
        pre_archive_policy=pre_archive_policy,
    )

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
        "book": {k: book_meta.get(k) for k in ("cagr", "vol", "sharpe", "maxdd", "start_usd", "end_usd", "n_plans_used", "first_plan", "cash_days")},
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
    mode: str = "frozen",
    pre_archive_policy: str = "cash",
) -> dict:
    mode = str(mode or "frozen").strip().lower()
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
    raise ValueError(f"Unknown mode={mode!r}; use frozen|replay|recompute")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-date", default="2026-07-08", help="Price-panel / frozen-anchor run date")
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument(
        "--mode",
        default="frozen",
        choices=("frozen", "replay", "recompute"),
        help="frozen=single plan; replay=archived plans; recompute=mirror+archived",
    )
    ap.add_argument(
        "--pre-archive-policy",
        default="cash",
        choices=("cash", "skip"),
        help="Before first archived plan: hold cash or start calendar at first plan",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output dir (default: notebooks/output/production_actual_bt[/_{mode}])",
    )
    args = ap.parse_args(argv)
    outdir = args.outdir
    if outdir is None:
        base = REPO / "notebooks" / "output" / "production_actual_bt"
        outdir = base if args.mode == "frozen" else base / args.mode
    run_production_actual_backtest(
        run_date=args.run_date,
        start=args.start,
        outdir=outdir,
        mode=args.mode,
        pre_archive_policy=args.pre_archive_policy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
