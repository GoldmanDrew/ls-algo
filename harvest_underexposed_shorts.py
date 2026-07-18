#!/usr/bin/env python3
"""Harvest under-exposed ETF shorts using position vs plan gaps.

By default, builds the discrepancy table from **live IBKR positions** (strategy shares × snapshot
marks) vs **proposed_trades.csv** — same merge rules as ``run_eod_pnl_email.load_position_discrepancies``
(Flex is not required). If IB is unreachable or live build fails, falls back to an existing
``position_discrepancies_all.csv`` (dated run or ``data/`` latest). Pass ``--discrepancy-csv`` to
force a specific file.

Two candidate lanes (both grow-only — harvest only ever SELLs the ETF, never covers):

* ``long_hedge`` — positive-delta ETFs (Bucket 1/2 levered/yieldboost). Short more ETF, then
  **BUY** the underlying ``|delta| × short_notional`` (delta-neutral long hedge).
* ``b4_pair`` — negative-delta inverse ETFs (Bucket 4 inverse-decay). Short more inverse ETF, then
  **SELL** the underlying ``r_live × short_notional`` where ``r_live`` preserves the pair's *current*
  underlying:inverse ratio (live actual; see :func:`resolve_b4_hedge_ratio`).

All underlying legs are then **netted by symbol** so a Bucket 1/2 buy and a Bucket 4 sell on the
same underlying collapse into a single order (:func:`net_underlying_orders`).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execute_trade_plan import (
    CoordinatorCancelService,
    configure_ib_error_log_filter,
    connect_ib,
    current_ib_positions,
    execute_leg,
    fetch_ibkr_short_availability_map,
    get_snapshot_price,
    is_short_unavailable_now,
    load_baseline_qty,
    norm_sym,
    run_dir,
    stop_requested,
    strategy_position_only,
    tprint,
    today_str,
)
from generate_trade_plan import load_blacklist
from trade_plan_targets import (
    _maybe_merge_optimal_targets,
    _resolve_target_basis_columns,
)
from ibkr_accounting import (
    EXCLUDE_SYMBOLS,
    SUPPLEMENTAL_ETF_MAP,
    canonical_symbol,
    load_etf_delta_map,
    load_etf_to_under_map,
    load_universe_from_screened,
    normalize_plan_etf_ticker,
)
from strategy_config import load_config

# Sleeves whose ETF leg is a structural short with a negative-delta (inverse) exposure.
# Names in these sleeves are grown by SELLING the underlying (not buying) to keep the
# pair's hedge ratio; the ratchet keeps the inverse ETF leg grow-only.
B4_INVERSE_SLEEVES: frozenset[str] = frozenset({"inverse_decay_bucket4"})
# r_live clip: the maintained underlying:inverse ratio can never exceed |beta| * this
# multiplier (guards a drifted/degenerate live position from propagating onto the add).
_R_LIVE_CLIP_MULT = 1.5


def harvest_execution_dir(run_date: str, cfg: dict | None = None) -> Path:
    """Absolute ``data/runs/<date>/execution``; creates parents if missing."""
    outdir = (_repo_root(cfg) / "data" / "runs" / run_date / "execution").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _repo_root(cfg: dict | None = None) -> Path:
    if cfg and cfg.get("_repo_root"):
        return Path(str(cfg["_repo_root"]))
    return Path(__file__).resolve().parent


def write_harvest_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a harvest artifact CSV, ensuring the parent directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def resolve_proposed_trades_path(run_date: str, paths_cfg: dict) -> Path:
    dated = run_dir(run_date) / "proposed_trades.csv"
    if dated.exists():
        return dated
    p = paths_cfg.get("proposed_trades_csv")
    if p:
        pp = Path(str(p))
        if pp.exists():
            return pp
    fb = Path("data") / "proposed_trades.csv"
    if fb.exists():
        return fb
    raise FileNotFoundError(
        "Could not locate proposed_trades.csv (try dated data/runs/<run-date>/proposed_trades.csv "
        "or paths.proposed_trades_csv in config)."
    )


def load_plan_pair_attrs(proposed_path: Path) -> tuple[dict[str, str], dict[str, float]]:
    """Return ``(etf_to_sleeve, etf_to_plan_ratio)`` keyed by canonical ETF symbol.

    ``plan_ratio`` = ``|long_usd / short_usd|`` per plan row (prefers ``optimal_*`` columns),
    i.e. the plan's intended underlying-notional-per-inverse-notional. NaN/inf where the
    short leg is zero; callers fall back to ``|beta|`` (delta-neutral) in that case.
    """
    try:
        df = pd.read_csv(proposed_path)
    except Exception:
        return {}, {}
    if df.empty or "ETF" not in df.columns:
        return {}, {}
    etf = df["ETF"].astype(str).map(canonical_symbol).map(normalize_plan_etf_ticker)
    long_col = "optimal_long_usd" if "optimal_long_usd" in df.columns else "long_usd"
    short_col = "optimal_short_usd" if "optimal_short_usd" in df.columns else "short_usd"
    longs = pd.to_numeric(df.get(long_col, 0.0), errors="coerce").fillna(0.0).abs()
    shorts = pd.to_numeric(df.get(short_col, 0.0), errors="coerce").fillna(0.0).abs()
    ratio = np.where(shorts > 1e-9, longs / shorts, np.nan)

    etf_to_sleeve: dict[str, str] = {}
    etf_to_plan_ratio: dict[str, float] = {}
    sleeves = df.get("sleeve")
    for i, e in enumerate(etf):
        if not e:
            continue
        if sleeves is not None:
            sv = str(sleeves.iloc[i]).strip().lower()
            if sv and sv != "nan":
                etf_to_sleeve.setdefault(e, sv)
        r = float(ratio[i])
        if np.isfinite(r) and r > 0 and e not in etf_to_plan_ratio:
            etf_to_plan_ratio[e] = r
    return etf_to_sleeve, etf_to_plan_ratio


def _resolve_latest_b4_detail_csv(run_date: str, cfg: dict | None = None) -> Path | None:
    runs = _repo_root(cfg) / "data" / "runs"
    pattern = str(runs / "*" / "accounting" / "net_exposure_bucket_4_detail.csv")
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
    dated: list[tuple[str, str]] = []
    for p in glob.glob(pattern):
        m = date_re.search(Path(p).as_posix())
        if m:
            dated.append((m.group(1), p))
    if not dated:
        return None
    on_or_before = [d for d in dated if d[0] <= run_date]
    chosen = max(on_or_before, key=lambda d: d[0]) if on_or_before else max(dated, key=lambda d: d[0])
    return Path(chosen[1])


def load_latest_b4_detail_maps(
    run_date: str, cfg: dict | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Load B4 exposure detail for live hedge-ratio resolution.

    Returns ``(etf_pair_ratio, underlying_notional)``:

    * ``etf_pair_ratio[etf]`` = ``|underlying_leg| / |etf_leg|`` when both legs exist in the
      detail file for that pair (preferred per-pair live ratio).
    * ``underlying_notional[underlying]`` = ``|B4 structural-short underlying notional|`` summed
      per underlying (fallback when the ETF leg is missing from detail).
    """
    path = _resolve_latest_b4_detail_csv(run_date, cfg)
    if path is None or not path.exists():
        return {}, {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, {}
    if df.empty or "leg_type" not in df.columns or "underlying" not in df.columns:
        return {}, {}

    df = df.copy()
    df["underlying"] = df["underlying"].astype(str).map(canonical_symbol)
    df["net_notional_usd"] = pd.to_numeric(df.get("net_notional_usd", 0.0), errors="coerce").fillna(0.0)
    leg = df["leg_type"].astype(str).str.strip().str.lower()

    und_legs = df[leg == "underlying"].copy()
    underlying_notional: dict[str, float] = {}
    if not und_legs.empty:
        grp = und_legs.groupby("underlying")["net_notional_usd"].sum().abs()
        underlying_notional = {k: float(v) for k, v in grp.items() if float(v) > 0.0}

    etf_pair_ratio: dict[str, float] = {}
    if "symbol" in df.columns:
        etf_legs = df[leg == "etf"].copy()
        etf_legs["symbol"] = etf_legs["symbol"].astype(str).map(canonical_symbol).map(normalize_plan_etf_ticker)
        und_by = und_legs.set_index("underlying")["net_notional_usd"].abs() if not und_legs.empty else pd.Series(dtype=float)
        for _, er in etf_legs.iterrows():
            etf_sym = str(er["symbol"])
            und_sym = str(er["underlying"])
            etf_n = abs(float(er["net_notional_usd"]))
            und_n = float(und_by.get(und_sym, 0.0))
            if etf_n > 1.0 and und_n > 1.0:
                etf_pair_ratio[etf_sym] = und_n / etf_n

    return etf_pair_ratio, underlying_notional


def load_latest_b4_detail_map(run_date: str, cfg: dict | None = None) -> dict[str, float]:
    """Return ``{underlying: |B4 structural-short underlying notional|}`` (legacy helper)."""
    _, underlying_notional = load_latest_b4_detail_maps(run_date, cfg)
    return underlying_notional


def resolve_b4_hedge_ratio(
    *,
    underlying: str,
    inverse_raw_notional: float,
    beta: float,
    etf: str | None = None,
    b4_detail_etf_ratio: dict[str, float] | None = None,
    b4_detail_map: dict[str, float] | None = None,
    live_net_under_notional: float = 0.0,
    plan_ratio: float | None = None,
    mode: str = "live",
    clip_mult: float = _R_LIVE_CLIP_MULT,
    eps: float = 1.0,
) -> tuple[float, str]:
    """Underlying-notional-per-inverse-notional ratio to preserve when growing a B4 short.

    Returns ``(r, source)``. ``mode``:
      * ``live`` (default): current actual ratio, resolved in order
        ``b4_detail`` (isolates B4 on shared names) -> ``live_net`` (net-short underlyings)
        -> ``plan_ratio`` -> ``delta_neutral`` (``|beta|``).
      * ``plan_ratio``: use the plan's ``|long/short|`` (fallback ``|beta|``).
      * ``delta_neutral``: always ``|beta|`` (fully cancel the added inverse delta).

    ``r`` is clipped to ``[0, |beta| * clip_mult]`` so a drifted live position cannot
    propagate an extreme ratio onto the increment.
    """
    beta = abs(float(beta))
    hi = beta * float(clip_mult) if beta > 0 else float("inf")

    def _clip(r: float) -> float:
        r = max(0.0, float(r))
        return min(r, hi) if math.isfinite(hi) else r

    def _plan_or_beta() -> tuple[float, str]:
        if plan_ratio is not None and np.isfinite(plan_ratio) and plan_ratio > 0:
            return _clip(plan_ratio), "plan_ratio"
        return _clip(beta), "delta_neutral"

    mode = str(mode or "live").strip().lower()
    if mode == "delta_neutral":
        return _clip(beta), "delta_neutral"
    if mode == "plan_ratio":
        return _plan_or_beta()

    # mode == "live"
    inv = abs(float(inverse_raw_notional or 0.0))
    if inv < eps:
        r, src = _plan_or_beta()
        return r, (src + "_no_inverse" if src == "plan_ratio" else "delta_neutral_no_inverse")

    etf_key = canonical_symbol(str(etf or "")).strip()
    if etf_key:
        etf_ratio = (b4_detail_etf_ratio or {}).get(etf_key)
        if etf_ratio is not None and np.isfinite(etf_ratio) and etf_ratio > 0:
            return _clip(float(etf_ratio)), "b4_detail_pair"

    b4u = (b4_detail_map or {}).get(underlying)
    if b4u is not None and abs(float(b4u)) > eps:
        return _clip(abs(float(b4u)) / inv), "b4_detail"

    lnu = float(live_net_under_notional or 0.0)
    if lnu < -eps:  # underlying is net short -> live net is a valid B4 short estimate
        return _clip(abs(lnu) / inv), "live_net"

    return _plan_or_beta()


def compute_underlying_delta_usd(
    *,
    lane: str,
    filled_inverse_notional: float,
    delta: float,
    r_live: float,
    buffer_pct: float,
) -> float:
    """Signed underlying notional to trade to hedge an ETF short add.

    Positive = BUY underlying (long_hedge / positive-delta ETF).
    Negative = SELL underlying (b4_pair / inverse ETF), sized by the maintained ``r_live``.
    """
    scale = max(0.0, 1.0 - float(buffer_pct))
    notional = abs(float(filled_inverse_notional))
    if str(lane) == "b4_pair":
        return -(notional * float(r_live)) * scale
    return +(notional * abs(float(delta))) * scale


def build_harvest_candidates(
    disc: pd.DataFrame,
    *,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    etf_to_sleeve: dict[str, str],
    plan_ratio_map: dict[str, float],
    blocked_symbols: set[str],
    b4_sleeves: frozenset[str] = B4_INVERSE_SLEEVES,
    top_n: int = 0,
) -> pd.DataFrame:
    """Under-exposed ETF candidates for both lanes (positive-delta long-hedge + negative-delta B4).

    Lane is driven by the ETF's delta sign (inverse ETFs carry the underlying's short); the plan
    sleeve/ratio are attached for the B4 hedge-ratio resolver and telemetry. Rows with delta == 0
    or an unmapped underlying are dropped.
    """
    if disc is None or disc.empty:
        return disc.iloc[0:0].copy() if disc is not None else pd.DataFrame()

    cands = disc[disc["under_exposed"]].copy()
    cands = cands[~cands["symbol"].isin(blocked_symbols)].copy()
    cands["underlying"] = cands["symbol"].map(etf_to_under)
    cands["delta"] = pd.to_numeric(cands["symbol"].map(etf_to_delta), errors="coerce")
    cands = cands[cands["underlying"].astype(str).str.len() > 0].copy()
    cands = cands[cands["delta"].notna() & (cands["delta"].abs() > 1e-9)].copy()

    cands["beta"] = cands["delta"].abs()
    cands["sleeve"] = cands["symbol"].map(etf_to_sleeve).fillna("")
    cands["plan_ratio"] = cands["symbol"].map(plan_ratio_map)
    cands["lane"] = np.where(cands["delta"] < 0, "b4_pair", "long_hedge")
    cands["under_hedge_dir"] = np.where(cands["delta"] < 0, "SELL", "BUY")

    cands = cands.sort_values("abs_discrepancy_usd", ascending=False).reset_index(drop=True)
    if top_n and top_n > 0:
        cands = cands.head(int(top_n)).copy()
    return cands


def net_underlying_orders(
    pair_delta_rows: list[dict[str, Any]],
    prices: dict[str, float],
    min_trade_usd: float,
) -> list[dict[str, Any]]:
    """Net signed underlying deltas by symbol into one order each.

    Each input row needs ``underlying``, ``under_delta_usd`` (signed; + BUY / - SELL) and
    optionally ``etf``. Output rows carry ``net_usd``, ``gross_buy_usd``, ``gross_sell_usd``,
    ``action`` (BUY/SELL/None), ``qty``, ``px_under``, ``decision`` and contributing ``etfs``.
    A near-total buy/sell cancellation collapses to ``decision == "skip"``.
    """
    agg: dict[str, dict[str, Any]] = {}
    for r in pair_delta_rows:
        u = norm_sym(str(r["underlying"]))
        d = float(r.get("under_delta_usd", 0.0) or 0.0)
        a = agg.setdefault(
            u,
            {"underlying": u, "net_usd": 0.0, "gross_buy_usd": 0.0, "gross_sell_usd": 0.0, "_etfs": []},
        )
        a["net_usd"] += d
        if d >= 0:
            a["gross_buy_usd"] += d
        else:
            a["gross_sell_usd"] += -d
        if r.get("etf"):
            a["_etfs"].append(norm_sym(str(r["etf"])))

    orders: list[dict[str, Any]] = []
    for u, a in agg.items():
        px = float(prices.get(u) or 0.0)
        net = float(a["net_usd"])
        rec: dict[str, Any] = {
            "underlying": u,
            "net_usd": net,
            "gross_buy_usd": float(a["gross_buy_usd"]),
            "gross_sell_usd": float(a["gross_sell_usd"]),
            "px_under": px,
            "etfs": ",".join(sorted({e for e in a["_etfs"] if e})),
        }
        if px <= 0:
            rec.update(action=None, qty=0, decision="skip", reason="no_price_for_underlying")
            orders.append(rec)
            continue
        qty = int(math.floor(abs(net) / px))
        if abs(net) < float(min_trade_usd) or qty <= 0:
            rec.update(action=None, qty=0, decision="skip", reason="below_min_trade_or_netted_out")
            orders.append(rec)
            continue
        rec.update(action=("BUY" if net > 0 else "SELL"), qty=qty, decision="trade", reason="")
        orders.append(rec)
    return orders


def _position_discrepancy_merge(
    plan: pd.DataFrame,
    actual: pd.DataFrame,
    screened_csv: Path,
    cfg: dict,
    *,
    target_basis: str = "optimal",
) -> pd.DataFrame:
    """Match ``run_eod_pnl_email.load_position_discrepancies`` merge (ETF-universe filter on symbols).

    ``target_basis`` selects the planned-notional columns:
      - ``optimal`` (default): use ``optimal_long_usd`` / ``optimal_short_usd`` so harvest fills
        the **standing structural gap** as soon as borrow comes back, even if today's executable
        was clipped by ``shares_available``.
      - ``executable``: legacy behavior, only fills gaps vs today's executable target.
      - ``max``: per-symbol max(|optimal|, |executable|), signed by sleeve direction.
    """
    strategy_tag = str((cfg.get("strategy", {}) or {}).get("tag", "")).strip()
    if strategy_tag and plan is not None and not plan.empty and "strategy_tag" in plan.columns:
        plan = plan[plan["strategy_tag"].astype(str) == strategy_tag].copy()

    if plan is None or plan.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "target_net_usd",
                "actual_net_usd",
                "discrepancy_usd",
                "abs_discrepancy_usd",
                "target_gross_usd",
                "actual_gross_usd",
                "gross_gap_usd",
                "under_exposed",
            ]
        )

    for c in ("ETF", "Underlying"):
        if c not in plan.columns:
            raise ValueError(f"proposed_trades.csv missing required column: {c}")
    plan = plan.copy()
    plan["ETF"] = plan["ETF"].astype(str).map(canonical_symbol).map(normalize_plan_etf_ticker)
    plan["Underlying"] = plan["Underlying"].astype(str).map(canonical_symbol)
    plan["long_usd"] = pd.to_numeric(plan.get("long_usd", 0.0), errors="coerce").fillna(0.0)
    plan["short_usd"] = pd.to_numeric(plan.get("short_usd", 0.0), errors="coerce").fillna(0.0)
    if "optimal_long_usd" in plan.columns:
        plan["optimal_long_usd"] = pd.to_numeric(plan["optimal_long_usd"], errors="coerce").fillna(0.0)
    if "optimal_short_usd" in plan.columns:
        plan["optimal_short_usd"] = pd.to_numeric(plan["optimal_short_usd"], errors="coerce").fillna(0.0)

    long_col, short_col = _resolve_target_basis_columns(plan, target_basis)

    if str(target_basis or "").strip().lower() == "max":
        # max basis: row-wise max magnitude across optimal/executable, sign preserved by source.
        # Long target: max of two non-negative columns.
        long_a = pd.to_numeric(plan.get("long_usd", 0.0), errors="coerce").fillna(0.0)
        long_b = pd.to_numeric(plan.get("optimal_long_usd", long_a), errors="coerce").fillna(long_a)
        plan["_harvest_long"] = np.where(long_a.abs() >= long_b.abs(), long_a, long_b)
        # Short target: max magnitude of two non-positive columns; pick the more-negative.
        short_a = pd.to_numeric(plan.get("short_usd", 0.0), errors="coerce").fillna(0.0)
        short_b = pd.to_numeric(plan.get("optimal_short_usd", short_a), errors="coerce").fillna(short_a)
        plan["_harvest_short"] = np.where(short_a.abs() >= short_b.abs(), short_a, short_b)
        long_col, short_col = "_harvest_long", "_harvest_short"

    target_under = (
        plan.groupby("Underlying", as_index=False)[long_col]
        .sum()
        .rename(columns={"Underlying": "symbol", long_col: "target_net_usd"})
    )
    target_etf = (
        plan.groupby("ETF", as_index=False)[short_col]
        .sum()
        .rename(columns={"ETF": "symbol", short_col: "target_net_usd"})
    )
    target = pd.concat([target_under, target_etf], ignore_index=True)
    target = target[target["symbol"].astype(bool)].copy()
    target = target.groupby("symbol", as_index=False)["target_net_usd"].sum()

    if actual is None or actual.empty:
        actual_df = pd.DataFrame(columns=["symbol", "actual_net_usd"])
    else:
        actual_df = actual.copy()
        actual_df["symbol"] = actual_df["symbol"].astype(str).map(canonical_symbol)
        actual_df["actual_net_usd"] = pd.to_numeric(actual_df["actual_net_usd"], errors="coerce").fillna(0.0)
        actual_df = actual_df.groupby("symbol", as_index=False)["actual_net_usd"].sum()

    allowed_etfs, _ = load_universe_from_screened(screened_csv)
    allowed_etfs |= set(SUPPLEMENTAL_ETF_MAP.keys())

    etf_to_under = load_etf_to_under_map(screened_csv)
    for e_sym, u_sym in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(e_sym, u_sym)

    blacklist_raw = ((cfg.get("strategy", {}) or {}).get("blacklist", [])) or []
    blacklist = {canonical_symbol(str(s)) for s in blacklist_raw if str(s).strip()}
    blocked_etfs = {s for s in blacklist if s in allowed_etfs}
    blocked_etfs |= {e for e, u in etf_to_under.items() if u in blacklist}

    merged = target.merge(actual_df, on="symbol", how="outer")
    merged["target_net_usd"] = pd.to_numeric(merged["target_net_usd"], errors="coerce").fillna(0.0)
    merged["actual_net_usd"] = pd.to_numeric(merged["actual_net_usd"], errors="coerce").fillna(0.0)
    merged["symbol"] = merged["symbol"].astype(str).map(canonical_symbol)
    merged = merged[merged["symbol"].isin(allowed_etfs)].copy()
    if blocked_etfs:
        merged = merged[~merged["symbol"].isin(blocked_etfs)].copy()

    merged["discrepancy_usd"] = merged["actual_net_usd"] - merged["target_net_usd"]
    merged["abs_discrepancy_usd"] = merged["discrepancy_usd"].abs()
    merged["target_gross_usd"] = merged["target_net_usd"].abs()
    merged["actual_gross_usd"] = merged["actual_net_usd"].abs()
    merged["gross_gap_usd"] = merged["actual_gross_usd"] - merged["target_gross_usd"]
    merged["under_exposed"] = merged["gross_gap_usd"] < -1e-9
    return merged.sort_values("abs_discrepancy_usd", ascending=False).reset_index(drop=True)


def build_discrepancy_from_live_ib(
    ib: Any,
    *,
    run_date: str,
    cfg: dict,
    screened_csv: Path,
    baseline_csv: Path,
    prefer_delayed: bool,
    paths_cfg: dict,
    target_basis: str = "optimal",
) -> pd.DataFrame:
    """Actual notionals from strategy-only IB positions × snapshot marks vs proposed plan.

    ``target_basis`` selects what counts as "the plan": ``optimal`` reads the structural-only
    target so the harvester fills the **standing** shortfall when borrow returns;
    ``executable`` reads only today's clipped target; ``max`` takes the row-wise max."""
    proposed_path = resolve_proposed_trades_path(run_date, paths_cfg)
    plan = pd.read_csv(proposed_path)
    plan = _maybe_merge_optimal_targets(plan, run_date)

    baseline = load_baseline_qty(baseline_csv)
    ib_pos = current_ib_positions(ib)
    strat = strategy_position_only(ib_pos, baseline)

    rows: list[dict[str, Any]] = []
    for sym_raw, sh in strat.items():
        sh_f = float(sh)
        if abs(sh_f) < 1e-9:
            continue
        sym = canonical_symbol(norm_sym(str(sym_raw)))
        if sym in EXCLUDE_SYMBOLS:
            continue
        try:
            px = float(get_snapshot_price(ib, sym_raw, prefer_delayed=prefer_delayed))
        except Exception:
            px = float("nan")
        if not (np.isfinite(px) and px > 0):
            tprint(f"[HARVEST] LIVE_DISC: skip {sym} (no snapshot price; shares={sh_f:g})")
            continue
        rows.append({"symbol": sym, "actual_net_usd": sh_f * px})

    actual_df = pd.DataFrame(rows)
    return _position_discrepancy_merge(plan, actual_df, screened_csv, cfg, target_basis=target_basis)


def resolve_discrepancy_csv(run_date: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Discrepancy CSV not found at --discrepancy-csv path: {p}")
    dated = run_dir(run_date) / "accounting" / "position_discrepancies_all.csv"
    if dated.exists():
        return dated
    latest = Path("data") / "position_discrepancies_all.csv"
    if latest.exists():
        return latest
    raise FileNotFoundError(
        "Could not find discrepancy CSV. Expected one of:\n"
        f"- {dated}\n"
        f"- {latest}"
    )


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Harvest under-exposed ETF shorts; hedge underlying, netted by symbol."
    )
    ap.add_argument("--run-date", default=None, help="YYYY-MM-DD (defaults to RUN_DATE or today).")
    ap.add_argument("--discrepancy-csv", default=None, help="Optional explicit discrepancy CSV path.")
    ap.add_argument("--top-n", type=int, default=30, help="Process top N under-exposed ETFs by |discrepancy|.")
    ap.add_argument(
        "--underhedge-buffer-pct",
        type=float,
        default=0.0025,
        help="Under-hedge buffer fraction (0.25%% default).",
    )
    ap.add_argument(
        "--max-short-usd-per-etf",
        type=float,
        default=0.0,
        help="Optional cap per ETF short notional. 0 means no cap.",
    )
    ap.add_argument(
        "--b4-hedge-ratio-source",
        choices=("live", "plan_ratio", "delta_neutral"),
        default="live",
        help=(
            "How the Bucket-4 underlying leg sizes its SELL. 'live' (default) preserves the "
            "pair's current actual underlying:inverse ratio (B4 detail -> live net -> plan -> "
            "delta-neutral). 'plan_ratio' uses the plan |long/short|. 'delta_neutral' always "
            "sells |beta| x added short."
        ),
    )
    ap.add_argument(
        "--no-net-underlyings",
        action="store_true",
        help="Disable netting of underlying legs by symbol (execute each pair's underlying leg separately).",
    )
    ap.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip confirmation prompt before live order placement.",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Deprecated flag (live is now default).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Force dry-run mode.")
    ap.add_argument(
        "--file-discrepancies-only",
        action="store_true",
        help="Skip live IB discrepancy build; use position_discrepancies_all.csv (dated or data/).",
    )
    ap.add_argument(
        "--target-basis",
        choices=("optimal", "executable", "max"),
        default="optimal",
        help=(
            "Which target columns drive harvest sizing. 'optimal' (default) uses the "
            "structural-only target so harvest fills the standing gap when borrow returns. "
            "'executable' uses today's clipped target (legacy). 'max' picks the larger "
            "magnitude per row."
        ),
    )
    args = ap.parse_args()

    run_date = args.run_date or os.environ.get("RUN_DATE") or today_str()
    dry_run = bool(args.dry_run)
    buffer_pct = float(args.underhedge_buffer_pct)
    ratio_mode = str(args.b4_hedge_ratio_source)
    net_underlyings = not bool(args.no_net_underlyings)

    cfg = load_config("config/strategy_config.yml")
    ibkr_cfg = cfg.get("ibkr", {}) or {}
    strat_cfg = cfg.get("strategy", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}
    reb_cfg = ((cfg.get("portfolio", {}) or {}).get("rebalance", {}) or {})

    strategy_tag = str(strat_cfg.get("tag", "")).strip()
    if not strategy_tag:
        raise ValueError("Missing strategy.tag in config.")

    host = str(ibkr_cfg.get("host", "127.0.0.1"))
    port = int(ibkr_cfg.get("port", 7496))
    client_id = int(ibkr_cfg.get("client_id", 41))
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))
    suppress_error_codes = [int(c) for c in ((ibkr_cfg.get("suppress_error_codes", [10089])) or [])]
    configure_ib_error_log_filter(suppress_error_codes)

    min_trade_usd = float(reb_cfg.get("min_trade_usd", 200.0))
    limit_bps = float(exec_cfg.get("limit_bps", 25.0))
    timeout_sec = float(exec_cfg.get("timeout_sec", 90))
    max_retries = int(exec_cfg.get("max_retries", 3))
    max_short_usd_per_etf = float(args.max_short_usd_per_etf or 0.0)

    screened_csv = Path(paths_cfg.get("screened_csv", "data/etf_screened_today.csv"))
    baseline_csv = Path(paths_cfg.get("baseline_csv", "data/baseline_snapshot.csv"))
    if not screened_csv.exists():
        raise FileNotFoundError(f"Screened CSV not found: {screened_csv}")

    outdir = harvest_execution_dir(run_date, cfg)
    candidates_path = outdir / "harvest_candidates.csv"
    attempted_path = outdir / "harvest_attempted_trades.csv"
    fills_path = outdir / "harvest_fills.csv"
    summary_path = outdir / "harvest_post_trade_summary.csv"
    netting_path = outdir / "harvest_underlying_netting.csv"
    disc_source_path = outdir / "harvest_discrepancy_source.txt"
    disc_input_path = outdir / "harvest_discrepancies_input.csv"

    def _write_empty_outputs() -> None:
        write_harvest_csv(attempted_path, pd.DataFrame())
        write_harvest_csv(fills_path, pd.DataFrame())
        write_harvest_csv(summary_path, pd.DataFrame())
        write_harvest_csv(netting_path, pd.DataFrame())

    disc: pd.DataFrame
    disc_source = ""
    if args.discrepancy_csv:
        discrepancy_csv = resolve_discrepancy_csv(run_date, args.discrepancy_csv)
        disc = pd.read_csv(discrepancy_csv)
        disc_source = f"explicit_file:{discrepancy_csv}"
    elif args.file_discrepancies_only:
        discrepancy_csv = resolve_discrepancy_csv(run_date, None)
        disc = pd.read_csv(discrepancy_csv)
        disc_source = f"file_only:{discrepancy_csv}"
    else:
        try:
            ib_pre = connect_ib(host, port, client_id + 510, coordinator=False)
            try:
                disc = build_discrepancy_from_live_ib(
                    ib_pre,
                    run_date=run_date,
                    cfg=cfg,
                    screened_csv=screened_csv,
                    baseline_csv=baseline_csv,
                    prefer_delayed=prefer_delayed,
                    paths_cfg=paths_cfg,
                    target_basis=args.target_basis,
                )
                disc_source = f"live_ib(target_basis={args.target_basis})"
                tprint(f"[HARVEST] target_basis={args.target_basis} (gap measured vs '{args.target_basis}' target)")
            finally:
                try:
                    ib_pre.disconnect()
                except Exception:
                    pass
        except Exception as ex:
            tprint(
                f"[HARVEST] Live discrepancy build failed ({ex}); "
                f"falling back to position_discrepancies_all.csv."
            )
            discrepancy_csv = resolve_discrepancy_csv(run_date, None)
            disc = pd.read_csv(discrepancy_csv)
            disc_source = f"file_fallback:{discrepancy_csv}"

    disc_source_path.parent.mkdir(parents=True, exist_ok=True)
    disc_source_path.write_text(disc_source.strip() + "\n", encoding="utf-8")
    if disc.empty:
        tprint(f"[HARVEST] Discrepancy table has no rows (source={disc_source})")
        write_harvest_csv(candidates_path, pd.DataFrame())
        _write_empty_outputs()
        return 0

    for col in ("symbol", "abs_discrepancy_usd", "gross_gap_usd", "under_exposed"):
        if col not in disc.columns:
            raise ValueError(f"Discrepancy table missing required column: {col}")

    disc["symbol"] = disc["symbol"].astype(str).map(canonical_symbol)
    disc["abs_discrepancy_usd"] = pd.to_numeric(disc["abs_discrepancy_usd"], errors="coerce").fillna(0.0)
    disc["gross_gap_usd"] = pd.to_numeric(disc["gross_gap_usd"], errors="coerce").fillna(0.0)
    disc["under_exposed"] = to_bool_series(disc["under_exposed"])
    write_harvest_csv(disc_input_path, disc)
    tprint(f"[HARVEST] Discrepancy input ({disc_source}) -> {disc_input_path}")

    _etf_to_under_raw, _etf_to_delta_raw = load_etf_delta_map(screened_csv)
    etf_to_under = {canonical_symbol(str(k)): canonical_symbol(str(v)) for k, v in _etf_to_under_raw.items()}
    etf_to_delta = {canonical_symbol(str(k)): float(v) for k, v in _etf_to_delta_raw.items()}

    proposed_path = resolve_proposed_trades_path(run_date, paths_cfg)
    etf_to_sleeve, plan_ratio_map = load_plan_pair_attrs(proposed_path)
    b4_detail_etf_ratio, b4_detail_map = load_latest_b4_detail_maps(run_date, cfg)
    tprint(
        f"[HARVEST] plan pair attrs: {len(etf_to_sleeve)} sleeves, {len(plan_ratio_map)} ratios; "
        f"B4 detail pairs: {len(b4_detail_etf_ratio)} etf ratios, {len(b4_detail_map)} underlyings "
        f"(ratio mode={ratio_mode}, net={net_underlyings})"
    )

    blacklist = {canonical_symbol(str(s)) for s in load_blacklist(cfg)}
    blocked_by_under = {etf for etf, und in etf_to_under.items() if und in blacklist}
    purgatory_symbols: set[str] = set()
    try:
        screened_policy = pd.read_csv(screened_csv)
        if "ETF" in screened_policy.columns:
            policy_block = pd.Series(False, index=screened_policy.index)
            for col in ("purgatory", "hard_exit_borrow"):
                if col in screened_policy.columns:
                    policy_block = policy_block | to_bool_series(screened_policy[col])
            purgatory_symbols |= set(
                screened_policy.loc[policy_block, "ETF"].astype(str).map(canonical_symbol)
            )
        plan_policy = pd.read_csv(proposed_path)
        if "ETF" in plan_policy.columns:
            plan_block = plan_policy.get(
                "execution_policy", pd.Series("", index=plan_policy.index)
            ).astype(str).str.lower().isin({"reduce_only", "hard_exit", "hold"})
            purgatory_symbols |= set(
                plan_policy.loc[plan_block, "ETF"].astype(str).map(canonical_symbol)
            )
    except Exception as ex:
        tprint(f"[HARVEST] WARNING: purgatory policy load failed ({ex}); using plan blockers only.")
    blocked_symbols = blacklist | blocked_by_under | purgatory_symbols

    cands = build_harvest_candidates(
        disc,
        etf_to_under=etf_to_under,
        etf_to_delta=etf_to_delta,
        etf_to_sleeve=etf_to_sleeve,
        plan_ratio_map=plan_ratio_map,
        blocked_symbols=blocked_symbols,
        top_n=int(args.top_n),
    )
    write_harvest_csv(candidates_path, cands)

    if cands.empty:
        tprint("[HARVEST] No valid under-exposed ETF candidates after filters.")
        _write_empty_outputs()
        return 0

    n_b4 = int((cands["lane"] == "b4_pair").sum())
    n_long = int((cands["lane"] == "long_hedge").sum())
    tprint(f"[HARVEST] candidates={len(cands)} (long_hedge={n_long}, b4_pair={n_b4})")

    # Short-availability for candidate ETFs (short sale) + B4-lane underlyings (netted SELL may
    # flip the name net-short and needs a locate).
    short_check_syms = list(cands["symbol"].tolist())
    short_check_syms += cands.loc[cands["lane"] == "b4_pair", "underlying"].tolist()
    short_map: dict[str, dict[str, Any]]
    try:
        short_map = fetch_ibkr_short_availability_map(sorted({str(s) for s in short_check_syms if s}))
    except Exception as ex:
        tprint(f"[HARVEST] WARNING: FTP short availability fetch failed ({ex}); continuing uncapped.")
        short_map = {}

    baseline = load_baseline_qty(baseline_csv)
    tprint(
        f"[HARVEST] run_date={run_date} candidates={len(cands)} dry_run={dry_run} "
        f"underhedge_buffer={buffer_pct:.4f}"
    )

    attempted_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    netting_rows: list[dict[str, Any]] = []

    try:
        ib = connect_ib(host, port, client_id + 510, coordinator=True)
    except Exception as ex:
        tprint(f"[HARVEST] ERROR: IB connection failed: {ex}")
        write_harvest_csv(attempted_path, pd.DataFrame(attempted_rows))
        write_harvest_csv(fills_path, pd.DataFrame(fill_rows))
        write_harvest_csv(
            summary_path,
            pd.DataFrame([{"status": "IB_CONNECTION_FAILED", "error": str(ex), "run_date": run_date}]),
        )
        write_harvest_csv(netting_path, pd.DataFrame())
        return 2

    cancel_service = CoordinatorCancelService(host=host, port=port)
    cancel_service.start()
    try:
        ib_pos = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)
        tprint(f"[HARVEST] Strategy-only symbols currently held: {len(strat_pos)}")

        # ---- Plan phase: size each ETF short leg + resolve the B4 hedge ratio -----------
        prices: dict[str, float] = {}

        def _px(sym: str) -> float:
            s = norm_sym(str(sym))
            if s in prices and prices[s] > 0:
                return prices[s]
            try:
                p = float(get_snapshot_price(ib, s, prefer_delayed=prefer_delayed))
            except Exception:
                p = 0.0
            prices[s] = p
            return p

        planned_rows: list[dict[str, Any]] = []
        for _, row in cands.iterrows():
            if stop_requested():
                tprint("[HARVEST] Stop requested; halting candidate planning.")
                break

            etf = norm_sym(str(row["symbol"]))
            under = norm_sym(str(row["underlying"]))
            delta = float(row["delta"])
            beta = abs(delta)
            lane = str(row["lane"])
            plan_ratio = row.get("plan_ratio")
            plan_ratio = float(plan_ratio) if pd.notna(plan_ratio) else None
            need_usd = max(0.0, -float(row.get("gross_gap_usd", 0.0) or 0.0))
            if max_short_usd_per_etf > 0:
                need_usd = min(need_usd, max_short_usd_per_etf)

            base_summary = {
                "symbol": etf,
                "underlying": under,
                "lane": lane,
                "requested_short_usd": need_usd,
                "requested_short_sh": 0,
                "filled_short_sh": 0,
                "r_live": None,
                "hedge_ratio_source": None,
                "intended_under_delta_usd": 0.0,
                "remaining_short_usd": need_usd,
            }

            if need_usd < min_trade_usd:
                summary_rows.append({**base_summary, "status": "SKIP_BELOW_MIN_TRADE_USD"})
                continue

            px_etf = _px(etf)
            px_under = _px(under)
            if not (px_etf > 0 and px_under > 0):
                summary_rows.append({**base_summary, "status": "SKIP_NO_PRICE"})
                continue

            requested_short_sh = int(math.floor(need_usd / px_etf)) if px_etf > 0 else 0
            if requested_short_sh <= 0:
                summary_rows.append({**base_summary, "status": "SKIP_ZERO_SHARES_AFTER_ROUNDING"})
                continue

            avail = (short_map.get(etf) or {}).get("available")
            borrow = (short_map.get(etf) or {}).get("borrow")
            if isinstance(avail, (int, float)) and int(avail) <= 0:
                summary_rows.append(
                    {**base_summary, "requested_short_sh": requested_short_sh, "status": "SKIP_FTP_AVAILABLE_ZERO"}
                )
                continue
            if isinstance(avail, (int, float)) and int(avail) > 0:
                requested_short_sh = min(requested_short_sh, int(avail))
            if requested_short_sh <= 0:
                summary_rows.append({**base_summary, "status": "SKIP_ZERO_AFTER_FTP_CAP"})
                continue

            # Resolve the maintained hedge ratio for B4 pairs from the live snapshot.
            r_live = beta
            r_src = "delta_neutral"
            if lane == "b4_pair":
                inv_raw = abs(float(strat_pos.get(etf, 0.0)) * px_etf)
                live_net_under = float(strat_pos.get(under, 0.0)) * px_under
                r_live, r_src = resolve_b4_hedge_ratio(
                    underlying=under,
                    etf=etf,
                    inverse_raw_notional=inv_raw,
                    beta=beta,
                    b4_detail_etf_ratio=b4_detail_etf_ratio,
                    b4_detail_map=b4_detail_map,
                    live_net_under_notional=live_net_under,
                    plan_ratio=plan_ratio,
                    mode=ratio_mode,
                )

            planned_rows.append(
                {
                    "symbol": etf,
                    "underlying": under,
                    "lane": lane,
                    "delta": delta,
                    "beta": beta,
                    "r_live": float(r_live),
                    "hedge_ratio_source": r_src,
                    "plan_ratio": plan_ratio,
                    "target_short_usd": need_usd,
                    "etf_px": px_etf,
                    "under_px": px_under,
                    "requested_short_sh": requested_short_sh,
                    "ftp_available": avail,
                    "ftp_borrow_annual": borrow,
                }
            )

        if not planned_rows:
            tprint("[HARVEST] No actionable candidates after planning filters.")
            write_harvest_csv(attempted_path, pd.DataFrame(attempted_rows))
            write_harvest_csv(fills_path, pd.DataFrame(fill_rows))
            write_harvest_csv(summary_path, pd.DataFrame(summary_rows))
            write_harvest_csv(netting_path, pd.DataFrame())
            return 0

        # ---- Pre-trade preview (assumes full ETF short fill) ----------------------------
        preview_df = pd.DataFrame(planned_rows).copy()
        preview_df["requested_short_notional_usd"] = preview_df["requested_short_sh"] * preview_df["etf_px"]
        preview_pair_deltas = [
            {
                "underlying": r["underlying"],
                "etf": r["symbol"],
                "under_delta_usd": compute_underlying_delta_usd(
                    lane=r["lane"],
                    filled_inverse_notional=r["requested_short_sh"] * r["etf_px"],
                    delta=r["delta"],
                    r_live=r["r_live"],
                    buffer_pct=buffer_pct,
                ),
            }
            for r in planned_rows
        ]
        preview_orders = (
            net_underlying_orders(preview_pair_deltas, prices, min_trade_usd)
            if net_underlyings
            else net_underlying_orders(
                [{**d, "underlying": f"{d['underlying']}|{d['etf']}"} for d in preview_pair_deltas],
                {f"{r['underlying']}|{r['symbol']}": r["under_px"] for r in planned_rows},
                min_trade_usd,
            )
        )

        tprint("")
        tprint("=" * 120)
        tprint("[HARVEST] PRE-TRADE PLAN — ETF short legs (what will be attempted)")
        tprint("=" * 120)
        tprint(
            preview_df[
                [
                    "symbol",
                    "underlying",
                    "lane",
                    "target_short_usd",
                    "etf_px",
                    "requested_short_sh",
                    "requested_short_notional_usd",
                    "r_live",
                    "hedge_ratio_source",
                    "ftp_available",
                ]
            ].to_string(index=False)
        )
        tprint("-" * 120)
        tprint("[HARVEST] PRE-TRADE PLAN — netted underlying legs (if full fill)")
        if preview_orders:
            tprint(
                pd.DataFrame(preview_orders)[
                    ["underlying", "action", "qty", "net_usd", "gross_buy_usd", "gross_sell_usd", "etfs", "decision"]
                ].to_string(index=False)
            )
        else:
            tprint("(none)")
        tprint("=" * 120)
        tprint("")

        if (not dry_run) and (not args.auto_approve):
            ans = input("[HARVEST] Approve live execution of this plan? (y/n): ").strip().lower()
            if ans != "y":
                tprint("[HARVEST] Execution cancelled by user.")
                for r in planned_rows:
                    summary_rows.append(
                        {
                            "symbol": str(r["symbol"]),
                            "underlying": str(r["underlying"]),
                            "lane": str(r["lane"]),
                            "requested_short_usd": float(r["target_short_usd"]),
                            "requested_short_sh": int(r["requested_short_sh"]),
                            "filled_short_sh": 0,
                            "r_live": float(r["r_live"]),
                            "hedge_ratio_source": str(r["hedge_ratio_source"]),
                            "intended_under_delta_usd": 0.0,
                            "remaining_short_usd": float(r["target_short_usd"]),
                            "status": "SKIP_USER_CANCELLED",
                        }
                    )
                write_harvest_csv(attempted_path, pd.DataFrame(attempted_rows))
                write_harvest_csv(fills_path, pd.DataFrame(fill_rows))
                write_harvest_csv(summary_path, pd.DataFrame(summary_rows))
                write_harvest_csv(netting_path, pd.DataFrame())
                return 0

        # ---- Phase A: execute all ETF short legs ----------------------------------------
        pair_delta_rows: list[dict[str, Any]] = []
        for row in planned_rows:
            if stop_requested():
                tprint("[HARVEST] Stop requested; halting ETF short execution loop.")
                break

            etf = norm_sym(str(row["symbol"]))
            under = norm_sym(str(row["underlying"]))
            lane = str(row["lane"])
            delta = float(row["delta"])
            r_live = float(row["r_live"])
            r_src = str(row["hedge_ratio_source"])
            need_usd = float(row["target_short_usd"])
            px_etf = float(row["etf_px"])
            requested_short_sh = int(row["requested_short_sh"])

            tprint(
                f"[HARVEST] TRY {etf} ({lane}): short {requested_short_sh} sh "
                f"(~${requested_short_sh * px_etf:,.0f}); underlying {under} "
                f"{'SELL' if lane == 'b4_pair' else 'BUY'} r={r_live:.3f} ({r_src})."
            )

            short_ref = f"{strategy_tag}|HARVEST_SHORT|{under}|{etf}"
            short_res = execute_leg(
                ib=ib,
                symbol=etf,
                action="SELL",
                qty=requested_short_sh,
                ref_price=px_etf,
                bps=limit_bps,
                order_ref=short_ref,
                exec_cfg=exec_cfg,
                timeout=timeout_sec,
                max_retries=max_retries,
                dry_run=dry_run,
                context=f"HARVEST|{etf}",
                cancel_service=cancel_service,
            )
            filled_short_sh = int(short_res.filled)
            attempted_rows.append(
                {
                    "symbol": etf,
                    "underlying": under,
                    "lane": lane,
                    "leg": "ETF_SHORT",
                    "action": "SELL",
                    "requested_sh": requested_short_sh,
                    "filled_sh": filled_short_sh,
                    "status": short_res.status,
                    "error_code": short_res.error_code,
                    "error_msg": short_res.error_msg,
                    "price_ref": px_etf,
                    "delta": delta,
                    "ftp_available": row.get("ftp_available"),
                    "ftp_borrow_annual": row.get("ftp_borrow_annual"),
                }
            )

            filled_notional = filled_short_sh * px_etf
            under_delta_usd = (
                compute_underlying_delta_usd(
                    lane=lane,
                    filled_inverse_notional=filled_notional,
                    delta=delta,
                    r_live=r_live,
                    buffer_pct=buffer_pct,
                )
                if filled_short_sh > 0
                else 0.0
            )

            summary_rows.append(
                {
                    "symbol": etf,
                    "underlying": under,
                    "lane": lane,
                    "requested_short_usd": need_usd,
                    "requested_short_sh": requested_short_sh,
                    "filled_short_sh": filled_short_sh,
                    "r_live": r_live,
                    "hedge_ratio_source": r_src,
                    "intended_under_delta_usd": under_delta_usd,
                    "remaining_short_usd": max(0.0, need_usd - filled_notional),
                    "status": f"SHORT_{short_res.status}" if filled_short_sh > 0 else f"NO_SHORT_FILL_{short_res.status}",
                }
            )
            fill_rows.append(
                {
                    "symbol": etf,
                    "underlying": under,
                    "lane": lane,
                    "filled_short_sh": filled_short_sh,
                    "etf_px": px_etf,
                    "under_px": float(row["under_px"]),
                    "delta": delta,
                    "r_live": r_live,
                    "short_fill_notional_usd": filled_notional,
                    "intended_under_delta_usd": under_delta_usd,
                }
            )
            if filled_short_sh > 0 and abs(under_delta_usd) > 0:
                pair_delta_rows.append(
                    {"underlying": under, "etf": etf, "lane": lane, "under_delta_usd": under_delta_usd}
                )

        # ---- Phase C: net underlying deltas by symbol and execute -----------------------
        if net_underlyings:
            netted = net_underlying_orders(pair_delta_rows, prices, min_trade_usd)
        else:
            # Per-pair execution: keep pairs distinct via a composite key, then unwrap.
            keyed = [{**d, "underlying": f"{d['underlying']}|{d['etf']}"} for d in pair_delta_rows]
            keyed_px = {f"{d['underlying']}|{d['etf']}": float(prices.get(d["underlying"]) or 0.0) for d in pair_delta_rows}
            netted = net_underlying_orders(keyed, keyed_px, min_trade_usd)
            for o in netted:
                o["underlying"] = o["underlying"].split("|", 1)[0]

        for o in netted:
            if stop_requested():
                tprint("[HARVEST] Stop requested; halting underlying execution loop.")
                break

            under = norm_sym(str(o["underlying"]))
            px_under = float(o.get("px_under") or prices.get(under) or 0.0)
            action = o.get("action")
            qty = int(o.get("qty") or 0)
            if o.get("decision") != "trade" or not action or qty <= 0 or px_under <= 0:
                netting_rows.append({**o, "requested_sh": 0, "filled_sh": 0, "status": f"SKIP_{o.get('reason', 'no_trade')}"})
                continue

            cur_under_sh = float(strat_pos.get(under, 0.0))
            locate_note = ""
            if action == "SELL":
                # Portion covered by an existing long needs no locate; the remainder is a true
                # short sale (structural short) and must pass the locate / FTP gate.
                long_cover_sh = int(min(qty, max(0.0, cur_under_sh)))
                short_sh = qty - long_cover_sh
                if short_sh > 0:
                    blocked, why = is_short_unavailable_now(under, short_map=short_map)
                    if blocked:
                        short_sh = 0
                        locate_note = f"|short_clipped_{why}"
                    else:
                        avail_u = (short_map.get(under) or {}).get("available")
                        if isinstance(avail_u, (int, float)) and int(avail_u) >= 0:
                            short_sh = min(short_sh, int(avail_u))
                qty = long_cover_sh + short_sh
                if qty <= 0:
                    netting_rows.append(
                        {**o, "requested_sh": int(o.get("qty") or 0), "filled_sh": 0, "status": f"SKIP_NO_LOCATE{locate_note}"}
                    )
                    continue

            under_ref = f"{strategy_tag}|HARVEST_HEDGE_NET|{under}|{action}"
            hedge_res = execute_leg(
                ib=ib,
                symbol=under,
                action=action,
                qty=qty,
                ref_price=px_under,
                bps=limit_bps,
                order_ref=under_ref,
                exec_cfg=exec_cfg,
                timeout=timeout_sec,
                max_retries=max_retries,
                dry_run=dry_run,
                context=f"HARVEST|{under}",
                cancel_service=cancel_service,
            )
            filled_under_sh = int(hedge_res.filled)
            attempted_rows.append(
                {
                    "symbol": under,
                    "underlying": under,
                    "lane": "underlying_net",
                    "leg": "UNDER_NET",
                    "action": action,
                    "requested_sh": qty,
                    "filled_sh": filled_under_sh,
                    "status": hedge_res.status,
                    "error_code": hedge_res.error_code,
                    "error_msg": hedge_res.error_msg,
                    "price_ref": px_under,
                    "delta": None,
                    "ftp_available": (short_map.get(under) or {}).get("available"),
                    "ftp_borrow_annual": (short_map.get(under) or {}).get("borrow"),
                }
            )
            netting_rows.append(
                {
                    **o,
                    "requested_sh": qty,
                    "filled_sh": filled_under_sh,
                    "filled_notional_usd": filled_under_sh * px_under,
                    "status": f"{action}_{hedge_res.status}{locate_note}",
                }
            )
    finally:
        try:
            cancel_service.stop()
        except Exception:
            pass
        try:
            ib.disconnect()
        except Exception:
            pass

    write_harvest_csv(attempted_path, pd.DataFrame(attempted_rows))
    write_harvest_csv(fills_path, pd.DataFrame(fill_rows))
    write_harvest_csv(summary_path, pd.DataFrame(summary_rows))
    write_harvest_csv(netting_path, pd.DataFrame(netting_rows))

    tprint(f"[HARVEST] Wrote candidates -> {candidates_path}")
    tprint(f"[HARVEST] Wrote attempted trades -> {attempted_path}")
    tprint(f"[HARVEST] Wrote fills -> {fills_path}")
    tprint(f"[HARVEST] Wrote post-trade summary -> {summary_path}")
    tprint(f"[HARVEST] Wrote underlying netting -> {netting_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
