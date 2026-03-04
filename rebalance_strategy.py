#!/usr/bin/env python3
"""
rebalance_strategy.py

Hybrid three-phase rebalancer for the ls-algo ETF long/short strategy.

Phase 1 — Cleanup:   Close ETF positions not in proposed_trades.csv
Phase 2 — Establish: Open both legs for new pairs with no/tiny current position
Phase 3 — Hedge:     Directional net-exposure correction (core_leveraged +
                     whitelist_stock only). Trades ONE leg per underlying to
                     minimise transaction count.

Usage:
    python rebalance_strategy.py [--dry-run] [--run-date YYYY-MM-DD]
                                 [--skip-phase-1] [--skip-phase-2] [--skip-phase-3]
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml
from ib_insync import IB, Trade

from execute_trade_plan import (
    # Thread / shutdown
    tprint, stop_requested, handle_sigint, ensure_thread_event_loop,
    PRINT_LOCK, SHUTDOWN,
    # Exposure logging
    EXPOSURE_COLS, compute_portfolio_notionals,
    append_csv_row, append_jsonl, safe_avg_fill_price,
    # Symbol helpers
    norm_sym, make_stock,
    ib_symbol_from_universal, universal_symbol_from_ib,
    IB_SYMBOL_MAP, REVERSE_IB_SYMBOL_MAP,
    # Path helpers
    today_str, run_dir,
    # IBKR connection + pricing
    connect_ib, get_snapshot_price, safe_price, ensure_price_coordinator,
    # Order construction + execution
    build_market_order, build_adaptive_market_order, build_limit_order,
    wait_for_trade_terminal, wait_for_trade_accepted,
    ExecResult, CancelRequest, CoordinatorCancelService, execute_leg,
    # Position helpers
    load_baseline_qty, current_ib_positions, strategy_position_only,
    target_shares_from_usd, fmt_dollars,
    # Screened data helpers
    build_borrow_by_etf, build_purgatory_set,
    # Short availability
    fetch_ibkr_short_availability_map, is_short_not_available,
    # Cleanup (Phase 1)
    build_cleanup_trades_to_match_plan, print_cleanup_trade_list,
    build_contract_cache, execute_cleanup_trades_parallel,
    # Sync + IO
    _sync_positions_after_external_trades,
    append_fills, resolve_plan_path, resolve_fills_path,
    write_execution_snapshot,
)
from execute_flow_program import get_account_equity
from ibkr_accounting import load_etf_beta_map
from generate_trade_plan import load_blacklist


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def rebalance_dir(run_date: str) -> Path:
    """data/runs/<run_date>/rebalance/"""
    d = run_dir(run_date) / "rebalance"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Plan loading
# ---------------------------------------------------------------------------

def load_plan(
    plan_path: Path,
    strategy_tag: str,
    flow_etfs: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load proposed_trades.csv filtered to strategy_tag.

    Returns:
        plan_df       — all plan rows for the tag (all sleeves)
        hedgeable_df  — core_leveraged + whitelist_stock, non-purgatory, non-flow
    """
    if not plan_path.exists():
        raise FileNotFoundError(f"Trade plan not found: {plan_path}")

    plan = pd.read_csv(plan_path)
    if "strategy_tag" in plan.columns:
        plan = plan[plan["strategy_tag"].astype(str) == strategy_tag].copy()

    if plan.empty:
        tprint(f"[PLAN] WARNING: No rows for strategy_tag={strategy_tag}; plan is empty.")
        empty: pd.DataFrame = pd.DataFrame()
        return empty, empty

    plan["ETF"]        = plan["ETF"].astype(str).map(norm_sym)
    plan["Underlying"] = plan["Underlying"].astype(str).map(norm_sym)
    plan["purgatory"]  = plan.get("purgatory", False).fillna(False).astype(bool) \
                         if "purgatory" in plan.columns else False
    plan["sleeve"]     = plan.get("sleeve", "core_leveraged") \
                         if "sleeve" in plan.columns else "core_leveraged"
    plan["long_usd"]   = pd.to_numeric(plan.get("long_usd", 0), errors="coerce").fillna(0.0)
    plan["short_usd"]  = pd.to_numeric(plan.get("short_usd", 0), errors="coerce").fillna(0.0)
    plan = plan.reset_index(drop=True)

    hedgeable_mask = (
        plan["sleeve"].isin({"core_leveraged", "whitelist_stock"})
        & (~plan["purgatory"])
        & (~plan["ETF"].isin(flow_etfs))
    )
    hedgeable_df = plan[hedgeable_mask].copy().reset_index(drop=True)

    return plan, hedgeable_df


# ---------------------------------------------------------------------------
# Phase 2 — Establish new positions
# ---------------------------------------------------------------------------

def build_establish_trades(
    *,
    plan: pd.DataFrame,
    strat_pos: Dict[str, float],
    prices: Dict[str, float],
    purgatory_etfs: Set[str],
    flow_etfs: Set[str],
    etf_to_under: Dict[str, str],
    min_trade_usd: float,
    establish_threshold_usd: float = 100.0,
) -> List[dict]:
    """
    Identify pairs in the plan that need to be established from scratch.
    A pair qualifies when BOTH legs are near-zero (< establish_threshold_usd).
    """
    trades: List[dict] = []
    seen_under: Set[str] = set()

    for _, row in plan.iterrows():
        etf   = norm_sym(str(row["ETF"]))
        under = norm_sym(str(row["Underlying"]))

        if etf in purgatory_etfs or etf in flow_etfs:
            continue
        if under in seen_under:
            continue

        long_usd  = float(row.get("long_usd", 0.0)  or 0.0)
        short_usd = float(row.get("short_usd", 0.0) or 0.0)
        if long_usd <= 0 and short_usd <= 0:
            continue

        cur_etf_notional   = abs(float(strat_pos.get(etf, 0.0)))   * float(prices.get(etf, 0.0))
        cur_under_notional = abs(float(strat_pos.get(under, 0.0))) * float(prices.get(under, 0.0))

        # Only establish if BOTH legs are near-zero
        if cur_etf_notional > establish_threshold_usd or cur_under_notional > establish_threshold_usd:
            continue

        seen_under.add(under)
        trades.append({
            "underlying": under,
            "etf":        etf,
            "long_usd":   long_usd,
            "short_usd":  short_usd,
        })

    tprint(f"[ESTABLISH] {len(trades)} new pairs to establish.")
    return trades


def _establish_worker(
    trade_info: dict,
    *,
    worker_idx: int,
    host: str,
    port: int,
    client_id: int,
    baseline: Dict[str, float],
    prices: Dict[str, float],
    prefer_delayed: bool,
    exec_cfg: dict,
    strategy_tag: str,
    run_date: str,
    limit_bps: float,
    timeout: float,
    max_retries: int,
    dry_run: bool,
    short_map: Dict[str, dict],
    cancel_service: CoordinatorCancelService,
    log_exposure_event,
    short_first: bool,
    log_lock: threading.Lock,
) -> List[dict]:
    ensure_thread_event_loop()
    if stop_requested():
        return []

    under     = trade_info["underlying"]
    etf       = trade_info["etf"]
    long_usd  = trade_info["long_usd"]
    short_usd = trade_info["short_usd"]
    local_fills: List[dict] = []

    try:
        ib_local = connect_ib(host, port, client_id + 200 + worker_idx)
    except Exception as e:
        tprint(f"[ESTABLISH][{under}] Worker IB connect failed: {e}")
        return []

    try:
        px_under = prices.get(under) or get_snapshot_price(ib_local, under, prefer_delayed)
        px_etf   = prices.get(etf)   or get_snapshot_price(ib_local, etf,   prefer_delayed)

        if not px_under or not px_etf:
            tprint(f"[ESTABLISH][{under}] No price for {under}={px_under} or {etf}={px_etf}; skipping.")
            return []

        with log_lock:
            prices[under] = float(px_under)
            prices[etf]   = float(px_etf)

        target_under_sh = target_shares_from_usd(long_usd,  px_under)
        target_etf_sh   = target_shares_from_usd(short_usd, px_etf)

        def exec_leg_local(sym: str, action: str, qty: int, px: float, ref: str) -> ExecResult:
            return execute_leg(
                ib=ib_local, symbol=sym, action=action, qty=qty,
                ref_price=px, bps=limit_bps, order_ref=ref,
                exec_cfg=exec_cfg, timeout=timeout, max_retries=max_retries,
                dry_run=dry_run, context=f"{under}|ESTABLISH",
                cancel_service=cancel_service,
            )

        legs = []
        if target_etf_sh > 0:
            legs.append(("SELL", etf,   target_etf_sh,   px_etf,   "etf"))
        if target_under_sh > 0:
            legs.append(("BUY",  under, target_under_sh, px_under, "under"))

        if short_first:
            legs.sort(key=lambda x: (0 if x[0] == "SELL" else 1))

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for action, sym, qty, px, leg_type in legs:
            if stop_requested():
                break

            # FTP gate for short leg
            if action == "SELL":
                sm    = short_map.get(sym, {})
                avail = sm.get("available")
                if avail is not None and avail <= 0:
                    tprint(f"[ESTABLISH][{under}] SKIP {sym}: FTP available=0.")
                    local_fills.append({
                        "filled_at": now, "run_date": run_date,
                        "strategy_tag": strategy_tag,
                        "pair_id": f"{under}__ESTABLISH",
                        "underlying": under, "etf": etf,
                        "px_under": px_under, "px_etf": px_etf,
                        "target_sh_under": target_under_sh,
                        "target_sh_etf":   target_etf_sh,
                        "delta_sh_under": 0, "delta_sh_etf": -target_etf_sh,
                        "filled_sh_under": 0, "filled_sh_etf": 0,
                        "notes": f"SKIP_FTP_AVAIL0 wants_short={target_etf_sh}",
                    })
                    continue

            order_ref = f"{strategy_tag}|{under}__ESTABLISH|{sym}|{leg_type.upper()}"
            res = exec_leg_local(sym, action, qty, px, order_ref)
            filled_signed = int(res.filled) if action == "BUY" else -int(res.filled)

            stage = "POST_ETF" if leg_type == "etf" else "POST_UNDER_ESTABLISH"
            log_exposure_event(
                stage=stage, pair_id=f"{under}__ESTABLISH",
                underlying=under, etf=etf, symbol=sym,
                delta_sh=(-qty if action == "SELL" else qty),
                filled_sh=filled_signed, trade=res.trade,
            )

            local_fills.append({
                "filled_at": now, "run_date": run_date,
                "strategy_tag": strategy_tag,
                "pair_id": f"{under}__ESTABLISH",
                "underlying": under, "etf": etf,
                "px_under": px_under, "px_etf": px_etf,
                "target_sh_under": target_under_sh,
                "target_sh_etf":   target_etf_sh,
                "delta_sh_under": (filled_signed if leg_type == "under" else 0),
                "delta_sh_etf":   (filled_signed if leg_type == "etf"   else 0),
                "filled_sh_under": (filled_signed if leg_type == "under" else 0),
                "filled_sh_etf":   (filled_signed if leg_type == "etf"   else 0),
                "notes": f"ESTABLISH_{leg_type.upper()} status={res.status}",
            })

        return local_fills

    finally:
        try:
            ib_local.disconnect()
        except Exception:
            pass


def execute_establish_parallel(
    *,
    establish_trades: List[dict],
    host: str,
    port: int,
    client_id: int,
    baseline: Dict[str, float],
    prices: Dict[str, float],
    prefer_delayed: bool,
    exec_cfg: dict,
    strategy_tag: str,
    run_date: str,
    limit_bps: float,
    timeout: float,
    max_retries: int,
    dry_run: bool,
    parallel_n: int,
    short_map: Dict[str, dict],
    cancel_service: CoordinatorCancelService,
    log_exposure_event,
    short_first: bool,
    log_lock: threading.Lock,
) -> List[dict]:
    if not establish_trades:
        return []

    all_fills: List[dict] = []
    n_workers = min(parallel_n, len(establish_trades))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _establish_worker, t,
                worker_idx=i,
                host=host, port=port, client_id=client_id,
                baseline=baseline, prices=prices,
                prefer_delayed=prefer_delayed, exec_cfg=exec_cfg,
                strategy_tag=strategy_tag, run_date=run_date,
                limit_bps=limit_bps, timeout=timeout, max_retries=max_retries,
                dry_run=dry_run, short_map=short_map,
                cancel_service=cancel_service,
                log_exposure_event=log_exposure_event,
                short_first=short_first, log_lock=log_lock,
            ): t
            for i, t in enumerate(establish_trades)
        }
        for fut in as_completed(futures):
            try:
                all_fills.extend(fut.result())
            except Exception as ex:
                tprint(f"[ESTABLISH] Worker raised: {ex}")

    return all_fills


# ---------------------------------------------------------------------------
# Phase 3 — Directional hedge math
# ---------------------------------------------------------------------------

def compute_beta_adjusted_net_notional(
    *,
    strat_pos: Dict[str, float],
    prices: Dict[str, float],
    underlying: str,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
) -> Tuple[float, float]:
    """
    Beta-adjusted net notional for one underlying group.

    Each symbol s in the group contributes:
        contribution = shares_s * price_s * beta_s
    where beta=1.0 for the spot underlying, and ETF shares are negative (short).

    Returns (net_notional, gross_notional).
        net > 0  =>  net long exposure
        net < 0  =>  net short exposure
    """
    net   = 0.0
    gross = 0.0

    for sym, sh_raw in strat_pos.items():
        sh = float(sh_raw)
        if sh == 0.0:
            continue

        is_under = (sym == underlying)
        is_etf   = (etf_to_under.get(sym) == underlying)

        if not is_under and not is_etf:
            continue

        beta = etf_to_beta.get(sym, 1.0) if is_etf else 1.0
        px   = float(prices.get(sym) or 0.0)
        if px <= 0.0:
            continue

        contrib = sh * px * beta
        net   += contrib
        gross += abs(contrib)

    return net, gross


def compute_target_gross_per_underlying(
    *,
    underlying: str,
    plan: pd.DataFrame,
    account_equity: float,
    gross_leverage: float,
) -> float:
    """
    target_gross = account_equity * gross_leverage * pair_weight

    pair_weight = this_underlying_long_usd / total_plan_long_usd
    (plan must already be filtered to hedgeable rows only)
    """
    total_long = float(plan["long_usd"].sum())
    if total_long <= 0.0:
        return 0.0

    this_long   = float(plan[plan["Underlying"] == underlying]["long_usd"].sum())
    pair_weight = this_long / total_long
    return account_equity * gross_leverage * pair_weight


def compute_hedge_delta(
    *,
    net_notional: float,
    target_gross: float,
    net_exposure_band: float,
) -> Tuple[bool, float]:
    """
    Determine whether a hedge is needed and by how much.

    Trigger:  |net_notional| > net_exposure_band * target_gross

    If triggered:
        target_net   = sign(net) * net_exposure_band * target_gross / 2
        correction   = target_net - net_notional

    correction < 0  =>  net too long  =>  add to ETF short
    correction > 0  =>  net too short =>  add to underlying long

    Returns (triggered, correction_usd).
    """
    if target_gross <= 0.0:
        return False, 0.0

    trigger_threshold = net_exposure_band * target_gross
    if abs(net_notional) <= trigger_threshold:
        return False, 0.0

    sign       = 1.0 if net_notional >= 0.0 else -1.0
    target_net = sign * (net_exposure_band * target_gross / 2.0)
    correction = target_net - net_notional
    return True, correction


def resolve_hedge_leg(
    *,
    underlying: str,
    correction_usd: float,
    strat_pos: Dict[str, float],
    prices: Dict[str, float],
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
    plan: pd.DataFrame,
) -> Tuple[Optional[str], str, int, float]:
    """
    Resolve which symbol/leg to trade to achieve the correction.

    correction < 0  =>  add to ETF short  (SELL ETF)
    correction > 0  =>  add to underlying long  (BUY underlying)

    ETF selection priority:
        1. ETFs already in position (negative shares)
        2. Largest absolute position (tiebreak)

    Returns (symbol, action, qty, ref_price).
    symbol is None if leg cannot be resolved.
    """
    if correction_usd == 0.0:
        return None, "SELL", 0, 0.0

    if correction_usd < 0:
        # Add to ETF short leg
        plan_etfs = set(plan["ETF"].astype(str))
        etf_candidates = [
            sym for sym, u in etf_to_under.items()
            if u == underlying and sym in plan_etfs
        ]
        if not etf_candidates:
            tprint(f"[HEDGE][{underlying}] No ETF candidates in plan; cannot add to short.")
            return None, "SELL", 0, 0.0

        # Sort: prefer already-shorted ETFs, then by largest abs position
        etf_candidates.sort(key=lambda s: (
            0 if float(strat_pos.get(s, 0.0)) < 0 else 1,
            -abs(float(strat_pos.get(s, 0.0))),
        ))

        for etf in etf_candidates:
            px   = float(prices.get(etf) or 0.0)
            beta = etf_to_beta.get(etf, 1.0)
            if px <= 0.0 or beta <= 0.0:
                continue
            qty = int(math.floor(abs(correction_usd) / (px * beta)))
            if qty <= 0:
                continue
            return etf, "SELL", qty, px

        tprint(f"[HEDGE][{underlying}] No valid ETF price available for short hedge.")
        return None, "SELL", 0, 0.0

    else:
        # Add to underlying long leg
        px = float(prices.get(underlying) or 0.0)
        if px <= 0.0:
            tprint(f"[HEDGE][{underlying}] No price for underlying; cannot add to long.")
            return None, "BUY", 0, 0.0
        qty = int(math.floor(abs(correction_usd) / px))
        if qty <= 0:
            return None, "BUY", 0, 0.0
        return underlying, "BUY", qty, px


def build_hedge_trades(
    *,
    hedgeable_plan: pd.DataFrame,
    strat_pos: Dict[str, float],
    prices: Dict[str, float],
    account_equity: float,
    gross_leverage: float,
    net_exposure_band: float,
    min_trade_usd: float,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
) -> List[dict]:
    """
    Iterate over all underlyings in hedgeable_plan. For each:
        1. Compute beta-adjusted net notional from live positions
        2. Compute target gross from live equity
        3. Check trigger
        4. If triggered, resolve the single leg to trade

    Returns list of trade dicts ordered by abs(correction_usd) descending
    (largest imbalances traded first).
    """
    underlyings = sorted(hedgeable_plan["Underlying"].unique())
    trades: List[dict] = []

    for under in underlyings:
        if stop_requested():
            break

        net_notional, _ = compute_beta_adjusted_net_notional(
            strat_pos=strat_pos, prices=prices, underlying=under,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
        )
        target_gross = compute_target_gross_per_underlying(
            underlying=under, plan=hedgeable_plan,
            account_equity=account_equity, gross_leverage=gross_leverage,
        )

        triggered, correction_usd = compute_hedge_delta(
            net_notional=net_notional, target_gross=target_gross,
            net_exposure_band=net_exposure_band,
        )

        net_pct = (net_notional / target_gross * 100.0) if target_gross > 0 else 0.0
        tprint(
            f"[HEDGE][{under:15s}] net={net_notional:>+12,.0f}  "
            f"tgt_gross={target_gross:>10,.0f}  net%={net_pct:>+6.1f}%  "
            f"{'TRIGGERED' if triggered else 'ok'}"
        )

        if not triggered:
            continue
        if abs(correction_usd) < min_trade_usd:
            tprint(
                f"[HEDGE][{under}] correction={correction_usd:+,.0f} "
                f"< min_trade_usd={min_trade_usd}; skipping."
            )
            continue

        symbol, action, qty, ref_px = resolve_hedge_leg(
            underlying=under, correction_usd=correction_usd,
            strat_pos=strat_pos, prices=prices,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
            plan=hedgeable_plan,
        )
        if symbol is None or qty <= 0:
            tprint(f"[HEDGE][{under}] Could not resolve hedge leg; skipping.")
            continue

        tprint(
            f"[HEDGE][{under}] -> {action} {qty} {symbol} @ ~{ref_px:.2f} "
            f"(correction={correction_usd:+,.0f})"
        )
        trades.append({
            "underlying":        under,
            "symbol":            symbol,
            "action":            action,
            "qty":               qty,
            "ref_price":         ref_px,
            "correction_usd":    correction_usd,
            "net_notional_before": net_notional,
            "target_gross":      target_gross,
        })

    # Largest corrections first
    trades.sort(key=lambda t: abs(t["correction_usd"]), reverse=True)
    tprint(
        f"[HEDGE] Checked {len(underlyings)} underlyings -> "
        f"{len(trades)} hedge trades queued."
    )
    return trades


def execute_hedge_pass_serial(
    *,
    hedge_trades: List[dict],
    ib: IB,
    baseline: Dict[str, float],
    prices: Dict[str, float],
    prefer_delayed: bool,
    exec_cfg: dict,
    strategy_tag: str,
    run_date: str,
    limit_bps: float,
    timeout: float,
    max_retries: int,
    dry_run: bool,
    short_map: Dict[str, dict],
    cancel_service: CoordinatorCancelService,
    log_exposure_event,
    net_exposure_band: float,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
) -> Tuple[List[dict], int, int]:
    """
    Execute Phase 3 hedge trades serially on the coordinator connection.

    For each trade:
      1. Re-read live position to confirm still triggered (prices may have moved)
      2. FTP gate for ETF shorts
      3. execute_leg on coordinator IB
      4. Log exposure event and append fill record

    Returns (fill_records, n_triggered, n_traded).
    """
    fill_records: List[dict] = []
    n_triggered = len(hedge_trades)
    n_traded    = 0

    for trade_info in hedge_trades:
        if stop_requested():
            tprint("[HEDGE] Shutdown requested; aborting hedge pass.")
            break

        under      = trade_info["underlying"]
        symbol     = trade_info["symbol"]
        action     = trade_info["action"]
        qty        = trade_info["qty"]
        ref_px     = trade_info["ref_price"]
        target_gross = trade_info["target_gross"]

        # Re-verify: re-read live position before each trade
        ib_pos_now = current_ib_positions(ib)
        strat_now  = strategy_position_only(ib_pos_now, baseline)
        net_now, _ = compute_beta_adjusted_net_notional(
            strat_pos=strat_now, prices=prices, underlying=under,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
        )
        triggered_now, _ = compute_hedge_delta(
            net_notional=net_now, target_gross=target_gross,
            net_exposure_band=net_exposure_band,
        )
        if not triggered_now:
            tprint(
                f"[HEDGE][{under}] No longer triggered after re-read "
                f"(net={net_now:+,.0f}); skipping."
            )
            n_triggered -= 1
            continue

        # Fresh price
        fresh_px   = get_snapshot_price(ib, symbol, prefer_delayed=prefer_delayed)
        px         = float(fresh_px or ref_px)
        prices[symbol] = px

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        px_under = float(prices.get(under) or 0.0)

        # FTP gate for new shorts
        if action == "SELL":
            sm    = short_map.get(symbol, {})
            avail = sm.get("available")
            if avail is not None and avail <= 0:
                tprint(f"[HEDGE][{under}] SKIP {symbol}: FTP available=0.")
                fill_records.append({
                    "filled_at": now, "run_date": run_date,
                    "strategy_tag": strategy_tag,
                    "pair_id": f"{under}__HEDGE",
                    "underlying": under, "etf": symbol,
                    "px_under": px_under, "px_etf": px,
                    "target_sh_under": 0, "target_sh_etf": 0,
                    "delta_sh_under": 0, "delta_sh_etf": -qty,
                    "filled_sh_under": 0, "filled_sh_etf": 0,
                    "notes": "HEDGE_SKIP_FTP_AVAIL0",
                })
                continue

        # Log pre-hedge state
        log_exposure_event(
            stage="HEDGE_PRE", pair_id=f"{under}__HEDGE",
            underlying=under,
            etf=(symbol if etf_to_under.get(symbol) == under else ""),
            symbol=symbol, delta_sh=0, filled_sh=0, trade=None,
        )

        order_ref = f"{strategy_tag}|{under}__HEDGE|{symbol}|{action}"
        res = execute_leg(
            ib=ib, symbol=symbol, action=action, qty=qty,
            ref_price=px, bps=limit_bps, order_ref=order_ref,
            exec_cfg=exec_cfg, timeout=timeout, max_retries=max_retries,
            dry_run=dry_run, context=f"{under}|HEDGE",
            cancel_service=cancel_service,
        )

        filled_signed = int(res.filled) if action == "BUY" else -int(res.filled)

        # Graceful 201 handling
        if res.status == "SHORT_BLOCKED":
            tprint(f"[HEDGE][{under}] {symbol} SHORT_BLOCKED (201). Logging and continuing.")
            fill_records.append({
                "filled_at": now, "run_date": run_date,
                "strategy_tag": strategy_tag,
                "pair_id": f"{under}__HEDGE",
                "underlying": under, "etf": symbol,
                "px_under": px_under, "px_etf": px,
                "target_sh_under": 0, "target_sh_etf": 0,
                "delta_sh_under": 0, "delta_sh_etf": -qty,
                "filled_sh_under": 0, "filled_sh_etf": 0,
                "notes": f"HEDGE_SHORT_BLOCKED_201 msg={res.error_msg}",
            })
            log_exposure_event(
                stage="POST_HEDGE", pair_id=f"{under}__HEDGE",
                underlying=under, etf=symbol, symbol=symbol,
                delta_sh=-qty, filled_sh=0, trade=res.trade,
            )
            continue

        log_exposure_event(
            stage="POST_HEDGE", pair_id=f"{under}__HEDGE",
            underlying=under,
            etf=(symbol if etf_to_under.get(symbol) == under else ""),
            symbol=symbol,
            delta_sh=(-qty if action == "SELL" else qty),
            filled_sh=filled_signed, trade=res.trade,
        )

        is_etf_trade = (etf_to_under.get(symbol) == under)
        fill_records.append({
            "filled_at": now, "run_date": run_date,
            "strategy_tag": strategy_tag,
            "pair_id": f"{under}__HEDGE",
            "underlying": under,
            "etf": (symbol if is_etf_trade else ""),
            "px_under": px_under,
            "px_etf":   (px if is_etf_trade else 0.0),
            "target_sh_under": 0, "target_sh_etf": 0,
            "delta_sh_under":  (qty if action == "BUY"  else 0),
            "delta_sh_etf":    (qty if action == "SELL" else 0),
            "filled_sh_under": (filled_signed if action == "BUY"  else 0),
            "filled_sh_etf":   (abs(filled_signed) if action == "SELL" else 0),
            "notes": (
                f"HEDGE_{action} status={res.status} "
                f"correction_usd={trade_info['correction_usd']:+,.0f}"
            ),
        })
        n_traded += 1

    return fill_records, n_triggered, n_traded


# ---------------------------------------------------------------------------
# Phase 3 — Parallel hedge execution
# ---------------------------------------------------------------------------

def _hedge_worker(
    trade_info: dict,
    *,
    worker_idx: int,
    host: str,
    port: int,
    client_id: int,
    baseline: Dict[str, float],
    prices: Dict[str, float],
    prefer_delayed: bool,
    exec_cfg: dict,
    strategy_tag: str,
    run_date: str,
    limit_bps: float,
    timeout: float,
    max_retries: int,
    dry_run: bool,
    short_map: Dict[str, dict],
    cancel_service: CoordinatorCancelService,
    log_exposure_event,
    net_exposure_band: float,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
    log_lock: threading.Lock,
) -> Tuple[List[dict], bool]:
    """Execute a single Phase 3 hedge trade on its own IB connection.

    Returns (fill_records, was_traded).
    """
    ensure_thread_event_loop()
    if stop_requested():
        return [], False

    under      = trade_info["underlying"]
    symbol     = trade_info["symbol"]
    action     = trade_info["action"]
    qty        = trade_info["qty"]
    ref_px     = trade_info["ref_price"]
    target_gross = trade_info["target_gross"]

    try:
        ib_local = connect_ib(host, port, client_id + 300 + worker_idx)
    except Exception as e:
        tprint(f"[HEDGE][{under}] Worker IB connect failed: {e}")
        return [], False

    try:
        # Re-verify: re-read live position before trading
        ib_pos_now = current_ib_positions(ib_local)
        strat_now  = strategy_position_only(ib_pos_now, baseline)
        net_now, _ = compute_beta_adjusted_net_notional(
            strat_pos=strat_now, prices=prices, underlying=under,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
        )
        triggered_now, _ = compute_hedge_delta(
            net_notional=net_now, target_gross=target_gross,
            net_exposure_band=net_exposure_band,
        )
        if not triggered_now:
            tprint(
                f"[HEDGE][{under}] No longer triggered after re-read "
                f"(net={net_now:+,.0f}); skipping."
            )
            return [], False

        # Fresh price
        fresh_px = get_snapshot_price(ib_local, symbol, prefer_delayed=prefer_delayed)
        px = float(fresh_px or ref_px)
        with log_lock:
            prices[symbol] = px

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        px_under = float(prices.get(under) or 0.0)

        # FTP gate for new shorts
        if action == "SELL":
            sm    = short_map.get(symbol, {})
            avail = sm.get("available")
            if avail is not None and avail <= 0:
                tprint(f"[HEDGE][{under}] SKIP {symbol}: FTP available=0.")
                return [{
                    "filled_at": now, "run_date": run_date,
                    "strategy_tag": strategy_tag,
                    "pair_id": f"{under}__HEDGE",
                    "underlying": under, "etf": symbol,
                    "px_under": px_under, "px_etf": px,
                    "target_sh_under": 0, "target_sh_etf": 0,
                    "delta_sh_under": 0, "delta_sh_etf": -qty,
                    "filled_sh_under": 0, "filled_sh_etf": 0,
                    "notes": "HEDGE_SKIP_FTP_AVAIL0",
                }], False

        # Log pre-hedge state
        log_exposure_event(
            stage="HEDGE_PRE", pair_id=f"{under}__HEDGE",
            underlying=under,
            etf=(symbol if etf_to_under.get(symbol) == under else ""),
            symbol=symbol, delta_sh=0, filled_sh=0, trade=None,
        )

        order_ref = f"{strategy_tag}|{under}__HEDGE|{symbol}|{action}"
        res = execute_leg(
            ib=ib_local, symbol=symbol, action=action, qty=qty,
            ref_price=px, bps=limit_bps, order_ref=order_ref,
            exec_cfg=exec_cfg, timeout=timeout, max_retries=max_retries,
            dry_run=dry_run, context=f"{under}|HEDGE",
            cancel_service=cancel_service,
        )

        filled_signed = int(res.filled) if action == "BUY" else -int(res.filled)

        # Graceful 201 handling
        if res.status == "SHORT_BLOCKED":
            tprint(f"[HEDGE][{under}] {symbol} SHORT_BLOCKED (201). Logging and continuing.")
            log_exposure_event(
                stage="POST_HEDGE", pair_id=f"{under}__HEDGE",
                underlying=under, etf=symbol, symbol=symbol,
                delta_sh=-qty, filled_sh=0, trade=res.trade,
            )
            return [{
                "filled_at": now, "run_date": run_date,
                "strategy_tag": strategy_tag,
                "pair_id": f"{under}__HEDGE",
                "underlying": under, "etf": symbol,
                "px_under": px_under, "px_etf": px,
                "target_sh_under": 0, "target_sh_etf": 0,
                "delta_sh_under": 0, "delta_sh_etf": -qty,
                "filled_sh_under": 0, "filled_sh_etf": 0,
                "notes": f"HEDGE_SHORT_BLOCKED_201 msg={res.error_msg}",
            }], False

        log_exposure_event(
            stage="POST_HEDGE", pair_id=f"{under}__HEDGE",
            underlying=under,
            etf=(symbol if etf_to_under.get(symbol) == under else ""),
            symbol=symbol,
            delta_sh=(-qty if action == "SELL" else qty),
            filled_sh=filled_signed, trade=res.trade,
        )

        is_etf_trade = (etf_to_under.get(symbol) == under)
        fill_rec = {
            "filled_at": now, "run_date": run_date,
            "strategy_tag": strategy_tag,
            "pair_id": f"{under}__HEDGE",
            "underlying": under,
            "etf": (symbol if is_etf_trade else ""),
            "px_under": px_under,
            "px_etf":   (px if is_etf_trade else 0.0),
            "target_sh_under": 0, "target_sh_etf": 0,
            "delta_sh_under":  (qty if action == "BUY"  else 0),
            "delta_sh_etf":    (qty if action == "SELL" else 0),
            "filled_sh_under": (filled_signed if action == "BUY"  else 0),
            "filled_sh_etf":   (abs(filled_signed) if action == "SELL" else 0),
            "notes": (
                f"HEDGE_{action} status={res.status} "
                f"correction_usd={trade_info['correction_usd']:+,.0f}"
            ),
        }
        return [fill_rec], True

    finally:
        try:
            ib_local.disconnect()
        except Exception:
            pass


def execute_hedge_pass_parallel(
    *,
    hedge_trades: List[dict],
    host: str,
    port: int,
    client_id: int,
    baseline: Dict[str, float],
    prices: Dict[str, float],
    prefer_delayed: bool,
    exec_cfg: dict,
    strategy_tag: str,
    run_date: str,
    limit_bps: float,
    timeout: float,
    max_retries: int,
    dry_run: bool,
    parallel_n: int,
    short_map: Dict[str, dict],
    cancel_service: CoordinatorCancelService,
    log_exposure_event,
    net_exposure_band: float,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
    log_lock: threading.Lock,
) -> Tuple[List[dict], int, int]:
    """Execute Phase 3 hedge trades in parallel (one IB connection per worker).

    Returns (fill_records, n_triggered, n_traded).
    """
    if not hedge_trades:
        return [], 0, 0

    all_fills: List[dict] = []
    n_triggered = len(hedge_trades)
    n_traded    = 0
    n_workers   = min(parallel_n, len(hedge_trades))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _hedge_worker, t,
                worker_idx=i,
                host=host, port=port, client_id=client_id,
                baseline=baseline, prices=prices,
                prefer_delayed=prefer_delayed, exec_cfg=exec_cfg,
                strategy_tag=strategy_tag, run_date=run_date,
                limit_bps=limit_bps, timeout=timeout, max_retries=max_retries,
                dry_run=dry_run, short_map=short_map,
                cancel_service=cancel_service,
                log_exposure_event=log_exposure_event,
                net_exposure_band=net_exposure_band,
                etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                log_lock=log_lock,
            ): t
            for i, t in enumerate(hedge_trades)
        }
        for fut in as_completed(futures):
            try:
                fills, was_traded = fut.result()
                all_fills.extend(fills)
                if was_traded:
                    n_traded += 1
            except Exception as ex:
                tprint(f"[HEDGE] Worker raised: {ex}")

    return all_fills, n_triggered, n_traded


# ---------------------------------------------------------------------------
# Phase summary printer
# ---------------------------------------------------------------------------

def print_phase_summary(
    *,
    phase: str,
    n_checked: int,
    n_triggered: int,
    n_traded: int,
    net_by_underlying: Dict[str, float],
    target_gross_by_underlying: Dict[str, float],
    net_exposure_band: float,
) -> None:
    tprint(f"\n{'='*78}")
    tprint(f"  {phase}")
    tprint(f"  checked={n_checked}  triggered={n_triggered}  traded={n_traded}")
    tprint(f"{'='*78}")
    if not net_by_underlying:
        tprint("  (no data)")
        return

    tprint(f"  {'UNDERLYING':<15} {'NET':>12} {'TGT_GROSS':>12} {'NET_PCT':>8}  STATUS")
    tprint(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*8}  {'-'*6}")
    for under in sorted(net_by_underlying):
        net = net_by_underlying[under]
        tgt = target_gross_by_underlying.get(under, 0.0)
        pct = (net / tgt * 100.0) if tgt > 0 else 0.0
        status = "OK" if abs(pct) <= net_exposure_band * 100.0 else "WARN"
        tprint(f"  {under:<15} {net:>12,.0f} {tgt:>12,.0f} {pct:>7.1f}%  {status}")
    tprint("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    signal.signal(signal.SIGINT, handle_sigint)

    parser = argparse.ArgumentParser(description="Hybrid three-phase strategy rebalancer.")
    parser.add_argument("--dry-run",      action="store_true", help="No orders placed.")
    parser.add_argument("--run-date",     default=None,        help="Override run date (YYYY-MM-DD).")
    parser.add_argument("--skip-phase-1", action="store_true", help="Skip cleanup pass.")
    parser.add_argument("--skip-phase-2", action="store_true", help="Skip establish pass.")
    parser.add_argument("--skip-phase-3", action="store_true", help="Skip hedge pass.")
    args = parser.parse_args()

    CONFIG_YML = Path("config/strategy_config.yml")
    if not CONFIG_YML.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_YML}")

    cfg       = yaml.safe_load(CONFIG_YML.read_text(encoding="utf-8")) or {}
    ibkr_cfg  = cfg.get("ibkr", {})      or {}
    strat_cfg = cfg.get("strategy", {})  or {}
    paths_cfg = cfg.get("paths", {})     or {}
    exec_cfg  = cfg.get("execution", {}) or {}
    port_cfg  = cfg.get("portfolio", {}) or {}
    reb_cfg   = port_cfg.get("rebalance", {}) or {}

    # IBKR connection
    host           = str(ibkr_cfg.get("host", "127.0.0.1"))
    port           = int(ibkr_cfg.get("port", 7496))
    client_id      = int(ibkr_cfg.get("client_id", 41))
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))

    exec_cfg = dict(exec_cfg)
    exec_cfg["prefer_delayed"] = prefer_delayed

    limit_bps   = float(exec_cfg.get("limit_bps", 25.0))
    timeout     = float(exec_cfg.get("timeout_sec", 90))
    max_retries = int(exec_cfg.get("max_retries", 3))
    short_first = bool(exec_cfg.get("short_first", True))
    parallel_n  = max(1, int(exec_cfg.get("parallel_n", 10)))
    auto_approve = bool(exec_cfg.get("auto_approve", False))

    dry_run = args.dry_run or bool(exec_cfg.get("dry_run", False))
    if dry_run:
        tprint("[DRY_RUN] Enabled. No orders will be placed.")

    # Strategy params
    strategy_tag = str(strat_cfg.get("tag", "")).strip()
    if not strategy_tag:
        raise ValueError("Missing strategy.tag in config.")
    gross_leverage = float(strat_cfg.get("gross_leverage", 4.0))
    blacklist      = load_blacklist(cfg)

    # Rebalance params
    net_exposure_band       = float(reb_cfg.get("net_exposure_band", 0.10))
    min_trade_usd           = float(reb_cfg.get("min_trade_usd", 500.0))
    establish_threshold_usd = float(reb_cfg.get("establish_threshold_usd", 100.0))

    run_date = args.run_date or today_str()
    rb_dir   = rebalance_dir(run_date)
    exposure_csv   = rb_dir / "exposure_log.csv"
    exposure_jsonl = rb_dir / "exposure_log.jsonl"
    fills_path     = rb_dir / "fills.csv"

    # Flow ETFs (excluded from Phase 3)
    flow_etfs: Set[str] = set(
        norm_sym(x)
        for x in (
            port_cfg.get("sleeves", {})
                    .get("flow_program", {})
                    .get("universe", {})
                    .get("shorts", []) or []
        )
    )
    tprint(f"[FLOW] {len(flow_etfs)} flow-program ETFs excluded from hedge phase.")

    # Load plan
    plan_path           = resolve_plan_path(run_date, paths_cfg)
    plan, hedgeable_df  = load_plan(plan_path, strategy_tag, flow_etfs)
    tprint(
        f"[PLAN] {len(plan)} total rows; "
        f"{len(hedgeable_df)} hedgeable (core+whitelist, non-purgatory, non-flow)."
    )

    # Load screened universe
    screened_csv = Path(paths_cfg.get("screened_csv", "data/etf_screened_today.csv"))
    if not screened_csv.exists():
        raise FileNotFoundError(f"Screened CSV not found: {screened_csv}")

    screened = pd.read_csv(screened_csv)
    screened["ETF"]        = screened["ETF"].astype(str).map(norm_sym)
    screened["Underlying"] = screened["Underlying"].astype(str).map(norm_sym)

    purgatory_etfs = build_purgatory_set(screened)
    borrow_by_etf  = build_borrow_by_etf(screened)

    # Build ETF -> underlying + beta maps from screened CSV
    _etf_to_under_raw, _etf_to_beta_raw = load_etf_beta_map(screened_csv)
    etf_to_under: Dict[str, str]   = {norm_sym(k): norm_sym(v)   for k, v in _etf_to_under_raw.items()}
    etf_to_beta:  Dict[str, float] = {norm_sym(k): float(v)      for k, v in _etf_to_beta_raw.items()}

    # leverage_by_etf_all for execute_cleanup_trades_parallel
    leverage_by_etf_all: Dict[str, float] = {
        norm_sym(str(r["ETF"])): float(r["Beta"])
        for _, r in screened.iterrows()
        if str(r["ETF"]).strip().upper() not in ("", "NAN")
    }

    baseline     = load_baseline_qty(Path(paths_cfg.get("baseline_csv", "data/baseline_snapshot.csv")))
    plan_etfs    = sorted(set(plan["ETF"].astype(str).str.upper()))

    # Short availability precheck
    try:
        short_map: Dict[str, dict] = fetch_ibkr_short_availability_map(plan_etfs)
        tprint(f"[SHORT] Loaded availability for {len(short_map)}/{len(plan_etfs)} plan ETFs.")
    except Exception as ex:
        tprint(f"[SHORT] WARNING: FTP precheck failed ({ex}); continuing without it.")
        short_map = {}

    # Connect IBKR coordinator
    tprint(f"[IB] Connecting coordinator: {host}:{port}  clientId={client_id}")
    ib = connect_ib(host, port, client_id, coordinator=True)

    cancel_service = CoordinatorCancelService(host=host, port=port)
    cancel_service.start()
    tprint("[CANCEL_COORD] Started cancel coordinator.")

    log_lock:  threading.Lock = threading.Lock()
    prices:    Dict[str, float] = {}
    all_fills: List[dict] = []

    try:
        ib_pos    = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)
        tprint(f"[POS] IB symbols={len(ib_pos)}, strategy-only={len(strat_pos)}")

        # ------------------------------------------------------------------
        # Logging closure — thread-safe, captures ib/baseline/prices by ref
        # ------------------------------------------------------------------
        def log_exposure_event(
            *, stage: str, pair_id: str, underlying: str, etf: str,
            symbol: str, delta_sh: int, filled_sh: int, trade,
        ) -> None:
            ib_pos_now  = current_ib_positions(ib)
            strat_now   = strategy_position_only(ib_pos_now, baseline)
            port_nots   = compute_portfolio_notionals(
                {k: int(round(float(v))) for k, v in strat_now.items()}, prices
            )
            mark_px    = float(prices[symbol]) if (prices.get(symbol) and symbol != "PORTFOLIO") else None
            fill_px    = safe_avg_fill_price(trade)
            used_px    = fill_px if fill_px is not None else mark_px
            pos_sh     = int(round(float(strat_now.get(symbol, 0.0)))) if symbol != "PORTFOLIO" else 0
            pos_notl   = (pos_sh * mark_px) if (mark_px and symbol != "PORTFOLIO") else None
            delta_notl = (int(filled_sh) * float(used_px)) if used_px else None

            row = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_date": run_date, "strategy_tag": strategy_tag,
                "stage": stage, "pair_id": pair_id,
                "underlying": underlying, "etf": etf, "symbol": symbol,
                "delta_sh": int(delta_sh), "filled_sh": int(filled_sh),
                "fill_avg_px": fill_px, "mark_px": mark_px,
                "delta_notional": delta_notl, "pos_sh": pos_sh,
                "pos_notional": pos_notl,
                **port_nots,
            }
            with log_lock:
                append_csv_row(exposure_csv, row)
                append_jsonl(exposure_jsonl, row)

        # ==================================================================
        # PHASE 1 — Cleanup
        # ==================================================================
        if not args.skip_phase_1:
            tprint("\n" + "=" * 60)
            tprint("  PHASE 1 — CLEANUP")
            tprint("=" * 60)

            cleanup_trades, impacted_under = build_cleanup_trades_to_match_plan(
                ib=ib,
                baseline=baseline,
                plan=plan,
                screened=screened,
                prices=prices,
                prefer_delayed=prefer_delayed,
                blacklist=blacklist,
                etf_to_under_all=etf_to_under,
                borrow_by_etf=borrow_by_etf,
                purgatory_etfs=purgatory_etfs,
                flow_short_etfs=flow_etfs,
            )
            print_cleanup_trade_list(cleanup_trades)

            if cleanup_trades:
                if auto_approve:
                    do_cleanup = True
                else:
                    ans = input("[CLEANUP] Approve executing these trades? (y/n): ").strip().lower()
                    do_cleanup = (ans == "y")

                if do_cleanup:
                    syms = [t["symbol"] for t in cleanup_trades] + list(impacted_under)
                    contract_cache = build_contract_cache(ib, syms)
                    execute_cleanup_trades_parallel(
                        approved_trades=cleanup_trades,
                        impacted_underlyings=impacted_under,
                        ib=ib,
                        baseline=baseline,
                        prices=prices,
                        prefer_delayed=prefer_delayed,
                        etf_to_under_all=etf_to_under,
                        leverage_by_etf_all=leverage_by_etf_all,
                        exec_cfg=exec_cfg,
                        strategy_tag=strategy_tag,
                        limit_bps=limit_bps,
                        timeout=timeout,
                        max_retries=max_retries,
                        dry_run=dry_run,
                        parallel_n=parallel_n,
                        contract_cache=contract_cache,
                        cfg=cfg,
                        cancel_service=cancel_service,
                    )
                    _sync_positions_after_external_trades(ib, timeout_s=15.0)
                else:
                    tprint("[CLEANUP] Skipped by user.")
            else:
                tprint("[CLEANUP] No positions to clean up.")
        else:
            tprint("[PHASE 1] Skipped (--skip-phase-1).")

        # Refresh positions + prefetch prices
        ib_pos    = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)

        # Only prefetch symbols that are in the plan or screened universe;
        # strat_pos may contain non-tradeable legacy holdings (CVRs, escrows, etc.)
        screened_universe: Set[str] = (
            set(etf_to_under.keys()) | set(etf_to_under.values())
        )
        all_symbols: Set[str] = (
            set(plan["ETF"].tolist()) | set(plan["Underlying"].tolist())
            | {s for s, sh in strat_pos.items()
               if float(sh) != 0.0 and s in screened_universe}
        )
        tprint(f"[PRICES] Prefetching {len(all_symbols)} symbols...")
        for sym in sorted(all_symbols):
            if stop_requested():
                break
            if sym not in prices:
                try:
                    prices[sym] = get_snapshot_price(ib, sym, prefer_delayed=prefer_delayed)
                except RuntimeError as e:
                    tprint(f"[PRICES] WARNING: {e} — skipping {sym}")
                    continue

        if stop_requested():
            tprint("[SHUTDOWN] Exiting after Phase 1.")
            return

        # ==================================================================
        # PHASE 2 — Establish new positions
        # ==================================================================
        if not args.skip_phase_2 and not plan.empty:
            tprint("\n" + "=" * 60)
            tprint("  PHASE 2 — ESTABLISH NEW POSITIONS")
            tprint("=" * 60)

            establish_trades = build_establish_trades(
                plan=plan,
                strat_pos=strat_pos,
                prices=prices,
                purgatory_etfs=purgatory_etfs,
                flow_etfs=flow_etfs,
                etf_to_under=etf_to_under,
                min_trade_usd=min_trade_usd,
                establish_threshold_usd=establish_threshold_usd,
            )
            if establish_trades:
                fills2 = execute_establish_parallel(
                    establish_trades=establish_trades,
                    host=host, port=port, client_id=client_id,
                    baseline=baseline, prices=prices,
                    prefer_delayed=prefer_delayed, exec_cfg=exec_cfg,
                    strategy_tag=strategy_tag, run_date=run_date,
                    limit_bps=limit_bps, timeout=timeout, max_retries=max_retries,
                    dry_run=dry_run, parallel_n=parallel_n,
                    short_map=short_map, cancel_service=cancel_service,
                    log_exposure_event=log_exposure_event,
                    short_first=short_first, log_lock=log_lock,
                )
                all_fills.extend(fills2)
                _sync_positions_after_external_trades(ib, timeout_s=10.0)
            else:
                tprint("[ESTABLISH] No new pairs to establish.")
        else:
            tprint("[PHASE 2] Skipped.")

        if stop_requested():
            tprint("[SHUTDOWN] Exiting after Phase 2.")
            return

        # Refresh positions for Phase 3
        ib_pos    = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)

        # ==================================================================
        # PHASE 3 — Directional hedge rebalance
        # ==================================================================
        if not args.skip_phase_3 and not hedgeable_df.empty:
            tprint("\n" + "=" * 60)
            tprint("  PHASE 3 — DIRECTIONAL HEDGE REBALANCE")
            tprint("=" * 60)

            account_equity = get_account_equity(ib)
            tprint(f"[EQUITY] NetLiquidation = ${account_equity:,.0f}")
            tprint(
                f"[HEDGE]  net_exposure_band={net_exposure_band*100:.0f}%  "
                f"gross_leverage={gross_leverage}x  "
                f"min_trade_usd=${min_trade_usd:,.0f}"
            )

            # Pre-hedge snapshot (for summary)
            pre_net: Dict[str, float] = {}
            pre_tgt: Dict[str, float] = {}
            for u in sorted(hedgeable_df["Underlying"].unique()):
                net, _ = compute_beta_adjusted_net_notional(
                    strat_pos=strat_pos, prices=prices, underlying=u,
                    etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                )
                tgt = compute_target_gross_per_underlying(
                    underlying=u, plan=hedgeable_df,
                    account_equity=account_equity, gross_leverage=gross_leverage,
                )
                pre_net[u] = net
                pre_tgt[u] = tgt

            hedge_trades = build_hedge_trades(
                hedgeable_plan=hedgeable_df,
                strat_pos=strat_pos, prices=prices,
                account_equity=account_equity, gross_leverage=gross_leverage,
                net_exposure_band=net_exposure_band, min_trade_usd=min_trade_usd,
                etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
            )

            fills3, n_triggered, n_traded = execute_hedge_pass_parallel(
                hedge_trades=hedge_trades,
                host=host, port=port, client_id=client_id,
                baseline=baseline, prices=prices,
                prefer_delayed=prefer_delayed, exec_cfg=exec_cfg,
                strategy_tag=strategy_tag, run_date=run_date,
                limit_bps=limit_bps, timeout=timeout, max_retries=max_retries,
                dry_run=dry_run, parallel_n=parallel_n,
                short_map=short_map, cancel_service=cancel_service,
                log_exposure_event=log_exposure_event,
                net_exposure_band=net_exposure_band,
                etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                log_lock=log_lock,
            )
            all_fills.extend(fills3)

            # Post-hedge net for summary
            ib_pos_final    = current_ib_positions(ib)
            strat_pos_final = strategy_position_only(ib_pos_final, baseline)
            post_net: Dict[str, float] = {}
            for u in pre_net:
                net, _ = compute_beta_adjusted_net_notional(
                    strat_pos=strat_pos_final, prices=prices, underlying=u,
                    etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                )
                post_net[u] = net

            print_phase_summary(
                phase="PHASE 3 — DIRECTIONAL HEDGE",
                n_checked=len(pre_net),
                n_triggered=n_triggered,
                n_traded=n_traded,
                net_by_underlying=post_net,
                target_gross_by_underlying=pre_tgt,
                net_exposure_band=net_exposure_band,
            )
        else:
            tprint("[PHASE 3] Skipped.")

        # Final portfolio snapshot
        log_exposure_event(
            stage="FINAL", pair_id="PORTFOLIO",
            underlying="", etf="", symbol="PORTFOLIO",
            delta_sh=0, filled_sh=0, trade=None,
        )

        if all_fills:
            append_fills(all_fills, fills_path)
            tprint(f"[FILLS] Wrote {len(all_fills)} fill records -> {fills_path}")

        tprint("\n[DONE] Rebalance complete.")

    finally:
        try:
            cancel_service.stop()
        except Exception:
            pass
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
