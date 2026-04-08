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
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
    configure_ib_error_log_filter,
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


def configure_market_data_mode(ib: IB, prefer_delayed: bool) -> None:
    """
    Configure market data mode once on the coordinator connection.
    Delayed mode avoids live entitlement warnings (10089) and is faster
    when live subscriptions are incomplete.
    """
    data_type = 3 if prefer_delayed else 1
    mode = "DELAYED(3)" if prefer_delayed else "LIVE(1)"
    try:
        ib.reqMarketDataType(data_type)
        setattr(ib, "_ls_algo_market_data_type", data_type)
        tprint(f"[PRICES] Market data mode set to {mode}.")
    except Exception as ex:
        tprint(f"[PRICES] WARNING: Failed to set market data mode {mode}: {ex}")


def _read_snapshot_ticker_price(ticker) -> Optional[float]:
    bid = safe_price(getattr(ticker, "bid", None)) or safe_price(getattr(ticker, "delayedBid", None))
    ask = safe_price(getattr(ticker, "ask", None)) or safe_price(getattr(ticker, "delayedAsk", None))
    last = safe_price(getattr(ticker, "last", None)) or safe_price(getattr(ticker, "delayedLast", None))
    close = safe_price(getattr(ticker, "close", None)) or safe_price(getattr(ticker, "delayedClose", None))
    mkt = safe_price(ticker.marketPrice())
    if bid and ask:
        return (bid + ask) / 2.0
    return last or close or mkt


def prefetch_prices_batched(
    *,
    ib: IB,
    symbols: Iterable[str],
    prices: Dict[str, float],
    prefer_delayed: bool,
    batch_size: int = 75,
) -> Tuple[int, int]:
    """
    Prefetch prices using batched reqTickers snapshots. Falls back to
    get_snapshot_price per symbol for unresolved names.
    Returns (resolved_count, requested_count).
    """
    requested = sorted({norm_sym(s) for s in symbols if str(s).strip() and norm_sym(s) not in prices})
    if not requested:
        return 0, 0

    resolved = 0
    batch_size = max(1, int(batch_size))

    for i in range(0, len(requested), batch_size):
        if stop_requested():
            break

        chunk = requested[i : i + batch_size]
        contracts = [make_stock(sym) for sym in chunk]

        qualified = []
        try:
            qualified = list(ib.qualifyContracts(*contracts) or [])
        except Exception:
            for c in contracts:
                try:
                    q = ib.qualifyContracts(c)
                    if q:
                        qualified.extend(q)
                except Exception:
                    continue

        if qualified:
            try:
                tickers = ib.reqTickers(*qualified)
            except Exception:
                tickers = []

            for tk in tickers:
                sym = norm_sym(str(getattr(getattr(tk, "contract", None), "symbol", "")))
                px = _read_snapshot_ticker_price(tk)
                if sym and px is not None:
                    prices[sym] = float(px)
                    resolved += 1

        # Fallback path for symbols still unresolved in this batch.
        for sym in chunk:
            if sym in prices:
                continue
            try:
                prices[sym] = float(get_snapshot_price(ib, sym, prefer_delayed=prefer_delayed))
                resolved += 1
            except RuntimeError as e:
                tprint(f"[PRICES] WARNING: {e} — skipping {sym}")

    return resolved, len(requested)


def blocked_short_symbols_from_fills(fill_records: List[dict]) -> Set[str]:
    """
    Extract ETF symbols that failed shorting this run (201/FTP=0) so
    later hedge passes don't keep retrying the same impossible short.
    """
    blocked: Set[str] = set()
    for rec in fill_records:
        note = str(rec.get("notes", "") or "")
        if ("HEDGE_SHORT_BLOCKED_201" not in note) and ("HEDGE_SKIP_FTP_AVAIL0" not in note):
            continue
        etf = norm_sym(str(rec.get("etf", "") or ""))
        if etf:
            blocked.add(etf)
    return blocked


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

        target_under_sh = target_shares_from_usd(abs(long_usd), px_under)
        # short_usd is stored as negative notional in proposed_trades.csv.
        # Convert to absolute shares for leg construction (SELL action is
        # encoded separately), otherwise target_etf_sh stays <= 0 and the
        # establish short leg is silently skipped.
        target_etf_sh   = target_shares_from_usd(abs(short_usd), px_etf)

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

        short_blocked = False
        short_requested_sh = 0
        short_filled_sh = 0

        for action, sym, qty, px, leg_type in legs:
            if stop_requested():
                break

            # If the short leg was fully blocked, skip the long to avoid
            # creating a naked long.  Phase 3's 3a/3b will catch any
            # remaining imbalance on the next hedge pass.
            if action == "BUY" and short_blocked:
                tprint(f"[ESTABLISH][{under}] Skipping long — short leg was blocked.")
                continue

            # If the ETF short only partially filled, scale the underlying
            # leg to match actual short coverage instead of opening the full
            # long target immediately.
            if action == "BUY" and leg_type == "under" and target_etf_sh > 0 and short_requested_sh > 0:
                fill_ratio = min(1.0, max(0.0, float(short_filled_sh) / float(short_requested_sh)))
                scaled_qty = int(math.floor(target_under_sh * fill_ratio))
                if scaled_qty <= 0:
                    tprint(
                        f"[ESTABLISH][{under}] Skipping long — short fill ratio={fill_ratio:.1%} "
                        f"({short_filled_sh}/{short_requested_sh}) gives 0 scaled shares."
                    )
                    continue
                if scaled_qty < qty:
                    tprint(
                        f"[ESTABLISH][{under}] Scaling long leg to short fill ratio={fill_ratio:.1%}: "
                        f"{qty} -> {scaled_qty} shares."
                    )
                    qty = scaled_qty

            # FTP gate for short leg
            if action == "SELL":
                sm    = short_map.get(sym, {})
                avail = sm.get("available")
                if avail is not None and avail <= 0:
                    tprint(f"[ESTABLISH][{under}] SKIP {sym}: FTP available=0.")
                    short_blocked = True
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

            # If the short filled zero shares (blocked, timed out, or failed),
            # skip the long leg to prevent naked longs.
            # Partial fills (res.filled > 0) are OK — Phase 3 handles the residual.
            if action == "SELL" and int(res.filled) == 0:
                short_blocked = True
            if action == "SELL" and leg_type == "etf":
                short_requested_sh += int(qty)
                short_filled_sh += int(abs(res.filled))

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

    import time as _time

    all_fills: List[dict] = []

    # Cap concurrent workers to avoid TWS connection limit (~8 max),
    # matching the Phase 3 hedge pattern.
    MAX_ESTABLISH_WORKERS = 6
    n_workers = min(parallel_n, len(establish_trades), MAX_ESTABLISH_WORKERS)
    stagger_delay = 1.5   # seconds between submitting each worker

    tprint(
        f"[ESTABLISH] Launching {len(establish_trades)} workers with "
        f"{n_workers} concurrent (stagger={stagger_delay}s)"
    )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for i, t in enumerate(establish_trades):
            if stop_requested():
                break
            if i > 0:
                _time.sleep(stagger_delay)
            fut = pool.submit(
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
            )
            futures[fut] = t

        for fut in as_completed(futures):
            try:
                all_fills.extend(fut.result())
            except Exception as ex:
                tprint(f"[ESTABLISH] Worker raised: {ex}")

    return all_fills


# ---------------------------------------------------------------------------
# Phase 3 — Directional hedge math
# ---------------------------------------------------------------------------

def has_blind_etfs(
    *,
    underlying: str,
    strat_pos: Dict[str, float],
    prices: Dict[str, float],
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
    flow_etfs: Optional[Set[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Check if an underlying has any ETFs that are:
        1. Positioned (non-zero shares in strat_pos)
        2. Positive beta (core pair, not flow/inverse)
        3. Cannot be priced (px <= 0)

    When this returns True, compute_beta_adjusted_net_notional is blind
    to those ETFs' contribution, and any hedge/reconcile trade on the
    underlying risks creating a naked position.

    Returns (is_blind, list_of_blind_etf_symbols).
    """
    if flow_etfs is None:
        flow_etfs = set()

    blind: List[str] = []
    for sym, u in etf_to_under.items():
        if u != underlying:
            continue
        if sym in flow_etfs:
            continue
        beta = etf_to_beta.get(sym, 1.0)
        if beta < 0:
            continue
        sh = float(strat_pos.get(sym, 0.0))
        if sh == 0.0:
            continue
        px = float(prices.get(sym) or 0.0)
        if px <= 0.0:
            blind.append(sym)

    return len(blind) > 0, blind


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

    Negative-beta ETFs (inverse products) are EXCLUDED entirely.
    Their exposure belongs to bucket 3 (flow program) and must not
    influence core pair hedge decisions.

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

        # Skip negative-beta ETFs — inverse products whose exposure
        # is managed separately in the flow program (bucket 3).
        if is_etf and beta < 0:
            continue

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
    net_trigger_long_pct: float,   # 0.02
    net_trigger_short_pct: float,  # 0.01
    net_target_long_pct: float,    # 0.01
    net_target_short_pct: float,   # 0.00
) -> Tuple[bool, float]:
    if target_gross <= 0.0:
        return False, 0.0

    if net_notional > 0:
        # Net long — need to add shorts
        trigger = net_trigger_long_pct * target_gross
        if net_notional <= trigger:
            return False, 0.0
        target_net = net_target_long_pct * target_gross
    else:
        # Net short — need to add longs
        trigger = net_trigger_short_pct * target_gross
        if abs(net_notional) <= trigger:
            return False, 0.0
        target_net = -net_target_short_pct * target_gross

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
    short_map: Optional[Dict[str, dict]] = None,
    blocked_short_etfs: Optional[Set[str]] = None,
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

    blocked_short_etfs = set(blocked_short_etfs or set())

    if correction_usd < 0:
        # Add to ETF short leg
        # Include plan ETFs AND ETFs already held short (even if dropped from today's plan)
        plan_etfs = set(plan["ETF"].astype(str))
        etf_candidates = [
            sym for sym, u in etf_to_under.items()
            if u == underlying
            and (sym in plan_etfs or float(strat_pos.get(sym, 0.0)) < 0)
        ]
        if blocked_short_etfs:
            etf_candidates = [s for s in etf_candidates if s not in blocked_short_etfs]
        if not etf_candidates:
            if blocked_short_etfs:
                tprint(
                    f"[HEDGE][{underlying}] No ETF candidates after blocked-short filter; "
                    "deferring to reconciliation."
                )
            else:
                tprint(f"[HEDGE][{underlying}] No ETF candidates (plan or positioned); cannot add to short.")
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
            # Cap to FTP available shares so we don't attempt more than
            # the borrow desk can supply.  The remaining imbalance will be
            # caught on the next hedge pass.
            if short_map:
                avail = (short_map.get(etf) or {}).get("available")
                if avail is not None and avail > 0:
                    qty = min(qty, int(avail))
                elif avail is not None and avail <= 0:
                    tprint(f"[HEDGE][{underlying}] {etf} FTP available=0; skipping.")
                    continue
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
    net_trigger_long_pct: float,
    net_trigger_short_pct: float,
    net_target_long_pct: float,
    net_target_short_pct: float,
    min_trade_usd: float,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
    short_map: Optional[Dict[str, dict]] = None,
    blocked_short_etfs: Optional[Set[str]] = None,
    flow_etfs: Optional[Set[str]] = None,
    blacklist: Optional[Set[str]] = None,
) -> List[dict]:
    """
    Iterate over ALL underlyings that have strategy positions — not just
    those in hedgeable_plan.  This catches orphaned positions from Phase 1
    cleanup, ETF delistings, forced short recalls, and any other event
    that removes one leg of a pair.

    For each underlying:
        1. Compute beta-adjusted net notional from live positions
        2. Compute target gross from live equity (0 if not in plan)
        3. If the underlying is the ONLY remaining leg (no ETF exposure):
           → generate a full close trade (orphan_close=True)
        4. Otherwise: normal threshold check + resolve single leg

    Negative-beta / flow-program ETFs are excluded so bucket-3 exposure
    is not disturbed.  Blacklisted underlyings are skipped entirely
    (they should not be traded by the strategy).

    Returns list of trade dicts ordered by abs(correction_usd) descending
    (largest imbalances traded first).
    """
    if flow_etfs is None:
        flow_etfs = set()
    if blacklist is None:
        blacklist = set()
    else:
        blacklist = set(blacklist)
    blocked_short_etfs = set(blocked_short_etfs or set())

    plan_underlyings: Set[str] = set(
        hedgeable_plan["Underlying"].unique()
    ) if not hedgeable_plan.empty else set()

    # ── Build the full set of underlyings with any strategy position ──
    # Include: direct underlying positions + underlyings mapped from ETFs
    # Exclude: flow ETFs and negative-beta (inverse) ETFs — bucket 3
    # Exclude: blacklisted underlyings (not managed by Phase 3)
    positioned_underlyings: Set[str] = set()
    for sym, sh in strat_pos.items():
        if float(sh) == 0.0:
            continue
        # Is this symbol an ETF?  Add its underlying unless it's
        # a flow ETF or a negative-beta (inverse) product.
        u = etf_to_under.get(sym)
        if u and sym not in flow_etfs:
            beta = etf_to_beta.get(sym, 1.0)
            if beta >= 0:
                positioned_underlyings.add(u)
        # Is this symbol an underlying itself?
        elif sym in set(etf_to_under.values()):
            positioned_underlyings.add(sym)

    # Remove blacklisted underlyings from the off-plan scan.
    # (plan_underlyings already excludes them via generate_trade_plan.)
    blacklisted_positioned = positioned_underlyings & blacklist
    if blacklisted_positioned:
        tprint(
            f"[HEDGE] Skipping {len(blacklisted_positioned)} blacklisted "
            f"positioned underlyings: {sorted(blacklisted_positioned)}"
        )
        positioned_underlyings -= blacklist

    underlyings = sorted(plan_underlyings | positioned_underlyings)
    trades: List[dict] = []

    # Diagnostic: warn about positioned symbols invisible to hedge math
    # (Actionable unmapped handling happens in detect_unmapped_positions
    #  before Phase 3; this is just a debug-level count.)
    all_mapped = set(etf_to_under.keys()) | set(etf_to_under.values())
    unmapped = {
        s for s, sh in strat_pos.items()
        if float(sh) != 0.0 and s not in all_mapped
    }
    if unmapped:
        tprint(
            f"[HEDGE] INFO: {len(unmapped)} positioned symbols not in ETF beta map "
            f"(CVRs, legacy, etc. — handled upstream if in screened CSV)."
        )

    for under in underlyings:
        if stop_requested():
            break

        in_plan = under in plan_underlyings

        # Guard: skip this underlying if we have no price for it.
        px_under = float(prices.get(under) or 0.0)
        if px_under <= 0.0:
            tprint(f"[HEDGE][{under}] WARNING: No price for underlying; skipping hedge check.")
            continue

        net_notional, actual_gross = compute_beta_adjusted_net_notional(
            strat_pos=strat_pos, prices=prices, underlying=under,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
        )

        # ── Check for positioned ETFs with no price ──────────────────
        # If any core ETF has shares but can't be priced, the net
        # notional is unreliable (blind ETF contribution is invisible).
        # We track this per-underlying but do NOT blanket-skip:
        #   - Priceable ETF trades are still safe (adding shorts always
        #     reduces net-long from the visible portion; worst case is
        #     mild over-hedge that self-corrects on the next run)
        #   - Underlying-side trades are BLOCKED (selling underlying
        #     based on inflated priceable-only net could create a
        #     naked short — the DKNG catastrophe)
        #   - Orphan detection already handles this correctly: blind
        #     ETFs with shares count as "has exposure", preventing
        #     false orphan classification
        is_blind, blind_syms = has_blind_etfs(
            underlying=under, strat_pos=strat_pos, prices=prices,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
            flow_etfs=flow_etfs,
        )
        if is_blind:
            tprint(
                f"[HEDGE][{under:15s}] BLIND: {blind_syms} positioned but "
                f"unpriceable — priceable-ETF trades only."
            )

        # ── Check if the underlying is the only remaining leg ────────
        # (all core ETF shorts gone — orphaned position)
        # Negative-beta (inverse) ETFs and flow ETFs are excluded from
        # this check — they belong to bucket 3 and don't constitute
        # a hedge for the underlying long.
        under_sh = float(strat_pos.get(under, 0.0))
        has_etf_exposure = False
        for sym, u in etf_to_under.items():
            if u != under:
                continue
            if sym in flow_etfs:
                continue
            beta = etf_to_beta.get(sym, 1.0)
            if beta < 0:
                continue
            if float(strat_pos.get(sym, 0.0)) != 0.0:
                has_etf_exposure = True
                break

        if not has_etf_exposure and under_sh != 0.0:
            # Orphaned: underlying is the only remaining non-flow leg.
            # Close the entire position to zero.
            # No min_trade_usd gate — orphans are risk-reduction trades
            # that should always execute regardless of size.
            close_notional = abs(under_sh * px_under)
            action = "SELL" if under_sh > 0 else "BUY"
            qty = abs(int(under_sh))

            label = "ORPHAN" if not in_plan else "ORPHAN(plan)"
            tprint(
                f"[HEDGE][{under:15s}] net={net_notional:>+12,.0f}  "
                f"tgt_gross={'0':>10s}  net%={'n/a':>6s}  "
                f"{label}: {action} {qty} to close (${close_notional:,.0f})"
            )

            if qty <= 0:
                continue

            trades.append({
                "underlying":        under,
                "symbol":            under,
                "action":            action,
                "qty":               qty,
                "ref_price":         px_under,
                "correction_usd":    -close_notional if action == "SELL" else close_notional,
                "net_notional_before": net_notional,
                "target_gross":      0.0,
                "orphan_close":      True,
            })
            continue

        # ── Normal hedge logic (has ETF exposure) ────────────────────
        target_gross = compute_target_gross_per_underlying(
            underlying=under, plan=hedgeable_plan,
            account_equity=account_equity, gross_leverage=gross_leverage,
        )
        # Use the larger of actual gross or plan-derived target gross as the
        # reference for the hedge band.  This prevents silently under-hedging
        # when actual positions exceed the plan's target allocation (e.g. a
        # pair that grew beyond its intended weight).
        ref_gross = max(actual_gross, target_gross)

        triggered, correction_usd = compute_hedge_delta(
            net_notional=net_notional, target_gross=ref_gross,
            net_trigger_long_pct=net_trigger_long_pct,
            net_trigger_short_pct=net_trigger_short_pct,
            net_target_long_pct=net_target_long_pct,
            net_target_short_pct=net_target_short_pct,
        )

        net_pct = (net_notional / ref_gross * 100.0) if ref_gross > 0 else 0.0
        status_tag = "TRIGGERED" if triggered else "ok"
        if not in_plan:
            status_tag += " (off-plan)"
        tprint(
            f"[HEDGE][{under:15s}] net={net_notional:>+12,.0f}  "
            f"tgt_gross={target_gross:>10,.0f}  net%={net_pct:>+6.1f}%  "
            f"{status_tag}"
        )

        if not triggered:
            continue
        # Off-plan corrections bypass min_trade_usd — these are
        # risk-reduction wind-down trades, not cost-optimised hedges.
        if in_plan and abs(correction_usd) < min_trade_usd:
            tprint(
                f"[HEDGE][{under}] correction={correction_usd:+,.0f} "
                f"< min_trade_usd={min_trade_usd}; skipping."
            )
            continue

        symbol, action, qty, ref_px = resolve_hedge_leg(
            underlying=under, correction_usd=correction_usd,
            strat_pos=strat_pos, prices=prices,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
            plan=hedgeable_plan, short_map=short_map,
            blocked_short_etfs=blocked_short_etfs,
        )
        if symbol is None or qty <= 0:
            tprint(f"[HEDGE][{under}] Could not resolve hedge leg; skipping.")
            continue

        # Block underlying-side trades when blind ETFs exist.
        # Priceable-only net overstates the imbalance (blind shorts are
        # invisible), so trading the underlying risks creating a naked
        # position.  Priceable ETF-side trades are safe — adding shorts
        # always reduces visible exposure.
        if is_blind and symbol == under:
            tprint(
                f"[HEDGE][{under}] BLIND: blocking {action} {qty} {under} "
                f"(underlying trade unsafe with unpriceable ETFs)."
            )
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
            "target_gross":      ref_gross,   # max(actual, plan) used for trigger
            "orphan_close":      False,
        })

    # Largest corrections first
    trades.sort(key=lambda t: abs(t["correction_usd"]), reverse=True)
    n_orphan = sum(1 for t in trades if t.get("orphan_close"))
    tprint(
        f"[HEDGE] Checked {len(underlyings)} underlyings "
        f"({len(underlyings) - len(plan_underlyings)} off-plan) -> "
        f"{len(trades)} hedge trades queued ({n_orphan} orphan closes)."
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
    net_trigger_long_pct: float,
    net_trigger_short_pct: float,
    net_target_long_pct: float,
    net_target_short_pct: float,
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
        is_orphan  = trade_info.get("orphan_close", False)

        # Re-verify: re-read live position before each trade
        ib_pos_now = current_ib_positions(ib)
        strat_now  = strategy_position_only(ib_pos_now, baseline)

        if is_orphan:
            # Orphan close: confirm position still exists, flatten entirely
            cur_sh = float(strat_now.get(symbol, 0.0))
            if (action == "SELL" and cur_sh <= 0) or (action == "BUY" and cur_sh >= 0):
                tprint(
                    f"[HEDGE][{under}] Orphan close: position already flat "
                    f"(sh={cur_sh:.0f}); skipping."
                )
                n_triggered -= 1
                continue
            qty = abs(int(cur_sh))
            if qty <= 0:
                n_triggered -= 1
                continue
        else:
            net_now, _ = compute_beta_adjusted_net_notional(
                strat_pos=strat_now, prices=prices, underlying=under,
                etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
            )
            triggered_now, _ = compute_hedge_delta(
                net_notional=net_now, target_gross=target_gross,
                net_trigger_long_pct=net_trigger_long_pct,
                net_trigger_short_pct=net_trigger_short_pct,
                net_target_long_pct=net_target_long_pct,
                net_target_short_pct=net_target_short_pct,
            )
            if not triggered_now:
                tprint(
                    f"[HEDGE][{under}] No longer triggered after re-read "
                    f"(net={net_now:+,.0f}); skipping."
                )
                n_triggered -= 1
                continue

            # Cap SELL-underlying to shares held — don't flip to short
            if action == "SELL" and symbol == under:
                cur_under_sh = float(strat_now.get(under, 0.0))
                if cur_under_sh > 0:
                    qty = min(qty, int(cur_under_sh))
                else:
                    tprint(
                        f"[HEDGE][{under}] Reconcile SELL but underlying "
                        f"sh={cur_under_sh:.0f}; cannot sell. Skipping."
                    )
                    continue

        # Fresh price
        fresh_px   = get_snapshot_price(ib, symbol, prefer_delayed=prefer_delayed)
        px         = float(fresh_px or ref_px)
        prices[symbol] = px

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        px_under = float(prices.get(under) or 0.0)

        # FTP gate for new shorts (ETF only — selling an existing
        # underlying long is not a short sale)
        if action == "SELL" and symbol != under:
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
            "delta_sh_under":  ((-qty if action == "SELL" else qty) if not is_etf_trade else 0),
            "delta_sh_etf":    (qty if (action == "SELL" and is_etf_trade) else 0),
            "filled_sh_under": (filled_signed if not is_etf_trade else 0),
            "filled_sh_etf":   (abs(filled_signed) if is_etf_trade else 0),
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
    net_trigger_long_pct: float,
    net_trigger_short_pct: float,
    net_target_long_pct: float,
    net_target_short_pct: float,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
    log_lock: threading.Lock,
) -> Tuple[List[dict], bool, bool]:
    """Execute a single Phase 3 hedge trade on its own IB connection.

    Returns (fill_records, was_traded, was_triggered).
    was_triggered=False means the trade was no longer needed after re-verification.
    """
    ensure_thread_event_loop()
    if stop_requested():
        return [], False, True        # count as triggered (shutdown, not re-check)

    under      = trade_info["underlying"]
    symbol     = trade_info["symbol"]
    action     = trade_info["action"]
    qty        = trade_info["qty"]
    ref_px     = trade_info["ref_price"]
    target_gross = trade_info["target_gross"]
    is_orphan  = trade_info.get("orphan_close", False)

    try:
        ib_local = connect_ib(host, port, client_id + 300 + worker_idx)
    except Exception as e:
        tprint(f"[HEDGE][{under}] Worker IB connect failed: {e}")
        return [], False, True        # connection failure, still counts as triggered

    try:
        # Re-verify: re-read live position and fresh price before trading.
        # Fetch fresh price FIRST so the net-notional calculation uses
        # current market, not a stale snapshot from build time.
        ib_pos_now = current_ib_positions(ib_local)
        strat_now  = strategy_position_only(ib_pos_now, baseline)

        fresh_px = get_snapshot_price(ib_local, symbol, prefer_delayed=prefer_delayed)
        px = float(fresh_px or ref_px)
        with log_lock:
            prices[symbol] = px

        # ── Orphan close: simplified re-verification ─────────────────
        # No threshold math — just confirm the position still exists,
        # then sell/buy ALL remaining shares to flatten.
        if is_orphan:
            cur_sh = float(strat_now.get(symbol, 0.0))
            if (action == "SELL" and cur_sh <= 0) or (action == "BUY" and cur_sh >= 0):
                tprint(
                    f"[HEDGE][{under}] Orphan close: position already flat "
                    f"(sh={cur_sh:.0f}); skipping."
                )
                return [], False, False
            qty = abs(int(cur_sh))
            if qty <= 0:
                return [], False, False
            tprint(f"[HEDGE][{under}] Orphan close re-verified: {action} {qty} {symbol}")

        else:
            # ── Normal hedge re-verification ─────────────────────────
            net_now, _ = compute_beta_adjusted_net_notional(
                strat_pos=strat_now, prices=prices, underlying=under,
                etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
            )
            triggered_now, correction_now = compute_hedge_delta(
                net_notional=net_now, target_gross=target_gross,
                net_trigger_long_pct=net_trigger_long_pct,
                net_trigger_short_pct=net_trigger_short_pct,
                net_target_long_pct=net_target_long_pct,
                net_target_short_pct=net_target_short_pct,
            )
            if not triggered_now:
                tprint(
                    f"[HEDGE][{under}] No longer triggered after re-read "
                    f"(net={net_now:+,.0f}); skipping."
                )
                return [], False, False   # was_triggered=False → decrement n_triggered

            # Check that the correction direction hasn't flipped since build time.
            if (correction_now < 0 and action != "SELL") or (correction_now > 0 and action != "BUY"):
                tprint(
                    f"[HEDGE][{under}] Correction direction flipped "
                    f"(was {action}, now {'SELL' if correction_now < 0 else 'BUY'}); skipping."
                )
                return [], False, True

            # Recompute qty from fresh correction + fresh price.
            if action == "SELL":
                beta = etf_to_beta.get(symbol, 1.0)
                qty = int(math.floor(abs(correction_now) / (px * beta))) if (px > 0 and beta > 0) else 0
            else:
                qty = int(math.floor(abs(correction_now) / px)) if px > 0 else 0

            if qty <= 0:
                tprint(f"[HEDGE][{under}] Re-computed qty=0 after fresh correction; skipping.")
                return [], False, True

            # Cap SELL-underlying to shares actually held long — never
            # flip a long position into a short via reconciliation.
            if action == "SELL" and symbol == under:
                cur_under_sh = float(strat_now.get(under, 0.0))
                if cur_under_sh > 0:
                    qty = min(qty, int(cur_under_sh))
                else:
                    tprint(
                        f"[HEDGE][{under}] Reconcile SELL but underlying "
                        f"sh={cur_under_sh:.0f}; cannot sell. Skipping."
                    )
                    return [], False, True

            # Apply FTP cap on re-computed qty (ETF shorts only)
            if action == "SELL" and short_map and symbol != under:
                avail = (short_map.get(symbol) or {}).get("available")
                if avail is not None and avail > 0:
                    qty = min(qty, int(avail))

            tprint(f"[HEDGE][{under}] Re-verified: correction={correction_now:+,.0f} -> qty={qty}")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        px_under = float(prices.get(under) or 0.0)

        # FTP gate for new shorts (ETF only — selling an existing
        # underlying long is not a short sale)
        if action == "SELL" and symbol != under:
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
                }], False, True

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
            }], False, True

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
            "pair_id": f"{under}__{'ORPHAN_CLOSE' if is_orphan else 'HEDGE'}",
            "underlying": under,
            "etf": (symbol if is_etf_trade else ""),
            "px_under": px_under,
            "px_etf":   (px if is_etf_trade else 0.0),
            "target_sh_under": 0, "target_sh_etf": 0,
            "delta_sh_under":  ((-qty if action == "SELL" else qty) if not is_etf_trade else 0),
            "delta_sh_etf":    (qty if (action == "SELL" and is_etf_trade) else 0),
            "filled_sh_under": (filled_signed if not is_etf_trade else 0),
            "filled_sh_etf":   (abs(filled_signed) if is_etf_trade else 0),
            "notes": (
                f"{'ORPHAN_CLOSE' if is_orphan else 'HEDGE'}_{action} "
                f"status={res.status} "
                f"correction_usd={trade_info['correction_usd']:+,.0f}"
            ),
        }
        return [fill_rec], True, True

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
    net_trigger_long_pct: float,
    net_trigger_short_pct: float,
    net_target_long_pct: float,
    net_target_short_pct: float,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
    log_lock: threading.Lock,
) -> Tuple[List[dict], int, int]:
    """Execute Phase 3 hedge trades in parallel (one IB connection per worker).

    TWS limits concurrent API connections (~8). Workers are capped at 6
    concurrent and staggered 1.5s apart to avoid overwhelming the gateway.

    Returns (fill_records, n_triggered, n_traded).
    """
    import time as _time

    if not hedge_trades:
        return [], 0, 0

    all_fills: List[dict] = []
    n_triggered = len(hedge_trades)
    n_traded    = 0

    # Cap concurrent workers to avoid TWS connection limit (default ~8 max)
    MAX_HEDGE_WORKERS = 6
    n_workers = min(parallel_n, len(hedge_trades), MAX_HEDGE_WORKERS)
    stagger_delay = 1.5   # seconds between submitting each worker

    tprint(
        f"[HEDGE] Launching {len(hedge_trades)} trades with "
        f"{n_workers} concurrent workers (stagger={stagger_delay}s)"
    )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for i, t in enumerate(hedge_trades):
            if stop_requested():
                break
            # Stagger submissions so connections don't all hit TWS at once
            if i > 0:
                _time.sleep(stagger_delay)
            fut = pool.submit(
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
                net_trigger_long_pct=net_trigger_long_pct,
                net_trigger_short_pct=net_trigger_short_pct,
                net_target_long_pct=net_target_long_pct,
                net_target_short_pct=net_target_short_pct,
                etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                log_lock=log_lock,
            )
            futures[fut] = t

        for fut in as_completed(futures):
            try:
                fills, was_traded, was_triggered = fut.result()
                all_fills.extend(fills)
                if was_traded:
                    n_traded += 1
                if not was_triggered:
                    n_triggered -= 1
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
    net_trigger_long_pct: float,
    net_trigger_short_pct: float,
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
        band = net_trigger_long_pct if net >= 0 else net_trigger_short_pct
        status = "OK" if abs(pct) <= band * 100.0 else "WARN"
        tprint(f"  {under:<15} {net:>12,.0f} {tgt:>12,.0f} {pct:>7.1f}%  {status}")
    tprint("")


def filter_near_duplicate_requeues(
    *,
    trades: List[dict],
    previous_attempts: Dict[Tuple[str, str], float],
    abs_tolerance_usd: float,
    rel_tolerance: float,
    label: str,
) -> List[dict]:
    """
    Drop trades that are near-identical requeues for the same symbol/action.
    This reduces repetitive submit/cancel churn across hedge passes when the
    correction has barely changed.
    """
    if not trades:
        return trades

    filtered: List[dict] = []
    skipped = 0
    for t in trades:
        key = (str(t.get("symbol", "")), str(t.get("action", "")))
        corr = float(t.get("correction_usd", 0.0))
        prev = previous_attempts.get(key)
        if prev is None:
            filtered.append(t)
            continue

        tol = max(abs_tolerance_usd, abs(prev) * rel_tolerance)
        if abs(corr - prev) <= tol:
            skipped += 1
            continue
        filtered.append(t)

    if skipped > 0:
        tprint(
            f"[HEDGE] {label}: skipped {skipped} near-duplicate requeues "
            f"(abs_tol=${abs_tolerance_usd:,.0f}, rel_tol={rel_tolerance:.0%})."
        )
    return filtered


def log_post_pass_unresolved_threshold(
    *,
    pass_label: str,
    trades: List[dict],
    unresolved_threshold_usd: float,
) -> int:
    """
    Log remaining triggered corrections above a dollar threshold.
    Returns the count above threshold.
    """
    unresolved = [
        t for t in trades
        if abs(float(t.get("correction_usd", 0.0))) >= unresolved_threshold_usd
    ]
    if not unresolved:
        tprint(
            f"[HEDGE] {pass_label}: no unresolved corrections "
            f">= ${unresolved_threshold_usd:,.0f}."
        )
        return 0

    top = ", ".join(
        f"{t['underlying']}:{t['symbol']} {t['correction_usd']:+,.0f}"
        for t in unresolved[:8]
    )
    tprint(
        f"[HEDGE] {pass_label}: {len(unresolved)} unresolved corrections "
        f">= ${unresolved_threshold_usd:,.0f}. Top: {top}"
    )
    return len(unresolved)


# ---------------------------------------------------------------------------
# Phase 3 final reconciliation — underlying-side corrections
# ---------------------------------------------------------------------------

def build_reconciliation_trades(
    *,
    strat_pos: Dict[str, float],
    prices: Dict[str, float],
    account_equity: float,
    gross_leverage: float,
    hedgeable_plan: pd.DataFrame,
    net_trigger_long_pct: float,
    net_trigger_short_pct: float,
    net_target_long_pct: float,
    net_target_short_pct: float,
    min_trade_usd: float,
    etf_to_under: Dict[str, str],
    etf_to_beta: Dict[str, float],
    flow_etfs: Set[str],
    blacklist: Set[str],
) -> List[dict]:
    """
    After the main hedge passes, some underlyings may still be outside
    threshold because:
        - ETF borrow was exhausted (partial fills across all passes)
        - All ETFs were FTP-blocked (no short supply at all)
        - ETF delisted / no valid candidates

    For each still-triggered underlying, generate a trade on the
    UNDERLYING itself:
        - net-long  → SELL underlying to reduce long exposure
        - net-short → BUY underlying to increase long exposure

    Only operates on underlyings that have at least one ETF still
    positioned (non-orphan).  Orphans are already handled by the
    orphan_close path in the main hedge loop.

    Returns trade list sorted by abs(correction_usd) descending.
    """
    blacklist = set(blacklist) if blacklist else set()
    plan_underlyings: Set[str] = set(
        hedgeable_plan["Underlying"].unique()
    ) if not hedgeable_plan.empty else set()

    # Same underlying scan as build_hedge_trades — all positioned
    all_under_values = set(etf_to_under.values())
    check_underlyings: Set[str] = set()
    for sym, sh in strat_pos.items():
        if float(sh) == 0.0:
            continue
        u = etf_to_under.get(sym)
        if u and sym not in flow_etfs:
            beta = etf_to_beta.get(sym, 1.0)
            if beta >= 0:
                check_underlyings.add(u)
        elif sym in all_under_values:
            check_underlyings.add(sym)
    check_underlyings -= blacklist

    trades: List[dict] = []

    for under in sorted(check_underlyings):
        if stop_requested():
            break

        px_under = float(prices.get(under) or 0.0)
        if px_under <= 0.0:
            continue

        under_sh = float(strat_pos.get(under, 0.0))

        # Skip orphans (no ETF exposure) — handled by main loop already
        has_etf = False
        for sym, u in etf_to_under.items():
            if u != under or sym in flow_etfs:
                continue
            beta = etf_to_beta.get(sym, 1.0)
            if beta < 0:
                continue
            if float(strat_pos.get(sym, 0.0)) != 0.0:
                has_etf = True
                break
        if not has_etf:
            continue

        # Guard: don't trade underlying if ETF prices are missing
        is_blind, blind_syms = has_blind_etfs(
            underlying=under, strat_pos=strat_pos, prices=prices,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
            flow_etfs=flow_etfs,
        )
        if is_blind:
            tprint(
                f"[RECONCILE][{under}] BLIND: {blind_syms} positioned but "
                f"unpriceable — skipping."
            )
            continue

        net_notional, actual_gross = compute_beta_adjusted_net_notional(
            strat_pos=strat_pos, prices=prices, underlying=under,
            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
        )
        target_gross = compute_target_gross_per_underlying(
            underlying=under, plan=hedgeable_plan,
            account_equity=account_equity, gross_leverage=gross_leverage,
        )
        ref_gross = max(actual_gross, target_gross)

        in_plan = under in plan_underlyings

        triggered, correction_usd = compute_hedge_delta(
            net_notional=net_notional, target_gross=ref_gross,
            net_trigger_long_pct=net_trigger_long_pct,
            net_trigger_short_pct=net_trigger_short_pct,
            net_target_long_pct=net_target_long_pct,
            net_target_short_pct=net_target_short_pct,
        )

        if not triggered:
            continue
        # Off-plan bypasses min_trade_usd — wind-down trades
        if in_plan and abs(correction_usd) < min_trade_usd:
            continue

        # Correction via the UNDERLYING, not the ETF.
        # correction < 0 (net-long)  → SELL underlying to reduce exposure
        # correction > 0 (net-short) → BUY underlying (same as normal)
        if correction_usd < 0:
            action = "SELL"
            qty = int(math.floor(abs(correction_usd) / px_under))
            # Don't sell more underlying than we hold long
            if under_sh > 0:
                qty = min(qty, int(under_sh))
            elif under_sh <= 0:
                # We're net-long but underlying is already flat/short —
                # this means net-long comes entirely from ETF beta.
                # Selling underlying would make us more short. Skip.
                tprint(
                    f"[RECONCILE][{under}] Net-long but underlying "
                    f"sh={under_sh:.0f}; cannot sell underlying."
                )
                continue
        else:
            action = "BUY"
            qty = int(math.floor(abs(correction_usd) / px_under))

        if qty <= 0:
            continue

        net_pct = (net_notional / ref_gross * 100.0) if ref_gross > 0 else 0.0
        tprint(
            f"[RECONCILE][{under:15s}] net={net_notional:>+12,.0f}  "
            f"net%={net_pct:>+6.1f}%  -> {action} {qty} {under} "
            f"@ ~{px_under:.2f} (correction={correction_usd:+,.0f})"
        )
        trades.append({
            "underlying":        under,
            "symbol":            under,
            "action":            action,
            "qty":               qty,
            "ref_price":         px_under,
            "correction_usd":    correction_usd,
            "net_notional_before": net_notional,
            "target_gross":      ref_gross,
            "orphan_close":      False,
        })

    trades.sort(key=lambda t: abs(t["correction_usd"]), reverse=True)
    if trades:
        tprint(
            f"[RECONCILE] {len(trades)} underlying-side corrections queued."
        )
    return trades


# ---------------------------------------------------------------------------
# Unmapped position detection + user-prompted close
# ---------------------------------------------------------------------------

def detect_unmapped_positions(
    *,
    strat_pos: Dict[str, float],
    prices: Dict[str, float],
    etf_to_under: Dict[str, str],
    screened_universe: Set[str],
    ib: IB,
    prefer_delayed: bool,
) -> List[dict]:
    """
    Find strategy positions that:
        1. Appear in etf_screened_today.csv (ETF or Underlying column)
        2. Are NOT in the ETF beta map used by Phase 3

    This catches symbols that the screener knows about but that somehow
    fell out of the etf_to_under mapping (ticker change between screener
    and beta load, column parse issue, etc.).

    Symbols not in the screened CSV at all (CVRs, personal holdings,
    legacy positions) are ignored — they're outside our universe.

    Returns list sorted by abs(notional) descending.
    """
    all_mapped = set(etf_to_under.keys()) | set(etf_to_under.values())
    unmapped: List[dict] = []

    for sym, sh_raw in strat_pos.items():
        sh = float(sh_raw)
        if sh == 0.0:
            continue
        if sym in all_mapped:
            continue
        # Only flag symbols that are in today's screened CSV
        if sym not in screened_universe:
            continue

        # Try to price it
        px = float(prices.get(sym) or 0.0)
        if px <= 0.0:
            try:
                px = float(get_snapshot_price(ib, sym, prefer_delayed=prefer_delayed) or 0.0)
                if px > 0:
                    prices[sym] = px
            except Exception:
                px = 0.0

        notional = sh * px if px > 0 else 0.0

        unmapped.append({
            "symbol":   sym,
            "shares":   int(sh),
            "price":    px,
            "notional": notional,
            "action":   "SELL" if sh > 0 else "BUY",
            "qty":      abs(int(sh)),
        })

    unmapped.sort(key=lambda d: abs(d["notional"]), reverse=True)
    return unmapped


def prompt_unmapped_close(
    unmapped: List[dict],
    auto_approve: bool,
) -> List[dict]:
    """
    Print a table of unmapped positions and prompt the user to decide
    what to do.  Returns the subset of positions approved for closing.

    This ALWAYS prompts (even with auto_approve=True) because unmapped
    positions are an abnormal condition that requires human judgement —
    a symbol may have legitimately dropped from the screened CSV, or
    it may be a data issue that would resolve on the next screener run.
    """
    if not unmapped:
        return []

    total_abs_notional = sum(abs(d["notional"]) for d in unmapped)

    tprint(f"\n{'='*90}")
    tprint(f"  UNMAPPED POSITIONS — {len(unmapped)} symbols not in today's ETF beta map")
    tprint(f"  These positions are invisible to Phase 3 hedge math.")
    tprint(f"{'='*90}")
    tprint(f"  {'SYMBOL':<12} {'SHARES':>10} {'PRICE':>10} {'NOTIONAL':>12}  ACTION")
    tprint(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12}  {'-'*6}")
    for d in unmapped:
        px_str = f"${d['price']:.2f}" if d["price"] > 0 else "no price"
        ntl_str = f"${d['notional']:>+11,.0f}" if d["price"] > 0 else "   unknown"
        tprint(
            f"  {d['symbol']:<12} {d['shares']:>+10,d} {px_str:>10} {ntl_str}  "
            f"{d['action']} {d['qty']}"
        )
    tprint(f"\n  Total absolute notional: ${total_abs_notional:,.0f}")
    tprint(f"{'='*90}")

    # Unpriced symbols cannot be traded.
    priceable = [d for d in unmapped if d["price"] > 0.0]
    unpriceable = [d for d in unmapped if d["price"] <= 0.0]

    if unpriceable:
        tprint(
            f"\n  WARNING: {len(unpriceable)} symbol(s) have no price and "
            f"cannot be closed: {[d['symbol'] for d in unpriceable]}"
        )

    if not priceable:
        tprint("  No priceable unmapped positions to close.")
        return []

    tprint(
        f"\n  {len(priceable)} priceable position(s) can be closed to flatten."
    )
    # Always require explicit human approval — never auto-approve unmapped
    # closes.  This is intentional: unmapped positions are an abnormal
    # condition (screener data issue, ticker change, etc.) and need review.
    tprint("  NOTE: auto_approve does NOT apply to unmapped closes.\n")
    ans = input("  Close ALL priceable unmapped positions? (y/n): ").strip().lower()

    if ans == "y":
        tprint(f"  Approved: closing {len(priceable)} unmapped positions.\n")
        return priceable
    else:
        tprint("  Skipped: unmapped positions left as-is.\n")
        return []


def execute_unmapped_closes(
    *,
    approved: List[dict],
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
    cancel_service: CoordinatorCancelService,
    log_exposure_event,
) -> List[dict]:
    """
    Execute close trades for unmapped positions serially on the
    coordinator IB connection.

    Re-reads live position before each trade to confirm the symbol
    is still held (it may have been closed externally).

    Returns fill records.
    """
    fill_records: List[dict] = []

    for trade_info in approved:
        if stop_requested():
            tprint("[UNMAPPED] Shutdown requested; aborting.")
            break

        symbol = trade_info["symbol"]
        action = trade_info["action"]

        # Re-read live position — confirm still held
        ib_pos_now = current_ib_positions(ib)
        strat_now  = strategy_position_only(ib_pos_now, baseline)
        cur_sh = float(strat_now.get(symbol, 0.0))

        if (action == "SELL" and cur_sh <= 0) or (action == "BUY" and cur_sh >= 0):
            tprint(
                f"[UNMAPPED][{symbol}] Already flat (sh={cur_sh:.0f}); skipping."
            )
            continue

        qty = abs(int(cur_sh))
        if qty <= 0:
            continue

        # Fresh price
        fresh_px = get_snapshot_price(ib, symbol, prefer_delayed=prefer_delayed)
        px = float(fresh_px or trade_info["price"])
        if px > 0:
            prices[symbol] = px

        if px <= 0.0:
            tprint(f"[UNMAPPED][{symbol}] No price; cannot close.")
            continue

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_exposure_event(
            stage="UNMAPPED_PRE", pair_id=f"{symbol}__UNMAPPED",
            underlying="", etf="", symbol=symbol,
            delta_sh=0, filled_sh=0, trade=None,
        )

        order_ref = f"{strategy_tag}|UNMAPPED|{symbol}|{action}"
        res = execute_leg(
            ib=ib, symbol=symbol, action=action, qty=qty,
            ref_price=px, bps=limit_bps, order_ref=order_ref,
            exec_cfg=exec_cfg, timeout=timeout, max_retries=max_retries,
            dry_run=dry_run, context=f"{symbol}|UNMAPPED",
            cancel_service=cancel_service,
        )

        filled_signed = int(res.filled) if action == "BUY" else -int(res.filled)

        log_exposure_event(
            stage="UNMAPPED_POST", pair_id=f"{symbol}__UNMAPPED",
            underlying="", etf="", symbol=symbol,
            delta_sh=(-qty if action == "SELL" else qty),
            filled_sh=filled_signed, trade=res.trade,
        )

        fill_records.append({
            "filled_at": now, "run_date": run_date,
            "strategy_tag": strategy_tag,
            "pair_id": f"{symbol}__UNMAPPED",
            "underlying": "", "etf": "",
            "px_under": 0.0, "px_etf": 0.0,
            "target_sh_under": 0, "target_sh_etf": 0,
            "delta_sh_under": 0, "delta_sh_etf": 0,
            "filled_sh_under": 0, "filled_sh_etf": 0,
            "notes": (
                f"UNMAPPED_CLOSE {action} {qty} "
                f"status={res.status}"
            ),
        })
        tprint(
            f"[UNMAPPED][{symbol}] {action} {qty} @ ~{px:.2f} "
            f"-> filled={abs(filled_signed)} status={res.status}"
        )

    return fill_records


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
    suppress_error_codes = ibkr_cfg.get("suppress_error_codes", [10089])
    suppress_error_codes = [int(c) for c in (suppress_error_codes or [])]

    configure_ib_error_log_filter(suppress_error_codes)
    if suppress_error_codes:
        tprint(f"[IB] Suppressing noisy API error codes: {sorted(set(suppress_error_codes))}")

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
    net_trigger_long_pct    = float(reb_cfg.get("net_trigger_long_pct", 0.02))
    net_trigger_short_pct   = float(reb_cfg.get("net_trigger_short_pct", 0.01))
    net_target_long_pct     = float(reb_cfg.get("net_target_long_pct", 0.01))
    net_target_short_pct    = float(reb_cfg.get("net_target_short_pct", 0.00))
    min_trade_usd           = float(reb_cfg.get("min_trade_usd", 500.0))
    requeue_abs_tolerance_usd = float(reb_cfg.get("requeue_abs_tolerance_usd", max(50.0, min_trade_usd * 0.25)))
    requeue_rel_tolerance     = float(reb_cfg.get("requeue_rel_tolerance", 0.10))
    post_pass_unresolved_usd  = float(reb_cfg.get("post_pass_unresolved_usd", min_trade_usd))
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
    configure_market_data_mode(ib, prefer_delayed=prefer_delayed)

    cancel_service = CoordinatorCancelService(host=host, port=port)
    cancel_service.start()
    tprint("[CANCEL_COORD] Started cancel coordinator.")

    log_lock:  threading.Lock = threading.Lock()
    prices:    Dict[str, float] = {}
    blocked_short_etfs: Set[str] = set()
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

        # Only prefetch symbols that are in the plan or beta-mapped universe;
        # strat_pos may contain non-tradeable legacy holdings (CVRs, escrows, etc.)
        beta_mapped_universe: Set[str] = (
            set(etf_to_under.keys()) | set(etf_to_under.values())
        )
        all_symbols: Set[str] = (
            set(plan["ETF"].tolist()) | set(plan["Underlying"].tolist())
            | {s for s, sh in strat_pos.items()
               if float(sh) != 0.0 and s in beta_mapped_universe}
        )
        tprint(f"[PRICES] Prefetching {len(all_symbols)} symbols...")
        n_px, n_req = prefetch_prices_batched(
            ib=ib,
            symbols=all_symbols,
            prices=prices,
            prefer_delayed=prefer_delayed,
            batch_size=75,
        )
        tprint(f"[PRICES] Prefetch complete: {n_px}/{n_req} resolved.")

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
        # PRE-PHASE-3 — Detect & prompt for unmapped positions
        # ==================================================================
        # Only flag symbols that appear in etf_screened_today.csv but
        # somehow aren't in the etf_to_under beta map.  Everything
        # outside the screened CSV (CVRs, personal holdings, legacy
        # positions) is not our universe and gets ignored.
        screened_universe: Set[str] = (
            set(screened["ETF"].astype(str).map(norm_sym))
            | set(screened["Underlying"].astype(str).map(norm_sym))
        )
        unmapped_positions = detect_unmapped_positions(
            strat_pos=strat_pos,
            prices=prices,
            etf_to_under=etf_to_under,
            screened_universe=screened_universe,
            ib=ib,
            prefer_delayed=prefer_delayed,
        )
        if unmapped_positions:
            approved_unmapped = prompt_unmapped_close(
                unmapped_positions, auto_approve=auto_approve,
            )
            if approved_unmapped:
                fills_unmap = execute_unmapped_closes(
                    approved=approved_unmapped,
                    ib=ib,
                    baseline=baseline,
                    prices=prices,
                    prefer_delayed=prefer_delayed,
                    exec_cfg=exec_cfg,
                    strategy_tag=strategy_tag,
                    run_date=run_date,
                    limit_bps=limit_bps,
                    timeout=timeout,
                    max_retries=max_retries,
                    dry_run=dry_run,
                    cancel_service=cancel_service,
                    log_exposure_event=log_exposure_event,
                )
                all_fills.extend(fills_unmap)
                if fills_unmap:
                    _sync_positions_after_external_trades(ib, timeout_s=10.0)
                    # Refresh positions so Phase 3 sees clean state
                    ib_pos    = current_ib_positions(ib)
                    strat_pos = strategy_position_only(ib_pos, baseline)

        if stop_requested():
            tprint("[SHUTDOWN] Exiting after unmapped close.")
            return

        # ==================================================================
        # PHASE 3 — Directional hedge rebalance
        # ==================================================================
        if not args.skip_phase_3:
            tprint("\n" + "=" * 60)
            tprint("  PHASE 3 — DIRECTIONAL HEDGE REBALANCE")
            tprint("=" * 60)

            account_equity = get_account_equity(ib)
            tprint(f"[EQUITY] NetLiquidation = ${account_equity:,.0f}")
            tprint(
                f"[HEDGE]  trigger_long={net_trigger_long_pct*100:.0f}%  "
                f"trigger_short={net_trigger_short_pct*100:.0f}%  "
                f"target_long={net_target_long_pct*100:.0f}%  "
                f"target_short={net_target_short_pct*100:.0f}%  "
                f"gross_leverage={gross_leverage}x  "
                f"min_trade_usd=${min_trade_usd:,.0f}"
            )

            # Pre-hedge snapshot (for summary) — include all positioned
            # underlyings, not just plan, so the summary reflects orphans.
            pre_net: Dict[str, float] = {}
            pre_tgt: Dict[str, float] = {}
            _all_under_values = set(etf_to_under.values())
            _snapshot_underlyings: Set[str] = set(
                hedgeable_df["Underlying"].unique()
            ) if not hedgeable_df.empty else set()
            for sym, sh in strat_pos.items():
                if float(sh) == 0.0:
                    continue
                u = etf_to_under.get(sym)
                if u and sym not in flow_etfs:
                    beta = etf_to_beta.get(sym, 1.0)
                    if beta >= 0:
                        _snapshot_underlyings.add(u)
                elif sym in _all_under_values:
                    _snapshot_underlyings.add(sym)
            _snapshot_underlyings -= set(blacklist)

            for u in sorted(_snapshot_underlyings):
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

            MAX_HEDGE_PASSES = 3
            triggered_underlyings: Set[str] = set()
            total_traded    = 0
            recent_attempts: Dict[Tuple[str, str], float] = {}

            for hedge_pass in range(1, MAX_HEDGE_PASSES + 1):
                if stop_requested():
                    break

                # Refresh positions at the start of each pass
                ib_pos    = current_ib_positions(ib)
                strat_pos = strategy_position_only(ib_pos, baseline)

                hedge_trades = build_hedge_trades(
                    hedgeable_plan=hedgeable_df,
                    strat_pos=strat_pos, prices=prices,
                    account_equity=account_equity, gross_leverage=gross_leverage,
                    net_trigger_long_pct=net_trigger_long_pct,
                    net_trigger_short_pct=net_trigger_short_pct,
                    net_target_long_pct=net_target_long_pct,
                    net_target_short_pct=net_target_short_pct,
                    min_trade_usd=min_trade_usd,
                    etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                    short_map=short_map,
                    blocked_short_etfs=blocked_short_etfs,
                    flow_etfs=flow_etfs,
                    blacklist=blacklist,
                )
                hedge_trades = filter_near_duplicate_requeues(
                    trades=hedge_trades,
                    previous_attempts=recent_attempts,
                    abs_tolerance_usd=requeue_abs_tolerance_usd,
                    rel_tolerance=requeue_rel_tolerance,
                    label=f"Pass {hedge_pass}",
                )

                if not hedge_trades:
                    tprint(f"[HEDGE] Pass {hedge_pass}: no trades needed — converged.")
                    break

                short_corrections = [t for t in hedge_trades if t["action"] == "SELL"]
                long_corrections  = [t for t in hedge_trades if t["action"] == "BUY"]
                blocked_short_skips = [
                    t for t in short_corrections
                    if t.get("symbol") in blocked_short_etfs and t.get("symbol") != t.get("underlying")
                ]
                if blocked_short_skips:
                    tprint(
                        f"[HEDGE] Pass {hedge_pass}: fast-fail skipped "
                        f"{len(blocked_short_skips)} queued blocked ETF shorts."
                    )
                    short_corrections = [
                        t for t in short_corrections
                        if not (t.get("symbol") in blocked_short_etfs and t.get("symbol") != t.get("underlying"))
                    ]

                tprint(
                    f"[HEDGE] Pass {hedge_pass}/{MAX_HEDGE_PASSES}: "
                    f"{len(short_corrections)} short corrections, "
                    f"{len(long_corrections)} long corrections (pre-rebuild)."
                )

                pass_traded = 0

                # ── 3a: Execute short legs first ─────────────────────────────
                # Short first so we know actual coverage before sizing longs.
                if short_corrections and not stop_requested():
                    for t in short_corrections:
                        recent_attempts[(t["symbol"], t["action"])] = float(t.get("correction_usd", 0.0))
                    fills3a, n_trig_a, n_trade_a = execute_hedge_pass_parallel(
                        hedge_trades=short_corrections,
                        host=host, port=port, client_id=client_id,
                        baseline=baseline, prices=prices,
                        prefer_delayed=prefer_delayed, exec_cfg=exec_cfg,
                        strategy_tag=strategy_tag, run_date=run_date,
                        limit_bps=limit_bps, timeout=timeout, max_retries=max_retries,
                        dry_run=dry_run, parallel_n=parallel_n,
                        short_map=short_map, cancel_service=cancel_service,
                        log_exposure_event=log_exposure_event,
                        net_trigger_long_pct=net_trigger_long_pct,
                        net_trigger_short_pct=net_trigger_short_pct,
                        net_target_long_pct=net_target_long_pct,
                        net_target_short_pct=net_target_short_pct,
                        etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                        log_lock=log_lock,
                    )
                    all_fills.extend(fills3a)
                    newly_blocked = blocked_short_symbols_from_fills(fills3a)
                    if newly_blocked - blocked_short_etfs:
                        just_added = sorted(newly_blocked - blocked_short_etfs)
                        tprint(f"[HEDGE] Marking {len(just_added)} ETF shorts as blocked this run: {just_added}")
                    blocked_short_etfs |= newly_blocked
                    triggered_underlyings.update(t["underlying"] for t in short_corrections)
                    total_traded    += n_trade_a
                    pass_traded     += n_trade_a

                    # Sync positions so long-leg sizing reflects actual short fills.
                    # If SMUP only filled 50 of 200, the rebuild below will see the
                    # real net and buy only as much underlying as is appropriate.
                    # 20s timeout: up to 6 parallel shorts may need extra propagation time.
                    _sync_positions_after_external_trades(
                        ib,
                        watch_syms=[t["symbol"] for t in short_corrections],
                        timeout_s=20.0,
                    )

                # ── 3b: Rebuild from fresh positions, then execute long legs ─
                # Do NOT use the long_corrections list built before 3a — it was
                # sized against planned shorts, not actual fills.
                if not stop_requested():
                    ib_pos    = current_ib_positions(ib)
                    strat_pos = strategy_position_only(ib_pos, baseline)

                    hedge_trades_b = build_hedge_trades(
                        hedgeable_plan=hedgeable_df,
                        strat_pos=strat_pos, prices=prices,
                        account_equity=account_equity, gross_leverage=gross_leverage,
                        net_trigger_long_pct=net_trigger_long_pct,
                        net_trigger_short_pct=net_trigger_short_pct,
                        net_target_long_pct=net_target_long_pct,
                        net_target_short_pct=net_target_short_pct,
                        min_trade_usd=min_trade_usd,
                        etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                        short_map=short_map,
                        blocked_short_etfs=blocked_short_etfs,
                        flow_etfs=flow_etfs,
                        blacklist=blacklist,
                    )
                    long_corrections_b = [t for t in hedge_trades_b if t["action"] == "BUY"]
                    long_corrections_b = filter_near_duplicate_requeues(
                        trades=long_corrections_b,
                        previous_attempts=recent_attempts,
                        abs_tolerance_usd=requeue_abs_tolerance_usd,
                        rel_tolerance=requeue_rel_tolerance,
                        label=f"Pass {hedge_pass}b",
                    )

                    if long_corrections_b:
                        tprint(
                            f"[HEDGE] Pass {hedge_pass}b: {len(long_corrections_b)} long "
                            f"corrections sized to actual short fills."
                        )
                        for t in long_corrections_b:
                            recent_attempts[(t["symbol"], t["action"])] = float(t.get("correction_usd", 0.0))
                        fills3b, n_trig_b, n_trade_b = execute_hedge_pass_parallel(
                            hedge_trades=long_corrections_b,
                            host=host, port=port, client_id=client_id,
                            baseline=baseline, prices=prices,
                            prefer_delayed=prefer_delayed, exec_cfg=exec_cfg,
                            strategy_tag=strategy_tag, run_date=run_date,
                            limit_bps=limit_bps, timeout=timeout, max_retries=max_retries,
                            dry_run=dry_run, parallel_n=parallel_n,
                            short_map=short_map, cancel_service=cancel_service,
                            log_exposure_event=log_exposure_event,
                            net_trigger_long_pct=net_trigger_long_pct,
                            net_trigger_short_pct=net_trigger_short_pct,
                            net_target_long_pct=net_target_long_pct,
                            net_target_short_pct=net_target_short_pct,
                            etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                            log_lock=log_lock,
                        )
                        all_fills.extend(fills3b)
                        newly_blocked = blocked_short_symbols_from_fills(fills3b)
                        if newly_blocked - blocked_short_etfs:
                            just_added = sorted(newly_blocked - blocked_short_etfs)
                            tprint(f"[HEDGE] Marking {len(just_added)} ETF shorts as blocked this run: {just_added}")
                        blocked_short_etfs |= newly_blocked
                        triggered_underlyings.update(t["underlying"] for t in long_corrections_b)
                        total_traded    += n_trade_b
                        pass_traded     += n_trade_b

                        _sync_positions_after_external_trades(
                            ib,
                            watch_syms=[t["symbol"] for t in long_corrections_b],
                            timeout_s=15.0,
                        )

                # Post-pass check: what still exceeds a material dollar threshold?
                ib_pos_post_pass = current_ib_positions(ib)
                strat_pos_post_pass = strategy_position_only(ib_pos_post_pass, baseline)
                post_pass_trades = build_hedge_trades(
                    hedgeable_plan=hedgeable_df,
                    strat_pos=strat_pos_post_pass, prices=prices,
                    account_equity=account_equity, gross_leverage=gross_leverage,
                    net_trigger_long_pct=net_trigger_long_pct,
                    net_trigger_short_pct=net_trigger_short_pct,
                    net_target_long_pct=net_target_long_pct,
                    net_target_short_pct=net_target_short_pct,
                    min_trade_usd=min_trade_usd,
                    etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                    short_map=short_map,
                    blocked_short_etfs=blocked_short_etfs,
                    flow_etfs=flow_etfs,
                    blacklist=blacklist,
                )
                unresolved_count = log_post_pass_unresolved_threshold(
                    pass_label=f"Post-pass {hedge_pass}",
                    trades=post_pass_trades,
                    unresolved_threshold_usd=post_pass_unresolved_usd,
                )
                if unresolved_count == 0:
                    tprint(f"[HEDGE] Pass {hedge_pass}: converged above dollar threshold.")
                    break

                if pass_traded == 0:
                    tprint(f"[HEDGE] Pass {hedge_pass}: no fills — stopping passes.")
                    break

            # ── Final reconciliation: underlying-side corrections ─────
            # After the main ETF-side hedge passes, some underlyings may
            # still be outside threshold (borrow exhausted, FTP blocked,
            # no ETF candidates).  Correct these by trading the underlying
            # directly — sell to reduce net-long, buy to reduce net-short.
            if not stop_requested():
                ib_pos    = current_ib_positions(ib)
                strat_pos = strategy_position_only(ib_pos, baseline)

                reconcile_trades = build_reconciliation_trades(
                    strat_pos=strat_pos,
                    prices=prices,
                    account_equity=account_equity,
                    gross_leverage=gross_leverage,
                    hedgeable_plan=hedgeable_df,
                    net_trigger_long_pct=net_trigger_long_pct,
                    net_trigger_short_pct=net_trigger_short_pct,
                    net_target_long_pct=net_target_long_pct,
                    net_target_short_pct=net_target_short_pct,
                    min_trade_usd=min_trade_usd,
                    etf_to_under=etf_to_under,
                    etf_to_beta=etf_to_beta,
                    flow_etfs=flow_etfs,
                    blacklist=blacklist,
                )

                if reconcile_trades:
                    tprint(
                        f"\n{'-'*60}\n"
                        f"  PHASE 3 RECONCILIATION — "
                        f"{len(reconcile_trades)} underlying-side corrections\n"
                        f"{'-'*60}"
                    )
                    fills_rec, n_trig_rec, n_trade_rec = execute_hedge_pass_parallel(
                        hedge_trades=reconcile_trades,
                        host=host, port=port, client_id=client_id,
                        baseline=baseline, prices=prices,
                        prefer_delayed=prefer_delayed, exec_cfg=exec_cfg,
                        strategy_tag=strategy_tag, run_date=run_date,
                        limit_bps=limit_bps, timeout=timeout,
                        max_retries=max_retries,
                        dry_run=dry_run, parallel_n=parallel_n,
                        short_map=short_map, cancel_service=cancel_service,
                        log_exposure_event=log_exposure_event,
                        net_trigger_long_pct=net_trigger_long_pct,
                        net_trigger_short_pct=net_trigger_short_pct,
                        net_target_long_pct=net_target_long_pct,
                        net_target_short_pct=net_target_short_pct,
                        etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                        log_lock=log_lock,
                    )
                    all_fills.extend(fills_rec)
                    total_traded += n_trade_rec

                    if fills_rec:
                        _sync_positions_after_external_trades(
                            ib,
                            watch_syms=[t["symbol"] for t in reconcile_trades],
                            timeout_s=15.0,
                        )
                    tprint(
                        f"[RECONCILE] Complete: "
                        f"{n_trade_rec}/{len(reconcile_trades)} traded."
                    )

            # Post-hedge net for summary — use actual ref_gross (same
            # denominator Phase 3 used for trigger/target) so the WARN
            # check reflects whether the pair is genuinely out-of-band.
            ib_pos_final    = current_ib_positions(ib)
            strat_pos_final = strategy_position_only(ib_pos_final, baseline)
            post_net: Dict[str, float] = {}
            post_ref_gross: Dict[str, float] = {}
            for u in pre_net:
                net, actual_gross = compute_beta_adjusted_net_notional(
                    strat_pos=strat_pos_final, prices=prices, underlying=u,
                    etf_to_under=etf_to_under, etf_to_beta=etf_to_beta,
                )
                plan_tgt = pre_tgt.get(u, 0.0)
                post_net[u] = net
                post_ref_gross[u] = max(actual_gross, plan_tgt)

            print_phase_summary(
                phase="PHASE 3 — DIRECTIONAL HEDGE",
                n_checked=len(pre_net),
                n_triggered=len(triggered_underlyings),
                n_traded=total_traded,
                net_by_underlying=post_net,
                target_gross_by_underlying=post_ref_gross,
                net_trigger_long_pct=net_trigger_long_pct,
                net_trigger_short_pct=net_trigger_short_pct,
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