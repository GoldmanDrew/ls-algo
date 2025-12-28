#!/usr/bin/env python3
"""
execute_trade_plan.py

Reads a proposed trade plan CSV and executes pair-by-pair with manual approval.

Key mechanics:
- Uses baseline_snapshot.csv to protect pre-existing positions.
- Places short leg first. Then sizes long leg based on filled short notional to
  keep the pair delta-neutral (per your ratio).
- Tags orders using orderRef so you can filter them in IBKR activity.

Safety:
- DRY_RUN=1 environment variable to simulate without placing orders.

Usage:
  python execute_trade_plan.py --strategy-tag MYTAG --host 127.0.0.1 --port 7497 --client-id 3
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from ib_insync import IB, Stock, Order, Trade, TagValue
from strategy_config import load_config

# ---------------------------
# Symbol normalization
# ---------------------------

IB_SYMBOL_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "BRK-B": ("BRK B", "NYSE"),
    "BRK-A": ("BRK A", "NYSE"),
}
REVERSE_IB_SYMBOL_MAP: Dict[str, str] = {ib_sym: uni for uni, (ib_sym, _) in IB_SYMBOL_MAP.items()}


def ib_symbol_from_universal(sym: str) -> Tuple[str, Optional[str]]:
    s = str(sym).strip().upper()
    if s in IB_SYMBOL_MAP:
        return IB_SYMBOL_MAP[s]
    return s, None


def universal_symbol_from_ib(sym: str) -> str:
    s = str(sym).strip().upper()
    return REVERSE_IB_SYMBOL_MAP.get(s, s)


def make_stock(symbol: str) -> Stock:
    ib_sym, primary = ib_symbol_from_universal(symbol)
    c = Stock(ib_sym, "SMART", "USD")
    if primary:
        c.primaryExchange = primary
    return c


# ---------------------------
# IBKR connection & pricing
# ---------------------------

def connect_ib(host: str, port: int, client_id: int) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR.")
    return ib


def safe_price(v) -> Optional[float]:
    try:
        x = float(v)
        return x if x > 0 else None
    except Exception:
        return None


def get_snapshot_price(ib: IB, symbol: str, prefer_delayed: bool = True) -> float:
    """
    Lightweight price fetch:
    - If prefer_delayed, request marketDataType=3 first (delayed), else live first.
    - Try bid/ask mid, last, close, marketPrice.
    - Fallback: 1D historical close.
    """
    sym_u = symbol.upper()
    contract = make_stock(sym_u)
    ib.qualifyContracts(contract)

    def try_type(data_type: int) -> Optional[float]:
        ib.reqMarketDataType(data_type)
        t = ib.reqMktData(contract, "", snapshot=True)
        for _ in range(12):
            ib.sleep(0.25)
            bid = safe_price(getattr(t, "bid", None)) or safe_price(getattr(t, "delayedBid", None))
            ask = safe_price(getattr(t, "ask", None)) or safe_price(getattr(t, "delayedAsk", None))
            last = safe_price(getattr(t, "last", None)) or safe_price(getattr(t, "delayedLast", None))
            close = safe_price(getattr(t, "close", None)) or safe_price(getattr(t, "delayedClose", None))
            mkt = safe_price(t.marketPrice())
            if bid and ask:
                return (bid + ask) / 2.0
            if last:
                return last
            if close:
                return close
            if mkt:
                return mkt
        return None

    px = None
    if prefer_delayed:
        px = try_type(3) or try_type(1)
    else:
        px = try_type(1) or try_type(3)

    try:
        ib.cancelMktData(contract)
    except Exception:
        pass

    if px is None:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
            keepUpToDate=False,
        )
        if bars:
            px = safe_price(bars[-1].close)

    if px is None:
        raise RuntimeError(f"No usable price for {sym_u}")
    return float(px)


# ---------------------------
# Orders
# ---------------------------

def build_limit_order(action: str, qty: int, ref_price: float, bps: float, order_ref: str) -> Order:
    """
    Simple limit order priced slightly through ref_price.
    """
    o = Order()
    o.action = action.upper()
    o.totalQuantity = int(qty)
    o.tif = "DAY"
    o.orderType = "LMT"
    offset = ref_price * (bps / 10_000.0)
    if o.action == "BUY":
        o.lmtPrice = round(ref_price + offset, 4)
    else:
        o.lmtPrice = round(ref_price - offset, 4)
    o.orderRef = order_ref
    return o


import time

TERMINAL = {"filled", "cancelled", "inactive", "api cancelled"}
ACCEPTED = {"presubmitted", "submitted"}

def wait_for_trade_accepted(ib: IB, trade: Trade, timeout: float = 15.0) -> bool:
    """Wait until IB accepts/transmits the order (Submitted/PreSubmitted) or it becomes terminal."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        status = (trade.orderStatus.status or "").lower()
        if status in ACCEPTED:
            return True
        if status in TERMINAL:
            return False
        ib.sleep(0.2)
    return False


def wait_for_trade_done(ib: IB, trade: Trade, timeout: float = 90.0) -> Trade:
    """Wait until terminal state; do not treat PendingSubmit as failure."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        status = (trade.orderStatus.status or "").lower()
        if status in TERMINAL:
            return trade
        ib.sleep(0.2)
    return trade




from typing import Tuple, Optional, Dict
from ib_insync import IB, Trade, MarketOrder  # ensure MarketOrder imported

def execute_leg(
    ib: IB,
    symbol: str,
    action: str,
    qty: int,
    ref_price: float,
    bps: float,
    order_ref: str,
    exec_cfg: Dict,                 # <-- NEW: pass execution config dict in
    timeout: float = 90.0,
    max_retries: int = 3,
    dry_run: bool = False,
) -> Tuple[int, Optional[Trade]]:
    """
    Option B: "walk the limit" (refresh price each retry + increasing aggressiveness),
    with optional market fallback on 3rd attempt if first 2 tries fail.
    Returns filled shares (int) and last Trade.
    """
    if qty <= 0:
        return 0, None

    contract = make_stock(symbol)
    ib.qualifyContracts(contract)

    filled_total = 0
    last_trade: Optional[Trade] = None

    aggressive_step_bps = float(exec_cfg.get("aggressive_bps_step", 25.0))
    market_fallback_third_try = bool(exec_cfg.get("market_fallback_third_try", True))

    for attempt in range(1, max_retries + 1):
        remain = qty - filled_total
        if remain <= 0:
            break

        # Always refresh reference price (live if possible)
        ref_px_now = get_snapshot_price(ib, symbol, prefer_delayed=True)

        # Attempt-specific aggressiveness
        bps_now = bps + (attempt - 1) * aggressive_step_bps

        # Decide order type: LMT (walk) vs MKT (fallback)
        use_market = (market_fallback_third_try and attempt == 3)

        if use_market:
            o = MarketOrder(action, remain)
            o.orderRef = f"{order_ref}|att{attempt}|MKT"
            lmt_str = "MKT"
        else:
            o = build_limit_order(
                action,
                remain,
                ref_px_now,
                bps_now,
                order_ref=f"{order_ref}|att{attempt}|LMT"
            )
            lmt_str = f"{o.lmtPrice:.4f}"

        print(
            f"[LEG] {symbol} {action} qty={remain} "
            f"ref_now={ref_px_now:.4f} bps_now={bps_now:.1f} px={lmt_str} refTag={o.orderRef}"
        )

        if dry_run:
            filled_total += remain
            continue

        trade = ib.placeOrder(contract, o)
        last_trade = trade

        accepted = wait_for_trade_accepted(ib, trade, timeout=float(exec_cfg.get("accept_timeout_sec", 15.0)))

        # If not accepted, do NOT assume "failed"; for MKT give it more time before doing anything drastic.
        if not accepted:
            if use_market:
                # Give market orders extra runway to get accepted
                accepted = wait_for_trade_accepted(ib, trade, timeout=float(exec_cfg.get("market_accept_timeout_sec", 45.0)))
            if not accepted:
                print(f"[LEG] {symbol} not accepted yet (status={trade.orderStatus.status}); continuing to next attempt.")
                continue

        # Once accepted, wait for fill/cancel
        done_timeout = float(exec_cfg.get("market_done_timeout_sec" if use_market else "limit_done_timeout_sec",
                                        180.0 if use_market else timeout))
        trade = wait_for_trade_done(ib, trade, timeout=done_timeout)

        status = (trade.orderStatus.status or "").lower()
        filled = int(trade.orderStatus.filled or 0)

        # remaining may be None; don't coerce it to 0
        raw_remaining = trade.orderStatus.remaining
        remaining = None if raw_remaining is None else int(raw_remaining)

        print(f"[LEG] status={status} filled={filled} remaining={remaining}")

        filled_total = min(qty, filled_total + filled)

        # Only cancel LIMIT orders that are still working after waiting
        if (not use_market) and status in ("presubmitted", "submitted", "pendingsubmit"):
            try:
                ib.cancelOrder(trade.order)
            except Exception:
                pass


    return int(filled_total), last_trade



# ---------------------------
# Baseline snapshot mechanics
# ---------------------------

def load_baseline_qty(path: Path) -> Dict[str, float]:
    if not path.exists():
        print(f"[BASELINE] No baseline file found at {path}. Treating baseline as empty.")
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return dict(df.groupby("symbol")["qty"].sum())


def current_ib_positions(ib: IB) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in ib.positions():
        sym = universal_symbol_from_ib(p.contract.symbol)
        out[sym] = out.get(sym, 0.0) + float(p.position)
    return out


def strategy_position_only(ib_pos: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    syms = set(ib_pos) | set(baseline)
    return {s: float(ib_pos.get(s, 0.0) - baseline.get(s, 0.0)) for s in syms}


# ---------------------------
# Executor main
# ---------------------------

def append_fills(rows: list[dict], fills_path: Path) -> None:
    fills_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if fills_path.exists():
        df_old = pd.read_csv(fills_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(fills_path, index=False)
    print(f"[FILLS] Appended {len(rows)} rows -> {fills_path}")


def main() -> None:
    """
    execute_trade_plan.py main()

    Now reads everything from: config/strategy_config.yml
      - ibkr.host / port / client_id / prefer_delayed
      - execution.limit_bps / timeout_sec / short_first / max_retries / dry_run
      - strategy.tag
      - paths.proposed_trades_csv / baseline_csv / fills_csv

    Optional CLI override:
      --strategy-tag  (if you want to run a different tag than what's in YAML)
    """

    import os
    import argparse
    from pathlib import Path

    import pandas as pd
    import yaml
    from datetime import datetime

    CONFIG_YML = Path("config/strategy_config.yml")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy-tag",
        default=None,
        help="Override strategy.tag from config/strategy_config.yml",
    )
    args = ap.parse_args()

    if not CONFIG_YML.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_YML}")

    cfg = yaml.safe_load(CONFIG_YML.read_text(encoding="utf-8")) or {}

    ibkr_cfg = cfg.get("ibkr", {}) or {}
    strat_cfg = cfg.get("strategy", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}

    # --- Strategy tag ---
    strategy_tag = args.strategy_tag or str(strat_cfg.get("tag", "")).strip()
    if not strategy_tag:
        raise ValueError("Missing strategy.tag in config/strategy_config.yml (or pass --strategy-tag).")

    # --- IBKR connection params ---
    host = str(ibkr_cfg.get("host", "127.0.0.1"))
    port = int(ibkr_cfg.get("port", 7497))
    client_id = int(ibkr_cfg.get("client_id", 3))
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))

    # --- Execution params ---
    limit_bps = float(exec_cfg.get("limit_bps", 10.0))
    timeout = float(exec_cfg.get("timeout_sec", 90))
    short_first = bool(exec_cfg.get("short_first", True))
    max_retries = int(exec_cfg.get("max_retries", 3))

    # Let env var DRY_RUN override config (handy for safety)
    if "DRY_RUN" in os.environ:
        dry_run = bool(int(os.getenv("DRY_RUN", "0")))
    else:
        dry_run = bool(exec_cfg.get("dry_run", False))

    if dry_run:
        print("[DRY_RUN] Enabled. No orders will be placed.")

    # --- Paths ---
    trade_plan_csv = Path(paths_cfg.get("proposed_trades_csv", "data/proposed_trades.csv"))
    baseline_csv = Path(paths_cfg.get("baseline_csv", "data/baseline_snapshot.csv"))

    # If your append_fills() uses a global FILLS_CSV, keep it as-is.
    # Otherwise, you can pass this into append_fills() if your implementation supports it.
    # fills_csv = Path(paths_cfg.get("fills_csv", "data/trade_fills.csv"))

    if not trade_plan_csv.exists():
        raise FileNotFoundError(f"Trade plan not found: {trade_plan_csv}")

    plan = pd.read_csv(trade_plan_csv)
    plan = plan[plan["strategy_tag"].astype(str) == strategy_tag].copy()
    if plan.empty:
        raise ValueError(f"No rows in {trade_plan_csv} for strategy_tag={strategy_tag}")

    # baseline protection
    baseline = load_baseline_qty(baseline_csv)

    ib = connect_ib(host, port, client_id)
    try:
        ib_pos = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)
        print(
            f"[POS] current IB symbols={len(ib_pos)}; "
            f"baseline symbols={len(baseline)}; "
            f"strategy-only symbols={len(strat_pos)}"
        )

        fills_to_append = []
        approve_all = False

        # NOTE: we still keep the manual approval loop (pair-by-pair),
        # since that's core to your workflow.
        for _, row in plan.iterrows():
            # --- New trade-plan schema support (proposed_trades.csv) ---
            u = str(row["Underlying"]).upper()
            e = str(row["ETF"]).upper()

            # pair_id: generate deterministically if not provided
            pair_id = str(row.get("pair_id", f"{u}__{e}"))

            # notionals come directly from the plan
            tu = float(row["long_usd"])   # target underlying (long) notional
            te = float(row["short_usd"])  # target ETF (short) notional

            # short_ratio is "short_notional / long_notional"
            # (used later to size the long off the filled short when short_first=True)
            sr = (te / tu) if tu else 0.0

            # lev_type is optional; derive something reasonable for logging / fills
            lev_mult = float(row.get("Leverage", 0.0) or 0.0)
            if lev_mult <= 1.0:
                lev_type = "CC"
            else:
                lev_type = f"{int(round(lev_mult))}X"


            print("\n" + "-" * 90)
            print(f"[PAIR] {pair_id}")
            print(f"  Target notionals: underlying=${tu:,.2f}  ETF(short)=${te:,.2f}  short_ratio={sr}")
            print(f"  Baseline-protected? baseline[u]={baseline.get(u,0):.0f} baseline[e]={baseline.get(e,0):.0f}")
            print(f"  Current strategy-only qty: u={strat_pos.get(u,0):.2f} e={strat_pos.get(e,0):.2f}")

            if not approve_all:
                ans = input("Approve this pair? [y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
                if ans == "q":
                    break
                if ans == "a":
                    approve_all = True
                if ans not in ("y", "a"):
                    continue

            # prices
            px_u = get_snapshot_price(ib, u, prefer_delayed=prefer_delayed)
            px_e = get_snapshot_price(ib, e, prefer_delayed=prefer_delayed)

            # Convert target notionals -> target shares (ETF short leg)
            target_sh_e = int(te // px_e)
            if target_sh_e <= 0:
                print("[PAIR] Skipping: ETF short size rounds to 0 shares.")
                continue

            order_base_ref = f"{strategy_tag}|{pair_id}"

            # Execute order legs (respect short_first config)
            if short_first:
                filled_e, trade_e = execute_leg(
                    ib,
                    symbol=e,
                    action="SELL",
                    qty=target_sh_e,
                    ref_price=px_e,
                    bps=limit_bps,
                    order_ref=f"{order_base_ref}|ETF_SHORT",
                    exec_cfg=exec_cfg,
                    timeout=timeout,
                    max_retries=max_retries,
                    dry_run=dry_run,
                )

                if filled_e <= 0:
                    print("[PAIR] ETF short leg did not fill. Skipping long leg.")
                    continue

                filled_short_notional = filled_e * px_e
                desired_under_notional = filled_short_notional / abs(sr)
                target_sh_u = int(desired_under_notional // px_u)
                if target_sh_u <= 0:
                    print("[PAIR] Long size rounds to 0. Attempting to buy 1 share as minimum.")
                    target_sh_u = 1

                filled_u, trade_u = execute_leg(
                    ib,
                    symbol=u,
                    action="BUY",
                    qty=target_sh_u,
                    ref_price=px_u,
                    bps=limit_bps,
                    order_ref=f"{order_base_ref}|UNDER_LONG",
                    timeout=timeout,
                    max_retries=max_retries,
                    dry_run=dry_run,
                )
            else:
                # Long-first mode (not typical for borrow realism, but supported)
                # Size long off target notional tu
                target_sh_u = int(tu // px_u)
                if target_sh_u <= 0:
                    print("[PAIR] Long size rounds to 0. Skipping.")
                    continue

                filled_u, trade_u = execute_leg(
                    ib,
                    symbol=u,
                    action="BUY",
                    qty=target_sh_u,
                    ref_price=px_u,
                    bps=limit_bps,
                    order_ref=f"{order_base_ref}|UNDER_LONG",
                    timeout=timeout,
                    max_retries=max_retries,
                    dry_run=dry_run,
                )

                if filled_u <= 0:
                    print("[PAIR] Underlying long leg did not fill. Skipping short leg.")
                    continue

                # Size short off filled long notional and ratio
                filled_long_notional = filled_u * px_u
                desired_short_notional = filled_long_notional * abs(sr)
                target_sh_e = int(desired_short_notional // px_e)
                if target_sh_e <= 0:
                    print("[PAIR] Short size rounds to 0. Skipping.")
                    continue

                filled_e, trade_e = execute_leg(
                    ib,
                    symbol=e,
                    action="SELL",
                    qty=target_sh_e,
                    ref_price=px_e,
                    bps=limit_bps,
                    order_ref=f"{order_base_ref}|ETF_SHORT",
                    timeout=timeout,
                    max_retries=max_retries,
                    dry_run=dry_run,
                )

            # Record fills
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fills_to_append.append(
                {
                    "filled_at": now,
                    "strategy_tag": strategy_tag,
                    "pair_id": pair_id,
                    "lev_type": lev_type,
                    "underlying": u,
                    "etf": e,
                    "short_ratio": sr,
                    "px_under": px_u,
                    "px_etf": px_e,
                    "target_sh_under": target_sh_u,
                    "target_sh_etf_short": target_sh_e,
                    "filled_sh_under": filled_u,
                    "filled_sh_etf_short": filled_e,
                    "filled_under_notional": filled_u * px_u,
                    "filled_etf_notional": filled_e * px_e,
                    "notes": "",
                }
            )

            # Refresh positions for next pair view
            ib_pos = current_ib_positions(ib)
            strat_pos = strategy_position_only(ib_pos, baseline)

        if fills_to_append:
            append_fills(fills_to_append)

        print("[DONE] Execution pass complete.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
