#!/usr/bin/env python3
"""
execute_trade_plan_updated.py

Reads a proposed trade plan CSV and executes bucket-by-bucket (Underlying groups),
optionally in parallel (N buckets at a time).

Key behavior (matches your “working properly” version):
- Execute ONLY ETFs in the PLAN.
- Hedge truth uses ALL HELD ETFs that are in the SCREENED universe (etf_screened_today.csv),
  even if those ETFs are NOT in the plan (e.g., AMDG).
- Underlying hedge is computed from ACTUAL post-ETF positions and leverage from SCREENED.

Parallel safety model:
- One IB() connection per worker thread (unique clientId per worker).
- Logging is guarded by a lock to avoid interleaved writes.
- Plan sanity check: no planned ETF symbol appears in multiple underlyings.

Ctrl+C (SIGINT) behavior:
- First Ctrl+C: stop launching new buckets; workers stop cooperatively between legs;
  in-flight orders are not force-cancelled by default.
- Second Ctrl+C: hard exit.

Usage:
  python execute_trade_plan_parallel.py
  DRY_RUN=1 python execute_trade_plan_parallel.py
"""

from __future__ import annotations

import argparse
import asyncio
import ftplib
import io
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

import pandas as pd
import yaml
from ib_insync import IB, Stock, Order, Trade, MarketOrder, TagValue


# =============================================================================
# Thread-safe printing + shutdown flag
# =============================================================================

PRINT_LOCK = threading.Lock()


def tprint(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


SHUTDOWN = threading.Event()

STOP_FILE = Path("STOP_EXECUTION")

def stop_requested() -> bool:
    return STOP_FILE.exists()


def handle_sigint(signum, frame):
    # First Ctrl+C => graceful; second Ctrl+C => hard exit
    if not stop_requested():
        tprint("\n[CTRL+C] Shutdown requested. Stopping new work…")
        SHUTDOWN.set()
    else:
        tprint("\n[CTRL+C] Forced exit.")
        sys.exit(1)


def ensure_thread_event_loop() -> asyncio.AbstractEventLoop:
    """
    ib_insync relies on an asyncio event loop. Worker threads created by ThreadPoolExecutor
    do not have one by default on Windows, so we must create and set it.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


# =============================================================================
# Exposure logging
# =============================================================================

EXPOSURE_COLS = [
    "ts",
    "run_date",
    "strategy_tag",
    "stage",          # PRE_GROUP, POST_ETF, POST_UNDER_GROUP, FINAL, etc.
    "pair_id",
    "underlying",
    "etf",
    "symbol",         # symbol traded or "PORTFOLIO"
    "delta_sh",
    "filled_sh",
    "fill_avg_px",
    "mark_px",
    "delta_notional",
    "pos_sh",
    "pos_notional",
    "gross_long",
    "gross_short",
    "net_notional",
]


def compute_portfolio_notionals(strat_pos: Dict[str, int], prices: Dict[str, float]) -> Dict[str, float]:
    gross_long = 0.0
    gross_short = 0.0
    net = 0.0
    for sym, sh in strat_pos.items():
        px = prices.get(sym)
        if px is None:
            continue
        notional = float(sh) * float(px)
        net += notional
        if notional >= 0:
            gross_long += notional
        else:
            gross_short += abs(notional)
    return {"gross_long": gross_long, "gross_short": gross_short, "net_notional": net}


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def safe_avg_fill_price(trade: Optional[Trade]) -> Optional[float]:
    try:
        if trade is None:
            return None
        px = float(trade.orderStatus.avgFillPrice or 0)
        return px if px > 0 else None
    except Exception:
        return None


# =============================================================================
# Short availability (IBKR FTP shortstock) – optional precheck
# =============================================================================

def fetch_ibkr_short_availability_map(
    symbols: List[str],
    ftp_host: str = "ftp2.interactivebrokers.com",
    ftp_user: str = "shortstock",
    ftp_pass: str = "",
    ftp_file: str = "usa.txt",
) -> Dict[str, Dict[str, Optional[float]]]:
    want = {s.upper().strip() for s in symbols if str(s).strip()}
    if not want:
        return {}

    ftp = ftplib.FTP(ftp_host)
    ftp.login(user=ftp_user, passwd=ftp_pass)

    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {ftp_file}", buf.write)
    ftp.quit()

    buf.seek(0)
    text = buf.getvalue().decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("#SYM|"):
            header_idx = i
            break
    if header_idx is None:
        tprint("[SHORT] WARNING: no '#SYM|' header; skipping short availability precheck.")
        return {}

    header_cols = [c.strip().lstrip("#").lower() for c in lines[header_idx].split("|")]
    data_lines = lines[header_idx + 1 :]

    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="|", header=None, engine="python")
    n_cols = min(len(header_cols), df.shape[1])
    df = df.iloc[:, :n_cols]
    df.columns = header_cols[:n_cols]

    if "sym" not in df.columns:
        tprint("[SHORT] WARNING: missing 'sym' col; skipping short availability precheck.")
        return {}

    df["sym"] = df["sym"].astype(str).str.upper().str.strip()

    if "available" in df.columns:
        df["available_int"] = pd.to_numeric(df["available"], errors="coerce")
    else:
        df["available_int"] = pd.NA

    fee = pd.to_numeric(df.get("feerate", pd.Series([pd.NA] * len(df))), errors="coerce") / 100.0
    rebate = pd.to_numeric(df.get("rebaterate", pd.Series([pd.NA] * len(df))), errors="coerce") / 100.0
    df["net_borrow_annual"] = (fee - rebate).clip(lower=0)

    sub = df[df["sym"].isin(want)].copy()

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for _, r in sub.iterrows():
        sym = str(r["sym"])
        avail = r.get("available_int", pd.NA)
        borrow = r.get("net_borrow_annual", pd.NA)
        out[sym] = {
            "available": None if pd.isna(avail) else int(avail),
            "borrow": None if pd.isna(borrow) else float(borrow),
        }
    return out


# =============================================================================
# Symbol normalization
# =============================================================================

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


# =============================================================================
# Paths / date helpers
# =============================================================================

def today_str() -> str:
    return date.today().isoformat()


def run_dir(run_date: str) -> Path:
    return Path("data") / "runs" / run_date


def exec_dir(run_date: str) -> Path:
    return run_dir(run_date) / "execution"


# =============================================================================
# IBKR connection & pricing
# =============================================================================

def connect_ib(host: str, port: int, client_id: int) -> IB:
    ensure_thread_event_loop()  # safe even in main thread
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR.")
    # IMPORTANT: only clientId=0 can auto-bind orders
    if client_id == 0:
        ib.reqAutoOpenOrders(True)
    try:
        ib.reqOpenOrders()
        ib.sleep(0.5)
    except Exception:
        pass
    return ib


def safe_price(v) -> Optional[float]:
    try:
        x = float(v)
        return x if x > 0 else None
    except Exception:
        return None


def get_snapshot_price(ib: IB, symbol: str, prefer_delayed: bool = True) -> float:
    sym_u = symbol.upper()
    contract = make_stock(sym_u)
    ib.qualifyContracts(contract)

    def read_ticker(t) -> Optional[float]:
        bid = safe_price(getattr(t, "bid", None)) or safe_price(getattr(t, "delayedBid", None))
        ask = safe_price(getattr(t, "ask", None)) or safe_price(getattr(t, "delayedAsk", None))
        last = safe_price(getattr(t, "last", None)) or safe_price(getattr(t, "delayedLast", None))
        close = safe_price(getattr(t, "close", None)) or safe_price(getattr(t, "delayedClose", None))
        mkt = safe_price(t.marketPrice())
        if bid and ask:
            return (bid + ask) / 2.0
        return last or close or mkt

    def snapshot_with_type(data_type: int) -> Optional[float]:
        ib.reqMarketDataType(data_type)
        t = ib.reqMktData(contract, "", snapshot=True)
        try:
            for _ in range(12):
                if stop_requested():
                    return None
                ib.sleep(0.25)
                px = read_ticker(t)
                if px is not None:
                    return px
        finally:
            try:
                req_id = getattr(t, "reqId", None) or getattr(t, "tickerId", None)
                if isinstance(req_id, int) and req_id > 0:
                    ib.client.cancelMktData(req_id)
            except Exception:
                pass
        return None

    px: Optional[float] = None
    if prefer_delayed:
        px = snapshot_with_type(4) or snapshot_with_type(3)
    else:
        px = snapshot_with_type(1) or snapshot_with_type(3)

    if px is None and not stop_requested():
        for _attempt in range(3):
            if stop_requested():
                break
            if not ib.isConnected():
                for _ in range(40):
                    if stop_requested():
                        break
                    ib.sleep(0.25)
                    if ib.isConnected():
                        break
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="2 D",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False,
                )
            except Exception:
                bars = []
            if bars:
                px = safe_price(bars[-1].close)
                if px is not None:
                    break
            ib.sleep(0.5)

    if px is None:
        raise RuntimeError(f"No usable price for {sym_u}")
    return float(px)


# =============================================================================
# Orders & execution
# =============================================================================

def build_market_order(action: str, qty: int, order_ref: str) -> Order:
    o = MarketOrder(action.upper(), int(qty))
    o.tif = "DAY"
    o.transmit = True
    o.orderRef = order_ref
    return o


def build_adaptive_market_order(action: str, qty: int, order_ref: str, priority: str = "Normal") -> Order:
    o = Order()
    o.action = action.upper()
    o.totalQuantity = int(qty)
    o.orderType = "MKT"
    o.tif = "DAY"
    o.transmit = True
    o.orderRef = order_ref
    o.algoStrategy = "Adaptive"
    o.algoParams = [TagValue("adaptivePriority", str(priority))]
    return o


TERMINAL: Set[str] = {"filled", "cancelled", "inactive"}
ACCEPTED: Set[str] = {"presubmitted", "submitted", "filled", "pendingsubmit"}


def wait_for_trade_terminal(ib: IB, trade: Trade, timeout: float = 180.0) -> Trade:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stop_requested():
            return trade
        st = (trade.orderStatus.status or "").lower()
        if st in TERMINAL:
            return trade
        try:
            ib.reqAllOpenOrders()
        except Exception:
            pass
        ib.sleep(0.2)
    return trade


def wait_for_trade_accepted(ib: IB, trade: Trade, timeout: float = 45.0) -> Tuple[bool, Trade]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stop_requested():
            return False, trade
        st = (trade.orderStatus.status or "").lower()
        if st in ACCEPTED:
            return True, trade
        if st in TERMINAL:
            return False, trade
        try:
            ib.reqAllOpenOrders()
        except Exception:
            pass
        ib.sleep(0.2)
    return False, trade


def execute_leg(
    *,
    ib: IB,
    symbol: str,
    action: str,
    qty: int,
    ref_price: float,
    bps: float,
    order_ref: str,
    exec_cfg: Dict,
    timeout: float = 90.0,
    max_retries: int = 3,
    dry_run: bool = False,
    context: str = "",
) -> Tuple[int, Optional[Trade]]:
    if qty <= 0:
        return 0, None
    if stop_requested():
        tprint(f"[{context}][LEG] Shutdown active; skipping {symbol} {action} qty={qty}")
        return 0, None

    contract = make_stock(symbol)
    ib.qualifyContracts(contract)

    filled_total = 0
    last_trade: Optional[Trade] = None

    order_style = str(exec_cfg.get("order_style", "ADAPTIVE_MKT")).strip().upper()
    market_done_timeout = float(exec_cfg.get("market_done_timeout_sec", 180.0))

    def last_ib_error(trade: Optional[Trade]) -> Tuple[Optional[int], Optional[str]]:
        if trade is None:
            return None, None
        try:
            logs = getattr(trade, "log", None) or []
            for entry in reversed(logs):
                code = getattr(entry, "errorCode", None)
                msg = getattr(entry, "message", None)
                if code and int(code) != 0:
                    return int(code), str(msg or "")
        except Exception:
            pass
        return None, None

    for attempt in range(1, max_retries + 1):
        if stop_requested():
            tprint(f"[{context}][LEG] Shutdown during retries; stopping {symbol}")
            break

        remain = qty - filled_total
        if remain <= 0:
            break

        if order_style == "ADAPTIVE_MKT":
            if "|UNDER_DELTA" in order_ref:
                priority = "Urgent"
            else:
                priority = "Patient" if attempt == 1 else ("Normal" if attempt == 2 else "Urgent")

            o = build_adaptive_market_order(
                action=action,
                qty=remain,
                order_ref=f"{order_ref}|att{attempt}|ADAPTIVE_MKT",
                priority=priority,
            )
            px_str = f"ADAPTIVE_MKT({priority})"
        else:
            o = build_market_order(
                action=action,
                qty=remain,
                order_ref=f"{order_ref}|att{attempt}|MKT",
            )
            px_str = "MKT"

        ctx = f"[{context}]" if context else ""
        tprint(
            f"{ctx}[LEG] {symbol} {action} qty={remain} px={px_str} "
            f"refTag={o.orderRef} clientId={ib.client.clientId}"
        )

        if dry_run:
            filled_total += remain
            continue

        trade = ib.placeOrder(contract, o)
        last_trade = trade

        try:
            ib.reqAllOpenOrders()
        except Exception:
            pass
        ib.sleep(0.3)

        accepted, trade = wait_for_trade_accepted(
            ib, trade, timeout=float(exec_cfg.get("market_accept_timeout_sec", 45.0))
        )

        if not accepted:
            st = (trade.orderStatus.status or "")
            code, msg = last_ib_error(trade)

            # HARD FAIL: cannot short this contract
            if code == 201 and "not available for short sale" in (msg or "").lower():
                tprint(
                    f"{ctx}[LEG] {symbol} HARD_REJECT code=201 not shortable; "
                    f"no more retries. status={st} orderId={trade.order.orderId} ref={trade.order.orderRef}"
                )
                break

            tprint(f"{ctx}[LEG] {symbol} not accepted (status={st} code={code} msg={msg}); retrying.")
            continue

        trade = wait_for_trade_terminal(ib, trade, timeout=market_done_timeout)
        status = (trade.orderStatus.status or "").lower()
        filled = int(trade.orderStatus.filled or 0)
        remaining = trade.orderStatus.remaining
        remaining = None if remaining is None else int(remaining)

        order_id = getattr(trade.order, "orderId", None)
        perm_id = getattr(trade.orderStatus, "permId", None) or getattr(trade.order, "permId", None)
        tprint(
            f"{ctx}[LEG] status={status} filled={filled} remaining={remaining} "
            f"orderId={order_id} permId={perm_id} refTag={trade.order.orderRef}"
        )

        filled_total = min(qty, filled_total + max(0, filled))

        if status not in TERMINAL and remaining and remaining > 0:
            try:
                ib.cancelOrder(trade.order)
            except Exception:
                pass

    return int(filled_total), last_trade


# =============================================================================
# Baseline snapshot mechanics
# =============================================================================

def load_baseline_qty(path: Path) -> Dict[str, float]:
    if not path.exists():
        tprint(f"[BASELINE] No baseline file found at {path}. Treating baseline as empty.")
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    if "qty" not in df.columns:
        raise ValueError(f"Baseline file {path} missing required column 'qty'. Columns={list(df.columns)}")
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


# =============================================================================
# IO helpers
# =============================================================================

def append_fills(rows: List[dict], fills_path: Path) -> None:
    fills_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if fills_path.exists():
        df_old = pd.read_csv(fills_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(fills_path, index=False)
    tprint(f"[FILLS] Appended {len(rows)} rows -> {fills_path}")


def resolve_plan_path(run_date: str, paths_cfg: dict) -> Path:
    dated = run_dir(run_date) / "proposed_trades.csv"
    if dated.exists():
        return dated
    return Path(paths_cfg.get("proposed_trades_csv", "data/proposed_trades.csv"))


def resolve_fills_path(run_date: str, paths_cfg: dict) -> Path:
    return exec_dir(run_date) / "fills.csv"


def write_execution_snapshot(run_date: str, df: pd.DataFrame, name: str) -> None:
    p = exec_dir(run_date) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    tprint(f"[EXEC] Wrote {p}")


# =============================================================================
# Sizing helpers
# =============================================================================

def target_shares_from_usd(notional_usd: float, px: float) -> int:
    if px <= 0:
        raise ValueError("Price must be > 0")
    return int(notional_usd / px)


def fmt_dollars(x: float) -> str:
    return f"${x:,.0f}"


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # Register Ctrl+C handler early
    signal.signal(signal.SIGINT, handle_sigint)

    CONFIG_YML = Path("config/strategy_config.yml")
    if not CONFIG_YML.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_YML}")

    cfg = yaml.safe_load(CONFIG_YML.read_text(encoding="utf-8")) or {}
    ibkr_cfg = cfg.get("ibkr", {}) or {}
    strat_cfg = cfg.get("strategy", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}

    parallel_n = max(1, int(exec_cfg.get("parallel_n", 1)))
    auto_approve_cli = bool(exec_cfg.get("auto_approve", False))
    run_date = today_str()

    strategy_tag = str(strat_cfg.get("tag", "")).strip()
    if not strategy_tag:
        raise ValueError("Missing strategy.tag in config/strategy_config.yml")

    host = str(ibkr_cfg.get("host", "127.0.0.1"))
    port = int(ibkr_cfg.get("port", 7497))
    client_id = int(ibkr_cfg.get("client_id", 3))
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))

    exec_cfg = dict(exec_cfg)
    exec_cfg["prefer_delayed"] = prefer_delayed

    limit_bps = float(exec_cfg.get("limit_bps", 10.0))
    timeout = float(exec_cfg.get("timeout_sec", 90))
    short_first = bool(exec_cfg.get("short_first", True))
    max_retries = int(exec_cfg.get("max_retries", 3))

    if "DRY_RUN" in os.environ:
        dry_run = bool(int(os.getenv("DRY_RUN", "0")))
    else:
        dry_run = bool(exec_cfg.get("dry_run", False))

    if dry_run:
        tprint("[DRY_RUN] Enabled. No orders will be placed.")

    plan_path = resolve_plan_path(run_date, paths_cfg)
    baseline_csv = Path(paths_cfg.get("baseline_csv", "data/baseline_snapshot.csv"))
    fills_path = resolve_fills_path(run_date, paths_cfg)

    if not plan_path.exists():
        raise FileNotFoundError(f"Trade plan not found: {plan_path}")

    plan = pd.read_csv(plan_path)
    if "strategy_tag" in plan.columns:
        plan = plan[plan["strategy_tag"].astype(str) == strategy_tag].copy()
    if plan.empty:
        raise ValueError(f"No rows in {plan_path} for strategy_tag={strategy_tag}")

    # Normalize + stable sort
    plan = plan.reset_index(drop=True)
    plan["_orig_idx"] = plan.index
    plan["Underlying"] = plan["Underlying"].astype(str).str.upper().str.strip()
    plan["ETF"] = plan["ETF"].astype(str).str.upper().str.strip()
    plan = plan.sort_values(by=["Underlying", "_orig_idx"], kind="mergesort").reset_index(drop=True)
    plan = plan.drop(columns=["_orig_idx"]).reset_index(drop=True)

    baseline = load_baseline_qty(baseline_csv)
    exec_dir(run_date).mkdir(parents=True, exist_ok=True)
    exposure_csv = exec_dir(run_date) / "exposure_log.csv"
    exposure_jsonl = exec_dir(run_date) / "exposure_log.jsonl"

    log_lock = threading.Lock()

    # Coordinator connection (clientId from config)
    ib = connect_ib(host, port, client_id)
    try:
        ib_pos = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)

        tprint(
            f"[POS] current IB symbols={len(ib_pos)}; "
            f"baseline symbols={len(baseline)}; "
            f"strategy-only symbols={len(strat_pos)}"
        )
        tprint(f"[PLAN] Using: {plan_path}")
        tprint(f"[BASELINE] Using: {baseline_csv}")
        tprint(f"[EXEC] Writing to: {exec_dir(run_date)}")

        fills_to_append: List[dict] = []

        # ---------------------------
        # PLAN-derived maps (execution truth)
        # ---------------------------
        if "Leverage" not in plan.columns:
            raise ValueError("Plan missing required column: Leverage")

        plan["ETF_U"] = plan["ETF"].astype(str).str.upper().str.strip()
        plan["UNDER_U"] = plan["Underlying"].astype(str).str.upper().str.strip()

        leverage_by_etf_plan: Dict[str, float] = {}
        under_to_etfs_planned: Dict[str, Set[str]] = {}

        for _, r in plan.iterrows():
            e = str(r["ETF_U"])
            u = str(r["UNDER_U"])
            lev = float(r["Leverage"])
            under_to_etfs_planned.setdefault(u, set()).add(e)
            if e in leverage_by_etf_plan and abs(leverage_by_etf_plan[e] - lev) > 1e-9:
                raise ValueError(f"Leverage mismatch for {e} in plan: {leverage_by_etf_plan[e]} vs {lev}")
            leverage_by_etf_plan[e] = lev

        # ---------------------------
        # SCREENED universe maps (hedge truth)
        # ---------------------------
        screened_csv = Path(paths_cfg.get("screened_csv", "data/etf_screened_today.csv"))
        if not screened_csv.exists():
            raise FileNotFoundError(f"Screened universe not found: {screened_csv}")

        screened = pd.read_csv(screened_csv)
        need_cols = {"Underlying", "ETF", "Leverage"}
        if not need_cols.issubset(set(screened.columns)):
            raise ValueError(f"{screened_csv} missing required columns {need_cols}. Columns={list(screened.columns)}")

        screened["Underlying"] = screened["Underlying"].astype(str).str.upper().str.strip()
        screened["ETF"] = screened["ETF"].astype(str).str.upper().str.strip()

        under_to_etfs_all: Dict[str, Set[str]] = {}
        leverage_by_etf_all: Dict[str, float] = {}
        for _, r in screened.iterrows():
            u = str(r["Underlying"])
            e = str(r["ETF"])
            lev = float(r["Leverage"])
            if not u or u == "NAN" or not e or e == "NAN":
                continue
            under_to_etfs_all.setdefault(u, set()).add(e)
            leverage_by_etf_all[e] = lev

        etf_to_under_all: Dict[str, str] = {}
        for u, etfs in under_to_etfs_all.items():
            for e in etfs:
                prev = etf_to_under_all.get(e)
                if prev is not None and prev != u:
                    raise ValueError(f"[SCREENED] ETF {e} maps to multiple underlyings: {prev} and {u}")
                etf_to_under_all[e] = u

        # ---------------------------
        # Plan sanity: ETF symbol appears in exactly one bucket (parallel safety)
        # ---------------------------
        etf_seen: Dict[str, str] = {}
        for _, r in plan.iterrows():
            e = str(r["ETF_U"])
            u = str(r["UNDER_U"])
            prev = etf_seen.get(e)
            if prev is not None and prev != u:
                raise ValueError(f"Parallel unsafe: planned ETF {e} appears under multiple underlyings {prev} and {u}.")
            etf_seen[e] = u

        # ---------------------------
        # Short availability snapshot (PLAN ETFs only)
        # ---------------------------
        etf_symbols_plan = sorted(set(plan["ETF_U"].astype(str).str.upper()))
        short_map: Dict[str, Dict[str, Optional[float]]] = {}
        try:
            short_map = fetch_ibkr_short_availability_map(etf_symbols_plan)
            tprint(f"[SHORT] Loaded availability for {len(short_map)}/{len(etf_symbols_plan)} plan ETFs from IBKR FTP.")
        except Exception as ex:
            tprint(f"[SHORT] WARNING: short availability precheck failed ({ex}); continuing without it.")
            short_map = {}

        # ---------------------------
        # Prefetch prices for plan symbols + held mapped ETFs (screened)
        # ---------------------------
        held_etfs_by_under: Dict[str, Set[str]] = {}
        for sym, sh0 in strat_pos.items():
            sh = int(round(float(sh0)))
            if sh == 0:
                continue
            u = etf_to_under_all.get(sym)
            if u is None:
                continue
            held_etfs_by_under.setdefault(u, set()).add(sym)

        plan_symbols = set(plan["UNDER_U"].tolist()) | set(plan["ETF_U"].tolist())
        held_mapped_etfs = set().union(*held_etfs_by_under.values()) if held_etfs_by_under else set()
        symbols = sorted(plan_symbols | held_mapped_etfs)

        prices: Dict[str, float] = {}
        for s in symbols:
            if stop_requested():
                tprint("[SHUTDOWN] Stopping during price prefetch.")
                break
            prices[s] = get_snapshot_price(ib, s, prefer_delayed=prefer_delayed)

        if stop_requested():
            tprint("[SHUTDOWN] Exiting before execution.")
            return

        px_df = pd.DataFrame([{"symbol": k, "price": v} for k, v in prices.items()]).sort_values("symbol")
        write_execution_snapshot(run_date, px_df, "prices_snapshot.csv")

        # ---------------------------
        # Skip underlyings whose TOTAL target shares would round/truncate to 0.
        # ---------------------------
        skip_underlyings = set()
        for u_sym, grp in plan.groupby(plan["UNDER_U"].astype(str).str.upper()):
            px_u = float(prices[u_sym])
            total_u_sh = 0
            for _, rr in grp.iterrows():
                tu = float(rr["long_usd"])
                total_u_sh += target_shares_from_usd(tu, px_u)
            if total_u_sh == 0:
                skip_underlyings.add(u_sym)

        if skip_underlyings:
            tprint(f"[SKIP] Underlyings with rounded-to-0 total target shares: {sorted(skip_underlyings)}")
            df_skip = plan[plan["UNDER_U"].astype(str).str.upper().isin(skip_underlyings)].copy()
            write_execution_snapshot(run_date, df_skip, "skipped_underlyings_rounded0.csv")
            plan = plan[~plan["UNDER_U"].astype(str).str.upper().isin(skip_underlyings)].copy()
            if plan.empty:
                tprint("[SKIP] All rows skipped due to rounded-to-0 underlying sizing. Nothing to execute.")
                return

        # ---------------------------
        # Logging helper (thread-safe)
        # ---------------------------
        def log_exposure_event(
            *,
            stage: str,
            pair_id: str,
            underlying: str,
            etf: str,
            symbol: str,
            delta_sh: int,
            filled_sh: int,
            trade: Optional[Trade],
        ):
            # Coordinator snapshot (portfolio-level)
            ib_pos_now = current_ib_positions(ib)
            strat_pos_now = strategy_position_only(ib_pos_now, baseline)

            port = compute_portfolio_notionals(
                {k: int(round(float(v))) for k, v in strat_pos_now.items()},
                prices,
            )

            mark_px = float(prices.get(symbol)) if prices.get(symbol) is not None else None
            fill_px = safe_avg_fill_price(trade)
            used_px = fill_px if fill_px is not None else mark_px

            pos_sh = int(round(float(strat_pos_now.get(symbol, 0.0)))) if symbol != "PORTFOLIO" else 0
            pos_notional = (pos_sh * mark_px) if (mark_px is not None and symbol != "PORTFOLIO") else None
            delta_notional = (int(filled_sh) * float(used_px)) if (used_px is not None) else None

            row = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_date": run_date,
                "strategy_tag": strategy_tag,
                "stage": stage,
                "pair_id": pair_id,
                "underlying": underlying,
                "etf": etf,
                "symbol": symbol,
                "delta_sh": int(delta_sh),
                "filled_sh": int(filled_sh),
                "fill_avg_px": fill_px,
                "mark_px": mark_px,
                "delta_notional": delta_notional,
                "pos_sh": pos_sh,
                "pos_notional": pos_notional,
                **port,
            }

            with log_lock:
                append_csv_row(exposure_csv, row)
                append_jsonl(exposure_jsonl, row)

        # ---------------------------
        # Shared helpers for workers
        # ---------------------------
        def get_lev_or_raise(etf: str) -> float:
            lev = leverage_by_etf_all.get(etf)
            if lev is None:
                raise ValueError(f"Missing leverage for ETF {etf} in screened universe; cannot hedge accurately.")
            return float(lev)

        def ensure_price_worker(ib_local: IB, sym: str) -> float:
            px = prices.get(sym)
            if px is not None:
                return float(px)
            px2 = get_snapshot_price(ib_local, sym, prefer_delayed=prefer_delayed)
            with log_lock:
                prices[sym] = float(px2)
            return float(px2)

        def print_bucket_exposure_breakdown(
            ib_local: IB,
            baseline_local: Dict[str, float],
            etf_to_under_local: Dict[str, str],
            u_sym: str,
            title: str,
        ) -> None:
            ib_pos_now = current_ib_positions(ib_local)
            strat_now_raw = strategy_position_only(ib_pos_now, baseline_local)
            strat_now = {k: int(round(float(v))) for k, v in strat_now_raw.items()}

            px_u = ensure_price_worker(ib_local, u_sym)
            u_sh = int(strat_now.get(u_sym, 0))
            E = u_sh * px_u

            held = []
            for sym, sh in strat_now.items():
                if sh == 0:
                    continue
                u = etf_to_under_local.get(sym)
                if u == u_sym:
                    held.append(sym)

            tprint(f"\n[{title}] Exposure breakdown for {u_sym}")
            tprint(f"  Underlying {u_sym}: sh={u_sh:+d} px={px_u:.4f} MV={fmt_dollars(u_sh*px_u)}")

            if held:
                tprint("  Held screened ETFs mapped to this underlying:")
                for etf in sorted(held):
                    sh = int(strat_now.get(etf, 0))
                    px_e = ensure_price_worker(ib_local, etf)
                    mv = sh * px_e
                    lev = get_lev_or_raise(etf)
                    contrib = mv * lev
                    E += contrib
                    tprint(f"    {etf}: sh={sh:+d} px={px_e:.4f} MV={fmt_dollars(mv)} lev={lev:+.2f} contrib={fmt_dollars(contrib)}")
            else:
                tprint("  Held screened ETFs mapped to this underlying: (none)")

            resid_dollars = E
            resid_sh_eq = resid_dollars / px_u if px_u else 0.0
            tprint(f"  Residual: {fmt_dollars(resid_dollars)}  ({resid_sh_eq:+.2f} {u_sym} shares eq)\n")

        # ---------------------------
        # Build bucket list (plan order)
        # ---------------------------
        underlying_order: List[str] = []
        seen_u: Set[str] = set()
        for u_sym in plan["UNDER_U"].astype(str).str.upper().tolist():
            if u_sym not in seen_u:
                underlying_order.append(u_sym)
                seen_u.add(u_sym)

        buckets: List[Tuple[str, pd.DataFrame]] = []
        for u_sym in underlying_order:
            grp = plan[plan["UNDER_U"].astype(str).str.upper() == u_sym].copy()
            if not grp.empty:
                buckets.append((u_sym, grp))

        # ---------------------------
        # Approvals (serial) -> Approved buckets
        # ---------------------------
        approved: List[Tuple[str, pd.DataFrame]] = []
        if parallel_n > 1 and not auto_approve_cli:
            for (u_sym, grp) in buckets:
                if stop_requested():
                    break
                ans = input(f"Approve bucket for {u_sym}? (y/n/q): ").strip().lower()
                if ans == "q":
                    SHUTDOWN.set()
                    break
                if ans == "y":
                    approved.append((u_sym, grp))
        else:
            approved = buckets

        if not approved or stop_requested():
            tprint("[DONE] No approved buckets to execute (or shutdown requested).")
            return

        # ---------------------------
        # Worker: execute a single underlying bucket
        # ---------------------------
        def execute_underlying_bucket(u_sym: str, grp: pd.DataFrame, worker_idx: int) -> List[dict]:
            ensure_thread_event_loop()
            if stop_requested():
                tprint(f"[{u_sym}] Shutdown before start; skipping bucket.")
                return []

            ib_local = connect_ib(host, port, client_id + 100 + worker_idx)
            local_fills: List[dict] = []
            try:
                ib_pos_local = current_ib_positions(ib_local)
                strat_pos_local = strategy_position_only(ib_pos_local, baseline)

                px_u = ensure_price_worker(ib_local, u_sym)

                bucket_target_etf_sh: Dict[str, int] = {}
                bucket_target_under_sh: int = 0

                for _, row in grp.iterrows():
                    e_sym = str(row["ETF_U"]).upper()
                    tu = float(row["long_usd"])
                    te = float(row["short_usd"])

                    px_e = ensure_price_worker(ib_local, e_sym)

                    bucket_target_under_sh += target_shares_from_usd(tu, px_u)
                    bucket_target_etf_sh[e_sym] = int(
                        bucket_target_etf_sh.get(e_sym, 0) + target_shares_from_usd(te, px_e)
                    )

                cur_u_sh = int(round(float(strat_pos_local.get(u_sym, 0.0))))
                target_u_sh = int(bucket_target_under_sh)

                bucket_delta_etf: Dict[str, int] = {}
                for e_sym, tgt_sh in bucket_target_etf_sh.items():
                    cur_e_sh = int(round(float(strat_pos_local.get(e_sym, 0.0))))
                    bucket_delta_etf[e_sym] = int(tgt_sh - cur_e_sh)

                delta_under_naive = int(target_u_sh - cur_u_sh)

                tprint("\n" + "=" * 110)
                tprint(f"[GROUP] Underlying bucket: {u_sym}")
                tprint(f"  Mark px: {u_sym}={px_u:.4f}")
                tprint(f"  Bucket pairs: {len(grp)}")
                tprint(f"  Planned target underlying shares (sum of rows): {target_u_sh:+d}")
                tprint(f"  Current strategy-only underlying shares: {cur_u_sh:+d}")
                tprint(f"  Naive underlying delta to planned target: {delta_under_naive:+d}")

                for e_sym in sorted(bucket_target_etf_sh.keys()):
                    px_e = ensure_price_worker(ib_local, e_sym)
                    cur_e_sh = int(round(float(strat_pos_local.get(e_sym, 0.0))))
                    tgt_e_sh = int(bucket_target_etf_sh[e_sym])
                    d_e = int(bucket_delta_etf[e_sym])
                    lev = leverage_by_etf_plan.get(e_sym)
                    if lev is None:
                        raise ValueError(f"Missing leverage for planned ETF {e_sym} in plan columns.")
                    tprint(
                        f"  ETF {e_sym}: px={px_e:.4f} lev={float(lev):+,.2f} "
                        f"cur={cur_e_sh:+d} tgt={tgt_e_sh:+d} delta={d_e:+d}"
                    )

                print_bucket_exposure_breakdown(
                    ib_local,
                    baseline,
                    etf_to_under_all,
                    u_sym,
                    title="PRE_GROUP_HEDGE_TRUTH",
                )

                log_exposure_event(
                    stage="PRE_GROUP",
                    pair_id=f"{u_sym}__GROUP",
                    underlying=u_sym,
                    etf="",
                    symbol="PORTFOLIO",
                    delta_sh=0,
                    filled_sh=0,
                    trade=None,
                )

                def exec_delta_local(symbol: str, delta: int, px: float, order_ref: str) -> Tuple[int, Optional[Trade]]:
                    if delta == 0:
                        return 0, None
                    action = "BUY" if delta > 0 else "SELL"
                    qty = abs(delta)
                    return execute_leg(
                        ib=ib_local,
                        symbol=symbol,
                        action=action,
                        qty=qty,
                        ref_price=px,
                        bps=limit_bps,
                        order_ref=order_ref,
                        exec_cfg=exec_cfg,
                        timeout=timeout,
                        max_retries=max_retries,
                        dry_run=dry_run,
                        context=u_sym,
                    )

                etf_items = list(bucket_delta_etf.items())

                def _etf_sort_key(item):
                    sym, d = item
                    return (0 if d < 0 else 1, sym)

                if short_first:
                    etf_items.sort(key=_etf_sort_key)
                else:
                    etf_items.sort(key=lambda x: x[0])

                for e_sym, d_e in etf_items:
                    if stop_requested():
                        tprint(f"[{u_sym}] Shutdown during ETF loop; aborting bucket.")
                        return local_fills

                    if d_e == 0:
                        continue

                    sm = short_map.get(e_sym)
                    if sm and sm.get("available") is not None and d_e < 0 and abs(d_e) > int(sm["available"]):
                        tprint(
                            f"[SHORT] WARNING: {e_sym} wants {abs(d_e)} shares short, "
                            f"but IBKR file shows only {sm['available']} available."
                        )

                    px_e = ensure_price_worker(ib_local, e_sym)
                    order_ref = f"{strategy_tag}|{u_sym}__GROUP|{e_sym}|ETF_DELTA"

                    filled_e_abs, trade_e = exec_delta_local(e_sym, d_e, px_e, order_ref)
                    filled_e = -filled_e_abs if d_e < 0 else filled_e_abs

                    log_exposure_event(
                        stage="POST_ETF",
                        pair_id=f"{u_sym}__GROUP",
                        underlying=u_sym,
                        etf=e_sym,
                        symbol=e_sym,
                        delta_sh=d_e,
                        filled_sh=filled_e,
                        trade=trade_e,
                    )

                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    local_fills.append(
                        {
                            "filled_at": now,
                            "run_date": run_date,
                            "strategy_tag": strategy_tag,
                            "pair_id": f"{u_sym}__GROUP",
                            "underlying": u_sym,
                            "etf": e_sym,
                            "px_under": px_u,
                            "px_etf": px_e,
                            "target_sh_under": target_u_sh,
                            "target_sh_etf": int(bucket_target_etf_sh.get(e_sym, 0)),
                            "delta_sh_under": 0,
                            "delta_sh_etf": d_e,
                            "filled_sh_under": 0,
                            "filled_sh_etf": filled_e,
                            "notes": "GROUP_ETF",
                        }
                    )

                print_bucket_exposure_breakdown(
                    ib_local,
                    baseline,
                    etf_to_under_all,
                    u_sym,
                    title="POST_ETFS_HEDGE_TRUTH",
                )

                def compute_bucket_resid_sh_local(u: str) -> float:
                    ib_pos_now = current_ib_positions(ib_local)
                    strat_now_raw = strategy_position_only(ib_pos_now, baseline)
                    strat_now = {k: int(round(float(v))) for k, v in strat_now_raw.items()}

                    px_u2 = ensure_price_worker(ib_local, u)
                    u_sh2 = int(strat_now.get(u, 0))
                    E = u_sh2 * px_u2

                    for sym, sh0 in strat_now.items():
                        sh = int(sh0)
                        if sh == 0:
                            continue
                        uu = etf_to_under_all.get(sym)
                        if uu != u:
                            continue
                        px_e2 = ensure_price_worker(ib_local, sym)
                        lev2 = get_lev_or_raise(sym)
                        E += sh * px_e2 * lev2

                    return E / px_u2

                resid_before = compute_bucket_resid_sh_local(u_sym)
                delta_under_hedge = int(round(-resid_before))
                tprint(f"[HEDGE] {u_sym}: resid_before={resid_before:+.2f}sh -> delta_under_hedge={delta_under_hedge:+d} sh")

                if stop_requested():
                    tprint(f"[{u_sym}] Shutdown before underlying hedge; aborting bucket.")
                    return local_fills

                if delta_under_hedge != 0:
                    order_ref = f"{strategy_tag}|{u_sym}__GROUP|UNDER_DELTA"
                    filled_u_abs, trade_u = exec_delta_local(u_sym, delta_under_hedge, px_u, order_ref)
                    filled_u = filled_u_abs if delta_under_hedge > 0 else -filled_u_abs

                    log_exposure_event(
                        stage="POST_UNDER_GROUP",
                        pair_id=f"{u_sym}__GROUP",
                        underlying=u_sym,
                        etf="",
                        symbol=u_sym,
                        delta_sh=delta_under_hedge,
                        filled_sh=filled_u,
                        trade=trade_u,
                    )

                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    local_fills.append(
                        {
                            "filled_at": now,
                            "run_date": run_date,
                            "strategy_tag": strategy_tag,
                            "pair_id": f"{u_sym}__GROUP",
                            "underlying": u_sym,
                            "etf": "",
                            "px_under": px_u,
                            "px_etf": None,
                            "target_sh_under": target_u_sh,
                            "target_sh_etf": None,
                            "delta_sh_under": delta_under_hedge,
                            "delta_sh_etf": 0,
                            "filled_sh_under": filled_u,
                            "filled_sh_etf": 0,
                            "notes": f"GROUP_UNDER_HEDGE resid_before={resid_before:+.2f}sh",
                        }
                    )
                else:
                    tprint(f"[GROUP] {u_sym}: already net-flat within rounding (resid_before={resid_before:+.2f}sh).")

                print_bucket_exposure_breakdown(
                    ib_local,
                    baseline,
                    etf_to_under_all,
                    u_sym,
                    title="POST_UNDER_HEDGE_TRUTH",
                )

                resid_after = compute_bucket_resid_sh_local(u_sym)
                tprint(f"[GROUP_NET] {u_sym}: resid_before={resid_before:+.2f}sh resid_after={resid_after:+.2f}sh")
                return local_fills

            finally:
                try:
                    ib_local.disconnect()
                except Exception:
                    pass

        # ---------------------------
        # Execute buckets (serial or parallel) with Ctrl+C support
        # ---------------------------
        if parallel_n == 1:
            for idx, (u_sym, grp) in enumerate(approved):
                if stop_requested():
                    tprint("[SHUTDOWN] Stopping before next bucket (serial).")
                    break
                fills_to_append.extend(execute_underlying_bucket(u_sym, grp, worker_idx=idx))
        else:
            with ThreadPoolExecutor(max_workers=parallel_n) as ex:
                futs = []
                for idx, (u_sym, grp) in enumerate(approved):
                    if stop_requested():
                        tprint("[SHUTDOWN] Skipping remaining buckets (not submitting new work).")
                        break
                    futs.append(ex.submit(execute_underlying_bucket, u_sym, grp, idx))

                try:
                    for fut in as_completed(futs):
                        if stop_requested():
                            tprint("[SHUTDOWN] Not waiting for remaining futures to complete.")
                            break
                        fills_to_append.extend(fut.result())
                except KeyboardInterrupt:
                    handle_sigint(None, None)

        # Final portfolio exposure snapshot (only if not shutdown mid-flight; adjust if you want always)
        if not stop_requested():
            log_exposure_event(
                stage="FINAL",
                pair_id="FINAL",
                underlying="",
                etf="",
                symbol="PORTFOLIO",
                delta_sh=0,
                filled_sh=0,
                trade=None,
            )

        if fills_to_append:
            append_fills(fills_to_append, fills_path)

        if stop_requested():
            tprint("[DONE] Exiting due to shutdown request.")
        else:
            tprint("[DONE] Execution pass complete.")

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# execute_trade_plan_parallel.py

# Reads a proposed trade plan CSV and executes bucket-by-bucket (Underlying groups),
# optionally in parallel (N buckets at a time).

# Key behavior (matches your “working properly” version):
# - Execute ONLY ETFs in the PLAN.
# - Hedge truth uses ALL HELD ETFs that are in the SCREENED universe (etf_screened_today.csv),
#   even if those ETFs are NOT in the plan (e.g., AMDG).
# - Underlying hedge is computed from ACTUAL post-ETF positions and leverage from SCREENED.

# Parallel safety model:
# - One IB() connection per worker thread (unique clientId per worker).
# - Logging is guarded by a lock to avoid interleaved writes.
# - Plan sanity check: no planned ETF symbol appears in multiple underlyings.

# Usage:
#   python execute_trade_plan_parallel.py --run-date 2026-01-20 --strategy-tag ETF_LS
#   python execute_trade_plan_parallel.py --run-date 2026-01-20 --strategy-tag ETF_LS --parallel 5 --auto-approve
#   DRY_RUN=1 python execute_trade_plan_parallel.py --run-date 2026-01-20 --strategy-tag ETF_LS --parallel 5 --auto-approve
# """

# from __future__ import annotations

# import argparse
# import ftplib
# import io
# import json
# import os
# import threading
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from datetime import date, datetime
# from pathlib import Path
# from typing import Dict, Optional, Tuple, List, Set

# import pandas as pd
# import yaml
# from ib_insync import IB, Stock, Order, Trade, MarketOrder, TagValue
# import asyncio

# import asyncio

# def ensure_thread_event_loop() -> asyncio.AbstractEventLoop:
#     """
#     ib_insync relies on an asyncio event loop. Worker threads created by ThreadPoolExecutor
#     do not have one by default on Windows, so we must create and set it.
#     """
#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#     return loop

# PRINT_LOCK = threading.Lock()

# def tprint(msg: str) -> None:
#     with PRINT_LOCK:
#         print(msg, flush=True)

# import signal
# import sys

# SHUTDOWN = threading.Event()


# # ---------------------------
# # Exposure logging
# # ---------------------------

# EXPOSURE_COLS = [
#     "ts",
#     "run_date",
#     "strategy_tag",
#     "stage",          # PRE_GROUP, POST_ETF, POST_UNDER_GROUP, FINAL, etc.
#     "pair_id",
#     "underlying",
#     "etf",
#     "symbol",         # symbol traded or "PORTFOLIO"
#     "delta_sh",
#     "filled_sh",
#     "fill_avg_px",
#     "mark_px",
#     "delta_notional",
#     "pos_sh",
#     "pos_notional",
#     "gross_long",
#     "gross_short",
#     "net_notional",
# ]


# def compute_portfolio_notionals(strat_pos: Dict[str, int], prices: Dict[str, float]) -> Dict[str, float]:
#     gross_long = 0.0
#     gross_short = 0.0
#     net = 0.0
#     for sym, sh in strat_pos.items():
#         px = prices.get(sym)
#         if px is None:
#             continue
#         notional = float(sh) * float(px)
#         net += notional
#         if notional >= 0:
#             gross_long += notional
#         else:
#             gross_short += abs(notional)
#     return {"gross_long": gross_long, "gross_short": gross_short, "net_notional": net}


# def append_csv_row(path: Path, row: dict) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     df = pd.DataFrame([row])
#     if path.exists():
#         df.to_csv(path, mode="a", header=False, index=False)
#     else:
#         df.to_csv(path, mode="w", header=True, index=False)


# def append_jsonl(path: Path, row: dict) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(row, default=str) + "\n")


# def safe_avg_fill_price(trade: Optional[Trade]) -> Optional[float]:
#     try:
#         if trade is None:
#             return None
#         px = float(trade.orderStatus.avgFillPrice or 0)
#         return px if px > 0 else None
#     except Exception:
#         return None


# # ---------------------------
# # Short availability (IBKR FTP shortstock) – optional precheck
# # ---------------------------

# def fetch_ibkr_short_availability_map(
#     symbols: List[str],
#     ftp_host: str = "ftp2.interactivebrokers.com",
#     ftp_user: str = "shortstock",
#     ftp_pass: str = "",
#     ftp_file: str = "usa.txt",
# ) -> Dict[str, Dict[str, Optional[float]]]:
#     want = {s.upper().strip() for s in symbols if str(s).strip()}
#     if not want:
#         return {}

#     ftp = ftplib.FTP(ftp_host)
#     ftp.login(user=ftp_user, passwd=ftp_pass)

#     buf = io.BytesIO()
#     ftp.retrbinary(f"RETR {ftp_file}", buf.write)
#     ftp.quit()

#     buf.seek(0)
#     text = buf.getvalue().decode("utf-8", errors="ignore")
#     lines = [ln for ln in text.splitlines() if ln.strip()]

#     header_idx = None
#     for i, ln in enumerate(lines):
#         if ln.startswith("#SYM|"):
#             header_idx = i
#             break
#     if header_idx is None:
#         print("[SHORT] WARNING: no '#SYM|' header; skipping short availability precheck.")
#         return {}

#     header_cols = [c.strip().lstrip("#").lower() for c in lines[header_idx].split("|")]
#     data_lines = lines[header_idx + 1 :]

#     df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="|", header=None, engine="python")
#     n_cols = min(len(header_cols), df.shape[1])
#     df = df.iloc[:, :n_cols]
#     df.columns = header_cols[:n_cols]

#     if "sym" not in df.columns:
#         print("[SHORT] WARNING: missing 'sym' col; skipping short availability precheck.")
#         return {}

#     df["sym"] = df["sym"].astype(str).str.upper().str.strip()

#     if "available" in df.columns:
#         df["available_int"] = pd.to_numeric(df["available"], errors="coerce")
#     else:
#         df["available_int"] = pd.NA

#     fee = pd.to_numeric(df.get("feerate", pd.Series([pd.NA] * len(df))), errors="coerce") / 100.0
#     rebate = pd.to_numeric(df.get("rebaterate", pd.Series([pd.NA] * len(df))), errors="coerce") / 100.0
#     df["net_borrow_annual"] = (fee - rebate).clip(lower=0)

#     sub = df[df["sym"].isin(want)].copy()

#     out: Dict[str, Dict[str, Optional[float]]] = {}
#     for _, r in sub.iterrows():
#         sym = str(r["sym"])
#         avail = r.get("available_int", pd.NA)
#         borrow = r.get("net_borrow_annual", pd.NA)
#         out[sym] = {
#             "available": None if pd.isna(avail) else int(avail),
#             "borrow": None if pd.isna(borrow) else float(borrow),
#         }
#     return out


# # ---------------------------
# # Symbol normalization
# # ---------------------------

# IB_SYMBOL_MAP: Dict[str, Tuple[str, Optional[str]]] = {
#     "BRK-B": ("BRK B", "NYSE"),
#     "BRK-A": ("BRK A", "NYSE"),
# }
# REVERSE_IB_SYMBOL_MAP: Dict[str, str] = {ib_sym: uni for uni, (ib_sym, _) in IB_SYMBOL_MAP.items()}


# def ib_symbol_from_universal(sym: str) -> Tuple[str, Optional[str]]:
#     s = str(sym).strip().upper()
#     if s in IB_SYMBOL_MAP:
#         return IB_SYMBOL_MAP[s]
#     return s, None


# def universal_symbol_from_ib(sym: str) -> str:
#     s = str(sym).strip().upper()
#     return REVERSE_IB_SYMBOL_MAP.get(s, s)


# def make_stock(symbol: str) -> Stock:
#     ib_sym, primary = ib_symbol_from_universal(symbol)
#     c = Stock(ib_sym, "SMART", "USD")
#     if primary:
#         c.primaryExchange = primary
#     return c


# # ---------------------------
# # Paths / date helpers
# # ---------------------------

# def today_str() -> str:
#     return date.today().isoformat()


# def run_dir(run_date: str) -> Path:
#     return Path("data") / "runs" / run_date


# def exec_dir(run_date: str) -> Path:
#     return run_dir(run_date) / "execution"


# # ---------------------------
# # IBKR connection & pricing
# # ---------------------------

# def connect_ib(host: str, port: int, client_id: int) -> IB:
#     ib = IB()
#     ib.connect(host, port, clientId=client_id)
#     if not ib.isConnected():
#         raise RuntimeError("Failed to connect to IBKR.")
#     if client_id == 0:
#         ib.reqAutoOpenOrders(True)
#     try:
#         ib.reqOpenOrders()
#         ib.sleep(0.5)
#     except Exception:
#         pass
#     return ib


# def safe_price(v) -> Optional[float]:
#     try:
#         x = float(v)
#         return x if x > 0 else None
#     except Exception:
#         return None


# def get_snapshot_price(ib: IB, symbol: str, prefer_delayed: bool = True) -> float:
#     sym_u = symbol.upper()
#     contract = make_stock(sym_u)
#     ib.qualifyContracts(contract)

#     def read_ticker(t) -> Optional[float]:
#         bid = safe_price(getattr(t, "bid", None)) or safe_price(getattr(t, "delayedBid", None))
#         ask = safe_price(getattr(t, "ask", None)) or safe_price(getattr(t, "delayedAsk", None))
#         last = safe_price(getattr(t, "last", None)) or safe_price(getattr(t, "delayedLast", None))
#         close = safe_price(getattr(t, "close", None)) or safe_price(getattr(t, "delayedClose", None))
#         mkt = safe_price(t.marketPrice())
#         if bid and ask:
#             return (bid + ask) / 2.0
#         return last or close or mkt

#     def snapshot_with_type(data_type: int) -> Optional[float]:
#         ib.reqMarketDataType(data_type)
#         t = ib.reqMktData(contract, "", snapshot=True)
#         try:
#             for _ in range(12):
#                 ib.sleep(0.25)
#                 px = read_ticker(t)
#                 if px is not None:
#                     return px
#         finally:
#             try:
#                 req_id = getattr(t, "reqId", None) or getattr(t, "tickerId", None)
#                 if isinstance(req_id, int) and req_id > 0:
#                     ib.client.cancelMktData(req_id)
#             except Exception:
#                 pass
#         return None

#     px: Optional[float] = None
#     if prefer_delayed:
#         px = snapshot_with_type(4) or snapshot_with_type(3)
#     else:
#         px = snapshot_with_type(1) or snapshot_with_type(3)

#     if px is None:
#         for _attempt in range(3):
#             if not ib.isConnected():
#                 for _ in range(40):
#                     ib.sleep(0.25)
#                     if ib.isConnected():
#                         break
#             try:
#                 bars = ib.reqHistoricalData(
#                     contract,
#                     endDateTime="",
#                     durationStr="2 D",
#                     barSizeSetting="1 day",
#                     whatToShow="TRADES",
#                     useRTH=True,
#                     formatDate=1,
#                     keepUpToDate=False,
#                 )
#             except Exception:
#                 bars = []
#             if bars:
#                 px = safe_price(bars[-1].close)
#                 if px is not None:
#                     break
#             ib.sleep(0.5)

#     if px is None:
#         raise RuntimeError(f"No usable price for {sym_u}")
#     return float(px)


# # ---------------------------
# # Orders & execution
# # ---------------------------

# def build_market_order(action: str, qty: int, order_ref: str) -> Order:
#     o = MarketOrder(action.upper(), int(qty))
#     o.tif = "DAY"
#     o.transmit = True
#     o.orderRef = order_ref
#     return o


# def build_adaptive_market_order(action: str, qty: int, order_ref: str, priority: str = "Normal") -> Order:
#     o = Order()
#     o.action = action.upper()
#     o.totalQuantity = int(qty)
#     o.orderType = "MKT"
#     o.tif = "DAY"
#     o.transmit = True
#     o.orderRef = order_ref
#     o.algoStrategy = "Adaptive"
#     o.algoParams = [TagValue("adaptivePriority", str(priority))]
#     return o


# TERMINAL: Set[str] = {"filled", "cancelled", "inactive"}
# ACCEPTED: Set[str] = {"presubmitted", "submitted", "filled", "pendingsubmit"}


# def wait_for_trade_terminal(ib: IB, trade: Trade, timeout: float = 180.0) -> Trade:
#     t0 = time.time()
#     while time.time() - t0 < timeout:
#         st = (trade.orderStatus.status or "").lower()
#         if st in TERMINAL:
#             return trade
#         try:
#             ib.reqAllOpenOrders()
#         except Exception:
#             pass
#         ib.sleep(0.2)
#     return trade


# def wait_for_trade_accepted(ib: IB, trade: Trade, timeout: float = 45.0) -> Tuple[bool, Trade]:
#     t0 = time.time()
#     while time.time() - t0 < timeout:
#         st = (trade.orderStatus.status or "").lower()
#         if st in ACCEPTED:
#             return True, trade
#         if st in TERMINAL:
#             return False, trade
#         try:
#             ib.reqAllOpenOrders()
#         except Exception:
#             pass
#         ib.sleep(0.2)
#     return False, trade


# def execute_leg(
#     *,
#     ib: IB,
#     symbol: str,
#     action: str,
#     qty: int,
#     ref_price: float,
#     bps: float,
#     order_ref: str,
#     exec_cfg: Dict,
#     timeout: float = 90.0,
#     max_retries: int = 3,
#     dry_run: bool = False,
#     context: str = ""
# ) -> Tuple[int, Optional[Trade]]:
#     if qty <= 0:
#         return 0, None

#     contract = make_stock(symbol)
#     ib.qualifyContracts(contract)

#     filled_total = 0
#     last_trade: Optional[Trade] = None

#     order_style = str(exec_cfg.get("order_style", "ADAPTIVE_MKT")).strip().upper()
#     market_done_timeout = float(exec_cfg.get("market_done_timeout_sec", 180.0))
#     def last_ib_error(trade: Optional[Trade]) -> Tuple[Optional[int], Optional[str]]:
#         if trade is None:
#             return None, None
#         try:
#             logs = getattr(trade, "log", None) or []
#             # scan from end for an errorCode
#             for entry in reversed(logs):
#                 code = getattr(entry, "errorCode", None)
#                 msg = getattr(entry, "message", None)
#                 if code and int(code) != 0:
#                     return int(code), str(msg or "")
#         except Exception:
#             pass
#         return None, None

#     for attempt in range(1, max_retries + 1):
#         remain = qty - filled_total
#         if remain <= 0:
#             break

#         if order_style == "ADAPTIVE_MKT":
#             # Underlying gets urgent
#             if "|UNDER_DELTA" in order_ref:
#                 priority = "Urgent"
#             else:
#                 priority = "Patient" if attempt == 1 else ("Normal" if attempt == 2 else "Urgent")

#             o = build_adaptive_market_order(
#                 action=action,
#                 qty=remain,
#                 order_ref=f"{order_ref}|att{attempt}|ADAPTIVE_MKT",
#                 priority=priority,
#             )
#             px_str = f"ADAPTIVE_MKT({priority})"
#         else:
#             o = build_market_order(
#                 action=action,
#                 qty=remain,
#                 order_ref=f"{order_ref}|att{attempt}|MKT",
#             )
#             px_str = "MKT"

#         ctx = f"[{context}]" if context else ""
#         tprint(
#             f"{ctx}[LEG] {symbol} {action} qty={remain} px={px_str} "
#             f"refTag={o.orderRef} clientId={ib.client.clientId}"
#         )
#         if dry_run:
#             filled_total += remain
#             continue

#         trade = ib.placeOrder(contract, o)
#         last_trade = trade

#         try:
#             ib.reqAllOpenOrders()
#         except Exception:
#             pass
#         ib.sleep(0.3)
        
#         accepted, trade = wait_for_trade_accepted(
#             ib, trade, timeout=float(exec_cfg.get("market_accept_timeout_sec", 45.0))
#         )

#         if not accepted:
#             st = (trade.orderStatus.status or "")
#             code, msg = last_ib_error(trade)

#             # ---- HARD FAIL: cannot short this contract ----
#             if code == 201 and "not available for short sale" in (msg or "").lower():
#                 tprint(
#                     f"[LEG] {symbol} HARD_REJECT code=201 not shortable; "
#                     f"no more retries. status={st} orderId={trade.order.orderId} ref={trade.order.orderRef}"
#                 )
#                 break  # <- stop retry loop for this leg

#             tprint(f"[LEG] {symbol} not accepted (status={st} code={code} msg={msg}); retrying.")
#             continue


#         trade = wait_for_trade_terminal(ib, trade, timeout=market_done_timeout)
#         status = (trade.orderStatus.status or "").lower()
#         filled = int(trade.orderStatus.filled or 0)
#         remaining = trade.orderStatus.remaining
#         remaining = None if remaining is None else int(remaining)

#         order_id = getattr(trade.order, "orderId", None)
#         perm_id = getattr(trade.orderStatus, "permId", None) or getattr(trade.order, "permId", None)
#         tprint(
#             f"{ctx}[LEG] status={status} filled={filled} remaining={remaining} "
#             f"orderId={order_id} permId={perm_id} refTag={trade.order.orderRef}"
#         )
#         filled_total = min(qty, filled_total + max(0, filled))

#         if status not in TERMINAL and remaining and remaining > 0:
#             try:
#                 ib.cancelOrder(trade.order)
#             except Exception:
#                 pass

#     return int(filled_total), last_trade


# # ---------------------------
# # Baseline snapshot mechanics
# # ---------------------------

# def load_baseline_qty(path: Path) -> Dict[str, float]:
#     if not path.exists():
#         print(f"[BASELINE] No baseline file found at {path}. Treating baseline as empty.")
#         return {}
#     df = pd.read_csv(path)
#     if df.empty:
#         return {}
#     df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
#     if "qty" not in df.columns:
#         raise ValueError(f"Baseline file {path} missing required column 'qty'. Columns={list(df.columns)}")
#     return dict(df.groupby("symbol")["qty"].sum())


# def current_ib_positions(ib: IB) -> Dict[str, float]:
#     out: Dict[str, float] = {}
#     for p in ib.positions():
#         sym = universal_symbol_from_ib(p.contract.symbol)
#         out[sym] = out.get(sym, 0.0) + float(p.position)
#     return out


# def strategy_position_only(ib_pos: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
#     syms = set(ib_pos) | set(baseline)
#     return {s: float(ib_pos.get(s, 0.0) - baseline.get(s, 0.0)) for s in syms}


# # ---------------------------
# # IO helpers
# # ---------------------------

# def append_fills(rows: List[dict], fills_path: Path) -> None:
#     fills_path.parent.mkdir(parents=True, exist_ok=True)
#     df_new = pd.DataFrame(rows)
#     if fills_path.exists():
#         df_old = pd.read_csv(fills_path)
#         df_out = pd.concat([df_old, df_new], ignore_index=True)
#     else:
#         df_out = df_new
#     df_out.to_csv(fills_path, index=False)
#     print(f"[FILLS] Appended {len(rows)} rows -> {fills_path}")


# def resolve_plan_path(run_date: str, paths_cfg: dict) -> Path:
#     dated = run_dir(run_date) / "proposed_trades.csv"
#     if dated.exists():
#         return dated
#     return Path(paths_cfg.get("proposed_trades_csv", "data/proposed_trades.csv"))


# def resolve_fills_path(run_date: str, paths_cfg: dict) -> Path:
#     return exec_dir(run_date) / "fills.csv"


# def write_execution_snapshot(run_date: str, df: pd.DataFrame, name: str) -> None:
#     p = exec_dir(run_date) / name
#     p.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(p, index=False)
#     print(f"[EXEC] Wrote {p}")


# # ---------------------------
# # Sizing helpers
# # ---------------------------

# def target_shares_from_usd(notional_usd: float, px: float) -> int:
#     if px <= 0:
#         raise ValueError("Price must be > 0")
#     return int(notional_usd / px)


# # ---------------------------
# # Pretty breakdown (your new prints)
# # ---------------------------

# def fmt_dollars(x: float) -> str:
#     s = f"${x:,.0f}"
#     return s

# def handle_sigint(signum, frame):
#     if not stop_requested():
#         print("\n[CTRL+C] Shutdown requested. Stopping new work…", flush=True)
#         SHUTDOWN.set()
#     else:
#         print("\n[CTRL+C] Forced exit.", flush=True)
#         sys.exit(1)


# # ---------------------------
# # Main
# # ---------------------------

# def main() -> None:

#     CONFIG_YML = Path("config/strategy_config.yml")

#     if not CONFIG_YML.exists():
#         raise FileNotFoundError(f"Config not found: {CONFIG_YML}")

#     cfg = yaml.safe_load(CONFIG_YML.read_text(encoding="utf-8")) or {}

#     ibkr_cfg = cfg.get("ibkr", {}) or {}
#     strat_cfg = cfg.get("strategy", {}) or {}
#     paths_cfg = cfg.get("paths", {}) or {}
#     exec_cfg = cfg.get("execution", {}) or {}

#     parallel_n = max(1, int(exec_cfg.get("parallel_n", 1)))
#     auto_approve_cli = bool(exec_cfg.get("auto_approve", False))
#     run_date = today_str()
#     # --- Strategy tag ---
#     strategy_tag = str(strat_cfg.get("tag", "")).strip()
#     if not strategy_tag:
#         raise ValueError("Missing strategy.tag in config/strategy_config.yml (or pass --strategy-tag).")

#     # --- IBKR connection params ---
#     host = str(ibkr_cfg.get("host", "127.0.0.1"))
#     port = int(ibkr_cfg.get("port", 7497))
#     client_id = int(ibkr_cfg.get("client_id", 3))
#     prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))

#     exec_cfg = dict(exec_cfg)
#     exec_cfg["prefer_delayed"] = prefer_delayed

#     # --- Execution params ---
#     limit_bps = float(exec_cfg.get("limit_bps", 10.0))
#     timeout = float(exec_cfg.get("timeout_sec", 90))
#     short_first = bool(exec_cfg.get("short_first", True))
#     max_retries = int(exec_cfg.get("max_retries", 3))

#     if "DRY_RUN" in os.environ:
#         dry_run = bool(int(os.getenv("DRY_RUN", "0")))
#     else:
#         dry_run = bool(exec_cfg.get("dry_run", False))

#     if dry_run:
#         print("[DRY_RUN] Enabled. No orders will be placed.")

#     # --- Paths ---
#     plan_path = resolve_plan_path(run_date, paths_cfg)
#     baseline_csv = Path(paths_cfg.get("baseline_csv", "data/baseline_snapshot.csv"))
#     fills_path = resolve_fills_path(run_date, paths_cfg)

#     if not plan_path.exists():
#         raise FileNotFoundError(f"Trade plan not found: {plan_path}")

#     plan = pd.read_csv(plan_path)
#     if "strategy_tag" in plan.columns:
#         plan = plan[plan["strategy_tag"].astype(str) == strategy_tag].copy()

#     if plan.empty:
#         raise ValueError(f"No rows in {plan_path} for strategy_tag={strategy_tag}")

#     # Normalize + stable sort
#     plan = plan.reset_index(drop=True)
#     plan["_orig_idx"] = plan.index
#     plan["Underlying"] = plan["Underlying"].astype(str).str.upper().str.strip()
#     plan["ETF"] = plan["ETF"].astype(str).str.upper().str.strip()

#     plan = plan.sort_values(by=["Underlying", "_orig_idx"], kind="mergesort").reset_index(drop=True)
#     plan = plan.drop(columns=["_orig_idx"]).reset_index(drop=True)

#     # baseline
#     baseline = load_baseline_qty(baseline_csv)
#     exec_dir(run_date).mkdir(parents=True, exist_ok=True)
#     exposure_csv = exec_dir(run_date) / "exposure_log.csv"
#     exposure_jsonl = exec_dir(run_date) / "exposure_log.jsonl"

#     log_lock = threading.Lock()

#     # Connect a “coordinator” IB (only for initial snapshots + price prefetch)
#     ib = connect_ib(host, port, client_id)
#     try:
#         ib_pos = current_ib_positions(ib)
#         strat_pos = strategy_position_only(ib_pos, baseline)

#         print(
#             f"[POS] current IB symbols={len(ib_pos)}; "
#             f"baseline symbols={len(baseline)}; "
#             f"strategy-only symbols={len(strat_pos)}"
#         )
#         print(f"[PLAN] Using: {plan_path}")
#         print(f"[BASELINE] Using: {baseline_csv}")
#         print(f"[EXEC] Writing to: {exec_dir(run_date)}")

#         fills_to_append: List[dict] = []

#         # ---------------------------
#         # PLAN-derived maps (execution truth)
#         # ---------------------------
#         if "Leverage" not in plan.columns:
#             raise ValueError("Plan missing required column: Leverage")

#         plan["ETF_U"] = plan["ETF"].astype(str).str.upper().str.strip()
#         plan["UNDER_U"] = plan["Underlying"].astype(str).str.upper().str.strip()

#         leverage_by_etf_plan: Dict[str, float] = {}
#         under_to_etfs_planned: Dict[str, Set[str]] = {}

#         for _, r in plan.iterrows():
#             e = str(r["ETF_U"])
#             u = str(r["UNDER_U"])
#             lev = float(r["Leverage"])
#             under_to_etfs_planned.setdefault(u, set()).add(e)
#             if e in leverage_by_etf_plan and abs(leverage_by_etf_plan[e] - lev) > 1e-9:
#                 raise ValueError(f"Leverage mismatch for {e} in plan: {leverage_by_etf_plan[e]} vs {lev}")
#             leverage_by_etf_plan[e] = lev

#         # ---------------------------
#         # SCREENED universe maps (hedge truth)
#         # ---------------------------
#         screened_csv = Path(paths_cfg.get("screened_csv", "data/etf_screened_today.csv"))
#         if not screened_csv.exists():
#             raise FileNotFoundError(f"Screened universe not found: {screened_csv}")

#         screened = pd.read_csv(screened_csv)
#         need_cols = {"Underlying", "ETF", "Leverage"}
#         if not need_cols.issubset(set(screened.columns)):
#             raise ValueError(f"{screened_csv} missing required columns {need_cols}. Columns={list(screened.columns)}")

#         screened["Underlying"] = screened["Underlying"].astype(str).str.upper().str.strip()
#         screened["ETF"] = screened["ETF"].astype(str).str.upper().str.strip()

#         under_to_etfs_all: Dict[str, Set[str]] = {}
#         leverage_by_etf_all: Dict[str, float] = {}
#         for _, r in screened.iterrows():
#             u = str(r["Underlying"])
#             e = str(r["ETF"])
#             lev = float(r["Leverage"])
#             if not u or u == "NAN" or not e or e == "NAN":
#                 continue
#             under_to_etfs_all.setdefault(u, set()).add(e)
#             leverage_by_etf_all[e] = lev

#         # ETF->Underlying map (screened)
#         etf_to_under_all: Dict[str, str] = {}
#         for u, etfs in under_to_etfs_all.items():
#             for e in etfs:
#                 prev = etf_to_under_all.get(e)
#                 if prev is not None and prev != u:
#                     raise ValueError(f"[SCREENED] ETF {e} maps to multiple underlyings: {prev} and {u}")
#                 etf_to_under_all[e] = u

#         # ---------------------------
#         # Plan sanity: ETF symbol appears in exactly one bucket (parallel safety)
#         # ---------------------------
#         etf_seen: Dict[str, str] = {}
#         for _, r in plan.iterrows():
#             e = str(r["ETF_U"])
#             u = str(r["UNDER_U"])
#             prev = etf_seen.get(e)
#             if prev is not None and prev != u:
#                 raise ValueError(
#                     f"Parallel unsafe: planned ETF {e} appears under multiple underlyings {prev} and {u}."
#                 )
#             etf_seen[e] = u

#         # ---------------------------
#         # Short availability snapshot (PLAN ETFs only)
#         # ---------------------------
#         etf_symbols_plan = sorted(set(plan["ETF_U"].astype(str).str.upper()))
#         short_map: Dict[str, Dict[str, Optional[float]]] = {}
#         try:
#             short_map = fetch_ibkr_short_availability_map(etf_symbols_plan)
#             print(f"[SHORT] Loaded availability for {len(short_map)}/{len(etf_symbols_plan)} plan ETFs from IBKR FTP.")
#         except Exception as ex:
#             print(f"[SHORT] WARNING: short availability precheck failed ({ex}); continuing without it.")
#             short_map = {}

#         # ---------------------------
#         # Prefetch prices for plan symbols + held mapped ETFs (screened)
#         # ---------------------------
#         # Identify HELD screened ETFs by reading current strategy-only positions + screened mapping.
#         held_etfs_by_under: Dict[str, Set[str]] = {}
#         for sym, sh0 in strat_pos.items():
#             sh = int(round(float(sh0)))
#             if sh == 0:
#                 continue
#             u = etf_to_under_all.get(sym)
#             if u is None:
#                 continue
#             held_etfs_by_under.setdefault(u, set()).add(sym)

#         plan_symbols = set(plan["UNDER_U"].tolist()) | set(plan["ETF_U"].tolist())
#         held_mapped_etfs = set().union(*held_etfs_by_under.values()) if held_etfs_by_under else set()
#         symbols = sorted(plan_symbols | held_mapped_etfs)

#         prices: Dict[str, float] = {}
#         for s in symbols:
#             prices[s] = get_snapshot_price(ib, s, prefer_delayed=prefer_delayed)

#         px_df = pd.DataFrame([{"symbol": k, "price": v} for k, v in prices.items()]).sort_values("symbol")
#         write_execution_snapshot(run_date, px_df, "prices_snapshot.csv")

#         # ---------------------------
#         # Skip underlyings whose TOTAL target shares would round/truncate to 0.
#         # (execution filter only; based on plan)
#         # ---------------------------
#         skip_underlyings = set()
#         for u_sym, grp in plan.groupby(plan["UNDER_U"].astype(str).str.upper()):
#             px_u = float(prices[u_sym])
#             total_u_sh = 0
#             for _, rr in grp.iterrows():
#                 tu = float(rr["long_usd"])
#                 total_u_sh += target_shares_from_usd(tu, px_u)
#             if total_u_sh == 0:
#                 skip_underlyings.add(u_sym)

#         if skip_underlyings:
#             print(f"[SKIP] Underlyings with rounded-to-0 total target shares: {sorted(skip_underlyings)}")
#             df_skip = plan[plan["UNDER_U"].astype(str).str.upper().isin(skip_underlyings)].copy()
#             write_execution_snapshot(run_date, df_skip, "skipped_underlyings_rounded0.csv")
#             plan = plan[~plan["UNDER_U"].astype(str).str.upper().isin(skip_underlyings)].copy()
#             if plan.empty:
#                 print("[SKIP] All rows skipped due to rounded-to-0 underlying sizing. Nothing to execute.")
#                 return

#         # ---------------------------
#         # Logging helper (thread-safe)
#         # ---------------------------
#         def log_exposure_event(
#             *,
#             stage: str,
#             pair_id: str,
#             underlying: str,
#             etf: str,
#             symbol: str,
#             delta_sh: int,
#             filled_sh: int,
#             trade: Optional[Trade],
#         ):
#             ib_pos_now = current_ib_positions(ib)  # coordinator snapshot (OK for portfolio-level)
#             strat_pos_now = strategy_position_only(ib_pos_now, baseline)

#             port = compute_portfolio_notionals(
#                 {k: int(round(float(v))) for k, v in strat_pos_now.items()},
#                 prices,
#             )

#             mark_px = float(prices.get(symbol)) if prices.get(symbol) is not None else None
#             fill_px = safe_avg_fill_price(trade)
#             used_px = fill_px if fill_px is not None else mark_px

#             pos_sh = int(round(float(strat_pos_now.get(symbol, 0.0)))) if symbol != "PORTFOLIO" else 0
#             pos_notional = (pos_sh * mark_px) if (mark_px is not None and symbol != "PORTFOLIO") else None
#             delta_notional = (int(filled_sh) * float(used_px)) if (used_px is not None) else None

#             row = {
#                 "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "run_date": run_date,
#                 "strategy_tag": strategy_tag,
#                 "stage": stage,
#                 "pair_id": pair_id,
#                 "underlying": underlying,
#                 "etf": etf,
#                 "symbol": symbol,
#                 "delta_sh": int(delta_sh),
#                 "filled_sh": int(filled_sh),
#                 "fill_avg_px": fill_px,
#                 "mark_px": mark_px,
#                 "delta_notional": delta_notional,
#                 "pos_sh": pos_sh,
#                 "pos_notional": pos_notional,
#                 **port,
#             }

#             with log_lock:
#                 append_csv_row(exposure_csv, row)
#                 append_jsonl(exposure_jsonl, row)

#         # ---------------------------
#         # Shared helpers for workers
#         # ---------------------------
#         def get_lev_or_raise(etf: str) -> float:
#             lev = leverage_by_etf_all.get(etf)
#             if lev is None:
#                 raise ValueError(f"Missing leverage for ETF {etf} in screened universe; cannot hedge accurately.")
#             return float(lev)

#         def ensure_price_worker(ib_local: IB, sym: str) -> float:
#             px = prices.get(sym)
#             if px is not None:
#                 return float(px)
#             px2 = get_snapshot_price(ib_local, sym, prefer_delayed=prefer_delayed)
#             with log_lock:
#                 prices[sym] = float(px2)
#             return float(px2)

#         def print_bucket_exposure_breakdown(
#             ib_local: IB,
#             baseline_local: Dict[str, float],
#             etf_to_under_local: Dict[str, str],
#             u_sym: str,
#             title: str,
#         ) -> None:
#             ib_pos_now = current_ib_positions(ib_local)
#             strat_now_raw = strategy_position_only(ib_pos_now, baseline_local)
#             strat_now = {k: int(round(float(v))) for k, v in strat_now_raw.items()}

#             px_u = ensure_price_worker(ib_local, u_sym)
#             u_sh = int(strat_now.get(u_sym, 0))
#             E = u_sh * px_u

#             # collect held screened ETFs mapped to this underlying
#             held = []
#             for sym, sh in strat_now.items():
#                 if sh == 0:
#                     continue
#                 u = etf_to_under_local.get(sym)
#                 if u == u_sym:
#                     held.append(sym)

#             print(f"\n[{title}] Exposure breakdown for {u_sym}")
#             print(f"  Underlying {u_sym}: sh={u_sh:+d} px={px_u:.4f} MV={fmt_dollars(u_sh*px_u)}")

#             if held:
#                 print("  Held screened ETFs mapped to this underlying:")
#                 for etf in sorted(held):
#                     sh = int(strat_now.get(etf, 0))
#                     px_e = ensure_price_worker(ib_local, etf)
#                     mv = sh * px_e
#                     lev = get_lev_or_raise(etf)
#                     contrib = mv * lev
#                     E += contrib
#                     print(f"    {etf}: sh={sh:+d} px={px_e:.4f} MV={fmt_dollars(mv)} lev={lev:+.2f} contrib={fmt_dollars(contrib)}")
#             else:
#                 print("  Held screened ETFs mapped to this underlying: (none)")

#             resid_dollars = E
#             resid_sh_eq = resid_dollars / px_u if px_u else 0.0
#             print(f"  Residual: {fmt_dollars(resid_dollars)}  ({resid_sh_eq:+.2f} {u_sym} shares eq)\n")

#         # ---------------------------
#         # Build bucket list (plan order)
#         # ---------------------------
#         underlying_order: List[str] = []
#         seen_u: Set[str] = set()
#         for u_sym in plan["UNDER_U"].astype(str).str.upper().tolist():
#             if u_sym not in seen_u:
#                 underlying_order.append(u_sym)
#                 seen_u.add(u_sym)

#         buckets: List[Tuple[str, pd.DataFrame]] = []
#         for u_sym in underlying_order:
#             grp = plan[plan["UNDER_U"].astype(str).str.upper() == u_sym].copy()
#             if grp.empty:
#                 continue
#             buckets.append((u_sym, grp))

#         # ---------------------------
#         # Approvals (serial) -> Approved buckets
#         # ---------------------------
#         approved: List[Tuple[str, pd.DataFrame]] = []
#         if parallel_n > 1 and not auto_approve_cli:
#             # Serial prompt for safety; then parallel execute approved.
#             for (u_sym, grp) in buckets:
#                 ans = input(f"Approve bucket for {u_sym}? (y/n/q): ").strip().lower()
#                 if ans == "q":
#                     break
#                 if ans == "y":
#                     approved.append((u_sym, grp))
#         else:
#             approved = buckets

#         if not approved:
#             print("[DONE] No approved buckets to execute.")
#             return

#         # ---------------------------
#         # Worker: execute a single underlying bucket
#         # ---------------------------
#         def execute_underlying_bucket(u_sym: str, grp: pd.DataFrame, worker_idx: int) -> List[dict]:
#             ensure_thread_event_loop()   # <-- MUST be before any ib_insync calls
#             ib_local = connect_ib(host, port, client_id + 100 + worker_idx)
#             local_fills: List[dict] = []
#             try:
#                 # Local snapshot for hedge truth (includes held ETFs not in plan)
#                 ib_pos_local = current_ib_positions(ib_local)
#                 strat_pos_local = strategy_position_only(ib_pos_local, baseline)

#                 # Print the PLAN summary (like your current)
#                 px_u = ensure_price_worker(ib_local, u_sym)

#                 # Compute plan targets for this bucket
#                 bucket_target_etf_sh: Dict[str, int] = {}
#                 bucket_target_under_sh: int = 0

#                 for _, row in grp.iterrows():
#                     e_sym = str(row["ETF_U"]).upper()
#                     tu = float(row["long_usd"])
#                     te = float(row["short_usd"])

#                     px_e = ensure_price_worker(ib_local, e_sym)

#                     bucket_target_under_sh += target_shares_from_usd(tu, px_u)
#                     bucket_target_etf_sh[e_sym] = int(bucket_target_etf_sh.get(e_sym, 0) + target_shares_from_usd(te, px_e))

#                 cur_u_sh = int(round(float(strat_pos_local.get(u_sym, 0.0))))
#                 target_u_sh = int(bucket_target_under_sh)

#                 bucket_delta_etf: Dict[str, int] = {}
#                 for e_sym, tgt_sh in bucket_target_etf_sh.items():
#                     cur_e_sh = int(round(float(strat_pos_local.get(e_sym, 0.0))))
#                     bucket_delta_etf[e_sym] = int(tgt_sh - cur_e_sh)

#                 delta_under_naive = int(target_u_sh - cur_u_sh)

#                 print("\n" + "=" * 110)
#                 print(f"[GROUP] Underlying bucket: {u_sym}")
#                 print(f"  Mark px: {u_sym}={px_u:.4f}")
#                 print(f"  Bucket pairs: {len(grp)}")
#                 print(f"  Planned target underlying shares (sum of rows): {target_u_sh:+d}")
#                 print(f"  Current strategy-only underlying shares: {cur_u_sh:+d}")
#                 print(f"  Naive underlying delta to planned target: {delta_under_naive:+d}")

#                 for e_sym in sorted(bucket_target_etf_sh.keys()):
#                     px_e = ensure_price_worker(ib_local, e_sym)
#                     cur_e_sh = int(round(float(strat_pos_local.get(e_sym, 0.0))))
#                     tgt_e_sh = int(bucket_target_etf_sh[e_sym])
#                     d_e = int(bucket_delta_etf[e_sym])
#                     lev = leverage_by_etf_plan.get(e_sym)
#                     if lev is None:
#                         raise ValueError(f"Missing leverage for planned ETF {e_sym} in plan columns.")
#                     print(
#                         f"  ETF {e_sym}: px={px_e:.4f} lev={float(lev):+,.2f} cur={cur_e_sh:+d} tgt={tgt_e_sh:+d} delta={d_e:+d}"
#                     )

#                 # Hedge truth print BEFORE any ETF trades
#                 print_bucket_exposure_breakdown(
#                     ib_local,
#                     baseline,
#                     etf_to_under_all,
#                     u_sym,
#                     title="PRE_GROUP_HEDGE_TRUTH",
#                 )

#                 log_exposure_event(
#                     stage="PRE_GROUP",
#                     pair_id=f"{u_sym}__GROUP",
#                     underlying=u_sym,
#                     etf="",
#                     symbol="PORTFOLIO",
#                     delta_sh=0,
#                     filled_sh=0,
#                     trade=None,
#                 )

#                 # Execute ETF legs (PLAN only)
#                 def exec_delta_local(symbol: str, delta: int, px: float, order_ref: str) -> Tuple[int, Optional[Trade]]:
#                     if delta == 0:
#                         return 0, None
#                     action = "BUY" if delta > 0 else "SELL"
#                     qty = abs(delta)
#                     return execute_leg(
#                         ib=ib_local,
#                         symbol=symbol,
#                         action=action,
#                         qty=qty,
#                         ref_price=px,
#                         bps=limit_bps,
#                         order_ref=order_ref,
#                         exec_cfg=exec_cfg,
#                         timeout=timeout,
#                         max_retries=max_retries,
#                         dry_run=dry_run,
#                         context=u_sym
#                     )

#                 etf_items = list(bucket_delta_etf.items())

#                 def _etf_sort_key(item):
#                     sym, d = item
#                     return (0 if d < 0 else 1, sym)

#                 if short_first:
#                     etf_items.sort(key=_etf_sort_key)
#                 else:
#                     etf_items.sort(key=lambda x: x[0])

#                 for e_sym, d_e in etf_items:
#                     if d_e == 0:
#                         continue

#                     sm = short_map.get(e_sym)
#                     if sm and sm.get("available") is not None and d_e < 0 and abs(d_e) > int(sm["available"]):
#                         print(
#                             f"[SHORT] WARNING: {e_sym} wants {abs(d_e)} shares short, "
#                             f"but IBKR file shows only {sm['available']} available."
#                         )

#                     px_e = ensure_price_worker(ib_local, e_sym)
#                     order_ref = f"{strategy_tag}|{u_sym}__GROUP|{e_sym}|ETF_DELTA"

#                     filled_e_abs, trade_e = exec_delta_local(e_sym, d_e, px_e, order_ref)
#                     filled_e = -filled_e_abs if d_e < 0 else filled_e_abs

#                     log_exposure_event(
#                         stage="POST_ETF",
#                         pair_id=f"{u_sym}__GROUP",
#                         underlying=u_sym,
#                         etf=e_sym,
#                         symbol=e_sym,
#                         delta_sh=d_e,
#                         filled_sh=filled_e,
#                         trade=trade_e,
#                     )

#                     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     local_fills.append(
#                         {
#                             "filled_at": now,
#                             "run_date": run_date,
#                             "strategy_tag": strategy_tag,
#                             "pair_id": f"{u_sym}__GROUP",
#                             "underlying": u_sym,
#                             "etf": e_sym,
#                             "px_under": px_u,
#                             "px_etf": px_e,
#                             "target_sh_under": target_u_sh,
#                             "target_sh_etf": int(bucket_target_etf_sh.get(e_sym, 0)),
#                             "delta_sh_under": 0,
#                             "delta_sh_etf": d_e,
#                             "filled_sh_under": 0,
#                             "filled_sh_etf": filled_e,
#                             "notes": "GROUP_ETF",
#                         }
#                     )

#                 # Hedge truth AFTER ETF trades but BEFORE underlying hedge
#                 print_bucket_exposure_breakdown(
#                     ib_local,
#                     baseline,
#                     etf_to_under_all,
#                     u_sym,
#                     title="POST_ETFS_HEDGE_TRUTH",
#                 )

#                 # Compute hedge residual using ALL HELD screened ETFs mapped to this underlying
#                 def compute_bucket_resid_sh_local(u: str) -> float:
#                     ib_pos_now = current_ib_positions(ib_local)
#                     strat_now_raw = strategy_position_only(ib_pos_now, baseline)
#                     strat_now = {k: int(round(float(v))) for k, v in strat_now_raw.items()}

#                     px_u2 = ensure_price_worker(ib_local, u)
#                     u_sh2 = int(strat_now.get(u, 0))
#                     E = u_sh2 * px_u2

#                     for sym, sh0 in strat_now.items():
#                         sh = int(sh0)
#                         if sh == 0:
#                             continue
#                         uu = etf_to_under_all.get(sym)
#                         if uu != u:
#                             continue
#                         px_e2 = ensure_price_worker(ib_local, sym)
#                         lev2 = get_lev_or_raise(sym)
#                         E += sh * px_e2 * lev2

#                     return E / px_u2

#                 resid_before = compute_bucket_resid_sh_local(u_sym)
#                 delta_under_hedge = int(round(-resid_before))
#                 print(f"[HEDGE] {u_sym}: resid_before={resid_before:+.2f}sh -> delta_under_hedge={delta_under_hedge:+d} sh")

#                 if delta_under_hedge != 0:
#                     order_ref = f"{strategy_tag}|{u_sym}__GROUP|UNDER_DELTA"
#                     filled_u_abs, trade_u = exec_delta_local(u_sym, delta_under_hedge, px_u, order_ref)
#                     filled_u = filled_u_abs if delta_under_hedge > 0 else -filled_u_abs

#                     log_exposure_event(
#                         stage="POST_UNDER_GROUP",
#                         pair_id=f"{u_sym}__GROUP",
#                         underlying=u_sym,
#                         etf="",
#                         symbol=u_sym,
#                         delta_sh=delta_under_hedge,
#                         filled_sh=filled_u,
#                         trade=trade_u,
#                     )

#                     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     local_fills.append(
#                         {
#                             "filled_at": now,
#                             "run_date": run_date,
#                             "strategy_tag": strategy_tag,
#                             "pair_id": f"{u_sym}__GROUP",
#                             "underlying": u_sym,
#                             "etf": "",
#                             "px_under": px_u,
#                             "px_etf": None,
#                             "target_sh_under": target_u_sh,
#                             "target_sh_etf": None,
#                             "delta_sh_under": delta_under_hedge,
#                             "delta_sh_etf": 0,
#                             "filled_sh_under": filled_u,
#                             "filled_sh_etf": 0,
#                             "notes": f"GROUP_UNDER_HEDGE resid_before={resid_before:+.2f}sh",
#                         }
#                     )
#                 else:
#                     print(f"[GROUP] {u_sym}: already net-flat within rounding (resid_before={resid_before:+.2f}sh).")

#                 # Hedge truth AFTER underlying hedge
#                 print_bucket_exposure_breakdown(
#                     ib_local,
#                     baseline,
#                     etf_to_under_all,
#                     u_sym,
#                     title="POST_UNDER_HEDGE_TRUTH",
#                 )

#                 resid_after = compute_bucket_resid_sh_local(u_sym)
#                 print(f"[GROUP_NET] {u_sym}: resid_before={resid_before:+.2f}sh resid_after={resid_after:+.2f}sh")

#                 return local_fills

#             finally:
#                 try:
#                     ib_local.disconnect()
#                 except Exception:
#                     pass

#         # ---------------------------
#         # Execute buckets (serial or parallel)
#         # ---------------------------
#         if parallel_n == 1:
#             for idx, (u_sym, grp) in enumerate(approved):
#                 fills_to_append.extend(execute_underlying_bucket(u_sym, grp, worker_idx=idx))
#         else:
#             # NOTE: true interactive prompts must be done before parallel; we already did that above.
#             with ThreadPoolExecutor(max_workers=parallel_n) as ex:
#                 futs = []
#                 for idx, (u_sym, grp) in enumerate(approved):
#                     futs.append(ex.submit(execute_underlying_bucket, u_sym, grp, idx))

#                 for fut in as_completed(futs):
#                     fills_to_append.extend(fut.result())

#         # Final portfolio exposure snapshot
#         log_exposure_event(
#             stage="FINAL",
#             pair_id="FINAL",
#             underlying="",
#             etf="",
#             symbol="PORTFOLIO",
#             delta_sh=0,
#             filled_sh=0,
#             trade=None,
#         )

#         if fills_to_append:
#             append_fills(fills_to_append, fills_path)

#         print("[DONE] Execution pass complete.")

#     finally:
#         try:
#             ib.disconnect()
#         except Exception:
#             pass


# if __name__ == "__main__":
#     main()
