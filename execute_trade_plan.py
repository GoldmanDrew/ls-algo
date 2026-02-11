#!/usr/bin/env python3
"""
execute_trade_plan_updated.py

UPDATED FOR PURGATORY RULES + SHORT-SALE SAFETY:

1) Cleanup pre-pass ONLY closes ETF positions that are:
   - strategy-held (post-baseline)
   - mapped in screened universe
   - NOT present in proposed_trades.csv
   - AND NOT in purgatory (per screened truth)

2) Execution NEVER opens new positions in purgatory ETFs:
   - Purgatory ETFs must have long_usd==0 and short_usd==0 in the plan (hard-check)
   - We skip sizing/trading for purgatory ETFs entirely in bucket execution
   - We also prevent plan-based delta math from implicitly closing purgatory (freeze deltas)

3) NEW: Graceful handling for IB Error 201 (not available for short sale)
   - We do NOT spam retries. We return a SHORT_BLOCKED status immediately.

4) NEW: Use IBKR FTP short availability as a hard gate for NEW shorts
   - If we need to INCREASE a short (delta < 0) and FTP available == 0 -> skip placing order.
   - If available is positive but less than requested -> cap the order to available (partial).
   - This avoids repeated 201 rejects and “multiple orders” churn.

Everything else remains consistent with your current working behavior.
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
import traceback
import threading
import time
import queue
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

from ib_insync import IB, Stock, Order, Trade, util
from ib_insync.objects import ExecutionFilter
import time
import datetime as dt
from typing import Optional, Iterable, Dict, Any, List, Tuple


import pandas as pd
import yaml
from ib_insync import IB, Stock, Order, Trade, MarketOrder, TagValue
from generate_trade_plan import load_blacklist


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
    return STOP_FILE.exists() or SHUTDOWN.is_set()

def handle_sigint(signum, frame):
    # First Ctrl+C => graceful; second Ctrl+C => hard exit
    if not stop_requested():
        tprint("\n[CTRL+C] Shutdown requested. Stopping new work…")
        SHUTDOWN.set()
    else:
        tprint("\n[CTRL+C] Forced exit.")
        sys.exit(1)

def ensure_thread_event_loop() -> asyncio.AbstractEventLoop:
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
    "ts","run_date","strategy_tag","stage","pair_id","underlying","etf","symbol",
    "delta_sh","filled_sh","fill_avg_px","mark_px","delta_notional","pos_sh",
    "pos_notional","gross_long","gross_short","net_notional",
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

def norm_sym(x: str) -> str:
    return str(x).upper().strip().replace(".", "-")

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

def connect_ib(host: str, port: int, client_id: int, coordinator: bool = False) -> IB:
    ensure_thread_event_loop()
    ib = IB()
    ib.RequestTimeout = 60
    ib.connect(host, port, clientId=client_id)
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR.")

    if coordinator:
        ib.reqAutoOpenOrders(True)

    try:
        ib.reqIds(-1)
        ib.sleep(0.5)
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

def build_limit_order(action: str, qty: int, lmt_price: float, order_ref: str) -> Order:
    o = Order()
    o.action = action.upper()
    o.totalQuantity = int(qty)
    o.orderType = "LMT"
    o.lmtPrice = float(lmt_price)
    o.tif = "DAY"
    o.transmit = True
    o.orderRef = order_ref
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
        ib.waitOnUpdate(timeout=0.5)
    return trade

def wait_for_trade_accepted(ib: IB, trade: Trade, timeout: float = 120.0) -> Tuple[bool, Trade]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stop_requested():
            return False, trade
        ib.sleep(0.25)
        st_raw = trade.orderStatus.status or ""
        st = st_raw.strip().lower()
        try:
            oid = int(getattr(trade.order, "orderId", 0) or 0)
        except Exception:
            oid = 0
        if st in ACCEPTED or oid > 0:
            return True, trade
        if st in TERMINAL:
            return False, trade
        try:
            ib.reqOpenOrders()
        except Exception:
            pass
    st_raw = trade.orderStatus.status or ""
    st = st_raw.strip().lower()
    if st in ACCEPTED:
        return True, trade
    return False, trade

def strategy_tag_from_order_ref(order_ref: str) -> str:
    try:
        s = str(order_ref or "")
        if "|" not in s:
            return ""
        return s.split("|", 1)[0].strip()
    except Exception:
        return ""

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

def is_short_not_available(code: Optional[int], msg: Optional[str]) -> bool:
    m = (msg or "").lower()
    return (code == 201) and ("not available for short sale" in m)

def is_mktdata_block(code: Optional[int], msg: Optional[str]) -> bool:
    m = (msg or "").lower()
    return (
        code in (354, 10089)
        or ("without having market data" in m)
        or ("requires additional subscription" in m)
    )

@dataclass
class ExecResult:
    filled: int
    trade: Optional[Trade]
    status: str                  # "FILLED"/"PARTIAL"/"NOFILL"/"SHORT_BLOCKED"/"FAILED"
    error_code: Optional[int] = None
    error_msg: Optional[str] = None


# =============================================================================
# Global cancellation coordinator (clientId=0)
# =============================================================================

@dataclass
class CancelRequest:
    symbol: str
    strategy_tag: str
    resp_q: "queue.Queue[int]"

class CoordinatorCancelService:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._q: "queue.Queue[Optional[CancelRequest]]" = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="IBCancelCoordinator", daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            raise RuntimeError("Cancel coordinator did not become ready (timeout).")

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)
        try:
            self._thread.join(timeout=5.0)
        except Exception:
            pass

    def cancel(self, symbol: str, strategy_tag: str, timeout: float = 10.0) -> int:
        if not strategy_tag:
            return 0
        if self._stop.is_set():
            return 0
        resp_q: "queue.Queue[int]" = queue.Queue(maxsize=1)
        req = CancelRequest(symbol=norm_sym(symbol), strategy_tag=str(strategy_tag).strip(), resp_q=resp_q)
        try:
            self._q.put(req, timeout=1.0)
        except Exception:
            return 0
        try:
            return int(resp_q.get(timeout=timeout))
        except Exception:
            return 0

    def _run(self) -> None:
        ensure_thread_event_loop()
        try:
            ib = connect_ib(self.host, self.port, client_id=0, coordinator=True)
            self._ready.set()
        except Exception as e:
            tprint(f"[CANCEL_COORD] FAILED to connect clientId=0: {type(e).__name__}: {e}")
            self._ready.set()
            return

        try:
            while not self._stop.is_set() and not stop_requested():
                try:
                    item = self._q.get(timeout=0.25)
                except queue.Empty:
                    try:
                        ib.reqOpenOrders()
                        ib.sleep(0.05)
                    except Exception:
                        pass
                    continue

                if item is None:
                    break

                try:
                    n = self._cancel_stale_orders_for_symbol_impl(
                        ib=ib,
                        symbol=item.symbol,
                        strategy_tag=item.strategy_tag,
                    )
                except Exception:
                    n = 0

                try:
                    item.resp_q.put_nowait(int(n))
                except Exception:
                    pass
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    def _cancel_stale_orders_for_symbol_impl(self, *, ib: IB, symbol: str, strategy_tag: str) -> int:
        symbol = norm_sym(symbol)
        strategy_tag = str(strategy_tag).strip()
        if not symbol or not strategy_tag:
            return 0

        cancelled = 0
        try:
            ib.reqOpenOrders()
            ib.sleep(0.10)
        except Exception:
            pass

        for tr in list(ib.openTrades()):
            try:
                st = (tr.orderStatus.status or "").strip().lower()
                if st in TERMINAL:
                    continue

                ref = str(getattr(tr.order, "orderRef", "") or "")
                if not ref.startswith(strategy_tag + "|"):
                    continue

                c = getattr(tr, "contract", None)
                if c is None:
                    continue

                sym = norm_sym(universal_symbol_from_ib(getattr(c, "symbol", "") or ""))
                if sym != symbol:
                    continue

                ib.cancelOrder(tr.order)
                cancelled += 1
            except Exception:
                pass

        if cancelled:
            try:
                ib.sleep(0.10)
            except Exception:
                pass

        return cancelled


# =============================================================================
# Execution primitive (uses cancel coordinator)
# =============================================================================

QUALIFY_LOCK = threading.Lock()

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
    contract: Optional[Stock] = None,
    cancel_service: Optional[CoordinatorCancelService] = None,
) -> ExecResult:
    if qty <= 0:
        return ExecResult(filled=0, trade=None, status="NOFILL")
    if stop_requested():
        tprint(f"[{context}][LEG] Shutdown active; skipping {symbol} {action} qty={qty}")
        return ExecResult(filled=0, trade=None, status="FAILED")

    symbol_u = norm_sym(symbol)

    if contract is None:
        contract = make_stock(symbol_u)
        with QUALIFY_LOCK:
            ib.qualifyContracts(contract)

    filled_total = 0
    last_trade_local: Optional[Trade] = None

    order_style = str(exec_cfg.get("order_style", "ADAPTIVE_MKT")).strip().upper()
    accept_timeout = float(exec_cfg.get("market_accept_timeout_sec", 120.0))
    done_timeout   = float(exec_cfg.get("market_done_timeout_sec", 300.0))

    strategy_tag_local = strategy_tag_from_order_ref(order_ref)

    def cancel_global() -> int:
        if dry_run:
            return 0
        if cancel_service is None:
            return 0
        if not strategy_tag_local:
            return 0
        return cancel_service.cancel(symbol_u, strategy_tag_local)

    for attempt in range(1, max_retries + 1):
        if stop_requested():
            break

        remain = qty - filled_total
        if remain <= 0:
            break

        n_cancel = cancel_global()
        if n_cancel:
            tprint(f"[{context}][CANCEL] {symbol_u}: cancelled {n_cancel} stale orders (global)")

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
        tprint(f"{ctx}[LEG] {symbol_u} {action} qty={remain} px={px_str} refTag={o.orderRef} clientId={ib.client.clientId}")

        if dry_run:
            filled_total += remain
            continue

        trade = ib.placeOrder(contract, o)
        last_trade_local = trade

        accepted, trade = wait_for_trade_accepted(ib, trade, timeout=accept_timeout)

        if not accepted:
            st = (trade.orderStatus.status or "").strip()
            code, msg = last_ib_error(trade)

            # HARD-GRACEFUL: short not available -> do not retry/spam
            if is_short_not_available(code, msg):
                tprint(f"{ctx}[LEG] {symbol_u} SHORT_BLOCKED (201): cannot increase short now. status={st}")
                return ExecResult(
                    filled=int(filled_total),
                    trade=trade,
                    status="SHORT_BLOCKED",
                    error_code=code,
                    error_msg=msg,
                )

            cancel_global()
            tprint(f"{ctx}[LEG] {symbol_u} NOT_ACK (status={st} code={code} msg={msg}); retrying.")
            ib.sleep(1.0)
            continue

        trade = wait_for_trade_terminal(ib, trade, timeout=done_timeout)
        last_trade_local = trade

        status = (trade.orderStatus.status or "").strip().lower()
        filled = int(trade.orderStatus.filled or 0)

        code, msg = last_ib_error(trade)

        # If it terminal-cancelled due to short block, do not retry.
        if status in ("cancelled", "inactive") and is_short_not_available(code, msg):
            tprint(f"{ctx}[LEG] {symbol_u} SHORT_BLOCKED (201) after submit: cannot increase short now. status={status}")
            filled_total = min(qty, filled_total + max(0, filled))
            return ExecResult(
                filled=int(filled_total),
                trade=trade,
                status="SHORT_BLOCKED",
                error_code=code,
                error_msg=msg,
            )

        # Market-data subscription block -> fallback LMT (existing behavior)
        if status == "cancelled" and is_mktdata_block(code, msg):
            cancel_global()
            adj = float(bps) / 10000.0
            lmt = ref_price * (1.0 + adj) if action.upper() == "BUY" else ref_price * (1.0 - adj)

            o2 = build_limit_order(
                action=action,
                qty=remain,
                lmt_price=lmt,
                order_ref=f"{order_ref}|att{attempt}|LMT_FALLBACK",
            )
            tprint(f"{ctx}[LEG] {symbol_u} fallback LMT @ {lmt:.4f} (code={code})")

            trade2 = ib.placeOrder(contract, o2)
            last_trade_local = trade2

            _ok2, trade2 = wait_for_trade_accepted(ib, trade2, timeout=accept_timeout)
            trade2 = wait_for_trade_terminal(ib, trade2, timeout=done_timeout)
            last_trade_local = trade2

            filled2 = int(trade2.orderStatus.filled or 0)
            filled_total = min(qty, filled_total + max(0, filled2))
            continue

        filled_total = min(qty, filled_total + max(0, filled))

        if status not in TERMINAL:
            try:
                ib.cancelOrder(trade.order)
                ib.sleep(0.5)
            except Exception:
                pass

    # Final status
    if dry_run:
        return ExecResult(filled=int(filled_total), trade=last_trade_local, status="FILLED")
    if filled_total <= 0:
        code, msg = last_ib_error(last_trade_local)
        return ExecResult(filled=0, trade=last_trade_local, status="FAILED", error_code=code, error_msg=msg)
    if filled_total < qty:
        return ExecResult(filled=int(filled_total), trade=last_trade_local, status="PARTIAL")
    return ExecResult(filled=int(filled_total), trade=last_trade_local, status="FILLED")


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
        sym = norm_sym(universal_symbol_from_ib(p.contract.symbol))
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
# Borrow parsing helpers
# =============================================================================

def _coerce_borrow_annual(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            if not s:
                return None
            if s.endswith("%"):
                return float(s[:-1]) / 100.0
            v = float(s)
        else:
            v = float(x)
    except Exception:
        return None
    if v < 0:
        return None
    if v > 1.5:
        return v / 100.0
    return v

def detect_borrow_col(df: pd.DataFrame) -> Optional[str]:
    if "borrow_current" in df.columns:
        return "borrow_current"
    for c in df.columns:
        if "borrow" in str(c).lower():
            return c
    return None

def build_borrow_by_etf(screened: pd.DataFrame) -> Dict[str, Optional[float]]:
    borrow_col = detect_borrow_col(screened)
    out: Dict[str, Optional[float]] = {}
    if borrow_col is None:
        return out
    tmp = screened.copy()
    tmp["ETF"] = tmp["ETF"].astype(str).map(norm_sym)
    tmp[borrow_col] = tmp[borrow_col].apply(_coerce_borrow_annual)
    for _, r in tmp.iterrows():
        e = str(r["ETF"])
        b = r[borrow_col]
        out[e] = None if pd.isna(b) else float(b)
    return out

def build_purgatory_set(screened: pd.DataFrame) -> Set[str]:
    """
    ETFs that are in purgatory per screened truth.
    Conservative: if column missing, return empty set.
    """
    if "purgatory" not in screened.columns:
        return set()
    tmp = screened.copy()
    tmp["ETF"] = tmp["ETF"].astype(str).map(norm_sym)
    purg = tmp.loc[tmp["purgatory"] == True, "ETF"]  # noqa: E712
    return set(purg.dropna().astype(str).tolist())


# =============================================================================
# Hedge truth helpers (screened truth)
# =============================================================================

def ensure_price_coordinator(ib: IB, sym: str, prices: Dict[str, float], prefer_delayed: bool) -> float:
    sym = norm_sym(sym)
    px = prices.get(sym)
    if px is not None:
        return float(px)
    px2 = get_snapshot_price(ib, sym, prefer_delayed=prefer_delayed)
    prices[sym] = float(px2)
    return float(px2)

def compute_group_residual_sh_equiv(
    ib: IB,
    *,
    baseline: Dict[str, float],
    prices: Dict[str, float],
    prefer_delayed: bool,
    u_sym: str,
    etf_to_under_all: Dict[str, str],
    leverage_by_etf_all: Dict[str, float],
) -> float:
    u_sym = norm_sym(u_sym)

    ib_pos_now = current_ib_positions(ib)
    strat_now_raw = strategy_position_only(ib_pos_now, baseline)
    strat_now = {norm_sym(k): int(round(float(v))) for k, v in strat_now_raw.items()}

    px_u = ensure_price_coordinator(ib, u_sym, prices, prefer_delayed)
    u_sh = int(strat_now.get(u_sym, 0))
    E = u_sh * px_u

    for sym, sh in strat_now.items():
        if sh == 0:
            continue
        uu = etf_to_under_all.get(sym)
        if uu != u_sym:
            continue
        px_e = ensure_price_coordinator(ib, sym, prices, prefer_delayed)
        lev = leverage_by_etf_all.get(sym)
        if lev is None:
            raise ValueError(f"Missing leverage for ETF {sym} in screened universe; cannot hedge accurately.")
        E += sh * px_e * float(lev)

    return E / px_u if px_u else 0.0


# =============================================================================
# Coordinator-wide cancel helpers
# =============================================================================

def cancel_all_strategy_orders(ib: IB, strategy_tag: str) -> int:
    strategy_tag = str(strategy_tag or "").strip()
    if not strategy_tag:
        return 0
    cancelled = 0
    try:
        ib.reqOpenOrders()
        ib.sleep(0.25)
    except Exception:
        pass

    for tr in list(ib.openTrades()):
        try:
            st = (tr.orderStatus.status or "").strip().lower()
            if st in TERMINAL:
                continue
            ref = str(getattr(tr.order, "orderRef", "") or "")
            if not ref.startswith(strategy_tag + "|"):
                continue
            ib.cancelOrder(tr.order)
            cancelled += 1
        except Exception:
            pass

    if cancelled:
        ib.sleep(0.25)
    return cancelled

def wait_until_no_strategy_open_orders(ib: IB, strategy_tag: str, timeout: float = 20.0) -> bool:
    strategy_tag = str(strategy_tag or "").strip()
    if not strategy_tag:
        return True
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stop_requested():
            return False
        try:
            ib.reqOpenOrders()
        except Exception:
            pass

        open_cnt = 0
        try:
            for tr in list(ib.openTrades()):
                st = (tr.orderStatus.status or "").strip().lower()
                if st in TERMINAL:
                    continue
                ref = str(getattr(tr.order, "orderRef", "") or "")
                if ref.startswith(strategy_tag + "|"):
                    open_cnt += 1
        except Exception:
            open_cnt = 0

        if open_cnt == 0:
            return True
        ib.sleep(0.5)
    return False


# =============================================================================
# Cleanup to match plan (UPDATED: skip purgatory)
# =============================================================================

def build_cleanup_trades_to_match_plan(
    *,
    ib: IB,
    baseline: Dict[str, float],
    plan: pd.DataFrame,
    screened: pd.DataFrame,
    prices: Dict[str, float],
    prefer_delayed: bool,
    blacklist: Set[str],
    etf_to_under_all: Dict[str, str],
    borrow_by_etf: Dict[str, Optional[float]],
    purgatory_etfs: Set[str],
) -> Tuple[List[dict], List[str]]:
    plan_etfs = set(plan["ETF"].astype(str).map(norm_sym).tolist()) if "ETF" in plan.columns else set()

    ib_pos_now = current_ib_positions(ib)
    strat_now_raw = strategy_position_only(ib_pos_now, baseline)
    strat_now = {norm_sym(k): int(round(float(v))) for k, v in strat_now_raw.items()}
    strat_syms = {s for s, sh in strat_now.items() if sh != 0}

    unwanted_etfs = []
    for sym in strat_syms:
        if sym in blacklist:
            continue
        if sym in purgatory_etfs:
            # Do NOT close purgatory positions even if not in plan
            continue
        if sym in etf_to_under_all:
            if sym not in plan_etfs:
                unwanted_etfs.append(sym)

    trades: List[dict] = []
    impacted_under = set()

    for e_sym in sorted(unwanted_etfs):
        sh = int(strat_now.get(e_sym, 0))
        if sh == 0:
            continue

        u = etf_to_under_all.get(e_sym)
        if not u or u in blacklist:
            continue

        px = ensure_price_coordinator(ib, e_sym, prices, prefer_delayed)
        delta = -sh
        action = "BUY" if delta > 0 else "SELL"
        qty = abs(delta)

        b = borrow_by_etf.get(e_sym)
        trades.append({
            "symbol": e_sym,
            "action": action,
            "qty": qty,
            "px": float(px),
            "notional": float(qty) * float(px),
            "borrow_annual": b,
            "reason": "CLEANUP_CLOSE_UNWANTED_ETF",
            "underlying_group": u,
        })
        impacted_under.add(u)

    return trades, sorted(impacted_under)


def print_cleanup_trade_list(trades: List[dict]) -> None:
    if not trades:
        tprint("[CLEANUP] No unwanted ETF positions found. No cleanup trades needed.")
        return
    tprint("\n" + "=" * 110)
    tprint("[CLEANUP] Proposed trades to CLOSE ETF legs not in proposed_trades.csv (excluding purgatory)")
    tprint("=" * 110)
    total_notional = 0.0
    for r in trades:
        total_notional += float(r.get("notional", 0.0) or 0.0)
        b = r.get("borrow_annual", None)
        b_str = "n/a" if b is None else f"{float(b):.2%}"
        tprint(
            f"  {r['symbol']:<8} {r['action']:<4} qty={int(r['qty']):>8,d} "
            f"px={float(r['px']):>10.4f} notional={fmt_dollars(float(r['notional'])):<12} "
            f"borrow={b_str:<8} group={r.get('underlying_group','')}"
        )
    tprint(f"\n[CLEANUP] Total gross notional (approx): {fmt_dollars(total_notional)}")
    tprint("=" * 110 + "\n")


# =============================================================================
# Contract cache helper
# =============================================================================

def build_contract_cache(ib: IB, symbols: List[str]) -> Dict[str, Stock]:
    uniq = []
    seen = set()
    for s in symbols:
        s = norm_sym(s)
        if not s or s in seen:
            continue
        uniq.append(s)
        seen.add(s)

    tprint(f"[QUALIFY] Building contract cache for {len(uniq)} symbols (serial)...")
    out: Dict[str, Stock] = {}

    for s in uniq:
        if stop_requested():
            break
        c = make_stock(s)
        try:
            with QUALIFY_LOCK:
                ib.qualifyContracts(c)
            out[s] = Stock(c.symbol, c.exchange, c.currency)
            out[s].primaryExchange = getattr(c, "primaryExchange", "") or ""
            out[s].conId = int(getattr(c, "conId", 0) or 0)
            if out[s].conId <= 0:
                raise RuntimeError("conId not set after qualify")
        except Exception as e:
            tprint(f"[QUALIFY] WARNING {s}: {type(e).__name__}: {e}")
    return out


# =============================================================================
# Cleanup executor (kept as in your file; uses execute_leg + hedging)
# NOTE: unchanged except for being called with purgatory-safe cleanup list.
# =============================================================================

def _make_worker_pool(*, parallel_n: int, host: str, port: int, base_client_id: int) -> List[IB]:
    ibs: List[IB] = []
    for i in range(parallel_n):
        cid = base_client_id + i
        ibw = connect_ib(host, port, cid)
        tprint(f"[CLEANUP] Worker connected clientId={ibw.client.clientId}")
        ibs.append(ibw)
        time.sleep(0.10)
    return ibs

# =============================================================================
# Cleanup executor (UPDATED)
# - Fixes “cleanup filled but main process doesn’t see it” by:
#   (1) forcing an executions pull (account-wide) after worker trades
#   (2) forcing a fresh positions snapshot on the coordinator connection
#   (3) waiting briefly for the coordinator connection to reflect new positions
# =============================================================================

def _ib_time_str_utc(t: dt.datetime) -> str:
    """IB ExecutionFilter 'time' wants 'YYYYMMDD HH:MM:SS'."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    t = t.astimezone(dt.timezone.utc)
    return t.strftime("%Y%m%d %H:%M:%S")


def _pull_executions_since(
    ib: IB,
    start_utc: dt.datetime,
    symbols: Optional[Iterable[str]] = None,
) -> List[Any]:
    """
    Pull executions since start_utc. This is account-wide and not tied
    to the worker clientId (unlike trade.fills callbacks).
    Returns the raw Fill objects from ib_insync.
    """
    symset = {norm_sym(s) for s in symbols} if symbols else None
    flt = ExecutionFilter(time=_ib_time_str_utc(start_utc))
    fills = ib.reqExecutions(flt)  # returns List[Fill]
    if symset is None:
        return fills
    out = []
    for f in fills:
        try:
            s = norm_sym(getattr(f.contract, "symbol", "") or "")
        except Exception:
            s = ""
        if s and s in symset:
            out.append(f)
    return out


def _sync_positions_after_external_trades(
    ib: IB,
    *,
    watch_syms: Optional[Iterable[str]] = None,
    timeout_s: float = 10.0,
) -> None:
    """
    Ensure the coordinator IB connection refreshes positions after trades
    placed on OTHER clientIds (workers).
    """
    watch = {norm_sym(s) for s in watch_syms} if watch_syms else set()

    # snapshot before
    before = current_ib_positions(ib)
    before_norm = {norm_sym(k): float(v) for k, v in before.items()}

    # force a positions refresh request (TWS pushes, but this nudges it)
    try:
        ib.reqPositions()
    except Exception:
        pass

    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        ib.sleep(0.5)
        now = current_ib_positions(ib)
        now_norm = {norm_sym(k): float(v) for k, v in now.items()}

        if not watch:
            # if no watch list, just accept after first refresh tick
            if now_norm != before_norm:
                return
            continue

        # if any watched symbol changed, we're synced enough
        changed = False
        for s in watch:
            if float(now_norm.get(s, 0.0) or 0.0) != float(before_norm.get(s, 0.0) or 0.0):
                changed = True
                break
        if changed:
            return

    # fall through: we tried; positions will still likely be correct soon,
    # but we don’t want to block forever.
    tprint(f"[CLEANUP] WARNING: coordinator positions may be stale after {timeout_s:.1f}s sync wait.")


def execute_cleanup_trades_parallel(
    *,
    approved_trades: List[dict],
    impacted_underlyings: List[str],
    ib: IB,
    baseline: Dict[str, float],
    prices: Dict[str, float],
    prefer_delayed: bool,
    etf_to_under_all: Dict[str, str],
    leverage_by_etf_all: Dict[str, float],
    exec_cfg: Dict,
    strategy_tag: str,
    limit_bps: float,
    timeout: float,
    max_retries: int,
    dry_run: bool,
    parallel_n: int,
    contract_cache: Dict[str, Stock],
    cfg: dict,
    cancel_service: Optional[CoordinatorCancelService],
) -> None:
    if not approved_trades and not impacted_underlyings:
        tprint("[CLEANUP] Nothing to do.")
        return

    # ---- record start time for executions pull (even if dry_run, harmless)
    cleanup_start_utc = dt.datetime.now(dt.timezone.utc)

    if approved_trades:
        syms = [norm_sym(t["symbol"]) for t in approved_trades]
        for s in syms:
            if stop_requested():
                return
            if s not in prices:
                prices[s] = get_snapshot_price(ib, s, prefer_delayed=prefer_delayed)

    host = str(exec_cfg.get("ib_host_override", cfg.get("ibkr", {}).get("host", "127.0.0.1")))
    port = int(exec_cfg.get("ib_port_override", cfg.get("ibkr", {}).get("port", 7497)))
    base_client = int(cfg.get("ibkr", {}).get("client_id", 3)) + 500

    n_workers = max(1, int(parallel_n))
    worker_ibs = _make_worker_pool(parallel_n=n_workers, host=host, port=port, base_client_id=base_client)

    try:
        def worker_execute_trade(tr: dict, worker_idx: int) -> None:
            ensure_thread_event_loop()
            if stop_requested():
                return

            sym = norm_sym(tr["symbol"])
            ib_local = worker_ibs[worker_idx % len(worker_ibs)]

            base = contract_cache.get(sym)
            if not base or int(getattr(base, "conId", 0) or 0) <= 0:
                raise RuntimeError(f"[CLEANUP] Missing cached conId for {sym}")

            c = Stock(base.symbol, base.exchange, base.currency)
            c.conId = int(base.conId)
            c.primaryExchange = getattr(base, "primaryExchange", "") or ""

            px = float(prices[sym])
            order_ref = f"{strategy_tag}|CLEANUP|{sym}|ETF_FLATTEN"

            _res = execute_leg(
                ib=ib_local,
                symbol=sym,
                action=str(tr["action"]),
                qty=int(tr["qty"]),
                ref_price=px,
                bps=limit_bps,
                order_ref=order_ref,
                exec_cfg=exec_cfg,
                timeout=timeout,
                max_retries=max_retries,
                dry_run=dry_run,
                context="CLEANUP",
                contract=c,
                cancel_service=cancel_service,
            )

        if approved_trades:
            tprint(f"[CLEANUP] Executing {len(approved_trades)} ETF close trades (parallel_n={n_workers}) ...")
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(worker_execute_trade, tr, i) for i, tr in enumerate(approved_trades)]
                for fut in as_completed(futs):
                    if stop_requested():
                        break
                    try:
                        fut.result()
                    except Exception as e:
                        tprint(f"[CLEANUP] WARNING: ETF cleanup worker error: {type(e).__name__}: {e}")
                        tprint(traceback.format_exc())

        if stop_requested():
            return

        # -----------------------------------------------------------------------------
        # NEW: pull executions since cleanup start on coordinator
        # This is the key to “seeing” what happened across clientIds.
        # -----------------------------------------------------------------------------
        if approved_trades and not dry_run:
            try:
                closed_syms = [norm_sym(t["symbol"]) for t in approved_trades]
                fills = _pull_executions_since(ib, cleanup_start_utc, symbols=closed_syms)
                if fills:
                    tprint(f"[CLEANUP] Observed {len(fills)} cleanup executions via reqExecutions().")
                else:
                    tprint("[CLEANUP] WARNING: No cleanup executions observed via reqExecutions() (may still be pending).")
            except Exception as e:
                tprint(f"[CLEANUP] WARNING: reqExecutions pull failed: {type(e).__name__}: {e}")

        # -----------------------------------------------------------------------------
        # NEW: force coordinator positions to refresh after worker trades
        # Without this, subsequent logic (including next run’s cleanup detection)
        # can think MSTW/MSTX are still open.
        # -----------------------------------------------------------------------------
        if approved_trades and not dry_run:
            try:
                watch = [norm_sym(t["symbol"]) for t in approved_trades]
                _sync_positions_after_external_trades(ib, watch_syms=watch, timeout_s=12.0)
            except Exception as e:
                tprint(f"[CLEANUP] WARNING: position sync failed: {type(e).__name__}: {e}")

        # ---- underlying hedge step (now uses refreshed coordinator positions)
        if impacted_underlyings:
            tprint(f"[CLEANUP] Hedging {len(impacted_underlyings)} impacted underlyings (serial on coordinator)...")
            for u_sym in impacted_underlyings:
                if stop_requested():
                    break
                u_sym = norm_sym(u_sym)
                resid_sh = compute_group_residual_sh_equiv(
                    ib,
                    baseline=baseline,
                    prices=prices,
                    prefer_delayed=prefer_delayed,
                    u_sym=u_sym,
                    etf_to_under_all=etf_to_under_all,
                    leverage_by_etf_all=leverage_by_etf_all,
                )
                delta_under = int(round(-resid_sh))
                if delta_under == 0:
                    continue
                px_u = ensure_price_coordinator(ib, u_sym, prices, prefer_delayed)
                action = "BUY" if delta_under > 0 else "SELL"
                qty = abs(delta_under)
                order_ref = f"{strategy_tag}|CLEANUP|{u_sym}|UNDER_DELTA"

                _resu = execute_leg(
                    ib=ib,
                    symbol=u_sym,
                    action=action,
                    qty=qty,
                    ref_price=px_u,
                    bps=limit_bps,
                    order_ref=order_ref,
                    exec_cfg=exec_cfg,
                    timeout=timeout,
                    max_retries=max_retries,
                    dry_run=dry_run,
                    context="CLEANUP_UNDER",
                    cancel_service=cancel_service,
                )

        tprint("[CLEANUP] Cleanup pass complete.\n")

    finally:
        for ibw in worker_ibs:
            try:
                ibw.disconnect()
            except Exception:
                pass



# =============================================================================
# Postpass hedge sweep (unchanged)
# =============================================================================

def hedge_all_screened_underlyings_postpass(
    *,
    ib: IB,
    baseline: Dict[str, float],
    prices: Dict[str, float],
    prefer_delayed: bool,
    underlyings: List[str],
    etf_to_under_all: Dict[str, str],
    leverage_by_etf_all: Dict[str, float],
    exec_cfg: Dict,
    strategy_tag: str,
    limit_bps: float,
    timeout: float,
    max_retries: int,
    dry_run: bool,
    cancel_service: Optional[CoordinatorCancelService],
) -> None:
    tprint("\n" + "=" * 110)
    tprint("[POSTPASS] Hedging ALL screened underlyings to net-flat using underlying-only trades.")
    tprint("=" * 110 + "\n")

    for u_sym in underlyings:
        u_sym = norm_sym(u_sym)
        if stop_requested():
            tprint("[SHUTDOWN] Stop requested during postpass hedge sweep.")
            return

        try:
            resid_sh = compute_group_residual_sh_equiv(
                ib,
                baseline=baseline,
                prices=prices,
                prefer_delayed=prefer_delayed,
                u_sym=u_sym,
                etf_to_under_all=etf_to_under_all,
                leverage_by_etf_all=leverage_by_etf_all,
            )
        except Exception as e:
            tprint(f"[POSTPASS] {u_sym}: ERROR computing residual: {type(e).__name__}: {e}")
            continue

        delta_under = int(round(-resid_sh))
        if delta_under == 0:
            continue

        px_u = ensure_price_coordinator(ib, u_sym, prices, prefer_delayed)
        tprint(f"[POSTPASS] {u_sym}: resid={resid_sh:+.2f}sh -> trade underlying delta={delta_under:+d} sh")

        action = "BUY" if delta_under > 0 else "SELL"
        qty = abs(delta_under)
        order_ref = f"{strategy_tag}|{u_sym}__POSTPASS|UNDER_DELTA"

        _res = execute_leg(
            ib=ib,
            symbol=u_sym,
            action=action,
            qty=qty,
            ref_price=px_u,
            bps=limit_bps,
            order_ref=order_ref,
            exec_cfg=exec_cfg,
            timeout=timeout,
            max_retries=max_retries,
            dry_run=dry_run,
            context=f"{u_sym}|POSTPASS",
            cancel_service=cancel_service,
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
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
    blacklist = load_blacklist(cfg)
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

    plan = plan.reset_index(drop=True)
    plan["_orig_idx"] = plan.index
    plan["Underlying"] = plan["Underlying"].astype(str).map(norm_sym)
    plan["ETF"] = plan["ETF"].astype(str).map(norm_sym)
    plan = plan.sort_values(by=["Underlying", "_orig_idx"], kind="mergesort").reset_index(drop=True)
    plan = plan.drop(columns=["_orig_idx"]).reset_index(drop=True)

    baseline = load_baseline_qty(baseline_csv)
    exec_dir(run_date).mkdir(parents=True, exist_ok=True)
    exposure_csv = exec_dir(run_date) / "exposure_log.csv"
    exposure_jsonl = exec_dir(run_date) / "exposure_log.jsonl"
    log_lock = threading.Lock()

    ib = connect_ib(host, port, client_id)
    cancel_service = CoordinatorCancelService(host=host, port=port)
    cancel_service.start()
    tprint("[CANCEL_COORD] Started global cancel coordinator (clientId=0).")

    try:
        ib_pos = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)

        tprint(f"[POS] current IB symbols={len(ib_pos)}; baseline symbols={len(baseline)}; strategy-only symbols={len(strat_pos)}")
        tprint(f"[PLAN] Using: {plan_path}")
        tprint(f"[BASELINE] Using: {baseline_csv}")
        tprint(f"[EXEC] Writing to: {exec_dir(run_date)}")

        fills_to_append: List[dict] = []
        prices: Dict[str, float] = {}

        if "Beta" not in plan.columns:
            raise ValueError("Plan missing required column: Beta")

        plan["ETF_U"] = plan["ETF"].astype(str).map(norm_sym)
        plan["UNDER_U"] = plan["Underlying"].astype(str).map(norm_sym)

        leverage_by_etf_plan: Dict[str, float] = {}
        under_to_etfs_planned: Dict[str, Set[str]] = {}

        for _, r in plan.iterrows():
            e = str(r["ETF_U"])
            u = str(r["UNDER_U"])
            lev = float(r["Beta"])
            under_to_etfs_planned.setdefault(u, set()).add(e)
            if e in leverage_by_etf_plan and abs(leverage_by_etf_plan[e] - lev) > 1e-9:
                raise ValueError(f"Beta mismatch for {e} in plan: {leverage_by_etf_plan[e]} vs {lev}")
            leverage_by_etf_plan[e] = lev

        screened_csv = Path(paths_cfg.get("screened_csv", "data/etf_screened_today.csv"))
        if not screened_csv.exists():
            raise FileNotFoundError(f"Screened universe not found: {screened_csv}")

        screened = pd.read_csv(screened_csv)
        need_cols = {"Underlying", "ETF", "Beta"}
        if not need_cols.issubset(set(screened.columns)):
            raise ValueError(f"{screened_csv} missing required columns {need_cols}. Columns={list(screened.columns)}")

        screened["Underlying"] = screened["Underlying"].astype(str).map(norm_sym)
        screened["ETF"] = screened["ETF"].astype(str).map(norm_sym)

        borrow_by_etf = build_borrow_by_etf(screened)
        purgatory_etfs = build_purgatory_set(screened)
        tprint(f"[PURGATORY] Loaded {len(purgatory_etfs)} purgatory ETFs from screened universe.")

        # HARD CHECK: purgatory ETFs must have zero targets in plan
        if "purgatory" in plan.columns:
            plan["purgatory"] = plan["purgatory"].fillna(False).astype(bool)
        if "purgatory" in plan.columns:
            bad = plan[(plan["ETF_U"].isin(purgatory_etfs)) & ((plan["long_usd"].abs() > 1e-9) | (plan["short_usd"].abs() > 1e-9))]
            if not bad.empty:
                raise ValueError(f"[PLAN] Purgatory ETFs have nonzero targets. Refuse. Examples: {bad[['ETF_U','long_usd','short_usd']].head(10).to_dict('records')}")

        under_to_etfs_all: Dict[str, Set[str]] = {}
        leverage_by_etf_all: Dict[str, float] = {}
        for _, r in screened.iterrows():
            u = str(r["Underlying"])
            e = str(r["ETF"])
            lev = float(r["Beta"])
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

        # PRE-PASS cleanup (UPDATED: skips purgatory)
        cleanup_trades, impacted_under = build_cleanup_trades_to_match_plan(
            ib=ib,
            baseline=baseline,
            plan=plan,
            screened=screened,
            prices=prices,
            prefer_delayed=prefer_delayed,
            blacklist=blacklist,
            etf_to_under_all=etf_to_under_all,
            borrow_by_etf=borrow_by_etf,
            purgatory_etfs=purgatory_etfs,
        )

        print_cleanup_trade_list(cleanup_trades)

        if cleanup_trades:
            ans = input("[CLEANUP] Approve executing these cleanup trades? (y/n): ").strip().lower()
            if ans != "y":
                tprint("[CLEANUP] Not approved. Skipping cleanup prepass.")
            else:
                syms_to_qualify = [t["symbol"] for t in cleanup_trades] + list(impacted_under)
                contract_cache = build_contract_cache(ib, syms_to_qualify)

                execute_cleanup_trades_parallel(
                    approved_trades=cleanup_trades,
                    impacted_underlyings=impacted_under,
                    ib=ib,
                    baseline=baseline,
                    prices=prices,
                    prefer_delayed=prefer_delayed,
                    etf_to_under_all=etf_to_under_all,
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

        ib_pos = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)

        # Plan sanity: no planned ETF in multiple underlyings
        etf_seen: Dict[str, str] = {}
        for _, r in plan.iterrows():
            e = str(r["ETF_U"])
            u = str(r["UNDER_U"])
            prev = etf_seen.get(e)
            if prev is not None and prev != u:
                raise ValueError(f"Parallel unsafe: planned ETF {e} appears under multiple underlyings {prev} and {u}.")
            etf_seen[e] = u

        # Short availability snapshot (PLAN ETFs only)
        etf_symbols_plan = sorted(set(plan["ETF_U"].astype(str).str.upper()))
        try:
            short_map = fetch_ibkr_short_availability_map(etf_symbols_plan)
            tprint(f"[SHORT] Loaded availability for {len(short_map)}/{len(etf_symbols_plan)} plan ETFs from IBKR FTP.")
        except Exception as ex:
            tprint(f"[SHORT] WARNING: short availability precheck failed ({ex}); continuing without it.")
            short_map = {}

        # Prefetch prices for plan symbols + held mapped ETFs (screened)
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

        # Logging helper (thread-safe)
        def log_exposure_event(*, stage: str, pair_id: str, underlying: str, etf: str, symbol: str, delta_sh: int, filled_sh: int, trade: Optional[Trade]):
            ib_pos_now = current_ib_positions(ib)
            strat_pos_now = strategy_position_only(ib_pos_now, baseline)
            port = compute_portfolio_notionals({k: int(round(float(v))) for k, v in strat_pos_now.items()}, prices)

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

        # Shared helpers for workers
        def get_lev_or_raise(etf: str) -> float:
            lev = leverage_by_etf_all.get(etf)
            if lev is None:
                raise ValueError(f"Missing leverage for ETF {etf} in screened universe; cannot hedge accurately.")
            return float(lev)

        def ensure_price_worker(ib_local: IB, sym: str) -> float:
            sym = norm_sym(sym)
            px = prices.get(sym)
            if px is not None:
                return float(px)
            px2 = get_snapshot_price(ib_local, sym, prefer_delayed=prefer_delayed)
            with log_lock:
                prices[sym] = float(px2)
            return float(px2)

        # Build bucket list (plan order)
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

        # Worker: execute a single underlying bucket
        def execute_underlying_bucket(u_sym: str, grp: pd.DataFrame, worker_idx: int) -> List[dict]:
            ensure_thread_event_loop()
            if stop_requested():
                tprint(f"[{u_sym}] Shutdown before start; skipping bucket.")
                return []

            local_fills: List[dict] = []
            try:
                ib_local = connect_ib(host, port, client_id + 100 + worker_idx)
            except Exception as e:
                tprint(f"[{u_sym}] Worker could not connect to IB: {type(e).__name__}: {e}")
                return []

            try:
                ib_pos_local = current_ib_positions(ib_local)
                strat_pos_local = strategy_position_only(ib_pos_local, baseline)

                px_u = ensure_price_worker(ib_local, u_sym)

                bucket_target_etf_sh: Dict[str, int] = {}
                bucket_target_under_sh: int = 0

                # -------------------------
                # UPDATED: skip purgatory ETFs entirely
                # and enforce zero targets if they appear
                # -------------------------
                for _, row in grp.iterrows():
                    e_sym = norm_sym(str(row["ETF_U"]))
                    tu = float(row.get("long_usd", 0.0) or 0.0)
                    te = float(row.get("short_usd", 0.0) or 0.0)

                    if e_sym in purgatory_etfs:
                        if abs(tu) > 1e-9 or abs(te) > 1e-9:
                            raise ValueError(
                                f"[PLAN] Purgatory ETF {e_sym} has nonzero targets "
                                f"(long_usd={tu}, short_usd={te}). Refuse."
                            )
                        continue

                    px_e = ensure_price_worker(ib_local, e_sym)
                    bucket_target_under_sh += target_shares_from_usd(tu, px_u)
                    bucket_target_etf_sh[e_sym] = int(
                        bucket_target_etf_sh.get(e_sym, 0) + target_shares_from_usd(te, px_e)
                    )

                cur_u_sh = int(round(float(strat_pos_local.get(u_sym, 0.0))))
                target_u_sh = int(bucket_target_under_sh)

                bucket_delta_etf: Dict[str, int] = {}
                for e_sym, tgt_sh in bucket_target_etf_sh.items():
                    e_sym = norm_sym(e_sym)

                    # Never trade purgatory ETFs (no opens, no closes)
                    if e_sym in purgatory_etfs:
                        bucket_delta_etf[e_sym] = 0
                        continue

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
                    e_sym = norm_sym(e_sym)
                    px_e = ensure_price_worker(ib_local, e_sym)
                    cur_e_sh = int(round(float(strat_pos_local.get(e_sym, 0.0))))
                    tgt_e_sh = int(bucket_target_etf_sh[e_sym])
                    d_e = int(bucket_delta_etf.get(e_sym, 0))
                    lev = leverage_by_etf_plan.get(e_sym)
                    if lev is None:
                        raise ValueError(f"Missing leverage for planned ETF {e_sym} in plan columns.")
                    flag = " (PURGATORY-FROZEN)" if e_sym in purgatory_etfs else ""
                    tprint(
                        f"  ETF {e_sym}{flag}: px={px_e:.4f} lev={float(lev):+,.2f} "
                        f"cur={cur_e_sh:+d} tgt={tgt_e_sh:+d} delta={d_e:+d}"
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

                def exec_delta_local(symbol: str, delta: int, px: float, order_ref: str) -> ExecResult:
                    if delta == 0:
                        return ExecResult(filled=0, trade=None, status="NOFILL")
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
                        cancel_service=cancel_service,
                    )

                etf_items = list(bucket_delta_etf.items())

                def _etf_sort_key(item):
                    sym, d = item
                    return (0 if d < 0 else 1, sym)

                if short_first:
                    etf_items.sort(key=_etf_sort_key)
                else:
                    etf_items.sort(key=lambda x: x[0])

                # ETF delta loop (UPDATED: FTP gate + explicit purgatory guard + graceful 201)
                for e_sym, d_e in etf_items:
                    e_sym = norm_sym(e_sym)
                    if e_sym in purgatory_etfs:
                        continue

                    if stop_requested():
                        tprint(f"[{u_sym}] Shutdown during ETF loop; aborting bucket.")
                        return local_fills

                    if d_e == 0:
                        continue

                    # --- FTP HARD GATE for NEW SHORTS (delta < 0 means SELL more / increase short)
                    capped_delta = d_e
                    sm = short_map.get(e_sym)
                    if d_e < 0 and sm and sm.get("available") is not None:
                        avail = int(sm["available"])
                        want = abs(d_e)

                        if avail <= 0:
                            tprint(f"[SHORT] SKIP {e_sym}: wants {want} new short shares but FTP available=0.")
                            # Log skip as a fill-like record
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
                                    "px_etf": float(prices.get(e_sym, 0.0) or 0.0),
                                    "target_sh_under": target_u_sh,
                                    "target_sh_etf": int(bucket_target_etf_sh.get(e_sym, 0)),
                                    "delta_sh_under": 0,
                                    "delta_sh_etf": d_e,
                                    "filled_sh_under": 0,
                                    "filled_sh_etf": 0,
                                    "notes": f"SKIP_FTP_AVAIL0 wants_short={want}",
                                }
                            )
                            continue

                        if avail < want:
                            capped_delta = -avail
                            tprint(f"[SHORT] CAP {e_sym}: wants {want} new short shares but FTP available={avail}. Capping order to {avail}.")

                    px_e = ensure_price_worker(ib_local, e_sym)
                    order_ref = f"{strategy_tag}|{u_sym}__GROUP|{e_sym}|ETF_DELTA"

                    res_e = exec_delta_local(e_sym, capped_delta, px_e, order_ref)

                    # Convert filled abs to signed (direction based on capped_delta)
                    filled_e_signed = 0
                    if capped_delta < 0:
                        filled_e_signed = -int(res_e.filled)
                    else:
                        filled_e_signed = int(res_e.filled)

                    # Graceful 201 handling: log and continue bucket (hedge will adapt)
                    if res_e.status == "SHORT_BLOCKED":
                        tprint(f"[{u_sym}] {e_sym} SHORT_BLOCKED (201). Leaving remaining delta unfilled and continuing.")
                        # also include in fills
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
                                "delta_sh_etf": int(capped_delta),
                                "filled_sh_under": 0,
                                "filled_sh_etf": int(filled_e_signed),
                                "notes": f"ETF_SHORT_BLOCKED_201 msg={res_e.error_msg}",
                            }
                        )
                        log_exposure_event(
                            stage="POST_ETF",
                            pair_id=f"{u_sym}__GROUP",
                            underlying=u_sym,
                            etf=e_sym,
                            symbol=e_sym,
                            delta_sh=int(capped_delta),
                            filled_sh=int(filled_e_signed),
                            trade=res_e.trade,
                        )
                        continue

                    log_exposure_event(
                        stage="POST_ETF",
                        pair_id=f"{u_sym}__GROUP",
                        underlying=u_sym,
                        etf=e_sym,
                        symbol=e_sym,
                        delta_sh=int(capped_delta),
                        filled_sh=int(filled_e_signed),
                        trade=res_e.trade,
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
                            "delta_sh_etf": int(capped_delta),
                            "filled_sh_under": 0,
                            "filled_sh_etf": int(filled_e_signed),
                            "notes": "GROUP_ETF",
                        }
                    )

                # Underlying hedge step (same)
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
                    res_u = exec_delta_local(u_sym, delta_under_hedge, px_u, order_ref)
                    filled_u_signed = int(res_u.filled) if delta_under_hedge > 0 else -int(res_u.filled)

                    log_exposure_event(
                        stage="POST_UNDER_GROUP",
                        pair_id=f"{u_sym}__GROUP",
                        underlying=u_sym,
                        etf="",
                        symbol=u_sym,
                        delta_sh=delta_under_hedge,
                        filled_sh=filled_u_signed,
                        trade=res_u.trade,
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
                            "filled_sh_under": filled_u_signed,
                            "filled_sh_etf": 0,
                            "notes": f"GROUP_UNDER_HEDGE resid_before={resid_before:+.2f}sh status={res_u.status}",
                        }
                    )
                else:
                    tprint(f"[GROUP] {u_sym}: already net-flat within rounding (resid_before={resid_before:+.2f}sh).")

                resid_after = compute_bucket_resid_sh_local(u_sym)
                tprint(f"[GROUP_NET] {u_sym}: resid_before={resid_before:+.2f}sh resid_after={resid_after:+.2f}sh")
                return local_fills

            finally:
                try:
                    ib_local.disconnect()
                except Exception:
                    pass

        # Execute buckets
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
                        try:
                            fills_to_append.extend(fut.result())
                        except Exception as e:
                            if stop_requested() or SHUTDOWN.is_set():
                                tprint(f"[SHUTDOWN] Worker ended with {type(e).__name__}: {e}")
                                continue
                            tprint(f"[ERROR] Worker failed: {type(e).__name__}: {e}")
                            continue
                except KeyboardInterrupt:
                    handle_sigint(None, None)

        # POST-PASS: hedge EVERY screened underlying (including those not in the plan)
        etf_to_under_all_filtered = {etf: under for etf, under in etf_to_under_all.items() if etf not in blacklist and under not in blacklist}
        leverage_by_etf_all_filtered = {etf: lev for etf, lev in leverage_by_etf_all.items() if etf in etf_to_under_all_filtered}

        under_to_etfs_all_filtered: Dict[str, List[str]] = {}
        for etf, under in etf_to_under_all_filtered.items():
            under_to_etfs_all_filtered.setdefault(under, []).append(etf)

        for under in list(under_to_etfs_all_filtered.keys()):
            under_to_etfs_all_filtered[under] = sorted(set(under_to_etfs_all_filtered[under]))
            if not under_to_etfs_all_filtered[under]:
                under_to_etfs_all_filtered.pop(under, None)

        all_screened_underlyings = sorted(under_to_etfs_all_filtered.keys())

        hedge_all_screened_underlyings_postpass(
            ib=ib,
            baseline=baseline,
            prices=prices,
            prefer_delayed=prefer_delayed,
            underlyings=all_screened_underlyings,
            etf_to_under_all=etf_to_under_all_filtered,
            leverage_by_etf_all=leverage_by_etf_all_filtered,
            exec_cfg=exec_cfg,
            strategy_tag=strategy_tag,
            limit_bps=limit_bps,
            timeout=timeout,
            max_retries=max_retries,
            dry_run=dry_run,
            cancel_service=cancel_service,
        )

        # Final portfolio exposure snapshot
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
            cancel_service.stop()
        except Exception:
            pass
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
