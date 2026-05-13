"""
execute_trade_plan.py — execution library for the LS strategy.

This module is a **library**: it defines the building blocks that the live
runners import. There is no longer a standalone CLI entry point — running
``python execute_trade_plan.py`` directly will refuse to start.

Live importers
--------------
* ``rebalance_strategy.py``   — primary runner; drives Cleanup, Establish,
                                 Resize (Phase 2b), and Hedge phases
* ``execute_flow_program.py`` — flow sleeve runner (weekly basket)
* ``harvest_underexposed_shorts.py`` — short-sleeve top-up
* ``phase2b_resize.py``       — imported via rebalance_strategy

What lives here
---------------
* IB connection + market-data helpers (``connect_ib``, ``get_snapshot_price``,
  ``ensure_price_coordinator``, ``configure_ib_error_log_filter``)
* Order construction (``build_market_order`` / ``build_adaptive_market_order`` /
  ``build_limit_order``) and the leg executor ``execute_leg`` that handles
  adaptive-market vs limit fallback, retries, FTP short-availability gating,
  and IB error 201 (SHORT_BLOCKED) graceful handling
* Position helpers (``current_ib_positions``, ``strategy_position_only``,
  ``load_baseline_qty``, ``_sync_positions_after_external_trades``)
* Cleanup pass: ``build_cleanup_trades_to_match_plan`` /
  ``execute_cleanup_trades_parallel`` — closes ETFs no longer in the plan
  (skips purgatory; never trades underlyings here)
* Snapshot / IO helpers (``resolve_plan_path``, ``resolve_fills_path``,
  ``write_execution_snapshot``, ``append_fills``, ``append_csv_row``,
  ``append_jsonl``)

Operating rules these helpers enforce
-------------------------------------
1. Cleanup never closes purgatory ETFs and never opens or hedges underlyings.
2. New positions are never opened in purgatory ETFs (callers are expected to
   filter; ``execute_leg`` itself is symbol-agnostic).
3. ``execute_leg`` returns ``SHORT_BLOCKED`` immediately on IB error 201
   (not available for short sale) without retry storms.
4. IBKR FTP short availability is a hard gate for new shorts: if available
   == 0 the leg is skipped; if available < requested the order is capped.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import ftplib
import io
import json
import logging
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, Order, Stock, TagValue, Trade
from ib_insync.objects import ExecutionFilter


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
# IB error-log filtering (noise suppression)
# =============================================================================

class _IBErrorCodeFilter(logging.Filter):
    def __init__(self, codes: Set[int]):
        super().__init__()
        self.codes = {int(c) for c in codes}

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        for code in self.codes:
            if f"Error {code}," in msg:
                return False
        return True


def configure_ib_error_log_filter(suppress_codes: Optional[Iterable[int]] = None) -> None:
    """
    Suppress selected noisy IB API error codes from logs while keeping
    all other warnings/errors visible.
    """
    codes = {10089} if suppress_codes is None else {int(c) for c in suppress_codes}
    if not codes:
        return

    for name in ("ib_insync.wrapper", "ib_insync.client", "ibapi.wrapper", "ibapi.client"):
        lg = logging.getLogger(name)
        key = f"_ls_algo_ib_filter_{'_'.join(str(c) for c in sorted(codes))}"
        if getattr(lg, key, False):
            continue
        lg.addFilter(_IBErrorCodeFilter(codes))
        setattr(lg, key, True)


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
    df["borrow_current_annual"] = fee

    sub = df[df["sym"].isin(want)].copy()

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for _, r in sub.iterrows():
        sym = str(r["sym"])
        avail = r.get("available_int", pd.NA)
        borrow = r.get("borrow_current_annual", pd.NA)
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
        # Only clientId=0 may call reqAutoOpenOrders; other client IDs
        # throw IB error 321 ("Only the default client can auto bind").
        if int(client_id) == 0:
            ib.reqAutoOpenOrders(True)
        else:
            tprint(
                f"[IB] coordinator=True with clientId={client_id}; "
                "skipping reqAutoOpenOrders (requires clientId=0)."
            )

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

def _ensure_market_data_type(ib: IB, data_type: int) -> None:
    """
    Set market data type at most once per IB connection, unless it changes.
    This avoids repeatedly calling reqMarketDataType for every symbol.
    """
    try:
        cur = getattr(ib, "_ls_algo_market_data_type", None)
        if cur == int(data_type):
            return
        ib.reqMarketDataType(int(data_type))
        setattr(ib, "_ls_algo_market_data_type", int(data_type))
    except Exception:
        pass

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
        _ensure_market_data_type(ib, data_type)
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
        # Use delayed streaming snapshots first and avoid switching
        # market-data type per symbol.
        px = snapshot_with_type(3)
    else:
        # Live first; delayed fallback if live entitlements are missing.
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

# When a 201 ("not available for short sale") rejection comes in, IB
# transitions the order PendingSubmit → Inactive → Cancelled(errorCode=201).
# The Inactive→Cancelled gap is small (typically < 500 ms) but the
# errorCode=201 entry is appended only with the Cancelled event.
#
# A naive ``wait_for_trade_terminal`` that returns at the first TERMINAL
# status would exit at "Inactive" with no errorCode in trade.log, so the
# caller's ``is_short_not_available(code, msg)`` check misses, the order
# falls into the retry path, and a *second* identical order is submitted
# only to be rejected with the same 201. To avoid that double‑attempt,
# we treat "Inactive" as transient and keep polling until either:
#   * status reaches a hard terminal ("cancelled" / "filled"), OR
#   * an errorCode (>0) gets appended to trade.log, OR
#   * a small grace window elapses (default 1.5 s).
INACTIVE_GRACE_SEC: float = 1.5
HARD_TERMINAL: Set[str] = {"filled", "cancelled"}


def _trade_has_error_code(trade: Trade) -> bool:
    try:
        for entry in (getattr(trade, "log", None) or []):
            code = getattr(entry, "errorCode", None)
            if code and int(code) != 0:
                return True
    except Exception:
        pass
    return False


def wait_for_trade_terminal(ib: IB, trade: Trade, timeout: float = 180.0) -> Trade:
    t0 = time.time()
    inactive_since: Optional[float] = None
    while time.time() - t0 < timeout:
        if stop_requested():
            return trade
        st = (trade.orderStatus.status or "").lower()
        if st in HARD_TERMINAL:
            return trade
        if st == "inactive":
            # Resolve "inactive" only when the cause is known (errorCode
            # logged) or after a short grace window, so the caller can
            # distinguish 201 (do-not-retry) from a benign Inactive that
            # later cancels.
            now = time.time()
            if inactive_since is None:
                inactive_since = now
            if _trade_has_error_code(trade):
                return trade
            if now - inactive_since >= INACTIVE_GRACE_SEC:
                return trade
        else:
            inactive_since = None
        ib.waitOnUpdate(timeout=0.25)
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


def is_short_unavailable_now(
    symbol: str,
    *,
    short_map: Optional[Dict[str, dict]] = None,
    screener_avail_map: Optional[Dict[str, int]] = None,
) -> Tuple[bool, str]:
    """Pre-submit "no locate" gate combining live FTP and screener evidence.

    Returns ``(blocked, reason)`` where ``reason`` is one of
    ``""``, ``"ftp_avail0"``, or ``"screener_avail0"``.

    Block conditions (positive evidence only — unknown stays "allow"):
      * FTP says ``available <= 0`` (existing behaviour).
      * Screener-CSV ``shares_available <= 0`` for this symbol — covers
        the FTP-missing case where ``daily_screener`` coerced NaN→0 and
        flagged ``borrow_missing_from_ftp=True``. Without this, a symbol
        absent from the live FTP feed (``avail=None``) would slip past
        the check and get rejected by IB with error 201.

    A symbol absent from BOTH maps is treated as "unknown → allow",
    matching the legacy behaviour for off-plan / off-screener tickers.
    """
    short_map = short_map or {}
    screener_avail_map = screener_avail_map or {}
    sym = str(symbol or "").upper().strip()
    if not sym:
        return False, ""
    avail = (short_map.get(sym) or {}).get("available")
    if avail is not None and int(avail) <= 0:
        return True, "ftp_avail0"
    sc_avail = screener_avail_map.get(sym)
    if sc_avail is not None and int(sc_avail) <= 0:
        return True, "screener_avail0"
    return False, ""

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
        cancelled_trades: List[Trade] = []
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
                cancelled_trades.append(tr)
            except Exception:
                pass

        # Wait for ALL cancelled orders to reach terminal state.
        # This prevents the caller from submitting a new order while the
        # old one is still live/filling — the root cause of double-fills.
        if cancelled_trades:
            CANCEL_CONFIRM_TIMEOUT = 10.0
            t0 = time.time()
            while time.time() - t0 < CANCEL_CONFIRM_TIMEOUT:
                all_done = True
                for tr in cancelled_trades:
                    st = (tr.orderStatus.status or "").strip().lower()
                    if st not in TERMINAL:
                        all_done = False
                        break
                if all_done:
                    break
                try:
                    ib.sleep(0.25)
                except Exception:
                    break
            # Log any that didn't reach terminal (rare but informative)
            for tr in cancelled_trades:
                st = (tr.orderStatus.status or "").strip().lower()
                if st not in TERMINAL:
                    try:
                        oid = int(getattr(tr.order, "orderId", 0) or 0)
                        tprint(
                            f"[CANCEL_COORD] WARNING: order {oid} for {symbol} "
                            f"still status={st} after cancel confirm wait."
                        )
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
    prefer_delayed_exec = bool(exec_cfg.get("prefer_delayed", True))
    # After IB error 354 (no market data), TWS rejects *adaptive / market*
    # algos until a quote stream exists. Retrying ADAPTIVE on att2/att3
    # just repeats 354 — flip to limit-only for the rest of this leg.
    mktdata_force_lmt = False

    def cancel_global() -> int:
        if dry_run:
            return 0
        if cancel_service is None:
            return 0
        if not strategy_tag_local:
            return 0
        return cancel_service.cancel(symbol_u, strategy_tag_local)

    def refresh_limit_reference_px() -> float:
        """Best-effort snapshot for LMT anchoring after 354 or when quotes are thin."""
        try:
            px = get_snapshot_price(ib, symbol_u, prefer_delayed=prefer_delayed_exec)
            if px is not None and float(px) > 0:
                return float(px)
        except Exception:
            pass
        return float(ref_price)

    def submit_limit_fallback(
        *,
        base_px: float,
        attempt: int,
        tag: str,
        ctx: str,
    ) -> int:
        """Place a limit at base_px±limit_bps; return filled shares from this order."""
        nonlocal last_trade_local
        adj = float(bps) / 10000.0
        lmt_px = base_px * (1.0 + adj) if action.upper() == "BUY" else base_px * (1.0 - adj)
        lmt_remain = qty - filled_total
        if lmt_remain <= 0:
            return 0
        o2 = build_limit_order(
            action=action,
            qty=lmt_remain,
            lmt_price=lmt_px,
            order_ref=f"{order_ref}|att{attempt}|{tag}",
        )
        tprint(f"{ctx}[LEG] {symbol_u} {tag} LMT @ {lmt_px:.4f} (ref={base_px:.4f})")
        trade2 = ib.placeOrder(contract, o2)
        last_trade_local = trade2
        _ok2, trade2 = wait_for_trade_accepted(ib, trade2, timeout=accept_timeout)
        trade2 = wait_for_trade_terminal(ib, trade2, timeout=done_timeout)
        last_trade_local = trade2
        return int(trade2.orderStatus.filled or 0)

    for attempt in range(1, max_retries + 1):
        if stop_requested():
            break

        remain = qty - filled_total
        if remain <= 0:
            break

        # Cancel any stale orders for this symbol and WAIT for them to
        # reach terminal state.  This is critical: without confirmation,
        # the old order can still fill while the new one is also live,
        # causing double/triple fills (the root cause of the BUY-SELL-BUY
        # cleanup bug).
        n_cancel = cancel_global()
        if n_cancel:
            ctx0 = f"[{context}]" if context else ""
            tprint(f"{ctx0}[CANCEL] {symbol_u}: cancelled {n_cancel} stale orders (global, confirmed terminal)")
            # Extra settle time: even after terminal confirmation, TWS may
            # still be processing internal state.  Give it a moment.
            ib.sleep(1.0)

        ctx = f"[{context}]" if context else ""

        # IB precautionary 354 path — only discretionary limits are accepted.
        if mktdata_force_lmt and order_style == "ADAPTIVE_MKT":
            base = refresh_limit_reference_px()
            filled2 = submit_limit_fallback(
                base_px=base, attempt=attempt, tag="LMT_ONLY", ctx=ctx,
            )
            filled_total = min(qty, filled_total + max(0, filled2))
            if attempt < max_retries and filled_total < qty:
                ib.sleep(1.5)
            continue

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

            # Before retrying, ensure this rejected order is truly dead.
            # Cancel it explicitly and wait for terminal, then collect any
            # partial fills it may have accumulated before rejection.
            try:
                ib.cancelOrder(trade.order)
            except Exception:
                pass
            trade = wait_for_trade_terminal(ib, trade, timeout=15.0)
            last_trade_local = trade
            rejected_filled = int(trade.orderStatus.filled or 0)
            if rejected_filled > 0:
                filled_total = min(qty, filled_total + rejected_filled)
                tprint(
                    f"{ctx}[LEG] {symbol_u} NOT_ACK but had {rejected_filled} partial fills "
                    f"before rejection; filled_total={filled_total}"
                )

            # Precautionary 354 can arrive before adaptive is "accepted" — do not
            # burn retries on another ADAPTIVE; go straight to LMT.
            if is_mktdata_block(code, msg) and order_style == "ADAPTIVE_MKT":
                mktdata_force_lmt = True
                base = refresh_limit_reference_px()
                tprint(
                    f"{ctx}[LEG] {symbol_u} NOT_ACK mktdata block (code={code}); "
                    f"LMT-only ref={base:.4f}"
                )
                filled2 = submit_limit_fallback(
                    base_px=base, attempt=attempt, tag="LMT_FALLBACK", ctx=ctx,
                )
                filled_total = min(qty, filled_total + max(0, filled2))
                if attempt < max_retries and filled_total < qty:
                    ib.sleep(1.5)
                continue

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
            # Collect any partial fills from the cancelled adaptive order
            filled_total = min(qty, filled_total + max(0, filled))
            cancel_global()
            mktdata_force_lmt = True
            # Refresh anchor — stale plan ref_price is a common reason LMT sits
            # far from the touch after a 354.
            base = refresh_limit_reference_px()
            tprint(f"{ctx}[LEG] {symbol_u} mktdata cancelled (code={code}); LMT fallback ref={base:.4f}")
            filled2 = submit_limit_fallback(
                base_px=base, attempt=attempt, tag="LMT_FALLBACK", ctx=ctx,
            )
            filled_total = min(qty, filled_total + max(0, filled2))
            continue

        filled_total = min(qty, filled_total + max(0, filled))

        if status not in TERMINAL:
            # Order didn't reach terminal within timeout — force cancel and
            # wait for confirmation before potentially retrying.
            try:
                ib.cancelOrder(trade.order)
            except Exception:
                pass
            trade = wait_for_trade_terminal(ib, trade, timeout=15.0)
            last_trade_local = trade
            # Re-read filled in case it filled during the cancel
            post_cancel_filled = int(trade.orderStatus.filled or 0)
            if post_cancel_filled > filled:
                extra = post_cancel_filled - filled
                filled_total = min(qty, filled_total + extra)
                tprint(
                    f"{ctx}[LEG] {symbol_u} picked up {extra} extra fills during cancel; "
                    f"filled_total={filled_total}"
                )

        # Settle between attempts: let TWS propagate state before next order
        if attempt < max_retries and filled_total < qty:
            ib.sleep(1.5)

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


def build_screener_avail_by_etf(screened: pd.DataFrame) -> Dict[str, int]:
    """ETF → ``shares_available`` int from the screened CSV.

    ``daily_screener.screen_universe`` already coerces NaN→0 and casts to
    int; symbols missing from the IBKR FTP file end up as 0 here. Used by
    ``is_short_unavailable_now`` as positive screener evidence that a
    locate is not available, even when the live FTP precheck returns
    ``avail=None`` (symbol absent from the FTP file at rebalance time).
    Conservative: if either column is missing, return ``{}``.
    """
    if "shares_available" not in screened.columns or "ETF" not in screened.columns:
        return {}
    tmp = screened[["ETF", "shares_available"]].copy()
    tmp["ETF"] = tmp["ETF"].astype(str).map(norm_sym)
    sh = pd.to_numeric(tmp["shares_available"], errors="coerce").fillna(0).astype(int)
    out: Dict[str, int] = {}
    for etf, val in zip(tmp["ETF"].tolist(), sh.tolist()):
        if not etf or etf == "NAN":
            continue
        out[str(etf)] = int(val)
    return out


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

# =============================================================================
# Cleanup: close ETF positions that are no longer in the plan
#   - skips purgatory ETFs (those are intentionally held / frozen)
#   - never opens or hedges underlyings here; that is rebalance_strategy's job
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
    flow_short_etfs: Set[str],   # <-- NEW
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
        if sym in flow_short_etfs:
            # Do NOT close flow-program shorts in cleanup
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

        # Borrow data — informational only for cleanup closes.
        # Cleanup always closes to zero (cover a short or sell a long),
        # neither of which requires borrow availability.  The price gate
        # below catches truly untradeable contracts (delisted, etc.).
        b = borrow_by_etf.get(e_sym, None)

        try:
            px = ensure_price_coordinator(ib, e_sym, prices, prefer_delayed)
        except (RuntimeError, Exception) as e:
            px = None
        if px is None:
            tprint(f"[CLEANUP] WARNING: No usable price for {e_sym}; skipping cleanup close for this symbol.")
            continue

        delta = -sh
        action = "BUY" if delta > 0 else "SELL"
        qty = abs(delta)

        trades.append({
            "symbol": e_sym,
            "action": action,
            "qty": qty,
            "px": float(px),
            "notional": float(qty) * float(px),
            "borrow_annual": float(b) if b is not None else None,
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

# ---------------------------------------------------------------------------
# Worker-pool sizing
# ---------------------------------------------------------------------------
# Hard ceiling on concurrent IB API connections from this process. TWS allows
# ~32 client IDs by default; we leave headroom for the coordinator (clientId=0
# / cleanup base / cancel coordinator / etc.) and any operator-side tools.
MAX_TWS_CLIENTS: int = 30


def pick_worker_count(
    *,
    n_trades: int,
    parallel_n_cfg: int,
    hard_cap: int,
    label: str = "",
) -> int:
    """Return the worker count for a phase.

    Workers = ``min(parallel_n_cfg, n_trades, hard_cap, MAX_TWS_CLIENTS)``,
    floored at 1. This guarantees we never spin up more IB connections than
    there are trades to execute (a 1-trade cleanup uses 1 worker, not 25)
    while still allowing each phase to scale up to its configured ceiling
    when the work justifies it.

    Emits a single ``tprint`` so the operator can see *why* the chosen
    count was picked.
    """
    n = max(1, int(n_trades))
    cfg = max(1, int(parallel_n_cfg))
    cap = max(1, int(hard_cap))
    chosen = min(n, cfg, cap, MAX_TWS_CLIENTS)
    chosen = max(1, chosen)
    tag = f"[{label}]" if label else "[WORKERS]"
    tprint(
        f"{tag} workers={chosen} (trades={n_trades}, cfg={parallel_n_cfg}, "
        f"phase_cap={hard_cap}, tws_cap={MAX_TWS_CLIENTS})"
    )
    return chosen


def _make_worker_pool(
    *,
    parallel_n: int,
    host: str,
    port: int,
    base_client_id: int,
    connect_stagger_sec: float = 0.25,
) -> List[IB]:
    """Open exactly ``parallel_n`` IB connections, sleeping briefly between
    each so TWS isn't asked to accept all of them in the same tick.

    ``connect_stagger_sec`` defaults to 0.25 s — at 25 workers that's a
    ~6 s ramp-up, well within TWS's per-second client tolerance and far
    below the default API request burst limits."""
    ibs: List[IB] = []
    for i in range(int(parallel_n)):
        cid = base_client_id + i
        ibw = connect_ib(host, port, cid)
        tprint(f"[CLEANUP] Worker connected clientId={ibw.client.clientId}")
        ibs.append(ibw)
        time.sleep(max(0.0, float(connect_stagger_sec)))
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


# =============================================================================
# Cleanup executor (UPDATED)
# - Keeps retrying until we can CONFIRM the ETF position is FLAT (strategy-only, post-baseline)
# - Cancels stale orders before retrying (via CoordinatorCancelService called inside execute_leg)
# - NO underlying hedging during cleanup
# =============================================================================

def _get_strategy_only_shares(ib: IB, sym: str, baseline: Dict[str, float]) -> int:
    """
    Strategy-only (post-baseline) shares for sym.
    """
    sym = norm_sym(sym)
    ib_pos_now = current_ib_positions(ib)
    strat_now_raw = strategy_position_only(ib_pos_now, baseline)
    return int(round(float(strat_now_raw.get(sym, 0.0) or 0.0)))


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
    # NOTE: we ignore impacted_underlyings now (cleanup hedging disabled)
    if not approved_trades:
        tprint("[CLEANUP] Nothing to do.")
        return

    # ---- record start time for executions pull (even if dry_run, harmless)
    cleanup_start_utc = dt.datetime.now(dt.timezone.utc)

    # Ensure we have prices for approved symbols
    syms = [norm_sym(t["symbol"]) for t in approved_trades]
    for s in syms:
        if stop_requested():
            return
        if s not in prices:
            prices[s] = get_snapshot_price(ib, s, prefer_delayed=prefer_delayed)

    host = str(exec_cfg.get("ib_host_override", cfg.get("ibkr", {}).get("host", "127.0.0.1")))
    port = int(exec_cfg.get("ib_port_override", cfg.get("ibkr", {}).get("port", 7497)))
    base_client = int(cfg.get("ibkr", {}).get("client_id", 3)) + 500

    cleanup_cap = int(exec_cfg.get("cleanup_max_workers", parallel_n) or parallel_n)
    connect_stagger_sec = float(exec_cfg.get("worker_connect_stagger_sec", 0.25))
    n_workers = pick_worker_count(
        n_trades=len(approved_trades),
        parallel_n_cfg=parallel_n,
        hard_cap=cleanup_cap,
        label="CLEANUP",
    )
    worker_ibs = _make_worker_pool(
        parallel_n=n_workers,
        host=host,
        port=port,
        base_client_id=base_client,
        connect_stagger_sec=connect_stagger_sec,
    )

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

        # ---------------------------------------------------------------------
        # First attempt: parallel submit (fast)
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Visibility: pull executions since cleanup start (account-wide)
        # ---------------------------------------------------------------------
        if not dry_run:
            try:
                closed_syms = [norm_sym(t["symbol"]) for t in approved_trades]
                fills = _pull_executions_since(ib, cleanup_start_utc, symbols=closed_syms)
                if fills:
                    tprint(f"[CLEANUP] Observed {len(fills)} cleanup executions via reqExecutions().")
                else:
                    tprint("[CLEANUP] WARNING: No cleanup executions observed via reqExecutions() (may still be pending).")
            except Exception as e:
                tprint(f"[CLEANUP] WARNING: reqExecutions pull failed: {type(e).__name__}: {e}")

        # ---------------------------------------------------------------------
        # Sync coordinator positions after worker trades
        # ---------------------------------------------------------------------
        if not dry_run:
            try:
                watch = [norm_sym(t["symbol"]) for t in approved_trades]
                _sync_positions_after_external_trades(ib, watch_syms=watch, timeout_s=15.0)
            except Exception as e:
                tprint(f"[CLEANUP] WARNING: position sync failed: {type(e).__name__}: {e}")

        # ---------------------------------------------------------------------
        # NEW: Confirm + retry until FLAT (serial on coordinator)
        # - Recomputes remaining shares each attempt (handles partial fills)
        # - Uses execute_leg() which cancels stale orders via cancel_service
        # ---------------------------------------------------------------------
        if not dry_run:
            max_confirm_retries = int(exec_cfg.get("cleanup_confirm_retries", 3))
            confirm_sync_timeout = float(exec_cfg.get("cleanup_confirm_sync_timeout_s", 10.0))

            # We may have multiple dicts for same symbol; de-dup by symbol
            seen_syms: Set[str] = set()
            ordered = []
            for tr in approved_trades:
                s = norm_sym(tr["symbol"])
                if s in seen_syms:
                    continue
                seen_syms.add(s)
                ordered.append(tr)

            for tr in ordered:
                if stop_requested():
                    return

                sym = norm_sym(tr["symbol"])

                # Quick check: already flat?
                sh0 = _get_strategy_only_shares(ib, sym, baseline)
                if sh0 == 0:
                    tprint(f"[CLEANUP] {sym} confirmed flat after initial submits.")
                    continue

                for att in range(1, max_confirm_retries + 1):
                    if stop_requested():
                        return

                    # refresh coordinator view
                    try:
                        _sync_positions_after_external_trades(ib, watch_syms=[sym], timeout_s=confirm_sync_timeout)
                    except Exception:
                        pass

                    sh_now = _get_strategy_only_shares(ib, sym, baseline)
                    if sh_now == 0:
                        tprint(f"[CLEANUP] {sym} confirmed flat after retry att{att-1}.")
                        break

                    # Flatten remaining shares (IMPORTANT: recompute qty each retry)
                    delta = -int(sh_now)
                    action = "BUY" if delta > 0 else "SELL"
                    qty = abs(delta)

                    # Ensure we have a price
                    px = ensure_price_coordinator(ib, sym, prices, prefer_delayed)

                    # Contract from cache
                    base = contract_cache.get(sym)
                    if not base or int(getattr(base, "conId", 0) or 0) <= 0:
                        raise RuntimeError(f"[CLEANUP] Missing cached conId for {sym}")

                    c = Stock(base.symbol, base.exchange, base.currency)
                    c.conId = int(base.conId)
                    c.primaryExchange = getattr(base, "primaryExchange", "") or ""

                    order_ref = f"{strategy_tag}|CLEANUP|{sym}|ETF_FLATTEN"
                    tprint(f"[CLEANUP] Retry att{att}: {sym} still open (sh={sh_now}); flatten {action} qty={qty} ...")

                    _res = execute_leg(
                        ib=ib,  # coordinator only for retries
                        symbol=sym,
                        action=action,
                        qty=qty,
                        ref_price=float(px),
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

                    # After attempt, sync + check again
                    try:
                        _sync_positions_after_external_trades(ib, watch_syms=[sym], timeout_s=confirm_sync_timeout)
                    except Exception:
                        pass

                    sh_after = _get_strategy_only_shares(ib, sym, baseline)
                    if sh_after == 0:
                        tprint(f"[CLEANUP] {sym} confirmed flat after retry att{att}.")
                        break

                    if att == max_confirm_retries:
                        tprint(f"[CLEANUP] ERROR: {sym} still not flat after {max_confirm_retries} confirm retries (sh={sh_after}).")

        # Cleanup never trades underlyings — that is rebalance_strategy's job.
        if impacted_underlyings:
            tprint(
                f"[CLEANUP] NOTE: {len(impacted_underlyings)} impacted underlying(s) "
                "passed in; cleanup ignores them by design."
            )

        tprint("[CLEANUP] Cleanup pass complete.\n")

    finally:
        for ibw in worker_ibs:
            try:
                ibw.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    sys.stderr.write(
        "execute_trade_plan.py is a library, not a runner. "
        "Use rebalance_strategy.py (or execute_flow_program.py / "
        "harvest_underexposed_shorts.py) instead.\n"
    )
    sys.exit(2)
