#!/usr/bin/env python3
"""
execute_flow_program.py

Automatic daily "flow sleeve" allocator:
- Deploys annual_deployment_rate * deployment_base each year
- Converts to DAILY deployment using 252 trading days
- Allocates by weights (fixed) across configured short tickers
- Trades only the incremental deployment for *today* (SELL shares)
- Updates a flow ledger CSV tracking cumulative deployed USD per ticker

Designed to run independently of generate_trade_plan.py / execute_trade_plan.py.
Schedule it daily (cron / Task Scheduler) after your main execution pass.
"""

from __future__ import annotations

import os
import math
import signal
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from ib_insync import IB, Stock  # type: ignore


TRADING_DAYS = 252


import math
import numpy as np
from datetime import datetime, timezone

from ib_insync import IB, Stock, util

# -----------------------------
# Robust price helpers (NO live mkt data required)
# -----------------------------

def _as_float(x):
    try:
        x = float(x)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan

def get_price_fallback_ib(
    ib: IB,
    sym: str,
    *,
    prefer_delayed: bool = True,
    timeout: float = 8.0,
) -> float:
    """
    Returns a usable price WITHOUT needing live market data subscription.
    Tries historical bars first (usually works even when reqMktData errors 10089).
    """
    sym = str(sym).upper().replace(".", "-")
    c = Stock(sym, "SMART", "USD")

    # qualify first (fast, deterministic)
    ib.qualifyContracts(c)

    # 1) intraday 1-min bars (last close)
    try:
        bars = ib.reqHistoricalData(
            c,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )
        if bars:
            px = _as_float(bars[-1].close)
            if np.isfinite(px) and px > 0:
                return float(px)
    except Exception:
        pass

    # 2) daily bars (last close)
    try:
        bars = ib.reqHistoricalData(
            c,
            endDateTime="",
            durationStr="5 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )
        if bars:
            px = _as_float(bars[-1].close)
            if np.isfinite(px) and px > 0:
                return float(px)
    except Exception:
        pass

    raise RuntimeError(f"No usable historical price for {sym} (bars empty).")

# -----------------------------
# Safe sizing
# -----------------------------
def usd_to_shares_floor(delta_usd: float, px: float) -> int:
    """
    Convert USD notional to shares, rounding toward zero but ensuring non-trivial.
    """
    px = _as_float(px)
    delta_usd = _as_float(delta_usd)
    if not np.isfinite(px) or px <= 0 or not np.isfinite(delta_usd):
        return 0
    return int(delta_usd / px)  # toward zero

# -----------------------------
# Order execution that doesn't require market data
# -----------------------------
def place_adaptive_mkt(
    ib: IB,
    sym: str,
    *,
    action: str,
    qty: int,
    order_ref: str,
    priority: str = "Patient",
):
    """
    Places Adaptive Market order (your existing style) which can work even without
    quote subscriptions, as long as IB allows blind trading / precautionary settings.
    """
    from ib_insync import Order

    sym = str(sym).upper().replace(".", "-")
    c = Stock(sym, "SMART", "USD")
    ib.qualifyContracts(c)

    o = Order(
        action=action,
        totalQuantity=int(qty),
        orderType="MKT",
        tif="DAY",
        orderRef=order_ref,
        algoStrategy="Adaptive",
        algoParams=[("adaptivePriority", priority)],
    )
    return ib.placeOrder(c, o)

def place_collared_limit_using_prev_close(
    ib: IB,
    sym: str,
    *,
    action: str,
    qty: int,
    prev_close: float,
    collar_bps: float = 300.0,  # 3%
    order_ref: str,
):
    """
    If you want to AVOID blind market orders, use last close and a wide collar.
    BUY: limit = close*(1+collar), SELL: limit = close*(1-collar)
    """
    from ib_insync import LimitOrder

    sym = str(sym).upper().replace(".", "-")
    c = Stock(sym, "SMART", "USD")
    ib.qualifyContracts(c)

    collar = float(collar_bps) / 1e4
    if action.upper() == "BUY":
        lmt = float(prev_close) * (1.0 + collar)
    else:
        lmt = float(prev_close) * (1.0 - collar)

    o = LimitOrder(
        action=action.upper(),
        totalQuantity=int(qty),
        lmtPrice=round(lmt, 4),
        tif="DAY",
        orderRef=order_ref,
    )
    return ib.placeOrder(c, o)

# -----------------------------
# FLOW daily execution snippet
# -----------------------------
def execute_flow_daily(
    ib: IB,
    *,
    flow_targets: pd.DataFrame,
    run_date: str,
    weights: dict[str, float],
    annual_deployment_rate: float,
    net_liq_usd: float,
    prefer_delayed: bool,
    dry_run: bool,
    strategy_tag: str,
    use_collared_limits: bool = False,
    collar_bps: float = 300.0,
):
    """
    Example:
    - Determine daily USD deployment = net_liq * annual_rate / 252
    - Allocate across tickers by weights
    - SELL that many shares (short more) each day
    """
    # daily budget
    daily_budget = float(net_liq_usd) * float(annual_deployment_rate) / 252.0

    # normalize weights
    w = {k.upper().replace(".", "-"): float(v) for k, v in weights.items()}
    wsum = sum(abs(v) for v in w.values())
    if wsum <= 0:
        print("[FLOW_DAILY] No weights; skipping.")
        return
    w = {k: v / wsum for k, v in w.items()}

    print(f"[FLOW_DAILY] daily_budget_usd={daily_budget:,.2f}")

    for sym, ww in w.items():
        sym = sym.upper().replace(".", "-")
        delta_usd = daily_budget * ww  # positive means add more short USD
        if abs(delta_usd) < 1.0:
            continue

        try:
            px = get_price_fallback_ib(ib, sym, prefer_delayed=prefer_delayed)
        except Exception as e:
            print(f"[FLOW_DAILY] SKIP {sym}: cannot get historical px ({type(e).__name__}: {e})")
            continue

        sh = -abs(usd_to_shares_floor(delta_usd, px))
        if sh == 0:
            continue

        action = "SELL"  # increasing short
        qty = abs(int(sh))
        order_ref = f"{strategy_tag}|FLOW_DAILY|{sym}|DEPLOY"

        print(f"[FLOW_DAILY] {sym}: px≈{px:.4f} deploy_usd={delta_usd:,.0f} -> {action} {qty} (short more)")

        if dry_run:
            continue

        try:
            if use_collared_limits:
                place_collared_limit_using_prev_close(
                    ib,
                    sym,
                    action=action,
                    qty=qty,
                    prev_close=px,
                    collar_bps=collar_bps,
                    order_ref=order_ref,
                )
            else:
                # Adaptive MKT path
                place_adaptive_mkt(
                    ib,
                    sym,
                    action=action,
                    qty=qty,
                    order_ref=order_ref,
                    priority="Patient",
                )
        except Exception as e:
            print(f"[FLOW_DAILY] ORDER FAIL {sym}: {type(e).__name__}: {e}")
            continue


# -----------------------------
# Small utilities
# -----------------------------
def tprint(msg: str) -> None:
    print(msg, flush=True)


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def norm_sym(x: str) -> str:
    return str(x).upper().replace(".", "-")


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)


def load_ledger(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["date", "ticker", "delta_usd", "cum_usd"])
    df = pd.read_csv(path)
    for c in ["date", "ticker"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).map(norm_sym)
    if "delta_usd" in df.columns:
        df["delta_usd"] = pd.to_numeric(df["delta_usd"], errors="coerce").fillna(0.0)
    if "cum_usd" in df.columns:
        df["cum_usd"] = pd.to_numeric(df["cum_usd"], errors="coerce")
    return df


def append_ledger(ledger_path: str, ledger_df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return ledger_df
    new_df = pd.DataFrame(rows)
    new_df["date"] = new_df["date"].astype(str)
    new_df["ticker"] = new_df["ticker"].astype(str).map(norm_sym)
    new_df["delta_usd"] = pd.to_numeric(new_df["delta_usd"], errors="coerce").fillna(0.0)

    df = pd.concat([ledger_df, new_df], ignore_index=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["cum_usd"] = df.groupby("ticker")["delta_usd"].cumsum()

    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ledger_path, index=False)
    return df


def connect_ib(host: str, port: int, client_id: int) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib


def get_account_equity(ib: IB) -> float:
    """
    Uses NetLiquidation as deployment base for 'current_equity'.
    """
    # ib.accountSummary() returns list of AccountValue items
    summ = ib.accountSummary()
    for av in summ:
        if av.tag == "NetLiquidation" and av.currency == "USD":
            v = safe_float(av.value, np.nan)
            if np.isfinite(v) and v > 0:
                return float(v)
    # fallback: first NetLiquidation regardless currency
    for av in summ:
        if av.tag == "NetLiquidation":
            v = safe_float(av.value, np.nan)
            if np.isfinite(v) and v > 0:
                return float(v)
    raise RuntimeError("Could not read NetLiquidation from IB accountSummary().")


def usd_to_shares_floor(abs_usd: float, px: float) -> int:
    """
    Convert USD notional to shares, rounding DOWN in absolute value (conservative).
    """
    if px <= 0 or (not np.isfinite(px)) or (not np.isfinite(abs_usd)):
        return 0
    return int(math.floor(abs_usd / px))

from ib_insync import Stock, MarketOrder  # add MarketOrder import at top

def execute_sell(ib: IB, sym: str, qty: int, order_ref: str) -> None:
    """
    Place a simple market SELL (increase short). Uses ib_insync MarketOrder.
    """
    if qty <= 0:
        return

    sym = norm_sym(sym)
    c = Stock(sym, "SMART", "USD")
    ib.qualifyContracts(c)

    o = MarketOrder("SELL", int(qty))
    o.orderRef = str(order_ref)

    tr = ib.placeOrder(c, o)

    # Give IB a moment to accept/submit (optional)
    ib.sleep(0.5)

def compute_daily_flow_allocations_usd(cfg: dict, equity_usd: float) -> pd.DataFrame:
    flow_cfg = cfg.get("portfolio", {}).get("sleeves", {}).get("flow_program", {}) or {}

    freq = str(flow_cfg.get("frequency", "W")).upper().strip()
    if freq != "D":
        raise ValueError(f"flow_program.frequency must be 'D' for this script. Got: {freq}")

    deployment_base = str(flow_cfg.get("deployment_base", "current_equity")).lower().strip()

    # --- Determine daily_budget ---
    if deployment_base in ("fixed_usd_per_day", "fixed"):
        daily_budget = float(flow_cfg.get("fixed_usd_per_day", 0.0) or 0.0)
        if daily_budget <= 0:
            return pd.DataFrame(columns=["ticker", "delta_usd"])
    elif deployment_base == "current_equity":
        annual_rate = float(flow_cfg.get("annual_deployment_rate", 0.0) or 0.0)
        if annual_rate <= 0:
            return pd.DataFrame(columns=["ticker", "delta_usd"])
        daily_budget = float(equity_usd) * float(annual_rate) / float(TRADING_DAYS)
    else:
        raise ValueError(
            f"Unsupported flow_program.deployment_base={deployment_base}. "
            f"Use 'fixed_usd_per_day' or 'current_equity'."
        )

    # --- Universe ---
    univ = (flow_cfg.get("universe", {}) or {}).get("shorts", []) or []
    tickers = [norm_sym(x) for x in univ if str(x).strip()]
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame(columns=["ticker", "delta_usd"])

    # --- Weights ---
    wcfg = (flow_cfg.get("weighting", {}) or {})
    method = str(wcfg.get("method", "fixed")).lower().strip()
    if method != "fixed":
        raise ValueError(f"Only weighting.method='fixed' supported. Got: {method}")

    weights_raw = (wcfg.get("weights", {}) or {})
    weights = {norm_sym(k): float(v) for k, v in weights_raw.items() if norm_sym(k) in tickers}

    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)

    normalize = bool(wcfg.get("normalize", True))
    if normalize:
        s = float(np.sum(np.abs(w)))
        if s <= 0:
            raise ValueError("flow_program.weighting.normalize=true but weights sum to 0.")
        w = w / s

    out = pd.DataFrame({"ticker": tickers, "weight": w})
    out["delta_usd"] = out["weight"] * float(daily_budget)

    # Convention: positive delta_usd => add more short USD
    out = out[out["delta_usd"].abs() > 1e-6].reset_index(drop=True)
    return out[["ticker", "delta_usd"]]


def main() -> None:
    signal.signal(signal.SIGINT, lambda *_: tprint("[SIGINT] Requested stop (Ctrl+C)."))

    CONFIG_YML = Path("config/strategy_config.yml")
    if not CONFIG_YML.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_YML}")

    cfg = yaml.safe_load(CONFIG_YML.read_text(encoding="utf-8")) or {}
    ibkr_cfg = cfg.get("ibkr", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}
    strat_cfg = cfg.get("strategy", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}

    host = str(ibkr_cfg.get("host", "127.0.0.1"))
    port = int(ibkr_cfg.get("port", 7497))
    client_id = int(ibkr_cfg.get("client_id", 77))  # pick a different id than your main runner
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))

    strategy_tag = str(strat_cfg.get("tag", "")).strip() or "FLOW"
    dry_run = bool(exec_cfg.get("dry_run", False))

    ledger_path = str(paths_cfg.get("flow_ledger_csv", "data/flow_ledger.csv"))

    tprint("\n" + "=" * 110)
    tprint("[FLOW_DAILY] Automatic daily flow sleeve runner")
    tprint("=" * 110 + "\n")
    tprint(f"[FLOW_DAILY] dry_run={dry_run} ledger={ledger_path}")

    ib = connect_ib(host, port, client_id)

    try:
        equity = get_account_equity(ib)
        tprint(f"[FLOW_DAILY] NetLiquidation (USD): {equity:,.2f}")

        alloc = compute_daily_flow_allocations_usd(cfg, equity_usd=equity)
        if alloc.empty:
            tprint("[FLOW_DAILY] No allocation for today (empty or annual_deployment_rate=0).")
            return

        # Price + share sizing
        prices: dict[str, float] = {}
        orders: list[dict] = []

        for _, r in alloc.iterrows():
            sym = norm_sym(r["ticker"])
            delta_usd = float(r["delta_usd"])

            try:
                px = get_price_fallback_ib(ib, sym, prefer_delayed=prefer_delayed)
                prices[sym] = px
            except Exception as e:
                tprint(f"[FLOW_DAILY] SKIP {sym}: hist price unavailable ({type(e).__name__}: {e})")
                continue


            # positive delta_usd means "add short"
            abs_usd = abs(delta_usd)
            sh = usd_to_shares_floor(abs_usd, px)

            if sh <= 0:
                tprint(f"[FLOW_DAILY] SKIP {sym}: delta_usd={delta_usd:,.0f} too small vs px={px:.2f}")
                continue

            orders.append(
                {
                    "ticker": sym,
                    "delta_usd": float(delta_usd),
                    "px": float(px),
                    "shares": int(sh),
                }
            )

        if not orders:
            tprint("[FLOW_DAILY] No actionable orders after pricing/sizing.")
            return

        # Execute sells
        run_date = today_str()
        ledger_rows: list[dict] = []

        tprint("\n" + "-" * 110)
        tprint("[FLOW_DAILY] Orders")
        tprint("-" * 110)
        for o in orders:
            tprint(f"  {o['ticker']}: SELL {o['shares']} sh  (~${o['delta_usd']:,.0f} @ {o['px']:.2f})")

        for o in orders:
            sym = o["ticker"]
            qty = int(o["shares"])
            delta_usd = float(o["delta_usd"])
            order_ref = f"{strategy_tag}|FLOW_DAILY|{sym}|{run_date}"

            if dry_run:
                tprint(f"[FLOW_DAILY][DRY_RUN] Would SELL {sym} qty={qty} ref={order_ref}")
            else:
                execute_sell(ib, sym, qty, order_ref=order_ref)

            ledger_rows.append({"date": run_date, "ticker": sym, "delta_usd": float(delta_usd)})

        # Update ledger AFTER attempting orders
        try:
            ledger_df = load_ledger(ledger_path)
            append_ledger(ledger_path, ledger_df, ledger_rows)
            tprint(f"\n[FLOW_DAILY] Ledger updated: {ledger_path}")
        except Exception as e:
            tprint(f"\n[FLOW_DAILY] WARNING: ledger update failed ({type(e).__name__}: {e})")

        total = float(sum(o["delta_usd"] for o in orders))
        tprint(f"\n[FLOW_DAILY] Done. Total deployed today (USD): {total:,.2f}")

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
