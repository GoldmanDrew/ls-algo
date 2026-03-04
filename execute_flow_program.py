#!/usr/bin/env python3
"""
execute_flow_program.py

Flow sleeve allocator with daily or weekly scheduling.

Modes:
  frequency=D  — immediate execution (SELL today, same as before)
  frequency=W  — schedule Adaptive MKT orders via IBKR goodAfterTime
                  for each configured weekday (e.g. Mon + Thu)

Designed to run independently of generate_trade_plan.py / execute_trade_plan.py.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from zoneinfo import ZoneInfo

from ib_insync import IB, MarketOrder, Stock, Order


TRADING_DAYS = 252
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def tprint(msg: str) -> None:
    print(msg, flush=True)


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def norm_sym(x: str) -> str:
    return str(x).upper().replace(".", "-")


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_float(x):
    try:
        x = float(x)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def usd_to_shares_floor(abs_usd: float, px: float) -> int:
    if px <= 0 or (not np.isfinite(px)) or (not np.isfinite(abs_usd)):
        return 0
    return int(math.floor(abs_usd / px))


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def place_adaptive_mkt(
    ib: IB,
    sym: str,
    *,
    action: str,
    qty: int,
    order_ref: str,
    priority: str = "Patient",
    good_after_time: str | None = None,
):
    """
    Places Adaptive Market order.  If good_after_time is set (IBKR format
    "YYYYMMDD HH:MM:SS" in exchange tz), the order activates at that time.
    """
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

    if good_after_time:
        o.goodAfterTime = good_after_time

    return ib.placeOrder(c, o)


def execute_sell(ib: IB, sym: str, qty: int, order_ref: str) -> None:
    if qty <= 0:
        return
    sym = norm_sym(sym)
    c = Stock(sym, "SMART", "USD")
    ib.qualifyContracts(c)
    o = MarketOrder("SELL", int(qty))
    o.orderRef = str(order_ref)
    ib.placeOrder(c, o)
    ib.sleep(0.5)


# ---------------------------------------------------------------------------
# IB connection + account
# ---------------------------------------------------------------------------

def connect_ib(host: str, port: int, client_id: int) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib


def get_account_equity(ib: IB) -> float:
    summ = ib.accountSummary()
    for av in summ:
        if av.tag == "NetLiquidation" and av.currency == "USD":
            v = safe_float(av.value, np.nan)
            if np.isfinite(v) and v > 0:
                return float(v)
    for av in summ:
        if av.tag == "NetLiquidation":
            v = safe_float(av.value, np.nan)
            if np.isfinite(v) and v > 0:
                return float(v)
    raise RuntimeError("Could not read NetLiquidation from IB accountSummary().")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Weekly scheduling helpers
# ---------------------------------------------------------------------------

def next_schedule_dates(
    schedule_days: list[int],
    schedule_time_et: str,
) -> list[tuple[str, str]]:
    """
    Compute the next occurrence of each scheduled weekday from today.

    Args:
        schedule_days: list of Python weekday ints (0=Mon .. 6=Sun)
        schedule_time_et: time string "HH:MM:SS" in US/Eastern

    Returns:
        List of (date_str "YYYY-MM-DD", gat_str "YYYYMMDD HH:MM:SS") tuples,
        sorted by date.  Only includes future dates (skips if already past).
    """
    now_et = datetime.now(ET)
    h, m, s = (int(x) for x in schedule_time_et.split(":"))

    results: list[tuple[str, str]] = []
    for wd in schedule_days:
        # Days until next occurrence of this weekday
        diff = (wd - now_et.weekday()) % 7
        if diff == 0:
            # Today is the target weekday — include only if time hasn't passed
            target = now_et.replace(hour=h, minute=m, second=s, microsecond=0)
            if target <= now_et:
                diff = 7  # already past, schedule next week
        target_date = (now_et + timedelta(days=diff)).date()
        date_str = target_date.strftime("%Y-%m-%d")
        gat_str = target_date.strftime("%Y%m%d") + f" {schedule_time_et}"
        results.append((date_str, gat_str))

    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Allocation computation
# ---------------------------------------------------------------------------

def _parse_universe_and_weights(flow_cfg: dict) -> tuple[list[str], np.ndarray]:
    """Parse tickers and normalized weights from flow config."""
    univ = (flow_cfg.get("universe", {}) or {}).get("shorts", []) or []
    tickers = sorted({norm_sym(x) for x in univ if str(x).strip()})
    if not tickers:
        return [], np.array([])

    wcfg = (flow_cfg.get("weighting", {}) or {})
    method = str(wcfg.get("method", "fixed")).lower().strip()
    if method != "fixed":
        raise ValueError(f"Only weighting.method='fixed' supported. Got: {method}")

    weights_raw = (wcfg.get("weights", {}) or {})
    weights = {norm_sym(k): float(v) for k, v in weights_raw.items() if norm_sym(k) in tickers}
    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)

    if bool(wcfg.get("normalize", True)):
        s = float(np.sum(np.abs(w)))
        if s <= 0:
            raise ValueError("flow_program.weighting.normalize=true but weights sum to 0.")
        w = w / s

    return tickers, w


def compute_flow_allocations_usd(
    cfg: dict,
    equity_usd: float,
) -> tuple[pd.DataFrame, str, list[int], str]:
    """
    Compute per-day flow allocations from config.

    Returns:
        (alloc_df, frequency, schedule_days, schedule_time_et)

    alloc_df has columns: ticker, delta_usd  (per-day budget)
    """
    flow_cfg = cfg.get("portfolio", {}).get("sleeves", {}).get("flow_program", {}) or {}
    freq = str(flow_cfg.get("frequency", "D")).upper().strip()
    deployment_base = str(flow_cfg.get("deployment_base", "current_equity")).lower().strip()

    # Determine per-day budget
    if freq == "D":
        if deployment_base in ("fixed_usd_per_day", "fixed"):
            per_day = float(flow_cfg.get("fixed_usd_per_day", 0.0) or 0.0)
        elif deployment_base == "current_equity":
            annual_rate = float(flow_cfg.get("annual_deployment_rate", 0.0) or 0.0)
            per_day = float(equity_usd) * annual_rate / float(TRADING_DAYS)
        else:
            raise ValueError(f"Unsupported deployment_base={deployment_base} for freq=D")
        schedule_days = []
        schedule_time = ""

    elif freq == "W":
        schedule_days = list(flow_cfg.get("schedule_days", [0, 3]))
        schedule_time = str(flow_cfg.get("schedule_time_et", "10:00:00"))
        n_days = len(schedule_days)
        if n_days <= 0:
            raise ValueError("flow_program.schedule_days must be non-empty for freq=W")

        if deployment_base in ("fixed_usd_per_week", "fixed"):
            weekly = float(flow_cfg.get("fixed_usd_per_week", 0.0) or 0.0)
            per_day = weekly / n_days
        elif deployment_base == "current_equity":
            annual_rate = float(flow_cfg.get("annual_deployment_rate", 0.0) or 0.0)
            weekly = float(equity_usd) * annual_rate / 52.0
            per_day = weekly / n_days
        else:
            raise ValueError(f"Unsupported deployment_base={deployment_base} for freq=W")
    else:
        raise ValueError(f"Unsupported flow_program.frequency={freq}. Use 'D' or 'W'.")

    if per_day <= 0:
        empty = pd.DataFrame(columns=["ticker", "delta_usd"])
        return empty, freq, schedule_days, schedule_time

    tickers, w = _parse_universe_and_weights(flow_cfg)
    if not tickers:
        empty = pd.DataFrame(columns=["ticker", "delta_usd"])
        return empty, freq, schedule_days, schedule_time

    out = pd.DataFrame({"ticker": tickers, "weight": w})
    out["delta_usd"] = out["weight"] * per_day
    out = out[out["delta_usd"].abs() > 1e-6].reset_index(drop=True)
    return out[["ticker", "delta_usd"]], freq, schedule_days, schedule_time


# Keep old name for backward compat (used by other modules)
def compute_daily_flow_allocations_usd(cfg: dict, equity_usd: float) -> pd.DataFrame:
    alloc, freq, _, _ = compute_flow_allocations_usd(cfg, equity_usd)
    return alloc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    signal.signal(signal.SIGINT, lambda *_: tprint("[SIGINT] Requested stop (Ctrl+C)."))

    parser = argparse.ArgumentParser(description="Flow sleeve allocator (daily or weekly GAT).")
    parser.add_argument("--dry-run", action="store_true", help="No orders placed.")
    args = parser.parse_args()

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
    client_id = int(ibkr_cfg.get("client_id", 77))
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))

    strategy_tag = str(strat_cfg.get("tag", "")).strip() or "FLOW"
    dry_run = args.dry_run or bool(exec_cfg.get("dry_run", False))

    ledger_path = str(paths_cfg.get("flow_ledger_csv", "data/flow_ledger.csv"))

    tprint("\n" + "=" * 110)
    tprint("[FLOW] Flow sleeve runner")
    tprint("=" * 110 + "\n")
    tprint(f"[FLOW] dry_run={dry_run} ledger={ledger_path}")

    ib = connect_ib(host, port, client_id)

    try:
        equity = get_account_equity(ib)
        tprint(f"[FLOW] NetLiquidation (USD): {equity:,.2f}")

        alloc, freq, schedule_days, schedule_time = compute_flow_allocations_usd(cfg, equity)
        if alloc.empty:
            tprint("[FLOW] No allocation (empty config or zero budget).")
            return

        # Price + share sizing (done once, reused for all scheduled days)
        prices: dict[str, float] = {}
        sized: list[dict] = []

        for _, r in alloc.iterrows():
            sym = norm_sym(r["ticker"])
            delta_usd = float(r["delta_usd"])

            try:
                px = get_price_fallback_ib(ib, sym, prefer_delayed=prefer_delayed)
                prices[sym] = px
            except Exception as e:
                tprint(f"[FLOW] SKIP {sym}: hist price unavailable ({type(e).__name__}: {e})")
                continue

            sh = usd_to_shares_floor(abs(delta_usd), px)
            if sh <= 0:
                tprint(f"[FLOW] SKIP {sym}: delta_usd={delta_usd:,.0f} too small vs px={px:.2f}")
                continue

            sized.append({
                "ticker": sym,
                "delta_usd": float(delta_usd),
                "px": float(px),
                "shares": int(sh),
            })

        if not sized:
            tprint("[FLOW] No actionable orders after pricing/sizing.")
            return

        # ---------------------------------------------------------------
        # Determine execution schedule
        # ---------------------------------------------------------------
        if freq == "W":
            schedule = next_schedule_dates(schedule_days, schedule_time)
            tprint(f"[FLOW] Weekly mode: scheduling {len(schedule)} days")
            for date_str, gat_str in schedule:
                tprint(f"  -> {date_str}  GAT={gat_str}")
        else:
            # Daily: single immediate execution (no GAT)
            schedule = [(today_str(), "")]

        per_day_total = float(sum(o["delta_usd"] for o in sized))
        tprint(f"[FLOW] Per-day budget: ${per_day_total:,.2f}  x {len(schedule)} day(s)")

        # ---------------------------------------------------------------
        # Submit orders for each scheduled day
        # ---------------------------------------------------------------
        ledger_rows: list[dict] = []

        for target_date, gat_str in schedule:
            tprint(f"\n{'─'*110}")
            label = f"GAT={gat_str}" if gat_str else "IMMEDIATE"
            tprint(f"[FLOW] {target_date}  ({label})")
            tprint(f"{'─'*110}")

            for o in sized:
                sym = o["ticker"]
                qty = int(o["shares"])
                delta_usd = float(o["delta_usd"])
                px = float(o["px"])
                order_ref = f"{strategy_tag}|FLOW|{sym}|{target_date}"

                tprint(f"  {sym}: SELL {qty} sh  (~${delta_usd:,.0f} @ {px:.2f})")

                if dry_run:
                    tprint(f"  [DRY_RUN] Would SELL {sym} qty={qty} ref={order_ref}")
                else:
                    try:
                        place_adaptive_mkt(
                            ib, sym,
                            action="SELL",
                            qty=qty,
                            order_ref=order_ref,
                            priority="Patient",
                            good_after_time=gat_str or None,
                        )
                    except Exception as e:
                        tprint(f"  ORDER FAIL {sym}: {type(e).__name__}: {e}")
                        continue

                ledger_rows.append({
                    "date": target_date,
                    "ticker": sym,
                    "delta_usd": float(delta_usd),
                })

        # Update ledger
        try:
            ledger_df = load_ledger(ledger_path)
            append_ledger(ledger_path, ledger_df, ledger_rows)
            tprint(f"\n[FLOW] Ledger updated: {ledger_path}")
        except Exception as e:
            tprint(f"\n[FLOW] WARNING: ledger update failed ({type(e).__name__}: {e})")

        grand_total = per_day_total * len(schedule)
        tprint(f"\n[FLOW] Done. Total scheduled (USD): ${grand_total:,.2f} across {len(schedule)} day(s).")

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
