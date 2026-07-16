#!/usr/bin/env python3
"""
execute_flow_program.py

Flow sleeve allocator with daily or weekly scheduling.

Modes:
  frequency=D  — immediate execution (SELL today, same as before)
  frequency=W  — schedule Adaptive MKT orders via IBKR goodAfterTime
                  for each configured weekday (e.g. Mon + Thu)

Also applies flow kill/flatten when configured:
  rules.kill_enabled + borrow > kill_borrow_floor AND net_edge_p50 < 0
  -> immediate BUY cover of live shorts, skip new adds, optional YAML scrub.

Designed to run independently of generate_trade_plan.py / execute_trade_plan.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from ib_insync import IB, MarketOrder, Stock, Order
from strategy_config import load_config
from execute_trade_plan import configure_ib_error_log_filter


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


def execute_buy(ib: IB, sym: str, qty: int, order_ref: str) -> None:
    """Cover a short (BUY to flatten)."""
    if qty <= 0:
        return
    sym = norm_sym(sym)
    c = Stock(sym, "SMART", "USD")
    ib.qualifyContracts(c)
    o = MarketOrder("BUY", int(qty))
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


def get_ib_short_shares(ib: IB) -> dict[str, int]:
    """Map symbol -> abs(short shares) for current IB short stock positions."""
    out: dict[str, int] = {}
    for p in ib.positions():
        try:
            qty = int(p.position)
        except Exception:
            continue
        if qty >= 0:
            continue
        sym = norm_sym(getattr(p.contract, "symbol", "") or "")
        if not sym:
            continue
        out[sym] = int(out.get(sym, 0) + abs(qty))
    return out


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
    if "carry_usd" not in df.columns:
        df["carry_usd"] = 0.0
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).map(norm_sym)
    if "delta_usd" in df.columns:
        df["delta_usd"] = pd.to_numeric(df["delta_usd"], errors="coerce").fillna(0.0)
    if "cum_usd" in df.columns:
        df["cum_usd"] = pd.to_numeric(df["cum_usd"], errors="coerce")
    return df

def get_carry_map(df):

    if df.empty:
        return {}

    return (
        df.sort_values("date")
        .groupby("ticker")["carry_usd"]
        .last()
        .to_dict()
    )


def ledger_tickers(df: pd.DataFrame) -> Set[str]:
    if df is None or df.empty or "ticker" not in df.columns:
        return set()
    return {norm_sym(x) for x in df["ticker"].tolist() if str(x).strip()}


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
# Kill / flatten (neg edge AND high borrow)
# ---------------------------------------------------------------------------

def load_screened_flow_metrics(screened_csv: str | Path) -> pd.DataFrame:
    """Load screener rows with ETF, net_edge_p50_annual, borrow_current."""
    path = Path(screened_csv)
    if not path.exists():
        return pd.DataFrame(columns=["ETF", "net_edge_p50_annual", "borrow_current"])
    df = pd.read_csv(path)
    if "ETF" not in df.columns:
        return pd.DataFrame(columns=["ETF", "net_edge_p50_annual", "borrow_current"])
    out = pd.DataFrame({"ETF": df["ETF"].astype(str).map(norm_sym)})
    out["net_edge_p50_annual"] = pd.to_numeric(
        df["net_edge_p50_annual"] if "net_edge_p50_annual" in df.columns else np.nan,
        errors="coerce",
    )
    borrow = pd.Series(np.nan, index=df.index, dtype=float)
    if "borrow_current" in df.columns:
        borrow = pd.to_numeric(df["borrow_current"], errors="coerce")
    elif "borrow_fee_annual" in df.columns:
        borrow = pd.to_numeric(df["borrow_fee_annual"], errors="coerce")
    out["borrow_current"] = borrow
    return out.drop_duplicates(subset=["ETF"], keep="last").reset_index(drop=True)


def build_flow_kill_set(
    screened: pd.DataFrame,
    candidates: Iterable[str],
    *,
    borrow_floor: float = 0.40,
    require_neg_edge: bool = True,
) -> tuple[Set[str], pd.DataFrame]:
    """
    Kill when borrow_current > borrow_floor AND (optionally) net_edge_p50_annual < 0.

    Missing borrow or edge => not killed (fail-closed on incomplete data).
    Returns (kill_symbols, detail_df).
    """
    cands = {norm_sym(x) for x in candidates if str(x).strip()}
    if not cands:
        return set(), pd.DataFrame()
    if screened is None or screened.empty:
        return set(), pd.DataFrame()

    sub = screened[screened["ETF"].isin(cands)].copy()
    if sub.empty:
        return set(), pd.DataFrame()

    edge = pd.to_numeric(sub["net_edge_p50_annual"], errors="coerce")
    borrow = pd.to_numeric(sub["borrow_current"], errors="coerce")
    known = edge.notna() & borrow.notna()
    hi_borrow = borrow > float(borrow_floor)
    neg_edge = edge < 0.0
    if require_neg_edge:
        kill_mask = known & hi_borrow & neg_edge
    else:
        kill_mask = known & hi_borrow

    sub["kill"] = kill_mask
    killed = {norm_sym(x) for x in sub.loc[kill_mask, "ETF"].tolist()}
    return killed, sub.sort_values("borrow_current", ascending=False).reset_index(drop=True)


def load_flow_alumni(path: str | Path, seeds: Iterable[str] | None = None) -> Set[str]:
    p = Path(path)
    out: Set[str] = set()
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                out |= {norm_sym(x) for x in raw if str(x).strip()}
            elif isinstance(raw, dict) and isinstance(raw.get("symbols"), list):
                out |= {norm_sym(x) for x in raw["symbols"] if str(x).strip()}
        except Exception:
            pass
    if seeds:
        out |= {norm_sym(x) for x in seeds if str(x).strip()}
    return out


def save_flow_alumni(path: str | Path, symbols: Iterable[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    syms = sorted({norm_sym(x) for x in symbols if str(x).strip()})
    p.write_text(json.dumps({"symbols": syms}, indent=2) + "\n", encoding="utf-8")


def scrub_flow_symbols_from_config(
    config_path: str | Path,
    symbols: Iterable[str],
) -> list[str]:
    """
    Remove symbols from flow_program universe.shorts and weighting.weights
    in strategy_config.yml while preserving surrounding comments/formatting.
    """
    path = Path(config_path)
    syms = {norm_sym(x) for x in symbols if str(x).strip()}
    if not syms or not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    in_flow = False
    in_shorts = False
    in_weights = False
    removed: list[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped == "flow_program:":
            in_flow = True
            in_shorts = False
            in_weights = False
            out.append(line)
            continue

        if in_flow and (
            stripped.startswith("# SPX")
            or (
                bool(line)
                and line[0] not in (" ", "	")
                and stripped.endswith(":")
                and not stripped.startswith("#")
            )
        ):
            in_flow = False
            in_shorts = False
            in_weights = False

        if in_flow and stripped == "shorts:":
            in_shorts = True
            in_weights = False
            out.append(line)
            continue

        if in_flow and stripped == "weights:":
            in_weights = True
            in_shorts = False
            out.append(line)
            continue

        if in_flow and in_shorts:
            if stripped.startswith("- "):
                tok = norm_sym(stripped[2:].split("#", 1)[0].strip())
                if tok in syms:
                    removed.append(tok)
                    continue
            elif stripped.endswith(":") and stripped != "shorts:":
                in_shorts = False

        if in_flow and in_weights and stripped and not stripped.startswith("#") and ":" in stripped:
            key = norm_sym(stripped.split(":", 1)[0].strip())
            if key in syms:
                removed.append(key)
                continue

        out.append(line)

    if removed:
        path.write_text("".join(out), encoding="utf-8")
    return sorted(set(removed))


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
    parser.add_argument(
        "--config",
        default="config/strategy_config.yml",
        help="Path to strategy_config.yml",
    )
    args = parser.parse_args()

    config_path = str(args.config)
    cfg = load_config(config_path)
    ibkr_cfg = cfg.get("ibkr", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}
    strat_cfg = cfg.get("strategy", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    flow_cfg = ((cfg.get("portfolio") or {}).get("sleeves") or {}).get("flow_program") or {}
    rules = flow_cfg.get("rules") or {}

    host = str(ibkr_cfg.get("host", "127.0.0.1"))
    port = int(ibkr_cfg.get("port", 7497))
    client_id = int(ibkr_cfg.get("client_id", 77))
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))
    suppress_error_codes = [int(c) for c in ((ibkr_cfg.get("suppress_error_codes", [10089])) or [])]
    configure_ib_error_log_filter(suppress_error_codes)
    if suppress_error_codes:
        tprint(f"[FLOW][IB] Suppressing noisy API error codes: {sorted(set(suppress_error_codes))}")

    strategy_tag = str(strat_cfg.get("tag", "")).strip() or "FLOW"
    dry_run = args.dry_run or bool(exec_cfg.get("dry_run", False))

    ledger_path = str(paths_cfg.get("flow_ledger_csv", "data/flow_ledger.csv"))
    screened_csv = str(paths_cfg.get("screened_csv", "data/etf_screened_today.csv"))

    hard_borrow_cap = float(rules.get("hard_borrow_cap", 0.30))
    kill_enabled = bool(rules.get("kill_enabled", True))
    kill_borrow_floor = float(rules.get("kill_borrow_floor", 0.40))
    kill_require_neg_edge = bool(rules.get("kill_require_neg_edge", True))
    kill_scrub_config = bool(rules.get("kill_scrub_config", True))
    alumni_json = str(rules.get("alumni_json", "data/flow_alumni.json"))
    seed_alumni = list(rules.get("seed_alumni") or [])

    tprint("\n" + "=" * 110)
    tprint("[FLOW] Flow sleeve runner")
    tprint("=" * 110 + "\n")
    tprint(f"[FLOW] dry_run={dry_run} ledger={ledger_path}")
    tprint(
        f"[FLOW] add gate: borrow<={hard_borrow_cap:.0%} and net_edge_p50>0 | "
        f"kill: enabled={kill_enabled} borrow>{kill_borrow_floor:.0%} "
        f"neg_edge={kill_require_neg_edge} scrub_config={kill_scrub_config}"
    )

    ib = connect_ib(host, port, client_id)

    try:
        equity = get_account_equity(ib)
        tprint(f"[FLOW] NetLiquidation (USD): {equity:,.2f}")

        ledger_df = load_ledger(ledger_path)
        universe, _ = _parse_universe_and_weights(flow_cfg)
        universe_set = set(universe)

        # ---------------------------------------------------------------
        # Kill / flatten: neg edge AND borrow above floor
        # Candidates = config universe U ledger history (catches orphans)
        # ---------------------------------------------------------------
        kill_set: set[str] = set()
        screened = load_screened_flow_metrics(screened_csv)
        short_shares = get_ib_short_shares(ib)

        if kill_enabled:
            alumni = load_flow_alumni(alumni_json, seeds=seed_alumni)
            alumni |= set(universe_set) | ledger_tickers(ledger_df)
            save_flow_alumni(alumni_json, alumni)
            # Config universe + ledger + alumni (orphans after YAML removal).
            candidates = set(alumni)
            kill_set, kill_detail = build_flow_kill_set(
                screened,
                candidates,
                borrow_floor=kill_borrow_floor,
                require_neg_edge=kill_require_neg_edge,
            )
            if kill_detail is not None and not kill_detail.empty:
                for _, row in kill_detail.iterrows():
                    flag = "KILL" if bool(row.get("kill")) else "ok"
                    edge = row.get("net_edge_p50_annual")
                    br = row.get("borrow_current")
                    tprint(
                        f"[FLOW][SCAN] {row['ETF']}: edge={edge} borrow={br} -> {flag}"
                    )
            if kill_set:
                tprint(f"[FLOW][KILL] Flatten candidates: {', '.join(sorted(kill_set))}")
            else:
                tprint("[FLOW][KILL] No names meet kill criteria.")

            # Immediate BUY covers for live shorts in kill_set
            flatten_rows: list[dict] = []
            for sym in sorted(kill_set):
                qty = int(short_shares.get(sym, 0) or 0)
                if qty <= 0:
                    tprint(f"[FLOW][KILL] {sym}: no IB short to cover")
                    continue
                tprint(f"[FLOW][KILL] COVER {sym}: BUY {qty} sh (flatten)")
                if not dry_run:
                    try:
                        execute_buy(
                            ib,
                            sym,
                            qty,
                            order_ref=f"{strategy_tag}|FLOW_KILL|{sym}|{today_str()}",
                        )
                    except Exception as e:
                        tprint(f"[FLOW][KILL] ORDER FAIL {sym}: {type(e).__name__}: {e}")
                        continue
                flatten_rows.append(
                    {
                        "date": today_str(),
                        "ticker": sym,
                        "delta_usd": 0.0,
                        "carry_usd": 0.0,
                    }
                )
            if flatten_rows:
                ledger_df = append_ledger(ledger_path, ledger_df, flatten_rows)
                tprint(f"[FLOW][KILL] Ledger carry zeroed for {len(flatten_rows)} name(s)")

            if kill_set and kill_scrub_config and not dry_run:
                removed = scrub_flow_symbols_from_config(config_path, kill_set)
                if removed:
                    tprint(
                        f"[FLOW][KILL] Scrubbed from config: {', '.join(removed)}"
                    )
                else:
                    tprint(
                        "[FLOW][KILL] Config scrub: nothing to remove "
                        "(already absent from universe/weights)"
                    )
            elif kill_set and kill_scrub_config and dry_run:
                tprint(
                    f"[FLOW][KILL] dry-run: would scrub from config: "
                    f"{', '.join(sorted(kill_set))}"
                )

        # ---------------------------------------------------------------
        # Adds: gated by hard_borrow_cap + positive edge; skip kill names
        # ---------------------------------------------------------------
        alloc, freq, schedule_days, schedule_time = compute_flow_allocations_usd(cfg, equity)
        if alloc.empty:
            tprint("[FLOW] No allocation (empty config or zero budget).")
            return

        metrics = screened.set_index("ETF") if not screened.empty else pd.DataFrame()
        kept_rows = []
        for _, r in alloc.iterrows():
            sym = norm_sym(r["ticker"])
            if sym in kill_set:
                tprint(f"[FLOW] SKIP ADD {sym}: on kill list")
                continue
            edge = br = float("nan")
            if not metrics.empty and sym in metrics.index:
                edge = float(pd.to_numeric(metrics.loc[sym, "net_edge_p50_annual"], errors="coerce"))
                br = float(pd.to_numeric(metrics.loc[sym, "borrow_current"], errors="coerce"))
            if not (edge == edge) or edge <= 0:
                tprint(f"[FLOW] SKIP ADD {sym}: net_edge_p50={edge} (need > 0)")
                continue
            if not (br == br) or br > hard_borrow_cap:
                tprint(
                    f"[FLOW] SKIP ADD {sym}: borrow={br} "
                    f"(need known and <= {hard_borrow_cap:.0%})"
                )
                continue
            kept_rows.append(r)

        if not kept_rows:
            tprint("[FLOW] No eligible adds after kill/edge/borrow gates.")
            return

        alloc = pd.DataFrame(kept_rows).reset_index(drop=True)

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

        if freq == "W":
            schedule = next_schedule_dates(schedule_days, schedule_time)
            tprint(f"[FLOW] Weekly mode: scheduling {len(schedule)} days")
            for date_str, gat_str in schedule:
                tprint(f"  -> {date_str}  GAT={gat_str}")
        else:
            schedule = [(today_str(), "")]

        per_day_total = float(sum(o["delta_usd"] for o in sized))
        tprint(f"[FLOW] Per-day budget: ${per_day_total:,.2f}  x {len(schedule)} day(s)")

        ledger_rows: list[dict] = []
        carry_map = get_carry_map(load_ledger(ledger_path))

        for target_date, gat_str in schedule:
            tprint(f"\n{'─'*110}")
            label = f"GAT={gat_str}" if gat_str else "IMMEDIATE"
            tprint(f"[FLOW] {target_date}  ({label})")
            tprint(f"{'─'*110}")

            for o in alloc.itertuples(index=False):
                sym = norm_sym(o.ticker)
                delta_usd = float(o.delta_usd)
                prior_carry = carry_map.get(sym, 0.0)
                effective_usd = delta_usd + prior_carry

                if sym not in prices:
                    try:
                        px = get_price_fallback_ib(ib, sym, prefer_delayed=prefer_delayed)
                        prices[sym] = px
                    except Exception as e:
                        tprint(f"[FLOW] SKIP {sym}: hist price unavailable ({type(e).__name__}: {e})")
                        continue
                else:
                    px = prices[sym]

                shares = usd_to_shares_floor(effective_usd, px)
                trade_usd = shares * px
                carry_usd = effective_usd - trade_usd

                if shares == 0:
                    tprint(f"[FLOW] CARRY {sym}: +${delta_usd:.0f} (total carry=${effective_usd:.0f})")
                    ledger_rows.append({
                        "date": target_date,
                        "ticker": sym,
                        "delta_usd": 0.0,
                        "carry_usd": effective_usd,
                    })
                    carry_map[sym] = effective_usd
                    continue

                tprint(f"{sym}: SELL {shares} sh (~${trade_usd:,.0f} @ {px:.2f})")

                if not dry_run:
                    try:
                        contract = Stock(sym, "SMART", "USD")
                        ib.qualifyContracts(contract)
                        order = Order(
                            action="SELL",
                            totalQuantity=shares,
                            orderType="MKT",
                            tif="DAY",
                            goodAfterTime=gat_str if gat_str else "",
                            orderRef=f"{strategy_tag}|FLOW|{sym}|{gat_str}",
                        )
                        ib.placeOrder(contract, order)
                    except Exception as e:
                        tprint(f"  ORDER FAIL {sym}: {type(e).__name__}: {e}")
                        continue

                ledger_rows.append({
                    "date": target_date,
                    "ticker": sym,
                    "delta_usd": trade_usd,
                    "carry_usd": carry_usd,
                })
                carry_map[sym] = carry_usd

        ledger_df = load_ledger(ledger_path)
        append_ledger(ledger_path, ledger_df, ledger_rows)
        tprint(f"\n[FLOW] Ledger updated: {ledger_path}")
        grand_total = sum(o["delta_usd"] for o in sized) * len(schedule)
        tprint(f"\n[FLOW] Done. Total scheduled (USD): ${grand_total:,.2f} across {len(schedule)} day(s).")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
