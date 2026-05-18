#!/usr/bin/env python3
"""
ibkr_accounting.py

Rebuild an accounting-grade PnL report from IBKR Flex XML exports,
plus beta-normalized net exposure by underlying.

UPDATED:
- Borrow fees from flex_borrow_fee_details.xml are now CUMULATIVE (all dates up
  to run_date) to match the YTD scope of FIFO PnL and cash transactions.
  Previously filtered to a single day, which understated borrow by ~100%.
- Universe filter simplified: any symbol whose resolved underlying is in
  allowed_underlyings is included (catches spot, ETF, and closed positions).
- Bond interest (e.g. bond coupons) is now categorized and included in PnL.
- Robust parsing: FIFO performance summary node varies by Flex query configuration.
  We try:
    1) FIFOPerformanceSummaryInBase
    2) FIFOPerformanceSummary (convert via fxRateToBase when available)
  If neither is present, trading PnL is set to 0 and the script still runs.
- Universe + ETF->Underlying mapping comes from: data/etf_screened_today.csv
  (NOT config/etf_cagr.csv)
- Net exposure by underlying (beta-normalized) is now computed here and
  output alongside PnL.  run_eod_pnl_email.py imports these functions.

PnL vectors per symbol:
  realized_pnl           FIFO performance summary (YTD cumulative)
  unrealized_pnl         FIFO performance summary (YTD cumulative)
  dividends              Cash transactions
  withholding_tax        Cash transactions
  pil_dividends          Cash transactions (Payment In Lieu - short positions)
  borrow_fees            Borrow Fee Details (cumulative) or cash fallback
  short_credit_interest  Cash transactions (allocated pro-rata to shorts)
  other_fees             Cash transactions
  bond_interest          Cash transactions (bond coupon payments)

Outputs (written to: data/runs/<RUN_DATE>/accounting/):
- pnl_by_symbol.csv
- pnl_by_underlying.csv (buckets 1+2+4 combined)
- pnl_by_underlying_b12.csv (legacy buckets 1+2 only)
- pnl_bucket_1.csv … pnl_bucket_4.csv
- pnl_bucket_4_by_pair.csv, pnl_bucket_4_by_symbol.csv
- pnl_by_pair.csv
- bucket4_pairs.csv
- net_exposure_by_underlying.csv (buckets 1+2+4)
- net_exposure_bucket_1.csv … net_exposure_bucket_4.csv
- net_exposure_bucket_4_detail.csv
- totals.json

Run:
  python ibkr_accounting.py YYYY-MM-DD
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / normalization  (safe to import — no side-effects)
# ──────────────────────────────────────────────────────────────────────────────
def canonical_symbol(sym: str) -> str:
    if sym is None:
        return ""
    s = str(sym).strip().upper()
    m = re.match(r"^([A-Z]{1,5})[ \-\.]([A-Z])$", s)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return s


def to_base(amount: float | str, fx: float | str) -> float:
    try:
        return float(amount) * float(fx)
    except Exception:
        try:
            return float(amount)
        except Exception:
            return 0.0


def yyyymmdd_from_run_date(run_date: str) -> str:
    return run_date.replace("-", "")


def yyyymmdd_normalize(date_str: str) -> str:
    """Normalize YYYY-MM-DD or YYYYMMDD to an 8-digit YYYYMMDD string."""
    return str(date_str or "").replace("-", "")[:8]


def find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        if (cur / "data").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not locate project root containing /data")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())

# Hard exclusion
EXCLUDE_SYMBOLS = {canonical_symbol("BRK.B"), canonical_symbol("BRKB")}


def _normalize_bucket_pair(b1: float, b2: float) -> tuple[float, float]:
    """Return a stable (b1,b2) pair with b1+b2=1."""
    try:
        x1 = float(b1)
    except Exception:
        x1 = 0.0
    try:
        x2 = float(b2)
    except Exception:
        x2 = 0.0
    x1 = max(0.0, x1)
    x2 = max(0.0, x2)
    s = x1 + x2
    if s <= 0:
        return 1.0, 0.0
    return x1 / s, x2 / s


def _normalize_bucket_triple(b1: float, b2: float, b4: float) -> tuple[float, float, float]:
    """Return stable (b1,b2,b4) with b1+b2+b4=1."""
    try:
        x1 = float(b1)
    except Exception:
        x1 = 0.0
    try:
        x2 = float(b2)
    except Exception:
        x2 = 0.0
    try:
        x4 = float(b4)
    except Exception:
        x4 = 0.0
    x1 = max(0.0, x1)
    x2 = max(0.0, x2)
    x4 = max(0.0, x4)
    s = x1 + x2 + x4
    if s <= 0:
        return 1.0, 0.0, 0.0
    return x1 / s, x2 / s, x4 / s


def _apply_signed_bucket_trade(
    qty: float,
    cost: float,
    dq: float,
    px: float,
) -> tuple[float, float, float]:
    """
    Update signed bucket position (qty shares, total cost basis).
    dq is signed shares added by the trade at px.
    Returns (new_qty, new_cost, realized_pnl).
    """
    if abs(dq) <= 1e-12:
        return qty, cost, 0.0
    if abs(qty) <= 1e-12:
        return dq, dq * px, 0.0
    if qty * dq > 0:
        return qty + dq, cost + dq * px, 0.0

    avg = cost / qty
    closed_shares = min(abs(dq), abs(qty))
    sign = 1.0 if qty > 0 else -1.0
    realized = closed_shares * (px - avg) * sign
    qty_after = qty + dq
    if abs(qty_after) <= 1e-12:
        return 0.0, 0.0, realized
    if qty * qty_after > 0:
        cost_after = cost * (qty_after / qty)
        return qty_after, cost_after, realized
    remainder = qty_after
    return remainder, remainder * px, realized


# Supplemental ETF→Underlying mappings for securities that were traded as part
# of the algo but are no longer in etf_screened_today.csv (e.g. closed positions
# from rotations).  These get merged into the CSV-based map so their realized
# PnL rolls up under the correct underlying.
SUPPLEMENTAL_ETF_MAP: dict[str, str] = {
    canonical_symbol("XRP"):  canonical_symbol("XRPZ"),
    canonical_symbol("GXRP"): canonical_symbol("XRPZ"),
    # Tradr ETFs — delisted/liquidated Jan–Mar 2026.
    # Kept here so PnL history (realized, borrow, etc.) continues
    # to map correctly even after the screened CSV drops them.
    canonical_symbol("AURU"): canonical_symbol("AUR"),
    canonical_symbol("AXUP"): canonical_symbol("AXON"),
    canonical_symbol("BKNU"): canonical_symbol("BKNG"),
    canonical_symbol("BLSX"): canonical_symbol("BLSH"),
    canonical_symbol("CELT"): canonical_symbol("CELH"),
    canonical_symbol("DASX"): canonical_symbol("DASH"),
    canonical_symbol("DKUP"): canonical_symbol("DKNG"),
    canonical_symbol("LYFX"): canonical_symbol("LYFT"),
    canonical_symbol("NETX"): canonical_symbol("NET"),
    canonical_symbol("NWMX"): canonical_symbol("NEM"),
    canonical_symbol("OKTX"): canonical_symbol("OKTA"),
    canonical_symbol("PXIU"): canonical_symbol("UPXI"),
    canonical_symbol("QSX"):  canonical_symbol("QS"),
    # Defiance 2× KEEL sleeve: BTFL superseded by KEEX; keep BTFL so legacy
    # Flex rows and closed lots still roll up to KEEL when screened CSV drops BTFL.
    canonical_symbol("BTFL"): canonical_symbol("KEEL"),
}

# Map legacy trade-plan tickers to the successor screened name so plan-vs-Flex
# position discrepancies aggregate on one line (e.g. BTFL -> KEEX vs KEEL).
PLAN_ETF_TICKER_NORMALIZATION: dict[str, str] = {
    canonical_symbol("BTFL"): canonical_symbol("KEEX"),
}


def normalize_plan_etf_ticker(sym: str) -> str:
    s = canonical_symbol(sym)
    return PLAN_ETF_TICKER_NORMALIZATION.get(s, s)


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_blacklist(config_yml: Path) -> set[str]:
    cfg = yaml.safe_load(config_yml.read_text(encoding="utf-8")) or {}
    bl = set()
    for sym in (cfg.get("strategy", {}) or {}).get("blacklist", []) or []:
        s = canonical_symbol(str(sym).upper().strip())
        if s:
            bl.add(s)

    bl_txt = ((cfg.get("paths", {}) or {}).get("blacklist_txt", "") or "").strip()
    if bl_txt:
        p = (Path(__file__).resolve().parent / bl_txt).resolve()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                bl.add(canonical_symbol(line.upper()))
    return bl


def _find_col(cols_lc: dict[str, str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in cols_lc:
            return cols_lc[cand]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Universe / ETF→Underlying mapping
# ──────────────────────────────────────────────────────────────────────────────
def load_etf_to_under_map(screened_csv: Path) -> dict[str, str]:
    """
    Read data/etf_screened_today.csv and return ETF->Underlying mapping.
    Column names vary; we search common variants.
    """
    u = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in u.columns}

    etf_col = _find_col(cols, ["etf", "symbol", "ticker", "etf_symbol", "etf_ticker"])
    under_col = _find_col(cols, ["underlying", "underlyingsymbol", "underlying_symbol", "root", "underlyingticker"])

    if etf_col is None or under_col is None:
        raise ValueError(
            f"etf_screened_today.csv must contain ETF + underlying columns. Found: {list(u.columns)}"
        )

    u = u[[etf_col, under_col]].dropna()
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)

    u = u.drop_duplicates(subset=[etf_col], keep="last")
    return dict(zip(u[etf_col], u[under_col]))


def load_universe_from_screened(screened_csv: Path) -> tuple[set[str], set[str]]:
    """
    Returns:
      - allowed_etfs: screened ETFs
      - allowed_underlyings: their mapped underlyings
    """
    u = pd.read_csv(screened_csv)
    cols = {c.lower(): c for c in u.columns}

    etf_col = _find_col(cols, ["etf", "symbol", "ticker", "etf_symbol", "etf_ticker"])
    under_col = _find_col(cols, ["underlying", "underlyingsymbol", "underlying_symbol", "root", "underlyingticker"])

    if etf_col is None or under_col is None:
        raise ValueError(
            f"etf_screened_today.csv must contain ETF + underlying columns. Found: {list(u.columns)}"
        )

    u = u[[etf_col, under_col]].dropna()
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)
    u = u[(u[etf_col].astype(bool)) & (u[under_col].astype(bool))].drop_duplicates()

    allowed_etfs = set(u[etf_col].tolist())
    allowed_underlyings = set(u[under_col].tolist())
    return allowed_etfs, allowed_underlyings


def load_etf_beta_map(screened_csv: Path) -> tuple[dict[str, str], dict[str, float]]:
    """
    Load ETF -> Underlying mapping and ETF -> Beta from etf_screened_today.csv.
    Returns (etf_to_under, etf_to_beta) dicts keyed by canonical symbol.

    Beta represents the ETF's sensitivity to its underlying (e.g. 2.0 for a
    2× levered ETF, -1.0 for an inverse, 1.0 for a plain wrapper).
    """
    u = pd.read_csv(screened_csv)
    cols_lc = {c.lower(): c for c in u.columns}

    etf_col = _find_col(cols_lc, ["etf", "symbol", "ticker", "etf_symbol"])
    under_col = _find_col(cols_lc, ["underlying", "underlyingsymbol", "underlying_symbol", "root"])
    beta_col = _find_col(cols_lc, ["beta", "leverage", "lev"])

    if etf_col is None or under_col is None:
        return {}, {}

    u = u[[etf_col, under_col] + ([beta_col] if beta_col else [])].dropna(subset=[etf_col, under_col])
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)

    etf_to_under = dict(zip(u[etf_col], u[under_col]))

    if beta_col:
        u[beta_col] = pd.to_numeric(u[beta_col], errors="coerce").fillna(1.0)
        etf_to_beta = dict(zip(u[etf_col], u[beta_col]))
    else:
        etf_to_beta = {k: 1.0 for k in etf_to_under}

    return etf_to_under, etf_to_beta


def build_bucket4_pair_registry(
    screened_csv: Path,
    *,
    flow_short_syms: set[str] | None = None,
    proposed_trades_csv: Path | None = None,
    partial_hedge_ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Canonical inverse-decay (bucket 4) ETF ↔ underlying pairs.

    Sources: screened CSV (β < 0, not flow shorts), supplemental ETF map,
    optional proposed_trades sleeve ``inverse_decay_bucket4``.
    """
    cols = ["etf", "underlying", "beta", "partial_hedge_ratio"]
    etf_to_under, etf_to_beta = load_etf_beta_map(screened_csv)
    for sym, und in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(sym, und)
        etf_to_beta.setdefault(sym, etf_to_beta.get(sym, -1.0))

    flow = flow_short_syms or set()
    rows: list[dict] = []
    for etf, under in etf_to_under.items():
        beta = float(etf_to_beta.get(etf, 0.0))
        if beta >= 0 or etf in flow:
            continue
        rows.append(
            {
                "etf": canonical_symbol(etf),
                "underlying": canonical_symbol(under),
                "beta": beta,
                "partial_hedge_ratio": float(partial_hedge_ratio),
            }
        )

    if proposed_trades_csv is not None and proposed_trades_csv.exists():
        try:
            plan = pd.read_csv(proposed_trades_csv)
            if not plan.empty and "sleeve" in plan.columns:
                b4 = plan[plan["sleeve"].astype(str) == "inverse_decay_bucket4"].copy()
                etf_col = _find_col({c.lower(): c for c in b4.columns}, ["etf", "symbol", "ticker"])
                under_col = _find_col(
                    {c.lower(): c for c in b4.columns},
                    ["underlying", "underlyingsymbol", "underlying_symbol", "root"],
                )
                if etf_col and under_col:
                    for _, r in b4.iterrows():
                        etf = canonical_symbol(str(r[etf_col]))
                        under = canonical_symbol(str(r[under_col]))
                        if not etf or not under:
                            continue
                        beta = float(etf_to_beta.get(etf, -1.0))
                        rows.append(
                            {
                                "etf": etf,
                                "underlying": under,
                                "beta": beta,
                                "partial_hedge_ratio": float(partial_hedge_ratio),
                            }
                        )
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["etf"], keep="last").sort_values(["underlying", "etf"])
    return out[cols].reset_index(drop=True)


def held_exposure_bucket124_weights(
    underlying: str,
    pos: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_beta: dict[str, float],
    *,
    flow_low_beta_bucket3_syms: set[str] | None = None,
    flow_short_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
) -> tuple[float, float, float]:
    """EOD beta-adjusted held weights for splitting spot PnL across b1/b2/b4."""
    forced = flow_low_beta_bucket3_syms or set()
    flow = flow_short_syms or set()
    b4_set = b4_etf_syms or set()
    u = canonical_symbol(underlying)
    w1 = w2 = w4 = 0.0
    if pos is None or pos.empty:
        if b4_set and u in {etf_to_under.get(e, "") for e in b4_set}:
            return 0.0, 0.0, 1.0
        return 1.0, 0.0, 0.0

    for sym, grp in pos.groupby("symbol"):
        etf = canonical_symbol(str(sym))
        if etf_to_under.get(etf) != u:
            continue
        beta = float(etf_to_beta.get(etf, 0.0))
        if etf in forced:
            continue
        notional = float(grp["positionValue_base"].abs().sum())
        if notional <= 1e-12:
            continue
        w = notional * abs(beta)
        if beta < 0 and etf not in flow:
            w4 += w
        elif beta > 1.5:
            w1 += w
        elif beta > 0:
            w2 += w
    if (w1 + w2 + w4) <= 1e-12:
        if b4_set and any(etf_to_under.get(e) == u for e in b4_set):
            return 0.0, 0.0, 1.0
        return 1.0, 0.0, 0.0
    return _normalize_bucket_triple(w1, w2, w4)


def compute_bucket4_pair_exposure(
    pos: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    underlying_b4_ratio: dict[str, float] | None = None,
    underlying_b4_qty: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Beta-normalized exposure for bucket-4 pairs (short underlying + short inverse ETF).

    When the same underlying is shared with buckets 1/2, pass ``underlying_b4_qty`` with
    the exact attributed share count for the stock leg (preferred). ``underlying_b4_ratio``
    scales the full IBKR line when share counts are unavailable.

    Returns (by_underlying, detail_by_symbol).
    """
    empty_u = pd.DataFrame(
        columns=["underlying", "symbols", "net_notional_usd", "gross_notional_usd", "n_legs"]
    )
    empty_d = pd.DataFrame(
        columns=["underlying", "symbol", "leg_type", "net_notional_usd", "gross_notional_usd"]
    )
    if pos.empty or registry.empty:
        return empty_u, empty_d

    reg = registry.copy()
    reg["etf"] = reg["etf"].astype(str).map(canonical_symbol)
    reg["underlying"] = reg["underlying"].astype(str).map(canonical_symbol)
    reg["beta"] = pd.to_numeric(reg["beta"], errors="coerce").fillna(-1.0)

    pos = pos.copy()
    pos["symbol"] = pos["symbol"].astype(str).map(canonical_symbol)

    detail_rows: list[dict] = []
    for _, pr in reg.iterrows():
        etf = str(pr["etf"])
        under = str(pr["underlying"])
        beta = float(pr["beta"])

        u_ratio = float((underlying_b4_ratio or {}).get(under, 1.0))
        u_qty = (underlying_b4_qty or {}).get(under)
        for sym, leg_type, mult_beta in ((under, "underlying", 1.0), (etf, "etf", beta)):
            rows = pos[pos["symbol"] == sym]
            if rows.empty:
                continue
            for _, r in rows.iterrows():
                fx = float(r["fxRateToBase"])
                mark = float(r["markPrice"]) * fx
                if leg_type == "underlying" and u_qty is not None:
                    eff_qty = float(u_qty)
                else:
                    leg_scale = 1.0 if leg_type == "etf" else u_ratio
                    eff_qty = float(r["position"]) * leg_scale
                mv = eff_qty * mult_beta * mark
                detail_rows.append(
                    {
                        "underlying": under,
                        "symbol": sym,
                        "leg_type": leg_type,
                        "net_notional_usd": mv,
                        "gross_notional_usd": abs(mv),
                    }
                )

    if not detail_rows:
        return empty_u, empty_d

    detail = pd.DataFrame(detail_rows)
    by_under = (
        detail.groupby("underlying", as_index=False)
        .agg(
            symbols=("symbol", lambda s: ", ".join(sorted(set(s.astype(str))))),
            net_notional_usd=("net_notional_usd", "sum"),
            gross_notional_usd=("gross_notional_usd", "sum"),
            n_legs=("symbol", "nunique"),
        )
        .sort_values("net_notional_usd", ascending=False)
    )
    return by_under, detail


# ──────────────────────────────────────────────────────────────────────────────
# XML Parsers
# ──────────────────────────────────────────────────────────────────────────────
def parse_fifo_perf(trades_xml: Path) -> pd.DataFrame:
    """
    Robust: tries multiple nodes.
    Returns columns: symbol, underlyingSymbol, description, realized_pnl, unrealized_pnl
    """
    r = ET.parse(trades_xml).getroot()

    def _rows_from_node(node, in_base: bool) -> list[dict]:
        rows = []
        for child in node:
            a = child.attrib
            sym = canonical_symbol(a.get("symbol", "") or "")
            if not sym:
                continue
            und = canonical_symbol(a.get("underlyingSymbol", "") or "")
            desc = a.get("description", "") or ""

            if in_base:
                realized = float(a.get("totalRealizedPnl", "0") or 0)
                unreal = float(a.get("totalUnrealizedPnl", "0") or 0)
            else:
                fx = float(a.get("fxRateToBase", "1") or 1)
                realized_local = float(a.get("totalRealizedPnl", a.get("realizedPnl", "0")) or 0)
                unreal_local = float(a.get("totalUnrealizedPnl", a.get("unrealizedPnl", "0")) or 0)
                realized = to_base(realized_local, fx)
                unreal = to_base(unreal_local, fx)

            rows.append(
                {
                    "symbol": sym,
                    "underlyingSymbol": und,
                    "description": desc,
                    "realized_pnl": float(realized),
                    "unrealized_pnl": float(unreal),
                }
            )
        return rows

    n1 = r.find(".//FIFOPerformanceSummaryInBase")
    if n1 is not None:
        df = pd.DataFrame(_rows_from_node(n1, in_base=True))
        return df[df["symbol"].astype(bool)] if not df.empty else df

    n2 = r.find(".//FIFOPerformanceSummary")
    if n2 is not None:
        df = pd.DataFrame(_rows_from_node(n2, in_base=False))
        return df[df["symbol"].astype(bool)] if not df.empty else df

    return pd.DataFrame(columns=["symbol", "underlyingSymbol", "description", "realized_pnl", "unrealized_pnl"])


def parse_trade_events(trades_xml: Path) -> pd.DataFrame:
    """Parse Trade rows with timing + order reference metadata."""
    r = ET.parse(trades_xml).getroot()
    node = r.find(".//Trades")
    if node is None:
        return pd.DataFrame(
            columns=[
                "dateTime", "symbol", "underlyingSymbol", "buySell", "quantity",
                "fifoPnlRealized_base", "tradePrice_base", "orderReference", "openCloseIndicator",
            ]
        )
    rows: list[dict] = []
    for child in node:
        if child.tag != "Trade":
            continue
        a = child.attrib
        sym = canonical_symbol(a.get("symbol", "") or "")
        if not sym:
            continue
        und = canonical_symbol(a.get("underlyingSymbol", "") or "")
        buy_sell = str(a.get("buySell", "") or "").upper()
        q_raw = float(a.get("quantity", "0") or 0.0)
        qty_signed = q_raw
        if q_raw >= 0:
            if buy_sell == "SELL":
                qty_signed = -q_raw
            elif buy_sell == "BUY":
                qty_signed = q_raw
        realized_local = float(a.get("fifoPnlRealized", "0") or 0.0)
        trade_price_local = float(a.get("tradePrice", "0") or 0.0)
        fx = float(a.get("fxRateToBase", "1") or 1.0)
        rows.append(
            {
                "dateTime": str(a.get("dateTime", "") or ""),
                "symbol": sym,
                "underlyingSymbol": und,
                "buySell": buy_sell,
                "quantity": float(qty_signed),
                "fifoPnlRealized_base": float(to_base(realized_local, fx)),
                "tradePrice_base": float(to_base(trade_price_local, fx)),
                "orderReference": str(a.get("orderReference", "") or ""),
                "openCloseIndicator": str(a.get("openCloseIndicator", "") or ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("dateTime").reset_index(drop=True)


def _bucket_from_beta(beta: float, *, force_bucket3: bool = False) -> str | None:
    if force_bucket3:
        return "bucket_3"
    if beta < 0:
        return "bucket_4"
    if beta > 1.5:
        return "bucket_1"
    if beta > 0:
        return "bucket_2"
    return None


def _bucket_hint_from_order_reference(
    order_ref: str,
    etf_to_beta_map: dict[str, float],
    flow_low_beta_bucket3_syms: set[str] | None = None,
) -> str | None:
    forced = flow_low_beta_bucket3_syms or set()
    if not order_ref:
        return None
    for tok in str(order_ref).split("|"):
        raw = tok.strip().upper()
        candidates = [raw]
        if "__" in raw:
            candidates.append(raw.split("__", 1)[1].strip())
        for item in candidates:
            cand = canonical_symbol(item)
            if cand in etf_to_beta_map:
                return _bucket_from_beta(float(etf_to_beta_map.get(cand, 0.0)), force_bucket3=(cand in forced))
    return None


def build_underlying_realized_bucket_ratio_map(
    trade_events: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_beta: dict[str, float],
    flow_low_beta_bucket3_syms: set[str] | None = None,
    *,
    flow_short_syms: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Build per-underlying realized allocation ratios from timestamped trades.
    For underlying trades with realized PnL:
      1) use orderReference ETF hint when present
      2) fallback to live ETF-position bucket mix at that timestamp (b1/b2/b4).
    """
    if trade_events.empty:
        return {}

    forced = flow_low_beta_bucket3_syms or set()
    flow = flow_short_syms or set()
    etf_pos_qty: dict[str, float] = defaultdict(float)
    realized_by_under_bucket: dict[str, dict[str, float]] = defaultdict(
        lambda: {"bucket_1": 0.0, "bucket_2": 0.0, "bucket_4": 0.0}
    )

    for _, row in trade_events.iterrows():
        sym = str(row.get("symbol", ""))
        qty = float(row.get("quantity", 0.0) or 0.0)
        realized = float(row.get("fifoPnlRealized_base", 0.0) or 0.0)

        is_etf = sym in etf_to_under
        if is_etf:
            etf_pos_qty[sym] += qty

        if is_etf or abs(realized) <= 1e-12:
            continue

        under = str(row.get("underlyingSymbol", "") or "")
        if not under:
            under = sym
        under = canonical_symbol(under)

        b_hint = _bucket_hint_from_order_reference(
            str(row.get("orderReference", "")),
            etf_to_beta,
            flow_low_beta_bucket3_syms=forced,
        )
        if b_hint in {"bucket_1", "bucket_2", "bucket_4"}:
            realized_by_under_bucket[under][b_hint] += abs(realized)
            continue

        w1 = 0.0
        w2 = 0.0
        w4 = 0.0
        for etf_sym, etf_under in etf_to_under.items():
            if etf_under != under:
                continue
            b = float(etf_to_beta.get(etf_sym, 0.0))
            if etf_sym in forced:
                continue
            pos_qty = float(etf_pos_qty.get(etf_sym, 0.0))
            if abs(pos_qty) <= 1e-12:
                continue
            w = abs(pos_qty) * abs(b)
            if b < 0 and etf_sym not in flow:
                w4 += w
            elif b > 1.5:
                w1 += w
            elif b > 0:
                w2 += w
        if (w1 + w2 + w4) <= 1e-12:
            realized_by_under_bucket[under]["bucket_1"] += abs(realized)
        else:
            r1, r2, r4 = _normalize_bucket_triple(w1, w2, w4)
            realized_by_under_bucket[under]["bucket_1"] += abs(realized) * r1
            realized_by_under_bucket[under]["bucket_2"] += abs(realized) * r2
            realized_by_under_bucket[under]["bucket_4"] += abs(realized) * r4

    ratio_map: dict[str, dict[str, float]] = {}
    for under, d in realized_by_under_bucket.items():
        b1 = float(d.get("bucket_1", 0.0))
        b2 = float(d.get("bucket_2", 0.0))
        b4 = float(d.get("bucket_4", 0.0))
        r1, r2, r4 = _normalize_bucket_triple(b1, b2, b4)
        ratio_map[under] = {"b1": r1, "b2": r2, "b4": r4}
    return ratio_map


def parse_open_positions(pos_xml: Path) -> pd.DataFrame:
    r = ET.parse(pos_xml).getroot()
    op = r.find(".//OpenPositions")
    if op is None:
        raise ValueError("Could not find OpenPositions in positions XML.")
    rows = []
    for node in op:
        a = node.attrib
        rows.append(
            {
                "symbol": canonical_symbol(a.get("symbol", "")),
                "underlyingSymbol": canonical_symbol(a.get("underlyingSymbol", "") or ""),
                "position": float(a.get("position", "0") or 0),
                "markPrice": float(a.get("markPrice", "0") or 0),
                "positionValue": float(a.get("positionValue", "0") or 0),
                "fxRateToBase": float(a.get("fxRateToBase", "1") or 1),
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["symbol"].astype(bool)]
    df["positionValue_base"] = df.apply(lambda r: to_base(r["positionValue"], r["fxRateToBase"]), axis=1)

    if (df["position"] < 0).sum() == 0 and (df["positionValue_base"] < 0).sum() > 0:
        df["is_short"] = df["positionValue_base"] < 0
    else:
        df["is_short"] = df["position"] < 0

    return df


def fetch_yfinance_closes(
    symbols: list[str],
    trade_date: str,
    *,
    batch_size: int = 50,
) -> dict[str, float]:
    """
    Fetch closing prices for *trade_date* from Yahoo Finance v8 API.

    Returns a dict  { canonical_symbol: close_price }.
    Only symbols that actually have a close on *trade_date* are included.

    Uses the Yahoo Finance chart API directly (not the yfinance library)
    because yfinance v0.2.40's auto_adjust=True returns incorrect prices.
    """
    import requests as _requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dt = datetime.strptime(trade_date, "%Y-%m-%d")
    # Yahoo v8 chart API: use period1/period2 as Unix timestamps
    period1 = int(dt.timestamp())
    period2 = int((dt + timedelta(days=1)).timestamp())

    def _fetch_one(sym: str) -> tuple[str, float | None]:
        yahoo_sym = sym.replace(".", "-")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_sym}"
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = _requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result = data["chart"]["result"][0]
            close = result["indicators"]["quote"][0]["close"][-1]
            if close and close > 0:
                return sym, float(close)
        except Exception:
            pass
        return sym, None

    closes: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_fetch_one, sym) for sym in symbols]
        for fut in as_completed(futures):
            sym, price = fut.result()
            if price is not None:
                closes[sym] = price

    return closes


def override_mark_prices(
    pos: pd.DataFrame,
    yf_closes: dict[str, float],
    *,
    corp_action_split_dates: dict[str, str] | None = None,
    run_date: str | None = None,
) -> pd.DataFrame:
    """
    Replace markPrice in the positions DataFrame with yfinance closes
    where available.  Recalculates positionValue and positionValue_base.

    Split-day guard
    ---------------
    On the ex-date of an IBKR-reported reverse split, the Flex position
    quantity for that symbol may still be on the PRE-split basis while
    Yahoo's close is already POST-split (or vice versa). Multiplying the
    two yields a 10×–25× phantom MtM swing. When ``corp_action_split_dates``
    is supplied (mapping ``canonical_symbol -> 'YYYY-MM-DD'``) and the
    symbol's ex-date matches ``run_date``, we KEEP the existing Flex
    ``markPrice`` for that row instead of overriding from Yahoo. The
    fallback message ``[ACCOUNTING][corp-action] {sym} reverted to Flex
    mark on split day`` is emitted to stdout for the EOD log.

    Returns a new DataFrame (does not modify in place).
    """
    if not yf_closes:
        return pos

    pos = pos.copy()
    sym_to_yf = dict(yf_closes)

    # Carve out symbols whose ex-date matches run_date — never override these
    # from Yahoo on the split day. The Flex markPrice is the safest fallback
    # because it is paired with the Flex quantity on the same date.
    if corp_action_split_dates and run_date:
        for sym, ex_date in corp_action_split_dates.items():
            if ex_date == run_date and sym in sym_to_yf:
                sym_to_yf.pop(sym, None)
                print(
                    f"[ACCOUNTING][corp-action] {sym} reverted to Flex mark on split day"
                )

    mask = pos["symbol"].isin(sym_to_yf)
    pos.loc[mask, "markPrice"] = pos.loc[mask, "symbol"].map(sym_to_yf)
    pos["positionValue"] = pos["position"] * pos["markPrice"]
    pos["positionValue_base"] = pos["positionValue"] * pos["fxRateToBase"]
    return pos


def parse_cash_transactions(cash_xml: Path) -> pd.DataFrame:
    r = ET.parse(cash_xml).getroot()
    ct = r.find(".//CashTransactions")
    if ct is None:
        raise ValueError("Could not find CashTransactions in cash XML.")
    rows = []
    for node in ct:
        a = node.attrib
        rows.append(
            {
                "date": (a.get("dateTime", "") or "")[:8],
                "type": a.get("type", ""),
                "currency": a.get("currency", ""),
                "symbol": canonical_symbol(a.get("symbol", "") or ""),
                "underlyingSymbol": canonical_symbol(a.get("underlyingSymbol", "") or ""),
                "description": a.get("description", ""),
                "amount": float(a.get("amount", "0") or 0),
                "fxRateToBase": float(a.get("fxRateToBase", "1") or 1),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "type",
                "currency",
                "symbol",
                "underlyingSymbol",
                "description",
                "amount",
                "fxRateToBase",
                "amount_base",
            ]
        )
    df["amount_base"] = df.apply(lambda r: to_base(r["amount"], r["fxRateToBase"]), axis=1)
    return df


def parse_borrow_fee_details(borrow_xml: Path, run_date: str) -> pd.DataFrame:
    """
    Parse Borrow Fee Details CUMULATIVE up to (and including) run_date.
    """
    cols = ["symbol", "underlyingSymbol", "borrowFeeRate", "borrowFee_base"]
    if not borrow_xml.exists():
        return pd.DataFrame(columns=cols)

    r = ET.parse(borrow_xml).getroot()
    target = yyyymmdd_from_run_date(run_date)
    rows: list[dict] = []

    nodes = r.findall(".//HardToBorrowDetail")
    if not nodes:
        nodes = r.findall(".//*[@valueDate]")

    for node in nodes:
        a = node.attrib
        vd = (a.get("valueDate", "") or "").strip()

        if vd > target:
            continue

        sym = canonical_symbol(a.get("symbol", "") or "")
        if not sym:
            continue

        fx = float(a.get("fxRateToBase", "1") or 1)
        borrow_fee = float(a.get("borrowFee", "0") or 0)
        borrow_fee_base = to_base(borrow_fee, fx)

        br = a.get("borrowFeeRate", "")
        try:
            borrow_rate = float(br) if br not in ("", None) else np.nan
        except Exception:
            borrow_rate = np.nan

        rows.append(
            {
                "symbol": sym,
                "underlyingSymbol": canonical_symbol(a.get("underlyingSymbol", "") or ""),
                "borrowFeeRate": borrow_rate,
                "borrowFee_base": float(borrow_fee_base),
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out

    out = (
        out.groupby("symbol", as_index=False)
        .agg(
            underlyingSymbol=("underlyingSymbol", "first"),
            borrowFeeRate=("borrowFeeRate", "max"),
            borrowFee_base=("borrowFee_base", "sum"),
        )
    )
    return out


def parse_borrow_fee_events(borrow_xml: Path, run_date: str) -> pd.DataFrame:
    """Parse borrow fee details as dated events up to run_date."""
    cols = ["date", "symbol", "underlyingSymbol", "borrowFee_base"]
    if not borrow_xml.exists():
        return pd.DataFrame(columns=cols)
    r = ET.parse(borrow_xml).getroot()
    target = yyyymmdd_from_run_date(run_date)
    rows: list[dict] = []
    nodes = r.findall(".//HardToBorrowDetail")
    if not nodes:
        nodes = r.findall(".//*[@valueDate]")
    for node in nodes:
        a = node.attrib
        vd = (a.get("valueDate", "") or "").strip()
        if not vd or vd > target:
            continue
        sym = canonical_symbol(a.get("symbol", "") or "")
        if not sym:
            continue
        fx = float(a.get("fxRateToBase", "1") or 1)
        fee = float(a.get("borrowFee", "0") or 0)
        rows.append(
            {
                "date": vd,
                "symbol": sym,
                "underlyingSymbol": canonical_symbol(a.get("underlyingSymbol", "") or ""),
                "borrowFee_base": float(to_base(fee, fx)),
            }
        )
    return pd.DataFrame(rows, columns=cols)


# ──────────────────────────────────────────────────────────────────────────────
# Cash categorization + allocation fallback
# ──────────────────────────────────────────────────────────────────────────────
def categorize_cash_row(row: pd.Series) -> str:
    t = row["type"]
    desc = (row["description"] or "").upper()

    if t == "Dividends":
        return "dividends"
    if t == "Withholding Tax":
        return "withholding_tax"
    if t == "Payment In Lieu Of Dividends":
        return "pil_dividends"
    if t == "Other Fees":
        return "other_fees"

    if t in ("Broker Interest Paid", "Broker Interest Received"):
        if "BORROW FEES" in desc:
            return "borrow_fees"
        if "SHORT CREDIT INTEREST" in desc:
            return "short_credit_interest"
        return "exclude_interest"

    if t == "Bond Interest Received":
        return "bond_interest"

    return "other"


def split_symbol_vs_account_level(rows: pd.DataFrame) -> tuple[pd.Series, float]:
    if rows.empty:
        return pd.Series(dtype=float), 0.0
    has_sym = rows["symbol"].astype(str).str.len() > 0
    direct = rows[has_sym].groupby("symbol")["amount_base"].sum()
    remainder = float(rows[~has_sym]["amount_base"].sum())
    return direct, remainder


def allocate_to_shorts(unallocated_amount: float, pos: pd.DataFrame) -> pd.Series:
    if pos.empty or unallocated_amount == 0:
        return pd.Series(dtype=float)

    shorts = pos[pos["is_short"]].copy() if "is_short" in pos.columns else pos[pos["position"] < 0].copy()
    if shorts.empty:
        return pd.Series(dtype=float)

    weights = shorts.groupby("symbol")["positionValue_base"].apply(lambda s: float(np.abs(s).sum()))
    wsum = float(weights.sum())
    if wsum == 0:
        return pd.Series(dtype=float)

    return (weights / wsum) * float(unallocated_amount)


# ──────────────────────────────────────────────────────────────────────────────
# Net Exposure calculation (beta-normalized)
# ──────────────────────────────────────────────────────────────────────────────
def compute_net_exposure(
    pos_xml: Path,
    screened_csv: Path,
    pnl_underlyings: set[str] | None = None,
    *,
    positions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute beta-normalized net exposure by underlying.

    For each position:
      mv_base = position × beta × markPrice × fxRateToBase

    Beta is the ETF's sensitivity to its underlying (e.g. 2.0 for a 2×
    levered ETF).  Spot / 1× holdings have beta = 1.0.

    Then group by underlying to get net_notional_usd and gross_notional_usd.

    If pnl_underlyings is provided, only include underlyings in that set
    (to keep the same universe as the PnL report).

    Parameters
    ----------
    positions_df : DataFrame, optional
        Pre-parsed positions (e.g. with yfinance-overridden markPrices).
        If provided, pos_xml is ignored for position data.

    Returns
    -------
    exposure_df : DataFrame
        Grouped by underlying with columns:
        underlying, symbols, net_notional_usd, gross_notional_usd, n_legs
    pos_detail_df : DataFrame
        Per-symbol rows with columns:
        underlying, symbol, net_notional_usd, gross_notional_usd
    """
    empty = pd.DataFrame(columns=["underlying", "symbols", "net_notional_usd", "gross_notional_usd", "n_legs"])
    empty_detail = pd.DataFrame(columns=["underlying", "symbol", "net_notional_usd", "gross_notional_usd"])

    if positions_df is not None:
        pos = positions_df.copy()
    else:
        pos = parse_open_positions(pos_xml)
    if pos.empty:
        return empty, empty_detail

    etf_to_under, etf_to_beta = load_etf_beta_map(screened_csv)

    # Map each position to its underlying and beta
    pos["is_etf"] = pos["symbol"].isin(etf_to_under)
    pos["underlying"] = np.where(
        pos["is_etf"],
        pos["symbol"].map(etf_to_under),
        pos["symbol"],
    )
    pos["beta"] = np.where(
        pos["is_etf"],
        pos["symbol"].map(etf_to_beta).astype(float),
        1.0,
    )
    pos["beta"] = pd.to_numeric(pos["beta"], errors="coerce").fillna(1.0)

    # Filter to strategy universe: only include positions whose symbol
    # is either an ETF in the beta map or an underlying mapped by at
    # least one ETF.  This excludes delisted ETFs (dropped from screened
    # CSV by daily_screener), personal holdings, CVRs, and other
    # non-strategy positions.
    strategy_universe = set(etf_to_under.keys()) | set(etf_to_under.values())
    pos = pos[pos["symbol"].isin(strategy_universe)].copy()

    if pos.empty:
        return empty, empty_detail

    # Beta-adjusted signed market value in base currency
    pos["mv_base"] = pos["position"] * pos["beta"] * pos["markPrice"] * pos["fxRateToBase"]
    pos["gross_mv_base"] = pos["mv_base"].abs()

    # Filter to PnL universe if provided
    if pnl_underlyings is not None:
        pos = pos[pos["underlying"].isin(pnl_underlyings)].copy()

    if pos.empty:
        return empty, empty_detail

    exposure = (
        pos.groupby("underlying", as_index=False)
        .agg(
            symbols=("symbol", lambda s: ", ".join(sorted(s.astype(str)))),
            net_notional_usd=("mv_base", "sum"),
            gross_notional_usd=("gross_mv_base", "sum"),
            n_legs=("symbol", "nunique"),
        )
        .sort_values("net_notional_usd", ascending=False)
    )

    pos_detail = (
        pos.groupby(["underlying", "symbol"], as_index=False)
        .agg(
            net_notional_usd=("mv_base", "sum"),
            gross_notional_usd=("gross_mv_base", "sum"),
        )
        .sort_values(["underlying", "net_notional_usd"], ascending=[True, False])
    )

    return exposure, pos_detail


def format_exposure_table(
    exposure_df: pd.DataFrame,
    pos_detail_df: pd.DataFrame | None = None,
) -> tuple[str, float, float]:
    """
    Format net exposure by underlying as a plain-text table.
    If pos_detail_df is provided, per-symbol net/gross rows are shown
    indented under each underlying.
    Returns (table_str, total_net, total_gross).
    """
    if exposure_df.empty:
        return "(no exposure data)", 0.0, 0.0

    tbl = exposure_df[["underlying", "net_notional_usd", "gross_notional_usd"]].copy()
    total_net = float(tbl["net_notional_usd"].sum())
    total_gross = float(tbl["gross_notional_usd"].sum())

    # Build all label candidates for column-width calculation
    all_labels = list(tbl["underlying"].astype(str)) + ["TOTAL"]
    if pos_detail_df is not None and not pos_detail_df.empty and "symbol" in pos_detail_df.columns:
        all_labels += ["  " + s for s in pos_detail_df["symbol"].astype(str)]
    u_w = max(10, max(len(s) for s in all_labels))

    all_net = list(tbl["net_notional_usd"]) + [total_net]
    all_gross = list(tbl["gross_notional_usd"]) + [total_gross]
    if pos_detail_df is not None and not pos_detail_df.empty:
        if "net_notional_usd" in pos_detail_df.columns:
            all_net += list(pos_detail_df["net_notional_usd"])
        if "gross_notional_usd" in pos_detail_df.columns:
            all_gross += list(pos_detail_df["gross_notional_usd"])
    net_w = max(14, max(len(f"{v:,.2f}") for v in all_net))
    gross_w = max(16, max(len(f"{v:,.2f}") for v in all_gross))

    lines: list[str] = []
    header = (
        f"{'UNDERLYING / SYMBOL'.ljust(u_w)}  "
        f"{'NET_NOTIONAL'.rjust(net_w)}  "
        f"{'GROSS_NOTIONAL'.rjust(gross_w)}"
    )
    lines.append(header)
    lines.append("-" * (u_w + 2 + net_w + 2 + gross_w))

    for _, r in tbl.iterrows():
        underlying = str(r["underlying"])
        n = f"{float(r['net_notional_usd']):,.2f}"
        g = f"{float(r['gross_notional_usd']):,.2f}"
        lines.append(f"{underlying.ljust(u_w)}  {n.rjust(net_w)}  {g.rjust(gross_w)}")

        # Indented per-symbol rows
        if pos_detail_df is not None and not pos_detail_df.empty and "underlying" in pos_detail_df.columns:
            syms = pos_detail_df[pos_detail_df["underlying"] == underlying].sort_values(
                "net_notional_usd", ascending=False
            )
            for _, sr in syms.iterrows():
                sym_label = ("  " + str(sr["symbol"])).ljust(u_w)
                sn = f"{float(sr['net_notional_usd']):,.2f}".rjust(net_w)
                sg = f"{float(sr['gross_notional_usd']):,.2f}".rjust(gross_w)
                lines.append(f"{sym_label}  {sn}  {sg}")

    lines.append("-" * (u_w + 2 + net_w + 2 + gross_w))
    lines.append(f"{'TOTAL'.ljust(u_w)}  {f'{total_net:,.2f}'.rjust(net_w)}  {f'{total_gross:,.2f}'.rjust(gross_w)}")

    return "\n".join(lines), total_net, total_gross


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(run_date: str | None = None, *, use_yfinance: bool | None = None) -> int:
    """
    Run the full accounting pipeline.

    Parameters
    ----------
    run_date : str, optional
        YYYY-MM-DD.  Defaults to sys.argv[1] or today.
    use_yfinance : bool | None
        If True, override Flex markPrices with yfinance closes for more
        timely mark-to-market.  If None (default), read from
        strategy_config.yml → accounting.use_yfinance (default False).
    """
    if run_date is None:
        run_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    run_dir = PROJECT_ROOT / "data" / "runs" / run_date / "ibkr_flex"

    flex_cash_path = run_dir / "flex_cash.xml"
    flex_positions_path = run_dir / "flex_positions.xml"
    flex_trades_path = run_dir / "flex_trades.xml"
    flex_borrow_details_path = run_dir / "flex_borrow_fee_details.xml"

    missing = [p for p in [flex_cash_path, flex_positions_path, flex_trades_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required IBKR Flex files:\n" + "\n".join(str(p) for p in missing))

    config_yml_path = Path(__file__).resolve().parent / "config" / "strategy_config.yml"
    if not config_yml_path.exists():
        raise FileNotFoundError(f"Missing strategy_config.yml at: {config_yml_path}")

    # Resolve use_yfinance from config if not explicitly passed
    cfg = yaml.safe_load(config_yml_path.read_text(encoding="utf-8")) or {}
    portfolio_cfg = cfg.get("portfolio", {}) or {}
    sleeves_cfg = portfolio_cfg.get("sleeves", {}) or {}
    flow_cfg = sleeves_cfg.get("flow_program", {}) or {}
    flow_universe_cfg = flow_cfg.get("universe", {}) or {}
    flow_shorts_cfg = flow_universe_cfg.get("shorts", []) or []
    flow_short_set = {canonical_symbol(x) for x in flow_shorts_cfg if str(x).strip()}
    if use_yfinance is None:
        use_yfinance = bool((cfg.get("accounting", {}) or {}).get("use_yfinance", False))
    split_method = str(
        (cfg.get("accounting", {}) or {}).get("bucket_split_method", "held_exposure")
    ).strip().lower()
    if split_method not in {"held_exposure", "universe_beta"}:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.bucket_split_method={split_method!r}; "
            "defaulting to 'held_exposure'"
        )
        split_method = "held_exposure"
    component_split_method = str(
        (cfg.get("accounting", {}) or {}).get("bucket_component_split_method", "lot_timed")
    ).strip().lower()
    if component_split_method not in {"lot_timed", "current_only"}:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.bucket_component_split_method={component_split_method!r}; "
            "defaulting to 'lot_timed'"
        )
        component_split_method = "lot_timed"
    bucket_state_path = PROJECT_ROOT / "data" / "accounting" / "underlying_bucket_state.csv"
    b4_partial_hedge_ratio = float(
        ((sleeves_cfg.get("inverse_decay_bucket4") or {}).get("rules") or {}).get(
            "partial_hedge_ratio", 1.0
        )
    )

    etf_screened_path = PROJECT_ROOT / "data" / "etf_screened_today.csv"
    if not etf_screened_path.exists():
        raise FileNotFoundError(f"Missing etf_screened_today.csv at: {etf_screened_path}")

    outdir: Path = run_dir.parent / "accounting"
    outdir.mkdir(parents=True, exist_ok=True)

    proposed_trades_path = run_dir.parent / "proposed_trades.csv"
    if not proposed_trades_path.exists():
        proposed_trades_path = PROJECT_ROOT / "data" / "proposed_trades.csv"

    fifo = parse_fifo_perf(flex_trades_path)
    trade_events = parse_trade_events(flex_trades_path)
    pos = parse_open_positions(flex_positions_path)
    cash = parse_cash_transactions(flex_cash_path)
    borrow_details = parse_borrow_fee_details(flex_borrow_details_path, run_date)
    borrow_fee_events = parse_borrow_fee_events(flex_borrow_details_path, run_date)

    # Pull <CorporateAction type="RS"> rows from the Flex cash XML and merge
    # them into ``data/splits_from_flex.csv``. The daily screener consumes
    # that CSV as the highest-precedence split source on its next run, and
    # ``override_mark_prices`` below uses today's CA set to skip Yahoo
    # overrides on the split's ex-date.
    corp_action_split_dates: dict[str, str] = {}
    try:
        from splits import parse_flex_corporate_action_splits  # type: ignore
        ca_splits_today = parse_flex_corporate_action_splits(flex_cash_path)
    except Exception as e:
        print(f"[ACCOUNTING][corp-action] split extraction failed: {e}")
        ca_splits_today = pd.DataFrame()

    if not ca_splits_today.empty:
        for _, r in ca_splits_today.iterrows():
            sym = canonical_symbol(str(r.get("symbol", "")))
            ex_date_v = str(r.get("ex_date", "")).strip()
            if sym and ex_date_v:
                corp_action_split_dates[sym] = ex_date_v

        flex_splits_csv = PROJECT_ROOT / "data" / "splits_from_flex.csv"
        flex_splits_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            cols = list(ca_splits_today.columns)
            if flex_splits_csv.is_file():
                existing = pd.read_csv(flex_splits_csv)
            else:
                existing = pd.DataFrame(columns=cols)
            combined = pd.concat([existing, ca_splits_today], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["symbol", "ex_date"], keep="last"
            )
            combined = combined.sort_values(by=["symbol", "ex_date"]).reset_index(drop=True)
            combined.to_csv(flex_splits_csv, index=False)
            n_today = len(ca_splits_today)
            print(
                f"[ACCOUNTING][corp-action] {n_today} RS row(s) merged into "
                f"{flex_splits_csv} ({len(combined)} total)"
            )
        except Exception as e:
            print(f"[ACCOUNTING][corp-action] failed to write splits CSV: {e}")

    # Exclusions
    if not fifo.empty:
        fifo = fifo[~fifo["symbol"].isin(EXCLUDE_SYMBOLS) & ~fifo["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if not pos.empty:
        pos = pos[~pos["symbol"].isin(EXCLUDE_SYMBOLS) & ~pos["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if not cash.empty:
        cash = cash[~cash["symbol"].isin(EXCLUDE_SYMBOLS) & ~cash["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if not borrow_details.empty:
        borrow_details = borrow_details[~borrow_details["symbol"].isin(EXCLUDE_SYMBOLS)].copy()

    if cash.empty:
        cash["category"] = pd.Series(dtype=str)
    else:
        cash["category"] = cash.apply(categorize_cash_row, axis=1)

    # Master universe (so cash / borrow rows don't disappear)
    all_syms = set()
    if not fifo.empty:
        all_syms |= set(fifo["symbol"].dropna().astype(str))
    if not pos.empty:
        all_syms |= set(pos["symbol"].dropna().astype(str))
    if not cash.empty:
        all_syms |= set(cash["symbol"].dropna().astype(str))
    if not borrow_details.empty:
        all_syms |= set(borrow_details["symbol"].dropna().astype(str))

    master = pd.DataFrame({"symbol": sorted(s for s in all_syms if s)})

    # Trading PnL by symbol (IBKR FIFO)
    if fifo.empty:
        fifo_agg = pd.DataFrame(columns=["symbol", "realized_pnl", "unrealized_pnl"])
    else:
        fifo_agg = fifo.groupby("symbol", as_index=False)[["realized_pnl", "unrealized_pnl"]].sum()

    df = master.merge(fifo_agg, on="symbol", how="left")
    df["realized_pnl"] = df.get("realized_pnl", 0.0).fillna(0.0)
    df["unrealized_pnl"] = df.get("unrealized_pnl", 0.0).fillna(0.0)

    # Metadata
    fifo_meta = (
        fifo.groupby("symbol", as_index=False).agg(
            underlyingSymbol=("underlyingSymbol", "first"),
            description=("description", "first"),
        )
        if not fifo.empty
        else pd.DataFrame(columns=["symbol", "underlyingSymbol", "description"])
    )
    pos_meta = (
        pos.groupby("symbol", as_index=False).agg(underlyingSymbol_pos=("underlyingSymbol", "first"))
        if not pos.empty
        else pd.DataFrame(columns=["symbol", "underlyingSymbol_pos"])
    )
    cash_meta = (
        cash.groupby("symbol", as_index=False).agg(underlyingSymbol_cash=("underlyingSymbol", "first"))
        if not cash.empty
        else pd.DataFrame(columns=["symbol", "underlyingSymbol_cash"])
    )
    bfd_meta = (
        borrow_details.groupby("symbol", as_index=False).agg(underlyingSymbol_bfd=("underlyingSymbol", "first"))
        if ("symbol" in borrow_details.columns and not borrow_details.empty)
        else pd.DataFrame(columns=["symbol", "underlyingSymbol_bfd"])
    )

    df = df.merge(fifo_meta, on="symbol", how="left")
    df = df.merge(pos_meta, on="symbol", how="left")
    df = df.merge(cash_meta, on="symbol", how="left")
    df = df.merge(bfd_meta, on="symbol", how="left")

    df["underlyingSymbol"] = (
        df.get("underlyingSymbol")
        .fillna(df.get("underlyingSymbol_pos"))
        .fillna(df.get("underlyingSymbol_cash"))
        .fillna(df.get("underlyingSymbol_bfd"))
        .fillna("")
    )
    df["description"] = df.get("description", "").fillna("")

    # Cash flows per symbol
    cash_sym = cash[cash["category"].isin(["dividends", "withholding_tax", "pil_dividends", "other_fees", "bond_interest"])].copy()
    cash_pivot = (
        cash_sym.pivot_table(index="symbol", columns="category", values="amount_base", aggfunc="sum", fill_value=0).reset_index()
        if not cash_sym.empty
        else pd.DataFrame({"symbol": df["symbol"].unique()})
    )
    cash_pivot.columns.name = None

    df = df.merge(cash_pivot, on="symbol", how="left")
    for c in ["dividends", "withholding_tax", "pil_dividends", "other_fees", "bond_interest"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    # Underlying mapping (source of truth: data/etf_screened_today.csv)
    etf_to_under = load_etf_to_under_map(etf_screened_path)
    for sym, und in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(sym, und)
    _, etf_to_beta_pre = load_etf_beta_map(etf_screened_path)
    for sym in SUPPLEMENTAL_ETF_MAP:
        etf_to_beta_pre.setdefault(sym, etf_to_beta_pre.get(sym, -1.0))
    b4_registry = build_bucket4_pair_registry(
        etf_screened_path,
        flow_short_syms=flow_short_set,
        proposed_trades_csv=proposed_trades_path,
        partial_hedge_ratio=b4_partial_hedge_ratio,
    )
    b4_registry.to_csv(outdir / "bucket4_pairs.csv", index=False)
    b4_etf_syms = set(b4_registry["etf"].astype(str).tolist()) if not b4_registry.empty else set()
    b4_underlyings = (
        set(b4_registry["underlying"].astype(str).tolist()) if not b4_registry.empty else set()
    )
    df["underlying"] = df["symbol"].map(etf_to_under)
    df["underlying"] = df["underlying"].fillna(df["underlyingSymbol"]).fillna(df["symbol"])

    # Drop Berkshire mapping
    df = df[(~df["symbol"].isin(EXCLUDE_SYMBOLS)) & (~df["underlying"].isin(EXCLUDE_SYMBOLS))].copy()

    # Pair label pre-filter
    df["pair"] = np.where(
        df["symbol"] == df["underlying"],
        df["underlying"] + " (spot)",
        df["underlying"] + " | " + df["symbol"],
    )

    # Blacklist
    blacklist = load_blacklist(config_yml_path)
    df = df[~df["symbol"].isin(blacklist)].copy()
    df = df[~df["underlying"].isin(blacklist)].copy()

    # Universe filter — whitelist approach.  A symbol can only appear
    # in PnL if it is a KNOWN part of our strategy universe:
    #   1. An ETF in the screened CSV or SUPPLEMENTAL_ETF_MAP
    #   2. An underlying mapped by one of those ETFs
    # This excludes non-strategy positions (CSU.DB, 3905.T, currencies)
    # while preserving historical PnL for delisted pairs (CELT/CELH, etc.)
    allowed_etfs, allowed_underlyings = load_universe_from_screened(etf_screened_path)
    allowed_underlyings |= set(SUPPLEMENTAL_ETF_MAP.values())
    allowed_etfs |= set(SUPPLEMENTAL_ETF_MAP.keys())
    allowed_symbols = allowed_etfs | allowed_underlyings

    df = df[df["symbol"].isin(allowed_symbols)].copy()

    # Rebuild pair labels post-filter
    df["pair"] = np.where(
        df["symbol"] == df["underlying"],
        df["underlying"] + " (spot)",
        df["underlying"] + " | " + df["symbol"],
    )

    # Borrow + SCR
    kept_syms = set(df["symbol"].unique())

    # ── yfinance mark-price override (only for kept symbols) ──
    yf_closes: dict[str, float] = {}
    if use_yfinance and kept_syms:
        # Also include the underlyings themselves (spot positions)
        yf_syms = sorted(kept_syms)
        yf_closes = fetch_yfinance_closes(yf_syms, run_date)

        if yf_closes:
            # 1) Adjust unrealized PnL:  delta = position × (yf − flex) × fx
            yf_unrealized_adj: dict[str, float] = {}
            for _, row in pos[pos["symbol"].isin(kept_syms)].iterrows():
                sym = row["symbol"]
                if sym in yf_closes:
                    price_diff = yf_closes[sym] - row["markPrice"]
                    adj = row["position"] * price_diff * row["fxRateToBase"]
                    yf_unrealized_adj[sym] = yf_unrealized_adj.get(sym, 0.0) + adj

            # On any symbol whose Flex CorporateAction reports a reverse-split
            # ex-date == run_date, drop the Yahoo unrealized adjustment too:
            # IBKR's FIFO PnL is consistent with the pre-split Flex quantity
            # and the post-split Yahoo close would create a 10× phantom delta.
            if corp_action_split_dates:
                split_today = {
                    sym for sym, ex in corp_action_split_dates.items() if ex == run_date
                }
                if split_today:
                    for s in split_today:
                        if s in yf_unrealized_adj:
                            yf_unrealized_adj.pop(s, None)
                    print(
                        f"[ACCOUNTING][corp-action] dropped Yahoo unrealized "
                        f"adjustment on split-day symbols: "
                        f"{sorted(split_today)}"
                    )

            df["unrealized_pnl"] += df["symbol"].map(yf_unrealized_adj).fillna(0.0)

            # 2) Override markPrice on positions (affects exposure calc downstream)
            pos = override_mark_prices(
                pos,
                yf_closes,
                corp_action_split_dates=corp_action_split_dates,
                run_date=run_date,
            )

            n_overridden = sum(
                1 for s in yf_syms
                if s in yf_closes
                and not (
                    corp_action_split_dates.get(s) == run_date
                )
            )
            print(f"[ACCOUNTING] yfinance: overrode {n_overridden}/{len(yf_syms)} markPrices for {run_date}")
        else:
            print(f"[ACCOUNTING] yfinance: no closes returned for {run_date} — keeping Flex markPrices")

    pos_kept_shorts = pos[(pos["symbol"].isin(kept_syms)) & (pos["is_short"])].copy()

    # SCR from cash
    scr_rows = cash[cash["category"] == "short_credit_interest"].copy()
    scr_direct, scr_remainder = split_symbol_vs_account_level(scr_rows)
    scr_alloc = allocate_to_shorts(scr_remainder, pos_kept_shorts)
    df["short_credit_interest"] = df["symbol"].map(scr_direct).fillna(0.0) + df["symbol"].map(scr_alloc).fillna(0.0)

    # Borrow fees: prefer borrow details
    borrow_details_kept = (
        borrow_details[borrow_details["symbol"].isin(kept_syms)].copy()
        if ("symbol" in borrow_details.columns)
        else pd.DataFrame()
    )
    if not borrow_details_kept.empty:
        bfd_map = borrow_details_kept.set_index("symbol")["borrowFee_base"].to_dict()
        df["borrow_fees"] = df["symbol"].map(bfd_map).fillna(0.0)
        borrow_mode = "borrow_fee_details_cumulative_by_symbol"
        borrow_total_account_source = float(borrow_details_kept["borrowFee_base"].sum())
    else:
        borrow_rows = cash[cash["category"] == "borrow_fees"].copy()
        borrow_direct, borrow_remainder = split_symbol_vs_account_level(borrow_rows)
        borrow_alloc = allocate_to_shorts(borrow_remainder, pos_kept_shorts)
        df["borrow_fees"] = df["symbol"].map(borrow_direct).fillna(0.0) + df["symbol"].map(borrow_alloc).fillna(0.0)
        borrow_mode = "cash_borrow_allocated_pro_rata_to_kept_shorts"
        borrow_total_account_source = float(borrow_rows["amount_base"].sum()) if not borrow_rows.empty else 0.0

    # Total PnL
    df["total_pnl"] = (
        df["realized_pnl"]
        + df["unrealized_pnl"]
        + df["dividends"]
        + df["withholding_tax"]
        + df["pil_dividends"]
        + df["borrow_fees"]
        + df["short_credit_interest"]
        + df["other_fees"]
        + df["bond_interest"]
    )

    base_cols = [
        "realized_pnl",
        "unrealized_pnl",
        "dividends",
        "withholding_tax",
        "pil_dividends",
        "borrow_fees",
        "short_credit_interest",
        "other_fees",
        "bond_interest",
        "total_pnl",
    ]
    df_pre_bucket = df.copy()

    # ── Bucket assignment ──
    # bucket_1: ETF beta > 1.5  (levered)
    # bucket_2: ETF beta 0 < β ≤ 1.5  (standard / low-lev)
    # bucket_3: ETF beta < 0  (inverse)
    # Spot positions (not in etf_to_beta_map) are split pro-rata by the
    # absolute betas of the ETFs sharing their underlying.
    _, etf_to_beta_map = load_etf_beta_map(etf_screened_path)
    flow_low_beta_bucket3_syms = {
        s for s in flow_short_set if 0 < float(etf_to_beta_map.get(s, 0.0)) <= 1.0
    }
    neg_beta_syms = {s for s, b in etf_to_beta_map.items() if b < 0}
    bucket3_etf_syms = set(neg_beta_syms) | set(flow_low_beta_bucket3_syms)

    is_etf = df["symbol"].isin(etf_to_beta_map)
    sym_beta = df["symbol"].map(etf_to_beta_map)

    # Assign buckets for ETF positions
    def _etf_bucket(b):
        if b < 0:
            return "bucket_3"
        elif b > 1.5:
            return "bucket_1"
        else:
            return "bucket_2"

    # Build a fallback split map from the ETF universe itself (per underlying).
    # This is used when we do not currently hold any ETF legs for an underlying.
    # Weight by absolute beta so underlyings with only >1.5 ETFs default to
    # bucket_1 for spot/fallback attribution.
    _all_rows = []
    for _s, _b in etf_to_beta_map.items():
        if _s in flow_low_beta_bucket3_syms:
            continue
        if _b < 0 and _s not in flow_short_set:
            _bkt = "bucket_4"
        elif _b <= 0:
            continue
        else:
            _bkt = "bucket_1" if _b > 1.5 else "bucket_2"
        _all_rows.append(
            {
                "underlying": etf_to_under.get(_s, _s),
                "_bkt": _bkt,
                "beta_weight": abs(float(_b)),
            }
        )

    if _all_rows:
        _all_df = pd.DataFrame(_all_rows)
        _all_beta_sums = (
            _all_df.groupby(["underlying", "_bkt"])["beta_weight"]
            .sum()
            .reset_index()
            .pivot(index="underlying", columns="_bkt", values="beta_weight")
            .fillna(0)
            .reset_index()
        )
        for bc in ["bucket_1", "bucket_2", "bucket_4"]:
            if bc not in _all_beta_sums.columns:
                _all_beta_sums[bc] = 0.0
        _all_beta_sums["_total"] = (
            _all_beta_sums["bucket_1"] + _all_beta_sums["bucket_2"] + _all_beta_sums["bucket_4"]
        )
        _fallback_ratio_map: dict[str, dict[str, float]] = {}
        for _, row in _all_beta_sums.iterrows():
            r1, r2, r4 = _normalize_bucket_triple(
                float(row["bucket_1"]),
                float(row["bucket_2"]),
                float(row["bucket_4"]),
            )
            _fallback_ratio_map[row["underlying"]] = {"b1": r1, "b2": r2, "b4": r4}
    else:
        _fallback_ratio_map = {}

    # Build ratio map from ETFs we actually hold (open positions only).
    # Weight by beta-adjusted absolute notional so the spot split reflects
    # current exposure. This map overrides the universe fallback above.
    _held_etf_syms = set(pos["symbol"].unique()) & set(etf_to_beta_map.keys())
    _held_positive = {
        s for s in _held_etf_syms
        if etf_to_beta_map.get(s, 0) > 0 and s not in flow_low_beta_bucket3_syms
    }

    _held_rows = []
    _held_b4 = {
        s
        for s in (set(pos["symbol"].unique()) & set(etf_to_beta_map.keys()))
        if etf_to_beta_map.get(s, 0) < 0 and s not in flow_short_set
    }
    for _s in set(_held_positive) | _held_b4:
        _b = float(etf_to_beta_map[_s])
        _pos_rows = pos[pos["symbol"] == _s]
        _notional = float(_pos_rows["positionValue_base"].abs().sum())
        _beta_adj_notional = _notional * abs(_b)
        if _b < 0:
            _bkt = "bucket_4"
        elif _b > 1.5:
            _bkt = "bucket_1"
        else:
            _bkt = "bucket_2"
        _held_rows.append({
            "underlying": etf_to_under.get(_s, _s),
            "_bkt": _bkt,
            "beta_adj_notional": _beta_adj_notional,
        })

    if _held_rows:
        _held_df = pd.DataFrame(_held_rows)
        _beta_sums = (
            _held_df.groupby(["underlying", "_bkt"])["beta_adj_notional"]
            .sum()
            .reset_index()
            .pivot(index="underlying", columns="_bkt", values="beta_adj_notional")
            .fillna(0)
            .reset_index()
        )
        for bc in ["bucket_1", "bucket_2", "bucket_4"]:
            if bc not in _beta_sums.columns:
                _beta_sums[bc] = 0.0
        _held_ratio_map: dict[str, dict[str, float]] = {}
        for _, row in _beta_sums.iterrows():
            r1, r2, r4 = _normalize_bucket_triple(
                float(row["bucket_1"]),
                float(row["bucket_2"]),
                float(row["bucket_4"]),
            )
            _held_ratio_map[row["underlying"]] = {"b1": r1, "b2": r2, "b4": r4}
    else:
        _held_ratio_map = {}

    # Base split map used for non-ETF rows (spot/fallback attribution).
    # - universe_beta: use ETF beta mix from screened universe
    # - held_exposure: held ETF exposure overrides universe fallback (default)
    _ratio_map = dict(_fallback_ratio_map)
    if split_method == "held_exposure":
        _ratio_map.update(_held_ratio_map)
    # If an underlying has no ETF mapping at all (orphan spot), attribute to bucket_1.
    _orphan_ratio = {"b1": 1.0, "b2": 0.0, "b4": 0.0}

    def _bucket_ratio_entry(underlying: str) -> tuple[dict[str, float], str]:
        u = str(underlying)
        if u in b4_underlyings and u not in _held_ratio_map and u not in _fallback_ratio_map:
            return {"b1": 0.0, "b2": 0.0, "b4": 1.0}, "b4_underlying_only"
        if split_method == "held_exposure" and u in _held_ratio_map:
            return _held_ratio_map[u], "held_exposure"
        if u in _fallback_ratio_map:
            return _fallback_ratio_map[u], "universe_beta_fallback"
        r1, r2, r4 = held_exposure_bucket124_weights(
            u,
            pos,
            etf_to_under,
            etf_to_beta_map,
            flow_low_beta_bucket3_syms=flow_low_beta_bucket3_syms,
            flow_short_syms=flow_short_set,
            b4_etf_syms=b4_etf_syms,
        )
        return {"b1": r1, "b2": r2, "b4": r4}, "held_exposure_eod"

    _realized_ratio_from_trades = build_underlying_realized_bucket_ratio_map(
        trade_events=trade_events,
        etf_to_under=etf_to_under,
        etf_to_beta=etf_to_beta_map,
        flow_low_beta_bucket3_syms=flow_low_beta_bucket3_syms,
        flow_short_syms=flow_short_set,
    )

    # Build a time-aware bucket ownership ledger for non-ETF rows.
    # This is event-driven from Trade rows:
    # - underlying BUY/SELL lots are bucket-tagged at trade time
    # - realized allocation uses the bucketed close amounts per trade
    # - unrealized/carry allocation uses bucket ownership state through time
    state_cols = [
        "run_date",
        "underlying",
        "qty_total",
        "qty_b1",
        "qty_b2",
        "qty_b4",
        "realized_b1",
        "realized_b2",
        "realized_b4",
        "ratio_b1",
        "ratio_b2",
        "ratio_b4",
        "ratio_source",
        "split_method",
    ]
    if bucket_state_path.exists():
        try:
            _state_hist = pd.read_csv(bucket_state_path)
        except Exception:
            _state_hist = pd.DataFrame(columns=state_cols)
    else:
        _state_hist = pd.DataFrame(columns=state_cols)
    for _c in state_cols:
        if _c not in _state_hist.columns:
            _state_hist[_c] = 0.0 if _c.startswith("qty_") or _c.startswith("ratio_") else ""
    _state_hist["run_date"] = _state_hist["run_date"].astype(str)
    _state_hist["underlying"] = _state_hist["underlying"].astype(str)
    for _c in [
        "qty_total",
        "qty_b1",
        "qty_b2",
        "qty_b4",
        "realized_b1",
        "realized_b2",
        "realized_b4",
        "ratio_b1",
        "ratio_b2",
        "ratio_b4",
    ]:
        _state_hist[_c] = pd.to_numeric(_state_hist[_c], errors="coerce").fillna(0.0)
    _prev_cutoff = ""
    _prev_cutoff_ymd = ""
    if not _state_hist.empty:
        _prior_dates = sorted(d for d in _state_hist["run_date"].astype(str).unique().tolist() if d < run_date)
        if _prior_dates:
            _prev_cutoff = _prior_dates[-1]
            _prev_cutoff_ymd = yyyymmdd_from_run_date(_prev_cutoff)

    _prev_state = (
        _state_hist[_state_hist["run_date"] < run_date]
        .sort_values("run_date")
        .groupby("underlying", as_index=False)
        .tail(1)
        .set_index("underlying")
    ) if not _state_hist.empty else pd.DataFrame(columns=state_cols).set_index(pd.Index([], dtype=str))

    _spot_qty = (
        pos.groupby("symbol", as_index=False)["position"].sum()
        .set_index("symbol")["position"]
        .to_dict()
    ) if not pos.empty else {}

    _state_rows: list[dict] = []
    _timing_rows: list[dict] = []
    _ratio_realized_map: dict[str, dict[str, float]] = {}
    _ratio_unrealized_map: dict[str, dict[str, float]] = {}
    _ratio_carry_map: dict[str, dict[str, float]] = {}
    _lot_components: dict[str, dict[str, dict[str, float]]] = {}
    _ledger_qty_map: dict[str, dict[str, float]] = {}
    _trade_underlyings: set[str] = set()
    if not trade_events.empty:
        for _, _tr in trade_events.iterrows():
            _tsym = str(_tr.get("symbol", "") or "")
            if _tsym in etf_to_under:
                continue
            _tu = canonical_symbol(str(_tr.get("underlyingSymbol", "") or _tsym))
            if _tu:
                _trade_underlyings.add(_tu)
    _prev_underlyings = (
        set(_prev_state.index.astype(str)) if not _prev_state.empty else set()
    )
    _all_underlyings = sorted(
        set(df["underlying"].dropna().astype(str))
        | _prev_underlyings
        | _trade_underlyings
    )
    _bucket_qty: dict[str, dict[str, float]] = defaultdict(
        lambda: {"bucket_1": 0.0, "bucket_2": 0.0, "bucket_4": 0.0}
    )
    _bucket_cost: dict[str, dict[str, float]] = defaultdict(
        lambda: {"bucket_1": 0.0, "bucket_2": 0.0, "bucket_4": 0.0}
    )
    _realized_amt: dict[str, dict[str, float]] = defaultdict(
        lambda: {"bucket_1": 0.0, "bucket_2": 0.0, "bucket_4": 0.0}
    )
    _etf_pos_qty: dict[str, float] = defaultdict(float)
    _state_series: dict[str, list[tuple[str, float, float, float]]] = defaultdict(list)
    _BUCKET_KEYS = ("bucket_1", "bucket_2", "bucket_4")

    # Optional manual opening-positions seed (pre-Flex-window holdings).
    _opening_seed: dict[str, dict[str, dict[str, float]]] = {}
    _opening_path = PROJECT_ROOT / "data" / "accounting" / "opening_positions.csv"
    if _opening_path.exists():
        try:
            _ops = pd.read_csv(_opening_path)
        except Exception as _exc:
            print(f"[ACCOUNTING] opening_positions.csv read failed: {_exc}")
            _ops = pd.DataFrame()
        if not _ops.empty:
            _ops["as_of_date"] = _ops["as_of_date"].astype(str)
            _ops["underlying"] = _ops["underlying"].astype(str).map(canonical_symbol)
            _ops = _ops[_ops["as_of_date"] < run_date].copy()
            _ops = (
                _ops.sort_values("as_of_date")
                .groupby("underlying", as_index=False)
                .tail(1)
            )
            for _, _row in _ops.iterrows():
                _u = str(_row["underlying"])
                if not _u:
                    continue
                _opening_seed[_u] = {
                    "bucket_1": {
                        "qty": float(_row.get("qty_b1", 0.0) or 0.0),
                        "cost": float(_row.get("cost_b1_base", 0.0) or 0.0)
                        if pd.notna(_row.get("cost_b1_base", float("nan")))
                        else float("nan"),
                    },
                    "bucket_2": {
                        "qty": float(_row.get("qty_b2", 0.0) or 0.0),
                        "cost": float(_row.get("cost_b2_base", 0.0) or 0.0)
                        if pd.notna(_row.get("cost_b2_base", float("nan")))
                        else float("nan"),
                    },
                    "bucket_4": {
                        "qty": float(_row.get("qty_b4", 0.0) or 0.0),
                        "cost": float(_row.get("cost_b4_base", 0.0) or 0.0)
                        if pd.notna(_row.get("cost_b4_base", float("nan")))
                        else float("nan"),
                    },
                }

    for _u in _all_underlyings:
        if _u in _prev_state.index:
            _pb1 = float(_prev_state.at[_u, "qty_b1"])
            _pb2 = float(_prev_state.at[_u, "qty_b2"])
            _pb4 = float(_prev_state.at[_u, "qty_b4"]) if "qty_b4" in _prev_state.columns else 0.0
            _bucket_qty[_u]["bucket_1"] = _pb1
            _bucket_qty[_u]["bucket_2"] = _pb2
            _bucket_qty[_u]["bucket_4"] = _pb4
            _realized_amt[_u]["bucket_1"] = float(
                _prev_state.at[_u, "realized_b1"] if "realized_b1" in _prev_state.columns else 0.0
            )
            _realized_amt[_u]["bucket_2"] = float(
                _prev_state.at[_u, "realized_b2"] if "realized_b2" in _prev_state.columns else 0.0
            )
            _realized_amt[_u]["bucket_4"] = float(
                _prev_state.at[_u, "realized_b4"] if "realized_b4" in _prev_state.columns else 0.0
            )
            _px_seed = 0.0
            _px_rows = pos[pos["symbol"] == _u]
            if not _px_rows.empty:
                _px_seed = float(_px_rows.iloc[0]["markPrice"]) * float(_px_rows.iloc[0]["fxRateToBase"])
            _bucket_cost[_u]["bucket_1"] = _pb1 * _px_seed
            _bucket_cost[_u]["bucket_2"] = _pb2 * _px_seed
            _bucket_cost[_u]["bucket_4"] = _pb4 * _px_seed
            if _prev_cutoff_ymd:
                _state_series[_u].append((_prev_cutoff_ymd, _pb1, _pb2, _pb4))
        elif _u in _opening_seed:
            _oseed = _opening_seed[_u]
            _pb1 = float(_oseed["bucket_1"]["qty"])
            _pb2 = float(_oseed["bucket_2"]["qty"])
            _pb4 = float(_oseed["bucket_4"]["qty"])
            _bucket_qty[_u]["bucket_1"] = _pb1
            _bucket_qty[_u]["bucket_2"] = _pb2
            _bucket_qty[_u]["bucket_4"] = _pb4
            _px_seed = 0.0
            _px_rows = pos[pos["symbol"] == _u]
            if not _px_rows.empty:
                _px_seed = float(_px_rows.iloc[0]["markPrice"]) * float(_px_rows.iloc[0]["fxRateToBase"])
            for _bk, _sk in (
                ("bucket_1", "bucket_1"),
                ("bucket_2", "bucket_2"),
                ("bucket_4", "bucket_4"),
            ):
                _cq = float(_oseed[_sk]["cost"])
                _bucket_cost[_u][_bk] = (
                    _cq if pd.notna(_cq) else float(_oseed[_sk]["qty"]) * _px_seed
                )

    def _sum_bucket_qty(_u: str, _b: str) -> float:
        return float(_bucket_qty[_u][_b])

    def _apply_trade_to_buckets(
        _u: str,
        _qty: float,
        _px: float,
        _w1: float,
        _w2: float,
        _w4: float,
        _ibkr_real: float,
    ) -> None:
        """Allocate a signed share trade across buckets using exact share counts."""
        _trade_model_real: dict[str, float] = {b: 0.0 for b in _BUCKET_KEYS}
        _weights = {"bucket_1": _w1, "bucket_2": _w2, "bucket_4": _w4}

        if _qty < -1e-12:
            _open_abs = {b: abs(_bucket_qty[_u][b]) for b in _BUCKET_KEYS}
            _open_tot = sum(_open_abs.values())
            if _open_tot > 1e-12:
                for _bk in _BUCKET_KEYS:
                    _dq = _qty * (_open_abs[_bk] / _open_tot)
                    if abs(_dq) <= 1e-12:
                        continue
                    _nq, _nc, _r = _apply_signed_bucket_trade(
                        _bucket_qty[_u][_bk], _bucket_cost[_u][_bk], _dq, _px
                    )
                    _bucket_qty[_u][_bk] = _nq
                    _bucket_cost[_u][_bk] = _nc
                    _trade_model_real[_bk] += _r
            else:
                for _bk, _w in _weights.items():
                    _dq = _qty * _w
                    if abs(_dq) <= 1e-12:
                        continue
                    _nq, _nc, _r = _apply_signed_bucket_trade(
                        _bucket_qty[_u][_bk], _bucket_cost[_u][_bk], _dq, _px
                    )
                    _bucket_qty[_u][_bk] = _nq
                    _bucket_cost[_u][_bk] = _nc
                    _trade_model_real[_bk] += _r
        else:
            for _bk, _w in _weights.items():
                _dq = _qty * _w
                if abs(_dq) <= 1e-12:
                    continue
                _nq, _nc, _r = _apply_signed_bucket_trade(
                    _bucket_qty[_u][_bk], _bucket_cost[_u][_bk], _dq, _px
                )
                _bucket_qty[_u][_bk] = _nq
                _bucket_cost[_u][_bk] = _nc
                _trade_model_real[_bk] += _r

        if abs(_ibkr_real) > 1e-12:
            _model_sum = sum(abs(v) for v in _trade_model_real.values())
            if _model_sum > 1e-12:
                for _bk in _BUCKET_KEYS:
                    _realized_amt[_u][_bk] += _ibkr_real * (abs(_trade_model_real[_bk]) / _model_sum)
            else:
                for _bk, _w in _weights.items():
                    _realized_amt[_u][_bk] += _ibkr_real * _w

    _ev = trade_events.copy()
    if not _ev.empty:
        _ev["date"] = _ev["dateTime"].astype(str).str.slice(0, 8)
        _target = yyyymmdd_from_run_date(run_date)
        _ev = _ev[_ev["date"] <= _target].copy()
        if _prev_cutoff_ymd:
            _ev = _ev[_ev["date"] > _prev_cutoff_ymd].copy()
        _ev = _ev.sort_values("dateTime").reset_index(drop=True)

    _minute_demand: dict[tuple[str, str], tuple[float, float, float]] = {}
    if not _ev.empty:
        _tmp = _ev[_ev["symbol"].isin(etf_to_under.keys())].copy()
        if not _tmp.empty:
            _tmp["minute_key"] = _tmp["dateTime"].astype(str).str.slice(0, 13)
            _tmp["under"] = _tmp["symbol"].map(etf_to_under)
            _tmp["beta"] = _tmp["symbol"].map(etf_to_beta_map).astype(float)
            _tmp["w"] = _tmp["quantity"].abs() * _tmp["tradePrice_base"].abs() * _tmp["beta"].abs()
            _tmp = _tmp[~_tmp["symbol"].isin(flow_low_beta_bucket3_syms)].copy()
            if not _tmp.empty:
                _tmp["_bkt"] = np.select(
                    [
                        _tmp["beta"].lt(0) & (~_tmp["symbol"].isin(flow_short_set)),
                        _tmp["beta"].gt(1.5),
                        _tmp["beta"].gt(0),
                    ],
                    ["bucket_4", "bucket_1", "bucket_2"],
                    default="",
                )
                _tmp = _tmp[_tmp["_bkt"].astype(bool)].copy()
                _agg = _tmp.groupby(["minute_key", "under", "_bkt"], as_index=False)["w"].sum()
                for (mk, uu), g in _agg.groupby(["minute_key", "under"]):
                    b1 = float(g.loc[g["_bkt"] == "bucket_1", "w"].sum())
                    b2 = float(g.loc[g["_bkt"] == "bucket_2", "w"].sum())
                    b4 = float(g.loc[g["_bkt"] == "bucket_4", "w"].sum())
                    _minute_demand[(str(mk), str(uu))] = (b1, b2, b4)

    def _weights_for_underlying_trade(_u: str, _ref: str, _dt: str) -> tuple[float, float, float]:
        _hint = _bucket_hint_from_order_reference(
            _ref,
            etf_to_beta_map,
            flow_low_beta_bucket3_syms=flow_low_beta_bucket3_syms,
        )
        if _hint == "bucket_1":
            return 1.0, 0.0, 0.0
        if _hint == "bucket_2":
            return 0.0, 1.0, 0.0
        if _hint == "bucket_4":
            return 0.0, 0.0, 1.0
        _mk = str(_dt)[:13] if _dt else ""
        _md = _minute_demand.get((_mk, _u))
        if _md is not None:
            _m1, _m2, _m4 = _md
            if (_m1 + _m2 + _m4) > 1e-12:
                return _normalize_bucket_triple(_m1, _m2, _m4)
        _w1 = 0.0
        _w2 = 0.0
        _w4 = 0.0
        for _es, _eu in etf_to_under.items():
            if _eu != _u:
                continue
            _b = float(etf_to_beta_map.get(_es, 0.0))
            if _es in flow_low_beta_bucket3_syms:
                continue
            _q = abs(float(_etf_pos_qty.get(_es, 0.0)))
            if _q <= 1e-12:
                continue
            _w = _q * abs(_b)
            if _b < 0 and _es not in flow_short_set:
                _w4 += _w
            elif _b > 1.5:
                _w1 += _w
            elif _b > 0:
                _w2 += _w
        if (_w1 + _w2 + _w4) <= 1e-12:
            _oq1 = abs(_sum_bucket_qty(_u, "bucket_1"))
            _oq2 = abs(_sum_bucket_qty(_u, "bucket_2"))
            _oq4 = abs(_sum_bucket_qty(_u, "bucket_4"))
            _w1, _w2, _w4 = _oq1, _oq2, _oq4
        if (_w1 + _w2 + _w4) <= 1e-12:
            return held_exposure_bucket124_weights(
                _u,
                pos,
                etf_to_under,
                etf_to_beta_map,
                flow_low_beta_bucket3_syms=flow_low_beta_bucket3_syms,
                flow_short_syms=flow_short_set,
                b4_etf_syms=b4_etf_syms,
            )
        return _normalize_bucket_triple(_w1, _w2, _w4)

    if not _ev.empty:
        _cur_date = ""
        _touched: set[str] = set()
        for _, _r in _ev.iterrows():
            _d = str(_r.get("date", "") or "")
            if not _d:
                continue
            if _cur_date and _d != _cur_date:
                for _u in _touched:
                    _state_series[_u].append(
                        (
                            _cur_date,
                            _sum_bucket_qty(_u, "bucket_1"),
                            _sum_bucket_qty(_u, "bucket_2"),
                            _sum_bucket_qty(_u, "bucket_4"),
                        )
                    )
                _touched = set()
            _cur_date = _d

            _sym = str(_r.get("symbol", "") or "")
            _qty = float(_r.get("quantity", 0.0) or 0.0)
            _real = float(_r.get("fifoPnlRealized_base", 0.0) or 0.0)
            _px = float(_r.get("tradePrice_base", 0.0) or 0.0)
            _oref = str(_r.get("orderReference", "") or "")
            _dt = str(_r.get("dateTime", "") or "")

            if _sym in etf_to_under:
                _etf_pos_qty[_sym] += _qty
                continue

            _u = canonical_symbol(str(_r.get("underlyingSymbol", "") or _sym))
            if not _u:
                _u = _sym
            _w1, _w2, _w4 = _weights_for_underlying_trade(_u, _oref, _dt)
            _touched.add(_u)
            _apply_trade_to_buckets(_u, _qty, _px, _w1, _w2, _w4, _real)

        if _cur_date:
            for _u in _touched:
                _state_series[_u].append(
                    (
                        _cur_date,
                        _sum_bucket_qty(_u, "bucket_1"),
                        _sum_bucket_qty(_u, "bucket_2"),
                        _sum_bucket_qty(_u, "bucket_4"),
                    )
                )

    def _ratio_at_date(_u: str, _d: str) -> tuple[float, float, float]:
        _series = _state_series.get(_u, [])
        for _entry in reversed(_series):
            if len(_entry) >= 4:
                _sd, _q1, _q2, _q4 = _entry[0], _entry[1], _entry[2], _entry[3]
            else:
                _sd, _q1, _q2 = _entry[0], _entry[1], _entry[2]
                _q4 = 0.0
            if yyyymmdd_normalize(_sd) <= yyyymmdd_normalize(_d):
                return _normalize_bucket_triple(abs(_q1), abs(_q2), abs(_q4))
        _r, _ = _bucket_ratio_entry(_u)
        return _normalize_bucket_triple(_r.get("b1", 1.0), _r.get("b2", 0.0), _r.get("b4", 0.0))

    _spot_carry_cols = {
        "motion_pnl",
        "motion_dividends",
        "motion_dividends",
        "dividends",
        "withholding_tax",
        "pil_dividends",
        "borrow_fees",
        "short_credit_interest",
        "other_fees",
        "bond_interest",
    }
    _carry_by_col: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"bucket_1": 0.0, "bucket_2": 0.0, "bucket_4": 0.0})
    )
    _carry_cats = {
        "motion_pnl",
        "motion_dividends",
        "dividends",
        "withholding_tax",
        "pil_dividends",
        "short_credit_interest",
        "other_fees",
        "bond_interest",
    }
    _cash_ev = cash[cash["category"].isin(_carry_cats)][["date", "symbol", "category", "amount_base"]].copy()
    _cash_ev["date"] = _cash_ev["date"].astype(str)
    _borrow_ev = pd.DataFrame(columns=["date", "symbol", "category", "amount_base"])
    if isinstance(borrow_fee_events, pd.DataFrame) and not borrow_fee_events.empty:
        _borrow_ev = borrow_fee_events[["date", "symbol", "borrowFee_base"]].rename(columns={"borrowFee_base": "amount_base"}).copy()
        _borrow_ev["date"] = _borrow_ev["date"].astype(str)
        _borrow_ev["category"] = "borrow_fees"
    _carry_ev = pd.concat([_cash_ev, _borrow_ev], ignore_index=True)
    for _, _r in _carry_ev.iterrows():
        _sym = canonical_symbol(str(_r.get("symbol", "") or ""))
        if not _sym or _sym in etf_to_beta_map:
            continue
        _d = str(_r.get("date", "") or "")
        _amt = float(_r.get("amount_base", 0.0) or 0.0)
        if abs(_amt) <= 1e-12 or not _d:
            continue
        _u = _sym
        _col = str(_r.get("category", "") or "")
        if _col not in _spot_carry_cols:
            _col = "other_fees"
        _cr1, _cr2, _cr4 = _ratio_at_date(_u, yyyymmdd_normalize(_d))
        _carry_by_col[_u][_col]["bucket_1"] += _amt * _cr1
        _carry_by_col[_u][_col]["bucket_2"] += _amt * _cr2
        _carry_by_col[_u][_col]["bucket_4"] += _amt * _cr4

    for _u in _all_underlyings:
        _ratio, _src = _bucket_ratio_entry(_u)
        _r1, _r2, _r4 = _normalize_bucket_triple(
            _ratio.get("b1", 0.0), _ratio.get("b2", 0.0), _ratio.get("b4", 0.0)
        )
        _cur_b1 = _sum_bucket_qty(_u, "bucket_1")
        _cur_b2 = _sum_bucket_qty(_u, "bucket_2")
        _cur_b4 = _sum_bucket_qty(_u, "bucket_4")
        _qty_total = _cur_b1 + _cur_b2 + _cur_b4

        if _u in _prev_state.index:
            _prev_b1 = float(_prev_state.at[_u, "qty_b1"])
            _prev_b2 = float(_prev_state.at[_u, "qty_b2"])
            _prev_b4 = float(_prev_state.at[_u, "qty_b4"]) if "qty_b4" in _prev_state.columns else 0.0
        else:
            _prev_b1 = 0.0
            _prev_b2 = 0.0
            _prev_b4 = 0.0

        _rw1 = abs(float(_realized_amt[_u]["bucket_1"]))
        _rw2 = abs(float(_realized_amt[_u]["bucket_2"]))
        _rw4 = abs(float(_realized_amt[_u]["bucket_4"]))
        if (_rw1 + _rw2 + _rw4) > 1e-12:
            _rz1, _rz2, _rz4 = _normalize_bucket_triple(_rw1, _rw2, _rw4)
        elif _u in _realized_ratio_from_trades:
            _rz1, _rz2, _rz4 = _normalize_bucket_triple(
                _realized_ratio_from_trades[_u].get("b1", _r1),
                _realized_ratio_from_trades[_u].get("b2", _r2),
                _realized_ratio_from_trades[_u].get("b4", _r4),
            )
        else:
            _rz1, _rz2, _rz4 = _r1, _r2, _r4
        _px_now = 0.0
        _px_rows = pos[pos["symbol"] == _u]
        if not _px_rows.empty:
            _px_now = float(_px_rows.iloc[0]["markPrice"]) * float(_px_rows.iloc[0]["fxRateToBase"])
        _um1 = (_cur_b1 * _px_now) - float(_bucket_cost[_u]["bucket_1"])
        _um2 = (_cur_b2 * _px_now) - float(_bucket_cost[_u]["bucket_2"])
        _um4 = (_cur_b4 * _px_now) - float(_bucket_cost[_u]["bucket_4"])
        if (abs(_um1) + abs(_um2) + abs(_um4)) > 1e-12:
            _ru1, _ru2, _ru4 = _normalize_bucket_triple(abs(_um1), abs(_um2), abs(_um4))
        else:
            _ru1, _ru2, _ru4 = _normalize_bucket_triple(abs(_cur_b1), abs(_cur_b2), abs(_cur_b4))
        _ca1 = sum(float(_carry_by_col[_u][c]["bucket_1"]) for c in _carry_by_col.get(_u, {}))
        _ca2 = sum(float(_carry_by_col[_u][c]["bucket_2"]) for c in _carry_by_col.get(_u, {}))
        _ca4 = sum(float(_carry_by_col[_u][c]["bucket_4"]) for c in _carry_by_col.get(_u, {}))
        if (abs(_ca1) + abs(_ca2) + abs(_ca4)) > 1e-12:
            _rc1, _rc2, _rc4 = _normalize_bucket_triple(abs(_ca1), abs(_ca2), abs(_ca4))
        else:
            _rc1, _rc2, _rc4 = _ru1, _ru2, _ru4

        _ratio_realized_map[_u] = {"b1": _rz1, "b2": _rz2, "b4": _rz4}
        _ratio_unrealized_map[_u] = {"b1": _ru1, "b2": _ru2, "b4": _ru4}
        _ratio_carry_map[_u] = {"b1": _rc1, "b2": _rc2, "b4": _rc4}

        _rz_real = float(_realized_amt[_u]["bucket_1"])
        _rz_real2 = float(_realized_amt[_u]["bucket_2"])
        _rz_real4 = float(_realized_amt[_u]["bucket_4"])
        _um_by_bkt = {"bucket_1": _um1, "bucket_2": _um2, "bucket_4": _um4}
        _lot_components[_u] = {}
        for _bk in _BUCKET_KEYS:
            _lot_components[_u][_bk] = {
                "realized_pnl": float(_realized_amt[_u][_bk]),
                "unrealized_pnl": float(_um_by_bkt[_bk]),
            }
            for _cc in _spot_carry_cols:
                _lot_components[_u][_bk][_cc] = float(
                    _carry_by_col.get(_u, {}).get(_cc, {}).get(_bk, 0.0)
                )

        _ibkr_qty = float(_spot_qty.get(_u, 0.0))
        _orphan_qty = _ibkr_qty - _qty_total
        _ledger_qty_map[_u] = dict(_bucket_qty[_u])

        _state_rows.append(
            {
                "run_date": run_date,
                "underlying": _u,
                "qty_total": _qty_total,
                "qty_b1": _cur_b1,
                "qty_b2": _cur_b2,
                "qty_b4": _cur_b4,
                "realized_b1": _rz_real,
                "realized_b2": _rz_real2,
                "realized_b4": _rz_real4,
                "ratio_b1": _r1,
                "ratio_b2": _r2,
                "ratio_b4": _r4,
                "ratio_source": _src,
                "split_method": split_method,
            }
        )
        _timing_rows.append(
            {
                "run_date": run_date,
                "underlying": _u,
                "ratio_source": _src,
                "prev_qty_b1": _prev_b1,
                "prev_qty_b2": _prev_b2,
                "prev_qty_b4": _prev_b4,
                "curr_qty_b1": _cur_b1,
                "curr_qty_b2": _cur_b2,
                "curr_qty_b4": _cur_b4,
                "ibkr_qty": _ibkr_qty,
                "orphan_qty": _orphan_qty,
                "ratio_current_b1": _r1,
                "ratio_current_b2": _r2,
                "ratio_current_b4": _r4,
                "ratio_realized_b1": _rz1,
                "ratio_realized_b2": _rz2,
                "ratio_realized_b4": _rz4,
                "ratio_unrealized_b1": _ru1,
                "ratio_unrealized_b2": _ru2,
                "ratio_unrealized_b4": _ru4,
                "ratio_carry_b1": _rc1,
                "ratio_carry_b2": _rc2,
                "ratio_carry_b4": _rc4,
            }
        )

    _state_today_df = pd.DataFrame(_state_rows, columns=state_cols)
    _state_merged = pd.concat([_state_hist[state_cols], _state_today_df], ignore_index=True)
    _state_merged = _state_merged.drop_duplicates(subset=["run_date", "underlying"], keep="last")
    _state_merged = _state_merged.sort_values(["run_date", "underlying"]).reset_index(drop=True)
    bucket_state_path.parent.mkdir(parents=True, exist_ok=True)
    _state_merged.to_csv(bucket_state_path, index=False)
    _timing_df = pd.DataFrame(_timing_rows)
    _timing_df.to_csv(outdir / "bucket_timing_state.csv", index=False)
    _lot_cost_rows = []
    for _u in _all_underlyings:
        _lot_cost_rows.append(
            {
                "run_date": run_date,
                "underlying": _u,
                "qty_b1": _sum_bucket_qty(_u, "bucket_1"),
                "qty_b2": _sum_bucket_qty(_u, "bucket_2"),
                "qty_b4": _sum_bucket_qty(_u, "bucket_4"),
                "cost_b1_base": float(_bucket_cost[_u]["bucket_1"]),
                "cost_b2_base": float(_bucket_cost[_u]["bucket_2"]),
                "cost_b4_base": float(_bucket_cost[_u]["bucket_4"]),
            }
        )
    pd.DataFrame(_lot_cost_rows).to_csv(outdir / "bucket_lot_cost_state.csv", index=False)

    _recon_rows: list[dict] = []
    for _u in _all_underlyings:
        _lb1 = float(_bucket_qty[_u]["bucket_1"])
        _lb2 = float(_bucket_qty[_u]["bucket_2"])
        _lb4 = float(_bucket_qty[_u]["bucket_4"])
        _ltot = _lb1 + _lb2 + _lb4
        _ibkr_qty = float(_spot_qty.get(_u, 0.0))
        _recon_rows.append(
            {
                "run_date": run_date,
                "underlying": _u,
                "qty_b1": _lb1,
                "qty_b2": _lb2,
                "qty_b4": _lb4,
                "ledger_total": _ltot,
                "ibkr_position": _ibkr_qty,
                "diff_ledger_minus_ibkr": _ltot - _ibkr_qty,
            }
        )
    if _recon_rows:
        _recon_df = pd.DataFrame(_recon_rows)
        _recon_df = _recon_df.sort_values(
            "diff_ledger_minus_ibkr", key=lambda s: s.abs(), ascending=False
        ).reset_index(drop=True)
        _recon_df.to_csv(outdir / "bucket_state_reconciliation.csv", index=False)
        _bad = _recon_df[_recon_df["diff_ledger_minus_ibkr"].abs() > 1.0]
        if not _bad.empty:
            print(
                f"[ACCOUNTING] reconciliation: {len(_bad)} underlying(s) drift > 1 share "
                f"(worst: {_bad.iloc[0]['underlying']} diff="
                f"{_bad.iloc[0]['diff_ledger_minus_ibkr']:+.1f}); "
                f"see {outdir / 'bucket_state_reconciliation.csv'}"
            )

    # Assign buckets:
    # - ETFs are ALWAYS bucketed by their own beta (independent of held status)
    # - Inverse ETFs (beta < 0):
    #     * flow program shorts -> bucket_3
    #     * all others          -> bucket_4
    # - Spot positions (non-ETF rows) split using ratio map by underlying
    df["bucket"] = ""
    neg_etf = is_etf & (sym_beta < 0)
    neg_flow = neg_etf & df["symbol"].isin(flow_short_set)
    neg_non_flow = neg_etf & (~df["symbol"].isin(flow_short_set))
    df.loc[neg_flow, "bucket"] = "bucket_3"
    df.loc[neg_non_flow, "bucket"] = "bucket_4"
    df.loc[is_etf & (sym_beta > 1.5), "bucket"] = "bucket_1"
    df.loc[is_etf & (sym_beta > 0) & (sym_beta <= 1.5), "bucket"] = "bucket_2"
    # Flow sleeve low-beta ETFs behave as inverse sleeve for attribution:
    # keep their ETF PnL/exposure in bucket_3 and remove their influence from bucket_2 splits.
    df.loc[is_etf & df["symbol"].isin(flow_low_beta_bucket3_syms), "bucket"] = "bucket_3"

    # Only non-ETF rows split by lot ledger (exact shares); current_only uses held ratios.
    needs_split = (~is_etf) & (df["bucket"] == "")
    fixed_rows = df[~needs_split].copy()
    split_source = df[needs_split].copy()
    component_cols = [c for c in base_cols if c != "total_pnl"]
    carry_cols = {
        "dividends",
        "withholding_tax",
        "pil_dividends",
        "borrow_fees",
        "short_credit_interest",
        "other_fees",
        "bond_interest",
    }
    if not split_source.empty:
        _ratio_meta = split_source["underlying"].astype(str).apply(_bucket_ratio_entry)
        split_source["_ratio_source"] = _ratio_meta.map(lambda x: x[1])
        split_source["_ratio_b1_current"] = _ratio_meta.map(lambda x: float(x[0]["b1"]))
        split_source["_ratio_b2_current"] = _ratio_meta.map(lambda x: float(x[0]["b2"]))
        split_source["_ratio_b4_current"] = _ratio_meta.map(lambda x: float(x[0].get("b4", 0.0)))
        _default_r = {"b1": 1.0, "b2": 0.0, "b4": 0.0}
        split_source["_ratio_b1_realized"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_realized_map.get(u, _ratio_carry_map.get(u, _default_r))["b1"]
        )
        split_source["_ratio_b2_realized"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_realized_map.get(u, _ratio_carry_map.get(u, _default_r))["b2"]
        )
        split_source["_ratio_b4_realized"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_realized_map.get(u, _ratio_carry_map.get(u, _default_r)).get("b4", 0.0)
        )
        split_source["_ratio_b1_unrealized"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_unrealized_map.get(u, _ratio_carry_map.get(u, _default_r))["b1"]
        )
        split_source["_ratio_b2_unrealized"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_unrealized_map.get(u, _ratio_carry_map.get(u, _default_r))["b2"]
        )
        split_source["_ratio_b4_unrealized"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_unrealized_map.get(u, _ratio_carry_map.get(u, _default_r)).get("b4", 0.0)
        )
        split_source["_ratio_b1_carry"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_carry_map.get(u, _default_r)["b1"]
        )
        split_source["_ratio_b2_carry"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_carry_map.get(u, _default_r)["b2"]
        )
        split_source["_ratio_b4_carry"] = split_source["underlying"].astype(str).map(
            lambda u: _ratio_carry_map.get(u, _default_r).get("b4", 0.0)
        )

    _lot_exact_cols = {"realized_pnl", "unrealized_pnl"} | carry_cols
    # The lot-timed override writes the per-underlying lot value onto
    # exactly ONE row per (underlying, bucket). When multiple symbols
    # share an underlying (e.g. GXRP / XRP / XRPZ -> underlying XRPZ),
    # writing the same value onto every row would multiply each bucket
    # sum by N and break conservation; writing it once keeps the
    # bucket-level sum equal to ``_lot_components[u][bucket][col]``.
    # Map each underlying to the index value of its first row in the
    # original ``split_source``. We use the actual DataFrame index (not a
    # reset positional one) so the comparison ``part.index == first``
    # works on identical labels in the copies below.
    if not split_source.empty:
        _ufirst_idx = (
            split_source.assign(_u=split_source["underlying"].astype(str))
            .groupby("_u")
            .apply(lambda g: g.index[0])
            .to_dict()
        )
    else:
        _ufirst_idx = {}
    split_parts: list[pd.DataFrame] = []
    for bkt_label in ["bucket_1", "bucket_2", "bucket_4"]:
        part = split_source.copy()
        part["bucket"] = bkt_label
        _rk = {"bucket_1": "b1", "bucket_2": "b2", "bucket_4": "b4"}[bkt_label]
        for col in component_cols:
            if component_split_method == "lot_timed" and col in _lot_exact_cols:
                _u_series = part["underlying"].astype(str)
                # All rows default to 0; the canonical first row per
                # underlying receives the full per-underlying lot value.
                part[col] = 0.0
                _is_first = part.index.to_series().eq(
                    _u_series.map(_ufirst_idx)
                )
                _vals = _u_series.map(
                    lambda u, _bk=bkt_label, _c=col: float(
                        _lot_components.get(u, {}).get(_bk, {}).get(_c, 0.0)
                    )
                )
                part.loc[_is_first, col] = _vals[_is_first]
            else:
                if component_split_method == "current_only":
                    _ratio_col = f"_ratio_{_rk}_current"
                elif col == "realized_pnl":
                    _ratio_col = f"_ratio_{_rk}_realized"
                elif col == "unrealized_pnl":
                    _ratio_col = f"_ratio_{_rk}_unrealized"
                elif col in carry_cols:
                    _ratio_col = f"_ratio_{_rk}_carry"
                else:
                    _ratio_col = f"_ratio_{_rk}_current"
                part[col] = part[col] * part[_ratio_col]
        part["total_pnl"] = part[component_cols].sum(axis=1)
        # Note: do NOT drop zero-component rows here. The lot-timed residual
        # fixup below needs the canonical bucket_1 row (which may legitimately
        # start at zero for an underlying whose lot ledger has no bucket_1
        # component) to attribute the IBKR-vs-lot residual onto. Dropping
        # zero rows is deferred until after the fixup.
        split_parts.append(part)

    # ── Lot-exact residual fixup ─────────────────────────────────────────
    # The lot-timed split copies per-bucket values out of ``_lot_components``
    # (built from our FIFO replay). When the FIFO ordering or cost basis
    # we computed differs from what IBKR's Flex statement reports as
    # ``fifoPnlRealized`` / ``fifoPnlUnrealized`` for the same underlying,
    # the per-bucket sum will not equal the IBKR-reported total. We close
    # the gap by adding the residual ``(ibkr_total - lot_sum)`` to the
    # bucket_1 row of each underlying. This preserves global conservation
    # without distorting per-bucket attribution beyond what IBKR's own
    # FIFO would also have done.
    if component_split_method == "lot_timed" and split_parts and not split_source.empty:
        # The lot-timed override above wrote ``_lot_components[u][b][col]``
        # onto exactly one row per (underlying, bucket). The remaining
        # gap vs the IBKR-reported totals is ``ibkr_total(u, col) -
        # sum_buckets(_lot_components[u][b][col])`` (or just
        # ``ibkr_total(u, col)`` when the underlying isn't in the lot
        # ledger). We attribute that residual to bucket_1's canonical
        # first row so the per-underlying / per-column sum equals the
        # IBKR-reported total.
        for _u in split_source["underlying"].astype(str).unique():
            _u_rows = split_source[split_source["underlying"].astype(str) == _u]
            if _u_rows.empty:
                continue
            _lot_sum_one_by_col = {
                _col: (
                    sum(
                        float(_lot_components.get(_u, {}).get(_bk, {}).get(_col, 0.0))
                        for _bk in _BUCKET_KEYS
                    )
                    if _u in _lot_components
                    else 0.0
                )
                for _col in _lot_exact_cols
                if _col in component_cols
            }
            for _col, _lot_sum in _lot_sum_one_by_col.items():
                _ibkr_val = float(_u_rows[_col].sum())
                _resid = _ibkr_val - _lot_sum
                if abs(_resid) <= 1e-6:
                    continue
                for _part in split_parts:
                    if _part.empty:
                        continue
                    _mask = (_part["bucket"] == "bucket_1") & (
                        _part["underlying"].astype(str) == _u
                    )
                    if not _mask.any():
                        continue
                    _first_idx = _part.index[_mask][0]
                    _part.at[_first_idx, _col] = (
                        float(_part.at[_first_idx, _col]) + _resid
                    )
                    _part.at[_first_idx, "total_pnl"] = float(
                        _part.loc[_first_idx, component_cols].sum()
                    )
                    break

    # Drop rows where all PnL components net to zero AFTER residual fixup.
    # Deferring this filter (rather than doing it inside the per-bucket
    # build loop) is required so that the lot-timed residual fixup above
    # can attribute the IBKR-vs-lot gap onto the canonical bucket_1 row
    # of each underlying even when that row started at zero.
    split_parts = [
        _p[_p[component_cols].abs().sum(axis=1) > 0].copy()
        for _p in split_parts
        if not _p.empty
    ]

    df = pd.concat([fixed_rows] + split_parts, ignore_index=True)
    _global_pre = df_pre_bucket[base_cols].sum(numeric_only=True)
    _global_post = df[base_cols].sum(numeric_only=True)
    _max_diff = (
        float((_global_post - _global_pre).abs().max())
        if not _global_post.empty
        else 0.0
    )
    # Tolerance: 1 cent. The lot-timed residual fixup sums many per-bucket
    # floats and then back-fills the canonical bucket_1 row of each
    # underlying, so per-column rounding can accumulate a few µ-dollars
    # across hundreds of symbols. Anything beyond 1 cent indicates a real
    # attribution bug (the kind that previously caused $1k+ breakages
    # when multiple symbols shared an underlying).
    if _max_diff > 1e-2:
        # Emit a per-column + worst-symbol breakdown before raising so
        # future conservation breakages don't require ad-hoc debugging.
        try:
            _per_col = (_global_post - _global_pre).round(6)
            _per_col = _per_col[_per_col.abs() > 1e-6]
            print(
                "[ACCOUNTING] conservation diff by column (post-pre):\n"
                + _per_col.to_string()
            )
            _worst_col = _per_col.abs().idxmax() if not _per_col.empty else None
            if _worst_col is not None and _worst_col in df_pre_bucket.columns:
                _pre_by_s = df_pre_bucket.groupby("symbol")[_worst_col].sum()
                _post_by_s = df.groupby("symbol")[_worst_col].sum()
                _diff_by_s = (_post_by_s - _pre_by_s).reindex(
                    _pre_by_s.index.union(_post_by_s.index), fill_value=0.0
                )
                _diff_by_s = _diff_by_s[_diff_by_s.abs() > 1e-6].sort_values(
                    key=lambda s: s.abs(), ascending=False
                )
                print(
                    f"[ACCOUNTING] worst column '{_worst_col}' top-10 symbols (post-pre):\n"
                    + _diff_by_s.head(10).round(2).to_string()
                )
                _pre_by_u = df_pre_bucket.groupby("underlying")[_worst_col].sum()
                _post_by_u = df.groupby("underlying")[_worst_col].sum()
                _diff_by_u = (_post_by_u - _pre_by_u).reindex(
                    _pre_by_u.index.union(_post_by_u.index), fill_value=0.0
                )
                _diff_by_u = _diff_by_u[_diff_by_u.abs() > 1e-6].sort_values(
                    key=lambda s: s.abs(), ascending=False
                )
                print(
                    f"[ACCOUNTING] worst column '{_worst_col}' top-10 underlyings (post-pre):\n"
                    + _diff_by_u.head(10).round(2).to_string()
                )
        except Exception as _diag_exc:
            print(f"[ACCOUNTING] conservation diagnostic failed: {_diag_exc}")
        raise AssertionError(
            "Bucket split conservation failed. "
            f"max_abs_diff={_max_diff:.8f}"
        )

    # ── PnL outputs ──
    pnl_by_symbol = (
        df[["symbol", "underlying", "bucket", "pair", "description"] + base_cols]
        .sort_values("total_pnl", ascending=False)
    )

    # pnl_by_underlying: combined bucket_1 + bucket_2 + bucket_4 (stock sleeves)
    df_b124 = df[df["bucket"].isin(["bucket_1", "bucket_2", "bucket_4"])]
    _under_agg = df_b124.groupby("underlying", as_index=False)[base_cols].sum()
    _under_syms = (
        df_b124.groupby("underlying")["symbol"]
        .apply(lambda s: ", ".join(sorted(set(s.astype(str)))))
        .reset_index()
        .rename(columns={"symbol": "symbols"})
    )
    pnl_by_underlying = (
        _under_agg.merge(_under_syms, on="underlying")
        [["underlying", "symbols"] + base_cols]
        .sort_values("total_pnl", ascending=False)
    )
    pnl_by_underlying_b12 = (
        df[df["bucket"].isin(["bucket_1", "bucket_2"])]
        .groupby("underlying", as_index=False)[base_cols]
        .sum()
        .sort_values("total_pnl", ascending=False)
    )

    # Bucket 1 PnL by underlying (levered ETFs + pro-rata spot)
    df_b1 = df[df["bucket"] == "bucket_1"]
    if not df_b1.empty:
        _b1_agg = df_b1.groupby("underlying", as_index=False)[base_cols].sum()
        _b1_syms = (
            df_b1.groupby("underlying")["symbol"]
            .apply(lambda s: ", ".join(sorted(set(s.astype(str)))))
            .reset_index()
            .rename(columns={"symbol": "symbols"})
        )
        pnl_bucket_1 = (
            _b1_agg.merge(_b1_syms, on="underlying")
            [["underlying", "symbols"] + base_cols]
            .sort_values("total_pnl", ascending=False)
        )
    else:
        pnl_bucket_1 = pd.DataFrame(columns=["underlying", "symbols"] + base_cols)

    # Bucket 2 PnL by underlying (standard ETFs + pro-rata spot)
    df_b2 = df[df["bucket"] == "bucket_2"]
    if not df_b2.empty:
        _b2_agg = df_b2.groupby("underlying", as_index=False)[base_cols].sum()
        _b2_syms = (
            df_b2.groupby("underlying")["symbol"]
            .apply(lambda s: ", ".join(sorted(set(s.astype(str)))))
            .reset_index()
            .rename(columns={"symbol": "symbols"})
        )
        pnl_bucket_2 = (
            _b2_agg.merge(_b2_syms, on="underlying")
            [["underlying", "symbols"] + base_cols]
            .sort_values("total_pnl", ascending=False)
        )
    else:
        pnl_bucket_2 = pd.DataFrame(columns=["underlying", "symbols"] + base_cols)

    # Bucket 3 PnL by symbol (inverse/hedge ETFs — keyed by ETF symbol, not underlying)
    pnl_bucket_3 = (
        df[df["bucket"] == "bucket_3"][["symbol", "description"] + base_cols]
        .sort_values("total_pnl", ascending=False)
    )
    df_b4 = df[df["bucket"] == "bucket_4"]
    if not df_b4.empty:
        _b4_agg = df_b4.groupby("underlying", as_index=False)[base_cols].sum()
        _b4_syms = (
            df_b4.groupby("underlying")["symbol"]
            .apply(lambda s: ", ".join(sorted(set(s.astype(str)))))
            .reset_index()
            .rename(columns={"symbol": "symbols"})
        )
        pnl_bucket_4 = (
            _b4_agg.merge(_b4_syms, on="underlying")
            [["underlying", "symbols"] + base_cols]
            .sort_values("total_pnl", ascending=False)
        )
    else:
        pnl_bucket_4 = pd.DataFrame(columns=["underlying", "symbols"] + base_cols)

    pnl_bucket_4_by_symbol = (
        df[df["bucket"] == "bucket_4"][["symbol", "underlying", "description"] + base_cols]
        .sort_values("total_pnl", ascending=False)
    )
    _pair_rows: list[dict] = []
    if not b4_registry.empty:
        for _, pr in b4_registry.iterrows():
            etf = str(pr["etf"])
            under = str(pr["underlying"])
            etf_pnl = df[(df["symbol"] == etf) & (df["bucket"] == "bucket_4")]
            und_pnl = df[(df["symbol"] == under) & (df["bucket"] == "bucket_4")]
            row = {"underlying": under, "etf": etf, "beta": float(pr["beta"])}
            for c in base_cols:
                row[c] = float(etf_pnl[c].sum()) + float(und_pnl[c].sum())
            _pair_rows.append(row)
    pnl_bucket_4_by_pair = (
        pd.DataFrame(_pair_rows).sort_values("total_pnl", ascending=False)
        if _pair_rows
        else pd.DataFrame(columns=["underlying", "etf", "beta"] + base_cols)
    )
    pnl_by_pair = (
        pd.concat(
            [
                pnl_bucket_4_by_pair.assign(bucket="bucket_4"),
                df[df["bucket"].isin(["bucket_1", "bucket_2"])][["underlying", "symbol", "bucket"] + base_cols],
            ],
            ignore_index=True,
        )
        if not df.empty
        else pd.DataFrame()
    )

    pnl_by_bucket = (
        df.groupby("bucket", as_index=False)[base_cols]
        .sum()
        .sort_values("total_pnl", ascending=False)
    )

    pnl_by_symbol.to_csv(outdir / "pnl_by_symbol.csv", index=False)
    pnl_by_underlying.to_csv(outdir / "pnl_by_underlying.csv", index=False)
    pnl_by_underlying_b12.to_csv(outdir / "pnl_by_underlying_b12.csv", index=False)
    pnl_bucket_1.to_csv(outdir / "pnl_bucket_1.csv", index=False)
    pnl_bucket_2.to_csv(outdir / "pnl_bucket_2.csv", index=False)
    pnl_bucket_3.to_csv(outdir / "pnl_bucket_3.csv", index=False)
    pnl_bucket_4.to_csv(outdir / "pnl_bucket_4.csv", index=False)
    pnl_bucket_4_by_symbol.to_csv(outdir / "pnl_bucket_4_by_symbol.csv", index=False)
    pnl_bucket_4_by_pair.to_csv(outdir / "pnl_bucket_4_by_pair.csv", index=False)
    if not pnl_by_pair.empty:
        pnl_by_pair.to_csv(outdir / "pnl_by_pair.csv", index=False)
    pnl_by_bucket.to_csv(outdir / "pnl_by_bucket.csv", index=False)

    # ── Net exposure (beta-normalized) ──
    bucket3_only_syms = set(flow_short_set) | set(flow_low_beta_bucket3_syms)
    pos_b124 = pos[(~pos["symbol"].isin(bucket3_only_syms))].copy()
    b124_underlyings = (
        set(pnl_by_underlying["underlying"].dropna().astype(str)) if not pnl_by_underlying.empty else set()
    )
    if not pos_b124.empty and b124_underlyings:
        exposure_df, exposure_detail_df = compute_net_exposure(
            flex_positions_path,
            etf_screened_path,
            b124_underlyings,
            positions_df=pos_b124,
        )
    else:
        exposure_df = pd.DataFrame(
            columns=["underlying", "symbols", "net_notional_usd", "gross_notional_usd", "n_legs"]
        )
        exposure_detail_df = pd.DataFrame(
            columns=["underlying", "symbol", "net_notional_usd", "gross_notional_usd"]
        )

    if not exposure_df.empty and not exposure_detail_df.empty:
        _d = exposure_detail_df.copy()
        _d["symbol"] = _d["symbol"].astype(str)
        _d["underlying"] = _d["underlying"].astype(str)
        _d["_is_etf"] = _d["symbol"].isin(etf_to_beta_map)
        _d["_beta"] = pd.to_numeric(_d["symbol"].map(etf_to_beta_map), errors="coerce").fillna(1.0)
        def _ledger_spot_ratio(_row: pd.Series, _bkt: str) -> float:
            if bool(_row["_is_etf"]):
                return 0.0
            _sym = str(_row["symbol"])
            _u = str(_row["underlying"])
            if _sym != _u:
                return 0.0
            _pos_q = float(_spot_qty.get(_sym, 0.0))
            if abs(_pos_q) <= 1e-12:
                return 0.0
            _lq = float(_ledger_qty_map.get(_u, {}).get(_bkt, 0.0))
            return _lq / _pos_q

        _d["_ratio_b1"] = np.where(
            _d["_is_etf"],
            np.where(_d["_beta"] > 1.5, 1.0, 0.0),
            _d.apply(lambda r: _ledger_spot_ratio(r, "bucket_1"), axis=1),
        )
        _d["_ratio_b2"] = np.where(
            _d["_is_etf"],
            np.where(
                ((_d["_beta"] > 0) & (_d["_beta"] <= 1.5))
                & (~_d["symbol"].isin(flow_low_beta_bucket3_syms)),
                1.0,
                0.0,
            ),
            _d.apply(lambda r: _ledger_spot_ratio(r, "bucket_2"), axis=1),
        )
        _d["_ratio_b4"] = np.where(
            _d["_is_etf"],
            np.where((_d["_beta"] < 0) & (~_d["symbol"].isin(flow_short_set)), 1.0, 0.0),
            _d.apply(lambda r: _ledger_spot_ratio(r, "bucket_4"), axis=1),
        )
        # Pure bucket-4 ETF legs: no b1/b2 residual
        _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b1"] = 0.0
        _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b2"] = 0.0
        _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b4"] = 1.0

        def _scale_detail(_detail: pd.DataFrame, ratio_col: str) -> pd.DataFrame:
            out = _detail[_detail[ratio_col] > 1e-12].copy()
            out["net_notional_usd"] = out["net_notional_usd"] * out[ratio_col]
            out["gross_notional_usd"] = out["gross_notional_usd"] * out[ratio_col]
            return out

        def _agg_bucket_exposure(_detail: pd.DataFrame) -> pd.DataFrame:
            if _detail.empty:
                return pd.DataFrame(
                    columns=["underlying", "symbols", "net_notional_usd", "gross_notional_usd", "n_legs"]
                )
            return (
                _detail.groupby("underlying", as_index=False)
                .agg(
                    symbols=("symbol", lambda s: ", ".join(sorted(set(s.astype(str))))),
                    net_notional_usd=("net_notional_usd", "sum"),
                    gross_notional_usd=("gross_notional_usd", "sum"),
                    n_legs=("symbol", "nunique"),
                )
                .sort_values("net_notional_usd", ascending=False)
            )

        exposure_b1_df = _agg_bucket_exposure(_scale_detail(_d, "_ratio_b1"))
        exposure_b2_df = _agg_bucket_exposure(_scale_detail(_d, "_ratio_b2"))
        exposure_b4_from_b124_df = _agg_bucket_exposure(_scale_detail(_d, "_ratio_b4"))
    else:
        exposure_b1_df = pd.DataFrame(
            columns=["underlying", "symbols", "net_notional_usd", "gross_notional_usd", "n_legs"]
        )
        exposure_b2_df = pd.DataFrame(
            columns=["underlying", "symbols", "net_notional_usd", "gross_notional_usd", "n_legs"]
        )
        exposure_b4_from_b124_df = pd.DataFrame(
            columns=["underlying", "symbols", "net_notional_usd", "gross_notional_usd", "n_legs"]
        )

    _b4_under_qty = {
        str(u): float(_ledger_qty_map.get(str(u), {}).get("bucket_4", 0.0))
        for u in b4_underlyings
    }
    exposure_b4_df, exposure_b4_detail_df = compute_bucket4_pair_exposure(
        pos, b4_registry, underlying_b4_qty=_b4_under_qty
    )
    _ = exposure_b4_from_b124_df  # ratio-split cross-check; pair exposure is canonical for B4

    exposure_b1_df.to_csv(outdir / "net_exposure_bucket_1.csv", index=False)
    exposure_b2_df.to_csv(outdir / "net_exposure_bucket_2.csv", index=False)
    exposure_df = exposure_df.sort_values("net_notional_usd", ascending=False)
    exposure_df.to_csv(outdir / "net_exposure_by_underlying.csv", index=False)
    if not exposure_b4_detail_df.empty:
        exposure_b4_detail_df.to_csv(outdir / "net_exposure_bucket_4_detail.csv", index=False)

    # Bucket 3 exposure (flow inverse / hedge)
    pos_neg = pos[pos["symbol"].isin(neg_beta_syms)].copy()
    pos_b3 = pos_neg[pos_neg["symbol"].isin(flow_short_set)].copy()
    pos_b3_pos_beta_flow = pos[pos["symbol"].isin(flow_low_beta_bucket3_syms)].copy()
    if not pos_b3_pos_beta_flow.empty:
        pos_b3 = pd.concat([pos_b3, pos_b3_pos_beta_flow], ignore_index=True)
    if not pos_b3.empty:
        pos_b3["beta"] = pos_b3["symbol"].map(etf_to_beta_map).fillna(1.0)
        pos_b3["mv_base"] = pos_b3["position"] * pos_b3["beta"] * pos_b3["markPrice"] * pos_b3["fxRateToBase"]
        pos_b3["gross_mv_base"] = pos_b3["mv_base"].abs()
        exposure_b3_df = (
            pos_b3.groupby("symbol", as_index=False)
            .agg(
                net_notional_usd=("mv_base", "sum"),
                gross_notional_usd=("gross_mv_base", "sum"),
                n_legs=("symbol", "nunique"),
            )
            .sort_values("net_notional_usd", ascending=False)
        )
    else:
        exposure_b3_df = pd.DataFrame(columns=["symbol", "net_notional_usd", "gross_notional_usd", "n_legs"])
    exposure_b3_df.to_csv(outdir / "net_exposure_bucket_3.csv", index=False)
    exposure_b4_df.to_csv(outdir / "net_exposure_bucket_4.csv", index=False)

    totals = {
        "run_date": run_date,
        "fifo_summary_present": bool(not fifo.empty),
        "total_realized_pnl": float(df["realized_pnl"].sum()),
        "total_unrealized_pnl": float(df["unrealized_pnl"].sum()),
        "total_dividends": float(df["dividends"].sum()),
        "total_withholding_tax": float(df["withholding_tax"].sum()),
        "total_pil_dividends": float(df["pil_dividends"].sum()),
        "total_borrow_fees": float(df["borrow_fees"].sum()),
        "total_short_credit_interest": float(df["short_credit_interest"].sum()),
        "total_other_fees": float(df["other_fees"].sum()),
        "total_bond_interest": float(df["bond_interest"].sum()),
        "total_pnl": float(df["total_pnl"].sum()),
        "excluded_cash_interest_base": float(cash.loc[cash["category"] == "exclude_interest", "amount_base"].sum()),
        "borrow_mode": borrow_mode,
        "borrow_total_account_source": float(borrow_total_account_source),
        "borrow_details_file_present": bool(flex_borrow_details_path.exists()),
        "borrow_details_rows_cumulative": int(borrow_details_kept.shape[0]) if isinstance(borrow_details_kept, pd.DataFrame) else 0,
        "universe_source": str(etf_screened_path),
        "universe_allowed_etfs": int(len(allowed_etfs)),
        "universe_allowed_underlyings": int(len(allowed_underlyings)),
        "kept_symbols": int(df["symbol"].nunique()) if not df.empty else 0,
        "kept_underlyings": int(df["underlying"].nunique()) if not df.empty else 0,
        "net_exposure_total": float(exposure_df["net_notional_usd"].sum()) if not exposure_df.empty else 0.0,
        "gross_exposure_total": float(exposure_df["gross_notional_usd"].sum()) if not exposure_df.empty else 0.0,
        "net_exposure_bucket_1": float(exposure_b1_df["net_notional_usd"].sum()) if not exposure_b1_df.empty else 0.0,
        "gross_exposure_bucket_1": float(exposure_b1_df["gross_notional_usd"].sum()) if not exposure_b1_df.empty else 0.0,
        "net_exposure_bucket_2": float(exposure_b2_df["net_notional_usd"].sum()) if not exposure_b2_df.empty else 0.0,
        "gross_exposure_bucket_2": float(exposure_b2_df["gross_notional_usd"].sum()) if not exposure_b2_df.empty else 0.0,
        "net_exposure_bucket_3": float(exposure_b3_df["net_notional_usd"].sum()) if not exposure_b3_df.empty else 0.0,
        "gross_exposure_bucket_3": float(exposure_b3_df["gross_notional_usd"].sum()) if not exposure_b3_df.empty else 0.0,
        "net_exposure_bucket_4": float(exposure_b4_df["net_notional_usd"].sum()) if not exposure_b4_df.empty else 0.0,
        "gross_exposure_bucket_4": float(exposure_b4_df["gross_notional_usd"].sum()) if not exposure_b4_df.empty else 0.0,
        "yfinance_override": bool(use_yfinance),
        "yfinance_symbols_overridden": len(yf_closes),
        "bucket_split_method": split_method,
        "bucket_component_split_method": component_split_method,
        "bucket3_flow_low_beta_symbols": sorted(flow_low_beta_bucket3_syms),
        "bucket_state_path": str(bucket_state_path),
        "bucket_realized_underlyings_from_trade_timing": int(len(_realized_ratio_from_trades)),
        "pnl_underlying_scope": "bucket_1_2_4",
        "bucket4_pairs": int(len(b4_registry)),
        "bucket_pnl": {
            row["bucket"]: float(row["total_pnl"])
            for _, row in pnl_by_bucket.iterrows()
        },
    }
    with open(outdir / "totals.json", "w", encoding="utf-8") as f:
        json.dump(totals, f, indent=2)

    if fifo.empty:
        print(
            "WARNING: No FIFO performance summary found in flex_trades.xml; trading PnL set to 0. "
            "Fix your Flex query to include FIFOPerformanceSummary(InBase) if you want realized/unrealized PnL."
        )

    n_exp = len(exposure_df) if not exposure_df.empty else 0
    print(f"[ACCOUNTING] PnL: {df['symbol'].nunique()} symbols, {df['underlying'].nunique()} underlyings")
    print(f"[ACCOUNTING] Bucket split method: {split_method} | component split: {component_split_method}")
    print(f"[ACCOUNTING] Trade-timed realized underlyings: {len(_realized_ratio_from_trades)}")
    print(
        f"[ACCOUNTING] Exposure (b124): {n_exp} underlyings, "
        f"net={totals['net_exposure_total']:,.0f}, gross={totals['gross_exposure_total']:,.0f}"
    )
    print(f"[ACCOUNTING] Bucket 1: net={totals['net_exposure_bucket_1']:,.0f}, gross={totals['gross_exposure_bucket_1']:,.0f}")
    print(f"[ACCOUNTING] Bucket 2: net={totals['net_exposure_bucket_2']:,.0f}, gross={totals['gross_exposure_bucket_2']:,.0f}")
    print(f"[ACCOUNTING] Bucket 3: net={totals['net_exposure_bucket_3']:,.0f}, gross={totals['gross_exposure_bucket_3']:,.0f}")
    print(f"[ACCOUNTING] Bucket 4: net={totals['net_exposure_bucket_4']:,.0f}, gross={totals['gross_exposure_bucket_4']:,.0f}")

    return 0


if __name__ == "__main__":
    _yf_flag = "--yf" in sys.argv
    _args = [a for a in sys.argv[1:] if a != "--yf"]
    _run_date = _args[0] if _args else None
    raise SystemExit(main(_run_date, use_yfinance=_yf_flag if _yf_flag else None))