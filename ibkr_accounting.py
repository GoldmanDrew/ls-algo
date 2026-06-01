#!/usr/bin/env python3
"""
ibkr_accounting.py

Rebuild an accounting-grade PnL report from IBKR Flex XML exports,
plus delta-normalized net exposure by underlying.

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
- Net exposure by underlying (delta-normalized) is now computed here and
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
- net_exposure_bucket_1.csv … net_exposure_bucket_4.csv (sleeve view; income shorts in b3)
- net_exposure_bucket_{1,2,4}_full.csv (audit view; pre-overlay income-short routing)
- net_exposure_spot_by_underlying.csv
- bucket_exposure_detail.csv, bucket_leg_classification.csv
- bucket_ratio_reconciliation.csv
- net_exposure_bucket_4_detail.csv
- totals.json

Run:
  python ibkr_accounting.py YYYY-MM-DD
"""

from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
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


def reconcile_spot_bucket_unrealized(
    qty_b1: float,
    qty_b2: float,
    qty_b4: float,
    cost_b1: float,
    cost_b2: float,
    cost_b4: float,
    ibkr_qty: float,
    px_now: float,
    *,
    flat_share_eps: float = 0.5,
) -> dict[str, float]:
    """Per-bucket spot unrealized PnL, reconciled to the physical line.

    The per-bucket FIFO ledger tracks an independent qty AND cost basis for each
    sleeve. When a physical round-trip is split across buckets (e.g. spot buys
    tagged ``bucket_1`` and later sells tagged ``bucket_4`` via inverse-ETF
    co-execution), the ledger can be left holding **offsetting opposite-sign
    lots** (long in one bucket, short in another) whose independent cost bases
    no longer net. Marking each bucket separately (``qty_b * px - cost_b``) then
    manufactures large equal-and-opposite "phantom" unrealized PnL that cancels
    at the account level but shreds the bucket attribution (e.g. DIA/XLK, fully
    closed yet showing ±$25k across B1/B4).

    This reconciles to two invariants before marking:

    1. **Flat physical → zero unrealized.** If ``|ibkr_qty|`` is under
       ``flat_share_eps`` the position is closed; there is nothing to mark, so
       all per-bucket unrealized collapses to the (residual ≈ 0) conserved
       total, parked on B1. Only realized PnL survives for closed names.
    2. **No bucket may oppose the physical net sign.** Any bucket whose ledger
       qty sign opposes ``sign(ibkr_qty)`` is a FIFO round-trip artifact: its
       qty and cost are folded into the same-sign buckets (pro-rata by |qty|)
       so the structural-short slice comes only from the explicit plan/implied
       inject path, never from a phantom ledger short.

    The conserved total ``Σ(qty_b * px - cost_b)`` is preserved exactly, so the
    account-level and per-underlying PnL are unchanged; only the split moves.
    """
    qtys = {"bucket_1": float(qty_b1), "bucket_2": float(qty_b2), "bucket_4": float(qty_b4)}
    costs = {"bucket_1": float(cost_b1), "bucket_2": float(cost_b2), "bucket_4": float(cost_b4)}
    px = float(px_now)

    def _mark(q: dict[str, float], c: dict[str, float]) -> dict[str, float]:
        return {b: q[b] * px - c[b] for b in qtys}

    um = _mark(qtys, costs)
    total_um = sum(um.values())

    # (1) Flat physical line: no position => no unrealized.
    if abs(float(ibkr_qty)) < float(flat_share_eps):
        return {"bucket_1": total_um, "bucket_2": 0.0, "bucket_4": 0.0}

    net_sign = 1.0 if float(ibkr_qty) > 0 else -1.0
    opposing = [b for b in qtys if qtys[b] * net_sign < -float(flat_share_eps)]
    if not opposing:
        return um

    # (2) Fold opposing-sign phantom lots into the aligned buckets pro-rata.
    aligned = [b for b in qtys if b not in opposing]
    add_q = sum(qtys[b] for b in opposing)
    add_c = sum(costs[b] for b in opposing)
    new_q = dict(qtys)
    new_c = dict(costs)
    for b in opposing:
        new_q[b] = 0.0
        new_c[b] = 0.0
    aligned_w = {b: abs(qtys[b]) for b in aligned}
    wsum = sum(aligned_w.values())
    if wsum > 1e-12:
        for b in aligned:
            new_q[b] += add_q * (aligned_w[b] / wsum)
            new_c[b] += add_c * (aligned_w[b] / wsum)
    elif aligned:
        new_q[aligned[0]] += add_q
        new_c[aligned[0]] += add_c
    return _mark(new_q, new_c)


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


def expand_blacklist(
    blacklist: set[str],
    etf_to_under: dict[str, str],
) -> tuple[set[str], set[str]]:
    """Expand strategy blacklist to all blocked symbols and underlyings.

    When an underlying is blacklisted (e.g. APLD), every ETF mapped to that
    underlying (APLX, APLZ, …) is excluded from exposure / capital metrics.
    When an ETF is blacklisted directly, its underlying is blocked too.
    """
    blocked_symbols: set[str] = set()
    blocked_underlyings: set[str] = set()
    bl = {canonical_symbol(str(s)) for s in blacklist if str(s).strip()}
    for sym in bl:
        blocked_symbols.add(sym)
        blocked_underlyings.add(sym)
    for etf, under in etf_to_under.items():
        e = canonical_symbol(str(etf))
        u = canonical_symbol(str(under))
        if not e or not u:
            continue
        if e in bl or u in bl:
            blocked_symbols.add(e)
            blocked_symbols.add(u)
            blocked_underlyings.add(u)
    return blocked_symbols, blocked_underlyings


def _filter_exposure_df(
    exposure_df: pd.DataFrame,
    blocked_underlyings: set[str],
    *,
    underlying_col: str = "underlying",
) -> pd.DataFrame:
    """Drop blacklisted underlyings from an exposure aggregate table."""
    if exposure_df.empty or not blocked_underlyings:
        return exposure_df
    if underlying_col not in exposure_df.columns:
        return exposure_df
    return exposure_df[
        ~exposure_df[underlying_col].astype(str).isin(blocked_underlyings)
    ].copy()


def _filter_positions_blacklist(
    pos: pd.DataFrame,
    blocked_symbols: set[str],
    blocked_underlyings: set[str],
    etf_to_under: dict[str, str],
) -> pd.DataFrame:
    """Remove blacklisted symbols / underlyings from a positions frame."""
    if pos.empty or (not blocked_symbols and not blocked_underlyings):
        return pos
    out = pos.copy()
    if "underlying" not in out.columns:
        under_sym = out.get("underlyingSymbol")
        if under_sym is not None:
            out["underlying"] = out["symbol"].map(etf_to_under).fillna(under_sym)
        else:
            out["underlying"] = out["symbol"].map(etf_to_under).fillna(out["symbol"])
    out = out[~out["symbol"].astype(str).isin(blocked_symbols)].copy()
    out = out[~out["underlying"].astype(str).isin(blocked_underlyings)].copy()
    return out


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


def load_etf_delta_map(screened_csv: Path) -> tuple[dict[str, str], dict[str, float]]:
    """
    Load ETF -> Underlying mapping and ETF -> Delta from etf_screened_today.csv.
    Returns (etf_to_under, etf_to_delta) dicts keyed by canonical symbol.

    Delta is the ETF's sensitivity to its underlying (e.g. 2.0 for a
    2× levered ETF, -1.0 for an inverse, 1.0 for a plain wrapper).
    """
    u = pd.read_csv(screened_csv)
    cols_lc = {c.lower(): c for c in u.columns}

    etf_col = _find_col(cols_lc, ["etf", "symbol", "ticker", "etf_symbol"])
    under_col = _find_col(cols_lc, ["underlying", "underlyingsymbol", "underlying_symbol", "root"])
    delta_col = _find_col(cols_lc, ["delta", "beta", "leverage", "lev"])

    if etf_col is None or under_col is None:
        return {}, {}

    u = u[[etf_col, under_col] + ([delta_col] if delta_col else [])].dropna(subset=[etf_col, under_col])
    u[etf_col] = u[etf_col].astype(str).str.upper().map(canonical_symbol)
    u[under_col] = u[under_col].astype(str).str.upper().map(canonical_symbol)

    etf_to_under = dict(zip(u[etf_col], u[under_col]))

    if delta_col:
        u[delta_col] = pd.to_numeric(u[delta_col], errors="coerce").fillna(1.0)
        etf_to_delta = dict(zip(u[etf_col], u[delta_col]))
    else:
        etf_to_delta = {k: 1.0 for k in etf_to_under}

    return etf_to_under, etf_to_delta


def _is_etf_leg(symbol: str, underlying: str, etf_to_under: dict[str, str]) -> bool:
    """True when ``symbol`` is a packaged ETF, not the spot underlying row.

    Classification keys off the *screener* mapping (``etf_to_under``) rather than
    the Flex-reported ``underlyingSymbol``. IBKR Flex sometimes echoes
    ``underlyingSymbol == symbol`` for a packaged ETF (e.g. APLZ trades tagged
    with underlyingSymbol=APLZ instead of APLD). Trusting that field mis-booked
    those ETF trades into the spot share ledger under the ETF ticker, manufacturing
    huge ledger-vs-IBKR drift (APLZ ledger -2000 vs position -42000). Benchmark
    self-maps (SPY->SPY, MSTR->MSTR) remain spot because the mapped underlying
    equals the symbol.
    """
    sym = canonical_symbol(symbol)
    if sym not in etf_to_under:
        return False
    mapped_und = canonical_symbol(etf_to_under.get(sym, sym) or sym)
    return sym != mapped_und


def complete_etf_maps_from_positions(
    pos: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
) -> tuple[dict[str, str], dict[str, float]]:
    """
  Extend ETF maps with 1× benchmark holdings (e.g. SPY, QQQ) that appear in
  the account but have no dedicated row in etf_screened_today.csv.

  Without this, compute_net_exposure treats them as spot (underlying in the
  strategy universe) while bucket exposure uses ledger ratios, which breaks
  bucket↔book reconciliation.
    """
    if pos is None or pos.empty:
        return etf_to_under, etf_to_delta
    unders_with_products = set(etf_to_under.values())
    out_under = dict(etf_to_under)
    out_delta = dict(etf_to_delta)
    for sym in pos["symbol"].astype(str).dropna().unique():
        s = canonical_symbol(sym)
        if not s or s in out_delta:
            continue
        if s in unders_with_products:
            out_under[s] = s
            out_delta[s] = 1.0
    return out_under, out_delta


def _spot_ledger_bucket_ratios(
    position_qty: float,
    ledger_qty: dict[str, float],
) -> tuple[float, float, float]:
    """
    Signed share ratios for a spot leg from FIFO ledger qty_b*.

    Caps attributed qty at the live IBKR line when the ledger is stale, and
    assigns any unattributed remainder to bucket_1 (orphan / pre-strategy shares).
    """
    if abs(position_qty) <= 1e-12:
        return 0.0, 0.0, 0.0
    b1 = float(ledger_qty.get("bucket_1", 0.0))
    b2 = float(ledger_qty.get("bucket_2", 0.0))
    b4 = float(ledger_qty.get("bucket_4", 0.0))
    ledger_total = b1 + b2 + b4
    if abs(ledger_total) > abs(position_qty) and abs(ledger_total) > 1e-12:
        scale = abs(position_qty) / abs(ledger_total)
        b1, b2, b4 = b1 * scale, b2 * scale, b4 * scale
        ledger_total = b1 + b2 + b4
    r1 = b1 / position_qty
    r2 = b2 / position_qty
    r4 = b4 / position_qty
    ratio_sum = r1 + r2 + r4
    if ratio_sum < 1.0 - 1e-9:
        r1 += 1.0 - ratio_sum
    elif ratio_sum > 1.0 + 1e-9:
        scale = 1.0 / ratio_sum
        r1, r2, r4 = r1 * scale, r2 * scale, r4 * scale
    return r1, r2, r4


def _spot_exposure_bucket_ratios(
    position_qty: float,
    ledger_qty: dict[str, float],
) -> tuple[float, float, float]:
    """
    Exposure-only spot split: tagged rebalance shares over tagged total.

    Uses qty_b* / (qty_b1 + qty_b2 + qty_b4) from the share ledger, applied to
    the full IBKR line. Stale ledgers larger than the live line are scaled down
    first (same cap as ``_spot_ledger_bucket_ratios``). Untagged IBKR shares
    inherit the tagged mix rather than diluting bucket 2 via qty_b2/ibkr_qty.
    """
    if abs(position_qty) <= 1e-12:
        return 0.0, 0.0, 0.0
    b1 = float(ledger_qty.get("bucket_1", 0.0))
    b2 = float(ledger_qty.get("bucket_2", 0.0))
    b4 = float(ledger_qty.get("bucket_4", 0.0))
    ledger_total = abs(b1) + abs(b2) + abs(b4)
    if ledger_total > abs(position_qty) and ledger_total > 1e-12:
        scale = abs(position_qty) / ledger_total
        b1, b2, b4 = b1 * scale, b2 * scale, b4 * scale
        ledger_total = abs(b1) + abs(b2) + abs(b4)
    if ledger_total <= 1e-12:
        return 1.0, 0.0, 0.0
    r1 = abs(b1) / ledger_total
    r2 = abs(b2) / ledger_total
    r4 = abs(b4) / ledger_total
    return r1, r2, r4


@dataclass(frozen=True)
class SpotBucketRatios:
    """Canonical b1/b2/b4 split for a physical spot line (sums to 1)."""

    b1: float
    b2: float
    b4: float
    source: str


@dataclass(frozen=True)
class HedgeRatioSpotMeta:
    """Per-underlying hedge-ratio spot targets vs allocation (exposure only)."""

    hedge_target_usd_b1: float
    hedge_target_usd_b2: float
    hedge_target_usd_b4: float
    hedge_alloc_usd_b1: float
    hedge_alloc_usd_b2: float
    hedge_target_qty_b1: float
    hedge_target_qty_b2: float
    hedge_target_qty_b4: float
    hedge_alloc_qty_b1: float
    hedge_alloc_qty_b2: float
    ledger_qty_b1: float
    ledger_qty_b2: float
    ledger_qty_b4: float
    plan_b4_qty: float
    ibkr_qty: float


def _build_b4_phr_by_etf(
    b4_registry: pd.DataFrame | None,
    *,
    partial_hedge_ratio_default: float = 0.75,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if b4_registry is None or b4_registry.empty:
        return out
    for _, row in b4_registry.iterrows():
        etf = canonical_symbol(str(row.get("etf", "") or ""))
        if not etf:
            continue
        try:
            phr = float(row.get("partial_hedge_ratio", partial_hedge_ratio_default))
        except (TypeError, ValueError):
            phr = partial_hedge_ratio_default
        if not np.isfinite(phr) or abs(phr) < 1e-12:
            phr = partial_hedge_ratio_default
        out[etf] = float(phr)
    return out


def _b12_hedge_spot_usd_for_etf(
    etf_net: float,
    beta: float,
    *,
    bucket: str,
    delta_floor: float,
) -> float:
    """
    Long spot USD to offset a short B1/B2 ETF leg in δ-normalized exposure.

    ``net_notional_usd`` already includes ETF δ (``qty × δ × mark``), so the
    paired spot slice is ``|net|`` — same units as ``compute_net_exposure``.
    (Trade-plan ``hr = 1/β`` applies to capital sizing, not this attribution.)
    """
    del beta, bucket, delta_floor  # bucket retained for call-site clarity
    if etf_net >= -1e-12:
        return 0.0
    return abs(float(etf_net))


def _b4_structural_hedge_usd_for_etf(etf_net: float, partial_hedge_ratio: float) -> float:
    """Informational B4 structural short USD (``phr × |etf_net|``); not ratio-split r4."""
    if abs(float(etf_net)) <= 1e-12:
        return 0.0
    return float(partial_hedge_ratio) * abs(float(etf_net))


def _hedge_ratio_finalize_spot_ratios(
    *,
    spot_net: float,
    need_b1: float,
    need_b2: float,
    has_b1: bool,
    has_b2: bool,
    has_b4: bool,
    b4_b12_only: bool,
) -> tuple[SpotBucketRatios, float, float]:
    """Allocate long spot to B2 hedge need first, then B1; orphan → B1 when held."""
    if abs(spot_net) <= 1e-12:
        return SpotBucketRatios(1.0, 0.0, 0.0, "hedge_ratio_flat"), 0.0, 0.0

    rem = float(spot_net)
    alloc_b2 = min(rem, float(need_b2)) if need_b2 > 1e-12 and rem > 1e-12 else 0.0
    rem -= alloc_b2
    alloc_b1 = min(rem, float(need_b1)) if need_b1 > 1e-12 and rem > 1e-12 else 0.0
    rem -= alloc_b1

    r1 = alloc_b1 / spot_net
    r2 = alloc_b2 / spot_net
    r4 = 0.0
    ratio_sum = r1 + r2 + r4
    if ratio_sum < 1.0 - 1e-9:
        orphan = 1.0 - ratio_sum
        if has_b1 or need_b1 > 1e-12:
            r1 += orphan
        elif has_b2 or need_b2 > 1e-12:
            r2 += orphan
        else:
            r1 += orphan
    elif ratio_sum > 1.0 + 1e-9:
        scale = 1.0 / ratio_sum
        r1, r2, r4 = r1 * scale, r2 * scale, r4 * scale

    r1, r2, r4 = apply_spot_bucket_eligibility(
        r1, r2, r4, has_b1_etf=has_b1, has_b2_etf=has_b2, has_b4_etf=has_b4
    )
    if b4_b12_only:
        s12 = r1 + r2
        if s12 > 1e-12:
            r1, r2 = r1 / s12, r2 / s12
        r4 = 0.0
    return SpotBucketRatios(r1, r2, r4, "hedge_ratio"), alloc_b1, alloc_b2


def _sleeve_balance_finalize_spot_ratios(
    *,
    spot_net: float,
    etf_b1: float,
    etf_b2: float,
    b4_b12_only: bool,
) -> SpotBucketRatios:
    """
    Sleeve-balance spot split: assign long spot to offset short ETF sleeves only.

    Ratios may sum to less than 1; remainder is **unbucketed** (not forced to B1).
    """
    if abs(spot_net) <= 1e-12:
        return SpotBucketRatios(1.0, 0.0, 0.0, "sleeve_balance_flat")

    r1 = r2 = 0.0
    if float(etf_b2) < -1e-12:
        r2 = min(1.0, max(0.0, -float(etf_b2) / float(spot_net)))
    if float(etf_b1) < -1e-12:
        r1 = min(1.0, max(0.0, -float(etf_b1) / float(spot_net)))
    s12 = r1 + r2
    if s12 > 1.0 + 1e-9:
        r1, r2 = r1 / s12, r2 / s12
    r4 = 0.0  # structural B4 carve applied later on registry names
    return SpotBucketRatios(r1, r2, r4, "sleeve_balance")


def sleeve_balance_spot_ratios_from_exposure_detail(
    detail: pd.DataFrame,
    *,
    etf_to_under: dict[str, str],
    etf_to_delta_map: dict[str, float],
    flow_short_set: set[str],
    b4_etf_syms: set[str],
    b4_spot_b12_only: set[str] | None = None,
) -> dict[str, SpotBucketRatios]:
    """Sleeve-balance spot split using the same leg nets as ``bucket_exposure_detail``."""
    if detail.empty:
        return {}
    d = detail.copy()
    d["underlying"] = d["underlying"].astype(str)
    d["symbol"] = d["symbol"].astype(str)
    d["net_notional_usd"] = pd.to_numeric(d["net_notional_usd"], errors="coerce").fillna(0.0)
    if "_is_etf" not in d.columns:
        d["_is_etf"] = d.apply(
            lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
            axis=1,
        )
    if "_beta" not in d.columns:
        d["_beta"] = pd.to_numeric(
            d["symbol"].map(etf_to_delta_map), errors="coerce"
        ).fillna(1.0)
    _b4_b12 = b4_spot_b12_only or set()
    out: dict[str, SpotBucketRatios] = {}
    for u, grp in d.groupby("underlying"):
        u_str = str(u)
        spot_net = etf_b1 = etf_b2 = 0.0
        for _, row in grp.iterrows():
            sym = str(row["symbol"])
            net = float(row["net_notional_usd"])
            if (not bool(row["_is_etf"])) and sym == u_str:
                spot_net += net
                continue
            if not bool(row["_is_etf"]):
                continue
            beta = float(row.get("_beta", etf_to_delta_map.get(sym, 1.0)))
            bkt, _ = classify_etf_leg_bucket(
                sym, beta, flow_short_set=flow_short_set, b4_etf_syms=b4_etf_syms
            )
            if bkt == "bucket_1":
                etf_b1 += net
            elif bkt == "bucket_2":
                etf_b2 += net
        out[u_str] = _sleeve_balance_finalize_spot_ratios(
            spot_net=spot_net,
            etf_b1=etf_b1,
            etf_b2=etf_b2,
            b4_b12_only=u_str in _b4_b12,
        )
    return out


def sleeve_balance_spot_bucket_ratios(
    underlying: str,
    pos: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    *,
    flow_short_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
    b4_registry_b12_only: bool = False,
) -> SpotBucketRatios:
    """Per-underlying sleeve-balance spot ratios from live positions."""
    flow = flow_short_syms or set()
    b4_set = b4_etf_syms or set()
    u = canonical_symbol(underlying)
    spot_net = etf_b1 = etf_b2 = 0.0

    if pos is None or pos.empty:
        return SpotBucketRatios(1.0, 0.0, 0.0, "sleeve_balance_empty")

    for sym, grp in pos.groupby("symbol"):
        sym_c = canonical_symbol(str(sym))
        if sym_c == u:
            spot_net = sum(
                _position_mv_base_notional(r, underlying=u, etf_to_delta=etf_to_delta)
                for _, r in grp.iterrows()
            )
            continue
        if etf_to_under.get(sym_c) != u:
            continue
        if not _is_etf_leg(sym_c, u, etf_to_under):
            continue
        net = sum(
            _position_mv_base_notional(r, underlying=u, etf_to_delta=etf_to_delta)
            for _, r in grp.iterrows()
        )
        beta = float(etf_to_delta.get(sym_c, 1.0))
        bkt, _ = classify_etf_leg_bucket(
            sym_c, beta, flow_short_set=flow, b4_etf_syms=b4_set
        )
        if bkt == "bucket_1":
            etf_b1 += net
        elif bkt == "bucket_2":
            etf_b2 += net

    return _sleeve_balance_finalize_spot_ratios(
        spot_net=spot_net,
        etf_b1=etf_b1,
        etf_b2=etf_b2,
        b4_b12_only=b4_registry_b12_only,
    )


def build_net_exposure_unbucketed(
    exposure_detail: pd.DataFrame,
    etf_to_under: dict[str, str],
) -> pd.DataFrame:
    """Physical spot not attributed to B1/B2/B4 (sleeve_balance orphan slice)."""
    cols = ["underlying", "net_notional_usd", "gross_notional_usd", "orphan_frac"]
    if exposure_detail.empty:
        return pd.DataFrame(columns=cols)
    d = exposure_detail.copy()
    d["underlying"] = d["underlying"].astype(str)
    d["symbol"] = d["symbol"].astype(str)
    if "_is_etf" not in d.columns:
        d["_is_etf"] = d.apply(
            lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
            axis=1,
        )
    rows: list[dict] = []
    for _, row in d.iterrows():
        sym = str(row["symbol"])
        u = str(row["underlying"])
        if bool(row["_is_etf"]) or sym != u:
            continue
        net = float(row["net_notional_usd"])
        gross = float(row.get("gross_notional_usd", abs(net)) or abs(net))
        r1 = float(row.get("_ratio_b1", 0.0) or 0.0)
        r2 = float(row.get("_ratio_b2", 0.0) or 0.0)
        r4 = float(row.get("_ratio_b4", 0.0) or 0.0)
        orphan_frac = max(0.0, 1.0 - r1 - r2 - r4)
        if orphan_frac <= 1e-9:
            continue
        rows.append(
            {
                "underlying": u,
                "net_notional_usd": net * orphan_frac,
                "gross_notional_usd": gross * orphan_frac,
                "orphan_frac": orphan_frac,
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out.sort_values("net_notional_usd", key=lambda s: s.abs(), ascending=False)


def load_yieldboost_etf_syms(screened_path: Path, plan_csv: Path | None = None) -> set[str]:
    """ETF symbols tagged ``is_yieldboost`` in the screener or trade plan."""
    out: set[str] = set()
    for path in (screened_path, plan_csv):
        if path is None or not Path(path).exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "is_yieldboost" not in df.columns:
            continue
        cols_lc = {c.lower(): c for c in df.columns}
        etf_col = _find_col(cols_lc, ["etf", "symbol", "ticker"])
        if not etf_col:
            continue
        flagged = df[df["is_yieldboost"].map(_plan_row_is_yieldboost_scalar)]
        for raw in flagged[etf_col].dropna().astype(str):
            sym = canonical_symbol(raw)
            if sym:
                out.add(sym)
    return out


def resolve_flow_inverse_bucket3_syms(
    flow_short_set: set[str],
    etf_to_delta_map: dict[str, float],
) -> set[str]:
    """Flow-program inverse ETFs only (β < 0). Low-β flow names stay in bucket 2."""
    return {
        s
        for s in flow_short_set
        if float(etf_to_delta_map.get(s, 0.0)) < 0.0
    }


def ledger_spot_bucket_ratios(
    ibkr_qty: float,
    ledger_qty: dict[str, float],
    *,
    b12_only: bool = False,
    has_b1_etf: bool = True,
    has_b2_etf: bool = True,
    has_b4_etf: bool = True,
) -> SpotBucketRatios:
    """FIFO share-ledger spot split (stable B1/B2 ratio-split exposure/PnL).

    When ``b12_only=True`` (inverse-decay registry names under ``ledger_fifo``)
    the spot line is split only across buckets 1 and 2; bucket-4 ratio-split net
    stays on ETF legs. Structural short underlying exposure remains in the
    separate B4 pair view (``net_exposure_bucket_4_detail``).
    """
    if b12_only:
        b1 = abs(float(ledger_qty.get("bucket_1", 0.0)))
        b2 = abs(float(ledger_qty.get("bucket_2", 0.0)))
        denom = b1 + b2
        if denom <= 1e-12:
            r1, r2, r4 = apply_spot_bucket_eligibility(
                1.0, 0.0, 0.0,
                has_b1_etf=has_b1_etf,
                has_b2_etf=has_b2_etf,
                has_b4_etf=False,
            )
            return SpotBucketRatios(r1, r2, r4, "ledger_fifo_b12")
        if denom > abs(ibkr_qty) and abs(ibkr_qty) > 1e-12:
            scale = abs(ibkr_qty) / denom
            b1 *= scale
            b2 *= scale
            denom = b1 + b2
        r1, r2, r4 = apply_spot_bucket_eligibility(
            b1 / denom,
            b2 / denom,
            0.0,
            has_b1_etf=has_b1_etf,
            has_b2_etf=has_b2_etf,
            has_b4_etf=False,
        )
        return SpotBucketRatios(r1, r2, r4, "ledger_fifo_b12")
    r1, r2, r4 = _spot_exposure_bucket_ratios(ibkr_qty, ledger_qty)
    r1, r2, r4 = apply_spot_bucket_eligibility(
        r1, r2, r4, has_b1_etf=has_b1_etf, has_b2_etf=has_b2_etf, has_b4_etf=has_b4_etf
    )
    return SpotBucketRatios(r1, r2, r4, "ledger_fifo")


def resolve_underlying_spot_ratios(
    *,
    underlying: str,
    ibkr_qty: float,
    ledger_qty: dict[str, float],
    plan_ratio: dict[str, float] | None = None,
    plan_b4_pnl_mode: str = "inject_slice",
    yieldboost_spot_b2: bool = False,
    ledger_r_b1: float | None = None,
    ledger_r_b2: float | None = None,
    b12_spot_split_method: str = "ledger_fifo",
    b4_registry_underlying: bool = False,
    etf_to_under: dict[str, str] | None = None,
    etf_to_delta: dict[str, float] | None = None,
    pos: pd.DataFrame | None = None,
    flow_short_syms: set[str] | None = None,
) -> SpotBucketRatios:
    """
    Canonical spot-line b1/b2/b4 ratios (sum to 1).

    Used for net exposure and spot PnL attribution. When
    ``yieldboost_spot_b2`` is True and a B2 ETF sleeve is held, spot maps
    100% to bucket 2 (matching ``apply_yieldboost_spot_b2_override``).

    When ``b12_spot_split_method='ledger_fifo'`` (default for stable B1/B2
    reporting) the FIFO ledger mix is returned and plan/B4 waterfall inject
    is skipped. Use ``held_exposure_waterfall`` for the legacy daily hedge-residual
    carve (B4 pair diagnostics still use ``build_hedge_residual_spot_ratio_map``).
    """
    u = canonical_symbol(underlying)
    ledger = {
        "bucket_1": float(ledger_qty.get("bucket_1", 0.0)),
        "bucket_2": float(ledger_qty.get("bucket_2", 0.0)),
        "bucket_4": float(ledger_qty.get("bucket_4", 0.0)),
    }
    has_b1 = has_b2 = has_b4 = True
    if etf_to_under is not None and etf_to_delta is not None and pos is not None:
        has_b1, has_b2, has_b4 = held_etf_bucket_flags_from_positions(
            u,
            pos,
            etf_to_under,
            etf_to_delta,
            flow_short_syms=flow_short_syms,
        )
    if str(b12_spot_split_method).strip().lower() == "ledger_fifo":
        sr = ledger_spot_bucket_ratios(
            ibkr_qty,
            ledger,
            b12_only=b4_registry_underlying,
            has_b1_etf=has_b1,
            has_b2_etf=has_b2,
            has_b4_etf=has_b4,
        )
        if yieldboost_spot_b2 and has_b2:
            b4_frac = max(float(sr.b4), 0.0)
            rem = max(0.0, 1.0 - b4_frac)
            return SpotBucketRatios(0.0, rem, b4_frac, "yieldboost_spot_b2")
        return sr

    r1, r2, r4 = _spot_exposure_bucket_ratios(ibkr_qty, ledger)
    lb1 = float(ledger_r_b1 if ledger_r_b1 is not None else r1)
    lb2 = float(ledger_r_b2 if ledger_r_b2 is not None else r2)
    b1_norm, b2_norm = _normalize_b1_b2_pair(lb1, lb2)

    if plan_ratio and plan_b4_pnl_mode == "full_override":
        pr1 = float(plan_ratio.get("b1", 0.0))
        pr2 = float(plan_ratio.get("b2", 0.0))
        pr4 = float(plan_ratio.get("b4", 0.0))
        fb1, fb2, fb4 = _normalize_bucket_triple(pr1, pr2, pr4)
        return SpotBucketRatios(fb1, fb2, fb4, "plan_full_override")

    if plan_ratio and float(plan_ratio.get("b4", 0.0)) > 1e-12:
        b4_frac = float(plan_ratio.get("b4", 0.0))
        rem = max(0.0, 1.0 - b4_frac)
        return SpotBucketRatios(
            rem * b1_norm,
            rem * b2_norm,
            b4_frac,
            "plan_inject_slice",
        )

    if yieldboost_spot_b2:
        b4_frac = max(float(r4), 0.0)
        rem = max(0.0, 1.0 - b4_frac)
        return SpotBucketRatios(rem * b1_norm, rem * b2_norm, b4_frac, "yieldboost_spot_b2")

    return SpotBucketRatios(r1, r2, r4, "ledger")


def _position_mv_base_notional(
    row: pd.Series,
    *,
    underlying: str,
    etf_to_delta: dict[str, float],
) -> float:
    """Signed β-adjusted notional (matches ``compute_net_exposure`` leg nets)."""
    sym = canonical_symbol(str(row.get("symbol", "") or ""))
    u = canonical_symbol(underlying)
    delta = (
        1.0
        if sym == u
        else float(etf_to_delta.get(sym, etf_to_delta.get(sym, 1.0)))
    )
    return (
        float(row.get("position", 0.0) or 0.0)
        * delta
        * float(row.get("markPrice", 0.0) or 0.0)
        * float(row.get("fxRateToBase", 1.0) or 1.0)
    )


def _sleeve_offset_spot_slice(spot_remaining: float, etf_bucket_net: float) -> float:
    """Spot notional in this bucket used to offset ``etf_bucket_net`` (opposite signs)."""
    if abs(spot_remaining) <= 1e-12 or abs(etf_bucket_net) <= 1e-12:
        return 0.0
    if spot_remaining > 0 and etf_bucket_net < 0:
        return min(spot_remaining, -etf_bucket_net)
    if spot_remaining < 0 and etf_bucket_net > 0:
        return max(spot_remaining, -etf_bucket_net)
    return 0.0


def sleeve_offset_spot_bucket_ratios(
    underlying: str,
    pos: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    *,
    flow_short_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
    b4_registry_b12_only: bool = False,
) -> SpotBucketRatios:
    """
    Per-sleeve spot split for net exposure: assign long spot to offset short ETFs
    in the same bucket (B1 levered, B2 yieldboost, B4 inverse). Unpaired spot
    remainder is attributed to B1 when held, else B2, so ratios still sum to 1.
    """
    flow = flow_short_syms or set()
    b4_set = b4_etf_syms or set()
    u = canonical_symbol(underlying)
    spot_net = 0.0
    etf_b1 = etf_b2 = etf_b4 = 0.0

    if pos is None or pos.empty:
        return SpotBucketRatios(1.0, 0.0, 0.0, "sleeve_offset_empty")

    for sym, grp in pos.groupby("symbol"):
        sym_c = canonical_symbol(str(sym))
        if sym_c == u:
            spot_net = sum(
                _position_mv_base_notional(r, underlying=u, etf_to_delta=etf_to_delta)
                for _, r in grp.iterrows()
            )
            continue
        if etf_to_under.get(sym_c) != u:
            continue
        if not _is_etf_leg(sym_c, u, etf_to_under):
            continue
        net = sum(
            _position_mv_base_notional(r, underlying=u, etf_to_delta=etf_to_delta)
            for _, r in grp.iterrows()
        )
        beta = float(etf_to_delta.get(sym_c, 1.0))
        bkt, _ = classify_etf_leg_bucket(
            sym_c, beta, flow_short_set=flow, b4_etf_syms=b4_set
        )
        if bkt == "bucket_1":
            etf_b1 += net
        elif bkt == "bucket_2":
            etf_b2 += net
        elif bkt == "bucket_4":
            etf_b4 += net

    has_b1, has_b2, has_b4 = held_etf_bucket_flags_from_positions(
        u,
        pos,
        etf_to_under,
        etf_to_delta,
        flow_short_syms=flow,
    )

    if abs(spot_net) <= 1e-12:
        return SpotBucketRatios(1.0, 0.0, 0.0, "sleeve_offset_flat")

    rem = float(spot_net)
    b1_alloc = _sleeve_offset_spot_slice(rem, etf_b1)
    rem -= b1_alloc
    b2_alloc = _sleeve_offset_spot_slice(rem, etf_b2)
    rem -= b2_alloc
    b4_alloc = 0.0 if b4_registry_b12_only else _sleeve_offset_spot_slice(rem, etf_b4)
    rem -= b4_alloc

    r1 = b1_alloc / spot_net
    r2 = b2_alloc / spot_net
    r4 = b4_alloc / spot_net
    ratio_sum = r1 + r2 + r4
    if ratio_sum < 1.0 - 1e-9:
        orphan = 1.0 - ratio_sum
        if has_b1:
            r1 += orphan
        elif has_b2:
            r2 += orphan
        else:
            r1 += orphan
    elif ratio_sum > 1.0 + 1e-9:
        scale = 1.0 / ratio_sum
        r1, r2, r4 = r1 * scale, r2 * scale, r4 * scale

    r1, r2, r4 = apply_spot_bucket_eligibility(
        r1, r2, r4, has_b1_etf=has_b1, has_b2_etf=has_b2, has_b4_etf=has_b4
    )
    if b4_registry_b12_only:
        s12 = r1 + r2
        if s12 > 1e-12:
            r1, r2 = r1 / s12, r2 / s12
        r4 = 0.0
    return SpotBucketRatios(r1, r2, r4, "sleeve_offset")


def sleeve_offset_spot_ratios_from_exposure_detail(
    detail: pd.DataFrame,
    *,
    etf_to_under: dict[str, str],
    etf_to_delta_map: dict[str, float],
    flow_short_set: set[str],
    b4_etf_syms: set[str],
    b4_spot_b12_only: set[str] | None = None,
) -> dict[str, SpotBucketRatios]:
    """Sleeve-offset ratios using the same leg nets as ``bucket_exposure_detail``."""
    if detail.empty:
        return {}
    d = detail.copy()
    d["underlying"] = d["underlying"].astype(str)
    d["symbol"] = d["symbol"].astype(str)
    d["net_notional_usd"] = pd.to_numeric(d["net_notional_usd"], errors="coerce").fillna(0.0)
    if "_is_etf" not in d.columns:
        d["_is_etf"] = d.apply(
            lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
            axis=1,
        )
    _b4_b12 = b4_spot_b12_only or set()
    out: dict[str, SpotBucketRatios] = {}
    for u, grp in d.groupby("underlying"):
        u_str = str(u)
        spot_net = etf_b1 = etf_b2 = etf_b4 = 0.0
        for _, row in grp.iterrows():
            sym = str(row["symbol"])
            net = float(row["net_notional_usd"])
            if (not bool(row["_is_etf"])) and sym == u_str:
                spot_net += net
                continue
            if not bool(row["_is_etf"]):
                continue
            beta = float(etf_to_delta_map.get(sym, 1.0))
            bkt, _ = classify_etf_leg_bucket(
                sym, beta, flow_short_set=flow_short_set, b4_etf_syms=b4_etf_syms
            )
            if bkt == "bucket_1":
                etf_b1 += net
            elif bkt == "bucket_2":
                etf_b2 += net
            elif bkt == "bucket_4":
                etf_b4 += net
        has_b1, has_b2, has_b4 = (
            abs(etf_b1) > 1e-12,
            abs(etf_b2) > 1e-12,
            abs(etf_b4) > 1e-12,
        )
        if abs(spot_net) <= 1e-12:
            out[u_str] = SpotBucketRatios(1.0, 0.0, 0.0, "sleeve_offset_flat")
            continue
        rem = float(spot_net)
        b1_alloc = _sleeve_offset_spot_slice(rem, etf_b1)
        rem -= b1_alloc
        b2_alloc = _sleeve_offset_spot_slice(rem, etf_b2)
        rem -= b2_alloc
        b4_alloc = (
            0.0
            if u_str in _b4_b12
            else _sleeve_offset_spot_slice(rem, etf_b4)
        )
        rem -= b4_alloc
        r1 = b1_alloc / spot_net
        r2 = b2_alloc / spot_net
        r4 = b4_alloc / spot_net
        ratio_sum = r1 + r2 + r4
        if ratio_sum < 1.0 - 1e-9:
            orphan = 1.0 - ratio_sum
            if has_b1 or etf_b1 != 0:
                r1 += orphan
            elif has_b2 or etf_b2 != 0:
                r2 += orphan
            else:
                r1 += orphan
        elif ratio_sum > 1.0 + 1e-9:
            scale = 1.0 / ratio_sum
            r1, r2, r4 = r1 * scale, r2 * scale, r4 * scale
        r1, r2, r4 = apply_spot_bucket_eligibility(
            r1, r2, r4, has_b1_etf=has_b1, has_b2_etf=has_b2, has_b4_etf=has_b4
        )
        if u_str in _b4_b12:
            s12 = r1 + r2
            if s12 > 1e-12:
                r1, r2 = r1 / s12, r2 / s12
            r4 = 0.0
        out[u_str] = SpotBucketRatios(r1, r2, r4, "sleeve_offset")
    return out


def hedge_ratio_spot_bucket_ratios(
    underlying: str,
    pos: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    *,
    flow_short_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
    b4_registry_b12_only: bool = False,
    b4_phr_by_etf: dict[str, float] | None = None,
    partial_hedge_ratio_default: float = 0.75,
    delta_floor: float = 0.25,
    ledger_qty: dict[str, float] | None = None,
    plan_b4_qty: float = 0.0,
    ibkr_qty: float = 0.0,
) -> tuple[SpotBucketRatios, HedgeRatioSpotMeta]:
    """
    Spot exposure split from trade-plan hedge ratios: B1/B2 long spot pairs with
    ``|short_etf| × |β|`` (hr = 1/|β|). B4 ``partial_hedge_ratio`` (default 0.75)
    is tracked for structural shorts only (ratio-split r4 stays 0 on B4-registry
    spot lines).
    """
    flow = flow_short_syms or set()
    b4_set = b4_etf_syms or set()
    phr_map = b4_phr_by_etf or {}
    u = canonical_symbol(underlying)
    spot_net = 0.0
    need_b1 = need_b2 = need_b4 = 0.0

    if pos is None or pos.empty:
        ratios = SpotBucketRatios(1.0, 0.0, 0.0, "hedge_ratio_empty")
        meta = HedgeRatioSpotMeta(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(ibkr_qty))
        return ratios, meta

    for sym, grp in pos.groupby("symbol"):
        sym_c = canonical_symbol(str(sym))
        if sym_c == u:
            spot_net = sum(
                _position_mv_base_notional(r, underlying=u, etf_to_delta=etf_to_delta)
                for _, r in grp.iterrows()
            )
            continue
        if etf_to_under.get(sym_c) != u:
            continue
        if not _is_etf_leg(sym_c, u, etf_to_under):
            continue
        net = sum(
            _position_mv_base_notional(r, underlying=u, etf_to_delta=etf_to_delta)
            for _, r in grp.iterrows()
        )
        beta = float(etf_to_delta.get(sym_c, 1.0))
        bkt, _ = classify_etf_leg_bucket(
            sym_c, beta, flow_short_set=flow, b4_etf_syms=b4_set
        )
        if bkt == "bucket_1":
            need_b1 += _b12_hedge_spot_usd_for_etf(
                net, beta, bucket=bkt, delta_floor=delta_floor
            )
        elif bkt == "bucket_2":
            need_b2 += _b12_hedge_spot_usd_for_etf(
                net, beta, bucket=bkt, delta_floor=delta_floor
            )
        elif bkt == "bucket_4":
            phr = float(phr_map.get(sym_c, partial_hedge_ratio_default))
            need_b4 += _b4_structural_hedge_usd_for_etf(net, phr)

    has_b1, has_b2, has_b4 = held_etf_bucket_flags_from_positions(
        u, pos, etf_to_under, etf_to_delta, flow_short_syms=flow
    )
    ratios, alloc_b1, alloc_b2 = _hedge_ratio_finalize_spot_ratios(
        spot_net=spot_net,
        need_b1=need_b1,
        need_b2=need_b2,
        has_b1=has_b1,
        has_b2=has_b2,
        has_b4=has_b4,
        b4_b12_only=b4_registry_b12_only,
    )
    _lq = ledger_qty or {}
    _l1 = float(_lq.get("bucket_1", _lq.get("b1", 0.0)) or 0.0)
    _l2 = float(_lq.get("bucket_2", _lq.get("b2", 0.0)) or 0.0)
    _l4 = float(_lq.get("bucket_4", _lq.get("b4", 0.0)) or 0.0)
    _iq = float(ibkr_qty)
    _mark = abs(spot_net / _iq) if abs(_iq) > 1e-12 else 0.0
    if _mark <= 1e-12:
        _px = pos[pos["symbol"].astype(str).map(canonical_symbol) == u]
        if not _px.empty:
            _mark = abs(
                float(_px.iloc[0]["markPrice"]) * float(_px.iloc[0]["fxRateToBase"])
            )
    _tq1 = need_b1 / _mark if _mark > 1e-12 else 0.0
    _tq2 = need_b2 / _mark if _mark > 1e-12 else 0.0
    _tq4 = need_b4 / _mark if _mark > 1e-12 else 0.0
    meta = HedgeRatioSpotMeta(
        hedge_target_usd_b1=need_b1,
        hedge_target_usd_b2=need_b2,
        hedge_target_usd_b4=need_b4,
        hedge_alloc_usd_b1=alloc_b1,
        hedge_alloc_usd_b2=alloc_b2,
        hedge_target_qty_b1=_tq1,
        hedge_target_qty_b2=_tq2,
        hedge_target_qty_b4=_tq4,
        hedge_alloc_qty_b1=alloc_b1 / _mark if _mark > 1e-12 else 0.0,
        hedge_alloc_qty_b2=alloc_b2 / _mark if _mark > 1e-12 else 0.0,
        ledger_qty_b1=_l1,
        ledger_qty_b2=_l2,
        ledger_qty_b4=_l4,
        plan_b4_qty=float(plan_b4_qty),
        ibkr_qty=_iq,
    )
    return ratios, meta


def hedge_ratio_spot_ratios_from_exposure_detail(
    detail: pd.DataFrame,
    *,
    etf_to_under: dict[str, str],
    etf_to_delta_map: dict[str, float],
    flow_short_set: set[str],
    b4_etf_syms: set[str],
    b4_spot_b12_only: set[str] | None = None,
    b4_phr_by_etf: dict[str, float] | None = None,
    partial_hedge_ratio_default: float = 0.75,
    delta_floor: float = 0.25,
    ledger_qty_by_u: dict[str, dict[str, float]] | None = None,
    plan_b4_qty_by_u: dict[str, float] | None = None,
    spot_qty_by_u: dict[str, float] | None = None,
) -> tuple[dict[str, SpotBucketRatios], dict[str, HedgeRatioSpotMeta]]:
    """Hedge-ratio spot split using the same leg nets as ``bucket_exposure_detail``."""
    if detail.empty:
        return {}, {}
    d = detail.copy()
    d["underlying"] = d["underlying"].astype(str)
    d["symbol"] = d["symbol"].astype(str)
    d["net_notional_usd"] = pd.to_numeric(d["net_notional_usd"], errors="coerce").fillna(0.0)
    if "_is_etf" not in d.columns:
        d["_is_etf"] = d.apply(
            lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
            axis=1,
        )
    if "_beta" not in d.columns:
        d["_beta"] = pd.to_numeric(
            d["symbol"].map(etf_to_delta_map), errors="coerce"
        ).fillna(1.0)
    _b4_b12 = b4_spot_b12_only or set()
    phr_map = b4_phr_by_etf or {}
    _ledger = ledger_qty_by_u or {}
    _plan_b4 = plan_b4_qty_by_u or {}
    _spot_q = spot_qty_by_u or {}
    out: dict[str, SpotBucketRatios] = {}
    meta_out: dict[str, HedgeRatioSpotMeta] = {}
    for u, grp in d.groupby("underlying"):
        u_str = str(u)
        spot_net = need_b1 = need_b2 = need_b4 = 0.0
        for _, row in grp.iterrows():
            sym = str(row["symbol"])
            net = float(row["net_notional_usd"])
            if (not bool(row["_is_etf"])) and sym == u_str:
                spot_net += net
                continue
            if not bool(row["_is_etf"]):
                continue
            beta = float(row.get("_beta", etf_to_delta_map.get(sym, 1.0)))
            bkt, _ = classify_etf_leg_bucket(
                sym, beta, flow_short_set=flow_short_set, b4_etf_syms=b4_etf_syms
            )
            if bkt == "bucket_1":
                need_b1 += _b12_hedge_spot_usd_for_etf(
                    net, beta, bucket=bkt, delta_floor=delta_floor
                )
            elif bkt == "bucket_2":
                need_b2 += _b12_hedge_spot_usd_for_etf(
                    net, beta, bucket=bkt, delta_floor=delta_floor
                )
            elif bkt == "bucket_4":
                phr = float(phr_map.get(sym, partial_hedge_ratio_default))
                need_b4 += _b4_structural_hedge_usd_for_etf(net, phr)
        has_b1, has_b2, has_b4 = (
            need_b1 > 1e-12,
            need_b2 > 1e-12,
            need_b4 > 1e-12,
        )
        ratios, alloc_b1, alloc_b2 = _hedge_ratio_finalize_spot_ratios(
            spot_net=spot_net,
            need_b1=need_b1,
            need_b2=need_b2,
            has_b1=has_b1,
            has_b2=has_b2,
            has_b4=has_b4,
            b4_b12_only=u_str in _b4_b12,
        )
        out[u_str] = ratios
        _lq = _ledger.get(u_str, {})
        _iq = float(_spot_q.get(u_str, 0.0))
        _mark = abs(spot_net / _iq) if abs(_iq) > 1e-12 else 0.0
        meta_out[u_str] = HedgeRatioSpotMeta(
            hedge_target_usd_b1=need_b1,
            hedge_target_usd_b2=need_b2,
            hedge_target_usd_b4=need_b4,
            hedge_alloc_usd_b1=alloc_b1,
            hedge_alloc_usd_b2=alloc_b2,
            hedge_target_qty_b1=need_b1 / _mark if _mark > 1e-12 else 0.0,
            hedge_target_qty_b2=need_b2 / _mark if _mark > 1e-12 else 0.0,
            hedge_target_qty_b4=need_b4 / _mark if _mark > 1e-12 else 0.0,
            hedge_alloc_qty_b1=alloc_b1 / _mark if _mark > 1e-12 else 0.0,
            hedge_alloc_qty_b2=alloc_b2 / _mark if _mark > 1e-12 else 0.0,
            ledger_qty_b1=float(_lq.get("bucket_1", 0.0)),
            ledger_qty_b2=float(_lq.get("bucket_2", 0.0)),
            ledger_qty_b4=float(_lq.get("bucket_4", 0.0)),
            plan_b4_qty=float(_plan_b4.get(u_str, 0.0)),
            ibkr_qty=_iq,
        )
    return out, meta_out


def resolve_underlying_spot_exposure_ratios(
    *,
    underlying: str,
    ibkr_qty: float,
    ledger_qty: dict[str, float],
    b12_spot_exposure_method: str = "sleeve_offset",
    b12_spot_split_method: str = "ledger_fifo",
    b4_registry_underlying: bool = False,
    pos: pd.DataFrame | None = None,
    etf_to_under: dict[str, str] | None = None,
    etf_to_delta: dict[str, float] | None = None,
    flow_short_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
    plan_ratio: dict[str, float] | None = None,
    plan_b4_pnl_mode: str = "inject_slice",
    yieldboost_spot_b2: bool = False,
) -> SpotBucketRatios:
    """
    Spot ratios for ratio-split **net exposure** only when ``b12_spot_pnl_method=ledger_fifo``.
    Otherwise spot PnL uses the same sleeve method via ``compose_spot_pnl_bucket_fractions``.

    ``sleeve_offset`` (default): long spot offsets short ETFs in the same bucket;
    unpaired remainder → B1 (or B2 if no B1 ETF).
    ``ledger_fifo`` / ``held_exposure_waterfall``: legacy paths via ``b12_spot_split_method``.
    """
    method = str(b12_spot_exposure_method or "sleeve_offset").strip().lower()
    if method in {"held_delta"}:
        method = "sleeve_offset"
    if method not in {
        "sleeve_offset",
        "ledger_fifo",
        "held_exposure_waterfall",
        "hedge_ratio",
        "sleeve_balance",
    }:
        method = "sleeve_offset"
    if method == "sleeve_balance":
        if pos is None or etf_to_under is None or etf_to_delta is None:
            return resolve_underlying_spot_ratios(
                underlying=underlying,
                ibkr_qty=ibkr_qty,
                ledger_qty=ledger_qty,
                plan_ratio=plan_ratio,
                plan_b4_pnl_mode=plan_b4_pnl_mode,
                yieldboost_spot_b2=yieldboost_spot_b2,
                b12_spot_split_method="ledger_fifo",
                b4_registry_underlying=b4_registry_underlying,
                etf_to_under=etf_to_under,
                etf_to_delta=etf_to_delta,
                pos=pos,
                flow_short_syms=flow_short_syms,
            )
        return sleeve_balance_spot_bucket_ratios(
            underlying,
            pos,
            etf_to_under,
            etf_to_delta,
            flow_short_syms=flow_short_syms,
            b4_etf_syms=b4_etf_syms,
            b4_registry_b12_only=b4_registry_underlying,
        )
    if method == "hedge_ratio":
        if pos is None or etf_to_under is None or etf_to_delta is None:
            return resolve_underlying_spot_ratios(
                underlying=underlying,
                ibkr_qty=ibkr_qty,
                ledger_qty=ledger_qty,
                plan_ratio=plan_ratio,
                plan_b4_pnl_mode=plan_b4_pnl_mode,
                yieldboost_spot_b2=yieldboost_spot_b2,
                b12_spot_split_method="ledger_fifo",
                b4_registry_underlying=b4_registry_underlying,
                etf_to_under=etf_to_under,
                etf_to_delta=etf_to_delta,
                pos=pos,
                flow_short_syms=flow_short_syms,
            )
        _hr_ratio, _ = hedge_ratio_spot_bucket_ratios(
            underlying,
            pos,
            etf_to_under,
            etf_to_delta,
            flow_short_syms=flow_short_syms,
            b4_etf_syms=b4_etf_syms,
            b4_registry_b12_only=b4_registry_underlying,
            ledger_qty=ledger_qty,
            ibkr_qty=ibkr_qty,
        )
        return _hr_ratio
    if method == "sleeve_offset":
        if pos is None or etf_to_under is None or etf_to_delta is None:
            return resolve_underlying_spot_ratios(
                underlying=underlying,
                ibkr_qty=ibkr_qty,
                ledger_qty=ledger_qty,
                plan_ratio=plan_ratio,
                plan_b4_pnl_mode=plan_b4_pnl_mode,
                yieldboost_spot_b2=yieldboost_spot_b2,
                b12_spot_split_method="ledger_fifo",
                b4_registry_underlying=b4_registry_underlying,
                etf_to_under=etf_to_under,
                etf_to_delta=etf_to_delta,
                pos=pos,
                flow_short_syms=flow_short_syms,
            )
        return sleeve_offset_spot_bucket_ratios(
            underlying,
            pos,
            etf_to_under,
            etf_to_delta,
            flow_short_syms=flow_short_syms,
            b4_etf_syms=b4_etf_syms,
            b4_registry_b12_only=b4_registry_underlying,
        )
    split = (
        "held_exposure_waterfall"
        if method == "held_exposure_waterfall"
        else str(b12_spot_split_method or "ledger_fifo").strip().lower()
    )
    return resolve_underlying_spot_ratios(
        underlying=underlying,
        ibkr_qty=ibkr_qty,
        ledger_qty=ledger_qty,
        plan_ratio=plan_ratio,
        plan_b4_pnl_mode=plan_b4_pnl_mode,
        yieldboost_spot_b2=yieldboost_spot_b2,
        b12_spot_split_method=split,
        b4_registry_underlying=b4_registry_underlying,
        etf_to_under=etf_to_under,
        etf_to_delta=etf_to_delta,
        pos=pos,
        flow_short_syms=flow_short_syms,
    )


def classify_etf_leg_bucket(
    symbol: str,
    beta: float,
    *,
    flow_short_set: set[str],
    b4_etf_syms: set[str] | None = None,
) -> tuple[str, str]:
    """Return (bucket, leg_class) for an ETF leg."""
    sym = canonical_symbol(symbol)
    b4_set = b4_etf_syms or set()
    if beta < 0:
        if sym in flow_short_set:
            return "bucket_3", "flow_inverse"
        return "bucket_4", "inverse_b4_etf"
    if sym in b4_set:
        return "bucket_4", "inverse_b4_etf"
    if beta > 1.5:
        return "bucket_1", "core_levered_etf"
    if beta > 0:
        leg = "flow_low_delta" if sym in flow_short_set else "yieldboost_etf"
        return "bucket_2", leg
    return "bucket_1", "core_levered_etf"


def _hedge_offset_spot(remaining: float, etf_net: float) -> float:
    """Spot notional assigned to a bucket to offset ``etf_net``, bounded by ``remaining``."""
    if abs(remaining) <= 1e-12 or abs(etf_net) <= 1e-12:
        return 0.0
    target = -etf_net
    if remaining > 0:
        return min(target, remaining) if target > 0 else 0.0
    return max(target, remaining) if target < 0 else 0.0


def hedge_residual_spot_slices(
    spot_net: float,
    etf_b1: float,
    etf_b4: float,
    *,
    b4_short_target: float = 0.0,
) -> tuple[float, float, float]:
    """
    Waterfall spot split for sleeve-level exposure.

    When ``b4_short_target`` is non-zero we carve that signed slice off the
    spot line first (typically negative for inverse-decay pairs where the
    sleeve structurally shorts the underlying alongside the short inverse
    ETF). The remaining spot then offsets B1 levered-ETF shorts; whatever
    is left lands in B2.

    When ``b4_short_target=0.0`` the function falls back to the legacy
    offset waterfall (B1 first, then B4) so existing call sites that don't
    supply an explicit structural target keep their old behaviour.
    """
    spot_b4 = float(b4_short_target)
    remaining = float(spot_net) - spot_b4
    spot_b1 = _hedge_offset_spot(remaining, float(etf_b1))
    remaining -= spot_b1
    if abs(spot_b4) <= 1e-12:
        spot_b4 = _hedge_offset_spot(remaining, float(etf_b4))
        remaining -= spot_b4
    return spot_b1, remaining, spot_b4


def plan_structural_spot_slices(
    spot_net: float,
    plan_b4_notional: float,
) -> tuple[float, float, float]:
    """
    Gross B4-plan decomposition on the physical spot line.

    ``plan_b4_notional`` is the signed structural short (negative USD). The B1
    long residual is whatever remains so spot_b1 + spot_b4 == spot_net.
    """
    spot_b4 = float(plan_b4_notional)
    spot_b1 = float(spot_net) - spot_b4
    return spot_b1, 0.0, spot_b4


def build_hedge_residual_spot_ratio_map(
    exposure_detail: pd.DataFrame,
    *,
    etf_to_under: dict[str, str],
    etf_to_delta_map: dict[str, float],
    flow_short_set: set[str],
    b4_etf_syms: set[str],
    plan_ratio_b4: dict[str, dict[str, float]] | None = None,
    plan_b4_qty_by_u: dict[str, float] | None = None,
    spot_qty: dict[str, float] | None = None,
    spot_marks: dict[str, float] | None = None,
    plan_sleeve_usd: dict[str, dict[str, float]] | None = None,
    etf_implied_short_usd: dict[str, float] | None = None,
    b4_attribution_mode: str = "etf_implied",
    b4_attribution_min_usd: float = 500.0,
) -> dict[str, SpotBucketRatios]:
    """
    Per-underlying spot ratios for bucket exposure.

    Source priority for the structural B4 short underlying leg:

    1. **plan_structural_short** — when ``plan_ratio_b4[u]`` carries an explicit
       sleeve target from the latest ``inverse_decay_bucket4`` trade plan row.
    2. **etf_implied_short** — when ``b4_attribution_mode == 'etf_implied'``
       and ``|etf_implied_short_usd[u]| >= b4_attribution_min_usd``. This is
       computed from held inverse ETF positions × β × partial_hedge_ratio
       and lets the accounting recover the structural short even after the
       rebalancer has netted the sleeve order into a single IBKR stock line
       and the FIFO ``qty_b4`` ledger has lost the tag.
    3. **hedge_residual** — fall back to the offset waterfall (no structural
       short carved out).

    Mode ``plan_only`` disables source (2); mode ``ledger_fifo`` disables both
    (1) and (2) so the legacy hedge-residual behaviour is restored.
    """
    if exposure_detail.empty:
        return {}
    plan_ratio_b4 = plan_ratio_b4 or {}
    plan_b4_qty_by_u = plan_b4_qty_by_u or {}
    spot_qty = spot_qty or {}
    spot_marks = spot_marks or {}
    plan_sleeve_usd = plan_sleeve_usd or {}
    etf_implied_short_usd = etf_implied_short_usd or {}
    attribution_mode = str(b4_attribution_mode or "etf_implied").strip().lower()
    if attribution_mode not in {"plan_only", "etf_implied", "ledger_fifo"}:
        attribution_mode = "etf_implied"
    d = exposure_detail.copy()
    d["underlying"] = d["underlying"].astype(str)
    d["symbol"] = d["symbol"].astype(str)
    if "_is_etf" not in d.columns:
        d["_is_etf"] = d.apply(
            lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
            axis=1,
        )
    if "_beta" not in d.columns:
        d["_beta"] = pd.to_numeric(d["symbol"].map(etf_to_delta_map), errors="coerce").fillna(1.0)
    d["net_notional_usd"] = pd.to_numeric(d["net_notional_usd"], errors="coerce").fillna(0.0)

    out: dict[str, SpotBucketRatios] = {}
    for u, grp in d.groupby("underlying"):
        etf_b1 = etf_b4 = 0.0
        spot_net = 0.0
        for _, row in grp.iterrows():
            sym = str(row["symbol"])
            net = float(row["net_notional_usd"])
            if bool(row["_is_etf"]):
                bkt, _ = classify_etf_leg_bucket(
                    sym,
                    float(row["_beta"]),
                    flow_short_set=flow_short_set,
                    b4_etf_syms=b4_etf_syms,
                )
                if bkt == "bucket_1":
                    etf_b1 += net
                elif bkt == "bucket_4":
                    etf_b4 += net
            elif sym == u:
                spot_net += net
        if abs(spot_net) <= 1e-12:
            continue
        if attribution_mode != "ledger_fifo" and u in plan_ratio_b4:
            _qty = float(spot_qty.get(u, 0.0))
            if abs(_qty) > 1e-12:
                _mark = abs(spot_net / _qty)
            else:
                _mark = float(spot_marks.get(u, 0.0))
            plan_b4_notional = float(plan_b4_qty_by_u.get(u, 0.0)) * _mark
            if abs(plan_b4_notional) <= 1e-9:
                plan_b4_notional = float((plan_sleeve_usd.get(u) or {}).get("b4", 0.0))
            spot_b1, spot_b2, spot_b4 = plan_structural_spot_slices(spot_net, plan_b4_notional)
            out[u] = SpotBucketRatios(
                spot_b1 / spot_net,
                spot_b2 / spot_net,
                spot_b4 / spot_net,
                "plan_structural_short",
            )
            continue
        if attribution_mode == "etf_implied":
            implied_b4_usd = float(etf_implied_short_usd.get(u, 0.0))
            if abs(implied_b4_usd) >= float(b4_attribution_min_usd):
                spot_b1, spot_b2, spot_b4 = plan_structural_spot_slices(spot_net, implied_b4_usd)
                out[u] = SpotBucketRatios(
                    spot_b1 / spot_net,
                    spot_b2 / spot_net,
                    spot_b4 / spot_net,
                    "etf_implied_short",
                )
                continue
        etf_gross = abs(etf_b1) + abs(etf_b4)
        if etf_gross <= 1e-12:
            out[u] = SpotBucketRatios(1.0, 0.0, 0.0, "hedge_residual_spot_only")
            continue
        spot_b1, spot_b2, spot_b4 = hedge_residual_spot_slices(spot_net, etf_b1, etf_b4)
        out[u] = SpotBucketRatios(
            spot_b1 / spot_net,
            spot_b2 / spot_net,
            spot_b4 / spot_net,
            "hedge_residual",
        )
    return out


def build_net_exposure_spot_by_underlying(
    exposure_detail: pd.DataFrame,
    spot_ratio_map: dict[str, SpotBucketRatios],
) -> pd.DataFrame:
    """Spot-line exposure split using canonical ``spot_ratio_map`` ratios."""
    if exposure_detail.empty or not spot_ratio_map:
        return pd.DataFrame(
            columns=[
                "underlying",
                "net_notional_usd",
                "gross_notional_usd",
                "ratio_spot_b1",
                "ratio_spot_b2",
                "ratio_spot_b4",
                "ratio_spot_source",
                "net_bucket_1",
                "net_bucket_2",
                "net_bucket_4",
            ]
        )
    spot = exposure_detail[
        (~exposure_detail["_is_etf"])
        & (exposure_detail["symbol"].astype(str) == exposure_detail["underlying"].astype(str))
    ].copy()
    rows: list[dict] = []
    for _, r in spot.iterrows():
        u = str(r["underlying"])
        sr = spot_ratio_map.get(u)
        if sr is None:
            continue
        net = float(r["net_notional_usd"])
        gross = float(r["gross_notional_usd"])
        rows.append(
            {
                "underlying": u,
                "net_notional_usd": net,
                "gross_notional_usd": gross,
                "ratio_spot_b1": sr.b1,
                "ratio_spot_b2": sr.b2,
                "ratio_spot_b4": sr.b4,
                "ratio_spot_source": sr.source,
                "net_bucket_1": net * sr.b1,
                "net_bucket_2": net * sr.b2,
                "net_bucket_4": net * sr.b4,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("net_notional_usd", key=lambda s: s.abs(), ascending=False)


def build_bucket_ratio_reconciliation(
    pnl_df: pd.DataFrame,
    spot_exposure_df: pd.DataFrame,
    spot_ratio_map: dict[str, SpotBucketRatios],
    *,
    min_abs_pnl_usd: float = 100.0,
    min_abs_net_usd: float = 1000.0,
) -> tuple[pd.DataFrame, float, float]:
    """
    Compare spot PnL bucket shares vs exposure spot shares (canonical ratios).

    Returns (detail dataframe, max exposure-vs-canonical diff, max pnl-vs-canonical diff).
    """
    cols = [
        "underlying",
        "pnl_total",
        "pnl_share_b1",
        "pnl_share_b2",
        "pnl_share_b4",
        "exp_net_total",
        "exp_share_b1",
        "exp_share_b2",
        "exp_share_b4",
        "ratio_spot_b1",
        "ratio_spot_b2",
        "ratio_spot_b4",
        "diff_b1",
        "diff_b2",
        "diff_b4",
        "diff_exp_b1",
        "diff_exp_b2",
        "diff_exp_b4",
        "diff_pnl_b1",
        "diff_pnl_b2",
        "diff_pnl_b4",
        "max_diff",
    ]
    if pnl_df.empty or not spot_ratio_map:
        return pd.DataFrame(columns=cols), 0.0, 0.0

    exp_by_u: dict[str, dict[str, float]] = {}
    if not spot_exposure_df.empty:
        for _, r in spot_exposure_df.iterrows():
            u = str(r["underlying"])
            exp_by_u[u] = {
                "net": float(r["net_notional_usd"]),
                "b1": float(r.get("net_bucket_1", 0.0)),
                "b2": float(r.get("net_bucket_2", 0.0)),
                "b4": float(r.get("net_bucket_4", 0.0)),
            }

    rows: list[dict] = []
    max_diff_exp = 0.0
    max_diff_pnl = 0.0
    spot_buckets = {"bucket_1", "bucket_2", "bucket_4"}
    for u, sr in spot_ratio_map.items():
        spot_pnl = pnl_df[
            (pnl_df["symbol"].astype(str) == u)
            & (pnl_df["underlying"].astype(str) == u)
            & pnl_df["bucket"].isin(spot_buckets)
        ]
        pnl_b1 = float(spot_pnl.loc[spot_pnl["bucket"] == "bucket_1", "total_pnl"].sum())
        pnl_b2 = float(spot_pnl.loc[spot_pnl["bucket"] == "bucket_2", "total_pnl"].sum())
        pnl_b4 = float(spot_pnl.loc[spot_pnl["bucket"] == "bucket_4", "total_pnl"].sum())
        pnl_total = pnl_b1 + pnl_b2 + pnl_b4
        exp = exp_by_u.get(u, {})
        exp_net = float(exp.get("net", 0.0))
        if abs(pnl_total) < min_abs_pnl_usd and abs(exp_net) < min_abs_net_usd:
            continue

        pnl_abs = abs(pnl_b1) + abs(pnl_b2) + abs(pnl_b4)
        if pnl_abs > 1e-6:
            pnl_s1, pnl_s2, pnl_s4 = abs(pnl_b1) / pnl_abs, abs(pnl_b2) / pnl_abs, abs(pnl_b4) / pnl_abs
        else:
            pnl_s1, pnl_s2, pnl_s4 = sr.b1, sr.b2, sr.b4

        if abs(exp_net) > 1e-6:
            exp_s1 = float(exp.get("b1", 0.0)) / exp_net
            exp_s2 = float(exp.get("b2", 0.0)) / exp_net
            exp_s4 = float(exp.get("b4", 0.0)) / exp_net
        else:
            exp_s1, exp_s2, exp_s4 = sr.b1, sr.b2, sr.b4

        d_exp1, d_exp2, d_exp4 = abs(exp_s1 - sr.b1), abs(exp_s2 - sr.b2), abs(exp_s4 - sr.b4)
        d_pnl1, d_pnl2, d_pnl4 = abs(pnl_s1 - sr.b1), abs(pnl_s2 - sr.b2), abs(pnl_s4 - sr.b4)
        row_max_exp = max(d_exp1, d_exp2, d_exp4)
        row_max_pnl = max(d_pnl1, d_pnl2, d_pnl4)
        row_max = max(row_max_exp, row_max_pnl)
        if abs(exp_net) >= min_abs_net_usd:
            max_diff_exp = max(max_diff_exp, row_max_exp)
        if abs(pnl_total) >= min_abs_pnl_usd:
            max_diff_pnl = max(max_diff_pnl, row_max_pnl)
        rows.append(
            {
                "underlying": u,
                "pnl_total": pnl_total,
                "pnl_share_b1": pnl_s1,
                "pnl_share_b2": pnl_s2,
                "pnl_share_b4": pnl_s4,
                "exp_net_total": exp_net,
                "exp_share_b1": exp_s1,
                "exp_share_b2": exp_s2,
                "exp_share_b4": exp_s4,
                "ratio_spot_b1": sr.b1,
                "ratio_spot_b2": sr.b2,
                "ratio_spot_b4": sr.b4,
                "diff_exp_b1": d_exp1,
                "diff_exp_b2": d_exp2,
                "diff_exp_b4": d_exp4,
                "diff_pnl_b1": d_pnl1,
                "diff_pnl_b2": d_pnl2,
                "diff_pnl_b4": d_pnl4,
                "max_diff": row_max,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("max_diff", ascending=False)
    return out, max_diff_exp, max_diff_pnl


def build_audit_full_bucket_exposure(
    exposure_detail: pd.DataFrame,
    *,
    etf_to_under: dict[str, str],
    etf_to_delta_map: dict[str, float],
    flow_short_set: set[str],
    b4_etf_syms: set[str],
    spot_ratio_map: dict[str, SpotBucketRatios],
    spot_qty: dict[str, float],
    ledger_qty_map: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit view mirroring primary bucket-1/2/4 exposure (β 0–1.5 ETFs → bucket 2)."""
    empty = pd.DataFrame(
        columns=["underlying", "symbols", "net_notional_usd", "gross_notional_usd", "n_legs"]
    )
    if exposure_detail.empty:
        return empty, empty, empty

    _d = exposure_detail.copy()
    _d["symbol"] = _d["symbol"].astype(str)
    _d["underlying"] = _d["underlying"].astype(str)
    if "_is_etf" not in _d.columns:
        _d["_is_etf"] = _d.apply(
            lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
            axis=1,
        )
    if "_beta" not in _d.columns:
        _d["_beta"] = pd.to_numeric(_d["symbol"].map(etf_to_delta_map), errors="coerce").fillna(1.0)

    def _audit_ratios(row: pd.Series) -> tuple[float, float, float]:
        sym = str(row["symbol"])
        u = str(row["underlying"])
        if bool(row["_is_etf"]):
            beta = float(row["_beta"])
            if beta < 0:
                if sym in flow_short_set:
                    return 0.0, 0.0, 0.0
                return 0.0, 0.0, 1.0
            if sym in b4_etf_syms:
                return 0.0, 0.0, 1.0
            if beta > 1.5:
                return 1.0, 0.0, 0.0
            if beta > 0:
                return 0.0, 1.0, 0.0
            return 1.0, 0.0, 0.0
        if sym != u:
            return 0.0, 0.0, 0.0
        sr = spot_ratio_map.get(u)
        if sr is not None:
            return sr.b1, sr.b2, sr.b4
        ledger = ledger_qty_map.get(u, {})
        return _spot_exposure_bucket_ratios(float(spot_qty.get(sym, 0.0)), ledger)

    ratios = _d.apply(_audit_ratios, axis=1, result_type="expand")
    _d["_ratio_b1"] = ratios[0]
    _d["_ratio_b2"] = ratios[1]
    _d["_ratio_b4"] = ratios[2]
    _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b1"] = 0.0
    _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b2"] = 0.0
    _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b4"] = 1.0
    _d = _normalize_exposure_bucket_ratios(
        _d,
        etf_to_delta_map=etf_to_delta_map,
        flow_short_set=flow_short_set,
        b4_etf_syms=b4_etf_syms,
    )

    def _scale(_detail: pd.DataFrame, col: str) -> pd.DataFrame:
        out = _detail[_detail[col].abs() > 1e-12].copy()
        out["net_notional_usd"] = out["net_notional_usd"] * out[col]
        out["gross_notional_usd"] = out["gross_notional_usd"] * out[col]
        return out

    def _agg(_detail: pd.DataFrame) -> pd.DataFrame:
        if _detail.empty:
            return empty.copy()
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

    b1 = _agg(_scale(_d, "_ratio_b1"))
    b2 = _agg(_scale(_d, "_ratio_b2"))
    b4 = _agg(_scale(_d, "_ratio_b4"))
    return b1, b2, b4


def _normalize_exposure_bucket_ratios(
    detail: pd.DataFrame,
    *,
    etf_to_delta_map: dict[str, float],
    flow_short_set: set[str],
    b4_etf_syms: set[str],
    b4_spot_b12_only_underlyings: set[str] | None = None,
    preserve_partial_spot_ratios: bool = False,
) -> pd.DataFrame:
    """Ensure each exposure leg's bucket ratios sum to 1 (or 0 if flat).

    When ``preserve_partial_spot_ratios`` (``sleeve_balance`` exposure), do not
    push unattributed spot orphan into B1 and do not renormalize B1+B2 to 1 on
    B4-registry names before the structural carve.
    """
    d = detail.copy()
    _b4_spot_b12 = b4_spot_b12_only_underlyings or set()
    for idx, row in d.iterrows():
        sym = str(row["symbol"])
        if sym in b4_etf_syms:
            continue
        is_spot = (not bool(row.get("_is_etf", False))) and sym == str(row["underlying"])
        if is_spot and str(row["underlying"]) in _b4_spot_b12:
            # Structural B4 names: ratio-split spot is B1/B2 only until carve.
            d.at[idx, "_ratio_b4"] = 0.0
            if not preserve_partial_spot_ratios:
                r1 = float(d.at[idx, "_ratio_b1"] or 0.0)
                r2 = float(d.at[idx, "_ratio_b2"] or 0.0)
                s12 = r1 + r2
                if s12 > 1e-12:
                    d.at[idx, "_ratio_b1"] = r1 / s12
                    d.at[idx, "_ratio_b2"] = r2 / s12
            continue
        r1 = float(row.get("_ratio_b1", 0.0) or 0.0)
        r2 = float(row.get("_ratio_b2", 0.0) or 0.0)
        r4 = float(row.get("_ratio_b4", 0.0) or 0.0)
        gross = float(row.get("gross_notional_usd", 0.0) or 0.0)
        if gross <= 1e-12:
            continue
        ratio_sum = r1 + r2 + r4
        # Signed plan split: B4 short + B1 long gross decomposition (sum == 1).
        is_spot = (not bool(row.get("_is_etf", False))) and sym == str(row["underlying"])
        if is_spot and r4 < -1e-12:
            continue
        if ratio_sum > 1e-12:
            if ratio_sum < 1.0 - 1e-9 and not (
                preserve_partial_spot_ratios and is_spot
            ):
                d.at[idx, "_ratio_b1"] = r1 + (1.0 - ratio_sum)
            elif ratio_sum > 1.0 + 1e-9:
                scale = 1.0 / ratio_sum
                d.at[idx, "_ratio_b1"] = r1 * scale
                d.at[idx, "_ratio_b2"] = r2 * scale
                d.at[idx, "_ratio_b4"] = r4 * scale
            continue
        beta = float(etf_to_delta_map.get(sym, 1.0))
        if bool(row.get("_is_etf", False)):
            if beta > 1.5:
                d.at[idx, "_ratio_b1"] = 1.0
            elif (beta > 0) and (beta <= 1.5):
                d.at[idx, "_ratio_b2"] = 1.0
            elif beta < 0 and sym not in flow_short_set:
                d.at[idx, "_ratio_b4"] = 1.0
            else:
                d.at[idx, "_ratio_b1"] = 1.0
        else:
            d.at[idx, "_ratio_b1"] = 1.0
    return d


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
    cols = ["etf", "underlying", "delta", "partial_hedge_ratio"]
    etf_to_under, etf_to_delta = load_etf_delta_map(screened_csv)
    for sym, und in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(sym, und)
        etf_to_delta.setdefault(sym, etf_to_delta.get(sym, -1.0))

    flow = flow_short_syms or set()
    rows: list[dict] = []
    for etf, under in etf_to_under.items():
        delta = float(etf_to_delta.get(etf, 0.0))
        if delta >= 0 or etf in flow:
            continue
        rows.append(
            {
                "etf": canonical_symbol(etf),
                "underlying": canonical_symbol(under),
                "delta": delta,
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
                        delta = float(etf_to_delta.get(etf, -1.0))
                        rows.append(
                            {
                                "etf": etf,
                                "underlying": under,
                                "delta": delta,
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


_PLAN_SLEEVE_TO_BUCKET: dict[str, str] = {
    "core_leveraged": "b1",
    "yieldboost": "b2",
    "inverse_decay_bucket4": "b4",
}


def _plan_row_is_yieldboost(row: pd.Series) -> bool:
    val = row.get("is_yieldboost", False)
    return _plan_row_is_yieldboost_scalar(val)


def _plan_row_is_yieldboost_scalar(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y"}
    return bool(val)


def _resolve_plan_row_beta(row: pd.Series, etf_to_delta: dict[str, float]) -> float:
    etf = canonical_symbol(str(row.get("ETF", "") or ""))
    for col in ("Delta", "delta"):
        if col in row.index:
            raw = row.get(col)
            if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                try:
                    beta = float(raw)
                except (TypeError, ValueError):
                    continue
                if abs(beta) > 1e-12:
                    return beta
    return float(etf_to_delta.get(etf, 0.0))


def _plan_row_bucket_key(
    row: pd.Series,
    etf_to_delta: dict[str, float],
    *,
    sleeve_first: bool,
) -> str | None:
    if sleeve_first and "sleeve" in row.index:
        sleeve = str(row.get("sleeve", "") or "").strip()
        bucket = _PLAN_SLEEVE_TO_BUCKET.get(sleeve)
        if bucket:
            return bucket
    beta = _resolve_plan_row_beta(row, etf_to_delta)
    if beta < 0:
        return "b4"
    if beta > 1.5:
        return "b1"
    if beta > 0:
        return "b2"
    if _plan_row_is_yieldboost(row):
        return "b2"
    return None


def merge_plan_etf_metadata(
    plan_csv: Path | None,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
) -> tuple[dict[str, str], dict[str, float]]:
    """Fill ETF→underlying and ETF→delta gaps from the latest trade plan."""
    out_under = dict(etf_to_under)
    out_delta = dict(etf_to_delta)
    if plan_csv is None or not Path(plan_csv).exists():
        return out_under, out_delta
    try:
        df = pd.read_csv(plan_csv)
    except Exception:
        return out_under, out_delta
    if df.empty:
        return out_under, out_delta
    cols_lc = {c.lower(): c for c in df.columns}
    etf_col = _find_col(cols_lc, ["etf", "symbol", "ticker"])
    under_col = _find_col(
        cols_lc, ["underlying", "underlyingsymbol", "underlying_symbol", "root"]
    )
    if not etf_col or not under_col:
        return out_under, out_delta
    for _, row in df.iterrows():
        etf = canonical_symbol(str(row.get(etf_col, "") or ""))
        under = canonical_symbol(str(row.get(under_col, "") or ""))
        if not etf or not under:
            continue
        out_under.setdefault(etf, under)
        if etf not in out_delta:
            beta = _resolve_plan_row_beta(row, out_delta)
            if abs(beta) > 1e-12:
                out_delta[etf] = beta
    return out_under, out_delta


def _normalize_b1_b2_pair(r_b1: float, r_b2: float) -> tuple[float, float]:
    s = abs(float(r_b1)) + abs(float(r_b2))
    if s <= 1e-12:
        return 1.0, 0.0
    return abs(float(r_b1)) / s, abs(float(r_b2)) / s


def ledger_pnl_split_b1_b2_ratios(
    *,
    orphan_frac: float,
    ledger_unreal_r_b1: float,
    ledger_unreal_r_b2: float,
    orphan_threshold: float = 0.10,
) -> tuple[float, float]:
    """B1/B2 weights for spot PnL overrides: always FIFO ledger unrealized mix.

    When the share ledger is stale (high orphan), do **not** fall back to
    held-ETF exposure weights — use ledger unrealized ratios, or 100% B1 if
    the ledger has no B2 sleeve.
    """
    if orphan_frac > orphan_threshold and (
        abs(ledger_unreal_r_b1) + abs(ledger_unreal_r_b2) <= 1e-12
    ):
        return 1.0, 0.0
    return float(ledger_unreal_r_b1), float(ledger_unreal_r_b2)


def collect_bucket2_underlyings(
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    *,
    flow_short_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
    pos: pd.DataFrame | None = None,
) -> set[str]:
    """Underlyings with at least one bucket-2 ETF in the delta map or open book."""
    flow = flow_short_syms or set()
    b4_set = b4_etf_syms or set()
    out: set[str] = set()
    for sym, beta in etf_to_delta.items():
        bkt, _ = classify_etf_leg_bucket(
            sym, float(beta), flow_short_set=flow, b4_etf_syms=b4_set
        )
        if bkt != "bucket_2":
            continue
        u = canonical_symbol(str(etf_to_under.get(sym, sym)))
        if u:
            out.add(u)
    if pos is not None and not pos.empty:
        for sym_raw in pos["symbol"].astype(str).unique():
            sym = canonical_symbol(sym_raw)
            if sym not in etf_to_under:
                continue
            beta = float(etf_to_delta.get(sym, 0.0))
            bkt, _ = classify_etf_leg_bucket(
                sym, beta, flow_short_set=flow, b4_etf_syms=b4_set
            )
            if bkt == "bucket_2":
                out.add(canonical_symbol(str(etf_to_under.get(sym, sym))))
    return out


def apply_yieldboost_spot_b2_override(
    *,
    lot_components: dict[str, dict[str, dict[str, float]]],
    underlying: str,
    df: pd.DataFrame,
    spot_carry_cols: set[str],
    r_b1: float,
    r_b2: float,
    r_b4: float = 0.0,
    force_all_b2: bool = False,
) -> None:
    """Attribute IBKR spot PnL to bucket 2 for yieldboost pairs (no B4 plan slice).

    When ``force_all_b2`` (held B2 ETF sleeve present), the full spot line
    rolls into bucket 2 after any B4 slice. Otherwise uses ``r_b1``/``r_b2``
    (typically FIFO ledger unrealized weights).
    """
    u_rows = df[
        (df["underlying"] == underlying)
        & (df["symbol"].astype(str) == df["underlying"].astype(str))
    ]
    if u_rows.empty:
        return
    override_cols = {"realized_pnl", "unrealized_pnl"} | set(spot_carry_cols)
    if force_all_b2:
        b1_norm, b2_norm = 0.0, 1.0
    else:
        b1_norm, b2_norm = _normalize_b1_b2_pair(r_b1, r_b2)
    b4_frac = max(float(r_b4), 0.0)
    for col in override_cols:
        if col not in u_rows.columns:
            continue
        total = float(u_rows[col].sum())
        if abs(total) <= 1e-9:
            for bk in ("bucket_1", "bucket_2", "bucket_4"):
                lot_components[underlying][bk][col] = 0.0
            continue
        b4_part = total * b4_frac
        remainder = total - b4_part
        lot_components[underlying]["bucket_4"][col] = b4_part
        lot_components[underlying]["bucket_1"][col] = remainder * b1_norm
        lot_components[underlying]["bucket_2"][col] = remainder * b2_norm


def b4_spot_pnl_inject_eligible(
    underlying: str,
    *,
    ledger_b4_qty: float,
    implied_b4_short_usd: dict[str, float],
    min_usd: float,
) -> bool:
    """Inject plan B4 unrealized only when FIFO ledger has tagged B4 spot qty."""
    _ = underlying, implied_b4_short_usd, min_usd
    return abs(float(ledger_b4_qty)) > 1e-9


def apply_plan_b4_spot_pnl_override(
    *,
    lot_components: dict[str, dict[str, dict[str, float]]],
    underlying: str,
    df: pd.DataFrame,
    plan_ratio: dict[str, float],
    spot_carry_cols: set[str],
    etf_to_delta_map: dict[str, float],
    mode: str,
    ledger_r_b1: float,
    ledger_r_b2: float,
    b4_frac_signed: float | None = None,
) -> None:
    """Redistribute IBKR spot PnL across buckets using a B4 ratio.

    ``plan_ratio`` historically came from the latest ``inverse_decay_bucket4``
    plan row and carried a normalised positive ``b4`` magnitude. With the
    canonical attribution rule the B4 sleeve runs as a structural *short*
    underlying leg, so the PnL split must use the **signed** ratio
    ``spot_b4 / spot_net`` (negative when short) to attribute correctly
    against the broker line. Callers pass that value via ``b4_frac_signed``;
    when omitted the function falls back to ``plan_ratio['b4']`` for
    backward compatibility.

    In ``inject_slice`` mode (default) **realized** PnL is left on the FIFO
    lot-ledger bucket assignments already present in ``lot_components``;
    only **unrealized** PnL and carry columns are re-split by the B4 ratio.
    This avoids re-carving cumulative trade PnL when plan exposure changes
    day-to-day (e.g. large hedge closes landing in B1 while B4 gets an
    opposite-sign realized slice). ``full_override`` still replaces all
    columns including realized.
    """
    u_rows = df[
        (df["underlying"] == underlying)
        & (df["symbol"].astype(str) == df["underlying"].astype(str))
    ]
    if u_rows.empty:
        return
    if mode == "full_override":
        override_cols = {"realized_pnl", "unrealized_pnl"} | set(spot_carry_cols)
    else:
        override_cols = {"unrealized_pnl"} | set(spot_carry_cols)
    bucket_keys = ("bucket_1", "bucket_2", "bucket_4")
    b1_norm, b2_norm = _normalize_b1_b2_pair(ledger_r_b1, ledger_r_b2)
    b4_frac = (
        float(b4_frac_signed)
        if b4_frac_signed is not None
        else float(plan_ratio.get("b4", 0.0))
    )

    for col in override_cols:
        if col not in u_rows.columns:
            continue
        total = float(u_rows[col].sum())
        if abs(total) <= 1e-9:
            for bk in bucket_keys:
                lot_components[underlying][bk][col] = 0.0
            continue
        if mode == "full_override":
            for bk, rk in (
                ("bucket_1", "b1"),
                ("bucket_2", "b2"),
                ("bucket_4", "b4"),
            ):
                lot_components[underlying][bk][col] = total * float(plan_ratio.get(rk, 0.0))
        else:
            b4_part = total * b4_frac
            remainder = total - b4_part
            lot_components[underlying]["bucket_4"][col] = b4_part
            lot_components[underlying]["bucket_1"][col] = remainder * b1_norm
            lot_components[underlying]["bucket_2"][col] = remainder * b2_norm


def collapse_spot_b4_pnl_into_b12(
    lot_components: dict[str, dict[str, dict[str, float]]],
    underlying: str,
    *,
    ledger_r_b1: float,
    ledger_r_b2: float,
    spot_carry_cols: set[str],
) -> None:
    """Move spot-line bucket_4 PnL into B1/B2 before live B4 trading begins."""
    u = lot_components.get(underlying)
    if not u or "bucket_4" not in u:
        return
    cols = {"realized_pnl", "unrealized_pnl"} | set(spot_carry_cols)
    b1n, b2n = _normalize_b1_b2_pair(ledger_r_b1, ledger_r_b2)
    for col in cols:
        v4 = float(u["bucket_4"].get(col, 0.0) or 0.0)
        if abs(v4) <= 1e-9:
            continue
        u["bucket_1"][col] = float(u["bucket_1"].get(col, 0.0) or 0.0) + v4 * b1n
        u["bucket_2"][col] = float(u["bucket_2"].get(col, 0.0) or 0.0) + v4 * b2n
        u["bucket_4"][col] = 0.0


def compose_spot_pnl_bucket_fractions(
    spot_exp_ratio: SpotBucketRatios,
    *,
    b4_frac_signed: float | None = None,
) -> tuple[float, float, float, str]:
    """
    Final spot-line b1/b2/b4 fractions for PnL (sum to 1).

    Mirrors exposure: sleeve-balance/offset B1/B2 hedge needs first, then carve
    the structural B4 short underlying slice when ``b4_frac_signed`` is set
    (``spot_b4 / spot_net``, negative when short).
    """
    hr_b1 = float(spot_exp_ratio.b1)
    hr_b2 = float(spot_exp_ratio.b2)
    r4 = float(spot_exp_ratio.b4)
    source = str(spot_exp_ratio.source)

    if b4_frac_signed is not None and abs(b4_frac_signed) > 1e-12:
        r4 = float(b4_frac_signed)
        rem = 1.0 - r4
        s12 = hr_b1 + hr_b2
        if s12 > 1e-12:
            r1 = rem * hr_b1 / s12
            r2 = rem * hr_b2 / s12
        else:
            r1, r2 = rem, 0.0
        source = f"{source}+b4_structural"
    else:
        r1, r2 = hr_b1, hr_b2
        s = r1 + r2 + r4
        if s < 1.0 - 1e-9:
            r1 += 1.0 - s
            source = f"{source}+orphan_b1"
        elif s > 1.0 + 1e-9:
            scale = 1.0 / s
            r1, r2, r4 = r1 * scale, r2 * scale, r4 * scale

    return r1, r2, r4, source


def apply_spot_pnl_bucket_split(
    *,
    lot_components: dict[str, dict[str, dict[str, float]]],
    underlying: str,
    df: pd.DataFrame,
    r_b1: float,
    r_b2: float,
    r_b4: float,
    spot_carry_cols: set[str],
    cols: set[str] | None = None,
) -> None:
    """Redistribute IBKR spot PnL across buckets using signed sleeve fractions."""
    u_rows = df[
        (df["underlying"] == underlying)
        & (df["symbol"].astype(str) == df["underlying"].astype(str))
    ]
    if u_rows.empty:
        return
    bucket_keys = ("bucket_1", "bucket_2", "bucket_4")
    fracs = {
        "bucket_1": float(r_b1),
        "bucket_2": float(r_b2),
        "bucket_4": float(r_b4),
    }
    override_cols = (
        cols
        if cols is not None
        else ({"realized_pnl", "unrealized_pnl"} | set(spot_carry_cols))
    )
    for col in override_cols:
        if col not in u_rows.columns:
            continue
        total = float(u_rows[col].sum())
        if abs(total) <= 1e-9:
            for bk in bucket_keys:
                lot_components[underlying][bk][col] = 0.0
            continue
        for bk in bucket_keys:
            lot_components[underlying][bk][col] = total * fracs[bk]


def load_plan_sleeve_bucket_usd(
    plan_csv: Path,
    etf_to_delta: dict[str, float],
    *,
    sleeve_first: bool = True,
) -> dict[str, dict[str, float]]:
    """For each underlying, signed ``long_usd`` per accounting bucket from the latest plan.

    Bucketing priority (when ``sleeve_first=True``):

      * ``sleeve`` column: ``yieldboost`` → b2, ``core_leveraged`` → b1,
        ``inverse_decay_bucket4`` → b4.
      * else ETF delta (plan ``Delta`` column, then ``etf_to_delta`` map):
        ``beta < 0`` → b4, ``beta > 1.5`` → b1, ``0 < beta <= 1.5`` → b2.
      * else ``is_yieldboost`` → b2.

    The returned values are the SIGNED ``long_usd`` contributions per bucket
    (B4 is typically negative because ``inverse_decay_bucket4`` rows carry a
    negative ``long_usd`` for the short underlying leg — see
    ``scripts/gtp_sizing_mirror.py``). Returns an empty dict if the plan file
    is missing, unreadable, empty, or missing required columns.

    This captures the strategy's intent as of the most recent
    ``generate_trade_plan`` run. Accounting uses these per-sleeve targets to
    attribute the bucket-4 short underlying slice even though the rebalancer
    nets B1 / yieldboost long + B4 short underlying orders into ONE signed
    IBKR stock position (see ``rebalance_strategy.build_establish_trades`` /
    ``phase2b_resize.build_resize_trades``). Without this overlay the FIFO
    share ledger tracks only the net IBKR line, so the B4 sleeve's structural
    short underlying shows as ``$0`` exposure and PnL.
    """
    if plan_csv is None or not Path(plan_csv).exists():
        return {}
    try:
        df = pd.read_csv(plan_csv)
    except Exception:
        return {}
    if df.empty or "Underlying" not in df.columns or "long_usd" not in df.columns:
        return {}
    df = df.copy()
    df["Underlying"] = df["Underlying"].astype(str).map(canonical_symbol)
    if "ETF" in df.columns:
        df["ETF"] = df["ETF"].astype(str).map(canonical_symbol)
    else:
        df["ETF"] = ""
    df["long_usd"] = pd.to_numeric(df["long_usd"], errors="coerce").fillna(0.0)
    df["_bucket"] = df.apply(
        lambda row: _plan_row_bucket_key(row, etf_to_delta, sleeve_first=sleeve_first),
        axis=1,
    )
    out: dict[str, dict[str, float]] = {}
    for u, grp in df.groupby("Underlying"):
        u_str = str(u or "")
        if not u_str:
            continue
        b1 = float(grp.loc[grp["_bucket"] == "b1", "long_usd"].sum())
        b2 = float(grp.loc[grp["_bucket"] == "b2", "long_usd"].sum())
        b4 = float(grp.loc[grp["_bucket"] == "b4", "long_usd"].sum())
        if abs(b1) + abs(b2) + abs(b4) <= 1e-9:
            continue
        out[u_str] = {"b1": b1, "b2": b2, "b4": b4}
    return out


def compute_plan_b4_structural_qty(
    plan_sleeve_usd: dict[str, dict[str, float]],
    underlying: str,
    spot_mark_base: float,
) -> float:
    """Signed share qty for the B4 structural short from plan ``long_usd`` (typically negative)."""
    weights = plan_sleeve_usd.get(canonical_symbol(str(underlying))) or {}
    b4_usd = float(weights.get("b4", 0.0))
    if abs(b4_usd) <= 1e-9 or abs(float(spot_mark_base)) <= 1e-12:
        return 0.0
    return b4_usd / float(spot_mark_base)


def compute_implied_b4_short(
    underlying: str,
    pos: pd.DataFrame,
    *,
    b4_registry: pd.DataFrame,
    spot_mark_base: float,
    partial_hedge_ratio_default: float = 0.75,
) -> tuple[float, float]:
    """
    Implied B4 structural short for ``underlying`` from held inverse-ETF positions.

    For each B4 registry pair mapped to ``underlying`` we sum the held ETF's
    delta-normalised market value (``position × delta × markPx × fxRateToBase``)
    — which yields the long-equivalent exposure created by being short the
    inverse ETF — and return its negative scaled by ``partial_hedge_ratio``.
    This is the structural short underlying leg the strategy *intends* to
    run against the held B4 inverse ETF book even when the latest plan no
    longer carries an explicit ``inverse_decay_bucket4`` row and the FIFO
    share ledger has been netted into a single broker stock line.

    Returns ``(short_usd, short_qty)``. Both are ≤ 0 when the held inverse
    ETF position is short (the typical case); both are ≥ 0 if the inverse
    ETF position is somehow long (signs flip naturally).
    """
    u = canonical_symbol(str(underlying))
    if pos is None or pos.empty or b4_registry is None or b4_registry.empty:
        return 0.0, 0.0

    reg = b4_registry.copy()
    reg["underlying"] = reg["underlying"].astype(str).map(canonical_symbol)
    reg["etf"] = reg["etf"].astype(str).map(canonical_symbol)
    reg = reg[reg["underlying"] == u]
    if reg.empty:
        return 0.0, 0.0

    pos_u = pos.copy()
    pos_u["symbol"] = pos_u["symbol"].astype(str).map(canonical_symbol)

    short_usd = 0.0
    for _, row in reg.iterrows():
        etf = str(row["etf"])
        try:
            delta = float(row.get("delta", -1.0))
        except (TypeError, ValueError):
            delta = -1.0
        try:
            phr = float(row.get("partial_hedge_ratio", partial_hedge_ratio_default))
        except (TypeError, ValueError):
            phr = partial_hedge_ratio_default
        if not np.isfinite(phr) or abs(phr) < 1e-12:
            phr = partial_hedge_ratio_default
        held = pos_u[pos_u["symbol"] == etf]
        if held.empty:
            continue
        for _, h in held.iterrows():
            try:
                qty = float(h.get("position", 0.0) or 0.0)
                fx = float(h.get("fxRateToBase", 1.0) or 1.0)
                mark_local = float(h.get("markPrice", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            mark_base = mark_local * fx
            if abs(qty) <= 1e-12 or mark_base <= 0:
                continue
            etf_beta_norm = qty * delta * mark_base
            short_usd += -phr * etf_beta_norm

    spot_mark = float(spot_mark_base or 0.0)
    short_qty = short_usd / spot_mark if abs(spot_mark) > 1e-12 else 0.0
    return short_usd, short_qty


def resolve_b4_plan_exposure_underlyings(
    *,
    mode: str,
    explicit: set[str],
    b4_underlyings: set[str],
    plan_sleeve_usd: dict[str, dict[str, float]],
    min_usd: float,
    etf_implied_short_usd: dict[str, float] | None = None,
    attribution_mode: str = "etf_implied",
) -> set[str]:
    """Underlyings whose pair exposure carries a structural short underlying leg.

    Sources, in priority:

    * Plan ``inverse_decay_bucket4`` sleeve USD with magnitude ≥ ``min_usd``
      (active only when ``mode='plan_structural'``).
    * ETF-implied short magnitude ≥ ``min_usd`` (active when
      ``attribution_mode='etf_implied'``).

    ``explicit`` (when set) restricts the result to its intersection with
    ``b4_underlyings``; otherwise both source signals are unioned and any
    name reaching the threshold is included.
    """
    if str(mode).strip().lower() != "plan_structural" and str(attribution_mode).strip().lower() != "etf_implied":
        return set()
    if explicit:
        return {canonical_symbol(str(u)) for u in explicit if canonical_symbol(str(u)) in b4_underlyings}

    etf_implied_short_usd = etf_implied_short_usd or {}
    attribution_mode = str(attribution_mode or "").strip().lower()
    out: set[str] = set()
    for u in b4_underlyings:
        u_str = canonical_symbol(str(u))
        if str(mode).strip().lower() == "plan_structural":
            b4_usd = abs(float((plan_sleeve_usd.get(u_str) or {}).get("b4", 0.0)))
            if b4_usd >= float(min_usd):
                out.add(u_str)
                continue
        if attribution_mode == "etf_implied":
            implied_usd = abs(float(etf_implied_short_usd.get(u_str, 0.0)))
            if implied_usd >= float(min_usd):
                out.add(u_str)
    return out


def resolve_b4_structural_short_usd_by_underlying(
    b4_underlyings: set[str] | list[str],
    *,
    b4_underlying_exposure_mode: str,
    b4_plan_exposure_underlyings: set[str],
    plan_b4_qty_by_u: dict[str, float],
    plan_sleeve_usd: dict[str, dict[str, float]],
    spot_qty: dict[str, float],
    spot_marks: dict[str, float],
    implied_b4_short_usd: dict[str, float],
    b4_attribution_mode: str,
    b4_attribution_min_usd: float,
) -> dict[str, float]:
    """
    Signed structural short underlying USD per B4 name for ratio-split exposure.

    Matches pair-view priority: plan structural qty/USD, then ETF-implied short.
    """
    if str(b4_attribution_mode or "").strip().lower() == "ledger_fifo":
        return {}
    out: dict[str, float] = {}
    for _u in b4_underlyings:
        u_str = canonical_symbol(str(_u))
        if (
            str(b4_underlying_exposure_mode).strip().lower() == "plan_structural"
            and u_str in b4_plan_exposure_underlyings
        ):
            _mark = float(spot_marks.get(u_str, 0.0))
            _qty = float(plan_b4_qty_by_u.get(u_str, 0.0))
            if abs(_qty) > 1e-9 and _mark > 1e-12:
                out[u_str] = _qty * _mark
                continue
            _plan_usd = float((plan_sleeve_usd.get(u_str) or {}).get("b4", 0.0))
            if abs(_plan_usd) >= float(b4_attribution_min_usd):
                out[u_str] = _plan_usd
                continue
        if str(b4_attribution_mode).strip().lower() == "etf_implied":
            _implied = float(implied_b4_short_usd.get(u_str, 0.0))
            if abs(_implied) >= float(b4_attribution_min_usd):
                out[u_str] = _implied
    return out


def apply_b4_structural_short_to_exposure_detail(
    detail: pd.DataFrame,
    structural_usd_by_u: dict[str, float],
    *,
    hedge_spot_ratio_map: dict[str, SpotBucketRatios] | None = None,
) -> pd.DataFrame:
    """
    Carve the B4 structural short underlying slice from each physical spot line.

    Preserves book totals: ``spot_b1 + spot_b2 + spot_b4 == spot_net`` per line.
    Remaining long spot after the structural short is split B1/B2 using
    ``hedge_spot_ratio_map`` (or all to B1 when absent).
    """
    if detail.empty or not structural_usd_by_u:
        return detail
    d = detail.copy()
    for u_str, struct_usd in structural_usd_by_u.items():
        if abs(float(struct_usd)) < 1e-9:
            continue
        mask = (
            (d["underlying"].astype(str) == u_str)
            & (d["symbol"].astype(str) == u_str)
            & (~d["_is_etf"])
        )
        if not mask.any():
            continue
        idx = d.index[mask][0]
        spot_net = float(d.at[idx, "net_notional_usd"])
        if abs(spot_net) <= 1e-12:
            continue
        hr = (hedge_spot_ratio_map or {}).get(u_str)
        r1_hr = float(hr.b1) if hr is not None else 1.0
        r2_hr = float(hr.b2) if hr is not None else 0.0
        spot_b4 = float(struct_usd)
        rem = spot_net - spot_b4
        s12 = r1_hr + r2_hr
        if s12 > 1e-12:
            spot_b1 = rem * r1_hr / s12
            spot_b2 = rem * r2_hr / s12
        else:
            spot_b1, spot_b2 = rem, 0.0
        d.at[idx, "_ratio_b1"] = spot_b1 / spot_net
        d.at[idx, "_ratio_b2"] = spot_b2 / spot_net
        d.at[idx, "_ratio_b4"] = spot_b4 / spot_net
        if "leg_class" in d.columns:
            d.at[idx, "leg_class"] = "spot_b4_structural"
    return d


def build_b4_plan_ledger_reconciliation(
    *,
    run_date: str,
    b4_underlyings: set[str],
    plan_sleeve_usd: dict[str, dict[str, float]],
    spot_qty: dict[str, float],
    ledger_qty_map: dict[str, dict[str, float]],
    spot_marks: dict[str, float],
    plan_exposure_underlyings: set[str],
    etf_implied_short_usd: dict[str, float] | None = None,
    etf_implied_short_qty: dict[str, float] | None = None,
    structural_short_source: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compare plan B4 targets, ETF-implied structural shorts, and FIFO ledger.

    The ``b4_source`` column reports which signal won at attribution time:
    ``plan`` (current ``inverse_decay_bucket4`` plan row), ``etf_implied``
    (held inverse ETFs × β × partial_hedge_ratio), ``ledger`` (FIFO
    ``qty_b4`` only), or ``none``. ``ledger_missing_b4`` fires whenever an
    intended structural short (plan or implied) is absent from the FIFO
    share ledger — useful when the rebalancer netted sleeve orders into a
    single broker stock line.
    """
    etf_implied_short_usd = etf_implied_short_usd or {}
    etf_implied_short_qty = etf_implied_short_qty or {}
    structural_short_source = structural_short_source or {}
    rows: list[dict] = []
    for u in sorted(b4_underlyings):
        u_str = canonical_symbol(str(u))
        plan_usd = float((plan_sleeve_usd.get(u_str) or {}).get("b4", 0.0))
        mark = float(spot_marks.get(u_str, 0.0))
        plan_qty = compute_plan_b4_structural_qty(plan_sleeve_usd, u_str, mark)
        implied_usd = float(etf_implied_short_usd.get(u_str, 0.0))
        implied_qty = float(etf_implied_short_qty.get(u_str, 0.0))
        ledger = ledger_qty_map.get(u_str, {})
        ledger_b4 = float(ledger.get("bucket_4", 0.0))
        ibkr = float(spot_qty.get(u_str, 0.0))
        orphan = ibkr - (
            float(ledger.get("bucket_1", 0.0))
            + float(ledger.get("bucket_2", 0.0))
            + ledger_b4
        )
        orphan_frac = abs(orphan) / abs(ibkr) if abs(ibkr) > 1e-9 else 0.0
        intent_usd = plan_usd if abs(plan_usd) > 1e-9 else implied_usd
        ledger_missing = abs(intent_usd) > 1e-9 and abs(ledger_b4) < 1.0
        b4_source = structural_short_source.get(u_str, "none")
        rows.append(
            {
                "run_date": run_date,
                "underlying": u_str,
                "plan_b4_usd": plan_usd,
                "plan_b4_qty": plan_qty,
                "etf_implied_short_usd": implied_usd,
                "etf_implied_short_qty": implied_qty,
                "b4_source": b4_source,
                "ledger_qty_b4": ledger_b4,
                "ibkr_qty": ibkr,
                "orphan_qty": orphan,
                "orphan_frac": orphan_frac,
                "ledger_missing_b4": ledger_missing,
                "plan_exposure_active": u_str in plan_exposure_underlyings,
            }
        )
    return pd.DataFrame(rows)


def held_exposure_bucket124_weights(
    underlying: str,
    pos: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    *,
    flow_short_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
) -> tuple[float, float, float]:
    """EOD delta-adjusted held weights for splitting spot PnL across b1/b2/b4."""
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
        beta = float(etf_to_delta.get(etf, 0.0))
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
    w1, w2, w4 = _normalize_bucket_triple(w1, w2, w4)
    has_b1, has_b2, has_b4 = w1 > 1e-12, w2 > 1e-12, w4 > 1e-12
    return apply_spot_bucket_eligibility(
        w1, w2, w4, has_b1_etf=has_b1, has_b2_etf=has_b2, has_b4_etf=has_b4
    )


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
    reg["delta"] = pd.to_numeric(reg["delta"], errors="coerce").fillna(-1.0)

    pos = pos.copy()
    pos["symbol"] = pos["symbol"].astype(str).map(canonical_symbol)

    detail_rows: list[dict] = []
    # Track which underlyings have already had their underlying leg emitted.
    # The registry can carry multiple inverse ETFs for the same underlying
    # (e.g. MSTR ↔ MSDD / MSTZ / SMST); emitting the underlying once per ETF
    # row would multiply its notional by N and break the by-underlying
    # aggregation (silently fine when ``qty_b4 == 0``, materially wrong as
    # soon as plan-aware B4 attribution surfaces a non-zero structural
    # short slice). The B4 pair view treats the underlying as ONE shared
    # short leg hedged by N inverse ETFs.
    _under_emitted: set[str] = set()
    for _, pr in reg.iterrows():
        etf = str(pr["etf"])
        under = str(pr["underlying"])
        delta = float(pr["delta"])

        u_ratio = float((underlying_b4_ratio or {}).get(under, 1.0))
        u_qty = (underlying_b4_qty or {}).get(under)
        legs: list[tuple[str, str, float]] = []
        if under not in _under_emitted:
            legs.append((under, "underlying", 1.0))
            _under_emitted.add(under)
        legs.append((etf, "etf", delta))
        for sym, leg_type, mult_delta in legs:
            rows = pos[pos["symbol"] == sym]
            if leg_type == "underlying" and u_qty is not None and rows.empty:
                # Plan-aware B4 may want the underlying leg even when we
                # hold no IBKR stock line for ``under`` (purely a notional
                # structural short). Synthesize a single row at the
                # plan-derived qty using a best-effort mark price.
                pseudo_mark = None
                _u_etf_rows = pos[pos["symbol"] == etf]
                if not _u_etf_rows.empty:
                    pseudo_mark = float(_u_etf_rows.iloc[0]["markPrice"]) * float(
                        _u_etf_rows.iloc[0]["fxRateToBase"]
                    )
                if pseudo_mark is None or pseudo_mark <= 0:
                    continue
                mv = float(u_qty) * mult_delta * pseudo_mark
                detail_rows.append(
                    {
                        "underlying": under,
                        "symbol": sym,
                        "leg_type": leg_type,
                        "net_notional_usd": mv,
                        "gross_notional_usd": abs(mv),
                    }
                )
                continue
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
                mv = eff_qty * mult_delta * mark
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


_LEDGER_REPLAY_ALL_SENTINELS = {"*", "ALL", "__ALL__"}


def resolve_ledger_full_replay_all(
    cfg_flag: object = False,
    replay_list: list | tuple | set | None = None,
    env_value: str | None = None,
) -> bool:
    """True when a FULL historical restate of the share ledger is requested.

    Triggered by any of: ``accounting.ledger_full_replay_all: true`` in config,
    a ``"*"`` / ``"ALL"`` sentinel inside ``ledger_full_replay_underlyings``, or
    the ``LS_LEDGER_FULL_RESTATE`` env var. A full restate rebuilds every
    underlying from the complete Flex trade history, discarding the
    carried-forward persisted seed (used to clear state booked under the old
    ``_is_etf_leg`` bug).
    """
    if bool(cfg_flag):
        return True
    for x in replay_list or []:
        if str(x).strip().upper() in _LEDGER_REPLAY_ALL_SENTINELS:
            return True
    return str(env_value or "").strip().lower() in {"1", "true", "yes", "on"}


def _bucket_from_delta(beta: float) -> str | None:
    if beta < 0:
        return "bucket_4"
    if beta > 1.5:
        return "bucket_1"
    if beta > 0:
        return "bucket_2"
    return None


def _bucket_hint_from_order_reference(
    order_ref: str,
    etf_to_delta_map: dict[str, float],
) -> str | None:
    if not order_ref:
        return None
    for tok in str(order_ref).split("|"):
        raw = tok.strip().upper()
        candidates = [raw]
        if "__" in raw:
            candidates.append(raw.split("__", 1)[1].strip())
        for item in candidates:
            cand = canonical_symbol(item)
            if cand in etf_to_delta_map:
                return _bucket_from_delta(float(etf_to_delta_map.get(cand, 0.0)))
    return None


def bucket_weights_from_order_reference(
    order_ref: str,
) -> tuple[float, float, float] | None:
    """Explicit B1/B2/B4 split tagged at execution (Phase 3 forward attribution).

    Recognizes, inside the pipe-delimited ``orderReference``:

    * ``LSB:<b1>:<b2>[:<b4>]`` -- long-spot bucket split in permille (e.g.
      ``LSB:600:400`` => 60% B1 / 40% B2). Stamped by ``rebalance_strategy`` /
      ``phase2b_resize`` on the netted underlying order from the plan's
      per-sleeve ``long_usd`` mix, so the FIFO share ledger records the true
      B1-vs-B2 long-spot division instead of inferring it.
    * bare ``B1`` / ``B2`` / ``B4`` (or ``BUCKET_1`` ...) single-bucket tags.

    Returns normalized (w1, w2, w4) or ``None`` when no explicit tag is present.
    """
    if not order_ref:
        return None
    for tok in str(order_ref).split("|"):
        t = tok.strip().upper()
        if t.startswith("LSB:"):
            parts = t[4:].split(":")
            try:
                vals = [float(x) for x in parts if x != ""]
            except ValueError:
                continue
            if len(vals) >= 2:
                b1 = vals[0]
                b2 = vals[1]
                b4 = vals[2] if len(vals) >= 3 else 0.0
                s = b1 + b2 + b4
                if s > 1e-9:
                    return b1 / s, b2 / s, b4 / s
        if t in ("B1", "BUCKET_1"):
            return 1.0, 0.0, 0.0
        if t in ("B2", "BUCKET_2"):
            return 0.0, 1.0, 0.0
        if t in ("B4", "BUCKET_4"):
            return 0.0, 0.0, 1.0
    return None


def _etf_bucket_demand_weights(
    underlying: str,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    etf_qty_by_sym: dict[str, float],
    *,
    flow_short_syms: set[str] | None = None,
) -> tuple[float, float, float]:
    """|qty| × |β| demand on *held* ETF legs for one underlying (b1/b2/b4)."""
    flow = flow_short_syms or set()
    u = canonical_symbol(underlying)
    w1 = w2 = w4 = 0.0
    for etf_sym, etf_under in etf_to_under.items():
        if etf_under != u:
            continue
        beta = float(etf_to_delta.get(etf_sym, 0.0))
        pos_qty = abs(float(etf_qty_by_sym.get(etf_sym, 0.0)))
        if pos_qty <= 1e-12:
            continue
        w = pos_qty * abs(beta)
        if beta < 0 and etf_sym not in flow:
            w4 += w
        elif beta > 1.5:
            w1 += w
        elif beta > 0:
            w2 += w
    return w1, w2, w4


def underlying_held_etf_bucket_flags(
    underlying: str,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    etf_qty_by_sym: dict[str, float],
    *,
    flow_short_syms: set[str] | None = None,
) -> tuple[bool, bool, bool]:
    """True when a non-flat ETF leg on ``underlying`` maps to that bucket."""
    w1, w2, w4 = _etf_bucket_demand_weights(
        underlying,
        etf_to_under,
        etf_to_delta,
        etf_qty_by_sym,
        flow_short_syms=flow_short_syms,
    )
    return w1 > 1e-12, w2 > 1e-12, w4 > 1e-12


def held_etf_bucket_flags_from_positions(
    underlying: str,
    pos: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    *,
    flow_short_syms: set[str] | None = None,
) -> tuple[bool, bool, bool]:
    u = canonical_symbol(underlying)
    qty_by_sym: dict[str, float] = {}
    if pos is not None and not pos.empty:
        for _, row in pos.iterrows():
            sym = canonical_symbol(str(row.get("symbol", "") or ""))
            if etf_to_under.get(sym) != u:
                continue
            if not _is_etf_leg(sym, u, etf_to_under):
                continue
            qty_by_sym[sym] = float(qty_by_sym.get(sym, 0.0)) + float(
                row.get("position", 0.0) or 0.0
            )
    return underlying_held_etf_bucket_flags(
        u,
        etf_to_under,
        etf_to_delta,
        qty_by_sym,
        flow_short_syms=flow_short_syms,
    )


def apply_spot_bucket_eligibility(
    w1: float,
    w2: float,
    w4: float,
    *,
    has_b1_etf: bool,
    has_b2_etf: bool,
    has_b4_etf: bool,
) -> tuple[float, float, float]:
    """
    Spot may only offset ETF sleeves that are actually held on the underlying.

    - B1 spot ↔ levered ETFs (β > 1.5)
    - B2 spot ↔ standard / yieldboost ETFs (0 < β ≤ 1.5)
    - B4 spot ↔ inverse-decay structural shorts (β < 0, non-flow)
    """
    if not has_b1_etf:
        w1 = 0.0
    if not has_b2_etf:
        w2 = 0.0
    if not has_b4_etf:
        w4 = 0.0
    if (w1 + w2 + w4) <= 1e-12:
        if has_b1_etf:
            return 1.0, 0.0, 0.0
        if has_b2_etf:
            return 0.0, 1.0, 0.0
        if has_b4_etf:
            return 0.0, 0.0, 1.0
        return 1.0, 0.0, 0.0
    return _normalize_bucket_triple(w1, w2, w4)


def spot_trade_bucket_weights(
    underlying: str,
    order_ref: str,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
    etf_pos_qty: dict[str, float],
    *,
    minute_demand: tuple[float, float, float] | None = None,
    ledger_bucket_qty: dict[str, float] | None = None,
    pos: pd.DataFrame | None = None,
    flow_short_syms: set[str] | None = None,
    b4_etf_syms: set[str] | None = None,
    yieldboost_spot_b2: bool = False,
) -> tuple[float, float, float]:
    """Allocate a spot share trade across b1/b2/b4 using held ETF sleeves only."""
    u = canonical_symbol(underlying)
    has_b1, has_b2, has_b4 = underlying_held_etf_bucket_flags(
        u,
        etf_to_under,
        etf_to_delta,
        etf_pos_qty,
        flow_short_syms=flow_short_syms,
    )
    if yieldboost_spot_b2 and has_b2:
        return 0.0, 1.0, 0.0

    # Explicit per-trade bucket split tagged at execution (Phase 3) takes
    # precedence over inferred weights: it encodes the actual sleeve intent.
    explicit = bucket_weights_from_order_reference(order_ref)
    if explicit is not None and sum(explicit) > 1e-12:
        return apply_spot_bucket_eligibility(
            explicit[0], explicit[1], explicit[2],
            has_b1_etf=has_b1, has_b2_etf=has_b2, has_b4_etf=has_b4,
        )

    hint = _bucket_hint_from_order_reference(order_ref, etf_to_delta)
    if hint == "bucket_1" and has_b1:
        return 1.0, 0.0, 0.0
    if hint == "bucket_2" and has_b2:
        return 0.0, 1.0, 0.0
    if hint == "bucket_4" and has_b4:
        return 0.0, 0.0, 1.0

    w1 = w2 = w4 = 0.0
    if minute_demand is not None:
        w1, w2, w4 = minute_demand
    if (w1 + w2 + w4) <= 1e-12:
        w1, w2, w4 = _etf_bucket_demand_weights(
            u, etf_to_under, etf_to_delta, etf_pos_qty, flow_short_syms=flow_short_syms
        )
    if (w1 + w2 + w4) <= 1e-12 and ledger_bucket_qty:
        w1 = abs(float(ledger_bucket_qty.get("bucket_1", 0.0)))
        w2 = abs(float(ledger_bucket_qty.get("bucket_2", 0.0)))
        w4 = abs(float(ledger_bucket_qty.get("bucket_4", 0.0)))
    if (w1 + w2 + w4) <= 1e-12:
        w1, w2, w4 = held_exposure_bucket124_weights(
            u,
            pos if pos is not None else pd.DataFrame(),
            etf_to_under,
            etf_to_delta,
            flow_short_syms=flow_short_syms,
            b4_etf_syms=b4_etf_syms,
        )
    return apply_spot_bucket_eligibility(
        w1, w2, w4, has_b1_etf=has_b1, has_b2_etf=has_b2, has_b4_etf=has_b4
    )


def build_underlying_realized_bucket_ratio_map(
    trade_events: pd.DataFrame,
    etf_to_under: dict[str, str],
    etf_to_delta: dict[str, float],
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

    flow = flow_short_syms or set()
    etf_pos_qty: dict[str, float] = defaultdict(float)
    realized_by_under_bucket: dict[str, dict[str, float]] = defaultdict(
        lambda: {"bucket_1": 0.0, "bucket_2": 0.0, "bucket_4": 0.0}
    )

    for _, row in trade_events.iterrows():
        sym = str(row.get("symbol", ""))
        qty = float(row.get("quantity", 0.0) or 0.0)
        realized = float(row.get("fifoPnlRealized_base", 0.0) or 0.0)

        under = str(row.get("underlyingSymbol", "") or "")
        if not under:
            under = sym
        under = canonical_symbol(under)

        is_etf = _is_etf_leg(sym, under, etf_to_under)
        if is_etf:
            etf_pos_qty[sym] += qty

        if is_etf or abs(realized) <= 1e-12:
            continue

        has_b1, has_b2, has_b4 = underlying_held_etf_bucket_flags(
            under,
            etf_to_under,
            etf_to_delta,
            etf_pos_qty,
            flow_short_syms=flow,
        )
        # Explicit per-trade split (Phase 3) wins over the ETF-symbol hint.
        explicit = bucket_weights_from_order_reference(str(row.get("orderReference", "")))
        if explicit is not None and sum(explicit) > 1e-12:
            ew1, ew2, ew4 = apply_spot_bucket_eligibility(
                explicit[0], explicit[1], explicit[2],
                has_b1_etf=has_b1, has_b2_etf=has_b2, has_b4_etf=has_b4,
            )
            tot_e = ew1 + ew2 + ew4
            if tot_e > 1e-12:
                realized_by_under_bucket[under]["bucket_1"] += abs(realized) * ew1 / tot_e
                realized_by_under_bucket[under]["bucket_2"] += abs(realized) * ew2 / tot_e
                realized_by_under_bucket[under]["bucket_4"] += abs(realized) * ew4 / tot_e
                continue

        b_hint = _bucket_hint_from_order_reference(
            str(row.get("orderReference", "")),
            etf_to_delta,
        )
        if b_hint == "bucket_1" and has_b1:
            realized_by_under_bucket[under]["bucket_1"] += abs(realized)
            continue
        if b_hint == "bucket_2" and has_b2:
            realized_by_under_bucket[under]["bucket_2"] += abs(realized)
            continue
        if b_hint == "bucket_4" and has_b4:
            realized_by_under_bucket[under]["bucket_4"] += abs(realized)
            continue

        w1, w2, w4 = _etf_bucket_demand_weights(
            under,
            etf_to_under,
            etf_to_delta,
            etf_pos_qty,
            flow_short_syms=flow,
        )
        if (w1 + w2 + w4) <= 1e-12:
            realized_by_under_bucket[under]["bucket_1"] += abs(realized)
        else:
            r1, r2, r4 = apply_spot_bucket_eligibility(
                w1, w2, w4, has_b1_etf=has_b1, has_b2_etf=has_b2, has_b4_etf=has_b4
            )
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
# Net Exposure calculation (delta-normalized)
# ──────────────────────────────────────────────────────────────────────────────
def compute_net_exposure(
    pos_xml: Path,
    screened_csv: Path,
    pnl_underlyings: set[str] | None = None,
    *,
    positions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute delta-normalized net exposure by underlying.

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

    etf_to_under, etf_to_delta = load_etf_delta_map(screened_csv)

    # Map each position to its underlying and beta
    pos["underlying"] = pos["symbol"].map(etf_to_under).fillna(pos["symbol"])
    pos["is_etf"] = pos.apply(
        lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
        axis=1,
    )
    pos["delta"] = np.where(
        pos["is_etf"],
        pos["symbol"].map(etf_to_delta).astype(float),
        1.0,
    )
    pos["delta"] = pd.to_numeric(pos["delta"], errors="coerce").fillna(1.0)

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
    pos["mv_base"] = pos["position"] * pos["delta"] * pos["markPrice"] * pos["fxRateToBase"]
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
    _acct_cfg = cfg.get("accounting", {}) or {}
    b12_spot_split_method = str(_acct_cfg.get("b12_spot_split_method", "ledger_fifo")).strip().lower()
    if b12_spot_split_method not in {"ledger_fifo", "held_exposure_waterfall"}:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.b12_spot_split_method="
            f"{b12_spot_split_method!r}; defaulting to 'ledger_fifo'"
        )
        b12_spot_split_method = "ledger_fifo"
    b12_spot_exposure_method = str(
        _acct_cfg.get("b12_spot_exposure_method", "sleeve_offset")
    ).strip().lower()
    if b12_spot_exposure_method == "held_delta":
        b12_spot_exposure_method = "sleeve_offset"
    if b12_spot_exposure_method not in {
        "sleeve_offset",
        "ledger_fifo",
        "held_exposure_waterfall",
        "hedge_ratio",
        "sleeve_balance",
    }:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.b12_spot_exposure_method="
            f"{b12_spot_exposure_method!r}; defaulting to 'sleeve_offset'"
        )
        b12_spot_exposure_method = "sleeve_offset"
    _b12_spot_pnl_method_raw = _acct_cfg.get("b12_spot_pnl_method")
    b12_spot_pnl_method = str(
        _b12_spot_pnl_method_raw
        if _b12_spot_pnl_method_raw is not None
        else b12_spot_exposure_method
    ).strip().lower()
    if b12_spot_pnl_method not in {
        "ledger_fifo",
        "sleeve_offset",
        "sleeve_balance",
        "hedge_ratio",
        "held_exposure_waterfall",
    }:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.b12_spot_pnl_method="
            f"{b12_spot_pnl_method!r}; defaulting to {b12_spot_exposure_method!r}"
        )
        b12_spot_pnl_method = b12_spot_exposure_method
    b12_hedge_delta_floor = float(_acct_cfg.get("b12_hedge_delta_floor", 0.25) or 0.25)
    b12_pnl_mode = str(_acct_cfg.get("b12_pnl_mode", "lot_timed_strict")).strip().lower()
    if b12_pnl_mode not in {"lot_timed_strict", "plan_b4_inject"}:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.b12_pnl_mode={b12_pnl_mode!r}; "
            "defaulting to 'lot_timed_strict'"
        )
        b12_pnl_mode = "lot_timed_strict"
    plan_b4_pnl_mode = str(_acct_cfg.get("plan_b4_pnl_mode", "inject_slice")).strip().lower()
    if plan_b4_pnl_mode not in {"inject_slice", "full_override"}:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.plan_b4_pnl_mode={plan_b4_pnl_mode!r}; "
            "defaulting to 'inject_slice'"
        )
        plan_b4_pnl_mode = "inject_slice"
    use_plan_b4_spot_pnl = b12_pnl_mode == "plan_b4_inject"
    plan_b4_inject_start_date = str(_acct_cfg.get("plan_b4_inject_start_date", "") or "").strip()
    plan_b4_inject_active = use_plan_b4_spot_pnl
    if plan_b4_inject_active and plan_b4_inject_start_date:
        plan_b4_inject_active = run_date >= plan_b4_inject_start_date
        if not plan_b4_inject_active:
            print(
                f"[ACCOUNTING] plan B4 spot inject disabled before {plan_b4_inject_start_date} "
                f"(run_date={run_date})"
            )
    plan_sleeve_bucketing = str(_acct_cfg.get("plan_sleeve_bucketing", "sleeve_first")).strip().lower()
    plan_sleeve_first = plan_sleeve_bucketing != "delta_first"
    _replay_raw = [
        str(x).strip()
        for x in (_acct_cfg.get("ledger_full_replay_underlyings") or [])
        if str(x).strip()
    ]
    # A ``"*"`` / ``"ALL"`` sentinel (or ``ledger_full_replay_all: true``) forces
    # a FULL historical restate: every underlying is rebuilt from the complete
    # Flex trade history, ignoring the carried-forward persisted seed. This is
    # the lever for Phase 4 -- it discards stale state booked under the old
    # ``_is_etf_leg`` bug and re-derives the ledger from trades alone.
    ledger_full_replay_all = resolve_ledger_full_replay_all(
        cfg_flag=_acct_cfg.get("ledger_full_replay_all", False),
        replay_list=_replay_raw,
        env_value=os.environ.get("LS_LEDGER_FULL_RESTATE", ""),
    )
    ledger_full_replay_explicit = {
        canonical_symbol(x)
        for x in _replay_raw
        if x.upper() not in _LEDGER_REPLAY_ALL_SENTINELS
    }
    ledger_full_replay_include_bucket2 = bool(
        _acct_cfg.get("ledger_full_replay_include_bucket2", True)
    )
    ledger_full_replay_underlyings: set[str] = set(ledger_full_replay_explicit)
    _yieldboost_spot_b2_cfg = {
        canonical_symbol(str(x))
        for x in (_acct_cfg.get("yieldboost_spot_b2_underlyings") or [])
        if str(x).strip()
    }
    yieldboost_spot_b2_pnl_override = bool(
        _acct_cfg.get("yieldboost_spot_b2_pnl_override", True)
    )
    yieldboost_spot_b2_trade_tag = bool(
        _acct_cfg.get(
            "yieldboost_spot_b2_trade_tag",
            yieldboost_spot_b2_pnl_override,
        )
    )
    yieldboost_spot_b2_underlyings = (
        _yieldboost_spot_b2_cfg if yieldboost_spot_b2_pnl_override else set()
    )
    yieldboost_spot_b2_trade_underlyings = (
        _yieldboost_spot_b2_cfg if yieldboost_spot_b2_trade_tag else set()
    )
    # Sleeve-rotation re-tag: when the FIFO lot tags point to a short-ETF
    # sleeve no longer held (e.g. a yieldBOOST sleeve closed while the long
    # spot persisted and now hedges the levered book), re-attribute the spot
    # P&L to the live sleeve capacity (``ratio_spot_exposure``). Only fires
    # when the lot tags diverge from the live sleeve beyond the tolerance, so
    # genuinely held/mixed sleeves are left untouched.
    spot_sleeve_rotation_retag = bool(
        _acct_cfg.get("spot_sleeve_rotation_retag", True)
    )
    # Min lot-share for a bucket to count as "holding stale lots".
    spot_sleeve_rotation_tol = float(
        _acct_cfg.get("spot_sleeve_rotation_tol", 0.05)
    )
    # Live-sleeve weight below which a bucket is treated as "dead" (sleeve no
    # longer held). The re-tag only fires when stale lots sit in a dead bucket.
    spot_sleeve_dead_threshold = float(
        _acct_cfg.get("spot_sleeve_dead_threshold", 0.05)
    )
    # Capacity cap: when open lot tags overweight B2 vs the live yieldBOOST sleeve
    # (but B2 is still held — not a dead-sleeve rotation), re-attribute spot
    # *realized* P&L to the live sleeve mix so excess lands in B1. Skips names
    # in ``spot_sleeve_capacity_exclude_underlyings`` (e.g. AMD, under-tagged).
    spot_sleeve_capacity_cap = bool(
        _acct_cfg.get("spot_sleeve_capacity_cap", True)
    )
    spot_sleeve_capacity_tol = float(
        _acct_cfg.get("spot_sleeve_capacity_tol", 0.05)
    )
    spot_sleeve_capacity_exclude = {
        canonical_symbol(str(x))
        for x in (_acct_cfg.get("spot_sleeve_capacity_exclude_underlyings") or [])
        if str(x).strip()
    }
    ledger_orphan_split_threshold = float(
        _acct_cfg.get("ledger_orphan_split_threshold", 0.10)
    )
    b4_underlying_exposure_mode = str(
        _acct_cfg.get("b4_underlying_exposure_mode", "ledger_fifo")
    ).strip().lower()
    if b4_underlying_exposure_mode not in {"ledger_fifo", "plan_structural"}:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.b4_underlying_exposure_mode="
            f"{b4_underlying_exposure_mode!r}; defaulting to 'ledger_fifo'"
        )
        b4_underlying_exposure_mode = "ledger_fifo"
    b4_plan_exposure_min_usd = float(_acct_cfg.get("b4_plan_exposure_min_usd", 500.0) or 500.0)
    b4_plan_exposure_underlyings_cfg = {
        canonical_symbol(str(x))
        for x in (_acct_cfg.get("b4_plan_exposure_underlyings") or [])
        if str(x).strip()
    }
    b4_attribution_mode = str(
        _acct_cfg.get("b4_underlying_attribution", "etf_implied")
    ).strip().lower()
    if b4_attribution_mode not in {"plan_only", "etf_implied", "ledger_fifo"}:
        print(
            f"[ACCOUNTING] WARNING: unknown accounting.b4_underlying_attribution="
            f"{b4_attribution_mode!r}; defaulting to 'etf_implied'"
        )
        b4_attribution_mode = "etf_implied"
    b4_attribution_min_usd = float(_acct_cfg.get("b4_attribution_min_usd", 500.0) or 500.0)
    b4_partial_hedge_ratio_default = float(
        _acct_cfg.get("b4_partial_hedge_ratio_default", 0.75) or 0.75
    )
    ledger_plan_seed_b4_underlyings = {
        canonical_symbol(str(x))
        for x in (_acct_cfg.get("ledger_plan_seed_b4_underlyings") or [])
        if str(x).strip()
    }
    spot_ratio_consistency_gate = bool(_acct_cfg.get("spot_ratio_consistency_gate", True))
    exposure_reconciliation_tol_gross_pct = float(
        _acct_cfg.get("exposure_reconciliation_tol_gross_pct", 0.001) or 0.001
    )
    exposure_reconciliation_tol_net_abs_usd = float(
        _acct_cfg.get("exposure_reconciliation_tol_net_abs_usd", 500.0) or 500.0
    )
    bucket_state_path = PROJECT_ROOT / "data" / "accounting" / "underlying_bucket_state.csv"
    b4_partial_hedge_ratio = float(
        ((sleeves_cfg.get("inverse_decay_bucket4") or {}).get("rules") or {}).get(
            "partial_hedge_ratio", 1.0
        )
    )

    dated_screened_path = PROJECT_ROOT / "data" / "runs" / run_date / "etf_screened_today.csv"
    etf_screened_path = (
        dated_screened_path
        if dated_screened_path.exists()
        else PROJECT_ROOT / "data" / "etf_screened_today.csv"
    )
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
    _, etf_to_delta_pre = load_etf_delta_map(etf_screened_path)
    for sym in SUPPLEMENTAL_ETF_MAP:
        etf_to_delta_pre.setdefault(sym, etf_to_delta_pre.get(sym, -1.0))
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
    b4_phr_by_etf = _build_b4_phr_by_etf(
        b4_registry,
        partial_hedge_ratio_default=b4_partial_hedge_ratio_default,
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

    blocked_symbols, blocked_underlyings = expand_blacklist(blacklist, etf_to_under)
    if blocked_underlyings:
        print(
            f"[ACCOUNTING] Blacklist exposure exclusion: "
            f"{sorted(blocked_underlyings)} ({len(blocked_symbols)} symbols)"
        )
    if not b4_registry.empty:
        b4_registry = b4_registry[
            ~b4_registry["underlying"].astype(str).isin(blocked_underlyings)
            & ~b4_registry["etf"].astype(str).isin(blocked_symbols)
        ].copy()
        b4_registry.to_csv(outdir / "bucket4_pairs.csv", index=False)
    b4_etf_syms = set(b4_registry["etf"].astype(str).tolist()) if not b4_registry.empty else set()
    b4_underlyings = (
        set(b4_registry["underlying"].astype(str).tolist()) if not b4_registry.empty else set()
    )

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
    # Spot positions (not in etf_to_under) are split pro-rata by the
    # absolute deltas of the ETFs sharing their underlying.
    etf_to_under, etf_to_delta_map = load_etf_delta_map(etf_screened_path)
    etf_to_under, etf_to_delta_map = merge_plan_etf_metadata(
        proposed_trades_path, etf_to_under, etf_to_delta_map
    )
    etf_to_under, etf_to_delta_map = complete_etf_maps_from_positions(
        pos, etf_to_under, etf_to_delta_map
    )
    flow_low_delta_syms = {
        s for s in flow_short_set if 0 < float(etf_to_delta_map.get(s, 0.0)) <= 1.5
    }
    flow_inverse_bucket3_syms = resolve_flow_inverse_bucket3_syms(
        flow_short_set, etf_to_delta_map
    )
    if flow_inverse_bucket3_syms:
        print(
            f"[ACCOUNTING] flow-program inverse bucket-3: "
            f"{len(flow_inverse_bucket3_syms)} symbol(s) "
            f"(e.g. {', '.join(sorted(flow_inverse_bucket3_syms)[:8])})"
        )
    if flow_low_delta_syms:
        print(
            f"[ACCOUNTING] flow low-beta (0-1.5) -> bucket 2: "
            f"{len(flow_low_delta_syms)} symbol(s) "
            f"(e.g. {', '.join(sorted(flow_low_delta_syms)[:8])})"
        )
    neg_beta_syms = {s for s, b in etf_to_delta_map.items() if b < 0}

    is_etf = df.apply(
        lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
        axis=1,
    )
    sym_beta = df["symbol"].map(etf_to_delta_map).fillna(1.0)

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
    for _s, _b in etf_to_delta_map.items():
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
                "delta_weight": abs(float(_b)),
            }
        )

    if _all_rows:
        _all_df = pd.DataFrame(_all_rows)
        _all_beta_sums = (
            _all_df.groupby(["underlying", "_bkt"])["delta_weight"]
            .sum()
            .reset_index()
            .pivot(index="underlying", columns="_bkt", values="delta_weight")
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
    # Weight by delta-adjusted absolute notional so the spot split reflects
    # current exposure. This map overrides the universe fallback above.
    _held_etf_syms = set(pos["symbol"].unique()) & set(etf_to_delta_map.keys())
    _held_positive = {
        s for s in _held_etf_syms
        if etf_to_delta_map.get(s, 0) > 0
    }

    _held_rows = []
    _held_b4 = {
        s
        for s in (set(pos["symbol"].unique()) & set(etf_to_delta_map.keys()))
        if etf_to_delta_map.get(s, 0) < 0 and s not in flow_short_set
    }
    for _s in set(_held_positive) | _held_b4:
        _b = float(etf_to_delta_map[_s])
        _pos_rows = pos[pos["symbol"] == _s]
        _notional = float(_pos_rows["positionValue_base"].abs().sum())
        _delta_adj_notional = _notional * abs(_b)
        if _b < 0:
            _bkt = "bucket_4"
        elif _b > 1.5:
            _bkt = "bucket_1"
        else:
            _bkt = "bucket_2"
        _held_rows.append({
            "underlying": etf_to_under.get(_s, _s),
            "_bkt": _bkt,
            "delta_adj_notional": _delta_adj_notional,
        })

    if _held_rows:
        _held_df = pd.DataFrame(_held_rows)
        _beta_sums = (
            _held_df.groupby(["underlying", "_bkt"])["delta_adj_notional"]
            .sum()
            .reset_index()
            .pivot(index="underlying", columns="_bkt", values="delta_adj_notional")
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

    if ledger_full_replay_include_bucket2:
        _b2_replay = collect_bucket2_underlyings(
            etf_to_under,
            etf_to_delta_map,
            flow_short_syms=flow_short_set,
            b4_etf_syms=b4_etf_syms,
            pos=pos,
        )
        ledger_full_replay_underlyings |= _b2_replay
    ledger_full_replay_underlyings |= ledger_full_replay_explicit

    # ── Plan-aware B4 underlying attribution ─────────────────────────────
    # ``rebalance_strategy.py`` runs an explicit short underlying leg for
    # every ``inverse_decay_bucket4`` row (negative ``long_usd``) and nets
    # it against B1/yieldboost long underlying orders into a SINGLE signed
    # IBKR stock trade per underlying (see ``build_establish_trades`` /
    # ``build_resize_trades``). The FIFO share ledger therefore cannot
    # recover the B4 structural slice on shared names like MSTR — the
    # ledger only sees the net broker line.
    #
    # We surface that slice here by reading the latest plan and computing
    # per-underlying sleeve magnitudes. The resulting ratios override the
    # held-exposure map for B4 underlyings so that:
    #   * net_exposure_bucket_4_detail.csv shows the structural short
    #     underlying leg next to the inverse ETFs;
    #   * spot PnL on shared names is split into B1/B2/B4 in proportion
    #     to the sleeve targets that produced the net stock position.
    #
    # Disable via ``accounting.plan_aware_b4_attribution: false`` to fall
    # back to pure FIFO/held-exposure attribution.
    _plan_aware_b4 = bool(
        (cfg.get("accounting", {}) or {}).get("plan_aware_b4_attribution", True)
    )
    _plan_sleeve_usd: dict[str, dict[str, float]] = {}
    _plan_ratio_b4: dict[str, dict[str, float]] = {}
    if _plan_aware_b4:
        _plan_sleeve_usd = load_plan_sleeve_bucket_usd(
            proposed_trades_path,
            etf_to_delta_map,
            sleeve_first=plan_sleeve_first,
        )
        for _u_p, _w_p in _plan_sleeve_usd.items():
            if _u_p not in b4_underlyings:
                continue
            _a1, _a2, _a4 = abs(_w_p["b1"]), abs(_w_p["b2"]), abs(_w_p["b4"])
            if _a4 <= 1e-9:
                continue
            _pr1, _pr2, _pr4 = _normalize_bucket_triple(_a1, _a2, _a4)
            _plan_ratio_b4[_u_p] = {"b1": _pr1, "b2": _pr2, "b4": _pr4}
            _held_ratio_map[_u_p] = {"b1": _pr1, "b2": _pr2, "b4": _pr4}
            if abs(_w_p["b2"]) > 1e-9 and _pr2 <= 1e-9:
                print(
                    f"[ACCOUNTING] WARNING: {_u_p} plan b2 sleeve "
                    f"${abs(_w_p['b2']):,.0f} but normalized ratio_b2=0 "
                    f"(check plan_sleeve_bucketing / ETF deltas)"
                )
        if _plan_ratio_b4:
            print(
                f"[ACCOUNTING] plan-aware B4 attribution active for "
                f"{len(_plan_ratio_b4)} underlying(s) "
                f"(e.g. {', '.join(sorted(_plan_ratio_b4)[:6])}); "
                f"b12 exposure={b12_spot_exposure_method} "
                f"spot_pnl={b12_spot_pnl_method} ledger={b12_spot_split_method} "
                f"pnl={b12_pnl_mode}"
            )
    # ``_b4_plan_exposure_underlyings`` and ``_b4_signed_frac`` are built
    # later, after ``_spot_marks`` and ``_implied_b4_short_usd`` are known.
    if ledger_full_replay_all:
        print(
            "[ACCOUNTING] FULL RESTATE: replaying ALL underlyings from complete "
            "trade history (carried-forward seeds discarded)."
        )
    elif ledger_full_replay_underlyings:
        print(
            f"[ACCOUNTING] ledger full replay for "
            f"{len(ledger_full_replay_underlyings)} underlying(s): "
            f"{', '.join(sorted(ledger_full_replay_underlyings))}"
        )
    if _yieldboost_spot_b2_cfg and not yieldboost_spot_b2_pnl_override:
        print(
            "[ACCOUNTING] yieldboost spot->B2 PnL override disabled "
            f"({len(_yieldboost_spot_b2_cfg)} name(s) in config); spot follows FIFO ledger"
        )
    elif yieldboost_spot_b2_underlyings:
        print(
            f"[ACCOUNTING] yieldboost spot->B2 override for "
            f"{len(yieldboost_spot_b2_underlyings)} underlying(s): "
            f"{', '.join(sorted(yieldboost_spot_b2_underlyings))}"
        )

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
            return _fallback_ratio_map[u], "universe_delta_fallback"
        r1, r2, r4 = held_exposure_bucket124_weights(
            u,
            pos,
            etf_to_under,
            etf_to_delta_map,
            flow_short_syms=flow_short_set,
            b4_etf_syms=b4_etf_syms,
        )
        return {"b1": r1, "b2": r2, "b4": r4}, "held_exposure_eod"

    _realized_ratio_from_trades = build_underlying_realized_bucket_ratio_map(
        trade_events=trade_events,
        etf_to_under=etf_to_under,
        etf_to_delta=etf_to_delta_map,
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
        "qty_b4_plan",
        "plan_b4_usd",
        "qty_b4_structural",
        "qty_b4_etf_implied",
        "etf_implied_b4_usd",
        "b4_structural_source",
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
            _state_hist[_c] = (
                0.0
                if _c.startswith("qty_")
                or _c.startswith("ratio_")
                or _c in {
                    "plan_b4_usd",
                    "etf_implied_b4_usd",
                    "realized_b1",
                    "realized_b2",
                    "realized_b4",
                }
                else ""
            )
    _state_hist["run_date"] = _state_hist["run_date"].astype(str)
    _state_hist["underlying"] = _state_hist["underlying"].astype(str)
    for _c in [
        "qty_total",
        "qty_b1",
        "qty_b2",
        "qty_b4",
        "qty_b4_plan",
        "plan_b4_usd",
        "qty_b4_structural",
        "qty_b4_etf_implied",
        "etf_implied_b4_usd",
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
    if ledger_full_replay_all:
        # Full restate: discard every carried-forward seed and the incremental
        # cutoff so the loop below applies the complete trade history.
        _prev_state = pd.DataFrame(columns=state_cols).set_index(pd.Index([], dtype=str))
        _prev_cutoff = ""
        _prev_cutoff_ymd = ""
    elif ledger_full_replay_underlyings and not _prev_state.empty:
        _prev_state = _prev_state.drop(
            index=[u for u in ledger_full_replay_underlyings if u in _prev_state.index],
            errors="ignore",
        )

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
            _tu = canonical_symbol(str(_tr.get("underlyingSymbol", "") or _tsym))
            if _is_etf_leg(_tsym, _tu, etf_to_under):
                continue
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
        _ev["_ledger_u"] = _ev.apply(
            lambda r: canonical_symbol(
                str(r.get("underlyingSymbol", "") or r.get("symbol", "") or "")
            ),
            axis=1,
        )
        _target = yyyymmdd_from_run_date(run_date)
        _ev = _ev[_ev["date"] <= _target].copy()
        if _prev_cutoff_ymd:
            _full_replay_mask = _ev["_ledger_u"].isin(ledger_full_replay_underlyings)
            _ev = _ev[(_ev["date"] > _prev_cutoff_ymd) | _full_replay_mask].copy()
        _ev["_etf_leg_sort"] = _ev.apply(
            lambda r: 0
            if _is_etf_leg(
                str(r.get("symbol", "") or ""),
                canonical_symbol(
                    str(r.get("underlyingSymbol", "") or r.get("symbol", "") or "")
                ),
                etf_to_under,
            )
            else 1,
            axis=1,
        )
        _ev = _ev.sort_values(["dateTime", "_etf_leg_sort"]).reset_index(drop=True)

    _minute_demand: dict[tuple[str, str], tuple[float, float, float]] = {}
    if not _ev.empty:
        _tmp = _ev[_ev["symbol"].isin(etf_to_under.keys())].copy()
        if not _tmp.empty:
            _tmp["minute_key"] = _tmp["dateTime"].astype(str).str.slice(0, 13)
            _tmp["under"] = _tmp["symbol"].map(etf_to_under)
            _tmp["delta"] = _tmp["symbol"].map(etf_to_delta_map).astype(float)
            _tmp["w"] = _tmp["quantity"].abs() * _tmp["tradePrice_base"].abs() * _tmp["delta"].abs()
            if not _tmp.empty:
                _tmp["_bkt"] = np.select(
                    [
                        _tmp["delta"].lt(0) & (~_tmp["symbol"].isin(flow_short_set)),
                        _tmp["delta"].gt(1.5),
                        _tmp["delta"].gt(0),
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
        _mk = str(_dt)[:13] if _dt else ""
        _md = _minute_demand.get((_mk, _u))
        return spot_trade_bucket_weights(
            _u,
            _ref,
            etf_to_under,
            etf_to_delta_map,
            _etf_pos_qty,
            minute_demand=_md,
            ledger_bucket_qty=_bucket_qty[_u],
            pos=pos,
            flow_short_syms=flow_short_set,
            b4_etf_syms=b4_etf_syms,
            yieldboost_spot_b2=_u in yieldboost_spot_b2_trade_underlyings,
        )

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

            _u = canonical_symbol(str(_r.get("underlyingSymbol", "") or _sym))
            if not _u:
                _u = _sym
            if _is_etf_leg(_sym, _u, etf_to_under):
                _etf_pos_qty[_sym] += _qty
                continue

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
        if not _sym or _sym in etf_to_delta_map:
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

    _spot_marks: dict[str, float] = {}
    for _sym, _q in _spot_qty.items():
        if abs(float(_q)) <= 1e-12:
            continue
        _px_rows = pos[pos["symbol"] == _sym]
        if not _px_rows.empty:
            _spot_marks[str(_sym)] = float(_px_rows.iloc[0]["markPrice"]) * float(
                _px_rows.iloc[0]["fxRateToBase"]
            )

    # ── ETF-implied B4 structural short (held inverse ETFs × β × hedge_ratio) ──
    # For every B4 registry underlying, compute the signed short underlying
    # notional the strategy intends to be running against its held inverse
    # ETF book. This recovers the structural short even when the rebalancer
    # netted the sleeve order into a single broker stock line and the FIFO
    # ``qty_b4`` ledger never tagged the slice. Used as fallback when the
    # current ``inverse_decay_bucket4`` plan no longer carries the row.
    _implied_b4_short_usd: dict[str, float] = {}
    _implied_b4_short_qty: dict[str, float] = {}
    for _u_b4 in b4_underlyings:
        _u_str_b4 = canonical_symbol(str(_u_b4))
        _mark_u_b4 = float(_spot_marks.get(_u_str_b4, 0.0))
        if _mark_u_b4 <= 1e-12:
            _px_rows_b4 = pos[pos["symbol"] == _u_str_b4]
            if not _px_rows_b4.empty:
                _mark_u_b4 = float(_px_rows_b4.iloc[0]["markPrice"]) * float(
                    _px_rows_b4.iloc[0]["fxRateToBase"]
                )
        if _mark_u_b4 <= 1e-12:
            continue
        _short_usd_b4, _short_qty_b4 = compute_implied_b4_short(
            _u_str_b4,
            pos,
            b4_registry=b4_registry,
            spot_mark_base=_mark_u_b4,
            partial_hedge_ratio_default=b4_partial_hedge_ratio_default,
        )
        if abs(_short_usd_b4) < b4_attribution_min_usd:
            continue
        _implied_b4_short_usd[_u_str_b4] = _short_usd_b4
        _implied_b4_short_qty[_u_str_b4] = _short_qty_b4

    # Resolve the set of underlyings whose pair exposure carries a structural
    # short underlying leg. Sources, in priority: ``plan_structural`` (latest
    # ``inverse_decay_bucket4`` sleeve target above ``b4_plan_exposure_min_usd``)
    # then ``etf_implied`` (held inverse ETFs above ``b4_attribution_min_usd``).
    _b4_plan_exposure_underlyings = resolve_b4_plan_exposure_underlyings(
        mode=b4_underlying_exposure_mode,
        explicit=b4_plan_exposure_underlyings_cfg,
        b4_underlyings=set(b4_underlyings),
        plan_sleeve_usd=_plan_sleeve_usd,
        min_usd=b4_plan_exposure_min_usd,
        etf_implied_short_usd=_implied_b4_short_usd,
        attribution_mode=b4_attribution_mode,
    )
    if _b4_plan_exposure_underlyings:
        print(
            f"[ACCOUNTING] structural B4 exposure for "
            f"{len(_b4_plan_exposure_underlyings)} underlying(s) "
            f"(attribution={b4_attribution_mode}, phr={b4_partial_hedge_ratio_default}): "
            f"{', '.join(sorted(_b4_plan_exposure_underlyings)[:8])}"
            f"{'...' if len(_b4_plan_exposure_underlyings) > 8 else ''}"
        )

    # Signed b4 fraction (spot_b4 / spot_net) per underlying. Plan-structural
    # wins when an ``inverse_decay_bucket4`` plan row exists; otherwise the
    # ETF-implied short fills in (when ``b4_attribution_mode=etf_implied``).
    # Used by the spot PnL override and the ledger reconciliation column.
    _b4_signed_frac: dict[str, float] = {}
    _b4_attribution_source: dict[str, str] = {}
    for _u_p in _plan_ratio_b4:
        _spot_net_u_p = float(_spot_qty.get(_u_p, 0.0)) * float(_spot_marks.get(_u_p, 0.0))
        if abs(_spot_net_u_p) <= 1e-12:
            continue
        _plan_b4_usd_p = float((_plan_sleeve_usd.get(_u_p) or {}).get("b4", 0.0))
        if abs(_plan_b4_usd_p) <= 1e-9:
            continue
        _b4_signed_frac[_u_p] = _plan_b4_usd_p / _spot_net_u_p
        _b4_attribution_source[_u_p] = "plan"
    if b4_attribution_mode == "etf_implied":
        for _u_imp, _short_usd_imp in _implied_b4_short_usd.items():
            if _u_imp in _b4_signed_frac:
                continue
            _spot_net_u_i = float(_spot_qty.get(_u_imp, 0.0)) * float(
                _spot_marks.get(_u_imp, 0.0)
            )
            if abs(_spot_net_u_i) <= 1e-12:
                continue
            _b4_signed_frac[_u_imp] = _short_usd_imp / _spot_net_u_i
            _b4_attribution_source[_u_imp] = "etf_implied"
    if _b4_signed_frac:
        _plan_count_b4 = sum(1 for v in _b4_attribution_source.values() if v == "plan")
        _impl_count_b4 = sum(1 for v in _b4_attribution_source.values() if v == "etf_implied")
        print(
            f"[ACCOUNTING] B4 spot attribution: {_plan_count_b4} plan + "
            f"{_impl_count_b4} etf_implied underlying(s) "
            f"(mode={b4_attribution_mode}, min_usd=${b4_attribution_min_usd:,.0f})"
        )

    _plan_b4_qty_by_u: dict[str, float] = {}
    if ledger_plan_seed_b4_underlyings:
        print(
            f"[ACCOUNTING] plan-seed B1/B2 ledger for "
            f"{len(ledger_plan_seed_b4_underlyings)} underlying(s): "
            f"{', '.join(sorted(ledger_plan_seed_b4_underlyings))}"
        )
    for _u in _all_underlyings:
        _u_str = str(_u)
        _mark_u = float(_spot_marks.get(_u_str, 0.0))
        _plan_b4_qty_by_u[_u_str] = compute_plan_b4_structural_qty(
            _plan_sleeve_usd, _u_str, _mark_u
        )
        if _u_str not in ledger_plan_seed_b4_underlyings or _u_str not in _plan_ratio_b4:
            continue
        _ibkr_seed = float(_spot_qty.get(_u_str, 0.0))
        _pr_seed = _plan_ratio_b4[_u_str]
        _pb1 = float(_pr_seed.get("b1", 0.0))
        _pb2 = float(_pr_seed.get("b2", 0.0))
        _phys_sum = _pb1 + _pb2
        if abs(_ibkr_seed) > 1e-9 and _phys_sum > 1e-12:
            _bucket_qty[_u_str]["bucket_1"] = _ibkr_seed * _pb1 / _phys_sum
            _bucket_qty[_u_str]["bucket_2"] = _ibkr_seed * _pb2 / _phys_sum
            _bucket_qty[_u_str]["bucket_4"] = 0.0

    # ── Canonical B4 structural short (Phase 2: observable & durable) ────────
    # The netted IBKR stock line carries no separable B4 FIFO trade, so the
    # structural short underlying is *derived*. Record ONE canonical value per
    # name with its provenance (``plan`` primary, ``etf_implied`` fallback) plus
    # both candidate magnitudes, so the recorded short does not vanish when a
    # name rotates out of ``proposed_trades.csv`` and the plan-vs-implied gap is
    # auditable over time. This same dict feeds the exposure pair view so the
    # persisted number equals the number used in net/gross exposure.
    _b4_struct_qty_by_u: dict[str, float] = {}
    _b4_struct_usd_by_u: dict[str, float] = {}
    _b4_struct_src_by_u: dict[str, str] = {}
    for _u in _all_underlyings:
        _u_str = str(_u)
        _plan_q = float(_plan_b4_qty_by_u.get(_u_str, 0.0))
        _plan_usd = float((_plan_sleeve_usd.get(_u_str) or {}).get("b4", 0.0))
        _imp_q = float(_implied_b4_short_qty.get(_u_str, 0.0))
        _imp_usd = float(_implied_b4_short_usd.get(_u_str, 0.0))
        if (
            b4_underlying_exposure_mode == "plan_structural"
            and _u_str in _b4_plan_exposure_underlyings
            and abs(_plan_q) > 1e-9
        ):
            _b4_struct_qty_by_u[_u_str] = _plan_q
            _b4_struct_usd_by_u[_u_str] = _plan_usd
            _b4_struct_src_by_u[_u_str] = "plan"
        elif (
            b4_attribution_mode == "etf_implied"
            and abs(_imp_q) > 1e-9
            and abs(_imp_usd) >= b4_attribution_min_usd
        ):
            _b4_struct_qty_by_u[_u_str] = _imp_q
            _b4_struct_usd_by_u[_u_str] = _imp_usd
            _b4_struct_src_by_u[_u_str] = "etf_implied"

    _spot_exposure_ratio_map: dict[str, SpotBucketRatios] = {}
    _hedge_spot_meta_map: dict[str, HedgeRatioSpotMeta] = {}

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
        _um_recon = reconcile_spot_bucket_unrealized(
            _cur_b1,
            _cur_b2,
            _cur_b4,
            float(_bucket_cost[_u]["bucket_1"]),
            float(_bucket_cost[_u]["bucket_2"]),
            float(_bucket_cost[_u]["bucket_4"]),
            float(_spot_qty.get(_u, 0.0)),
            _px_now,
        )
        _um1 = _um_recon["bucket_1"]
        _um2 = _um_recon["bucket_2"]
        _um4 = _um_recon["bucket_4"]
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

        # Plan-aware B4 override: when the latest plan carries an explicit
        # ``inverse_decay_bucket4`` short underlying target for ``_u`` we
        # inject the B4 structural slice into spot PnL. In ``inject_slice``
        # mode (default) the FIFO lot-ledger split for B1/B2 is preserved;
        # ``full_override`` replaces all bucket ratios with plan magnitudes.
        if _u in _plan_ratio_b4 and plan_b4_pnl_mode == "full_override":
            _pr = _plan_ratio_b4[_u]
            _ratio_realized_map[_u] = dict(_pr)
            _ratio_unrealized_map[_u] = dict(_pr)
            _ratio_carry_map[_u] = dict(_pr)

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

        # Plan-aware B4 spot PnL: inject the structural B4 slice from plan
        # targets without clobbering FIFO B1/B2 lot attribution (inject_slice).
        _yb_has_b1, _yb_has_b2, _yb_has_b4 = held_etf_bucket_flags_from_positions(
            _u,
            pos,
            etf_to_under,
            etf_to_delta_map,
            flow_short_syms=flow_short_set,
        )
        _ibkr_qty = float(_spot_qty.get(_u, 0.0))
        _orphan_qty = _ibkr_qty - _qty_total
        _ledger_qty_map[_u] = dict(_bucket_qty[_u])
        _ledger_for_ratio = {
            "bucket_1": _cur_b1,
            "bucket_2": _cur_b2,
            "bucket_4": _cur_b4,
        }
        _plan_b4_qty_row = float(_plan_b4_qty_by_u.get(str(_u), 0.0))
        if b12_spot_exposure_method == "hedge_ratio":
            _spot_exp_ratio, _hr_meta = hedge_ratio_spot_bucket_ratios(
                str(_u),
                pos,
                etf_to_under,
                etf_to_delta_map,
                flow_short_syms=flow_short_set,
                b4_etf_syms=b4_etf_syms,
                b4_registry_b12_only=_u in b4_underlyings,
                b4_phr_by_etf=b4_phr_by_etf,
                partial_hedge_ratio_default=b4_partial_hedge_ratio_default,
                delta_floor=b12_hedge_delta_floor,
                ledger_qty=_ledger_for_ratio,
                plan_b4_qty=_plan_b4_qty_row,
                ibkr_qty=_ibkr_qty,
            )
            _hedge_spot_meta_map[str(_u)] = _hr_meta
        else:
            _spot_exp_ratio = resolve_underlying_spot_exposure_ratios(
                underlying=str(_u),
                ibkr_qty=_ibkr_qty,
                ledger_qty=_ledger_for_ratio,
                b12_spot_exposure_method=b12_spot_exposure_method,
                b12_spot_split_method=b12_spot_split_method,
                b4_registry_underlying=_u in b4_underlyings,
                pos=pos,
                etf_to_under=etf_to_under,
                etf_to_delta=etf_to_delta_map,
                flow_short_syms=flow_short_set,
                b4_etf_syms=b4_etf_syms,
                plan_ratio=_plan_ratio_b4.get(_u),
                plan_b4_pnl_mode=plan_b4_pnl_mode,
                yieldboost_spot_b2=(
                    _u in yieldboost_spot_b2_underlyings and _yb_has_b2
                ),
            )
        _spot_exposure_ratio_map[str(_u)] = _spot_exp_ratio

        _ibkr_qty_pre = _ibkr_qty
        if abs(_ibkr_qty_pre) > 1e-9:
            _orphan_frac_pre = abs(_ibkr_qty_pre - _qty_total) / abs(_ibkr_qty_pre)
        else:
            _orphan_frac_pre = 0.0
        _split_r_b1, _split_r_b2 = ledger_pnl_split_b1_b2_ratios(
            orphan_frac=_orphan_frac_pre,
            ledger_unreal_r_b1=_ru1,
            ledger_unreal_r_b2=_ru2,
            orphan_threshold=ledger_orphan_split_threshold,
        )
        _pnl_split_mode = "lot_timed"
        if _u in yieldboost_spot_b2_underlyings:
            apply_yieldboost_spot_b2_override(
                lot_components=_lot_components,
                underlying=_u,
                df=df,
                spot_carry_cols=_spot_carry_cols,
                r_b1=_split_r_b1,
                r_b2=_split_r_b2,
                r_b4=_r4,
                force_all_b2=_yb_has_b2,
            )
            _pnl_split_mode = "yieldboost_spot_b2"
        elif b12_spot_pnl_method != "ledger_fifo":
            _pnl_r1, _pnl_r2, _pnl_r4, _pnl_src = compose_spot_pnl_bucket_fractions(
                _spot_exp_ratio,
                b4_frac_signed=_b4_signed_frac.get(_u),
            )
            apply_spot_pnl_bucket_split(
                lot_components=_lot_components,
                underlying=_u,
                df=df,
                r_b1=_pnl_r1,
                r_b2=_pnl_r2,
                r_b4=_pnl_r4,
                spot_carry_cols=_spot_carry_cols,
            )
            _pnl_split_mode = f"sleeve_pnl:{_pnl_src}"
        elif plan_b4_inject_active and _u in _plan_ratio_b4 and b4_spot_pnl_inject_eligible(
            _u,
            ledger_b4_qty=_cur_b4,
            implied_b4_short_usd=_implied_b4_short_usd,
            min_usd=b4_attribution_min_usd,
        ):
            apply_plan_b4_spot_pnl_override(
                lot_components=_lot_components,
                underlying=_u,
                df=df,
                plan_ratio=_plan_ratio_b4[_u],
                spot_carry_cols=_spot_carry_cols,
                etf_to_delta_map=etf_to_delta_map,
                mode=plan_b4_pnl_mode,
                ledger_r_b1=_split_r_b1,
                ledger_r_b2=_split_r_b2,
                b4_frac_signed=_b4_signed_frac.get(_u),
            )
            _pnl_split_mode = plan_b4_pnl_mode
        elif plan_b4_inject_active and _u in _b4_signed_frac and b4_spot_pnl_inject_eligible(
            _u,
            ledger_b4_qty=_cur_b4,
            implied_b4_short_usd=_implied_b4_short_usd,
            min_usd=b4_attribution_min_usd,
        ):
            apply_plan_b4_spot_pnl_override(
                lot_components=_lot_components,
                underlying=_u,
                df=df,
                plan_ratio={"b1": 0.0, "b2": 0.0, "b4": 0.0},
                spot_carry_cols=_spot_carry_cols,
                etf_to_delta_map=etf_to_delta_map,
                mode="inject_slice",
                ledger_r_b1=_split_r_b1,
                ledger_r_b2=_split_r_b2,
                b4_frac_signed=_b4_signed_frac[_u],
            )
            _pnl_split_mode = "inject_slice"

        # ── Sleeve-rotation re-tag ──────────────────────────────────────────
        # The plain lot-timed path freezes spot P&L in whatever bucket the
        # FIFO lot was tagged at open. When a short-ETF sleeve is closed but
        # the long spot persists (sleeve rotation), FIFO keeps closing the
        # stale-tagged lots, mis-booking realized P&L to a sleeve that is no
        # longer held (e.g. NVDA spot frozen in B2/yieldBOOST while the live
        # short book is entirely levered/B1). Re-split the spot P&L to the
        # live sleeve capacity (``_spot_exp_ratio``) when the lot tags diverge
        # materially from it. Names whose lot tags already match the live
        # sleeve (genuinely held yieldBOOST, e.g. SMCI/MSTR) are untouched.
        if (
            spot_sleeve_rotation_retag
            and _pnl_split_mode == "lot_timed"
            and abs(_ibkr_qty) > 1e-9
        ):
            _lot_tot = abs(_cur_b1) + abs(_cur_b2) + abs(_cur_b4)
            if _lot_tot > 1e-9:
                _lot_share = {
                    "b1": abs(_cur_b1) / _lot_tot,
                    "b2": abs(_cur_b2) / _lot_tot,
                    "b4": abs(_cur_b4) / _lot_tot,
                }
                _live_w = {
                    "b1": float(_spot_exp_ratio.b1),
                    "b2": float(_spot_exp_ratio.b2),
                    "b4": float(_spot_exp_ratio.b4),
                }
                # Fire only when stale lots are parked in a bucket whose live
                # short-ETF sleeve is effectively gone (rotation). Names whose
                # sleeves are all still held keep their lot-timed FIFO split.
                _has_dead_sleeve = any(
                    _lot_share[_b] > spot_sleeve_rotation_tol
                    and _live_w[_b] < spot_sleeve_dead_threshold
                    for _b in ("b1", "b2", "b4")
                )
                if _has_dead_sleeve:
                    _rt1, _rt2, _rt4, _rt_src = compose_spot_pnl_bucket_fractions(
                        _spot_exp_ratio,
                        b4_frac_signed=_b4_signed_frac.get(_u),
                    )
                    apply_spot_pnl_bucket_split(
                        lot_components=_lot_components,
                        underlying=_u,
                        df=df,
                        r_b1=_rt1,
                        r_b2=_rt2,
                        r_b4=_rt4,
                        spot_carry_cols=_spot_carry_cols,
                    )
                    _pnl_split_mode = f"sleeve_rotation_retag:{_rt_src}"

        # ── Live-sleeve capacity cap (realized spot only) ───────────────────
        # When lot tags park more shares in B2 than the live yieldBOOST sleeve
        # can justify (e.g. SMCI lot 99% B2 vs live 72%), cap spot *realized*
        # at the live sleeve fraction and push excess to B1. Only on lot_timed
        # names (dead-sleeve retag already handled separately). Does not touch
        # unrealized/carry or excluded underlyings (AMD is under-tagged B2).
        if (
            spot_sleeve_capacity_cap
            and _pnl_split_mode == "lot_timed"
            and str(_u) not in spot_sleeve_capacity_exclude
            and abs(_ibkr_qty) > 1e-9
        ):
            _lot_tot_cap = abs(_cur_b1) + abs(_cur_b2) + abs(_cur_b4)
            if _lot_tot_cap > 1e-9:
                _lot_r2_cap = abs(_cur_b2) / _lot_tot_cap
                _live_b2_cap = float(_spot_exp_ratio.b2)
                if (
                    _lot_r2_cap > _live_b2_cap + spot_sleeve_capacity_tol
                    and _live_b2_cap >= spot_sleeve_dead_threshold
                ):
                    _cap1, _cap2, _cap4, _cap_src = compose_spot_pnl_bucket_fractions(
                        _spot_exp_ratio,
                        b4_frac_signed=_b4_signed_frac.get(_u),
                    )
                    apply_spot_pnl_bucket_split(
                        lot_components=_lot_components,
                        underlying=_u,
                        df=df,
                        r_b1=_cap1,
                        r_b2=_cap2,
                        r_b4=_cap4,
                        spot_carry_cols=_spot_carry_cols,
                        cols={"realized_pnl"},
                    )
                    _pnl_split_mode = f"sleeve_capacity_cap:{_cap_src}"

        if not plan_b4_inject_active and abs(_ibkr_qty) > 1e-9:
            collapse_spot_b4_pnl_into_b12(
                _lot_components,
                _u,
                ledger_r_b1=_split_r_b1,
                ledger_r_b2=_split_r_b2,
                spot_carry_cols=_spot_carry_cols,
            )

        # PnL spot ratios (FIFO ledger path / diagnostics).
        _spot_pnl_ratio = resolve_underlying_spot_ratios(
            underlying=str(_u),
            ibkr_qty=_ibkr_qty,
            ledger_qty=_ledger_for_ratio,
            plan_ratio=_plan_ratio_b4.get(_u),
            plan_b4_pnl_mode=plan_b4_pnl_mode,
            yieldboost_spot_b2=(
                _u in yieldboost_spot_b2_underlyings and _yb_has_b2
            ),
            ledger_r_b1=None,
            ledger_r_b2=None,
            b12_spot_split_method=b12_spot_split_method,
            b4_registry_underlying=_u in b4_underlyings,
            etf_to_under=etf_to_under,
            etf_to_delta=etf_to_delta_map,
            pos=pos,
            flow_short_syms=flow_short_set,
        )

        _state_rows.append(
            {
                "run_date": run_date,
                "underlying": _u,
                "qty_total": _qty_total,
                "qty_b1": _cur_b1,
                "qty_b2": _cur_b2,
                "qty_b4": _cur_b4,
                "qty_b4_plan": float(_plan_b4_qty_by_u.get(str(_u), 0.0)),
                "plan_b4_usd": float((_plan_sleeve_usd.get(str(_u)) or {}).get("b4", 0.0)),
                "qty_b4_structural": float(_b4_struct_qty_by_u.get(str(_u), 0.0)),
                "qty_b4_etf_implied": float(_implied_b4_short_qty.get(str(_u), 0.0)),
                "etf_implied_b4_usd": float(_implied_b4_short_usd.get(str(_u), 0.0)),
                "b4_structural_source": _b4_struct_src_by_u.get(str(_u), "none"),
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
        _plan_b2_usd = float((_plan_sleeve_usd.get(_u) or {}).get("b2", 0.0))
        _plan_b4_usd_row = float((_plan_sleeve_usd.get(_u) or {}).get("b4", 0.0))
        if _u in _b4_plan_exposure_underlyings and abs(_plan_b4_qty_row) > 1e-9:
            _expo_b4_src = "plan"
        elif abs(_cur_b4) > 1e-9:
            _expo_b4_src = "ledger"
        else:
            _expo_b4_src = "none"
        _hr_meta_row = _hedge_spot_meta_map.get(str(_u))
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
                "plan_b4_frac": float((_plan_ratio_b4.get(_u) or {}).get("b4", 0.0)),
                "plan_b4_usd": _plan_b4_usd_row,
                "plan_b4_qty": _plan_b4_qty_row,
                "exposure_b4_qty_source": _expo_b4_src,
                "ledger_b2_frac": _ru2,
                "pnl_split_mode": _pnl_split_mode,
                "plan_b2_usd": _plan_b2_usd,
                "ratio_spot_b1": _spot_pnl_ratio.b1,
                "ratio_spot_b2": _spot_pnl_ratio.b2,
                "ratio_spot_b4": _spot_pnl_ratio.b4,
                "ratio_spot_source": _spot_pnl_ratio.source,
                "ratio_spot_exposure_b1": _spot_exp_ratio.b1,
                "ratio_spot_exposure_b2": _spot_exp_ratio.b2,
                "ratio_spot_exposure_b4": _spot_exp_ratio.b4,
                "ratio_spot_exposure_source": _spot_exp_ratio.source,
                "hedge_target_usd_b1": (
                    _hr_meta_row.hedge_target_usd_b1 if _hr_meta_row else 0.0
                ),
                "hedge_target_usd_b2": (
                    _hr_meta_row.hedge_target_usd_b2 if _hr_meta_row else 0.0
                ),
                "hedge_target_usd_b4": (
                    _hr_meta_row.hedge_target_usd_b4 if _hr_meta_row else 0.0
                ),
                "hedge_alloc_usd_b1": (
                    _hr_meta_row.hedge_alloc_usd_b1 if _hr_meta_row else 0.0
                ),
                "hedge_alloc_usd_b2": (
                    _hr_meta_row.hedge_alloc_usd_b2 if _hr_meta_row else 0.0
                ),
                "hedge_target_qty_b1": (
                    _hr_meta_row.hedge_target_qty_b1 if _hr_meta_row else 0.0
                ),
                "hedge_target_qty_b2": (
                    _hr_meta_row.hedge_target_qty_b2 if _hr_meta_row else 0.0
                ),
                "hedge_target_qty_b4": (
                    _hr_meta_row.hedge_target_qty_b4 if _hr_meta_row else 0.0
                ),
                "hedge_alloc_qty_b1": (
                    _hr_meta_row.hedge_alloc_qty_b1 if _hr_meta_row else 0.0
                ),
                "hedge_alloc_qty_b2": (
                    _hr_meta_row.hedge_alloc_qty_b2 if _hr_meta_row else 0.0
                ),
            }
        )
        if _u in _plan_ratio_b4 and abs(_ru2 - _r2) > 0.25:
            print(
                f"[ACCOUNTING] NOTE: {_u} ledger unrealized b2={_ru2:.1%} vs "
                f"held/plan current b2={_r2:.1%} (mode={plan_b4_pnl_mode})"
            )
        if abs(_ibkr_qty) > 1e-9 and abs(_orphan_qty) / abs(_ibkr_qty) > 0.10:
            print(
                f"[ACCOUNTING] NOTE: {_u} orphan_qty={_orphan_qty:.1f} "
                f"({abs(_orphan_qty) / abs(_ibkr_qty):.1%} of ibkr_qty={_ibkr_qty:.1f})"
            )

    _state_today_df = pd.DataFrame(_state_rows, columns=state_cols)
    _state_merged = pd.concat([_state_hist[state_cols], _state_today_df], ignore_index=True)
    _state_merged = _state_merged.drop_duplicates(subset=["run_date", "underlying"], keep="last")
    _state_merged = _state_merged.sort_values(["run_date", "underlying"]).reset_index(drop=True)
    # Purge ETF-leg contamination: rows whose ``underlying`` is actually a
    # packaged ETF (e.g. APLZ) leaked into the spot share-ledger state under
    # the pre-fix ``_is_etf_leg`` bug. They carry a spurious spot-only ledger
    # (~0) that never reconciles against the full ETF position. The ledger only
    # tracks true spot underlyings, so drop these rows across all dates.
    if not _state_merged.empty:
        _etf_state_mask = _state_merged["underlying"].apply(
            lambda _su: _is_etf_leg(str(_su), str(_su), etf_to_under)
        )
        _n_purged = int(_etf_state_mask.sum())
        if _n_purged:
            print(
                f"[ACCOUNTING] purged {_n_purged} ETF-leg row(s) from "
                f"underlying_bucket_state.csv "
                f"(e.g. {', '.join(sorted(set(_state_merged.loc[_etf_state_mask, 'underlying'].astype(str)))[:6])})"
            )
            _state_merged = _state_merged[~_etf_state_mask].copy()
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
        # The share ledger only tracks true spot underlyings. ETF tickers can
        # leak into _all_underlyings via stale persisted state rows (created
        # before the _is_etf_leg fix); comparing their spot-only ledger (~0)
        # against the full ETF IBKR position manufactures false drift
        # (e.g. APLZ ledger -2000 vs position -42000). Skip ETF legs so the
        # reconciliation reflects genuine spot-underlying drift only.
        if _is_etf_leg(_u, _u, etf_to_under):
            continue
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
    # - ETFs are ALWAYS bucketed by leg class (independent of held status)
    # - Inverse ETFs (beta < 0):
    #     * flow program shorts -> bucket_3
    #     * all others          -> bucket_4
    # - Core levered ETFs (beta > 1.5) -> bucket_1
    # - Standard / yieldboost / flow low-β (0 < β ≤ 1.5) -> bucket_2
    # - Spot positions split using lot ledger + unified spot ratios
    df["bucket"] = ""
    df["leg_class"] = ""
    neg_etf = is_etf & (sym_beta < 0)
    neg_flow = neg_etf & df["symbol"].isin(flow_short_set)
    neg_non_flow = neg_etf & (~df["symbol"].isin(flow_short_set))
    df.loc[neg_flow, "bucket"] = "bucket_3"
    df.loc[neg_flow, "leg_class"] = "flow_inverse"
    df.loc[neg_non_flow, "bucket"] = "bucket_4"
    df.loc[neg_non_flow, "leg_class"] = "inverse_b4_etf"
    df.loc[is_etf & df["symbol"].isin(b4_etf_syms), "bucket"] = "bucket_4"
    df.loc[is_etf & df["symbol"].isin(b4_etf_syms), "leg_class"] = "inverse_b4_etf"
    # Flow-program low-β yieldBOOST shorts (0 < β ≤ 1.5, e.g. NVYY/TSYY) belong
    # to the flow program -> bucket_3 for P&L (their P&L is dominated by the
    # PIL/dividend carry of the flow short, not the directional yieldBOOST
    # sleeve). NOTE: this only re-labels the *P&L* bucket. The ETF leg is left
    # in the B2 hedge sleeve for spot-ratio / exposure purposes so the
    # proportional underlying spot split (e.g. NVDA/TSLA) is unaffected.
    _flow_low_b3 = (
        is_etf
        & df["symbol"].isin(flow_low_delta_syms)
        & (df["bucket"] == "")
    )
    df.loc[_flow_low_b3, "bucket"] = "bucket_3"
    df.loc[_flow_low_b3, "leg_class"] = "flow_low_delta"
    df.loc[
        is_etf & (sym_beta > 1.5) & (df["bucket"] == ""),
        "bucket",
    ] = "bucket_1"
    df.loc[
        is_etf & (sym_beta > 1.5) & (df["leg_class"] == ""),
        "leg_class",
    ] = "core_levered_etf"
    _b2_etf = (
        is_etf
        & (sym_beta > 0)
        & (sym_beta <= 1.5)
        & (df["bucket"] == "")
    )
    df.loc[_b2_etf, "bucket"] = "bucket_2"
    df.loc[
        _b2_etf & df["symbol"].isin(flow_short_set) & (df["leg_class"] == ""),
        "leg_class",
    ] = "flow_low_delta"
    df.loc[
        _b2_etf & (df["leg_class"] == ""),
        "leg_class",
    ] = "yieldboost_etf"

    # Only non-ETF rows split by lot ledger (exact shares); current_only uses held ratios.
    needs_split = (~is_etf) & (df["bucket"] == "")
    df.loc[needs_split, "leg_class"] = "spot"
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
        # Multiple inverse ETFs can share one underlying (MSTR ↔ MSDD /
        # MSTZ / SMST). Spread the underlying's B4-attributed PnL evenly
        # across its pairs so the sum across pair rows ties back to the
        # underlying-level total instead of being multiplied by N.
        _under_etf_count = (
            b4_registry.groupby("underlying")["etf"].nunique().to_dict()
        )
        for _, pr in b4_registry.iterrows():
            etf = str(pr["etf"])
            under = str(pr["underlying"])
            etf_pnl = df[(df["symbol"] == etf) & (df["bucket"] == "bucket_4")]
            und_pnl = df[(df["symbol"] == under) & (df["bucket"] == "bucket_4")]
            n_etfs = max(int(_under_etf_count.get(under, 1) or 1), 1)
            row = {"underlying": under, "etf": etf, "delta": float(pr["delta"])}
            for c in base_cols:
                row[c] = float(etf_pnl[c].sum()) + (
                    float(und_pnl[c].sum()) / n_etfs
                )
            _pair_rows.append(row)
    pnl_bucket_4_by_pair = (
        pd.DataFrame(_pair_rows).sort_values("total_pnl", ascending=False)
        if _pair_rows
        else pd.DataFrame(columns=["underlying", "etf", "delta"] + base_cols)
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

    # ── Net exposure (delta-normalized) ──
    bucket3_only_syms = set(flow_inverse_bucket3_syms)
    pos_b124 = pos[(~pos["symbol"].isin(bucket3_only_syms))].copy()
    pos_b124 = _filter_positions_blacklist(
        pos_b124, blocked_symbols, blocked_underlyings, etf_to_under
    )
    b124_underlyings = (
        set(pnl_by_underlying["underlying"].dropna().astype(str)) if not pnl_by_underlying.empty else set()
    )
    # Underlyings registered as part of bucket-4 structural pairs MUST be
    # included in the book exposure aggregate even if they have zero
    # realized/unrealized P&L today (e.g. brand-new structural shorts
    # like APLD). Otherwise ``net_exposure_by_underlying.csv`` omits
    # them, while ``net_exposure_bucket_4.csv`` still includes the leg,
    # breaking bucket↔book reconciliation by the orphan's notional.
    if b4_underlyings:
        b124_underlyings = b124_underlyings | {str(u) for u in b4_underlyings}
    b124_underlyings = {u for u in b124_underlyings if u not in blocked_underlyings}
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
        _d["_is_etf"] = _d.apply(
            lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
            axis=1,
        )
        _d["_beta"] = pd.to_numeric(_d["symbol"].map(etf_to_delta_map), errors="coerce").fillna(1.0)

        _b4_spot_b12_only_expo: set[str] = set()
        if b12_spot_exposure_method in {
            "ledger_fifo",
            "sleeve_offset",
            "hedge_ratio",
            "sleeve_balance",
        }:
            _b4_spot_b12_only_expo = set(_b4_plan_exposure_underlyings)
            for _u_s12e, _short_usd_s12e in _implied_b4_short_usd.items():
                if abs(float(_short_usd_s12e)) >= float(b4_attribution_min_usd):
                    _b4_spot_b12_only_expo.add(str(_u_s12e))

        # Ratio-split B1/B2/B4 net exposure (``b12_spot_exposure_method``).
        _hedge_spot_ratio_map: dict[str, SpotBucketRatios] = {}
        if b12_spot_exposure_method == "held_exposure_waterfall":
            _hedge_spot_ratio_map = build_hedge_residual_spot_ratio_map(
                _d,
                etf_to_under=etf_to_under,
                etf_to_delta_map=etf_to_delta_map,
                flow_short_set=flow_short_set,
                b4_etf_syms=b4_etf_syms,
                plan_ratio_b4=_plan_ratio_b4,
                plan_b4_qty_by_u=_plan_b4_qty_by_u,
                spot_qty=_spot_qty,
                spot_marks=_spot_marks,
                plan_sleeve_usd=_plan_sleeve_usd,
                etf_implied_short_usd=_implied_b4_short_usd,
                b4_attribution_mode=b4_attribution_mode,
                b4_attribution_min_usd=b4_attribution_min_usd,
            )
        elif b12_spot_exposure_method == "sleeve_offset":
            _hedge_spot_ratio_map = sleeve_offset_spot_ratios_from_exposure_detail(
                _d,
                etf_to_under=etf_to_under,
                etf_to_delta_map=etf_to_delta_map,
                flow_short_set=flow_short_set,
                b4_etf_syms=b4_etf_syms,
                b4_spot_b12_only=_b4_spot_b12_only_expo,
            )
            _spot_exposure_ratio_map.update(_hedge_spot_ratio_map)
        elif b12_spot_exposure_method == "hedge_ratio":
            _hedge_spot_ratio_map, _ = hedge_ratio_spot_ratios_from_exposure_detail(
                _d,
                etf_to_under=etf_to_under,
                etf_to_delta_map=etf_to_delta_map,
                flow_short_set=flow_short_set,
                b4_etf_syms=b4_etf_syms,
                b4_spot_b12_only=_b4_spot_b12_only_expo,
                b4_phr_by_etf=b4_phr_by_etf,
                partial_hedge_ratio_default=b4_partial_hedge_ratio_default,
                delta_floor=b12_hedge_delta_floor,
                ledger_qty_by_u=_ledger_qty_map,
                plan_b4_qty_by_u=_plan_b4_qty_by_u,
                spot_qty_by_u=_spot_qty,
            )
            _spot_exposure_ratio_map.update(_hedge_spot_ratio_map)
        elif b12_spot_exposure_method == "sleeve_balance":
            _hedge_spot_ratio_map = sleeve_balance_spot_ratios_from_exposure_detail(
                _d,
                etf_to_under=etf_to_under,
                etf_to_delta_map=etf_to_delta_map,
                flow_short_set=flow_short_set,
                b4_etf_syms=b4_etf_syms,
                b4_spot_b12_only=_b4_spot_b12_only_expo,
            )
            _spot_exposure_ratio_map.update(_hedge_spot_ratio_map)
        else:
            _hedge_spot_ratio_map = dict(_spot_exposure_ratio_map)

        def _leg_bucket_ratios(_row: pd.Series) -> tuple[float, float, float]:
            _sym = str(_row["symbol"])
            _u = str(_row["underlying"])
            if bool(_row["_is_etf"]):
                _beta = float(_row["_beta"])
                _bkt, _ = classify_etf_leg_bucket(
                    _sym,
                    _beta,
                    flow_short_set=flow_short_set,
                    b4_etf_syms=b4_etf_syms,
                )
                if _bkt == "bucket_1":
                    return 1.0, 0.0, 0.0
                if _bkt == "bucket_2":
                    return 0.0, 1.0, 0.0
                if _bkt == "bucket_4":
                    return 0.0, 0.0, 1.0
                return 0.0, 0.0, 0.0
            if _sym != _u:
                return 0.0, 0.0, 0.0
            _spot_ratio = _hedge_spot_ratio_map.get(_u)
            if _spot_ratio is not None:
                return _spot_ratio.b1, _spot_ratio.b2, _spot_ratio.b4
            _pos_q = float(_spot_qty.get(_sym, 0.0))
            _ledger = _ledger_qty_map.get(_u, {})
            return _spot_exposure_bucket_ratios(_pos_q, _ledger)

        _spot_ratios = _d.apply(_leg_bucket_ratios, axis=1, result_type="expand")
        _d["_ratio_b1"] = _spot_ratios[0]
        _d["_ratio_b2"] = _spot_ratios[1]
        _d["_ratio_b4"] = _spot_ratios[2]
        _d["leg_class"] = _d.apply(
            lambda r: (
                "spot"
                if (not bool(r["_is_etf"]) and str(r["symbol"]) == str(r["underlying"]))
                else classify_etf_leg_bucket(
                    str(r["symbol"]),
                    float(r["_beta"]),
                    flow_short_set=flow_short_set,
                    b4_etf_syms=b4_etf_syms,
                )[1]
            ),
            axis=1,
        )
        # Pure bucket-4 ETF legs: no b1/b2 residual
        _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b1"] = 0.0
        _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b2"] = 0.0
        _d.loc[_d["symbol"].isin(b4_etf_syms), "_ratio_b4"] = 1.0
        _d = _normalize_exposure_bucket_ratios(
            _d,
            etf_to_delta_map=etf_to_delta_map,
            flow_short_set=flow_short_set,
            b4_etf_syms=b4_etf_syms,
            b4_spot_b12_only_underlyings=_b4_spot_b12_only_expo,
        )
        _b4_structural_usd_map = resolve_b4_structural_short_usd_by_underlying(
            b4_underlyings,
            b4_underlying_exposure_mode=b4_underlying_exposure_mode,
            b4_plan_exposure_underlyings=_b4_plan_exposure_underlyings,
            plan_b4_qty_by_u=_plan_b4_qty_by_u,
            plan_sleeve_usd=_plan_sleeve_usd,
            spot_qty=_spot_qty,
            spot_marks=_spot_marks,
            implied_b4_short_usd=_implied_b4_short_usd,
            b4_attribution_mode=b4_attribution_mode,
            b4_attribution_min_usd=b4_attribution_min_usd,
        )
        if _b4_structural_usd_map:
            _d = apply_b4_structural_short_to_exposure_detail(
                _d,
                _b4_structural_usd_map,
                hedge_spot_ratio_map=_hedge_spot_ratio_map,
            )

        def _scale_detail(_detail: pd.DataFrame, ratio_col: str) -> pd.DataFrame:
            out = _detail[_detail[ratio_col].abs() > 1e-12].copy()
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
        exposure_b1_full_df, exposure_b2_full_df, exposure_b4_full_df = build_audit_full_bucket_exposure(
            _d,
            etf_to_under=etf_to_under,
            etf_to_delta_map=etf_to_delta_map,
            flow_short_set=flow_short_set,
            b4_etf_syms=b4_etf_syms,
            spot_ratio_map=_hedge_spot_ratio_map,
            spot_qty=_spot_qty,
            ledger_qty_map=_ledger_qty_map,
        )
        spot_exposure_by_underlying_df = build_net_exposure_spot_by_underlying(
            _d, _hedge_spot_ratio_map
        )
        bucket_exposure_detail_df = _d[
            [
                "underlying",
                "symbol",
                "leg_class",
                "net_notional_usd",
                "gross_notional_usd",
                "_ratio_b1",
                "_ratio_b2",
                "_ratio_b4",
            ]
        ].copy()
        exposure_unbucketed_df = build_net_exposure_unbucketed(_d, etf_to_under)
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
        exposure_b1_full_df = exposure_b2_full_df = exposure_b4_full_df = exposure_b1_df.copy()
        spot_exposure_by_underlying_df = build_net_exposure_spot_by_underlying(
            pd.DataFrame(), _spot_exposure_ratio_map
        )
        bucket_exposure_detail_df = pd.DataFrame(
            columns=[
                "underlying",
                "symbol",
                "leg_class",
                "net_notional_usd",
                "gross_notional_usd",
                "_ratio_b1",
                "_ratio_b2",
                "_ratio_b4",
            ]
        )
        _hedge_spot_ratio_map = {}
        exposure_unbucketed_df = pd.DataFrame(
            columns=["underlying", "net_notional_usd", "gross_notional_usd", "orphan_frac"]
        )

    # ── Pair-exposure underlying qty ─────────────────────────────────────
    # Pair view: underlying leg uses, in priority,
    #   1. plan structural short qty (when ``b4_underlying_exposure_mode=plan_structural``
    #      and the latest ``inverse_decay_bucket4`` plan row hits ``b4_plan_exposure_min_usd``)
    #   2. ETF-implied structural short qty (when ``b4_attribution_mode=etf_implied``
    #      and the implied short hits ``b4_attribution_min_usd``)
    #   3. FIFO ``qty_b4`` from the share ledger
    # Use the canonical structural short (plan -> etf_implied) computed once
    # above and persisted to the state file, so the exposure pair view and the
    # recorded ledger value cannot drift apart. Fall back to FIFO ``qty_b4``
    # only when neither plan nor implied produced a structural short.
    _b4_under_qty: dict[str, float] = {}
    for _u_b4 in b4_underlyings:
        _u_str = str(_u_b4)
        _struct_q = float(_b4_struct_qty_by_u.get(_u_str, 0.0))
        if abs(_struct_q) > 1e-9:
            _b4_under_qty[_u_str] = _struct_q
        else:
            _b4_under_qty[_u_str] = float(
                _ledger_qty_map.get(_u_str, {}).get("bucket_4", 0.0)
            )
    exposure_b4_df, exposure_b4_detail_df = compute_bucket4_pair_exposure(
        _filter_positions_blacklist(pos, blocked_symbols, blocked_underlyings, etf_to_under),
        b4_registry,
        underlying_b4_qty=_b4_under_qty,
    )
    # Ratio-split B4 (totals ``net_exposure_bucket_4``) includes inverse ETF legs plus
    # the structural short underlying carved from each physical spot line. Pair CSV
    # is the same economics with per-leg detail rows.

    if blocked_underlyings:
        exposure_df = _filter_exposure_df(exposure_df, blocked_underlyings)
        exposure_b1_df = _filter_exposure_df(exposure_b1_df, blocked_underlyings)
        exposure_b2_df = _filter_exposure_df(exposure_b2_df, blocked_underlyings)
        exposure_b4_from_b124_df = _filter_exposure_df(
            exposure_b4_from_b124_df, blocked_underlyings
        )
        exposure_b1_full_df = _filter_exposure_df(exposure_b1_full_df, blocked_underlyings)
        exposure_b2_full_df = _filter_exposure_df(exposure_b2_full_df, blocked_underlyings)
        exposure_b4_full_df = _filter_exposure_df(exposure_b4_full_df, blocked_underlyings)
        spot_exposure_by_underlying_df = spot_exposure_by_underlying_df[
            ~spot_exposure_by_underlying_df["underlying"].astype(str).isin(blocked_underlyings)
        ].copy()
        exposure_b4_df = _filter_exposure_df(exposure_b4_df, blocked_underlyings)
        if not exposure_b4_detail_df.empty and "underlying" in exposure_b4_detail_df.columns:
            exposure_b4_detail_df = exposure_b4_detail_df[
                ~exposure_b4_detail_df["underlying"].astype(str).isin(blocked_underlyings)
            ].copy()

    exposure_b1_df.to_csv(outdir / "net_exposure_bucket_1.csv", index=False)
    exposure_b2_df.to_csv(outdir / "net_exposure_bucket_2.csv", index=False)
    exposure_b1_full_df.to_csv(outdir / "net_exposure_bucket_1_full.csv", index=False)
    exposure_b2_full_df.to_csv(outdir / "net_exposure_bucket_2_full.csv", index=False)
    exposure_b4_full_df.to_csv(outdir / "net_exposure_bucket_4_full.csv", index=False)
    if not spot_exposure_by_underlying_df.empty:
        spot_exposure_by_underlying_df.to_csv(
            outdir / "net_exposure_spot_by_underlying.csv", index=False
        )
    if not bucket_exposure_detail_df.empty:
        bucket_exposure_detail_df.to_csv(outdir / "bucket_exposure_detail.csv", index=False)
    if not exposure_unbucketed_df.empty:
        exposure_unbucketed_df.to_csv(outdir / "net_exposure_unbucketed.csv", index=False)
    leg_class_df = df[["symbol", "underlying", "bucket", "leg_class", "total_pnl"]].copy()
    leg_class_df.to_csv(outdir / "bucket_leg_classification.csv", index=False)
    bucket_ratio_recon_df, spot_ratio_max_diff_exp, spot_ratio_max_diff_pnl = build_bucket_ratio_reconciliation(
        df,
        spot_exposure_by_underlying_df,
        _hedge_spot_ratio_map,
    )
    if not bucket_ratio_recon_df.empty:
        bucket_ratio_recon_df.to_csv(outdir / "bucket_ratio_reconciliation.csv", index=False)
    spot_ratio_gate_passed = spot_ratio_max_diff_exp <= 0.001
    spot_ratio_gate_mode = "exposure_vs_plan_or_hedge_residual"
    exposure_df = exposure_df.sort_values("net_notional_usd", ascending=False)
    exposure_df.to_csv(outdir / "net_exposure_by_underlying.csv", index=False)
    if not exposure_b4_detail_df.empty:
        exposure_b4_detail_df.to_csv(outdir / "net_exposure_bucket_4_detail.csv", index=False)

    _b4_plan_recon_df = build_b4_plan_ledger_reconciliation(
        run_date=run_date,
        b4_underlyings=set(b4_underlyings),
        plan_sleeve_usd=_plan_sleeve_usd,
        spot_qty=_spot_qty,
        ledger_qty_map=_ledger_qty_map,
        spot_marks=_spot_marks,
        plan_exposure_underlyings=_b4_plan_exposure_underlyings,
        etf_implied_short_usd=_implied_b4_short_usd,
        etf_implied_short_qty=_implied_b4_short_qty,
        structural_short_source=_b4_attribution_source,
    )
    if not _b4_plan_recon_df.empty:
        _b4_plan_recon_df.to_csv(outdir / "b4_plan_ledger_reconciliation.csv", index=False)
        _n_missing = int(_b4_plan_recon_df["ledger_missing_b4"].sum())
        if _n_missing:
            print(
                f"[ACCOUNTING] B4 ledger reconciliation: {_n_missing} name(s) "
                f"with intended structural short (plan or implied) but ledger "
                f"qty_b4~=0 — see {outdir / 'b4_plan_ledger_reconciliation.csv'}"
            )

    # Bucket 3 exposure: flow-program inverse ETFs only (β < 0).
    pos_b3 = pos[pos["symbol"].isin(flow_inverse_bucket3_syms)].copy()
    pos_b3 = _filter_positions_blacklist(
        pos_b3, blocked_symbols, blocked_underlyings, etf_to_under
    )
    if not pos_b3.empty:
        pos_b3["delta"] = pos_b3["symbol"].map(etf_to_delta_map).fillna(1.0)
        pos_b3["mv_base"] = pos_b3["position"] * pos_b3["delta"] * pos_b3["markPrice"] * pos_b3["fxRateToBase"]
        pos_b3["gross_mv_base"] = pos_b3["mv_base"].abs()
        pos_b3["overlay_type"] = "flow_program"
        exposure_b3_df = (
            pos_b3.groupby(["symbol", "overlay_type"], as_index=False)
            .agg(
                net_notional_usd=("mv_base", "sum"),
                gross_notional_usd=("gross_mv_base", "sum"),
                n_legs=("symbol", "nunique"),
            )
            .sort_values("net_notional_usd", ascending=False)
        )
    else:
        exposure_b3_df = pd.DataFrame(
            columns=["symbol", "overlay_type", "net_notional_usd", "gross_notional_usd", "n_legs"]
        )
    if blocked_symbols and not exposure_b3_df.empty and "symbol" in exposure_b3_df.columns:
        exposure_b3_df = exposure_b3_df[
            ~exposure_b3_df["symbol"].astype(str).isin(blocked_symbols)
        ].copy()
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
        # Split attribution (b124 ratios) sums to book with b1+b2; pair CSV is separate.
        "net_exposure_bucket_4": float(exposure_b4_from_b124_df["net_notional_usd"].sum())
        if not exposure_b4_from_b124_df.empty
        else 0.0,
        "gross_exposure_bucket_4": float(exposure_b4_from_b124_df["gross_notional_usd"].sum())
        if not exposure_b4_from_b124_df.empty
        else 0.0,
        "net_exposure_unbucketed": float(exposure_unbucketed_df["net_notional_usd"].sum())
        if not exposure_unbucketed_df.empty
        else 0.0,
        "gross_exposure_unbucketed": float(exposure_unbucketed_df["gross_notional_usd"].sum())
        if not exposure_unbucketed_df.empty
        else 0.0,
        "net_exposure_bucket_4_pair": float(exposure_b4_df["net_notional_usd"].sum())
        if not exposure_b4_df.empty
        else 0.0,
        "gross_exposure_bucket_4_pair": float(exposure_b4_df["gross_notional_usd"].sum())
        if not exposure_b4_df.empty
        else 0.0,
        "yfinance_override": bool(use_yfinance),
        "yfinance_symbols_overridden": len(yf_closes),
        "bucket_split_method": split_method,
        "bucket_component_split_method": component_split_method,
        "b12_spot_split_method": b12_spot_split_method,
        "b12_spot_exposure_method": b12_spot_exposure_method,
        "b12_spot_pnl_method": b12_spot_pnl_method,
        "b12_pnl_mode": b12_pnl_mode,
        "plan_b4_pnl_mode": plan_b4_pnl_mode,
        "plan_sleeve_bucketing": plan_sleeve_bucketing,
        "b4_underlying_exposure_mode": b4_underlying_exposure_mode,
        "b4_plan_exposure_underlyings": sorted(_b4_plan_exposure_underlyings),
        "b4_underlying_attribution": b4_attribution_mode,
        "b4_attribution_min_usd": float(b4_attribution_min_usd),
        "b4_partial_hedge_ratio_default": float(b4_partial_hedge_ratio_default),
        "b4_etf_implied_underlyings": sorted(_implied_b4_short_usd),
        "b4_attribution_sources": {
            "plan": sorted([k for k, v in _b4_attribution_source.items() if v == "plan"]),
            "etf_implied": sorted(
                [k for k, v in _b4_attribution_source.items() if v == "etf_implied"]
            ),
        },
        "bucket2_flow_low_delta_symbols": sorted(flow_low_delta_syms),
        "bucket3_flow_program_symbols": sorted(flow_inverse_bucket3_syms),
        "net_exposure_bucket_1_full": float(exposure_b1_full_df["net_notional_usd"].sum())
        if not exposure_b1_full_df.empty
        else 0.0,
        "net_exposure_bucket_2_full": float(exposure_b2_full_df["net_notional_usd"].sum())
        if not exposure_b2_full_df.empty
        else 0.0,
        "net_exposure_bucket_4_full": float(exposure_b4_full_df["net_notional_usd"].sum())
        if not exposure_b4_full_df.empty
        else 0.0,
        "spot_ratio_max_diff": float(spot_ratio_max_diff_exp),
        "spot_ratio_max_diff_pnl": float(spot_ratio_max_diff_pnl),
        "spot_ratio_gate_passed": bool(spot_ratio_gate_passed),
        "spot_ratio_gate_mode": spot_ratio_gate_mode,
        "exposure_reconciliation_tol_gross_pct": exposure_reconciliation_tol_gross_pct,
        "exposure_reconciliation_tol_net_abs_usd": exposure_reconciliation_tol_net_abs_usd,
        "bucket_state_path": str(bucket_state_path),
        "bucket_realized_underlyings_from_trade_timing": int(len(_realized_ratio_from_trades)),
        "pnl_underlying_scope": "bucket_1_2_4",
        "bucket4_pairs": int(len(b4_registry)),
        "excluded_underlyings_exposure": sorted(blocked_underlyings),
        "excluded_symbols_exposure": sorted(blocked_symbols),
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
    print(
        f"[ACCOUNTING] B1/B2: exposure_spot={b12_spot_exposure_method} | "
        f"pnl_spot={b12_spot_pnl_method} | "
        f"pnl={b12_pnl_mode} | component={component_split_method}"
    )
    print(f"[ACCOUNTING] Bucket split method: {split_method} | plan_b4_pnl_mode: {plan_b4_pnl_mode}")
    print(f"[ACCOUNTING] Trade-timed realized underlyings: {len(_realized_ratio_from_trades)}")
    print(
        f"[ACCOUNTING] Exposure (b124): {n_exp} underlyings, "
        f"net={totals['net_exposure_total']:,.0f}, gross={totals['gross_exposure_total']:,.0f}"
    )
    print(f"[ACCOUNTING] Bucket 1: net={totals['net_exposure_bucket_1']:,.0f}, gross={totals['gross_exposure_bucket_1']:,.0f}")
    print(f"[ACCOUNTING] Bucket 2: net={totals['net_exposure_bucket_2']:,.0f}, gross={totals['gross_exposure_bucket_2']:,.0f}")
    print(f"[ACCOUNTING] Bucket 3: net={totals['net_exposure_bucket_3']:,.0f}, gross={totals['gross_exposure_bucket_3']:,.0f}")
    print(f"[ACCOUNTING] Bucket 4: net={totals['net_exposure_bucket_4']:,.0f}, gross={totals['gross_exposure_bucket_4']:,.0f}")
    if float(totals.get("net_exposure_unbucketed", 0.0)) != 0.0:
        print(
            f"[ACCOUNTING] Unbucketed spot: net={totals['net_exposure_unbucketed']:,.0f}, "
            f"gross={totals.get('gross_exposure_unbucketed', 0.0):,.0f}"
        )

    # ── Reconciliation gate ────────────────────────────────────────────
    # Bucket gross/net components (b1+b2+b4 ratio split) must sum to the book
    # aggregate within configured tolerance. Bucket 3 is an overlay and is
    # excluded from the sum. Under ``ledger_fifo``, spot is ratio-split across
    # B1/B2 (B4 structural names keep spot off ``_ratio_b4`` only).
    _book_gross = float(totals["gross_exposure_total"])
    _book_net = float(totals["net_exposure_total"])
    _bucket_gross_sum = (
        float(totals["gross_exposure_bucket_1"])
        + float(totals["gross_exposure_bucket_2"])
        + float(totals["gross_exposure_bucket_4"])
    )
    _bucket_net_sum = (
        float(totals["net_exposure_bucket_1"])
        + float(totals["net_exposure_bucket_2"])
        + float(totals["net_exposure_bucket_4"])
        + float(totals.get("net_exposure_unbucketed", 0.0))
    )
    if abs(_book_gross) > 1e-6:
        _gross_diff_pct = abs(_bucket_gross_sum - _book_gross) / abs(_book_gross)
    else:
        _gross_diff_pct = 0.0
    _net_diff_abs = abs(_bucket_net_sum - _book_net)
    print(
        f"[ACCOUNTING] Reconciliation: bucket_sum_gross={_bucket_gross_sum:,.0f} "
        f"(diff {_gross_diff_pct:.4%}), bucket_sum_net={_bucket_net_sum:,.0f} "
        f"(abs diff ${_net_diff_abs:,.0f})"
    )
    _exposure_gate_failed = (
        _gross_diff_pct > exposure_reconciliation_tol_gross_pct
        or _net_diff_abs > exposure_reconciliation_tol_net_abs_usd
    )
    if _exposure_gate_failed:
        print(
            "[ACCOUNTING][ERROR] Bucket exposures do not reconcile to book aggregate. "
            f"Gross diff {_gross_diff_pct:.4%} (tol {exposure_reconciliation_tol_gross_pct:.4%}), "
            f"net abs diff ${_net_diff_abs:,.0f} "
            f"(tol ${exposure_reconciliation_tol_net_abs_usd:,.0f})."
        )
        return 2

    if spot_ratio_consistency_gate and not spot_ratio_gate_passed:
        print(
            "[ACCOUNTING][ERROR] Spot exposure vs canonical ratio gate failed: "
            f"max share diff {spot_ratio_max_diff_exp:.4%} > 0.10%. "
            f"See {outdir / 'bucket_ratio_reconciliation.csv'}"
        )
        return 2

    if spot_ratio_max_diff_pnl > 0.05:
        print(
            f"[ACCOUNTING] NOTE: spot PnL vs canonical max diff "
            f"{spot_ratio_max_diff_pnl:.2%} (informational)"
        )
    elif spot_ratio_max_diff_exp > 0.001:
        print(
            f"[ACCOUNTING] NOTE: spot exposure vs canonical max diff "
            f"{spot_ratio_max_diff_exp:.4%}"
        )

    return 0


if __name__ == "__main__":
    _yf_flag = "--yf" in sys.argv
    _args = [a for a in sys.argv[1:] if a != "--yf"]
    _run_date = _args[0] if _args else None
    raise SystemExit(main(_run_date, use_yfinance=_yf_flag if _yf_flag else None))