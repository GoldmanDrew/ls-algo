#!/usr/bin/env python3
"""
tax_lot_view.py

Open-position tax-lot inventory plus realized-P&L simulation, used by
the Stage 2 tax router for Phase 2b resize decisions.

Backed by IBKR Flex's ``OpenPositions`` section with
``levelOfDetail="LOT"`` (configured once in Flex Query Manager). Each
lot row carries cost basis, open date, and holding period info.

This module does **not** override IBKR's tax-lot matching — the API
does not support per-order SpecID. It only *predicts* realized P&L
under an assumed account default lot method (typically HIFO or
MaxLossUtilization, set once in TWS Account Configuration).

Public API
----------
    Lot, LotView, EstPnL              dataclasses
    parse_open_position_lots(xml)     XML -> {symbol: [Lot, ...]}
    lots_from_flex_rows(rows)         dict-row sequence -> {sym: [Lot]}
    load_lot_views_for_run_dir(rd)    finds latest flex_positions.xml
    LotView.estimate_realized_pnl(...)  HIFO/FIFO/LIFO/MAXLOSS sim
    LotView.st_lt_split(days)         (st_qty, lt_qty) tuple
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

LotMethod = Literal["HIFO", "FIFO", "LIFO", "MAXLOSS"]


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Lot:
    """One open tax lot."""
    symbol: str
    side: str                       # "long" or "short"
    qty: float                      # positive magnitude
    open_date: date
    cost_basis_per_share: float
    lot_id: str = ""                # synthetic stable id


@dataclass
class EstPnL:
    """Result of a simulated lot close."""
    realized_pnl_usd: float = 0.0
    qty_consumed: float = 0.0
    lots_consumed: List[Tuple[str, float, float, int]] = field(default_factory=list)
    # tuples: (lot_id, qty_taken, basis_per_share, holding_days)
    st_qty: float = 0.0
    lt_qty: float = 0.0


@dataclass
class LotView:
    """Per-symbol lot inventory and realized-P&L simulator."""
    symbol: str
    lots: List[Lot] = field(default_factory=list)
    asof: date = field(default_factory=date.today)

    @property
    def total_qty(self) -> float:
        return sum(lot.qty for lot in self.lots)

    @property
    def avg_cost(self) -> float:
        if not self.lots or self.total_qty == 0:
            return 0.0
        total_basis = sum(lot.qty * lot.cost_basis_per_share for lot in self.lots)
        return total_basis / self.total_qty

    def st_lt_split(self, st_lt_holding_days: int = 365) -> Tuple[float, float]:
        """Return (st_qty, lt_qty) split based on holding period as of self.asof.

        IRS rule: long-term means held more than one year (>365 days).
        We treat lots opened ON or BEFORE (asof - st_lt_holding_days) as LT.
        """
        st = lt = 0.0
        boundary = self.asof - timedelta(days=st_lt_holding_days)
        for lot in self.lots:
            if lot.open_date > boundary:
                st += lot.qty
            else:
                lt += lot.qty
        return st, lt

    def estimate_realized_pnl(
        self,
        qty: float,
        current_px: float,
        *,
        method: LotMethod = "HIFO",
        st_lt_holding_days: int = 365,
        prefer_lt: bool = False,
    ) -> EstPnL:
        """Simulate closing ``qty`` shares at ``current_px``.

        Parameters
        ----------
        qty : float
            Number of shares to close (positive magnitude; this is a SELL on
            a long lot, BTC on a short lot).
        current_px : float
            Mark price for the close.
        method : "HIFO" | "FIFO" | "LIFO" | "MAXLOSS"
            Lot ordering. "MAXLOSS" picks the lots that maximize realized
            loss (or minimize realized gain) given current_px.
        st_lt_holding_days : int
            Boundary in days; lots older than this are LT.
        prefer_lt : bool
            If True, LT lots are consumed before ST regardless of method.

        Returns
        -------
        EstPnL
            Pure simulation; ``self.lots`` is not mutated.
        """
        if qty <= 0 or not self.lots or current_px <= 0:
            return EstPnL()

        if method == "HIFO":
            ordered = sorted(self.lots, key=lambda l: -l.cost_basis_per_share)
        elif method == "FIFO":
            ordered = sorted(self.lots, key=lambda l: l.open_date)
        elif method == "LIFO":
            ordered = sorted(self.lots, key=lambda l: -l.open_date.toordinal())
        elif method == "MAXLOSS":
            # For longs: realized = qty * (px - basis); minimize -> highest basis
            # For shorts: realized = qty * (basis - px); minimize -> lowest basis
            # Use signed-loss-per-share key so the same comparator works.
            def _ml_key(lot: Lot) -> float:
                if lot.side == "long":
                    return -(lot.cost_basis_per_share - current_px)  # most negative pnl first
                else:
                    return -(current_px - lot.cost_basis_per_share)
            ordered = sorted(self.lots, key=_ml_key)
        else:
            ordered = list(self.lots)

        if prefer_lt:
            boundary = self.asof - timedelta(days=st_lt_holding_days)
            lt_lots = [l for l in ordered if l.open_date <= boundary]
            st_lots = [l for l in ordered if l.open_date > boundary]
            ordered = lt_lots + st_lots

        consumed: List[Tuple[str, float, float, int]] = []
        remaining = float(qty)
        realized = 0.0
        st_consumed = lt_consumed = 0.0
        boundary = self.asof - timedelta(days=st_lt_holding_days)

        for lot in ordered:
            if remaining <= 0:
                break
            take = min(remaining, lot.qty)
            holding_days = (self.asof - lot.open_date).days
            if lot.side == "long":
                pnl = take * (current_px - lot.cost_basis_per_share)
            else:
                pnl = take * (lot.cost_basis_per_share - current_px)
            realized += pnl
            consumed.append((lot.lot_id, take, lot.cost_basis_per_share, holding_days))
            if lot.open_date > boundary:
                st_consumed += take
            else:
                lt_consumed += take
            remaining -= take

        return EstPnL(
            realized_pnl_usd=realized,
            qty_consumed=float(qty) - remaining,
            lots_consumed=consumed,
            st_qty=st_consumed,
            lt_qty=lt_consumed,
        )


# --------------------------------------------------------------------------
# Flex XML / dict-row parsing
# --------------------------------------------------------------------------

def _parse_flex_date(s: str) -> Optional[date]:
    """Flex date: 'YYYYMMDD;HHMMSS', 'YYYYMMDD HHMMSS', 'YYYYMMDD', or empty."""
    if not s:
        return None
    head = str(s).split(";")[0].split(" ")[0].strip()
    if len(head) >= 8 and head[:8].isdigit():
        try:
            return datetime.strptime(head[:8], "%Y%m%d").date()
        except ValueError:
            return None
    try:
        return date.fromisoformat(head[:10])
    except ValueError:
        return None


def lots_from_flex_rows(rows: Iterable[dict]) -> dict[str, List[Lot]]:
    """Convert a sequence of dict rows (one per Flex OpenPositions LOT) into
    ``{symbol: [Lot, ...]}``. Tolerant to missing / alternate field names so
    tests can construct rows directly without simulating XML.
    """
    by_sym: dict[str, List[Lot]] = {}
    for r in rows:
        sym = str(r.get("symbol") or r.get("sym") or "").upper().strip()
        if not sym:
            continue
        try:
            qty = float(r.get("position", r.get("qty", 0)) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty == 0:
            continue
        side = "long" if qty > 0 else "short"
        try:
            cost_basis = float(
                r.get("costBasisPrice",
                      r.get("cost_basis_per_share",
                            r.get("cost_basis", 0))) or 0
            )
        except (TypeError, ValueError):
            cost_basis = 0.0

        open_dt_raw = (
            r.get("openDateTime")
            or r.get("holdingPeriodDateTime")
            or r.get("open_date")
            or ""
        )
        open_d = _parse_flex_date(str(open_dt_raw)) or date.today()

        lot_id = str(
            r.get("originatingOrderID")
            or r.get("lot_id")
            or f"{sym}_{open_d.isoformat()}_{abs(qty):.0f}"
        )

        by_sym.setdefault(sym, []).append(
            Lot(
                symbol=sym, side=side, qty=abs(qty),
                open_date=open_d, cost_basis_per_share=float(cost_basis),
                lot_id=lot_id,
            )
        )
    return by_sym


def parse_open_position_lots(pos_xml: Path) -> List[dict]:
    """Parse Flex OpenPositions with ``levelOfDetail="LOT"`` into raw dict rows.

    Returns a list of dicts (one per lot). Use ``lots_from_flex_rows`` to
    convert to ``{symbol: [Lot, ...]}``. The two-step shape mirrors the
    existing ``parse_open_positions`` style in ibkr_accounting.py.
    """
    pos_xml = Path(pos_xml)
    root = ET.parse(pos_xml).getroot()
    op = root.find(".//OpenPositions")
    if op is None:
        raise ValueError(f"No OpenPositions section in {pos_xml}")
    rows: List[dict] = []
    for node in op:
        a = node.attrib
        # Skip summary-level rows; only consume LOT-level
        if str(a.get("levelOfDetail", "")).upper() != "LOT":
            continue
        rows.append(
            {
                "symbol":            a.get("symbol", ""),
                "underlyingSymbol":  a.get("underlyingSymbol", ""),
                "position":          a.get("position", "0"),
                "costBasisPrice":    a.get("costBasisPrice", "0"),
                "costBasisMoney":    a.get("costBasisMoney", "0"),
                "markPrice":         a.get("markPrice", "0"),
                "openDateTime":      a.get("openDateTime", ""),
                "holdingPeriodDateTime": a.get("holdingPeriodDateTime", ""),
                "originatingOrderID": a.get("originatingOrderID", ""),
            }
        )
    return rows


def load_lot_views_for_run_dir(
    run_dir: Path,
    *,
    fallback_search_root: Optional[Path] = None,
    asof: Optional[date] = None,
) -> dict[str, LotView]:
    """Locate the latest ``flex_positions.xml`` and build per-symbol LotViews.

    Parameters
    ----------
    run_dir : Path
        Preferred location (e.g. ``data/runs/<today>/accounting/``).
    fallback_search_root : Path, optional
        If ``run_dir/flex_positions.xml`` is absent, walk this root for the
        most recent ``flex_positions.xml`` (typically ``data/runs``).
    asof : date, optional
        Override "today" when computing holding period boundaries (testing).

    Returns
    -------
    dict[str, LotView]
        Empty dict if no XML is found — callers must handle this case
        (the tax router treats absent lot data as "pass through").
    """
    target = Path(run_dir) / "flex_positions.xml"
    chosen: Optional[Path] = target if target.exists() else None
    if chosen is None and fallback_search_root is not None:
        latest_mtime = -1.0
        for p in Path(fallback_search_root).rglob("flex_positions.xml"):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > latest_mtime:
                latest_mtime = m
                chosen = p
    if chosen is None:
        return {}

    try:
        rows = parse_open_position_lots(chosen)
    except (ET.ParseError, ValueError):
        return {}

    by_sym = lots_from_flex_rows(rows)
    asof_d = asof or date.today()
    return {
        sym: LotView(symbol=sym, lots=lots, asof=asof_d)
        for sym, lots in by_sym.items()
    }
