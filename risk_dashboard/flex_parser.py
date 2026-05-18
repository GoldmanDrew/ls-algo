"""Parse the Flex XML files saved by ``ibkr_flex.py``.

Only the sections needed by the risk dashboard are parsed; the existing
``ibkr_accounting.py`` is the source of truth for full accounting. The
parsers here are tolerant of missing files / sections so the dashboard
can be built from a partial run (e.g. flex_trades timed out) without
crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET


@dataclass
class FlexPosition:
    symbol: str
    description: str
    quantity: float
    mark_price: float
    position_value: float
    asset_category: str
    underlying: str | None
    multiplier: float
    fx_rate_to_base: float
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class FlexBorrowFee:
    symbol: str
    value_date: str
    quantity: float
    collateral_amount: float
    fee_rate_pct: float
    interest: float
    raw: dict[str, str] = field(default_factory=dict)


def _to_float(v: str | None, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _iter_elements(path: Path, tag: str) -> Iterable[ET.Element]:
    """Yield ``tag`` elements from a Flex XML file (no DOM build).

    Flex XMLs can be large (tens of MB), so iterparse keeps memory flat.
    """
    if not path.is_file():
        return
    for _, elem in ET.iterparse(str(path), events=("end",)):
        if elem.tag == tag:
            yield elem
            elem.clear()


def parse_positions(flex_positions_xml: Path) -> list[FlexPosition]:
    """Read ``OpenPosition`` rows from a positions Flex XML."""
    out: list[FlexPosition] = []
    for elem in _iter_elements(flex_positions_xml, "OpenPosition"):
        a = elem.attrib
        out.append(
            FlexPosition(
                symbol=a.get("symbol", "").strip(),
                description=a.get("description", "").strip(),
                quantity=_to_float(a.get("position")),
                mark_price=_to_float(a.get("markPrice")),
                position_value=_to_float(a.get("positionValue")),
                asset_category=a.get("assetCategory", "").strip(),
                underlying=(a.get("underlyingSymbol") or None),
                multiplier=_to_float(a.get("multiplier"), 1.0),
                fx_rate_to_base=_to_float(a.get("fxRateToBase"), 1.0),
                raw=dict(a),
            )
        )
    return out


def parse_borrow_fee_details(flex_borrow_xml: Path) -> list[FlexBorrowFee]:
    """Read borrow-fee detail rows. The IBKR section name is
    ``SLBOpenContract``/``SLBActivity``/``SLBFee`` depending on the
    Flex template -- try the most common ones."""
    out: list[FlexBorrowFee] = []
    for tag in ("SLBFee", "SLBActivity", "SLBOpenContract"):
        for elem in _iter_elements(flex_borrow_xml, tag):
            a = elem.attrib
            out.append(
                FlexBorrowFee(
                    symbol=a.get("symbol", "").strip(),
                    value_date=a.get("valueDate", "") or a.get("date", ""),
                    quantity=_to_float(a.get("quantity")),
                    collateral_amount=_to_float(a.get("collateralAmount")),
                    fee_rate_pct=_to_float(a.get("feeRate")) * 100.0
                    if "feeRate" in a
                    else _to_float(a.get("feeRatePct")),
                    interest=_to_float(a.get("interest")),
                    raw=dict(a),
                )
            )
    return out


def summarize_positions(positions: list[FlexPosition]) -> dict[str, Any]:
    """High-level summary used on Page 1 of the dashboard."""
    if not positions:
        return {
            "n_positions": 0,
            "long_notional_usd": 0.0,
            "short_notional_usd": 0.0,
            "gross_notional_usd": 0.0,
            "net_notional_usd": 0.0,
        }

    long_n = sum(p.position_value for p in positions if p.position_value > 0)
    short_n = sum(p.position_value for p in positions if p.position_value < 0)
    return {
        "n_positions": len(positions),
        "long_notional_usd": long_n,
        "short_notional_usd": short_n,
        "gross_notional_usd": long_n + abs(short_n),
        "net_notional_usd": long_n + short_n,
    }


def summarize_borrow(borrow_rows: list[FlexBorrowFee]) -> dict[str, Any]:
    """High-level borrow summary."""
    if not borrow_rows:
        return {
            "n_rows": 0,
            "total_interest_usd": 0.0,
            "max_fee_rate_pct": 0.0,
            "names_over_30pct": [],
            "names_over_90pct": [],
        }

    by_symbol: dict[str, dict[str, float]] = {}
    for r in borrow_rows:
        s = by_symbol.setdefault(
            r.symbol, {"interest": 0.0, "fee_rate_pct": 0.0, "rows": 0.0}
        )
        s["interest"] += r.interest
        s["fee_rate_pct"] = max(s["fee_rate_pct"], r.fee_rate_pct)
        s["rows"] += 1

    total_interest = sum(s["interest"] for s in by_symbol.values())
    max_rate = max((s["fee_rate_pct"] for s in by_symbol.values()), default=0.0)
    over_30 = sorted(
        [
            {"symbol": k, "fee_rate_pct": v["fee_rate_pct"]}
            for k, v in by_symbol.items()
            if v["fee_rate_pct"] >= 30.0
        ],
        key=lambda d: -d["fee_rate_pct"],
    )
    over_90 = [d for d in over_30 if d["fee_rate_pct"] >= 90.0]
    fee_rate_by_symbol = {
        k: float(v["fee_rate_pct"]) for k, v in by_symbol.items()
    }
    return {
        "n_rows": len(borrow_rows),
        "n_symbols": len(by_symbol),
        "total_interest_usd": total_interest,
        "max_fee_rate_pct": max_rate,
        "names_over_30pct": over_30,
        "names_over_90pct": over_90,
        "fee_rate_by_symbol": fee_rate_by_symbol,
    }
