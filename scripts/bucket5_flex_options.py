"""Bucket 5 Production B — contract-safe Flex option accounting.

A parallel, option-aware ingestion path for the B5 put book (plan section 5.2).
It reads the raw Flex XMLs directly and keys everything on ``conId`` plus
expiry / strike / right / trading class / multiplier — SPX/XSP option rows
never flow through the stock bucket logic that aggregates by ``symbol``.

Attribution rule: an option trade belongs to B5 iff its ``orderReference``
starts with ``B5P|``. Open option positions are matched to B5 via the ledger's
conId lot book (positions have no orderRef in Flex), so 0DTE SPXW inventory —
which is same-day and should never appear overnight — can never be claimed.

Outputs (written into the accounting output directory):
- ``b5_option_trades.csv``     every B5P| option trade with full contract identity
- ``b5_option_positions.csv``  open option positions matched to B5 ledger lots
- ``pnl_bucket_5_options.csv`` one-row daily put-book summary
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

B5_ORDER_REF_PREFIX = "B5P|"
OPTION_ASSET_CATEGORIES = {"OPT", "FOP"}


def _f(a: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(a.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def parse_option_trades(trades_xml: Path) -> pd.DataFrame:
    """All OPTION trades from flex_trades.xml with contract identity preserved."""
    cols = [
        "dateTime", "conId", "symbol", "underlyingSymbol", "assetCategory",
        "putCall", "expiry", "strike", "multiplier", "tradingClass",
        "buySell", "quantity", "tradePrice", "tradeMoney", "ibCommission",
        "fifoPnlRealized", "orderReference", "openCloseIndicator", "execId", "orderID",
    ]
    if not trades_xml.exists():
        return pd.DataFrame(columns=cols)
    r = ET.parse(trades_xml).getroot()
    node = r.find(".//Trades")
    if node is None:
        return pd.DataFrame(columns=cols)
    rows: list[dict] = []
    for child in node:
        if child.tag != "Trade":
            continue
        a = child.attrib
        if str(a.get("assetCategory", "")).upper() not in OPTION_ASSET_CATEGORIES:
            continue
        fx = _f(a, "fxRateToBase", 1.0) or 1.0
        rows.append({
            "dateTime": str(a.get("dateTime", "") or ""),
            "conId": str(a.get("conid", a.get("conId", "")) or ""),
            "symbol": str(a.get("symbol", "") or "").strip(),
            "underlyingSymbol": str(a.get("underlyingSymbol", "") or "").strip(),
            "assetCategory": str(a.get("assetCategory", "") or ""),
            "putCall": str(a.get("putCall", "") or ""),
            "expiry": str(a.get("expiry", "") or ""),
            "strike": _f(a, "strike"),
            "multiplier": _f(a, "multiplier", 100.0),
            "tradingClass": str(a.get("tradingClass", "") or ""),
            "buySell": str(a.get("buySell", "") or "").upper(),
            "quantity": _f(a, "quantity"),
            "tradePrice": _f(a, "tradePrice") * fx,
            "tradeMoney": _f(a, "tradeMoney") * fx,
            "ibCommission": _f(a, "ibCommission") * fx,
            "fifoPnlRealized": _f(a, "fifoPnlRealized") * fx,
            "orderReference": str(a.get("orderReference", "") or ""),
            "openCloseIndicator": str(a.get("openCloseIndicator", "") or ""),
            "execId": str(a.get("execId", a.get("ibExecID", "")) or ""),
            "orderID": str(a.get("ibOrderID", a.get("orderID", "")) or ""),
        })
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("dateTime").reset_index(drop=True) if not df.empty else df


def parse_option_positions(pos_xml: Path) -> pd.DataFrame:
    """All OPTION open positions from flex_positions.xml, keyed by conId."""
    cols = [
        "conId", "symbol", "underlyingSymbol", "assetCategory", "putCall",
        "expiry", "strike", "multiplier", "tradingClass",
        "position", "markPrice", "positionValue_base", "costBasis_base",
    ]
    if not pos_xml.exists():
        return pd.DataFrame(columns=cols)
    r = ET.parse(pos_xml).getroot()
    node = r.find(".//OpenPositions")
    if node is None:
        return pd.DataFrame(columns=cols)
    rows: list[dict] = []
    for child in node:
        a = child.attrib
        if str(a.get("assetCategory", "")).upper() not in OPTION_ASSET_CATEGORIES:
            continue
        fx = _f(a, "fxRateToBase", 1.0) or 1.0
        rows.append({
            "conId": str(a.get("conid", a.get("conId", "")) or ""),
            "symbol": str(a.get("symbol", "") or "").strip(),
            "underlyingSymbol": str(a.get("underlyingSymbol", "") or "").strip(),
            "assetCategory": str(a.get("assetCategory", "") or ""),
            "putCall": str(a.get("putCall", "") or ""),
            "expiry": str(a.get("expiry", "") or ""),
            "strike": _f(a, "strike"),
            "multiplier": _f(a, "multiplier", 100.0),
            "tradingClass": str(a.get("tradingClass", "") or ""),
            "position": _f(a, "position"),
            "markPrice": _f(a, "markPrice"),
            "positionValue_base": _f(a, "positionValue") * fx,
            "costBasis_base": _f(a, "costBasisMoney", _f(a, "costBasis")) * fx,
        })
    return pd.DataFrame(rows, columns=cols)


def build_b5_option_books(
    *,
    flex_trades_path: Path,
    flex_positions_path: Path,
    outdir: Path,
    ledger_conids: set[str] | None = None,
    print_fn=print,
) -> dict:
    """Write the three B5 put-book artifacts; returns the summary dict.

    ``ledger_conids``: conIds of B5 ledger lots (open or historical). Positions
    are attributed to B5 only when their conId is in this set; trades only when
    their orderReference is in the B5P| namespace. Passing None loads the
    default ledger.
    """
    if ledger_conids is None:
        try:
            from scripts.bucket5_ledger import Bucket5Ledger
        except ImportError:  # pragma: no cover
            from bucket5_ledger import Bucket5Ledger  # type: ignore
        ledger_conids = set(Bucket5Ledger().build_lots().keys())

    trades = parse_option_trades(flex_trades_path)
    b5_trades = (
        trades[trades["orderReference"].astype(str).str.startswith(B5_ORDER_REF_PREFIX)]
        if not trades.empty else trades
    )
    positions = parse_option_positions(flex_positions_path)
    b5_positions = (
        positions[positions["conId"].astype(str).isin({str(c) for c in ledger_conids})]
        if not positions.empty else positions
    )

    non_b5_option_trades = int(len(trades) - len(b5_trades)) if not trades.empty else 0
    non_b5_option_positions = int(len(positions) - len(b5_positions)) if not positions.empty else 0

    outdir.mkdir(parents=True, exist_ok=True)
    b5_trades.to_csv(outdir / "b5_option_trades.csv", index=False)
    b5_positions.to_csv(outdir / "b5_option_positions.csv", index=False)

    premium_paid = float(-b5_trades.loc[b5_trades["quantity"] > 0, "tradeMoney"].sum()) if not b5_trades.empty else 0.0
    sale_proceeds = float(b5_trades.loc[b5_trades["quantity"] < 0, "tradeMoney"].abs().sum()) if not b5_trades.empty else 0.0
    realized = float(b5_trades["fifoPnlRealized"].sum()) if not b5_trades.empty else 0.0
    commissions = float(b5_trades["ibCommission"].sum()) if not b5_trades.empty else 0.0
    mark_value = float(b5_positions["positionValue_base"].sum()) if not b5_positions.empty else 0.0
    cost_basis = float(b5_positions["costBasis_base"].sum()) if not b5_positions.empty else 0.0
    open_contracts = int(b5_positions["position"].sum()) if not b5_positions.empty else 0

    summary = {
        "n_b5_option_trades": int(len(b5_trades)),
        "n_b5_option_positions": int(len(b5_positions)),
        "open_contracts": open_contracts,
        "put_mark_value_usd": mark_value,
        "put_cost_basis_usd": cost_basis,
        "unrealized_pnl_usd": mark_value - cost_basis,
        "realized_pnl_usd": realized,
        "premium_paid_usd": abs(premium_paid),
        "sale_proceeds_usd": sale_proceeds,
        "commissions_usd": commissions,
        "non_b5_option_trades_excluded": non_b5_option_trades,
        "non_b5_option_positions_excluded": non_b5_option_positions,
    }
    pd.DataFrame([summary]).to_csv(outdir / "pnl_bucket_5_options.csv", index=False)
    print_fn(
        f"[B5-ACCT] Put book: {summary['n_b5_option_positions']} open lots "
        f"({open_contracts} contracts), mark ${mark_value:,.0f}, "
        f"realized ${realized:,.0f}; excluded {non_b5_option_trades} non-B5 option trades "
        f"(0DTE/other namespaces)."
    )
    return summary
