#!/usr/bin/env python3
"""
Convert IBKR "Portfolio" CSV export (Financial Instrument, Position, Last, …)
into minimal Flex XML stubs so `ibkr_accounting.py` can find the expected files.

Also prints delta-normalized net exposure by underlying (same as
`compute_net_exposure` in ibkr_accounting.py).

Usage:
  python scripts/ibkr_portfolio_csv_to_flex.py path/to/export.csv 2026-05-15
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Project imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ibkr_accounting import (  # noqa: E402
    EXCLUDE_SYMBOLS,
    PROJECT_ROOT,
    canonical_symbol,
    compute_net_exposure,
    load_etf_delta_map,
    parse_open_positions,
    yyyymmdd_from_run_date,
)


def _parse_num(x) -> float | None:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.upper() in ("NOMD", "—", "-"):
        return None
    s = s.replace(",", "")
    neg = False
    if s.startswith("'-"):
        neg = True
        s = s[2:]
    elif s.startswith("-"):
        neg = True
        s = s[1:]
    s = re.sub(r"^[CD]", "", s)  # Last column quirks like C16.09
    s = s.strip()
    if not s or s.upper() == "NOMD":
        return None
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def _price_from_row(row: pd.Series) -> float | None:
    last = _parse_num(row.get("Last"))
    if last is not None and last > 0:
        return last
    bid, ask = _parse_num(row.get("Bid")), _parse_num(row.get("Ask"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    ap = _parse_num(row.get("Avg Price"))
    if ap is not None and ap > 0:
        return ap
    return None


def csv_to_positions_df(csv_path: Path, etf_screened: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, dtype=str, encoding_errors="replace")
    col_sym = "Financial Instrument"
    if col_sym not in raw.columns:
        raise ValueError(f"Expected column {col_sym!r}; got {list(raw.columns)[:8]}…")

    etf_to_under, etf_to_delta = load_etf_delta_map(etf_screened)

    rows: list[dict] = []
    for _, r in raw.iterrows():
        sym = canonical_symbol(str(r.get(col_sym, "") or "").strip())
        if not sym or sym in EXCLUDE_SYMBOLS:
            continue
        pos = _parse_num(r.get("Position"))
        if pos is None or abs(pos) < 1e-12:
            continue
        px = _price_from_row(r)
        if px is None or px <= 0:
            continue
        und = etf_to_under.get(sym, sym)
        und = canonical_symbol(str(und))
        if und in EXCLUDE_SYMBOLS:
            continue
        pv = float(pos) * float(px)
        rows.append(
            {
                "symbol": sym,
                "underlyingSymbol": und,
                "position": float(pos),
                "markPrice": float(px),
                "positionValue": float(pv),
                "fxRateToBase": 1.0,
            }
        )

    if not rows:
        raise ValueError("No valid positions (need numeric Position and Last/Bid/Ask/Avg Price).")

    df = pd.DataFrame(rows)
    df["positionValue_base"] = df["positionValue"] * df["fxRateToBase"]
    if (df["position"] < 0).sum() == 0 and (df["positionValue_base"] < 0).sum() > 0:
        df["is_short"] = df["positionValue_base"] < 0
    else:
        df["is_short"] = df["position"] < 0
    return df


def write_flex_stubs(out_dir: Path, positions_df: pd.DataFrame, run_date: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ymd = yyyymmdd_from_run_date(run_date)
    acct = "CSVIMPORT"

    op_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<FlexQueryResponse queryName="Positions" type="AF">',
        '<FlexStatements count="1">',
        f'<FlexStatement accountId="{acct}" fromDate="{ymd}" toDate="{ymd}" period="Snapshot" whenGenerated="{ymd};000000">',
        "<OpenPositions>",
    ]
    for _, row in positions_df.iterrows():
        sym = str(row["symbol"]).replace("&", "&amp;")
        und = str(row["underlyingSymbol"]).replace("&", "&amp;")
        desc = sym
        op_lines.append(
            f'<OpenPosition accountId="{acct}" currency="USD" fxRateToBase="1" '
            f'assetCategory="STK" subCategory="COMMON" symbol="{sym}" description="{desc}" '
            f'underlyingSymbol="{und}" '
            f'position="{row["position"]:.8f}" markPrice="{row["markPrice"]:.8f}" '
            f'positionValue="{row["positionValue"]:.8f}" '
            f'reportDate="{ymd}" />'
        )
    op_lines.extend(["</OpenPositions>", "</FlexStatement>", "</FlexStatements>", "</FlexQueryResponse>"])
    (out_dir / "flex_positions.xml").write_text("\n".join(op_lines), encoding="utf-8")

    cash_xml = """<?xml version="1.0" encoding="UTF-8"?>
<FlexQueryResponse queryName="Cash" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="CSVIMPORT" fromDate="{ymd}" toDate="{ymd}" period="YTD" whenGenerated="{ymd};000000">
<CashTransactions>
</CashTransactions>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>
""".format(ymd=ymd)
    (out_dir / "flex_cash.xml").write_text(cash_xml, encoding="utf-8")

    trades_xml = """<?xml version="1.0" encoding="UTF-8"?>
<FlexQueryResponse queryName="Trades" type="AF">
<FlexStatements count="1">
<FlexStatement accountId="CSVIMPORT" fromDate="{ymd}" toDate="{ymd}" period="YTD" whenGenerated="{ymd};000000">
<FIFOPerformanceSummaryInBase>
</FIFOPerformanceSummaryInBase>
<Trades>
</Trades>
</FlexStatement>
</FlexStatements>
</FlexQueryResponse>
""".format(ymd=ymd)
    (out_dir / "flex_trades.xml").write_text(trades_xml, encoding="utf-8")


def pair_label(sym: str, und: str) -> str:
    if sym == und:
        return f"{und} (spot)"
    return f"{und} | {sym}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("run_date", help="YYYY-MM-DD")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to data/runs/<run_date>/ibkr_flex/",
    )
    args = ap.parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    run_date = args.run_date.strip()
    out_dir = args.out_dir or (PROJECT_ROOT / "data" / "runs" / run_date / "ibkr_flex")
    screened = PROJECT_ROOT / "data" / "etf_screened_today.csv"

    pos_df = csv_to_positions_df(csv_path, screened)
    write_flex_stubs(out_dir, pos_df, run_date)

    # Validate positions XML round-trip
    parsed = parse_open_positions(out_dir / "flex_positions.xml")
    exp_df, detail_df = compute_net_exposure(
        out_dir / "flex_positions.xml",
        screened,
        pnl_underlyings=None,
        positions_df=parsed,
    )

    out_detail = out_dir.parent / "net_exposure_by_pair_from_csv.csv"
    if not detail_df.empty:
        dd = detail_df.copy()
        dd["pair"] = [pair_label(str(s), str(u)) for s, u in zip(dd["symbol"], dd["underlying"])]
        dd = dd.sort_values(["underlying", "net_notional_usd"], ascending=[True, False])
        dd.to_csv(out_detail, index=False)

    agg_path = out_dir.parent / "net_exposure_by_underlying_from_csv.csv"
    exp_df.to_csv(agg_path, index=False)

    print(f"Wrote: {out_dir / 'flex_positions.xml'}")
    print(f"Wrote: {out_dir / 'flex_cash.xml'}")
    print(f"Wrote: {out_dir / 'flex_trades.xml'}")
    print(f"Wrote: {agg_path}")
    print(f"Wrote: {out_detail}")
    print()
    print("Beta-normalized net exposure by underlying (USD, strategy universe only):")
    print(exp_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
