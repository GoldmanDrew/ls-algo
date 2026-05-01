"""
Scrape YMAX fund-of-funds weights, map / expand underlying sleeves, aggregate
implied reference exposure, and plot the top 20 names.

Methodology (approximate):
  - YMAX weights from stockanalysis.com/etf/ymax/holdings/ (changes over time).
  - Single-ticker YieldMax ETFs: full YMAX child weight -> that underlying.
  - Multi-holding YieldMax ETFs (LFGY, GPTY, CHPY, ULTY, SOXY): allocate the
    YMAX child weight across inner rows with a valid equity/ETF ticker,
    proportional to the row's % of the sum of those row weights (options/cash
    rows labeled n/a are excluded).
  - OARK: sleeve weight attributed to ARKK (fund references ARKK options).
  - QDTY: sleeve weight attributed to QQQ (Nasdaq-100 0DTE sleeve proxy).
  - MSST: sleeve weight attributed to MSTR (MSTR-linked structure).

This is a sleeve-weight attribution, not an options delta or risk-neutral hedge.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (compatible; ls-algo research)"}

OUT_REPO = Path(__file__).resolve().parents[1] / "Levered Research"
OUT_DROPBOX = Path(r"C:\Users\werdn\Dropbox\Levered ETFs\Levered Research")

# Single-name YieldMax option-income ETFs held by YMAX (ticker -> underlying)
SINGLE_STOCK = {
    "AMDY": "AMD",
    "AMZY": "AMZN",
    "HOOY": "HOOD",
    "TSMY": "TSM",
    "HIYY": "HIMS",
    "MARO": "MARA",
    "XYZY": "XYZ",
    "PYPY": "PYPL",
    "GOOY": "GOOGL",
    "NFLY": "NFLX",
    "ABNY": "ABNB",
    "FBY": "META",
    "APLY": "AAPL",
    "MSFO": "MSFT",
    "NVDY": "NVDA",
    "MSTY": "MSTR",
    "JPO": "JPM",
}

# YMAX children whose basket is scraped from the child's holdings page
PORTFOLIO_FUNDS = frozenset({"LFGY", "GPTY", "CHPY", "ULTY", "SOXY"})

# Proxies for sleeves that are mostly derivatives / index-linked in the scrape
PROXY_FULL_SLEEVE = {
    "OARK": "ARKK",
    "QDTY": "QQQ",
    "MSST": "MSTR",
}


def _fetch_holdings_table(symbol: str) -> list[list[str]]:
    url = f"https://stockanalysis.com/etf/{symbol.lower()}/holdings/"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for tr in soup.select("table tbody tr"):
        cols = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(cols) >= 4:
            rows.append(cols)
    return rows


def _parse_pct(s: str) -> float | None:
    s = s.strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def _inner_equity_weights(symbol: str) -> list[tuple[str, float]]:
    """Return (ticker, weight_pct) from child fund holdings; renormalize."""
    rows = _fetch_holdings_table(symbol)
    raw: list[tuple[str, float]] = []
    for c in rows:
        t = c[1].strip()
        w = _parse_pct(c[3])
        if w is None or w <= 0:
            continue
        if t == "n/a" or not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", t):
            continue
        raw.append((t, w))
    s = sum(w for _, w in raw)
    if s <= 0:
        return []
    return [(t, w / s) for t, w in raw]


def ymax_holdings() -> list[tuple[str, float]]:
    rows = _fetch_holdings_table("YMAX")
    out: list[tuple[str, float]] = []
    for c in rows:
        etf = c[1].strip()
        w = _parse_pct(c[3])
        if w is None or not etf:
            continue
        out.append((etf, w / 100.0))
    return out


def aggregate_exposure() -> dict[str, float]:
    exp: dict[str, float] = defaultdict(float)
    for etf, w_parent in ymax_holdings():
        if etf in SINGLE_STOCK:
            exp[SINGLE_STOCK[etf]] += w_parent
        elif etf in PROXY_FULL_SLEEVE:
            exp[PROXY_FULL_SLEEVE[etf]] += w_parent
        elif etf in PORTFOLIO_FUNDS:
            inner = _inner_equity_weights(etf)
            for t, w_rel in inner:
                exp[t] += w_parent * w_rel
        else:
            # Unknown sleeve: skip rather than guess
            pass
    return dict(exp)


def main() -> None:
    exp = aggregate_exposure()
    ranked = sorted(exp.items(), key=lambda x: -x[1])[:20]
    names = [a for a, _ in ranked]
    vals = [b * 100.0 for _, b in ranked]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(names)), vals[::-1], color="steelblue", edgecolor="navy", linewidth=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Implied sleeve weight (% of YMAX NAV)", fontsize=11)
    ax.set_title(
        "YMAX — top 20 aggregated reference exposures (approx.)\n"
        "Single-stock sleeves at 100% of child weight; portfolio sleeves from "
        "child holdings %; OARK→ARKK, QDTY→QQQ, MSST→MSTR",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.35)
    fig.tight_layout()

    OUT_REPO.mkdir(parents=True, exist_ok=True)
    if OUT_DROPBOX.parent.exists():
        OUT_DROPBOX.mkdir(parents=True, exist_ok=True)

    png = "ymax_top20_underlying_exposure.png"
    csv = "ymax_top20_underlying_exposure.csv"
    fig.savefig(OUT_REPO / png, dpi=160)
    print(f"Wrote {OUT_REPO / png}")

    if OUT_DROPBOX.parent.exists():
        fig.savefig(OUT_DROPBOX / png, dpi=160)
        print(f"Wrote {OUT_DROPBOX / png}")

    with open(OUT_REPO / csv, "w", encoding="utf-8") as f:
        f.write("ticker,implied_weight_pct_of_ymax_nav\n")
        for n, v in ranked:
            f.write(f"{n},{v * 100:.4f}\n")
    print(f"Wrote {OUT_REPO / csv}")
    if OUT_DROPBOX.parent.exists():
        shutil.copy2(OUT_REPO / csv, OUT_DROPBOX / csv)
        print(f"Wrote {OUT_DROPBOX / csv}")

    plt.close(fig)


if __name__ == "__main__":
    main()
