#!/usr/bin/env python3
"""scripts/calibrate_covered_call_prior.py

One-off (re-runnable) calibration of the covered-call ``mu_beta`` constant
that ``beta_estimator.BetaPrior.for_row`` uses for ``covered_call_1x``
(and, by default, ``scraped_income``).

The goal is to anchor the prior in realised history rather than guessing.
The script:

  1. Pulls ~3 years of total-return prices (Yahoo v8 chart API or
     yfinance fallback — same code path as the screener) for the
     standard 1× covered-call sleeve and their underlyings.
  2. Computes a robust EWMA hedge-beta vs the underlying using the same
     :func:`beta_estimator.compute_beta_for_hedging` routine but with
     a deliberately weak prior (``tau = 0``) so the output is data-only.
  3. Reports the median, the inter-quartile range, and writes
     ``data/beta_priors_calibration.json``.

The screener default is hard-coded at ``0.55`` if the calibration falls
within ``[0.45, 0.65]``; otherwise the calibrated value should be used
explicitly (commit the new constant in ``beta_estimator`` with a
reference to this script and the run date).

Run:
    python scripts/calibrate_covered_call_prior.py
    python scripts/calibrate_covered_call_prior.py --period 5y
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beta_estimator import (  # noqa: E402  — path setup above
    BetaPrior,
    compute_beta_for_hedging,
)


# Standard 1× covered-call / option-income ETFs and their underlyings.
# Mirrors ``daily_screener.covered_call_pairs`` to stay in sync.
_PAIRS: list[tuple[str, str]] = [
    ("QYLD", "QQQ"),
    ("QYLG", "QQQ"),
    ("QQQX", "QQQ"),
    ("JEPQ", "QQQ"),
    ("XYLD", "SPY"),
    ("XYLG", "SPY"),
    ("JEPI", "SPY"),
    ("SPYI", "SPY"),
    ("RYLD", "IWM"),
]


def _fetch_yahoo_v8(symbol: str, period: str = "3y") -> pd.Series | None:
    """Fetch adj-close TR series from Yahoo v8 chart API."""
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace('.', '-')}"
    params = {"range": period, "interval": "1d", "events": "div,splits"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("chart") or {}).get("result") or []
        if not result:
            return None
        ts = result[0]["timestamp"]
        adjclose = result[0]["indicators"]["adjclose"][0]["adjclose"]
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize()
        s = pd.Series(adjclose, index=idx, name=symbol, dtype=float).dropna()
        return s if not s.empty else None
    except Exception as exc:
        print(f"[CC-CAL] {symbol}: yahoo v8 failed ({exc!s})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="3y", help="Yahoo period (default 3y)")
    ap.add_argument(
        "--out",
        default=str(ROOT / "data" / "beta_priors_calibration.json"),
        help="Output JSON path",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    for etf, und in _PAIRS:
        s_etf = _fetch_yahoo_v8(etf, args.period)
        s_und = _fetch_yahoo_v8(und, args.period)
        if s_etf is None or s_und is None:
            print(f"[CC-CAL] skip {etf}/{und} (missing price series)")
            continue
        # Data-only: τ=0 → posterior is pure robust EWMA estimate.
        prior = BetaPrior(
            mu=0.0,
            tau=0.0,
            source="calibration_seed",
            product_class="covered_call_1x",
            allow_sign_inflation=False,
        )
        res = compute_beta_for_hedging(s_etf, s_und, prior)
        rows.append(
            {
                "etf": etf,
                "underlying": und,
                "beta": res.beta,
                "beta_se": res.beta_se,
                "n_obs": res.n_obs,
                "horizon": res.horizon,
                "quality": res.quality,
            }
        )
        print(
            f"[CC-CAL] {etf:6s} vs {und:5s}  β={res.beta:+.3f}  "
            f"se={res.beta_se:.3f}  n={res.n_obs}  q={res.quality}"
        )

    if not rows:
        print("[CC-CAL] No usable rows — aborting")
        return 1

    betas = np.array([r["beta"] for r in rows], dtype=float)
    median = float(np.median(betas))
    p25, p75 = (float(x) for x in np.percentile(betas, [25, 75]))

    out = {
        "asof": date.today().isoformat(),
        "period": args.period,
        "n_rows": len(rows),
        "median": median,
        "p25": p25,
        "p75": p75,
        "rows": rows,
        "default_recommended": 0.55 if 0.45 <= median <= 0.65 else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"\n[CC-CAL] median β={median:.3f}  IQR=[{p25:.3f}, {p75:.3f}]  "
        f"→ {out_path}"
    )
    if 0.45 <= median <= 0.65:
        print(
            "[CC-CAL] Within tolerance: keep cc_default_mu = 0.55 in beta_estimator."
        )
    else:
        print(
            "[CC-CAL] Out of tolerance: update cc_default_mu in beta_estimator "
            "and reference this run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
