"""
Synthetic UVIX / SVIX history from Cboe SHORTVOL (+ inferred LONGVOL).

UVIX & SVIX began trading 2022-03-30. This module extends history back to
2005-12-20 (SHORTVOL index inception) by:

  1. Downloading ^SHORTVOL daily closes (yfinance).
  2. Inferring LONGVOL index returns as **-SHORTVOL returns** (calibrated on the
     live overlap: corr ≈ 0.97 between log(UVIX)/2 and -log(SHORTVOL)).
  3. Simulating daily-reset ETP prices:
       SVIX  = -1× SHORTVOL daily move − fees
       UVIX  = +2× LONGVOL daily move − fees
  4. Backward-anchoring simulated prices to **live** UVIX/SVIX on the splice
     date so the series is continuous through 2022-03-30.

Fee drag (annual, from Volatility Shares prospectus / Six Figure Investing):
  UVIX  2.78%  (1.65% mgmt + 1.13% op)
  SVIX  1.98%  (1.35% mgmt + 0.63% op)

Validated at splice: prints max abs % error vs live over the first 5 sessions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.bucket5_data import (
        INCEPTION,
        UVIX,
        SVIX,
        _download,
        _read_cache,
        _to_bday_index,
        _write_cache,
    )
except ImportError:
    from bucket5_data import (  # type: ignore
        INCEPTION,
        UVIX,
        SVIX,
        _download,
        _read_cache,
        _to_bday_index,
        _write_cache,
    )

SYNTH_START = "2005-12-20"
SPLICE = pd.Timestamp(INCEPTION)

UVIX_FEE_ANN = 0.0278
SVIX_FEE_ANN = 0.0198

CACHE_SYNTH = Path("data/cache/bucket5/synthetic")


def _load_shortvol(start: str = SYNTH_START, end: str | None = None) -> pd.Series:
    cached = CACHE_SYNTH / "SHORTVOL_index.csv"
    if cached.is_file():
        s = pd.read_csv(cached, index_col=0, parse_dates=True)["close"]
        s.index = pd.to_datetime(s.index).normalize()
    else:
        s = _download("^SHORTVOL", start, end)
        CACHE_SYNTH.mkdir(parents=True, exist_ok=True)
        s.rename("close").to_frame().to_csv(cached)
    s = _to_bday_index(s)
    if end:
        s = s.loc[: pd.Timestamp(end)]
    return s.loc[s.index >= pd.Timestamp(start)]


def _infer_longvol(shortvol: pd.Series) -> pd.Series:
    """LONGVOL index levels from -SHORTVOL log returns, base=100."""
    lr = np.log(shortvol).diff()
    long_lr = -lr
    lv = 100.0 * np.exp(long_lr.fillna(0.0).cumsum())
    lv.index = shortvol.index
    return lv.rename("longvol")


def _simulate_from_index(
    index: pd.Series,
    leverage: float,
    fee_ann: float,
    anchor_price: float,
    anchor_date: pd.Timestamp,
) -> pd.Series:
    """Daily-reset leveraged ETP, backward-anchored to ``anchor_price`` at ``anchor_date``."""
    idx = index.sort_index()
    r = idx.pct_change().fillna(0.0)
    fee_d = fee_ann / 252.0
    etp_r = leverage * r - fee_d

    # Forward simulate from first date with arbitrary start=1, then scale.
    fwd = (1.0 + etp_r).cumprod()
    if anchor_date not in fwd.index:
        raise ValueError(f"anchor_date {anchor_date.date()} not in index")
    scale = anchor_price / float(fwd.loc[anchor_date])
    return (fwd * scale).rename(index.name or "sim")


def build_synthetic_prices(
    *,
    start: str = SYNTH_START,
    end: str | None = None,
    splice: pd.Timestamp = SPLICE,
) -> pd.DataFrame:
    """UVIX/SVIX closes: synthetic pre-splice, live post-splice."""
    shortvol = _load_shortvol(start, end)
    longvol = _infer_longvol(shortvol)

    live_uvix = _to_bday_index(_download(UVIX, str(splice.date()), end))
    live_svix = _to_bday_index(_download(SVIX, str(splice.date()), end))

    if splice not in shortvol.index:
        raise RuntimeError(f"Splice {splice.date()} missing from SHORTVOL history")
    if splice not in live_uvix.index or splice not in live_svix.index:
        raise RuntimeError(f"Splice {splice.date()} missing from live UVIX/SVIX")

    uvix_sim = _simulate_from_index(
        longvol, leverage=2.0, fee_ann=UVIX_FEE_ANN,
        anchor_price=float(live_uvix.loc[splice]), anchor_date=splice,
    ).rename("uvix")
    svix_sim = _simulate_from_index(
        shortvol, leverage=-1.0, fee_ann=SVIX_FEE_ANN,
        anchor_price=float(live_svix.loc[splice]), anchor_date=splice,
    ).rename("svix")

    pre = pd.concat([uvix_sim, svix_sim], axis=1).loc[:splice]
    post = pd.concat(
        [live_uvix.rename("uvix"), live_svix.rename("svix")], axis=1
    ).loc[splice:]
    out = pd.concat([pre.iloc[:-1], post]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out["synthetic"] = out.index < splice
    return out


def validate_splice(panel: pd.DataFrame, splice: pd.Timestamp = SPLICE, days: int = 5) -> pd.Series:
    """Max abs pct error vs live over first ``days`` sessions at/after splice."""
    live_u = _to_bday_index(_download(UVIX, str(splice.date()), None))
    live_s = _to_bday_index(_download(SVIX, str(splice.date()), None))
    w = panel.loc[panel.index >= splice].head(days)
    err_u = ((w["uvix"] - live_u.reindex(w.index)) / live_u.reindex(w.index)).abs().max()
    err_s = ((w["svix"] - live_s.reindex(w.index)) / live_s.reindex(w.index)).abs().max()
    return pd.Series({"uvix_max_pct_err": err_u, "svix_max_pct_err": err_s})


def load_extended_uvix_svix(
    start: str = SYNTH_START,
    end: str | None = None,
    *,
    use_synthetic: bool = True,
) -> pd.DataFrame:
    """Return uvix/svix with optional synthetic pre-2022 extension."""
    if not use_synthetic or pd.Timestamp(start) >= SPLICE:
        u = _to_bday_index(_download(UVIX, max(start, INCEPTION), end))
        s = _to_bday_index(_download(SVIX, max(start, INCEPTION), end))
        px = pd.concat([u.rename("uvix"), s.rename("svix")], axis=1).dropna()
        px["synthetic"] = False
        return px
    px = build_synthetic_prices(start=start, end=end)
    return px.loc[px.index >= pd.Timestamp(start)]


if __name__ == "__main__":
    px = build_synthetic_prices()
    print(f"synthetic: {px.index.min().date()} -> {px.index.max().date()} ({len(px)} rows)")
    print(f"synthetic days: {px['synthetic'].sum()}  live days: {(~px['synthetic']).sum()}")
    print(px.loc[SPLICE - pd.Timedelta(days=3): SPLICE + pd.Timedelta(days=3)].round(3))
    print("splice validation:", validate_splice(px).round(4).to_dict())
