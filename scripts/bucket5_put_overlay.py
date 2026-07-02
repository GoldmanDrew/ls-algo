"""
SPX put-roll overlay for Bucket 5 (6M buy -> roll at 3M remaining).

Uses ThetaData EOD when ThetaTerminal is reachable; otherwise prices puts with
Black-Scholes using ^GSPC spot and VIX/100 as implied vol (skew bump for OTM).

Combined with the short-UVIX / short-SVIX carry book in ``bucket5_full_bt.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

try:
    from scripts.bucket5_data import ThetaSpxClient, load_series, _to_bday_index
except ImportError:
    from bucket5_data import ThetaSpxClient, load_series, _to_bday_index  # type: ignore


SPX = "^GSPC"
PricingMode = Literal["bs", "theta"]


@dataclass
class PutOverlayConfig:
    otm_pct: float = 0.10           # strike = spot * (1 - otm_pct)
    buy_dte: int = 126              # ~6M trading days at entry
    roll_dte: int = 63              # roll when this many DTE remain (~3M left)
    premium_frac_equity: float = 0.005   # hard cap: 0.5% equity per roll
    carry_budget_frac: float = 0.35      # spend up to 35% of trailing book gain
    carry_lookback: int = 63             # trailing days for gain budget
    contract_multiplier: int = 100
    risk_free: float = 0.04
    # --- SPX volatility-skew model (placeholder until ThetaData is wired) ------
    # Effective IV(strike) = atm_iv * atm_iv_scale + skew_vol_per_10pct *
    # (otm_pct / 0.10), so deep-OTM puts are NOT priced flat (which would make
    # them look almost free). Default ~ +3 vol pts per 10% OTM at 3-6M tenor,
    # which is in the right ballpark for SPX put skew. Tune / replace with the
    # ThetaData implied surface when ThetaTerminal is running.
    skew_vol_per_10pct: float = 0.03
    atm_iv_scale: float = 1.0
    min_days_between_rolls: int = 55     # prevent end-of-sample roll spam


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def effective_iv(atm_iv: float, otm_pct: float, cfg: "PutOverlayConfig") -> float:
    """Skew-adjusted IV for a put ``otm_pct`` below spot."""
    return max(0.05, atm_iv * cfg.atm_iv_scale + cfg.skew_vol_per_10pct * (otm_pct / 0.10))


def bs_put(spot: float, strike: float, t_years: float, vol: float, r: float) -> float:
    """European put premium per share (not x100)."""
    if spot <= 0 or strike <= 0 or t_years <= 1 / 365:
        return max(strike - spot, 0.0)
    vol = max(vol, 0.05)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _trading_dte(expiry: pd.Timestamp | None, dt: pd.Timestamp) -> int:
    if expiry is None or pd.isna(expiry) or expiry <= dt:
        return 0
    return max(0, int(np.busday_count(dt.date(), expiry.date())))


def _pick_expiry(index: pd.DatetimeIndex, dt: pd.Timestamp, target_dte: int, min_dte: int) -> pd.Timestamp | None:
    """Index date ``target_dte`` trading rows after ``dt``, or None if too near end."""
    future = index[index > dt]
    if len(future) <= min_dte + 5:
        return None
    pos = min(len(future) - 1, target_dte)
    exp = future[pos]
    if _trading_dte(exp, dt) < min_dte:
        return None
    return exp


def load_spx_spot(start: str, end: str | None = None) -> pd.Series:
    return _to_bday_index(load_series(SPX, start, end)).rename("spx")


def run_put_overlay(
    dates: pd.DatetimeIndex,
    equity: pd.Series,
    spot: pd.Series,
    iv: pd.Series,
    cfg: PutOverlayConfig | None = None,
    *,
    carry_pnl: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate long SPX puts rolled 6M -> 3M on ``dates`` aligned to ``equity``.

    Returns daily DataFrame with put MTM, premium flows, and combined equity.
    """
    cfg = cfg or PutOverlayConfig()
    idx = dates.intersection(equity.index).intersection(spot.index).intersection(iv.index)
    if len(idx) < 2:
        raise RuntimeError("Insufficient overlap for put overlay.")

    eq = equity.reindex(idx).ffill()
    spx = spot.reindex(idx).ffill()
    vol = iv.reindex(idx).ffill().clip(lower=0.08, upper=1.5)
    carry = (
        carry_pnl.reindex(idx).fillna(0.0)
        if carry_pnl is not None
        else pd.Series(0.0, index=idx)
    )

    # Put state
    strike = np.nan
    expiry: pd.Timestamp | None = None
    theta_exp: date | None = None
    contracts = 0
    cash_put = 0.0  # cumulative premium spent (negative) + sale proceeds

    rows: list[dict] = []
    roll_count = 0
    days_since_roll = 10_000

    for i, dt in enumerate(idx):
        s = float(spx.loc[dt])
        atm = float(vol.loc[dt])  # ATM IV proxy (VIX/100); skew applied per-strike
        eq_i = float(eq.loc[dt])
        eq_start = float(eq.iloc[max(0, i - cfg.carry_lookback)])

        dte = _trading_dte(expiry, dt) if expiry is not None else 0
        min_gap = cfg.min_days_between_rolls
        need_roll = (
            (i == 0 and contracts == 0)
            or (
                contracts > 0
                and 0 < dte <= cfg.roll_dte
                and days_since_roll >= min_gap
            )
        )

        premium_spent = 0.0
        premium_received = 0.0

        if need_roll and s > 0 and eq_i > 0:
            if contracts > 0 and expiry is not None:
                t_rem = max(dte / 252.0, 1 / 365)
                iv_sell = effective_iv(atm, max(0.0, 1.0 - strike / s), cfg)
                px = bs_put(s, strike, t_rem, iv_sell, cfg.risk_free)
                premium_received = px * cfg.contract_multiplier * contracts
                cash_put += premium_received
                contracts = 0
                expiry = None
                theta_exp = None

            # Budget: equity cap always; further cap by trailing book gain when positive.
            trail_gain = max(0.0, eq_i - eq_start)
            budget = cfg.premium_frac_equity * eq_i
            if trail_gain > 0:
                budget = min(budget, cfg.carry_budget_frac * trail_gain)

            new_expiry = _pick_expiry(idx, dt, cfg.buy_dte, cfg.roll_dte + 10)
            if new_expiry is not None and budget > 100:
                strike = s * (1.0 - cfg.otm_pct)
                t_buy = max(_trading_dte(new_expiry, dt) / 252.0, 1 / 252)
                px_buy = None
                try:
                    from scripts.bucket5_theta import put_mid_on_date
                except ImportError:
                    from bucket5_theta import put_mid_on_date  # type: ignore
                theta_hit = put_mid_on_date(
                    pd.Timestamp(dt), s, cfg.otm_pct, _trading_dte(new_expiry, dt)
                )
                if theta_hit is not None:
                    px_buy, meta = theta_hit
                    strike = float(meta.get("strike", strike))
                    theta_exp = meta.get("exp")
                else:
                    theta_exp = None
                if px_buy is None:
                    iv_buy = effective_iv(atm, cfg.otm_pct, cfg)
                    px_buy = bs_put(s, strike, t_buy, iv_buy, cfg.risk_free)
                if px_buy > 1e-6:
                    contracts = max(1, int(budget / (px_buy * cfg.contract_multiplier)))
                    premium_spent = px_buy * cfg.contract_multiplier * contracts
                    cash_put -= premium_spent
                    expiry = new_expiry
                    roll_count += 1
                    days_since_roll = 0
        days_since_roll += 1

        # Mark-to-market
        put_mtm = 0.0
        iv_mtm = effective_iv(atm, cfg.otm_pct, cfg)
        if contracts > 0 and expiry is not None and not math.isnan(strike):
            t_rem = max(_trading_dte(expiry, dt) / 252.0, 1 / 365)
            iv_mtm = effective_iv(atm, max(0.0, 1.0 - strike / s), cfg)
            px_mtm = None
            if theta_exp is not None:
                try:
                    from scripts.bucket5_theta import put_mtm_on_date
                except ImportError:
                    from bucket5_theta import put_mtm_on_date  # type: ignore
                px_mtm = put_mtm_on_date(pd.Timestamp(dt), strike, theta_exp)
            if px_mtm is None:
                px_mtm = bs_put(s, strike, t_rem, iv_mtm, cfg.risk_free)
            put_mtm = px_mtm * cfg.contract_multiplier * contracts

        rows.append(
            {
                "date": dt,
                "spx": s,
                "iv": iv_mtm,
                "equity_carry": eq_i,
                "put_strike": strike,
                "put_expiry": expiry,
                "put_contracts": contracts,
                "put_dte": dte if expiry is not None else np.nan,
                "put_mtm": put_mtm,
                "premium_spent": premium_spent,
                "premium_received": premium_received,
                "put_cash_flow": premium_received - premium_spent,
                "combined_equity": eq_i + put_mtm + cash_put,
            }
        )

    out = pd.DataFrame(rows).set_index("date")
    out["combined_ret"] = out["combined_equity"].pct_change().fillna(0.0)
    out["carry_ret"] = out["equity_carry"].pct_change().fillna(0.0)
    out["put_pnl"] = out["put_mtm"].diff().fillna(0.0) + out["put_cash_flow"]
    out["drawdown_combined"] = out["combined_equity"].div(out["combined_equity"].cummax()).sub(1.0)
    out.attrs["roll_count"] = roll_count
    return out


# ===========================================================================
# 1x3 put ratio backspread overlay (sell 1 nearer put, buy 3 deeper puts)
# ===========================================================================
@dataclass
class BackspreadConfig:
    otm_near: float = 0.12          # short leg strike = spot*(1-otm_near)
    otm_far: float = 0.30           # long leg strike = spot*(1-otm_far)
    far_ratio: int = 3              # buy this many far puts per 1 near sold (1x3)
    buy_dte: int = 126
    roll_dte: int = 63
    long_premium_frac_equity: float = 0.005   # budget = spend on the long (far) leg
    contract_multiplier: int = 100
    risk_free: float = 0.04
    skew_vol_per_10pct: float = 0.03
    atm_iv_scale: float = 1.0
    min_days_between_rolls: int = 55

    def _put_cfg(self) -> PutOverlayConfig:
        return PutOverlayConfig(
            skew_vol_per_10pct=self.skew_vol_per_10pct,
            atm_iv_scale=self.atm_iv_scale,
            risk_free=self.risk_free,
        )


def run_backspread_overlay(
    dates: pd.DatetimeIndex,
    equity: pd.Series,
    spot: pd.Series,
    iv: pd.Series,
    cfg: BackspreadConfig | None = None,
) -> pd.DataFrame:
    """Long SPX put 1x``far_ratio`` backspread, rolled 6M->3M.

    Per structure: SHORT 1 put at ``otm_near``, LONG ``far_ratio`` puts at
    ``otm_far``. Sized so the long leg premium ~= ``long_premium_frac_equity`` *
    equity each roll (the short leg partially or fully finances it -> net cost is
    often a small debit or a credit). Returns daily MTM + combined equity.
    """
    cfg = cfg or BackspreadConfig()
    pcfg = cfg._put_cfg()
    idx = dates.intersection(equity.index).intersection(spot.index).intersection(iv.index)
    if len(idx) < 2:
        raise RuntimeError("Insufficient overlap for backspread overlay.")
    eq = equity.reindex(idx).ffill()
    spx = spot.reindex(idx).ffill()
    vol = iv.reindex(idx).ffill().clip(lower=0.08, upper=1.5)

    k_near = k_far = np.nan
    expiry: pd.Timestamp | None = None
    structs = 0
    cash = 0.0
    rows: list[dict] = []
    roll_count = 0
    days_since_roll = 10_000
    mult = cfg.contract_multiplier

    def _val(s, atm, knear, kfar, t):
        money_n = max(0.0, 1.0 - knear / s)
        money_f = max(0.0, 1.0 - kfar / s)
        pn = bs_put(s, knear, t, effective_iv(atm, money_n, pcfg), cfg.risk_free)
        pf = bs_put(s, kfar, t, effective_iv(atm, money_f, pcfg), cfg.risk_free)
        return cfg.far_ratio * pf - pn  # net structure value (long bias)

    for i, dt in enumerate(idx):
        s = float(spx.loc[dt])
        atm = float(vol.loc[dt])
        eq_i = float(eq.loc[dt])
        dte = _trading_dte(expiry, dt) if expiry is not None else 0
        need_roll = (i == 0 and structs == 0) or (
            structs > 0 and 0 < dte <= cfg.roll_dte and days_since_roll >= cfg.min_days_between_rolls
        )
        net_flow = 0.0
        if need_roll and s > 0 and eq_i > 0:
            if structs > 0 and expiry is not None:
                t_rem = max(dte / 252.0, 1 / 365)
                proceeds = _val(s, atm, k_near, k_far, t_rem) * mult * structs
                cash += proceeds
                net_flow += proceeds
                structs = 0
                expiry = None
            new_expiry = _pick_expiry(idx, dt, cfg.buy_dte, cfg.roll_dte + 10)
            if new_expiry is not None:
                k_near = s * (1.0 - cfg.otm_near)
                k_far = s * (1.0 - cfg.otm_far)
                t_buy = max(_trading_dte(new_expiry, dt) / 252.0, 1 / 252)
                far_px = bs_put(s, k_far, t_buy, effective_iv(atm, cfg.otm_far, pcfg), cfg.risk_free)
                long_budget = cfg.long_premium_frac_equity * eq_i
                if far_px > 1e-6:
                    structs = max(1, int(long_budget / (cfg.far_ratio * far_px * mult)))
                    cost = _val(s, atm, k_near, k_far, t_buy) * mult * structs
                    cash -= cost
                    net_flow -= cost
                    expiry = new_expiry
                    roll_count += 1
                    days_since_roll = 0
        days_since_roll += 1

        mtm = 0.0
        if structs > 0 and expiry is not None and not math.isnan(k_far):
            t_rem = max(_trading_dte(expiry, dt) / 252.0, 1 / 365)
            mtm = _val(s, atm, k_near, k_far, t_rem) * mult * structs
        rows.append(
            {
                "date": dt,
                "spx": s,
                "equity_carry": eq_i,
                "k_near": k_near,
                "k_far": k_far,
                "structures": structs,
                "bs_mtm": mtm,
                "bs_cash_flow": net_flow,
                "combined_equity": eq_i + mtm + cash,
            }
        )
    out = pd.DataFrame(rows).set_index("date")
    out["combined_ret"] = out["combined_equity"].pct_change().fillna(0.0)
    out["drawdown_combined"] = out["combined_equity"].div(out["combined_equity"].cummax()).sub(1.0)
    out.attrs["roll_count"] = roll_count
    return out


def hedge_crash_value(
    *,
    spot: float,
    atm_iv: float,
    dte: int,
    spx_drop: float,
    vix_mult: float,
    days: int,
    kind: str,
    otm_far: float,
    otm_near: float = 0.12,
    far_ratio: int = 3,
    long_premium_frac_equity: float = 0.005,
    skew_vol_per_10pct: float = 0.03,
    risk_free: float = 0.04,
    contract_multiplier: int = 100,
) -> float:
    """Crash P&L of a hedge as a fraction of equity, sized by long-leg budget.

    ``kind`` in {"deep_put", "backspread"}. Both spend ``long_premium_frac_equity``
    on the long (far) leg so they're comparable; the backspread additionally
    shorts 1 nearer put per ``far_ratio`` longs (a credit that buys more longs
    of the SAME budget is captured by sizing on the long leg only).
    """
    pcfg = PutOverlayConfig(skew_vol_per_10pct=skew_vol_per_10pct, risk_free=risk_free)
    t0 = max(dte / 252.0, 1 / 365)
    t1 = max((dte - days) / 252.0, 1 / 365)
    s1 = spot * (1.0 + spx_drop)
    atm1 = atm_iv * vix_mult
    k_far = spot * (1.0 - otm_far)
    far0 = bs_put(spot, k_far, t0, effective_iv(atm_iv, otm_far, pcfg), risk_free)
    far1 = bs_put(s1, k_far, t1, effective_iv(atm1, max(0.0, 1 - k_far / s1), pcfg), risk_free)
    # budget buys the long leg; express per $1 equity
    long_units = long_premium_frac_equity / max(far0 * contract_multiplier, 1e-9)
    if kind == "deep_put":
        return long_units * contract_multiplier * (far1 - far0)
    # backspread: long far_ratio far + short 1 near, per structure
    k_near = spot * (1.0 - otm_near)
    near0 = bs_put(spot, k_near, t0, effective_iv(atm_iv, otm_near, pcfg), risk_free)
    near1 = bs_put(s1, k_near, t1, effective_iv(atm1, max(0.0, 1 - k_near / s1), pcfg), risk_free)
    structs = long_premium_frac_equity / max(far_ratio * far0 * contract_multiplier, 1e-9)
    val0 = far_ratio * far0 - near0
    val1 = far_ratio * far1 - near1
    return structs * contract_multiplier * (val1 - val0)
