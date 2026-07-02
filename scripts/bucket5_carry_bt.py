"""
Bucket 5 two-leg carry engine: SHORT UVIX vs SHORT SVIX ("positive-carry tail
risk", leg 1 + leg 2 of the product; the SPX put overlay is layered later).

Both legs are short. The "simple ratio" dial (2012 sixfigureinvesting article)
is::

    rho = (SVIX short notional) / (UVIX short notional)

    rho = 0     naked short UVIX         -> max carry, max (unbounded) tail
    rho = 2     first-order vol-neutral  -> min carry, min *linear* tail
    0 < rho < 2 net short vol + positive carry  <- target regime; the residual
                convex tail is what the SPX 6M->3M put overlay is meant to cover

Vol-future-return betas are UVIX = +2, SVIX = -1. With both legs short at dollar
notionals (-U, -S) the net vol-beta notional is::

    2*(-U) + (-1)*(-S) = S - 2U = U*(rho - 2)

so rho < 2 leaves a net-short-vol exposure (the thing that blows up in a spike).

The mark-to-market loop mirrors ``scripts/bucket4_dynamic_bt.py`` (borrow, short-
proceeds credit, fees, slippage, rebalance cadence) but is written explicitly in
UVIX/SVIX/rho terms.

Run directly for a static-rho sweep + the VIX/VIX3M dynamic-rho comparison::

    python scripts/bucket5_carry_bt.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.bucket5_data import (
        DEFAULT_BORROW_SVIX,
        DEFAULT_BORROW_UVIX,
        load_vol_panel,
        rebalance_dates,
    )
except ImportError:  # when run as a script from repo root
    from bucket5_data import (  # type: ignore
        DEFAULT_BORROW_SVIX,
        DEFAULT_BORROW_UVIX,
        load_vol_panel,
        rebalance_dates,
    )

# Vol-future-return betas of the two ETPs.
BETA_UVIX = 2.0
BETA_SVIX = -1.0


# ---------------------------------------------------------------------------
# rho policies
# ---------------------------------------------------------------------------
def static_rho(index: pd.DatetimeIndex, rho: float) -> pd.Series:
    return pd.Series(float(rho), index=index, name="rho")


def rho_from_ratio(
    ratio: pd.Series,
    *,
    rho_min: float = 0.5,
    rho_max: float = 2.0,
    r_lo: float = 0.85,
    r_hi: float = 0.95,
) -> pd.Series:
    """Dynamic rho from the VIX/VIX3M term-structure ratio.

    Steep contango (low ratio) -> rho_min (lean into carry). As the ratio climbs
    toward backwardation (the 2012 article's ~0.917-0.95 danger band) rho rises
    to rho_max (more short-SVIX hedge, less net short vol). Linear between
    ``r_lo`` and ``r_hi``, clipped outside.
    """
    frac = ((ratio - r_lo) / (r_hi - r_lo)).clip(0.0, 1.0)
    return (rho_min + (rho_max - rho_min) * frac).rename("rho")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_carry_backtest(
    panel: pd.DataFrame,
    rho_daily: pd.Series,
    rebal_dates: pd.DatetimeIndex,
    *,
    initial_capital: float = 1_000_000.0,
    gross_multiplier: float = 1.0,
    gross_daily: pd.Series | None = None,
    borrow_uvix_annual: float = DEFAULT_BORROW_UVIX,
    borrow_svix_annual: float = DEFAULT_BORROW_SVIX,
    short_proceeds_annual: float = 0.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 20.0,
) -> pd.DataFrame:
    """Mark-to-market the short-UVIX / short-SVIX book.

    ``panel`` needs columns ``uvix`` and ``svix`` (see ``load_vol_panel``).
    ``rho_daily`` is the target SVIX/UVIX notional ratio (sampled on rebalance
    days). Gross book = ``gross_mult(t) * equity`` where ``gross_mult(t)`` is
    ``gross_daily`` when supplied (regime de-risking) else the scalar
    ``gross_multiplier``.
    """
    px = panel[["uvix", "svix"]].dropna().copy()
    rho_aligned = rho_daily.reindex(px.index).ffill().bfill()
    if gross_daily is not None:
        gross_aligned = gross_daily.reindex(px.index).ffill().bfill()
    else:
        gross_aligned = pd.Series(float(gross_multiplier), index=px.index)
    is_rebal = px.index.isin(rebal_dates)
    is_rebal[0] = True  # always establish on day 1

    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0
    borrow_u_d = borrow_uvix_annual / 252.0
    borrow_s_d = borrow_svix_annual / 252.0
    proceeds_d = short_proceeds_annual / 252.0

    u_sh = 0.0
    s_sh = 0.0
    cash = float(initial_capital)
    rows: list[dict] = []

    for i, (dt, row) in enumerate(px.iterrows()):
        up = float(row["uvix"])
        sp = float(row["svix"])
        u_notional = u_sh * up
        s_notional = s_sh * sp

        # Financing: borrow on both short legs, optional credit on proceeds.
        borrow_cost = abs(u_notional) * borrow_u_d + abs(s_notional) * borrow_s_d
        proceeds_credit = (abs(u_notional) + abs(s_notional)) * proceeds_d
        financing = proceeds_credit - borrow_cost
        cash += financing
        equity = cash + u_notional + s_notional

        rebal_fee = slip = 0.0
        rho = float(rho_aligned.loc[dt])
        gmult = float(gross_aligned.loc[dt])
        if is_rebal[i]:
            target_gross = max(0.0, gmult * equity)
            u_target_notional = target_gross / (1.0 + rho)
            s_target_notional = target_gross - u_target_notional
            tgt_u_pos = -u_target_notional  # short
            tgt_s_pos = -s_target_notional  # short
            d_u = tgt_u_pos - u_notional
            d_s = tgt_s_pos - s_notional
            traded = abs(d_u) + abs(d_s)
            rebal_fee = traded * fee_rate
            slip = traded * slip_rate
            cash -= d_u + d_s + rebal_fee + slip
            u_sh = tgt_u_pos / up if up > 0 else 0.0
            s_sh = tgt_s_pos / sp if sp > 0 else 0.0
            u_notional, s_notional = u_sh * up, s_sh * sp
            equity = cash + u_notional + s_notional

        beta_notional = BETA_UVIX * u_notional + BETA_SVIX * s_notional
        rows.append(
            {
                "date": dt,
                "uvix": up,
                "svix": sp,
                "rho": rho,
                "gross_mult": gmult,
                "cash": cash,
                "u_notional": u_notional,
                "s_notional": s_notional,
                "gross": abs(u_notional) + abs(s_notional),
                "equity": equity,
                "beta_notional": beta_notional,
                "borrow_cost": borrow_cost,
                "financing_pnl": financing,
                "rebalance": bool(is_rebal[i]),
                "rebalance_friction": rebal_fee + slip,
            }
        )

    bt = pd.DataFrame(rows).set_index("date")
    bt["ret"] = bt["equity"].pct_change().fillna(0.0)
    bt["drawdown"] = bt["equity"].div(bt["equity"].cummax()).sub(1.0)
    bt["net_vol_beta_frac"] = np.where(
        bt["equity"].abs() > 1e-9, bt["beta_notional"] / bt["equity"], np.nan
    )
    return bt


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _worst_rolling(ret: pd.Series, window: int) -> float:
    if len(ret) < window:
        return float("nan")
    logg = np.log1p(ret.clip(lower=-0.999))
    roll = logg.rolling(window).sum().dropna()
    return float(np.expm1(roll.min())) if len(roll) else float("nan")


def perf_stats(bt: pd.DataFrame) -> pd.Series:
    n = len(bt)
    if n < 2:
        return pd.Series(dtype=float)
    ret = bt["ret"]
    total = bt["equity"].iloc[-1] / bt["equity"].iloc[0] - 1.0
    ann = 252 / (n - 1)
    cagr = (1 + total) ** ann - 1 if total > -1 else np.nan
    vol = ret.std() * np.sqrt(252)
    downside = ret[ret < 0].std() * np.sqrt(252)
    years = (n - 1) / 252.0
    avg_eq = bt["equity"].mean()
    return pd.Series(
        {
            "CAGR": cagr,
            "Total Return": total,
            "Ann Vol": vol,
            "Sharpe": (ret.mean() * 252 / vol) if vol > 0 else np.nan,
            "Sortino": (ret.mean() * 252 / downside) if downside > 0 else np.nan,
            "Max Drawdown": bt["drawdown"].min(),
            "Worst 1d": ret.min(),
            "Worst 5d": _worst_rolling(ret, 5),
            "Worst 21d": _worst_rolling(ret, 21),
            "Pct Up Days": float((ret > 0).mean()),
            "Calmar": (cagr / abs(bt["drawdown"].min())) if bt["drawdown"].min() < 0 else np.nan,
            "Ann Carry %eq": (bt["financing_pnl"].sum() / years / avg_eq) if years > 0 else np.nan,
            "Total Borrow $": bt["borrow_cost"].sum(),
            "Total Friction $": bt["rebalance_friction"].sum(),
            "Avg Net Vol Beta": bt["net_vol_beta_frac"].mean(),
            "Rebalances": int(bt["rebalance"].sum()),
        }
    )


def spike_pnl_fraction(rho: float, g: float, *, gross_multiplier: float = 1.0) -> float:
    """Instantaneous P&L (fraction of equity) for a 1-day vol-future shock ``g``.

    UVIX return = max(2g, -1), SVIX return = max(-g, -1) (neither price can go
    below zero). Useful for sizing the put overlay to the *jump*, not the
    diffusion. e.g. ``spike_pnl_fraction(1.0, 0.8)`` ~ a +80% one-day VIX-future
    move (Volmageddon-ish).
    """
    r_u = max(2.0 * g, -1.0)
    r_s = max(-g, -1.0)
    u_frac = 1.0 / (1.0 + rho)
    s_frac = rho / (1.0 + rho)
    # short legs: pnl = -(U*r_u + S*r_s); scaled by gross/equity
    return -gross_multiplier * (u_frac * r_u + s_frac * r_s)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    rebal_freq: str = "W-FRI"
    static_rhos: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    initial_capital: float = 1_000_000.0
    gross_multiplier: float = 1.0


def run_all(cfg: RunConfig | None = None, *, refresh: bool = False) -> dict:
    cfg = cfg or RunConfig()
    panel = load_vol_panel(refresh=refresh)
    rebal = rebalance_dates(panel.index, cfg.rebal_freq)

    results: dict[str, pd.DataFrame] = {}
    for rho in cfg.static_rhos:
        bt = run_carry_backtest(
            panel,
            static_rho(panel.index, rho),
            rebal,
            initial_capital=cfg.initial_capital,
            gross_multiplier=cfg.gross_multiplier,
        )
        results[f"static_rho={rho:.2f}"] = bt

    dyn = run_carry_backtest(
        panel,
        rho_from_ratio(panel["ratio"]),
        rebal,
        initial_capital=cfg.initial_capital,
        gross_multiplier=cfg.gross_multiplier,
    )
    results["dynamic_ratio"] = dyn

    summary = pd.DataFrame({name: perf_stats(bt) for name, bt in results.items()}).T
    return {"panel": panel, "results": results, "summary": summary}


def _save_outputs(out: dict) -> Path:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    dest = Path("data/runs") / today / "bucket5"
    dest.mkdir(parents=True, exist_ok=True)
    out["summary"].to_csv(dest / "bucket5_carry_summary.csv")

    # Spike profile across rho (tail sensitivity to a Volmageddon-size shock).
    shocks = [0.1, 0.25, 0.5, 0.8, 1.2]
    spike = pd.DataFrame(
        {f"g={g:+.2f}": {f"rho={r:.1f}": spike_pnl_fraction(r, g)
                          for r in (0.0, 0.5, 1.0, 1.5, 2.0)}
         for g in shocks}
    )
    spike.to_csv(dest / "bucket5_spike_profile.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 6))
        for name, bt in out["results"].items():
            ax.plot(bt.index, bt["equity"], label=name, lw=1.4)
        ax.set_title("Bucket 5 — short UVIX / short SVIX carry (equity)")
        ax.set_ylabel("Equity ($)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(dest / "bucket5_equity_curves.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] plot skipped: {e}")
    return dest


if __name__ == "__main__":
    out = run_all()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Bucket 5 carry backtest (live UVIX/SVIX) ===")
    print(f"sample: {out['panel'].index.min().date()} -> {out['panel'].index.max().date()}")
    print(out["summary"].round(3).to_string())
    dest = _save_outputs(out)
    print(f"\nsaved -> {dest}")
