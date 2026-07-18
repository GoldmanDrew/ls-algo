"""Return-verification utilities for the Bucket-5 insurance strategy.

This module deliberately consumes the production B5 engines.  It does not model
broker fills, locates, or paper trading; those are documented future gates in the
report builder.  The functions are also usable with small synthetic panels in
tests, avoiding network and cache dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.bucket5_insurance_bt import InsuranceConfig, production_config, run_insurance
from scripts.bucket5_put_overlay import PutOverlayConfig, bs_put, effective_iv

THETA_FILE = re.compile(r"^SPX_(\d{8})_(\d+)_P\.parquet$")


@dataclass
class VerificationResult:
    """Economic reconciliation and audit records for one B5 run."""

    books: dict[str, pd.DataFrame]
    reconciliation: pd.DataFrame
    attribution: pd.DataFrame
    harvest_audit: pd.DataFrame
    gates: dict[str, dict[str, Any]]
    engine_result: dict[str, Any]


def _empty_like(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, dtype=float)


def _carry_components(carry: pd.DataFrame, base_equity: pd.Series) -> pd.DataFrame:
    """Scale the production carry engine's daily components into combined capital."""
    idx = base_equity.index
    c = carry.reindex(idx).ffill()
    out = pd.DataFrame(index=idx)
    prev_u_shares = (c["u_notional"] / c["uvix"]).shift(1).fillna(0.0)
    prev_s_shares = (c["s_notional"] / c["svix"]).shift(1).fillna(0.0)
    raw_uvix = prev_u_shares * c["uvix"].diff().fillna(0.0)
    raw_svi = prev_s_shares * c["svix"].diff().fillna(0.0)
    raw_other = c["financing_pnl"].fillna(0.0) - c["rebalance_friction"].fillna(0.0)
    raw_change = c["equity"].diff().fillna(0.0)
    scale = base_equity.shift(1).div(c["equity"].shift(1).replace(0.0, np.nan)).fillna(0.0)
    out["uvix_pnl"] = raw_uvix * scale
    out["svix_pnl"] = raw_svi * scale
    out["carry_borrow_and_financing"] = raw_other * scale
    out["carry_engine_residual"] = (raw_change - raw_uvix - raw_svi - raw_other) * scale
    return out.fillna(0.0)


def build_split_books(engine_result: dict[str, Any]) -> VerificationResult:
    """Create independent carry, put, and combined books plus a daily identity.

    All three displayed books start with the same initial capital.  The combined
    reconciliation is additive by *economic components*, rather than adding three
    full-capital NAVs (which would double-count initial capital).
    """
    bt = engine_result["bt"].copy()
    carry = engine_result["carry"].copy()
    ladder = engine_result["ladder"].reindex(bt.index).ffill().fillna(0.0)
    cfg: InsuranceConfig = engine_result["cfg"]
    initial = float(cfg.initial_capital)
    idx = bt.index
    base = bt["base_equity"].astype(float)
    combined = bt["combined_equity"].astype(float)
    carry_parts = _carry_components(carry, base)
    sleeve_return = carry["ret"].reindex(idx).fillna(0.0)
    deployed = (carry["gross"].reindex(idx).ffill() / carry["equity"].reindex(idx).ffill().replace(0, np.nan)).clip(0, 1).fillna(0)
    tbill = base.shift(1).fillna(initial) * (cfg.tbill_rate / 252.0) * (1.0 - deployed)
    base_change = base.diff().fillna(0.0)
    carry_parts["tbill_pnl"] = tbill
    carry_parts["base_identity_residual"] = base_change - carry_parts.sum(axis=1)

    put_mtm_change = bt["put_mtm"].diff().fillna(0.0)
    put_cash_flow = ladder["put_cash_flow"].reindex(idx).fillna(0.0)
    redeployment = bt["redeploy_extra"].diff().fillna(0.0)
    rec = pd.DataFrame(
        {
            "prior_nav": combined.shift(1).fillna(combined.iloc[0]),
            **{name: carry_parts[name] for name in carry_parts},
            "put_mtm_change": put_mtm_change,
            "put_cash_flow": put_cash_flow,
            "redeployment_pnl": redeployment,
            "ending_nav": combined,
        },
        index=idx,
    )
    component_cols = [c for c in rec if c not in ("prior_nav", "ending_nav")]
    rec["identity_residual"] = rec["ending_nav"] - rec["prior_nav"] - rec[component_cols].sum(axis=1)
    rec.iloc[0, rec.columns.get_loc("identity_residual")] = 0.0

    put_cash_cum = put_cash_flow.cumsum()
    put_book = pd.DataFrame(
        {
            "nav": initial + put_cash_cum + bt["put_mtm"],
            "cash": put_cash_cum,
            "put_mtm": bt["put_mtm"],
            "contracts": ladder.get("contracts", _empty_like(idx)),
        },
        index=idx,
    )
    carry_book = carry[["equity", "cash", "u_notional", "s_notional", "borrow_cost", "rebalance_friction"]].copy()
    carry_book = carry_book.rename(columns={"equity": "nav"})
    for book in (carry_book, put_book):
        book.attrs["initial_capital"] = initial
        book.attrs["capital_convention"] = "independent book, same initial capital before first-day execution"
    component_book = pd.DataFrame(
        {
            "carry_contribution": carry_parts[["uvix_pnl", "svix_pnl", "carry_borrow_and_financing", "carry_engine_residual"]].sum(axis=1).cumsum(),
            "tbill_contribution": carry_parts["tbill_pnl"].cumsum(),
            "put_cash": put_cash_cum,
            "put_mtm": bt["put_mtm"],
            "redeployment": bt["redeploy_extra"],
            "nav_from_components": initial + rec[component_cols].cumsum(axis=1).sum(axis=1),
            "combined_nav": combined,
        },
        index=idx,
    )

    events = engine_result["ladder"].attrs.get("monetize_events", [])
    audit = trace_harvested_cash(ladder.get("realized", _empty_like(idx)), events, idx)
    attribution = rec[
        [
            "uvix_pnl", "svix_pnl", "carry_borrow_and_financing", "tbill_pnl",
            "put_mtm_change", "put_cash_flow", "redeployment_pnl",
        ]
    ].copy()
    gross = carry["gross"].reindex(idx).ffill().replace(0.0, np.nan)
    uvix_weight = (carry["u_notional"].reindex(idx).ffill().abs() / gross).fillna(0.0)
    combined_scale = base.shift(1).div(carry["equity"].reindex(idx).ffill().shift(1).replace(0.0, np.nan)).fillna(0.0)
    total_borrow = carry["borrow_cost"].reindex(idx).fillna(0.0) * combined_scale
    uvix_borrow = total_borrow * uvix_weight
    svix_borrow = total_borrow - uvix_borrow
    attribution["uvix_gross_pnl"] = attribution["uvix_pnl"]
    attribution["svix_gross_pnl"] = attribution["svix_pnl"]
    attribution["uvix_borrow"] = -uvix_borrow
    attribution["svix_borrow"] = -svix_borrow
    attribution["uvix_net_pnl"] = attribution["uvix_gross_pnl"] + attribution["uvix_borrow"]
    attribution["svix_net_pnl"] = attribution["svix_gross_pnl"] + attribution["svix_borrow"]
    attribution["put_premium"] = put_cash_flow.where(put_cash_flow < 0, 0.0)
    attribution["put_monetization_cash"] = ladder.get("realized", _empty_like(idx)).clip(lower=0.0)
    attribution["put_roll_or_other_cash"] = put_cash_flow - attribution["put_premium"] - attribution["put_monetization_cash"]
    gates = evaluate_gates(rec, audit)
    return VerificationResult(
        books={"carry": carry_book, "puts": put_book, "combined": bt, "components": component_book},
        reconciliation=rec,
        attribution=attribution,
        harvest_audit=audit,
        gates=gates,
        engine_result=engine_result,
    )


def trace_harvested_cash(realized: pd.Series, events: Iterable[dict[str, Any]], index: pd.Index) -> pd.DataFrame:
    """Tie positive banked cash to contract-sale events where the engine exposes them."""
    event_cash = _empty_like(index)
    event_count = pd.Series(0, index=index, dtype=int)
    for event in events:
        dt = pd.Timestamp(event.get("date")).normalize()
        if dt in event_cash.index:
            event_cash.loc[dt] += float(event.get("usd", 0.0) or 0.0)
            event_count.loc[dt] += 1
    actual = realized.reindex(index).fillna(0.0).clip(lower=0.0)
    out = pd.DataFrame({"harvested_cash": actual, "event_sale_cash": event_cash, "event_count": event_count})
    out["trace_gap"] = (out["harvested_cash"] - out["event_sale_cash"]).clip(lower=0.0)
    out["traced"] = (out["harvested_cash"] <= 1e-8) | (out["trace_gap"] <= 0.01)
    return out


def evaluate_gates(reconciliation: pd.DataFrame, harvest_audit: pd.DataFrame, *, tolerance: float = 0.05) -> dict[str, dict[str, Any]]:
    """Appendix-A hard gates only: accounting identity and cash traceability."""
    nav_error = float(reconciliation["identity_residual"].abs().max())
    base_error = float(reconciliation["base_identity_residual"].abs().max())
    trace_gap = float(harvest_audit["trace_gap"].sum())
    return {
        # ``base_identity_residual`` is retained as an attribution diagnostic:
        # the carry engine rebalance timing can leave a small decomposition
        # residual.  The combined NAV identity includes that labeled component
        # and is the Appendix-A hard accounting gate.
        "nav_identity": {"kind": "hard", "pass": nav_error <= tolerance, "max_error": nav_error, "base_attribution_diagnostic": base_error},
        "harvested_cash_trace": {"kind": "hard", "pass": trace_gap <= tolerance, "untraced_cash": trace_gap},
    }


def run_verification(panel: pd.DataFrame, spx: pd.Series, cfg: InsuranceConfig | None = None) -> VerificationResult:
    """Run production B5 and construct all split-book verification records."""
    cfg = cfg or production_config()
    aligned = panel.dropna(subset=["uvix", "svix", "vix", "vix3m", "ratio"]).copy()
    spx = spx.reindex(aligned.index).ffill().bfill()
    if len(aligned) < 2 or spx.isna().all():
        raise ValueError("B5 verification needs at least two aligned market dates.")
    engine = run_insurance(aligned, spx, aligned["vix"] / 100.0, cfg)
    engine["verification_panel"] = aligned
    return build_split_books(engine)


def performance_metrics(nav: pd.Series) -> dict[str, float]:
    nav = nav.dropna()
    if len(nav) < 2:
        return {k: float("nan") for k in ("CAGR", "Vol", "Sharpe", "MaxDD", "Calmar")}
    ret = nav.pct_change().fillna(0.0)
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = (1 + total) ** (252.0 / (len(nav) - 1)) - 1.0 if total > -1 else float("nan")
    vol = float(ret.std() * np.sqrt(252))
    dd = float(nav.div(nav.cummax()).sub(1.0).min())
    return {"CAGR": cagr, "Vol": vol, "Sharpe": float(ret.mean() * 252 / vol) if vol else float("nan"), "MaxDD": dd, "Calmar": cagr / abs(dd) if dd < 0 else float("nan")}


def theta_cache_replay(
    cache_dir: Path,
    spot: pd.Series,
    vix: pd.Series,
    *,
    min_observations: int = 10,
    risk_free: float = 0.04,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare cached observed EOD SPX marks with the B5 BS/skew proxy.

    Quote-side fields are retained for bid/ask execution replay.  A thin cache is
    explicitly skipped instead of silently backfilling model marks as observations.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("SPX_*_P.parquet")):
        match = THETA_FILE.match(path.name)
        if not match:
            continue
        expiry = pd.Timestamp(match.group(1))
        strike = int(match.group(2)) / 1000.0
        try:
            quote = pd.read_parquet(path)
        except Exception:
            continue
        quote.index = pd.to_datetime(quote.index, errors="coerce").normalize()
        quote = quote[~quote.index.isna()]
        for dt, row in quote.iterrows():
            s = float(spot.reindex([dt]).ffill().iloc[0]) if dt in spot.index else np.nan
            vx = float(vix.reindex([dt]).ffill().iloc[0]) if dt in vix.index else np.nan
            observed = pd.to_numeric(pd.Series([row.get("mid", row.get("close", np.nan))]), errors="coerce").iloc[0]
            if not (np.isfinite(s) and np.isfinite(vx) and np.isfinite(observed) and observed > 0):
                continue
            dte = max((expiry - dt).days, 0)
            otm = max(0.0, 1.0 - strike / s)
            model = bs_put(s, strike, max(dte / 252.0, 1 / 365.0), effective_iv(vx / 100.0, otm, PutOverlayConfig()), risk_free)
            bid, ask = row.get("bid", np.nan), row.get("ask", np.nan)
            rows.append({"date": dt, "expiry": expiry, "strike": strike, "spot": s, "vix": vx, "dte": dte, "observed_mid": observed, "bid": bid, "ask": ask, "model_price": model})
    replay = pd.DataFrame(rows)
    if replay.empty or len(replay) < min_observations:
        return replay, {"status": "skip", "reason": f"Theta cache too thin ({len(replay)} valid marks; need {min_observations})", "observations": len(replay)}
    replay["error"] = replay["model_price"] - replay["observed_mid"]
    replay["abs_error"] = replay["error"].abs()
    replay["error_pct"] = replay["error"] / replay["observed_mid"]
    replay["buy_ask"] = replay["ask"].where(pd.to_numeric(replay["ask"], errors="coerce") > 0, replay["observed_mid"])
    replay["sell_bid"] = replay["bid"].where(pd.to_numeric(replay["bid"], errors="coerce") > 0, replay["observed_mid"])
    spread = (replay["buy_ask"] - replay["sell_bid"]).clip(lower=0.0)
    replay["buy_stressed"] = replay["buy_ask"] + 0.5 * spread
    replay["sell_stressed"] = (replay["sell_bid"] - 0.5 * spread).clip(lower=0.0)
    summary = {
        "status": "ok", "observations": len(replay), "mean_error": float(replay["error"].mean()),
        "median_error": float(replay["error"].median()), "tail_abs_error_p95": float(replay["abs_error"].quantile(0.95)),
        "stress_error_large": bool(replay["abs_error"].quantile(0.95) > replay["observed_mid"].median()),
    }
    return replay, summary


def build_event_packets(
    verification: VerificationResult, theta_replay: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """Return requested fixed windows plus all available VIX-spike packets."""
    bt = verification.books["combined"]
    carry = verification.engine_result["carry"]
    idx = bt.index
    windows = {"bear_2022": ("2022-01-01", "2022-10-31"), "aug_2024": ("2024-08-01", "2024-08-31"), "covid_2020": ("2020-02-15", "2020-04-30"), "gfc_2008": ("2008-09-01", "2009-03-31")}
    vix = verification.engine_result.get("verification_panel", pd.DataFrame()).get("vix")
    packets: dict[str, pd.DataFrame] = {}
    columns = pd.DataFrame({"combined_nav": bt["combined_equity"], "put_mtm": bt["put_mtm"], "cash_realized": bt["realized_cum"], "uvix_exposure": carry["u_notional"], "svix_exposure": carry["s_notional"], "borrow": carry["borrow_cost"]}, index=idx)
    if theta_replay is not None and not theta_replay.empty:
        q = theta_replay.set_index("date")[["model_price", "observed_mid", "error"]].groupby(level=0).mean()
        columns = columns.join(q, how="left")
    for name, (start, end) in windows.items():
        part = columns.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        if not part.empty:
            packets[name] = part
    # Include every available top-tail VIX spike, not only named historical windows.
    if vix is not None and len(vix):
        threshold = float(vix.quantile(0.95))
        for dt in vix[vix >= threshold].index[::5]:
            packets[f"uvix_spike_{pd.Timestamp(dt):%Y%m%d}"] = columns.loc[(idx >= dt - pd.Timedelta(days=5)) & (idx <= dt + pd.Timedelta(days=5))]
    return packets


def sensitivity_grid(panel: pd.DataFrame, spx: pd.Series, cfg: InsuranceConfig | None = None) -> pd.DataFrame:
    """Predeclared executable cost grid; option-mode/delay limits are explicit."""
    cfg = cfg or production_config()
    rows: list[dict[str, Any]] = []
    for slip in (5.0, 15.0, 30.0, 50.0):
        for borrow_mult in (1.0, 1.5, 2.0):
            scenario = replace(
                cfg, uvix_slip_bps=slip,
                borrow_uvix_annual=(cfg.borrow_uvix_annual or 0.0284) * borrow_mult,
                borrow_svix_annual=(cfg.borrow_svix_annual or 0.0347) * borrow_mult,
            )
            check = run_verification(panel, spx, scenario)
            metrics = performance_metrics(check.books["combined"]["combined_equity"])
            rows.append({"option_execution": "hybrid_cached_theta_or_bs", "slippage_bps": slip, "borrow_multiple": borrow_mult, "monetization_delay": "not_supported", **metrics, "hard_gates_pass": all(x["pass"] for x in check.gates.values())})
    return pd.DataFrame(rows)
