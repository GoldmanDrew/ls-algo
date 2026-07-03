"""Dashboard helpers for Bucket 5 backtest panel (assumptions, live book)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


def build_assumption_sections(cfg: dict, meta: dict, assumptions: dict) -> list[dict[str, Any]]:
    """Structured assumption blocks for dashboard <details> rendering."""
    mon = cfg.get("monetize") or {}
    reg = cfg.get("regime") or {}
    redeploy = cfg.get("redeploy") or {}
    hb = cfg.get("hedge_budget") or {}
    rungs = cfg.get("rungs") or []
    bs = cfg.get("backspread") or {}

    def _tiers(name: str) -> str:
        tiers = mon.get(name) or []
        return ", ".join(f"{t[0]}→{t[1]:.0%}" for t in tiers if isinstance(t, (list, tuple)) and len(t) >= 2) or "—"

    rung_txt = "; ".join(
        f"{100 * float(r.get('otm_pct', 0)):.0f}% OTM @ {100 * float(r.get('per_roll_frac', 0)):.2f}%/roll"
        for r in rungs
    ) or "—"

    return [
        {
            "title": "Carry sleeve (UVIX / SVIX)",
            "rows": [
                ("Sleeve gross fraction", f"{100 * float(cfg.get('sleeve_frac', 0)):.1f}%"),
                ("UVIX borrow (annual)", f"{100 * float(assumptions.get('borrow_uvix_annual', 0)):.2f}%"),
                ("SVIX borrow (annual)", f"{100 * float(assumptions.get('borrow_svix_annual', 0)):.2f}%"),
                ("ETP slippage", f"{cfg.get('uvix_slip_bps', 5)} bps"),
                ("Commission", f"{cfg.get('fee_bps', 1)} bp"),
            ],
        },
        {
            "title": "Collateral & cadence",
            "rows": [
                ("T-bill yield on idle cash", f"{100 * float(cfg.get('tbill_rate', 0)):.2f}%"),
                ("Base rebalance days", str(cfg.get("base_days", 14))),
                ("Cadence stress k", str(cfg.get("cadence_k", 6))),
            ],
        },
        {
            "title": "Regime policy (VIX/VIX3M ratio)",
            "rows": [
                ("Rho contango / backwardation", f"{reg.get('rho_contango', 1)} / {reg.get('rho_backwardation', 2)}"),
                ("Gross contango / backwardation", f"{reg.get('gross_contango', 1)} / {reg.get('gross_backwardation', 0.35)}"),
            ],
        },
        {
            "title": "SPX put hedge",
            "rows": [
                ("Hedge kind", str(cfg.get("hedge_kind", "ladder"))),
                ("Ladder rungs", rung_txt),
                ("Dynamic budget", "on" if hb.get("enabled") else "off"),
                ("Contango / stress premium mult", f"{hb.get('contango_mult', 1.2)} / {hb.get('stress_mult', 0.85)}" if hb else "—"),
                ("Hybrid ladder share", f"{100 * float(cfg.get('hybrid_ladder_frac', 0.7)):.0f}%" if cfg.get("hedge_kind") == "hybrid" else "—"),
            ],
        },
        {
            "title": "Monetize policy",
            "rows": [
                ("Profit tiers (mult → sell)", _tiers("profit_tiers")),
                ("VIX tiers", _tiers("vix_tiers")),
                ("Giveback fraction / min mult", f"{mon.get('giveback_frac', 0.35)} / {mon.get('giveback_min_mult', 2)}"),
                ("Runner fraction / harvest mult", f"{mon.get('runner_frac', 0.15)} / {mon.get('runner_mult', 12)}"),
                ("Bank on full exit", f"{mon.get('bank_frac', 0.6):.0%}"),
                ("Re-arm fresh puts", "yes" if mon.get("rearm", True) else "no"),
            ],
        },
        {
            "title": "Redeploy harvested cash",
            "rows": [
                ("Sleeve wt contango / backwardation", f"{redeploy.get('sleeve_w_contango', 0.2)} / {redeploy.get('sleeve_w_backwardation', 0.65)}"),
            ],
        },
        {
            "title": "Data & pricing",
            "rows": [
                ("Era", meta.get("era", "—")),
                ("Date range", f"{meta.get('start', '?')} → {meta.get('end', '?')}"),
                ("Pricing mode", meta.get("pricing_mode", "—")),
                ("Synthetic history days", str(meta.get("synthetic_days", 0))),
                ("Rebalances in sample", str(meta.get("rebalances", "—"))),
            ],
        },
    ]


def load_live_b5_book(run_date: str | None = None) -> dict[str, Any]:
    """Current B5 proposed book vs model assumptions."""
    run_date = run_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    for path in (
        REPO / "data" / "runs" / run_date / "proposed_trades.csv",
        REPO / "data" / "proposed_trades.csv",
    ):
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "sleeve" not in df.columns:
            continue
        b5 = df[df["sleeve"].astype(str).eq("volatility_etp_bucket5")].copy()
        if b5.empty:
            return {"run_date": run_date, "rows": []}
        rows = []
        for _, r in b5.iterrows():
            rows.append({
                "etf": str(r.get("ETF", "")),
                "underlying": str(r.get("Underlying", "")),
                "proposed_gross_usd": float(pd.to_numeric(r.get("gross_target_usd"), errors="coerce") or 0),
                "optimal_gross_usd": float(pd.to_numeric(r.get("optimal_gross_target_usd"), errors="coerce") or 0),
                "borrow_annual": float(pd.to_numeric(r.get("borrow_current"), errors="coerce") or 0),
                "shares_available": float(pd.to_numeric(r.get("shares_available"), errors="coerce") or 0),
                "locate_ok": float(pd.to_numeric(r.get("shares_available"), errors="coerce") or 0) > 0,
            })
        return {"run_date": run_date, "rows": rows}
    return {"run_date": run_date, "rows": []}


def regime_from_series(ratio_series: list[list[Any]] | None) -> dict[str, Any]:
    if not ratio_series:
        return {}
    last = ratio_series[-1]
    ratio = float(last[1]) if len(last) >= 2 else None
    if ratio is None:
        return {}
    label = "backwardation (stress)" if ratio < 0.88 else "contango (calm)" if ratio > 1.0 else "neutral"
    return {"date": last[0], "ratio": ratio, "label": label}
