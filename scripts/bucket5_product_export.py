"""
Bucket 5 product-dashboard exporters (SPX-0DTE-aligned schema helpers).

Produces strategy_guide, daily[], marks_by_date, events_by_date, regime_panels
from a ``run_insurance`` result dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts.bucket5_dashboard_helpers import build_assumption_sections, load_live_b5_book
from scripts.bucket5_insurance_bt import HedgeBudgetPolicy, InsuranceConfig


def build_b5_insurance_strategy_guide(
    summary: dict[str, Any] | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plain-language strategy guide (SPX-0DTE StrategyGuide schema)."""
    summary = summary or {}
    meta = meta or {}
    cagr = summary.get("combined_CAGR")
    sharpe = summary.get("combined_Sharpe")
    maxdd = summary.get("combined_MaxDD")
    realized = summary.get("realized_$")
    start = meta.get("start", "2008-01-01")
    end = meta.get("end", "today")
    synth = int(meta.get("synthetic_days") or 0)

    results = []
    if cagr is not None:
        results.append({"label": "Combined CAGR", "value": f"{100 * float(cagr):.1f}%"})
    if sharpe is not None:
        results.append({"label": "Sharpe", "value": f"{float(sharpe):.2f}"})
    if maxdd is not None:
        results.append({"label": "Max drawdown", "value": f"{100 * float(maxdd):.1f}%"})
    if realized is not None:
        results.append({"label": "Harvested put cash", "value": f"${float(realized):,.0f}"})
    results.append({"label": "Sample", "value": f"{start} → {end}"})
    if synth:
        results.append({"label": "Synthetic UVIX/SVIX days", "value": f"{synth:,}"})

    return {
        "title": "Bucket 5 — Positive-Carry Tail Insurance",
        "subtitle": (
            "Short UVIX / short SVIX carry funds a ladder of SPX puts; "
            "monetize put convexity in crashes so insurance becomes cash, not a temporary mark."
        ),
        "sections": [
            {
                "title": "What this strategy does",
                "paragraphs": [
                    (
                        "Most short-volatility strategies make money slowly in calm markets and then "
                        "give it back in a crash. Bucket 5 is designed as a self-funding insurance "
                        "policy: earn carry from short-vol ETPs, park most capital in T-bills, spend "
                        "part of that income on longer-dated SPX puts, and sell those puts into "
                        "spikes so crash gains are banked."
                    ),
                    (
                        "This product dashboard shows the research insurance stack (dual-short + puts). "
                        "The live GTP book only holds a tiny volatility-ETP sleeve (~0.25% of gross) "
                        "as a placeholder risk budget — not the full dual-short + put product."
                    ),
                ],
            },
            {
                "title": "Carry engine (short UVIX / short SVIX)",
                "paragraphs": [
                    (
                        "UVIX is leveraged long-vol and bleeds in contango; shorting it harvests that "
                        "bleed. SVIX is roughly the opposite exposure, so shorting SVIX pays in a "
                        "vol spike. The book shorts both and dials the mix with rho = "
                        "SVIX-short / UVIX-short notional."
                    ),
                ],
                "bullets": [
                    "Calm contango → lean into short UVIX (more carry).",
                    "Flattening / backwardation → raise rho toward vol-neutral and cut sleeve gross.",
                    "Only ~20% of equity is deployed in the dual-short sleeve; the rest earns T-bill yield.",
                ],
            },
            {
                "title": "Regime policy and adaptive cadence",
                "bullets": [
                    "Signal: VIX / VIX3M term-structure ratio.",
                    "Deep contango (ratio ≤ 0.88): rho ≈ 1.0, full sleeve gross.",
                    "Backwardation (ratio ≥ 1.00): rho ≈ 2.0, sleeve cut to ~35% of calm size.",
                    "Rebalance clock: ~14 trading days in calm markets; speeds up toward ~2 days in stress "
                    "(same shape as Bucket 4 cadence, driven by the term structure).",
                ],
            },
            {
                "title": "SPX put ladder",
                "paragraphs": [
                    (
                        "Carry alone is not insurance. Part of sleeve + bill income buys a ladder of "
                        "longer-dated SPX puts (buy ~6M DTE, roll ~3M). Production spends about 2.4% "
                        "of equity in premium per roll across 10% / 20% / 30% OTM rungs, with a dynamic "
                        "budget that spends more when puts are cheap."
                    ),
                ],
            },
            {
                "title": "Monetizing puts — the real edge",
                "paragraphs": [
                    (
                        "Holding puts through a spike and then watching them decay is how hedged "
                        "short-vol books lose the plot. Monetization sells a planned fraction into "
                        "strength so the crash becomes a cash payout."
                    ),
                ],
                "bullets": [
                    "Profit tiers: scale out around 3× / 5× / 8× of cost.",
                    "VIX overrides: monetize on extreme VIX levels even if multiples are uneven.",
                    "Giveback stop: once doubled, sell down if MTM falls ~35% from peak.",
                    "Re-arm: on full exit, bank most proceeds and optionally buy fresh puts at the new spot.",
                    "Redeploy: lean harvested cash into the sleeve after spikes (when forward carry is richest); "
                    "park more in bills when calm.",
                ],
            },
            {
                "title": "Risks",
                "bullets": [
                    "UVIX can gap; puts do not perfectly offset ETP dislocations or halts.",
                    "Monetization can be early or late; V-shaped crashes may harvest little.",
                    "Long calm regimes create premium drag if carry/bills cannot fund the ladder.",
                    "SPX puts vs VIX-ETP carry is a basis trade — they usually spike together, not always one-for-one.",
                    "Extended history uses synthetic UVIX/SVIX before live inception; treat deep drawdowns as research stress.",
                ],
            },
            {
                "title": "How to read this dashboard",
                "bullets": [
                    "Overview: strategy guide, KPIs, equity / drawdown / attribution, crash scenarios.",
                    "Regime: term structure, rho, sleeve gross, rebalance cadence.",
                    "Daily: pick any backtest day for carry marks, put cashflows, and monetize events; "
                    "days tagged · live also show the GTP vol-ETP sleeve book.",
                ],
            },
        ],
        "results": results,
    }


def _series_pairs(s: pd.Series, *, nd: int = 4) -> list[list[Any]]:
    out: list[list[Any]] = []
    for dt, val in s.items():
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        out.append([pd.Timestamp(dt).strftime("%Y-%m-%d"), round(float(val), nd)])
    return out


def export_daily_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Full daily path for the product dashboard (not downsampled)."""
    bt = res["bt"]
    carry = res["carry"]
    ladder = res["ladder"]
    cfg: InsuranceConfig = res["cfg"]
    rebal = set(pd.DatetimeIndex(res["rebal"]))

    deployed = (carry["gross"] / carry["equity"].replace(0, np.nan)).clip(0, 1).fillna(0.0)
    tbill_daily = (cfg.tbill_rate / 252.0) * (1.0 - deployed)
    put_cf = ladder["put_cash_flow"].reindex(bt.index).fillna(0.0)
    realized = (
        ladder["realized"].reindex(bt.index).fillna(0.0)
        if "realized" in ladder.columns
        else pd.Series(0.0, index=bt.index)
    )

    rows: list[dict[str, Any]] = []
    for dt in bt.index:
        d = pd.Timestamp(dt)
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "combined_equity": round(float(bt.at[dt, "combined_equity"]), 2),
                "combined_ret": round(float(bt.at[dt, "combined_ret"]), 6),
                "drawdown_pct": round(float(bt.at[dt, "drawdown"]), 6),
                "sleeve_ret": round(float(carry["ret"].reindex([dt]).fillna(0.0).iloc[0]), 6),
                "tbill_ret": round(float(tbill_daily.reindex([dt]).fillna(0.0).iloc[0]), 6),
                "base_equity": round(float(bt.at[dt, "base_equity"]), 2),
                "put_mtm": round(float(bt.at[dt, "put_mtm"]), 2),
                "put_cash_flow": round(float(put_cf.loc[dt]), 2),
                "put_cash_cum": round(float(bt.at[dt, "put_cash_cum"]), 2),
                "realized_day": round(float(realized.loc[dt]), 2),
                "redeploy_extra": round(float(bt.at[dt, "redeploy_extra"]), 2),
                "rho": round(float(bt.at[dt, "rho"]), 4),
                "gross_frac": round(float(bt.at[dt, "gross_frac"]), 4),
                "ratio": round(float(bt.at[dt, "ratio"]), 4) if np.isfinite(bt.at[dt, "ratio"]) else None,
                "rebalance_flag": bool(d in rebal or d.normalize() in rebal),
            }
        )
    return rows


def export_day_marks(res: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Per-day mark rows (carry legs + put cashflow) for the Daily tab."""
    carry = res["carry"]
    ladder = res["ladder"]
    bt = res["bt"]
    put_cf = ladder["put_cash_flow"].reindex(bt.index).fillna(0.0)
    put_mtm = ladder["put_mtm"].reindex(bt.index).fillna(0.0)
    put_mtm_chg = put_mtm.diff().fillna(put_mtm)

    out: dict[str, list[dict[str, Any]]] = {}
    for dt in bt.index:
        d = pd.Timestamp(dt).strftime("%Y-%m-%d")
        c = carry.reindex([dt]).iloc[0] if dt in carry.index else None
        marks: list[dict[str, Any]] = []
        if c is not None:
            marks.append(
                {
                    "kind": "carry_leg",
                    "name": "UVIX short",
                    "notional_usd": round(float(c["u_notional"]), 2),
                    "price": round(float(c["uvix"]), 4),
                    "financing_pnl": round(float(c["financing_pnl"]) * (
                        abs(float(c["u_notional"])) / max(abs(float(c["u_notional"])) + abs(float(c["s_notional"])), 1e-9)
                    ), 2),
                    "rebalance": bool(c["rebalance"]),
                }
            )
            marks.append(
                {
                    "kind": "carry_leg",
                    "name": "SVIX short",
                    "notional_usd": round(float(c["s_notional"]), 2),
                    "price": round(float(c["svix"]), 4),
                    "financing_pnl": round(float(c["financing_pnl"]) * (
                        abs(float(c["s_notional"])) / max(abs(float(c["u_notional"])) + abs(float(c["s_notional"])), 1e-9)
                    ), 2),
                    "rebalance": bool(c["rebalance"]),
                }
            )
        marks.append(
            {
                "kind": "put_overlay",
                "name": "SPX put ladder",
                "mtm_usd": round(float(put_mtm.loc[dt]), 2),
                "mtm_chg_usd": round(float(put_mtm_chg.loc[dt]), 2),
                "cash_flow_usd": round(float(put_cf.loc[dt]), 2),
            }
        )
        out[d] = marks
    return out


def export_events_by_date(res: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    events = res["ladder"].attrs.get("monetize_events") or []
    out: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        d = str(e.get("date") or "")[:10]
        if not d:
            continue
        out.setdefault(d, []).append(
            {
                "kind": e.get("kind"),
                "usd": e.get("usd"),
                "otm_pct": e.get("otm_pct"),
                "mult": e.get("mult"),
                "vix": e.get("vix"),
                "contracts_sold": e.get("contracts_sold"),
            }
        )
    return out


def export_regime_panels(res: dict[str, Any], panel: pd.DataFrame) -> dict[str, Any]:
    bt = res["bt"]
    cfg: InsuranceConfig = res["cfg"]
    rebal = pd.DatetimeIndex(res["rebal"])
    ratio = bt["ratio"]
    vix = panel["vix"].reindex(bt.index).ffill()

    # Approximate cadence interval at each rebalance
    intervals: list[list[Any]] = []
    for i, d in enumerate(rebal):
        if i + 1 < len(rebal):
            gap = int((rebal[i + 1] - d).days)
            intervals.append([d.strftime("%Y-%m-%d"), gap])

    hb = cfg.hedge_budget or HedgeBudgetPolicy(enabled=False)
    budget_mult = []
    for dt in bt.index:
        r = float(ratio.loc[dt]) if np.isfinite(ratio.loc[dt]) else np.nan
        vx = float(vix.loc[dt]) if dt in vix.index and np.isfinite(vix.loc[dt]) else np.nan
        budget_mult.append([pd.Timestamp(dt).strftime("%Y-%m-%d"), round(float(hb.multiplier(r, vx)), 4)])

    return {
        "ratio": _series_pairs(ratio),
        "rho": _series_pairs(bt["rho"]),
        "gross_frac": _series_pairs(bt["gross_frac"]),
        "vix": _series_pairs(vix),
        "cadence_interval_days": intervals,
        "put_budget_mult": budget_mult,
        "r_lo": cfg.regime.r_lo,
        "r_hi": cfg.regime.r_hi,
        "rebalance_dates": [d.strftime("%Y-%m-%d") for d in rebal],
    }


def build_run_payload(
    *,
    run_id: str,
    label: str,
    res: dict[str, Any],
    panel: pd.DataFrame,
    summary: dict[str, Any],
    crash: dict[str, Any],
    meta: dict[str, Any],
    assumptions: dict[str, Any],
    include_guide: bool = False,
) -> dict[str, Any]:
    cfg: InsuranceConfig = res["cfg"]
    cfg_dict = {
        "sleeve_frac": cfg.sleeve_frac,
        "tbill_rate": cfg.tbill_rate,
        "base_days": cfg.base_days,
        "cadence_k": cfg.cadence_k,
        "hedge_kind": cfg.hedge_kind,
        "uvix_slip_bps": cfg.uvix_slip_bps,
        "fee_bps": cfg.fee_bps,
        "rungs": [{"otm_pct": r.otm_pct, "per_roll_frac": r.per_roll_frac} for r in cfg.rungs],
        "regime": {
            "rho_contango": cfg.regime.rho_contango,
            "rho_backwardation": cfg.regime.rho_backwardation,
            "gross_contango": cfg.regime.gross_contango,
            "gross_backwardation": cfg.regime.gross_backwardation,
            "r_lo": cfg.regime.r_lo,
            "r_hi": cfg.regime.r_hi,
        },
        "monetize": None
        if cfg.monetize is None
        else {
            "profit_tiers": list(cfg.monetize.profit_tiers),
            "vix_tiers": list(cfg.monetize.vix_tiers),
            "giveback_frac": cfg.monetize.giveback_frac,
            "giveback_min_mult": cfg.monetize.giveback_min_mult,
            "bank_frac": cfg.monetize.bank_frac,
            "rearm": cfg.monetize.rearm,
            "runner_frac": cfg.monetize.runner_frac,
            "runner_mult": cfg.monetize.runner_mult,
        },
        "redeploy": None
        if cfg.redeploy is None
        else {
            "sleeve_w_contango": cfg.redeploy.sleeve_w_contango,
            "sleeve_w_backwardation": cfg.redeploy.sleeve_w_backwardation,
        },
        "hedge_budget": None
        if cfg.hedge_budget is None
        else {
            "enabled": cfg.hedge_budget.enabled,
            "contango_mult": cfg.hedge_budget.contango_mult,
            "stress_mult": cfg.hedge_budget.stress_mult,
        },
    }
    daily = export_daily_rows(res)
    events = export_events_by_date(res)
    # Keep full daily[] for charts; sparsify marks to rebalance / cashflow / event / month-start days.
    marks_full = export_day_marks(res)
    marks: dict[str, list] = {}
    for row in daily:
        d = row["date"]
        keep = (
            row["rebalance_flag"]
            or abs(float(row["put_cash_flow"] or 0)) > 1.0
            or abs(float(row["realized_day"] or 0)) > 1.0
            or d in events
            or d.endswith("-01")
        )
        if keep:
            marks[d] = marks_full.get(d, [])
    # Always attach marks for event days even if not in daily loop edge cases
    for d, ev in events.items():
        marks.setdefault(d, marks_full.get(d, []))

    meta_out = {
        **meta,
        "date_range": f"{meta.get('start', '?')} → {meta.get('end', '?')}",
        "assumptions": assumptions,
        "config": cfg_dict,
        "assumption_sections": build_assumption_sections(cfg_dict, meta, assumptions),
    }
    if include_guide:
        meta_out["strategy_guide"] = build_b5_insurance_strategy_guide(summary, meta=meta)

    return {
        "id": run_id,
        "label": label,
        "meta": meta_out,
        "summary": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in summary.items()},
        "crash": crash,
        "daily": daily,
        "marks_by_date": marks,
        "events_by_date": events,
        "regime_panels": export_regime_panels(res, panel),
        "equity_series": [[r["date"], r["combined_equity"]] for r in daily],
        "drawdown_series": [[r["date"], r["drawdown_pct"]] for r in daily],
    }


def build_live_days(run_date: str | None = None) -> dict[str, Any]:
    """Live GTP vol-ETP sleeve marks (not the full insurance product)."""
    book = load_live_b5_book(run_date)
    days: dict[str, Any] = {}
    rd = book.get("run_date")
    if rd and book.get("rows"):
        days[str(rd)] = {
            "mode": "gtp_sleeve",
            "note": "Live Bucket 5 is the tiny volatility-ETP sleeve in the L/S book — not dual-short + puts.",
            "positions": book["rows"],
            "proposed_gross_usd": round(sum(float(r.get("proposed_gross_usd") or 0) for r in book["rows"]), 2),
        }

    # Attach recent accounting PnL if present
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    runs = repo / "data" / "runs"
    if runs.is_dir():
        for child in sorted(runs.iterdir(), reverse=True)[:40]:
            pnl = child / "accounting" / "pnl_bucket_5.csv"
            if not pnl.is_file():
                continue
            try:
                df = pd.read_csv(pnl)
            except Exception:
                continue
            total = float(pd.to_numeric(df.get("total_pnl"), errors="coerce").fillna(0).sum()) if "total_pnl" in df.columns else None
            if total is None and "pnl" in df.columns:
                total = float(pd.to_numeric(df["pnl"], errors="coerce").fillna(0).sum())
            entry = days.setdefault(
                child.name,
                {"mode": "accounting", "note": "Accounting attribution for live B5 sleeve.", "positions": []},
            )
            entry["marked_pnl"] = round(total, 2) if total is not None else None
            if len(days) >= 15:
                break
    return {"days": days}
