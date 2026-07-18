"""Per-pair Bucket-4 trade ledger vs production cadence model.

Reconstructs every executed B4 trade from production-actual artifacts
(``pair_stats.rebalance_dates`` + ``pair_daily_pnl`` legs / txn) and labels
each fill against the TR/VCR cadence calendar + plan membership.

This answers: *how did each B4 pair trade in the sim, and is that how live
``rebalance_strategy`` + ``bucket4_cadence_gate`` would behave?*

Important realism notes (sim vs live)
-------------------------------------
* Sim (``b4_execution=cadence`` + §9 knobs): B4 add/true-drop only on
  ``b4_membership_clock`` (default ``operator_5d``); on model cadence day
  rebuilds legs from gross × dynamic h, then Phase-2b bands + ratchet
  cover guard (``b4_apply_resize_bands`` / ``b4_ratchet_execution_guard``).
* Live: operator runs every ``operator_check_days`` (5); cadence gate marks
  pairs due/defer; **Phase-2b bands + ratchet cover rules apply**.
* ``pair_daily.is_rebalance`` is **book-level** (any pair traded that day) —
  do not use it as a per-pair trade flag. Use ``rebalance_dates`` / txn.
* Ledger reasons: ``enter_operator`` / ``exit_operator`` (membership day),
  ``enter_plan_add`` / ``exit_plan_drop`` (legacy every_plan labels),
  ``cadence_resize`` (on model cadence), ``off_cadence_resize``, ``resize``.

CLI::

    python -m scripts.b4_pair_trade_audit \\
        --out-dir notebooks/output/production_actual_bt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.b4_backtest_pair_charts import build_pair_path_bundle
from scripts.b4_historical_audit import load_b4_plan_history

REPO = Path(__file__).resolve().parents[1]
B4_SLEEVE = "inverse_decay_bucket4"
GROSS_EPS = 1.0  # USD; treat sub-$1 as flat


def _norm(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _parse_rebal_dates(raw: object) -> list[pd.Timestamp]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    out: list[pd.Timestamp] = []
    for part in s.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(pd.Timestamp(part).normalize())
        except Exception:
            continue
    return sorted(set(out))


def _legs_on(daily: pd.DataFrame, d: pd.Timestamp) -> tuple[float, float, float]:
    """Return (etf_usd, und_usd, txn_cost) on date d; NaNs if missing."""
    sub = daily[daily["date"] == d]
    if sub.empty:
        return float("nan"), float("nan"), 0.0
    row = sub.iloc[-1]
    return (
        float(pd.to_numeric(row.get("etf_usd"), errors="coerce") or 0.0),
        float(pd.to_numeric(row.get("underlying_usd"), errors="coerce") or 0.0),
        float(pd.to_numeric(row.get("txn_cost"), errors="coerce") or 0.0),
    )


def _legs_before(daily: pd.DataFrame, d: pd.Timestamp) -> tuple[float, float]:
    prior = daily[daily["date"] < d]
    if prior.empty:
        return 0.0, 0.0
    row = prior.iloc[-1]
    return (
        float(pd.to_numeric(row.get("etf_usd"), errors="coerce") or 0.0),
        float(pd.to_numeric(row.get("underlying_usd"), errors="coerce") or 0.0),
    )


def _classify(
    *,
    prev_gross: float,
    new_gross: float,
    on_model_cadence: bool | None,
    in_plan: bool | None,
    in_plan_prev: bool | None,
    on_operator_day: bool | None = None,
) -> str:
    entering = prev_gross <= GROSS_EPS and new_gross > GROSS_EPS
    exiting = prev_gross > GROSS_EPS and new_gross <= GROSS_EPS
    if entering:
        if on_operator_day is True:
            return "enter_operator"
        if in_plan_prev is False and in_plan is True:
            return "enter_plan_add"
        return "enter"
    if exiting:
        if on_operator_day is True:
            return "exit_operator"
        if in_plan is False and in_plan_prev is True:
            return "exit_plan_drop"
        return "exit"
    if on_model_cadence is True:
        return "cadence_resize"
    if on_model_cadence is False:
        return "off_cadence_resize"
    return "resize"  # model cadence not computed


def _plan_membership(
    plan_hist: pd.DataFrame,
    etf: str,
    d: pd.Timestamp,
) -> bool | None:
    if plan_hist is None or plan_hist.empty:
        return None
    sub = plan_hist[plan_hist["ETF"].map(_norm) == _norm(etf)]
    if sub.empty:
        return None
    # Plan dated on/before d (Friday plan stamped as plan date).
    dated = sub[sub["date"] <= d]
    if dated.empty:
        return False
    last_d = dated["date"].max()
    # Only count as in-plan if last plan within ~10 calendar days (stale = out).
    if (d - last_d).days > 10:
        return False
    row = dated[dated["date"] == last_d].iloc[-1]
    g_raw = row.get("gross_target_usd", row.get("gross"))
    g = float(pd.to_numeric(g_raw, errors="coerce") or 0.0)
    return g > GROSS_EPS


def build_b4_trade_ledger(
    *,
    pair_stats: pd.DataFrame,
    pair_daily: pd.DataFrame,
    plan_hist: pd.DataFrame | None = None,
    price_panel: dict[str, pd.DataFrame] | None = None,
    start: str | pd.Timestamp | None = None,
    fill_yahoo: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trade_ledger, pair_summary)."""
    ps = pair_stats.copy()
    if "sleeve" in ps.columns:
        ps = ps[ps["sleeve"].astype(str).str.strip().str.lower().eq(B4_SLEEVE)].copy()
    if ps.empty:
        return pd.DataFrame(), pd.DataFrame()

    daily_all = pair_daily.copy()
    daily_all["date"] = pd.to_datetime(daily_all["date"], errors="coerce").dt.normalize()
    daily_all = daily_all.dropna(subset=["date"])
    if "sleeve" in daily_all.columns:
        daily_all = daily_all[
            daily_all["sleeve"].astype(str).str.strip().str.lower().eq(B4_SLEEVE)
        ].copy()
    daily_all["ETF"] = daily_all["ETF"].map(_norm)

    if plan_hist is not None and not plan_hist.empty:
        ph = plan_hist.copy()
        ph["date"] = pd.to_datetime(ph["date"], errors="coerce").dt.normalize()
        ph["ETF"] = ph["ETF"].map(_norm)
    else:
        ph = pd.DataFrame()

    trade_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # Operator membership days (same clock as sim default operator_5d).
    all_dates = sorted(pd.DatetimeIndex(daily_all["date"].unique()))
    try:
        from scripts.production_actual_backtest import _membership_day_set

        operator_days = _membership_day_set(
            pd.DatetimeIndex(all_dates), mode="operator_5d", check_days=5
        )
    except Exception:
        operator_days = set()

    for _, prow in ps.iterrows():
        etf = _norm(prow.get("ETF"))
        und = _norm(prow.get("Underlying"))
        daily = daily_all[daily_all["ETF"] == etf].sort_values("date")
        if daily.empty:
            continue

        rebals = _parse_rebal_dates(prow.get("rebalance_dates"))
        model_rebal_set: set[pd.Timestamp] = set()
        model_h = pd.Series(dtype=float)
        cadence_note = ""
        if price_panel is not None and etf in price_panel:
            bundle = build_pair_path_bundle(
                etf=etf,
                underlying=und,
                pair_daily=daily_all,
                prices=price_panel[etf],
                start=start,
                fill_yahoo=fill_yahoo,
            )
            if bundle.get("ok"):
                model_rebal_set = {
                    pd.Timestamp(x).normalize() for x in bundle.get("model_rebals", [])
                }
                model_h = bundle.get("model_h")
                if not isinstance(model_h, pd.Series):
                    model_h = pd.Series(dtype=float)
            else:
                cadence_note = str(bundle.get("reason", "cadence_unavailable"))
        elif price_panel is not None:
            cadence_note = "etf_missing_from_panel"

        n_enter = n_exit = n_cadence = n_off = 0
        txn_total = 0.0
        intervals: list[float] = []
        prev_trade: pd.Timestamp | None = None

        for d in rebals:
            if start is not None and d < pd.Timestamp(start):
                continue
            prev_a, prev_b = _legs_before(daily, d)
            new_a, new_b, txn = _legs_on(daily, d)
            if not np.isfinite(new_a):
                new_a, new_b = 0.0, 0.0
            prev_g = abs(prev_a) + abs(prev_b)
            new_g = abs(new_a) + abs(new_b)
            turn = abs(new_a - prev_a) + abs(new_b - prev_b)
            on_cad: bool | None
            if model_rebal_set:
                on_cad = d in model_rebal_set
            elif cadence_note:
                on_cad = None
            else:
                # No panel requested — leave cadence unknown.
                on_cad = None
            in_plan = _plan_membership(ph, etf, d) if not ph.empty else None
            in_plan_prev = (
                _plan_membership(ph, etf, d - pd.Timedelta(days=1)) if not ph.empty else None
            )
            on_op = (d in operator_days) if operator_days else None
            reason = _classify(
                prev_gross=prev_g,
                new_gross=new_g,
                on_model_cadence=on_cad,
                in_plan=in_plan,
                in_plan_prev=in_plan_prev,
                on_operator_day=on_op,
            )
            if reason.startswith("enter"):
                n_enter += 1
            elif reason.startswith("exit"):
                n_exit += 1
            elif reason == "cadence_resize":
                n_cadence += 1
            elif reason == "off_cadence_resize":
                n_off += 1
            # "resize" (cadence unknown) counted neither as hit nor miss

            book_h = abs(new_b) / abs(new_a) if abs(new_a) > 1e-9 else float("nan")
            mh = float("nan")
            if len(model_h) and d in model_h.index and np.isfinite(model_h.loc[d]):
                mh = float(model_h.loc[d])
            elif len(model_h):
                mh_ff = model_h.reindex(pd.DatetimeIndex([d])).ffill()
                if len(mh_ff) and np.isfinite(mh_ff.iloc[0]):
                    mh = float(mh_ff.iloc[0])

            gap = float((d - prev_trade).days) if prev_trade is not None else float("nan")
            if np.isfinite(gap):
                intervals.append(gap)
            prev_trade = d
            txn_total += float(txn or 0.0)

            trade_rows.append(
                {
                    "ETF": etf,
                    "Underlying": und,
                    "date": d,
                    "reason": reason,
                    "on_model_cadence": on_cad,
                    "on_operator_day": on_op,
                    "in_plan": in_plan,
                    "prev_etf_usd": prev_a,
                    "prev_underlying_usd": prev_b,
                    "prev_gross": prev_g,
                    "etf_usd": new_a,
                    "underlying_usd": new_b,
                    "gross": new_g,
                    "turnover_usd": turn,
                    "txn_cost": float(txn or 0.0),
                    "book_h": book_h,
                    "model_h": mh,
                    "h_gap": (book_h - mh) if np.isfinite(book_h) and np.isfinite(mh) else np.nan,
                    "days_since_prev_trade": gap,
                    "weekday": int(d.weekday()),  # Mon=0
                }
            )

        # Rapid churn: enter/exit within 5 calendar days
        etf_trades = [r for r in trade_rows if r["ETF"] == etf]
        churn_flags = 0
        for i in range(1, len(etf_trades)):
            a, b = etf_trades[i - 1], etf_trades[i]
            if (
                a["reason"].startswith("enter")
                and b["reason"].startswith("exit")
                and (b["date"] - a["date"]).days <= 5
            ) or (
                a["reason"].startswith("exit")
                and b["reason"].startswith("enter")
                and (b["date"] - a["date"]).days <= 5
            ):
                churn_flags += 1

        n_model = len(model_rebal_set)
        n_resize_unknown = sum(1 for r in etf_trades if r["reason"] == "resize")
        # Share of non-enter/exit trades that landed on model cadence.
        resize_n = n_cadence + n_off
        cadence_hit = (n_cadence / resize_n) if resize_n else float("nan")
        membership_share = (
            (n_enter + n_exit) / len(etf_trades) if etf_trades else float("nan")
        )

        realism = []
        if n_off > 0:
            realism.append(f"off_cadence_resize={n_off}")
        if churn_flags > 0:
            realism.append(f"rapid_membership_churn={churn_flags}")
        if membership_share >= 0.7 and len(etf_trades) >= 4:
            realism.append("membership_dominated")
        if resize_n and n_cadence == resize_n:
            realism.append("resize_all_on_cadence")
        if cadence_note:
            realism.append(cadence_note)
        if not realism:
            realism.append("ok")

        summary_rows.append(
            {
                "ETF": etf,
                "Underlying": und,
                "pnl_usd": float(pd.to_numeric(prow.get("pnl_usd"), errors="coerce") or 0.0),
                "txn_cost_usd": float(
                    pd.to_numeric(prow.get("txn_cost_usd", prow.get("txn_cost")), errors="coerce")
                    or txn_total
                ),
                "n_trades": len(etf_trades),
                "n_enter": n_enter,
                "n_exit": n_exit,
                "n_cadence_resize": n_cadence,
                "n_cadence_rebal": n_cadence,  # alias for older notebook columns
                "n_off_cadence_resize": n_off,
                "n_resize_unknown_cadence": n_resize_unknown,
                "membership_trade_share": membership_share,
                "cadence_hit_rate": cadence_hit,
                "n_model_cadence_days_in_window": n_model,
                "median_days_between_trades": float(np.nanmedian(intervals)) if intervals else np.nan,
                "mean_days_between_trades": float(np.nanmean(intervals)) if intervals else np.nan,
                "rapid_churn_events": churn_flags,
                "realism_flags": ";".join(realism),
            }
        )

    ledger = pd.DataFrame(trade_rows)
    if not ledger.empty:
        ledger = ledger.sort_values(["ETF", "date"]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("pnl_usd", ascending=False).reset_index(drop=True)
    return ledger, summary


def realism_checklist() -> pd.DataFrame:
    """Static sim-vs-live checklist (documentation table)."""
    rows = [
        {
            "dimension": "Membership (add/drop)",
            "sim": "B4 add/true-drop only on operator_check_days (default 5) or weekly_fri",
            "live": "Phase-1 establish/cleanup only when rebalance_strategy runs (~5d)",
            "match": "yes — b4_membership_clock=operator_5d (every_plan = legacy A/B)",
        },
        {
            "dimension": "Purgatory",
            "sim": "reduce-only toward model_*; missing/zero model share-holds",
            "live": "executable 0 on plan; Phase-2b trim-only; no Phase-1 close",
            "match": "yes — purgatory_model_zero_policy=hold",
        },
        {
            "dimension": "When B4 may resize",
            "sim": "TR/VCR cadence day + Phase-2b bands (12%/4%/$250)",
            "live": "Phase-2b resize if cadence gate due on operator run",
            "match": "mostly — same bands; sim can fire mid-week on exact cadence day",
        },
        {
            "dimension": "Resize sizing",
            "sim": "gross×h then _resize_band_target when b4_apply_resize_bands",
            "live": "Phase-2b toward plan USD with enter/exit bands + min trade $",
            "match": "yes when b4_apply_resize_bands=true",
        },
        {
            "dimension": "Inverse ratchet",
            "sim": "grow-only floor + cover pin/trim-cap on ledger (plan ratchet_* flags)",
            "live": "GTP floor + Phase-2b allow_inverse_cover / ratchet_released",
            "match": "yes — b4_ratchet_execution_guard (floors from sim plan, not Flex)",
        },
        {
            "dimension": "Empty B4 plan",
            "sim": "exec gross~0 → share-hold open B4 (no true-drop wipe)",
            "live": "no screened/sizing → keep last book; do not Phase-1 wipe into cash",
            "match": "yes — b4_empty_plan_policy=hold (archive-gap anti-glitch)",
        },
        {
            "dimension": "Costs",
            "sim": "20 bp slip + $0.0035/share on traded notional",
            "live": "broker slip/commission; borrow from IBKR",
            "match": "approximate",
        },
    ]
    return pd.DataFrame(rows)


def run_audit(
    out_dir: Path,
    *,
    start: str | None = None,
    with_panel: bool = True,
    fill_yahoo: bool = False,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    ps = pd.read_csv(out_dir / "pair_stats.csv")
    pdaily = pd.read_csv(out_dir / "pair_daily_pnl.csv", parse_dates=["date"])
    plans_dir = out_dir / "plans"
    plan_hist = load_b4_plan_history(plans_dir) if plans_dir.is_dir() else pd.DataFrame()

    panel = None
    if with_panel:
        try:
            from scripts.sizing_tilt_cadence_bt import load_price_panel

            run_date = str(pdaily["date"].max().date()) if len(pdaily) else None
            panel = load_price_panel(run_date) if run_date else None
        except Exception:
            panel = None

    ledger, summary = build_b4_trade_ledger(
        pair_stats=ps,
        pair_daily=pdaily,
        plan_hist=plan_hist,
        price_panel=panel,
        start=start,
        fill_yahoo=fill_yahoo,
    )
    checklist = realism_checklist()

    paths = {
        "ledger": out_dir / "b4_pair_trade_ledger.csv",
        "summary": out_dir / "b4_pair_trade_summary.csv",
        "checklist": out_dir / "b4_sim_vs_live_checklist.csv",
    }
    ledger.to_csv(paths["ledger"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checklist.to_csv(paths["checklist"], index=False)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "notebooks" / "output" / "production_actual_bt",
    )
    ap.add_argument("--start", default=None)
    ap.add_argument("--no-panel", action="store_true", help="Skip model cadence overlay")
    ap.add_argument("--fill-yahoo", action="store_true")
    args = ap.parse_args()
    paths = run_audit(
        args.out_dir,
        start=args.start,
        with_panel=not args.no_panel,
        fill_yahoo=args.fill_yahoo,
    )
    summary = pd.read_csv(paths["summary"])
    ledger = pd.read_csv(paths["ledger"])
    print(f"wrote {paths['ledger']} ({len(ledger)} trades)")
    print(f"wrote {paths['summary']} ({len(summary)} pairs)")
    print(f"wrote {paths['checklist']}")
    if not summary.empty:
        print("\n=== B4 pair trade summary (by pnl) ===")
        cols = [
            c
            for c in (
                "ETF",
                "n_trades",
                "n_enter",
                "n_exit",
                "n_cadence_rebal",
                "n_off_cadence_resize",
                "cadence_hit_rate",
                "median_days_between_trades",
                "rapid_churn_events",
                "pnl_usd",
                "realism_flags",
            )
            if c in summary.columns
        ]
        print(summary[cols].round(2).to_string(index=False))
        print("\n=== reason mix ===")
        if not ledger.empty:
            print(ledger["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
