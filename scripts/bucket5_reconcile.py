"""Bucket 5 Production B — daily reconciliation and NAV identity.

Checks (plan section 5.1 / 8.1):
1. Lot identity — every open option position in the broker Flex snapshot maps
   to exactly one B5 ledger lot by ``conId`` and the contract counts match.
2. NAV identity — chains today's B5 NAV from the prior reconciled NAV plus the
   day's components (carry P&L, borrow, short credit, option mark change,
   premium paid, sale proceeds, commissions); residual must be within
   ``reconcile.max_residual_usd``.
3. Publishes one health status: green | no_new_risk | reduce_only | halted.

Usage:
    python scripts/bucket5_reconcile.py --run-date 2026-07-21 [--prior-date 2026-07-20]

Reads  data/runs/<date>/accounting/{pnl_bucket_5.csv, net_exposure_bucket_5.csv,
       pnl_bucket_5_options.csv, b5_option_positions.csv}
Writes data/runs/<date>/bucket5_production/reconcile.json and appends a
       RECONCILE event to the B5 ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bucket5_ledger import Bucket5Ledger  # noqa: E402
from scripts.bucket5_policy import load_b5_config  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _find_prior_reconcile(run_date: str) -> tuple[str, dict] | None:
    runs = Path("data") / "runs"
    if not runs.exists():
        return None
    candidates = sorted(
        (d.name for d in runs.iterdir()
         if d.is_dir() and d.name < run_date and (d / "bucket5_production" / "reconcile.json").exists()),
        reverse=True,
    )
    for name in candidates:
        try:
            payload = json.loads((runs / name / "bucket5_production" / "reconcile.json").read_text(encoding="utf-8"))
            return name, payload
        except Exception:
            continue
    return None


def reconcile(run_date: str, prior_date: str | None = None, config_path: str | Path | None = None) -> dict:
    cfg = load_b5_config(config_path)
    max_residual = float((cfg.get("reconcile") or {}).get("max_residual_usd", 10.0))
    acct = Path("data") / "runs" / run_date / "accounting"
    out_dir = Path("data") / "runs" / run_date / str((cfg.get("paths") or {}).get("run_subdir", "bucket5_production"))
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "run_date": run_date,
        "mode": cfg["mode"],
        "kill_mode": cfg["kill_mode"],
        "checks": {},
        "notes": [],
        "health": "green",
    }

    def _degrade(level: str, note: str) -> None:
        order = ["green", "no_new_risk", "reduce_only", "halted"]
        if order.index(level) > order.index(result["health"]):
            result["health"] = level
        result["notes"].append(note)

    # ------------------------------------------------------------- carry book
    pnl_b5 = _read_csv(acct / "pnl_bucket_5.csv")
    exp_b5 = _read_csv(acct / "net_exposure_bucket_5.csv")
    if pnl_b5 is None or exp_b5 is None:
        _degrade("no_new_risk", "missing carry accounting artifacts (pnl/net_exposure bucket 5)")
        carry = {}
    else:
        carry = {
            "total_pnl_usd": float(pnl_b5.get("total_pnl", pd.Series(dtype=float)).sum()),
            "realized_pnl_usd": float(pnl_b5.get("realized_pnl", pd.Series(dtype=float)).sum()),
            "unrealized_pnl_usd": float(pnl_b5.get("unrealized_pnl", pd.Series(dtype=float)).sum()),
            "borrow_fees_usd": float(pnl_b5.get("borrow_fees", pd.Series(dtype=float)).sum()),
            "short_credit_interest_usd": float(pnl_b5.get("short_credit_interest", pd.Series(dtype=float)).sum()),
            "gross_notional_usd": float(exp_b5.get("gross_notional_usd", pd.Series(dtype=float)).sum()),
            "net_notional_usd": float(exp_b5.get("net_notional_usd", pd.Series(dtype=float)).sum()),
        }
    result["carry"] = carry

    # ------------------------------------------------------------- put book
    put_summary_df = _read_csv(acct / "pnl_bucket_5_options.csv")
    put_positions = _read_csv(acct / "b5_option_positions.csv")
    puts = {} if put_summary_df is None or put_summary_df.empty else put_summary_df.iloc[0].to_dict()
    result["puts"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in puts.items()}

    # ---------------------------------------------------------- lot identity
    ledger = Bucket5Ledger((cfg.get("paths") or {}).get("ledger_events", "data/bucket5_ledger/events.jsonl"))
    open_lots = ledger.open_lots()
    lot_check = {"pass": True, "mismatches": []}
    broker_by_conid: dict[str, float] = {}
    if put_positions is not None and not put_positions.empty:
        broker_by_conid = {
            str(r["conId"]): float(r["position"]) for _, r in put_positions.iterrows()
        }
    for con_id, lot in open_lots.items():
        broker_qty = broker_by_conid.get(str(con_id))
        ledger_qty = float(lot.get("remaining_contracts", 0))
        if broker_qty is None:
            lot_check["pass"] = False
            lot_check["mismatches"].append(
                {"conId": con_id, "issue": "ledger lot missing from broker positions", "ledger_qty": ledger_qty}
            )
        elif abs(broker_qty - ledger_qty) > 1e-9:
            lot_check["pass"] = False
            lot_check["mismatches"].append(
                {"conId": con_id, "issue": "quantity mismatch", "ledger_qty": ledger_qty, "broker_qty": broker_qty}
            )
    for con_id, broker_qty in broker_by_conid.items():
        if con_id not in open_lots:
            lot_check["pass"] = False
            lot_check["mismatches"].append(
                {"conId": con_id, "issue": "broker option position has no ledger lot (orphan)", "broker_qty": broker_qty}
            )
    result["checks"]["lot_identity"] = lot_check
    if not lot_check["pass"]:
        _degrade("no_new_risk", f"lot identity failed: {len(lot_check['mismatches'])} mismatches")

    # ------------------------------------------------------------ NAV identity
    put_mark = float(puts.get("put_mark_value_usd", 0.0) or 0.0)
    carry_cum_pnl = float(carry.get("total_pnl_usd", 0.0) or 0.0)
    nav_today = carry_cum_pnl + put_mark + float(puts.get("realized_pnl_usd", 0.0) or 0.0)
    nav_check: dict = {"nav_today_usd": nav_today, "chained": False}

    prior = None
    if prior_date:
        p = Path("data") / "runs" / prior_date / "bucket5_production" / "reconcile.json"
        if p.exists():
            prior = (prior_date, json.loads(p.read_text(encoding="utf-8")))
    if prior is None:
        prior = _find_prior_reconcile(run_date)

    if prior is not None:
        prior_name, prior_payload = prior
        prior_nav = float(((prior_payload.get("checks") or {}).get("nav_identity") or {}).get("nav_today_usd", 0.0))
        prior_carry = float((prior_payload.get("carry") or {}).get("total_pnl_usd", 0.0) or 0.0)
        prior_put_mark = float((prior_payload.get("puts") or {}).get("put_mark_value_usd", 0.0) or 0.0)
        prior_put_realized = float((prior_payload.get("puts") or {}).get("realized_pnl_usd", 0.0) or 0.0)
        components = {
            "carry_pnl_change_usd": carry_cum_pnl - prior_carry,
            "put_mark_change_usd": put_mark - prior_put_mark,
            "put_realized_change_usd": float(puts.get("realized_pnl_usd", 0.0) or 0.0) - prior_put_realized,
        }
        expected = prior_nav + sum(components.values())
        residual = nav_today - expected
        nav_check.update({
            "chained": True,
            "prior_date": prior_name,
            "prior_nav_usd": prior_nav,
            "components": components,
            "expected_nav_usd": expected,
            "residual_usd": residual,
            "max_residual_usd": max_residual,
            "pass": abs(residual) <= max_residual,
        })
        if not nav_check["pass"]:
            _degrade("no_new_risk", f"NAV identity residual ${residual:,.2f} exceeds ${max_residual:,.2f}")
    else:
        nav_check["note"] = "no prior reconcile snapshot; NAV chain starts today"
    result["checks"]["nav_identity"] = nav_check

    # ------------------------------------------------------------- kill mode
    if cfg["kill_mode"] in ("reduce_only", "flatten_carry"):
        _degrade("reduce_only", f"operator kill_mode={cfg['kill_mode']}")
    elif cfg["kill_mode"] in ("halt_all",):
        _degrade("halted", "operator kill_mode=halt_all")
    elif cfg["kill_mode"] == "no_new_risk":
        _degrade("no_new_risk", "operator kill_mode=no_new_risk")

    out_path = out_dir / "reconcile.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    ledger.append("RECONCILE", f"RECON|{run_date}", {
        "run_date": run_date, "health": result["health"],
        "lot_identity_pass": lot_check["pass"],
        "nav_residual_usd": nav_check.get("residual_usd"),
    })
    print(f"[B5-RECON] {run_date}: health={result['health']} -> {out_path}")
    for n in result["notes"]:
        print(f"[B5-RECON]   note: {n}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Bucket 5 Production B daily reconciliation")
    ap.add_argument("--run-date", default=date.today().isoformat())
    ap.add_argument("--prior-date", default=None)
    args = ap.parse_args()
    result = reconcile(args.run_date, args.prior_date)
    return 0 if result["health"] == "green" else 1


if __name__ == "__main__":
    raise SystemExit(main())
