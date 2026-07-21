"""Bucket 5 Production B — live-panel JSON generator for the risk dashboard.

Assembles the real B5 book (mode, health, carry positions/targets, option lots,
reconciliation status) into ``bucket5_live.json``. The dashboard build
(``risk_dashboard/build_site.py``) merges it into ``latest.json`` as the
``bucket5_live`` aux panel, and the B5 Product tab renders it above the
research backtest so the live book and the backtested product are shown
side by side.

EOD mode (default) reads the accounting outputs and the B5 manifest/ledger —
no broker connection required:

    python scripts/bucket5_monitor.py --run-date 2026-07-21

Optional intraday refresh with live IBKR marks (positions filtered to UVIX/
SVIX and ledger conIds only; read-only):

    python scripts/bucket5_monitor.py --run-date 2026-07-21 --live
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bucket5_ledger import Bucket5Ledger  # noqa: E402
from scripts.bucket5_policy import load_b5_config  # noqa: E402

CARRY_SYMBOLS = ("UVIX", "SVIX")


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_run_with(fname: str, run_date: str) -> Path | None:
    runs = Path("data") / "runs"
    if not runs.exists():
        return None
    for d in sorted((p for p in runs.iterdir() if p.is_dir() and p.name <= run_date), reverse=True):
        p = d / "bucket5_production" / fname
        if p.exists():
            return p
    return None


def build_live_payload(run_date: str, config_path: str | Path | None = None) -> dict:
    cfg = load_b5_config(config_path)
    cap = cfg.get("capital") or {}
    acct = Path("data") / "runs" / run_date / "accounting"

    manifest = _read_json(Path("data") / "runs" / run_date / "bucket5_production" / "decision_manifest.json")
    if manifest is None:
        mp = _latest_run_with("decision_manifest.json", run_date)
        manifest = _read_json(mp) if mp else None
    reconcile = _read_json(Path("data") / "runs" / run_date / "bucket5_production" / "reconcile.json")
    if reconcile is None:
        rp = _latest_run_with("reconcile.json", run_date)
        reconcile = _read_json(rp) if rp else None

    # ------------------------------------------------------------- carry book
    pnl_b5 = _read_csv(acct / "pnl_bucket_5.csv")
    exp_b5 = _read_csv(acct / "net_exposure_bucket_5_detail.csv")
    if exp_b5 is None:
        exp_b5 = _read_csv(acct / "net_exposure_bucket_5.csv")
    carry_positions = []
    if exp_b5 is not None and not exp_b5.empty:
        for _, r in exp_b5.iterrows():
            carry_positions.append({
                "symbol": str(r.get("symbol", r.get("underlying", ""))),
                "net_notional_usd": float(r.get("net_notional_usd", 0.0) or 0.0),
                "gross_notional_usd": float(r.get("gross_notional_usd", 0.0) or 0.0),
            })
    carry = {
        "positions": carry_positions,
        "gross_notional_usd": float(sum(abs(p["gross_notional_usd"]) for p in carry_positions)),
        "net_notional_usd": float(sum(p["net_notional_usd"] for p in carry_positions)),
        "cum_pnl_usd": float(pnl_b5["total_pnl"].sum()) if pnl_b5 is not None and "total_pnl" in (pnl_b5.columns if pnl_b5 is not None else []) else None,
        "borrow_fees_usd": float(pnl_b5["borrow_fees"].sum()) if pnl_b5 is not None and "borrow_fees" in pnl_b5.columns else None,
        "policy_targets": (manifest or {}).get("carry"),
    }

    # ------------------------------------------------------------- put book
    ledger = Bucket5Ledger((cfg.get("paths") or {}).get("ledger_events", "data/bucket5_ledger/events.jsonl"))
    lots = []
    for lot in ledger.open_lots().values():
        lots.append({
            "conId": lot.get("conId"),
            "local_symbol": lot.get("local_symbol"),
            "expiry": lot.get("expiry"),
            "strike": lot.get("strike"),
            "right": lot.get("right"),
            "remaining_contracts": int(lot.get("remaining_contracts", 0)),
            "entry_contracts": int(lot.get("entry_contracts", 0)),
            "cost_basis_usd": float(lot.get("cost_basis_usd", 0.0)),
            "realized_cash_usd": float(lot.get("realized_cash_usd", 0.0)),
            "peak_mult": float(lot.get("peak_mult", 1.0)),
            "profit_tiers_fired": lot.get("profit_tiers_fired", []),
            "vix_tiers_fired": lot.get("vix_tiers_fired", []),
        })
    put_summary = _read_csv(acct / "pnl_bucket_5_options.csv")
    puts = {
        "lots": lots,
        "open_contracts": int(sum(l["remaining_contracts"] for l in lots)),
        "accounting": (put_summary.iloc[0].to_dict() if put_summary is not None and not put_summary.empty else None),
    }

    # ------------------------------------------------------------- intents
    intents_csv = Path("data") / "runs" / run_date / "bucket5_production" / "option_intents.csv"
    intents_df = _read_csv(intents_csv)
    intents = intents_df.to_dict("records") if intents_df is not None else []

    health = (reconcile or {}).get("health") or (manifest or {}).get("health") or "unknown"
    payload = {
        "schema": "bucket5_live.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "mode": cfg["mode"],
        "kill_mode": cfg["kill_mode"],
        "health": health,
        "strategy_version": str(cfg.get("strategy_version", "B5PROD-1")),
        "capital": {
            "b5_allocated_nav": float(cap.get("b5_allocated_nav", 0.0)),
            "ramp_factor": float(cap.get("ramp_factor", 0.0)),
            "effective_b5_nav": float(cap.get("b5_allocated_nav", 0.0)) * float(cap.get("ramp_factor", 0.0)),
            "max_carry_gross_usd": float(cap.get("max_carry_gross_usd", 0.0)),
        },
        "signals": (manifest or {}).get("signals"),
        "carry": carry,
        "puts": puts,
        "option_intents_today": intents,
        "reconcile": {
            "health": (reconcile or {}).get("health"),
            "notes": (reconcile or {}).get("notes", []),
            "nav_identity": ((reconcile or {}).get("checks") or {}).get("nav_identity"),
            "lot_identity_pass": (((reconcile or {}).get("checks") or {}).get("lot_identity") or {}).get("pass"),
        },
    }
    return payload


def refresh_live_marks(payload: dict, cfg: dict) -> dict:
    """Optional: overwrite carry marks with live IBKR positions (read-only)."""
    try:
        from ib_insync import IB
    except ImportError:
        print("[B5-MON] ib_insync not available; skipping live marks.")
        return payload
    import yaml

    scfg = yaml.safe_load(Path("config/strategy_config.yml").read_text(encoding="utf-8")) or {}
    ib_cfg = scfg.get("ibkr") or {}
    ib = IB()
    try:
        ib.connect(str(ib_cfg.get("host", "127.0.0.1")), int(ib_cfg.get("port", 7496)), clientId=87)
        ledger_conids = {str(l["conId"]) for l in payload["puts"]["lots"]}
        carry_rows, put_rows = [], []
        for p in ib.positions():
            c = p.contract
            if c.secType == "STK" and c.symbol.upper() in CARRY_SYMBOLS:
                carry_rows.append({
                    "symbol": c.symbol.upper(),
                    "position": float(p.position),
                    "avg_cost": float(p.avgCost),
                })
            elif c.secType == "OPT" and str(c.conId) in ledger_conids:
                put_rows.append({
                    "conId": str(c.conId),
                    "position": float(p.position),
                    "avg_cost": float(p.avgCost),
                })
        payload["live_broker"] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "carry_positions": carry_rows,
            "put_positions": put_rows,
        }
    except Exception as ex:
        print(f"[B5-MON] live refresh failed ({ex}); EOD payload retained.")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Bucket 5 Production B live-panel generator")
    ap.add_argument("--run-date", default=date.today().isoformat())
    ap.add_argument("--live", action="store_true", help="Also poll IBKR for live positions (read-only)")
    ap.add_argument("--out", default=None, help="Output JSON (default from config paths.live_panel_json)")
    args = ap.parse_args()

    cfg = load_b5_config()
    payload = build_live_payload(args.run_date)
    if args.live:
        payload = refresh_live_marks(payload, cfg)

    out = Path(args.out or (cfg.get("paths") or {}).get("live_panel_json", "risk_dashboard/data/bucket5_live.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Per-run-date copy so dated dashboard snapshots pick it up.
    dated = Path("data") / "runs" / args.run_date / "bucket5_live.json"
    dated.parent.mkdir(parents=True, exist_ok=True)
    dated.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[B5-MON] mode={payload['mode']} health={payload['health']} -> {out} (+ {dated})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
