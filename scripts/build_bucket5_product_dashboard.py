#!/usr/bin/env python3
"""
Build Bucket 5 product dashboard JSON (SPX-0DTE-aligned).

Writes ``risk_dashboard/data/bucket5_product.json`` and optionally copies to
etf-dashboard.

Usage::

    python scripts/build_bucket5_product_dashboard.py
    python scripts/build_bucket5_product_dashboard.py --quick
    python scripts/build_bucket5_product_dashboard.py --copy-etf-dashboard
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_backtest_api import (  # noqa: E402
    list_presets,
    resolve_borrow_rates,
)
from scripts.bucket5_data import INCEPTION, load_series, load_vol_panel  # noqa: E402
from scripts.bucket5_insurance_bt import (  # noqa: E402
    EXTENDED_START,
    crash_payoff,
    production_config,
    run_insurance,
    summarize,
)
from scripts.bucket5_product_export import (  # noqa: E402
    build_live_days,
    build_run_payload,
)
from scripts.bucket5_put_overlay import load_spx_spot  # noqa: E402
from scripts.bucket5_spy_put_grid_bt import (  # noqa: E402
    _attach_puts,
    _build_idle_base,
    _scale_rungs,
)

OUT = REPO / "risk_dashboard" / "data" / "bucket5_product.json"


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _run_standard(
    *,
    preset_label: str,
    era: str,
    panel: pd.DataFrame,
    spx: pd.Series,
    iv: pd.Series,
) -> tuple[dict, dict, dict, dict]:
    cfg = production_config()
    u_borrow, s_borrow = resolve_borrow_rates(panel_index=panel.index)
    cfg = replace(cfg, borrow_uvix_annual=u_borrow, borrow_svix_annual=s_borrow)
    res = run_insurance(panel, spx, iv, cfg)
    summary = summarize(res, spx, panel)
    summary["realized_$"] = float(res["ladder"].attrs.get("realized_total", 0.0))
    summary["redeploy_extra_$"] = float(res["bt"]["redeploy_extra"].iloc[-1])
    crash = crash_payoff(res, spx, panel)
    crash_out = {
        "crash_mild_-20%": crash.get("mild_-20%"),
        "crash_severe_-30%": crash.get("severe_-30%"),
        "crash_volmageddon_-40%": crash.get("volmageddon_-40%"),
    }
    for k in ("crash_mild_-20%", "crash_severe_-30%", "crash_volmageddon_-40%"):
        summary.pop(k, None)
    meta = {
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "n_days": int(len(res["bt"])),
        "synthetic_days": int(panel["synthetic"].sum()) if "synthetic" in panel.columns else 0,
        "era": era,
        "rebalances": int(res["bt"].attrs.get("rebalances", len(res["rebal"]))),
        "pricing_mode": (
            "bs_skew"
            if ("synthetic" in panel.columns and int(panel["synthetic"].sum()) > len(panel) * 0.5)
            else "theta_mid"
        ),
        "preset": "B_production",
        "label_note": preset_label,
    }
    assumptions = {
        "borrow_uvix_annual": u_borrow,
        "borrow_svix_annual": s_borrow,
        "uvix_slip_bps": cfg.uvix_slip_bps,
        "fee_bps": cfg.fee_bps,
        "tbill_rate": cfg.tbill_rate,
        "sleeve_frac": cfg.sleeve_frac,
    }
    return res, summary, crash_out, {"meta": meta, "assumptions": assumptions}


def _run_spy_tilt(
    *,
    spy_idle_frac: float,
    put_mult: float,
    panel: pd.DataFrame,
    spx: pd.Series,
    iv: pd.Series,
    spy: pd.Series,
) -> tuple[dict, dict, dict, dict]:
    cfg = production_config()
    u_borrow, s_borrow = resolve_borrow_rates(panel_index=panel.index)
    cfg = replace(
        cfg,
        borrow_uvix_annual=u_borrow,
        borrow_svix_annual=s_borrow,
        rungs=_scale_rungs(cfg.rungs, put_mult),
    )
    base = _build_idle_base(panel, spy, spy_idle_frac=spy_idle_frac, cfg=cfg)
    res = _attach_puts(panel, spx, iv, base, cfg)
    summary = summarize(res, spx, panel)
    summary["realized_$"] = float(res["ladder"].attrs.get("realized_total", 0.0))
    summary["redeploy_extra_$"] = float(res["bt"]["redeploy_extra"].iloc[-1])
    summary["spy_idle_frac"] = spy_idle_frac
    summary["put_mult"] = put_mult
    crash = crash_payoff(res, spx, panel)
    crash_out = {
        "crash_mild_-20%": crash.get("mild_-20%"),
        "crash_severe_-30%": crash.get("severe_-30%"),
        "crash_volmageddon_-40%": crash.get("volmageddon_-40%"),
    }
    for k in ("crash_mild_-20%", "crash_severe_-30%", "crash_volmageddon_-40%"):
        summary.pop(k, None)
    meta = {
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "n_days": int(len(res["bt"])),
        "synthetic_days": int(panel["synthetic"].sum()) if "synthetic" in panel.columns else 0,
        "era": "extended",
        "rebalances": int(res["bt"].attrs.get("rebalances", len(res["rebal"]))),
        "pricing_mode": "mixed",
        "preset": f"spy{int(spy_idle_frac*100):02d}_puts{put_mult:.2f}x",
        "label_note": f"{int(spy_idle_frac*100)}% SPY idle collateral + puts {put_mult:.2f}×",
    }
    assumptions = {
        "borrow_uvix_annual": u_borrow,
        "borrow_svix_annual": s_borrow,
        "uvix_slip_bps": cfg.uvix_slip_bps,
        "fee_bps": cfg.fee_bps,
        "tbill_rate": cfg.tbill_rate,
        "sleeve_frac": cfg.sleeve_frac,
        "spy_idle_frac": spy_idle_frac,
        "put_mult": put_mult,
    }
    return res, summary, crash_out, {"meta": meta, "assumptions": assumptions}


def build_payload(*, quick: bool = False, run_date: str | None = None) -> dict:
    run_date = run_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    t0 = time.time()
    runs: list[dict] = []

    # Primary: production extended
    print("[b5-product] loading extended panel ...")
    panel_x = load_vol_panel(start=EXTENDED_START, use_synthetic=True)
    spx_x = load_spx_spot(panel_x.index.min().strftime("%Y-%m-%d"))
    iv_x = (panel_x["vix"] / 100.0).rename("iv")
    print("[b5-product] run B_extended (primary) ...")
    res, summary, crash, pack = _run_standard(
        preset_label=list_presets().get("B_production", "B production"),
        era="extended",
        panel=panel_x,
        spx=spx_x,
        iv=iv_x,
    )
    primary = build_run_payload(
        run_id="B_extended",
        label="B production ★ (extended)",
        res=res,
        panel=panel_x,
        summary=summary,
        crash=crash,
        meta=pack["meta"],
        assumptions=pack["assumptions"],
        include_guide=True,
    )
    runs.append(primary)
    regime_panels = primary["regime_panels"]

    # Secondary: live-era production
    print("[b5-product] loading live panel ...")
    panel_l = load_vol_panel(start=INCEPTION, use_synthetic=False)
    spx_l = load_spx_spot(panel_l.index.min().strftime("%Y-%m-%d"))
    iv_l = (panel_l["vix"] / 100.0).rename("iv")
    print("[b5-product] run B_live ...")
    res_l, summary_l, crash_l, pack_l = _run_standard(
        preset_label=list_presets().get("B_production", "B production"),
        era="live",
        panel=panel_l,
        spx=spx_l,
        iv=iv_l,
    )
    runs.append(
        build_run_payload(
            run_id="B_live",
            label="B production ★ (live era)",
            res=res_l,
            panel=panel_l,
            summary=summary_l,
            crash=crash_l,
            meta=pack_l["meta"],
            assumptions=pack_l["assumptions"],
            include_guide=False,
        )
    )

    if not quick:
        print("[b5-product] run research tilt 30% SPY idle + puts 2.00x ...")
        spy = load_series("SPY", start=panel_x.index.min().strftime("%Y-%m-%d")).rename("spy")
        res_t, summary_t, crash_t, pack_t = _run_spy_tilt(
            spy_idle_frac=0.30,
            put_mult=2.0,
            panel=panel_x,
            spx=spx_x,
            iv=iv_x,
            spy=spy,
        )
        runs.append(
            build_run_payload(
                run_id="R_spy30_puts2x",
                label="Research: 30% SPY idle + puts 2.00×",
                res=res_t,
                panel=panel_x,
                summary=summary_t,
                crash=crash_t,
                meta=pack_t["meta"],
                assumptions=pack_t["assumptions"],
                include_guide=False,
            )
        )

    live = build_live_days(run_date)
    payload = {
        "schema": "bucket5_product_dashboard.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "primary_run_id": "B_extended",
        "runs": runs,
        "live": live,
        "live_enabled": bool(live.get("days")),
        "regime_panels": regime_panels,
        "presets": list_presets(),
        "lab_command": "streamlit run scripts/bucket5_lab.py",
        "repo": "ls-algo",
        "build_seconds": round(time.time() - t0, 1),
        "notes": {
            "live_vs_research": (
                "Daily · live tags show the GTP volatility-ETP sleeve (~0.25% gross). "
                "Overview / Regime / Daily backtest marks are the insurance product research stack."
            ),
        },
    }
    return _sanitize(payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Skip research tilt run (faster CI)")
    ap.add_argument("--run-date", default=None)
    ap.add_argument("--copy-etf-dashboard", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    payload = build_payload(quick=args.quick, run_date=args.run_date)
    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[b5-product] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {payload['build_seconds']}s)")
    for r in payload["runs"]:
        m = r["summary"]
        print(
            f"  {r['id']}: CAGR={100*float(m.get('combined_CAGR') or 0):.2f}% "
            f"Sharpe={float(m.get('combined_Sharpe') or 0):.2f} "
            f"MaxDD={100*float(m.get('combined_MaxDD') or 0):.2f}% "
            f"days={len(r['daily'])}"
        )

    if args.copy_etf_dashboard:
        for root in (REPO.parent / "etf-dashboard", Path.home() / "Projects" / "quant" / "etf-dashboard"):
            if root.is_dir():
                dest = root / "data" / "bucket5_product.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, dest)
                print(f"[b5-product] copied -> {dest}")
                # also keep legacy filename as alias for one release
                legacy = root / "data" / "bucket5_insurance_backtest.json"
                shutil.copy2(out, legacy)
                print(f"[b5-product] copied legacy alias -> {legacy}")
                _sync_product_ui(root)
    return 0


def _sync_product_ui(etf_root: Path) -> None:
    """Copy canonical B5 Product JS/CSS into etf-dashboard assets/."""
    pairs = [
        (REPO / "site" / "assets" / "js" / "bucket5_product.js", etf_root / "assets" / "bucket5_product.js"),
        (REPO / "site" / "assets" / "css" / "bucket5_product.css", etf_root / "assets" / "bucket5_product.css"),
    ]
    for src, dest in pairs:
        if not src.is_file():
            print(f"[b5-product] skip UI sync (missing {src})")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"[b5-product] synced UI -> {dest}")


if __name__ == "__main__":
    raise SystemExit(main())
