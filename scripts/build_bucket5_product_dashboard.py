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
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_backtest_api import (  # noqa: E402
    list_presets,
    resolve_borrow_rates,
)
from scripts.bucket5_data import load_vol_panel  # noqa: E402
from scripts.bucket5_insurance_bt import (  # noqa: E402
    EXTENDED_START,
    crash_payoff,
    production_config,
    reverse_solve_put_contracts,
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

    # Publish one canonical full-history run; research/live comparison curves stay out of B5.
    print("[b5-product] loading extended panel ...")
    panel_x = load_vol_panel(start=EXTENDED_START, use_synthetic=True)
    spx_x = load_spx_spot(panel_x.index.min().strftime("%Y-%m-%d"))
    iv_x = (panel_x["vix"] / 100.0).rename("iv")
    print("[b5-product] run B_full_2008_present (primary) ...")
    res, summary, crash, pack = _run_standard(
        preset_label=list_presets().get("B_production", "B production"),
        era="extended",
        panel=panel_x,
        spx=spx_x,
        iv=iv_x,
    )
    primary = build_run_payload(
        run_id="B_full_2008_present",
        label="B production FULL backtest (2008-present, 2x puts)",
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

    strategy_doc = yaml.safe_load(
        (REPO / "config" / "strategy_config.yml").read_text(encoding="utf-8")
    ) or {}
    strategy_cfg = strategy_doc.get("strategy") or {}
    account_nav = float(strategy_cfg.get("capital_usd", 1_200_000.0))
    total_target_gross = account_nav * float(strategy_cfg.get("gross_leverage", 4.0))
    b5_cfg = (
        (((strategy_doc.get("portfolio") or {}).get("sleeves") or {}).get("inverse_decay_bucket4") or {})
        .get("rules", {})
        .get("volatility_etp_bucket5", {})
    ) or {}
    cfg = production_config()
    b5_pair_gross = total_target_gross * float(b5_cfg.get("target_weight", 0.01))
    effective_b5_nav = b5_pair_gross / max(float(cfg.sleeve_frac), 1e-12)
    last_date = panel_x.index.max()
    put_sizing = reverse_solve_put_contracts(
        equity_usd=effective_b5_nav,
        spx_spot=float(spx_x.reindex(panel_x.index).ffill().loc[last_date]),
        atm_iv=float(iv_x.loc[last_date]),
        rungs=cfg.rungs,
        hedge_budget=cfg.hedge_budget,
        ratio=float(panel_x.loc[last_date, "ratio"]),
        vix=float(panel_x.loc[last_date, "vix"]),
    )
    put_sizing["as_of"] = str(last_date.date())
    put_sizing["account_nav_usd"] = account_nav
    put_sizing["b5_pair_gross_usd"] = b5_pair_gross
    put_sizing["effective_b5_nav_usd"] = effective_b5_nav
    put_sizing["quote_note"] = (
        "Modeled sizing; replace modeled prices with executable SPX asks before trading."
    )
    primary["put_sizing"] = put_sizing

    live = build_live_days(run_date)
    payload = {
        "schema": "bucket5_product_dashboard.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "primary_run_id": "B_full_2008_present",
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
                "Daily live tags show the independent GTP volatility-ETP sleeve (1% true pair gross). "
                "All charts use the single Production B full-history backtest from 2008 to present."
            ),
        },
    }
    return _sanitize(payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Compatibility flag; the single full run is always built")
    ap.add_argument("--run-date", default=None)
    ap.add_argument("--copy-etf-dashboard", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    payload = build_payload(quick=args.quick, run_date=args.run_date)
    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically. Large dashboard JSON files can be observed by local servers/
    # sync tools; replacing a completed temp file avoids partial reads and Windows locks.
    tmp_out = out.with_name(out.name + ".tmp")
    tmp_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_out.replace(out)
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
