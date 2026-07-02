"""
Bucket 5 insurance backtest API — single entry for lab, dashboard, and CI.

Usage::

    from scripts.bucket5_backtest_api import run_backtest, list_presets

    result = run_backtest(preset="B_production", era="live")
    result = run_backtest(params={"sleeve_frac": 0.20, ...}, era="extended")
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_data import (  # noqa: E402
    DEFAULT_BORROW_SVIX,
    DEFAULT_BORROW_UVIX,
    INCEPTION,
    load_vol_panel,
)
from scripts.bucket5_insurance_bt import (  # noqa: E402
    BackspreadHedge,
    EXTENDED_START as INSURANCE_EXTENDED_START,
    HedgeBudgetPolicy,
    InsuranceConfig,
    LadderRung,
    MonetizeConfig,
    RedeployPolicy,
    RegimePolicy,
    build_ladder,
    production_config,
    run_insurance,
    summarize,
)
from scripts.bucket5_put_overlay import load_spx_spot  # noqa: E402

Era = Literal["live", "extended"]
CACHE_DIR = REPO / "data" / "cache" / "bucket5" / "backtests"
PRESETS_PATH = REPO / "config" / "bucket5_presets.yml"
ENGINE_FILES = (
    REPO / "scripts" / "bucket5_insurance_bt.py",
    REPO / "scripts" / "bucket5_backtest_api.py",
)


def _engine_sha() -> str:
    h = hashlib.sha256()
    for p in ENGINE_FILES:
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _etf_dashboard_roots() -> list[Path]:
    roots = [
        REPO.parent / "etf-dashboard",
        REPO / "etf-dashboard",
        Path.home() / "Projects" / "quant" / "etf-dashboard",
    ]
    return [p for p in roots if p.is_dir()]


def load_borrow_history_series(symbol: str) -> pd.Series | None:
    """Load daily borrow feerate (annual fraction) from etf-dashboard borrow_history.json."""
    sym = symbol.upper()
    for root in _etf_dashboard_roots():
        path = root / "data" / "borrow_history.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec = (payload.get("symbols") or {}).get(sym) or (payload.get("by_symbol") or {}).get(sym)
        if not rec:
            continue
        rows = rec.get("history") or rec.get("rows") or []
        if not rows:
            continue
        dates, rates = [], []
        for row in rows:
            d = row.get("date") or row.get("asof")
            rate = row.get("feerate") or row.get("borrow_annual") or row.get("rate")
            if d is None or rate is None:
                continue
            r = float(rate)
            if r > 1.0:
                r /= 100.0
            dates.append(pd.Timestamp(d))
            rates.append(r)
        if dates:
            s = pd.Series(rates, index=pd.DatetimeIndex(dates)).sort_index()
            return s.rename(sym)
    return None


def resolve_borrow_rates(
    *,
    borrow_mode: str = "fixed",
    borrow_uvix: float | None = None,
    borrow_svix: float | None = None,
    panel_index: pd.DatetimeIndex | None = None,
) -> tuple[float, float]:
    if borrow_mode == "etf_history" and panel_index is not None:
        u = load_borrow_history_series("UVIX")
        s = load_borrow_history_series("SVIX")
        if u is not None and s is not None:
            u_a = float(u.reindex(panel_index, method="ffill").dropna().mean())
            s_a = float(s.reindex(panel_index, method="ffill").dropna().mean())
            if np.isfinite(u_a) and np.isfinite(s_a):
                return u_a, s_a
    return (
        borrow_uvix if borrow_uvix is not None else DEFAULT_BORROW_UVIX,
        borrow_svix if borrow_svix is not None else DEFAULT_BORROW_SVIX,
    )


def _dc_to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return {f.name: _dc_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_dc_to_dict(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _parse_optional_dc(cls, raw: Any):
    if raw is None:
        return None
    if isinstance(raw, cls):
        return raw
    if not isinstance(raw, dict):
        return raw
    kwargs = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        if is_dataclass(f.type):
            kwargs[f.name] = _parse_optional_dc(f.type, val)
            continue
        if f.name in ("profit_tiers", "vix_tiers") and isinstance(val, list):
            kwargs[f.name] = tuple(tuple(x) for x in val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def config_from_dict(params: dict[str, Any]) -> InsuranceConfig:
    p = dict(params)
    if "rungs_from_ladder" in p:
        spec = p.pop("rungs_from_ladder")
        p["rungs"] = build_ladder(
            [tuple(x) for x in spec["strikes"]],
            float(spec["total_per_roll"]),
        )
    if "rungs" in p and p["rungs"] and isinstance(p["rungs"][0], dict):
        p["rungs"] = [LadderRung(**r) for r in p["rungs"]]
    for key, cls in (
        ("regime", RegimePolicy),
        ("hedge_budget", HedgeBudgetPolicy),
        ("monetize", MonetizeConfig),
        ("redeploy", RedeployPolicy),
        ("backspread", BackspreadHedge),
    ):
        if key in p:
            p[key] = _parse_optional_dc(cls, p[key])
    allowed = {f.name for f in fields(InsuranceConfig)}
    return InsuranceConfig(**{k: v for k, v in p.items() if k in allowed})


def load_presets_yaml() -> dict[str, Any]:
    if not PRESETS_PATH.is_file():
        return {"presets": {}, "defaults": {}}
    return yaml.safe_load(PRESETS_PATH.read_text(encoding="utf-8")) or {}


def list_presets() -> dict[str, str]:
    doc = load_presets_yaml()
    return {pid: spec.get("label", pid) for pid, spec in (doc.get("presets") or {}).items()}


def config_from_preset(preset_id: str) -> InsuranceConfig:
    doc = load_presets_yaml()
    spec = (doc.get("presets") or {}).get(preset_id)
    if spec is None:
        raise KeyError(f"Unknown preset: {preset_id}")
    if spec.get("preset_factory") == "production_config":
        return production_config()
    defaults = doc.get("defaults") or {}
    merged = {**defaults, **{k: v for k, v in spec.items() if k not in ("label", "preset_factory")}}
    return config_from_dict(merged)


def _era_start(era: Era) -> tuple[str, bool]:
    if era == "extended":
        return INSURANCE_EXTENDED_START, True
    return INCEPTION, False


def _cache_key(cfg: InsuranceConfig, era: Era, end: str | None, methodology: dict) -> str:
    payload = {
        "cfg": _dc_to_dict(cfg),
        "era": era,
        "end": end,
        "methodology": methodology,
        "engine": _engine_sha(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _downsample_series(s: pd.Series, max_points: int = 800) -> list[list[Any]]:
    s = s.dropna()
    if len(s) <= max_points:
        idx = s.index
    else:
        step = max(1, len(s) // max_points)
        idx = s.index[::step]
        if s.index[-1] not in idx:
            idx = idx.union(pd.Index([s.index[-1]]))
    return [[d.strftime("%Y-%m-%d"), round(float(s.loc[d]), 4)] for d in idx]


def _pricing_mode(panel: pd.DataFrame) -> str:
    if "synthetic" not in panel.columns:
        return "theta_mid"
    synth = int(panel["synthetic"].sum())
    if synth > len(panel) * 0.5:
        return "bs_skew"
    return "theta_mid"


def run_backtest(
    params: dict[str, Any] | InsuranceConfig | None = None,
    *,
    preset: str | None = None,
    era: Era = "live",
    end: str | None = None,
    refresh_data: bool = False,
    include_series: bool = True,
    max_series_points: int = 800,
    save: bool = False,
    run_id: str | None = None,
    use_cache: bool = True,
    methodology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    methodology = methodology or {}
    if preset and params is not None:
        raise ValueError("Specify preset OR params, not both")
    if preset:
        cfg = config_from_preset(preset)
    elif isinstance(params, InsuranceConfig):
        cfg = params
    elif params:
        cfg = config_from_dict(params)
    else:
        cfg = production_config()

    start, use_synthetic = _era_start(era)
    cache_key = _cache_key(cfg, era, end, methodology)
    cache_metrics = CACHE_DIR / f"{cache_key}_metrics.json"
    cache_series = CACHE_DIR / f"{cache_key}_series.parquet"

    if use_cache and cache_metrics.is_file() and (not include_series or cache_series.is_file()):
        out = json.loads(cache_metrics.read_text(encoding="utf-8"))
        if include_series and cache_series.is_file():
            ser = pd.read_parquet(cache_series)
            out["series"] = {c: _downsample_series(ser[c], max_series_points) for c in ser.columns}
        out["cached"] = True
        return out

    panel = load_vol_panel(start=start, end=end, use_synthetic=use_synthetic, refresh=refresh_data)
    spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"), end)
    iv = (panel["vix"] / 100.0).rename("iv")

    u_borrow, s_borrow = resolve_borrow_rates(
        borrow_mode=str(methodology.get("borrow_mode", "fixed")),
        borrow_uvix=methodology.get("borrow_uvix"),
        borrow_svix=methodology.get("borrow_svix"),
        panel_index=panel.index,
    )
    stress_mult = float(methodology.get("borrow_stress_mult", 1.0))
    if stress_mult != 1.0:
        u_borrow *= stress_mult
        s_borrow *= stress_mult

    cfg = InsuranceConfig(
        **{
            **{f.name: getattr(cfg, f.name) for f in fields(InsuranceConfig)},
            "borrow_uvix_annual": u_borrow,
            "borrow_svix_annual": s_borrow,
        }
    )

    res = run_insurance(panel, spx, iv, cfg)
    metrics = summarize(res, spx, panel)
    metrics["realized_$"] = float(res["ladder"].attrs.get("realized_total", 0.0))
    metrics["redeploy_extra_$"] = float(res["bt"]["redeploy_extra"].iloc[-1])
    metrics["hedge_kind"] = cfg.hedge_kind
    metrics["dynamic_budget"] = cfg.hedge_budget is not None and bool(cfg.hedge_budget.enabled)

    bt = res["bt"]
    synth_days = int(panel["synthetic"].sum()) if "synthetic" in panel.columns else 0

    meta = {
        "start": str(panel.index.min().date()),
        "end": str(panel.index.max().date()),
        "n_days": int(len(bt)),
        "synthetic_days": synth_days,
        "era": era,
        "rebalances": int(bt.attrs.get("rebalances", 0)),
        "pricing_mode": _pricing_mode(panel),
        "borrow_uvix_annual": u_borrow,
        "borrow_svix_annual": s_borrow,
        "cache_key": cache_key,
        "engine_sha": _engine_sha(),
    }

    crash = {
        "crash_mild_-20%": metrics.pop("crash_mild_-20%", None),
        "crash_severe_-30%": metrics.pop("crash_severe_-30%", None),
        "crash_volmageddon_-40%": metrics.pop("crash_volmageddon_-40%", None),
    }

    series_cols = [
        "combined_equity", "drawdown", "ratio", "rho", "gross_frac",
        "sleeve_equity", "base_equity", "put_mtm", "realized_cum", "redeploy_extra",
    ]
    ser_df = bt[[c for c in series_cols if c in bt.columns]].copy()
    ser_df.attrs = {}

    result: dict[str, Any] = {
        "schema": "bucket5_backtest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "config": _dc_to_dict(cfg),
        "meta": meta,
        "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
        "crash": crash,
        "assumptions": {
            "borrow_uvix_annual": u_borrow,
            "borrow_svix_annual": s_borrow,
            "uvix_slip_bps": cfg.uvix_slip_bps,
            "fee_bps": cfg.fee_bps,
            "tbill_rate": cfg.tbill_rate,
            "sleeve_frac": cfg.sleeve_frac,
        },
        "cached": False,
    }

    events = res["ladder"].attrs.get("monetize_events", [])
    if events:
        result["monetize_events"] = events

    if include_series:
        result["series"] = {c: _downsample_series(ser_df[c], max_series_points) for c in ser_df.columns}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_payload = {k: v for k, v in result.items() if k != "series"}
    cache_metrics.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    ser_df.to_parquet(cache_series)

    if save:
        rid = run_id or cache_key
        dest = REPO / "data" / "runs" / pd.Timestamp.today().strftime("%Y-%m-%d") / "bucket5" / "lab" / rid
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(json.dumps(result["config"], indent=2), encoding="utf-8")
        (dest / "metrics.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        ser_df.to_parquet(dest / "series.parquet")
        result["artifacts"] = {"dir": str(dest.relative_to(REPO))}

    return result


def run_compare(
    specs: list[tuple[str, dict[str, Any] | str]],
    *,
    era: Era = "live",
    **kwargs: Any,
) -> dict[str, Any]:
    variants = {}
    for label, spec in specs:
        if isinstance(spec, str):
            variants[label] = run_backtest(preset=spec, era=era, **kwargs)
        else:
            variants[label] = run_backtest(params=spec, era=era, **kwargs)
    return {"schema": "bucket5_compare.v1", "era": era, "variants": variants}


def build_dashboard_payload(
    *,
    run_date: str | None = None,
    variants: list[tuple[str, str, Era]] | None = None,
) -> dict[str, Any]:
    run_date = run_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    if variants is None:
        variants = [
            ("B_live", "B_production", "live"),
            ("F_live", "F_dynamic_deep30", "live"),
            ("B_extended", "B_production", "extended"),
            ("F_extended", "F_dynamic_deep30", "extended"),
        ]

    out_variants: dict[str, Any] = {}
    for key, preset, era in variants:
        r = run_backtest(preset=preset, era=era, max_series_points=600)
        out_variants[key] = {
            "preset": preset,
            "era": era,
            "label": list_presets().get(preset, preset),
            "metrics": r["metrics"],
            "crash": r["crash"],
            "meta": r["meta"],
            "assumptions": r["assumptions"],
            "series": {
                "combined_equity": r.get("series", {}).get("combined_equity", []),
                "drawdown": r.get("series", {}).get("drawdown", []),
            },
        }

    return {
        "schema": "bucket5_backtest_panel.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "variants": out_variants,
        "presets": list_presets(),
        "lab_command": "streamlit run scripts/bucket5_lab.py",
        "repo": "ls-algo",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "presets", "dashboard"])
    ap.add_argument("--preset", default="B_production")
    ap.add_argument("--era", default="live", choices=["live", "extended"])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    if args.command == "presets":
        for k, v in list_presets().items():
            print(f"  {k}: {v}")
        return 0
    if args.command == "dashboard":
        payload = build_dashboard_payload()
        out = REPO / "risk_dashboard" / "data" / "bucket5_backtest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")
        return 0

    r = run_backtest(preset=args.preset, era=args.era, save=args.save)
    m = r["metrics"]
    print(
        f"{args.preset} ({args.era}): CAGR={m.get('combined_CAGR', 0) * 100:.2f}% "
        f"Vol={m.get('combined_Vol', 0) * 100:.2f}% MaxDD={m.get('combined_MaxDD', 0) * 100:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
