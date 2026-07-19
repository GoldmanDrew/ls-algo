"""Authoritative Bucket 4 dashboard export from production-actual ledgers.

This module does not reimplement strategy logic.  It packages the outputs from
``run_prod_replay_backtest`` into a deterministic, versioned contract consumed
by etf-dashboard.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

SCHEMA = "bucket4_production_replay.v1"
PAIR_SCHEMA = "bucket4_production_pair.v1"
B4_SLEEVE = "inverse_decay_bucket4"
REQUIRED_FILES = (
    "report.json",
    "daily_nav.csv",
    "sleeve_daily_pnl.csv",
    "pair_daily_pnl.csv",
    "pair_stats.csv",
    "rebalance_audit.csv",
    "pending_target_audit.csv",
)
OPTIONAL_AUDIT_FILES = (
    "b4_plan_history.csv",
    "b4_plan_history_daily.csv",
    "b4_pair_trade_ledger.csv",
    "b4_pair_trade_summary.csv",
    "prod_sizing_diag.csv",
    "rebalance_audit.csv",
    "pending_target_audit.csv",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    if pd.isna(value):
        return None
    return str(value)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def git_provenance(repo: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = _run("rev-parse", "HEAD")
        branch = _run("branch", "--show-current")
        status_raw = _run("status", "--porcelain", "--untracked-files=all")
        ignored_prefixes = (
            "notebooks/output/production_actual_bt/",
            "risk_dashboard/data/bucket4_production_replay/",
            "__pycache__/",
        )
        status_lines = []
        for line in status_raw.splitlines():
            rel = line[3:].replace("\\", "/")
            if rel.startswith(ignored_prefixes) or "/__pycache__/" in rel or rel.endswith(".pyc"):
                continue
            status_lines.append(line)
        status = "\n".join(status_lines)
        dirty = bool(status_lines)
        patch = b""
        if dirty:
            tracked = [line[3:] for line in status_lines if not line.startswith("?? ")]
            if tracked:
                patch += subprocess.check_output(
                    ["git", "diff", "--binary", "HEAD", "--", *tracked], cwd=repo
                )
            untracked = [
                line[3:]
                for line in status_lines
                if line.startswith("?? ") and (repo / line[3:]).is_file()
            ]
            for rel in sorted(untracked):
                p = repo / rel
                patch += rel.encode("utf-8") + b"\0" + p.read_bytes() + b"\0"
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "working_tree_hash": hashlib.sha256(patch).hexdigest() if dirty else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "branch": None, "dirty": None, "working_tree_hash": None}


def _read_csv(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path)
    unnamed = [c for c in out.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        out = out.drop(columns=unnamed)
    return out


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_value(row) for row in frame.to_dict(orient="records")]


def _perf(nav: pd.Series) -> dict[str, Any]:
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if len(nav) < 2:
        return {"cagr": None, "ann_vol": None, "sharpe": None, "max_drawdown": None}
    ret = nav.pct_change(fill_method=None).fillna(0.0)
    years = max((len(nav) - 1) / 252.0, 1.0 / 252.0)
    cagr = (float(nav.iloc[-1]) / float(nav.iloc[0])) ** (1.0 / years) - 1.0 if nav.iloc[0] > 0 and nav.iloc[-1] > 0 else None
    vol = float(ret.std(ddof=1) * math.sqrt(252.0)) if len(ret) > 1 else 0.0
    mean = float(ret.mean() * 252.0)
    dd = nav / nav.cummax() - 1.0
    return {
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": mean / vol if vol > 1e-12 else None,
        "max_drawdown": float(dd.min()),
    }


def _allocate_pair_reconciliation(pair_daily: pd.DataFrame, sleeve_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Allocate shared-financing residual so pair rows tie exactly to B4 sleeve PnL."""
    out = pair_daily.copy()
    out["date"] = out["date"].astype(str)
    out["daily_pnl_raw"] = pd.to_numeric(out["daily_pnl"], errors="coerce").fillna(0.0)
    sleeve = sleeve_daily.set_index(sleeve_daily["date"].astype(str))
    target = pd.to_numeric(sleeve[B4_SLEEVE], errors="coerce").fillna(0.0)
    raw_by_day = out.groupby("date")["daily_pnl_raw"].sum()
    residual_by_day = target.subtract(raw_by_day, fill_value=0.0)
    out["ledger_allocation_pnl"] = 0.0
    for day, residual in residual_by_day.items():
        if abs(float(residual)) <= 1e-12:
            continue
        mask = out["date"].eq(str(day))
        if not mask.any():
            continue
        gross = (
            pd.to_numeric(out.loc[mask, "etf_usd"], errors="coerce").fillna(0.0).abs()
            + pd.to_numeric(out.loc[mask, "underlying_usd"], errors="coerce").fillna(0.0).abs()
        )
        weights = gross / float(gross.sum()) if float(gross.sum()) > 1e-12 else pd.Series(1.0 / int(mask.sum()), index=gross.index)
        out.loc[mask, "ledger_allocation_pnl"] = weights * float(residual)
    out["daily_pnl"] = out["daily_pnl_raw"] + out["ledger_allocation_pnl"]
    after = out.groupby("date")["daily_pnl"].sum()
    after_residual = target.subtract(after, fill_value=0.0)
    return out, {
        "max_abs_before_usd": float(residual_by_day.abs().max()) if len(residual_by_day) else 0.0,
        "cumulative_before_usd": float(residual_by_day.sum()) if len(residual_by_day) else 0.0,
        "max_abs_after_usd": float(after_residual.abs().max()) if len(after_residual) else 0.0,
        "cumulative_after_usd": float(after_residual.sum()) if len(after_residual) else 0.0,
    }


def _event_map(source_dir: Path, expected_end: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = source_dir / "b4_pair_trade_ledger.csv"
    if not path.exists():
        return {}
    frame = _read_csv(path)
    if frame.empty or "date" not in frame or str(frame["date"].astype(str).max()) != str(expected_end):
        # Never attach a stale audit ledger to a freshly generated replay.
        return {}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = (str(row.get("ETF", "")).upper(), str(row.get("date", "")))
        result[key] = _json_value(row)
    return result


def _pair_payload(etf: str, frame: pd.DataFrame, stats_row: Mapping[str, Any], events: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    d = frame.sort_values("date").copy()
    d["date"] = d["date"].astype(str)
    for col in (
        "etf_usd", "underlying_usd", "Delta", "price_pnl", "borrow_cost",
        "short_credit", "margin_cost", "txn_cost", "daily_pnl",
        "daily_pnl_raw", "ledger_allocation_pnl", "cum_pnl",
    ):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    d["cum_pnl_reconciled"] = d["daily_pnl"].cumsum()
    gross = d["etf_usd"].abs() + d["underlying_usd"].abs()
    h = np.where(
        (d["etf_usd"].abs() > 1e-12) & (d["Delta"].abs() > 1e-12),
        d["underlying_usd"].abs() / (d["etf_usd"].abs() * d["Delta"].abs()),
        np.nan,
    )
    reasons: list[str] = []
    event_rows: list[dict[str, Any]] = []
    for day in d["date"]:
        event = events.get((etf, day))
        reasons.append(str(event.get("reason", "")) if event else "")
        if event:
            event_rows.append(dict(event))
    basis = max(float(gross.replace(0.0, np.nan).dropna().iloc[0]) if (gross > 0).any() else 0.0, 1.0)
    equity = basis + d["cum_pnl_reconciled"]
    ret = equity.pct_change(fill_method=None).fillna(0.0)
    summary = {
        "etf": etf,
        "underlying": str(d["Underlying"].iloc[-1]).upper(),
        "production_status": "production_policy_replay",
        "model_status": "authoritative_export",
        "entry_date": str(d["date"].iloc[0]),
        "latest_date": str(d["date"].iloc[-1]),
        "n_days": int(len(d)),
        "n_rebalances": int(pd.to_numeric(d.get("is_rebalance", 0), errors="coerce").fillna(0).astype(bool).sum()),
        "total_borrow": float(d["borrow_cost"].sum()),
        "total_fees": float(d["txn_cost"].sum()),
        "final_equity": float(equity.iloc[-1] / basis),
        "actual_pnl_usd": float(d["daily_pnl"].sum()),
        "notional_basis_usd": basis,
        "h_last": float(pd.Series(h).dropna().iloc[-1]) if pd.Series(h).notna().any() else None,
        "mean_h": float(pd.Series(h).dropna().mean()) if pd.Series(h).notna().any() else None,
        **_perf(equity),
    }
    for key, value in stats_row.items():
        if key not in summary and key not in {"ETF", "Underlying", "sleeve"}:
            summary[str(key)] = _json_value(value)
    return {
        "schema": PAIR_SCHEMA,
        "etf": etf,
        "underlying": summary["underlying"],
        "in_production_book": True,
        "production_status": "production_policy_replay",
        "ledger_mode": "actual_dollar",
        "notional_basis_usd": basis,
        "summary": summary,
        "daily": {
            "dates": d["date"].tolist(),
            "equity": [float(x / basis) for x in equity],
            "equity_dollars": [float(x) for x in equity],
            "ret": [float(x) for x in ret],
            "net_pnl": [float(x / basis) for x in d["cum_pnl_reconciled"]],
            "net_pnl_dollars": [float(x) for x in d["cum_pnl_reconciled"]],
            "daily_pnl_dollars": [float(x) for x in d["daily_pnl"]],
            "price_pnl_cum_dollars": [float(x) for x in d["price_pnl"].cumsum()],
            "borrow_cost": [float(x / basis) for x in d["borrow_cost"]],
            "borrow_cost_dollars": [float(x) for x in d["borrow_cost"]],
            "borrow_cost_cum_dollars": [float(x) for x in d["borrow_cost"].cumsum()],
            "short_credit_dollars": [float(x) for x in d["short_credit"]],
            "margin_cost_dollars": [float(x) for x in d["margin_cost"]],
            "rebalance_fee": [float(x / basis) for x in d["txn_cost"]],
            "txn_cost_dollars": [float(x) for x in d["txn_cost"]],
            "txn_cost_cum_dollars": [float(x) for x in d["txn_cost"].cumsum()],
            "ledger_allocation_pnl_dollars": [float(x) for x in d["ledger_allocation_pnl"]],
            "etf_usd": [float(x) for x in d["etf_usd"]],
            "underlying_usd": [float(x) for x in d["underlying_usd"]],
            "gross_exposure_dollars": [float(x) for x in gross],
            "h_used": [float(x) if math.isfinite(float(x)) else None for x in h],
            "rebalance": [int(bool(x)) for x in pd.to_numeric(d.get("is_rebalance", 0), errors="coerce").fillna(0)],
            "rebalance_reason": reasons,
            "active_plan_date": d.get("active_plan_date", pd.Series("", index=d.index)).astype(str).tolist(),
        },
        "rebalance_log": event_rows,
    }


def build_contract(source_dir: Path, repo_root: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    missing = [name for name in REQUIRED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"production replay missing required files: {', '.join(missing)}")
    report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    if str(report.get("mode", "")).lower() != "prod":
        raise ValueError("report.json is not a production replay")
    sleeve_daily = _read_csv(source_dir / "sleeve_daily_pnl.csv")
    pair_daily_all = _read_csv(source_dir / "pair_daily_pnl.csv")
    pair_stats_all = _read_csv(source_dir / "pair_stats.csv")
    b4_daily = pair_daily_all.loc[pair_daily_all["sleeve"].astype(str).eq(B4_SLEEVE)].copy()
    b4_stats = pair_stats_all.loc[pair_stats_all["sleeve"].astype(str).eq(B4_SLEEVE)].copy()
    if b4_daily.empty:
        raise ValueError("pair_daily_pnl.csv contains no Bucket 4 rows")
    end = str(sleeve_daily["date"].astype(str).max())
    if end != str(report.get("end")):
        raise ValueError(f"source end-date mismatch: report={report.get('end')} ledger={end}")
    b4_daily, pair_recon = _allocate_pair_reconciliation(b4_daily, sleeve_daily)
    events = _event_map(source_dir, end)
    stats_by_etf = {
        str(row.get("ETF", "")).upper(): row for row in b4_stats.to_dict(orient="records")
    }
    pairs = {
        etf: _pair_payload(etf, frame, stats_by_etf.get(etf, {}), events)
        for etf, frame in b4_daily.groupby(b4_daily["ETF"].astype(str).str.upper(), sort=True)
    }
    budget = _finite((report.get("budgets_usd") or {}).get(B4_SLEEVE), 0.0)
    if budget <= 0:
        raise ValueError("report has no positive Bucket 4 budget")
    daily = sleeve_daily.sort_values("date").copy()
    pnl = pd.to_numeric(daily[B4_SLEEVE], errors="coerce").fillna(0.0)
    nav = budget + pnl.cumsum()
    book = {
        "schema": "bucket4_production_book.v1",
        "sleeve": B4_SLEEVE,
        "initial_capital_usd": budget,
        "dates": daily["date"].astype(str).tolist(),
        "daily_pnl_usd": [float(x) for x in pnl],
        "cumulative_pnl_usd": [float(x) for x in pnl.cumsum()],
        "nav_usd": [float(x) for x in nav],
        "equity": [float(x / budget) for x in nav],
        "returns": [float(x) for x in nav.pct_change(fill_method=None).fillna(0.0)],
        "price_pnl_usd": [float(x) for x in pd.to_numeric(daily[f"{B4_SLEEVE}__price_pnl"], errors="coerce").fillna(0.0)],
        "borrow_cost_usd": [float(x) for x in pd.to_numeric(daily[f"{B4_SLEEVE}__borrow_cost"], errors="coerce").fillna(0.0)],
        "short_credit_usd": [float(x) for x in pd.to_numeric(daily[f"{B4_SLEEVE}__short_credit"], errors="coerce").fillna(0.0)],
        "margin_cost_usd": [float(x) for x in pd.to_numeric(daily[f"{B4_SLEEVE}__margin_cost"], errors="coerce").fillna(0.0)],
        "txn_cost_usd": [float(x) for x in pd.to_numeric(daily[f"{B4_SLEEVE}__txn_cost"], errors="coerce").fillna(0.0)],
        "gross_cap_usd": [float(x) for x in pd.to_numeric(daily[f"{B4_SLEEVE}__gross_cap"], errors="coerce").fillna(0.0)],
        "net_cap_usd": [float(x) for x in pd.to_numeric(daily[f"{B4_SLEEVE}__net_cap"], errors="coerce").fillna(0.0)],
        "summary": {**_perf(nav), "final_nav_usd": float(nav.iloc[-1]), "net_pnl_usd": float(pnl.sum()), "n_days": int(len(nav))},
    }
    input_hashes = {name: sha256_file(source_dir / name) for name in REQUIRED_FILES}
    config_path = repo_root / "config" / "strategy_config.yml"
    if config_path.is_file():
        input_hashes["config/strategy_config.yml"] = sha256_file(config_path)
    manifest = {
        "schema": SCHEMA,
        "authoritative": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {k: _json_value(report.get(k)) for k in ("mode", "run_date", "start", "end", "capital_usd", "gross_leverage")},
        "source": git_provenance(repo_root),
        "input_hashes": input_hashes,
        "resolved_policy": _json_value(report.get("rebalance_knobs") or {}),
        "resolved_policy_hash": sha256_payload(report.get("rebalance_knobs") or {}),
        "limitations": _json_value(report.get("limitations") or []),
        "counts": {"pairs": len(pairs), "book_days": len(daily), "pair_days": len(b4_daily)},
        "reconciliation": {
            "pair_to_sleeve": pair_recon,
            "book_max_abs_residual_usd": float(pd.to_numeric(daily.get("pnl_recon_residual", 0.0), errors="coerce").fillna(0.0).abs().max()),
        },
        "artifacts": {"book": "book.json", "pairs": "pairs", "audit": "audit"},
    }
    if pair_recon["max_abs_after_usd"] > 0.01:
        raise ValueError(f"pair-to-sleeve reconciliation failed: {pair_recon}")
    return {"manifest": manifest, "book": book, "pairs": pairs}


def export_contract(source_dir: Path, out_dir: Path, repo_root: Path) -> dict[str, Any]:
    contract = build_contract(source_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_dir / "pairs"
    audit_dir = out_dir / "audit"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "book.json", contract["book"])
    for etf, payload in contract["pairs"].items():
        write_json(pairs_dir / f"{etf}.json", payload)
    for name in OPTIONAL_AUDIT_FILES:
        src = source_dir / name
        if src.is_file():
            shutil.copyfile(src, audit_dir / name)
    manifest = dict(contract["manifest"])
    manifest["output_hashes"] = {
        "book.json": sha256_file(out_dir / "book.json"),
        **{f"pairs/{etf}.json": sha256_file(pairs_dir / f"{etf}.json") for etf in sorted(contract["pairs"])},
    }
    write_json(out_dir / "manifest.json", manifest)
    contract["manifest"] = manifest
    return contract


def validate_contract(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("authoritative") is not True:
        raise ValueError("not an authoritative Bucket 4 production contract")
    for rel, expected in (manifest.get("output_hashes") or {}).items():
        path = root / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"contract hash mismatch: {rel}")
    recon = ((manifest.get("reconciliation") or {}).get("pair_to_sleeve") or {})
    if abs(_finite(recon.get("max_abs_after_usd"))) > 0.01:
        raise ValueError("contract pair-to-sleeve reconciliation is not exact")
    return manifest
