#!/usr/bin/env python3
"""Harvest under-exposed ETF shorts using position vs plan gaps.

By default, builds the discrepancy table from **live IBKR positions** (strategy shares × snapshot
marks) vs **proposed_trades.csv** — same merge rules as ``run_eod_pnl_email.load_position_discrepancies``
(Flex is not required). If IB is unreachable or live build fails, falls back to an existing
``position_discrepancies_all.csv`` (dated run or ``data/`` latest). Pass ``--discrepancy-csv`` to
force a specific file.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execute_trade_plan import (
    CoordinatorCancelService,
    configure_ib_error_log_filter,
    connect_ib,
    current_ib_positions,
    execute_leg,
    fetch_ibkr_short_availability_map,
    get_snapshot_price,
    load_baseline_qty,
    norm_sym,
    run_dir,
    stop_requested,
    strategy_position_only,
    tprint,
    today_str,
)
from generate_trade_plan import load_blacklist
from ibkr_accounting import (
    EXCLUDE_SYMBOLS,
    SUPPLEMENTAL_ETF_MAP,
    canonical_symbol,
    load_etf_beta_map,
    load_etf_to_under_map,
    load_universe_from_screened,
    normalize_plan_etf_ticker,
)
from strategy_config import load_config


def resolve_proposed_trades_path(run_date: str, paths_cfg: dict) -> Path:
    dated = run_dir(run_date) / "proposed_trades.csv"
    if dated.exists():
        return dated
    p = paths_cfg.get("proposed_trades_csv")
    if p:
        pp = Path(str(p))
        if pp.exists():
            return pp
    fb = Path("data") / "proposed_trades.csv"
    if fb.exists():
        return fb
    raise FileNotFoundError(
        "Could not locate proposed_trades.csv (try dated data/runs/<run-date>/proposed_trades.csv "
        "or paths.proposed_trades_csv in config)."
    )


def _position_discrepancy_merge(
    plan: pd.DataFrame,
    actual: pd.DataFrame,
    screened_csv: Path,
    cfg: dict,
) -> pd.DataFrame:
    """Match ``run_eod_pnl_email.load_position_discrepancies`` merge (ETF-universe filter on symbols)."""
    strategy_tag = str((cfg.get("strategy", {}) or {}).get("tag", "")).strip()
    if strategy_tag and plan is not None and not plan.empty and "strategy_tag" in plan.columns:
        plan = plan[plan["strategy_tag"].astype(str) == strategy_tag].copy()

    if plan is None or plan.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "target_net_usd",
                "actual_net_usd",
                "discrepancy_usd",
                "abs_discrepancy_usd",
                "target_gross_usd",
                "actual_gross_usd",
                "gross_gap_usd",
                "under_exposed",
            ]
        )

    for c in ("ETF", "Underlying"):
        if c not in plan.columns:
            raise ValueError(f"proposed_trades.csv missing required column: {c}")
    plan = plan.copy()
    plan["ETF"] = plan["ETF"].astype(str).map(canonical_symbol).map(normalize_plan_etf_ticker)
    plan["Underlying"] = plan["Underlying"].astype(str).map(canonical_symbol)
    plan["long_usd"] = pd.to_numeric(plan.get("long_usd", 0.0), errors="coerce").fillna(0.0)
    plan["short_usd"] = pd.to_numeric(plan.get("short_usd", 0.0), errors="coerce").fillna(0.0)

    target_under = (
        plan.groupby("Underlying", as_index=False)["long_usd"]
        .sum()
        .rename(columns={"Underlying": "symbol", "long_usd": "target_net_usd"})
    )
    target_etf = (
        plan.groupby("ETF", as_index=False)["short_usd"]
        .sum()
        .rename(columns={"ETF": "symbol", "short_usd": "target_net_usd"})
    )
    target = pd.concat([target_under, target_etf], ignore_index=True)
    target = target[target["symbol"].astype(bool)].copy()
    target = target.groupby("symbol", as_index=False)["target_net_usd"].sum()

    if actual is None or actual.empty:
        actual_df = pd.DataFrame(columns=["symbol", "actual_net_usd"])
    else:
        actual_df = actual.copy()
        actual_df["symbol"] = actual_df["symbol"].astype(str).map(canonical_symbol)
        actual_df["actual_net_usd"] = pd.to_numeric(actual_df["actual_net_usd"], errors="coerce").fillna(0.0)
        actual_df = actual_df.groupby("symbol", as_index=False)["actual_net_usd"].sum()

    allowed_etfs, _ = load_universe_from_screened(screened_csv)
    allowed_etfs |= set(SUPPLEMENTAL_ETF_MAP.keys())

    etf_to_under = load_etf_to_under_map(screened_csv)
    for e_sym, u_sym in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(e_sym, u_sym)

    blacklist_raw = ((cfg.get("strategy", {}) or {}).get("blacklist", [])) or []
    blacklist = {canonical_symbol(str(s)) for s in blacklist_raw if str(s).strip()}
    blocked_etfs = {s for s in blacklist if s in allowed_etfs}
    blocked_etfs |= {e for e, u in etf_to_under.items() if u in blacklist}

    merged = target.merge(actual_df, on="symbol", how="outer")
    merged["target_net_usd"] = pd.to_numeric(merged["target_net_usd"], errors="coerce").fillna(0.0)
    merged["actual_net_usd"] = pd.to_numeric(merged["actual_net_usd"], errors="coerce").fillna(0.0)
    merged["symbol"] = merged["symbol"].astype(str).map(canonical_symbol)
    merged = merged[merged["symbol"].isin(allowed_etfs)].copy()
    if blocked_etfs:
        merged = merged[~merged["symbol"].isin(blocked_etfs)].copy()

    merged["discrepancy_usd"] = merged["actual_net_usd"] - merged["target_net_usd"]
    merged["abs_discrepancy_usd"] = merged["discrepancy_usd"].abs()
    merged["target_gross_usd"] = merged["target_net_usd"].abs()
    merged["actual_gross_usd"] = merged["actual_net_usd"].abs()
    merged["gross_gap_usd"] = merged["actual_gross_usd"] - merged["target_gross_usd"]
    merged["under_exposed"] = merged["gross_gap_usd"] < -1e-9
    return merged.sort_values("abs_discrepancy_usd", ascending=False).reset_index(drop=True)


def build_discrepancy_from_live_ib(
    ib: Any,
    *,
    run_date: str,
    cfg: dict,
    screened_csv: Path,
    baseline_csv: Path,
    prefer_delayed: bool,
    paths_cfg: dict,
) -> pd.DataFrame:
    """Actual notionals from strategy-only IB positions × snapshot marks vs proposed plan."""
    proposed_path = resolve_proposed_trades_path(run_date, paths_cfg)
    plan = pd.read_csv(proposed_path)

    baseline = load_baseline_qty(baseline_csv)
    ib_pos = current_ib_positions(ib)
    strat = strategy_position_only(ib_pos, baseline)

    rows: list[dict[str, Any]] = []
    for sym_raw, sh in strat.items():
        sh_f = float(sh)
        if abs(sh_f) < 1e-9:
            continue
        sym = canonical_symbol(norm_sym(str(sym_raw)))
        if sym in EXCLUDE_SYMBOLS:
            continue
        try:
            px = float(get_snapshot_price(ib, sym_raw, prefer_delayed=prefer_delayed))
        except Exception:
            px = float("nan")
        if not (np.isfinite(px) and px > 0):
            tprint(f"[HARVEST] LIVE_DISC: skip {sym} (no snapshot price; shares={sh_f:g})")
            continue
        rows.append({"symbol": sym, "actual_net_usd": sh_f * px})

    actual_df = pd.DataFrame(rows)
    return _position_discrepancy_merge(plan, actual_df, screened_csv, cfg)


def resolve_discrepancy_csv(run_date: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Discrepancy CSV not found at --discrepancy-csv path: {p}")
    dated = run_dir(run_date) / "accounting" / "position_discrepancies_all.csv"
    if dated.exists():
        return dated
    latest = Path("data") / "position_discrepancies_all.csv"
    if latest.exists():
        return latest
    raise FileNotFoundError(
        "Could not find discrepancy CSV. Expected one of:\n"
        f"- {dated}\n"
        f"- {latest}"
    )


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Harvest under-exposed ETF shorts and hedge with underlying buys."
    )
    ap.add_argument("--run-date", default=None, help="YYYY-MM-DD (defaults to RUN_DATE or today).")
    ap.add_argument("--discrepancy-csv", default=None, help="Optional explicit discrepancy CSV path.")
    ap.add_argument("--top-n", type=int, default=30, help="Process top N under-exposed ETFs by |discrepancy|.")
    ap.add_argument(
        "--underhedge-buffer-pct",
        type=float,
        default=0.0025,
        help="Under-hedge buffer fraction (0.25%% default).",
    )
    ap.add_argument(
        "--max-short-usd-per-etf",
        type=float,
        default=0.0,
        help="Optional cap per ETF short notional. 0 means no cap.",
    )
    ap.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip confirmation prompt before live order placement.",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Deprecated flag (live is now default).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Force dry-run mode.")
    ap.add_argument(
        "--file-discrepancies-only",
        action="store_true",
        help="Skip live IB discrepancy build; use position_discrepancies_all.csv (dated or data/).",
    )
    args = ap.parse_args()

    run_date = args.run_date or os.environ.get("RUN_DATE") or today_str()
    dry_run = bool(args.dry_run)

    cfg = load_config("config/strategy_config.yml")
    ibkr_cfg = cfg.get("ibkr", {}) or {}
    strat_cfg = cfg.get("strategy", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}
    reb_cfg = ((cfg.get("portfolio", {}) or {}).get("rebalance", {}) or {})

    strategy_tag = str(strat_cfg.get("tag", "")).strip()
    if not strategy_tag:
        raise ValueError("Missing strategy.tag in config.")

    host = str(ibkr_cfg.get("host", "127.0.0.1"))
    port = int(ibkr_cfg.get("port", 7496))
    client_id = int(ibkr_cfg.get("client_id", 41))
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))
    suppress_error_codes = [int(c) for c in ((ibkr_cfg.get("suppress_error_codes", [10089])) or [])]
    configure_ib_error_log_filter(suppress_error_codes)

    min_trade_usd = float(reb_cfg.get("min_trade_usd", 200.0))
    limit_bps = float(exec_cfg.get("limit_bps", 25.0))
    timeout_sec = float(exec_cfg.get("timeout_sec", 90))
    max_retries = int(exec_cfg.get("max_retries", 3))
    max_short_usd_per_etf = float(args.max_short_usd_per_etf or 0.0)

    screened_csv = Path(paths_cfg.get("screened_csv", "data/etf_screened_today.csv"))
    baseline_csv = Path(paths_cfg.get("baseline_csv", "data/baseline_snapshot.csv"))
    if not screened_csv.exists():
        raise FileNotFoundError(f"Screened CSV not found: {screened_csv}")

    outdir = run_dir(run_date) / "execution"
    outdir.mkdir(parents=True, exist_ok=True)
    candidates_path = outdir / "harvest_candidates.csv"
    attempted_path = outdir / "harvest_attempted_trades.csv"
    fills_path = outdir / "harvest_fills.csv"
    summary_path = outdir / "harvest_post_trade_summary.csv"
    disc_source_path = outdir / "harvest_discrepancy_source.txt"
    disc_input_path = outdir / "harvest_discrepancies_input.csv"

    disc: pd.DataFrame
    disc_source = ""
    if args.discrepancy_csv:
        discrepancy_csv = resolve_discrepancy_csv(run_date, args.discrepancy_csv)
        disc = pd.read_csv(discrepancy_csv)
        disc_source = f"explicit_file:{discrepancy_csv}"
    elif args.file_discrepancies_only:
        discrepancy_csv = resolve_discrepancy_csv(run_date, None)
        disc = pd.read_csv(discrepancy_csv)
        disc_source = f"file_only:{discrepancy_csv}"
    else:
        try:
            ib_pre = connect_ib(host, port, client_id + 510, coordinator=False)
            try:
                disc = build_discrepancy_from_live_ib(
                    ib_pre,
                    run_date=run_date,
                    cfg=cfg,
                    screened_csv=screened_csv,
                    baseline_csv=baseline_csv,
                    prefer_delayed=prefer_delayed,
                    paths_cfg=paths_cfg,
                )
                disc_source = "live_ib"
            finally:
                try:
                    ib_pre.disconnect()
                except Exception:
                    pass
        except Exception as ex:
            tprint(
                f"[HARVEST] Live discrepancy build failed ({ex}); "
                f"falling back to position_discrepancies_all.csv."
            )
            discrepancy_csv = resolve_discrepancy_csv(run_date, None)
            disc = pd.read_csv(discrepancy_csv)
            disc_source = f"file_fallback:{discrepancy_csv}"

    disc_source_path.write_text(disc_source.strip() + "\n", encoding="utf-8")
    if disc.empty:
        tprint(f"[HARVEST] Discrepancy table has no rows (source={disc_source})")
        pd.DataFrame().to_csv(candidates_path, index=False)
        pd.DataFrame().to_csv(attempted_path, index=False)
        pd.DataFrame().to_csv(fills_path, index=False)
        pd.DataFrame().to_csv(summary_path, index=False)
        return 0

    for col in ("symbol", "abs_discrepancy_usd", "gross_gap_usd", "under_exposed"):
        if col not in disc.columns:
            raise ValueError(f"Discrepancy table missing required column: {col}")

    disc["symbol"] = disc["symbol"].astype(str).map(canonical_symbol)
    disc["abs_discrepancy_usd"] = pd.to_numeric(disc["abs_discrepancy_usd"], errors="coerce").fillna(0.0)
    disc["gross_gap_usd"] = pd.to_numeric(disc["gross_gap_usd"], errors="coerce").fillna(0.0)
    disc["under_exposed"] = to_bool_series(disc["under_exposed"])
    disc.to_csv(disc_input_path, index=False)
    tprint(f"[HARVEST] Discrepancy input ({disc_source}) -> {disc_input_path}")

    _etf_to_under_raw, _etf_to_beta_raw = load_etf_beta_map(screened_csv)
    etf_to_under = {canonical_symbol(str(k)): canonical_symbol(str(v)) for k, v in _etf_to_under_raw.items()}
    etf_to_beta = {canonical_symbol(str(k)): float(v) for k, v in _etf_to_beta_raw.items()}

    blacklist = {canonical_symbol(str(s)) for s in load_blacklist(cfg)}
    blocked_by_under = {etf for etf, und in etf_to_under.items() if und in blacklist}
    blocked_symbols = blacklist | blocked_by_under

    cands = disc[disc["under_exposed"]].copy()
    cands = cands[~cands["symbol"].isin(blocked_symbols)].copy()
    cands["underlying"] = cands["symbol"].map(etf_to_under)
    cands["beta"] = cands["symbol"].map(etf_to_beta)
    cands = cands[cands["underlying"].astype(str).str.len() > 0].copy()
    cands = cands[pd.to_numeric(cands["beta"], errors="coerce").fillna(0.0) > 0.0].copy()
    cands = cands.sort_values("abs_discrepancy_usd", ascending=False).reset_index(drop=True)
    if args.top_n > 0:
        cands = cands.head(int(args.top_n)).copy()
    cands.to_csv(candidates_path, index=False)

    if cands.empty:
        tprint("[HARVEST] No valid under-exposed ETF candidates after filters.")
        pd.DataFrame().to_csv(attempted_path, index=False)
        pd.DataFrame().to_csv(fills_path, index=False)
        pd.DataFrame().to_csv(summary_path, index=False)
        return 0

    short_map: dict[str, dict[str, Any]]
    try:
        short_map = fetch_ibkr_short_availability_map(cands["symbol"].tolist())
    except Exception as ex:
        tprint(f"[HARVEST] WARNING: FTP short availability fetch failed ({ex}); continuing uncapped.")
        short_map = {}

    baseline = load_baseline_qty(baseline_csv)
    tprint(
        f"[HARVEST] run_date={run_date} candidates={len(cands)} dry_run={dry_run} "
        f"underhedge_buffer={args.underhedge_buffer_pct:.4f}"
    )

    attempted_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    try:
        ib = connect_ib(host, port, client_id + 510, coordinator=True)
    except Exception as ex:
        tprint(f"[HARVEST] ERROR: IB connection failed: {ex}")
        pd.DataFrame(attempted_rows).to_csv(attempted_path, index=False)
        pd.DataFrame(fill_rows).to_csv(fills_path, index=False)
        pd.DataFrame(
            [{"status": "IB_CONNECTION_FAILED", "error": str(ex), "run_date": run_date}]
        ).to_csv(summary_path, index=False)
        return 2

    cancel_service = CoordinatorCancelService(host=host, port=port)
    cancel_service.start()
    try:
        ib_pos = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)
        tprint(f"[HARVEST] Strategy-only symbols currently held: {len(strat_pos)}")
        planned_rows: list[dict[str, Any]] = []
        for _, row in cands.iterrows():
            if stop_requested():
                tprint("[HARVEST] Stop requested; halting candidate planning.")
                break

            etf = norm_sym(str(row["symbol"]))
            under = norm_sym(str(row["underlying"]))
            beta = float(row["beta"])
            need_usd = max(0.0, -float(row.get("gross_gap_usd", 0.0) or 0.0))
            if max_short_usd_per_etf > 0:
                need_usd = min(need_usd, max_short_usd_per_etf)

            if need_usd < min_trade_usd:
                summary_rows.append(
                    {
                        "symbol": etf,
                        "underlying": under,
                        "requested_short_usd": need_usd,
                        "requested_short_sh": 0,
                        "filled_short_sh": 0,
                        "filled_under_buy_sh": 0,
                        "remaining_short_usd": need_usd,
                        "residual_beta_usd_after_hedge": 0.0,
                        "status": "SKIP_BELOW_MIN_TRADE_USD",
                    }
                )
                continue

            try:
                px_etf = float(get_snapshot_price(ib, etf, prefer_delayed=prefer_delayed))
                px_under = float(get_snapshot_price(ib, under, prefer_delayed=prefer_delayed))
            except Exception as ex:
                summary_rows.append(
                    {
                        "symbol": etf,
                        "underlying": under,
                        "requested_short_usd": need_usd,
                        "requested_short_sh": 0,
                        "filled_short_sh": 0,
                        "filled_under_buy_sh": 0,
                        "remaining_short_usd": need_usd,
                        "residual_beta_usd_after_hedge": 0.0,
                        "status": f"SKIP_NO_PRICE: {ex}",
                    }
                )
                continue

            requested_short_sh = int(math.floor(need_usd / px_etf)) if px_etf > 0 else 0
            if requested_short_sh <= 0:
                summary_rows.append(
                    {
                        "symbol": etf,
                        "underlying": under,
                        "requested_short_usd": need_usd,
                        "requested_short_sh": 0,
                        "filled_short_sh": 0,
                        "filled_under_buy_sh": 0,
                        "remaining_short_usd": need_usd,
                        "residual_beta_usd_after_hedge": 0.0,
                        "status": "SKIP_ZERO_SHARES_AFTER_ROUNDING",
                    }
                )
                continue

            avail = (short_map.get(etf) or {}).get("available")
            borrow = (short_map.get(etf) or {}).get("borrow")
            if isinstance(avail, (int, float)) and int(avail) <= 0:
                summary_rows.append(
                    {
                        "symbol": etf,
                        "underlying": under,
                        "requested_short_usd": need_usd,
                        "requested_short_sh": requested_short_sh,
                        "filled_short_sh": 0,
                        "filled_under_buy_sh": 0,
                        "remaining_short_usd": need_usd,
                        "residual_beta_usd_after_hedge": 0.0,
                        "status": "SKIP_FTP_AVAILABLE_ZERO",
                    }
                )
                continue
            if isinstance(avail, (int, float)) and int(avail) > 0:
                requested_short_sh = min(requested_short_sh, int(avail))

            if requested_short_sh <= 0:
                summary_rows.append(
                    {
                        "symbol": etf,
                        "underlying": under,
                        "requested_short_usd": need_usd,
                        "requested_short_sh": 0,
                        "filled_short_sh": 0,
                        "filled_under_buy_sh": 0,
                        "remaining_short_usd": need_usd,
                        "residual_beta_usd_after_hedge": 0.0,
                        "status": "SKIP_ZERO_AFTER_FTP_CAP",
                    }
                )
                continue

            planned_rows.append(
                {
                    "symbol": etf,
                    "underlying": under,
                    "beta": beta,
                    "target_short_usd": need_usd,
                    "etf_px": px_etf,
                    "under_px": px_under,
                    "requested_short_sh": requested_short_sh,
                    "ftp_available": avail,
                    "ftp_borrow_annual": borrow,
                }
            )

        if not planned_rows:
            tprint("[HARVEST] No actionable candidates after planning filters.")
            pd.DataFrame(attempted_rows).to_csv(attempted_path, index=False)
            pd.DataFrame(fill_rows).to_csv(fills_path, index=False)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            return 0

        preview_df = pd.DataFrame(planned_rows).copy()
        preview_df["requested_short_notional_usd"] = (
            preview_df["requested_short_sh"] * preview_df["etf_px"]
        )
        preview_df["planned_under_buy_sh_if_full_fill"] = (
            (
                (preview_df["requested_short_sh"] * preview_df["etf_px"] * preview_df["beta"])
                / preview_df["under_px"]
            )
            * max(0.0, 1.0 - float(args.underhedge_buffer_pct))
        ).apply(lambda x: int(math.floor(float(x))) if pd.notna(x) else 0)

        tprint("")
        tprint("=" * 120)
        tprint("[HARVEST] PRE-TRADE PLAN (what will be attempted)")
        tprint("=" * 120)
        tprint(
            preview_df[
                [
                    "symbol",
                    "underlying",
                    "target_short_usd",
                    "etf_px",
                    "requested_short_sh",
                    "requested_short_notional_usd",
                    "planned_under_buy_sh_if_full_fill",
                    "ftp_available",
                ]
            ].to_string(index=False)
        )
        tprint("=" * 120)
        tprint("")

        if (not dry_run) and (not args.auto_approve):
            ans = input("[HARVEST] Approve live execution of this plan? (y/n): ").strip().lower()
            if ans != "y":
                tprint("[HARVEST] Execution cancelled by user.")
                summary_rows.extend(
                    {
                        "symbol": str(r["symbol"]),
                        "underlying": str(r["underlying"]),
                        "requested_short_usd": float(r["target_short_usd"]),
                        "requested_short_sh": int(r["requested_short_sh"]),
                        "filled_short_sh": 0,
                        "filled_under_buy_sh": 0,
                        "remaining_short_usd": float(r["target_short_usd"]),
                        "residual_beta_usd_after_hedge": 0.0,
                        "status": "SKIP_USER_CANCELLED",
                    }
                    for r in planned_rows
                )
                pd.DataFrame(attempted_rows).to_csv(attempted_path, index=False)
                pd.DataFrame(fill_rows).to_csv(fills_path, index=False)
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
                return 0

        for row in planned_rows:
            if stop_requested():
                tprint("[HARVEST] Stop requested; halting execution loop.")
                break

            etf = norm_sym(str(row["symbol"]))
            under = norm_sym(str(row["underlying"]))
            beta = float(row["beta"])
            need_usd = float(row["target_short_usd"])
            px_etf = float(row["etf_px"])
            px_under = float(row["under_px"])
            requested_short_sh = int(row["requested_short_sh"])
            avail = row.get("ftp_available")
            borrow = row.get("ftp_borrow_annual")

            tprint(
                f"[HARVEST] TRY {etf}: short {requested_short_sh} sh (~${requested_short_sh * px_etf:,.0f}) "
                f"then hedge {under} from actual fills."
            )

            short_ref = f"{strategy_tag}|HARVEST_SHORT|{under}|{etf}"
            short_res = execute_leg(
                ib=ib,
                symbol=etf,
                action="SELL",
                qty=requested_short_sh,
                ref_price=px_etf,
                bps=limit_bps,
                order_ref=short_ref,
                exec_cfg=exec_cfg,
                timeout=timeout_sec,
                max_retries=max_retries,
                dry_run=dry_run,
                context=f"HARVEST|{etf}",
                cancel_service=cancel_service,
            )
            attempted_rows.append(
                {
                    "symbol": etf,
                    "underlying": under,
                    "leg": "ETF_SHORT",
                    "action": "SELL",
                    "requested_sh": requested_short_sh,
                    "filled_sh": int(short_res.filled),
                    "status": short_res.status,
                    "error_code": short_res.error_code,
                    "error_msg": short_res.error_msg,
                    "price_ref": px_etf,
                    "beta": beta,
                    "ftp_available": avail,
                    "ftp_borrow_annual": borrow,
                }
            )

            filled_short_sh = int(short_res.filled)
            if filled_short_sh <= 0:
                summary_rows.append(
                    {
                        "symbol": etf,
                        "underlying": under,
                        "requested_short_usd": need_usd,
                        "requested_short_sh": requested_short_sh,
                        "filled_short_sh": 0,
                        "filled_under_buy_sh": 0,
                        "remaining_short_usd": need_usd,
                        "residual_beta_usd_after_hedge": 0.0,
                        "status": f"NO_SHORT_FILL_{short_res.status}",
                    }
                )
                continue

            hedge_notional_usd = filled_short_sh * px_etf * beta
            hedge_target_sh = (
                int(
                    math.floor(
                        (hedge_notional_usd / px_under)
                        * max(0.0, 1.0 - float(args.underhedge_buffer_pct))
                    )
                )
                if px_under > 0
                else 0
            )

            filled_under_sh = 0
            hedge_status = "HEDGE_SKIPPED"
            if hedge_target_sh > 0 and (hedge_target_sh * px_under) >= min_trade_usd:
                hedge_ref = f"{strategy_tag}|HARVEST_HEDGE|{under}|{etf}"
                hedge_res = execute_leg(
                    ib=ib,
                    symbol=under,
                    action="BUY",
                    qty=hedge_target_sh,
                    ref_price=px_under,
                    bps=limit_bps,
                    order_ref=hedge_ref,
                    exec_cfg=exec_cfg,
                    timeout=timeout_sec,
                    max_retries=max_retries,
                    dry_run=dry_run,
                    context=f"HARVEST|{under}",
                    cancel_service=cancel_service,
                )
                filled_under_sh = int(hedge_res.filled)
                hedge_status = hedge_res.status
                attempted_rows.append(
                    {
                        "symbol": etf,
                        "underlying": under,
                        "leg": "UNDER_HEDGE",
                        "action": "BUY",
                        "requested_sh": hedge_target_sh,
                        "filled_sh": filled_under_sh,
                        "status": hedge_res.status,
                        "error_code": hedge_res.error_code,
                        "error_msg": hedge_res.error_msg,
                        "price_ref": px_under,
                        "beta": beta,
                        "ftp_available": None,
                        "ftp_borrow_annual": None,
                    }
                )

            remaining_short_usd = max(0.0, need_usd - (filled_short_sh * px_etf))
            residual_beta_usd = (filled_short_sh * px_etf * beta) - (filled_under_sh * px_under)
            summary_rows.append(
                {
                    "symbol": etf,
                    "underlying": under,
                    "requested_short_usd": need_usd,
                    "requested_short_sh": requested_short_sh,
                    "filled_short_sh": filled_short_sh,
                    "filled_under_buy_sh": filled_under_sh,
                    "remaining_short_usd": remaining_short_usd,
                    "residual_beta_usd_after_hedge": residual_beta_usd,
                    "status": f"SHORT_{short_res.status}|HEDGE_{hedge_status}",
                }
            )
            fill_rows.append(
                {
                    "symbol": etf,
                    "underlying": under,
                    "filled_short_sh": filled_short_sh,
                    "filled_under_buy_sh": filled_under_sh,
                    "etf_px": px_etf,
                    "under_px": px_under,
                    "beta": beta,
                    "short_fill_notional_usd": filled_short_sh * px_etf,
                    "hedge_fill_notional_usd": filled_under_sh * px_under,
                    "residual_beta_usd_after_hedge": residual_beta_usd,
                }
            )
    finally:
        try:
            cancel_service.stop()
        except Exception:
            pass
        try:
            ib.disconnect()
        except Exception:
            pass

    pd.DataFrame(attempted_rows).to_csv(attempted_path, index=False)
    pd.DataFrame(fill_rows).to_csv(fills_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    tprint(f"[HARVEST] Wrote candidates -> {candidates_path}")
    tprint(f"[HARVEST] Wrote attempted trades -> {attempted_path}")
    tprint(f"[HARVEST] Wrote fills -> {fills_path}")
    tprint(f"[HARVEST] Wrote post-trade summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
