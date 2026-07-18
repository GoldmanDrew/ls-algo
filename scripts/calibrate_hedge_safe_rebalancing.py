"""Compare full-chase, legacy pacing, and hedge-safe replay on one cached input.

The script is deliberately read-only with respect to strategy configuration and
cached plans. Each arm receives the same normalized plan timeline, price panel,
cost model, and B4 execution settings; only ``turnover_pace_mode`` changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.production_actual_backtest import (  # noqa: E402
    STOCK_SLEEVES,
    rebalance_knobs,
    simulate_book_from_plan_timeline,
    sleeve_budgets_usd,
    normalize_plan,
)
from scripts.sizing_tilt_cadence_bt import load_price_panel  # noqa: E402
from strategy_config import load_config  # noqa: E402


ARMS = (
    ("full_chase_off", "off"),
    ("current_pacing_legacy", "legacy"),
    ("hedge_safe_v1", "hedge_safe_v1"),
)


def load_cached_plan_timeline(
    plans_dir: Path, *, start: pd.Timestamp, end: pd.Timestamp
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Load date-named cached plans once without touching their source files."""
    timeline: dict[pd.Timestamp, pd.DataFrame] = {}
    for path in sorted(Path(plans_dir).glob("*.csv")):
        try:
            day = pd.Timestamp(path.stem).normalize()
        except Exception:
            continue
        if day < start or day > end:
            continue
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        plan = normalize_plan(raw, source_date=str(day.date()))
        if not plan.empty:
            timeline[day] = plan
    return timeline


def _delta_by_plan(
    timeline: dict[pd.Timestamp, pd.DataFrame],
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for day, plan in timeline.items():
        for _, row in plan.iterrows():
            delta = pd.to_numeric(row.get("Delta"), errors="coerce")
            if pd.notna(delta) and np.isfinite(float(delta)):
                out[(str(day.date()), str(row.get("ETF", "")).upper())] = float(delta)
    return out


def hedge_drift_diagnostics(
    pair_daily: pd.DataFrame,
    timeline: dict[pd.Timestamp, pd.DataFrame],
    *,
    long_trigger: float,
    short_trigger: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute stock-residual Phase-3 drift plus informational all-sleeve drift."""
    if pair_daily is None or pair_daily.empty:
        return pd.DataFrame(), {
            "hedge_breach_group_days": 0,
            "hedge_too_long_group_days": 0,
            "hedge_too_short_group_days": 0,
            "max_abs_hedge_net_pct": np.nan,
            "orphan_pair_days": 0,
            "missing_delta_rows": 0,
            "missing_delta_group_days": 0,
            "raw_all_sleeve_hedge_breach_group_days": 0,
            "raw_all_sleeve_max_abs_hedge_net_pct": np.nan,
            "raw_all_sleeve_missing_delta_rows": 0,
            "raw_all_sleeve_missing_delta_group_days": 0,
        }
    df = pair_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "sleeve" not in df.columns:
        # Backward-compatible diagnostics for older pair ledgers/tests.
        df["sleeve"] = STOCK_SLEEVES[0]
    lookup = _delta_by_plan(timeline)
    fallback_delta = pd.Series([
        lookup.get((str(plan_date), str(etf).upper()), np.nan)
        for plan_date, etf in zip(df["active_plan_date"], df["ETF"])
    ], index=df.index, dtype=float)
    position_delta = pd.to_numeric(
        df.get("Delta", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    position_valid = position_delta.notna() & np.isfinite(position_delta)
    df["_delta"] = position_delta.where(position_valid, fallback_delta)
    df["_delta_source"] = np.where(
        position_valid,
        "position",
        np.where(fallback_delta.notna(), "active_plan_fallback", "missing"),
    )
    df["_position_gross"] = (
        pd.to_numeric(df["underlying_usd"], errors="coerce").fillna(0.0).abs()
        + pd.to_numeric(df["etf_usd"], errors="coerce").fillna(0.0).abs()
    )
    df["_net"] = (
        pd.to_numeric(df["underlying_usd"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["etf_usd"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["_delta"], errors="coerce")
    )
    df["_gross"] = (
        pd.to_numeric(df["underlying_usd"], errors="coerce").fillna(0.0).abs()
        + (
            pd.to_numeric(df["etf_usd"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["_delta"], errors="coerce")
        ).abs()
    )
    def _aggregate(
        scope: pd.DataFrame, prefix: str
    ) -> tuple[pd.DataFrame, int, pd.DataFrame]:
        exposed = scope[scope["_position_gross"] > 1e-6].copy()
        missing = exposed[exposed["_delta"].isna()]
        missing_groups = missing[["date", "Underlying"]].drop_duplicates()
        if missing_groups.empty:
            complete = exposed
        else:
            incomplete_keys = pd.MultiIndex.from_frame(missing_groups)
            row_keys = pd.MultiIndex.from_frame(
                exposed[["date", "Underlying"]]
            )
            complete = exposed.loc[~row_keys.isin(incomplete_keys)]
        out = (
            complete
            .groupby(["date", "Underlying"], as_index=False)
            .agg(
                **{
                    f"{prefix}hedge_net_usd": ("_net", "sum"),
                    f"{prefix}hedge_gross_usd": ("_gross", "sum"),
                    f"{prefix}n_pairs": ("ETF", "nunique"),
                }
            )
        )
        pct_col = f"{prefix}hedge_net_pct"
        breach_col = f"{prefix}breach"
        out[pct_col] = np.where(
            out[f"{prefix}hedge_gross_usd"] > 0,
            out[f"{prefix}hedge_net_usd"] / out[f"{prefix}hedge_gross_usd"],
            np.nan,
        )
        out[breach_col] = np.where(
            out[pct_col] > float(long_trigger),
            "too_long",
            np.where(out[pct_col] < -float(short_trigger), "too_short", ""),
        )
        return out, int(len(missing)), missing_groups

    stock_df = df[df["sleeve"].astype(str).isin(STOCK_SLEEVES)].copy()
    stock_drift, stock_missing_rows, stock_missing_groups = _aggregate(
        stock_df, "stock_"
    )
    raw_drift, raw_missing_rows, raw_missing_groups = _aggregate(
        df, "raw_all_sleeve_"
    )
    drift = raw_drift.merge(
        stock_drift, on=["date", "Underlying"], how="outer"
    ).sort_values(["date", "Underlying"])
    for groups, column in (
        (stock_missing_groups, "stock_missing_delta"),
        (raw_missing_groups, "raw_all_sleeve_missing_delta"),
    ):
        marker = groups.copy()
        marker[column] = True
        drift = drift.merge(
            marker, on=["date", "Underlying"], how="outer"
        )
        drift[column] = drift[column].fillna(False)
    drift["hedge_net_usd"] = drift.get("stock_hedge_net_usd")
    drift["hedge_gross_usd"] = drift.get("stock_hedge_gross_usd")
    drift["n_pairs"] = drift.get("stock_n_pairs")
    drift["hedge_net_pct"] = drift.get("stock_hedge_net_pct")
    drift["breach"] = drift.get("stock_breach").fillna("")

    etf_open = (
        pd.to_numeric(stock_df["etf_usd"], errors="coerce").fillna(0.0).abs()
        > 1e-6
    )
    und_open = (
        pd.to_numeric(stock_df["underlying_usd"], errors="coerce")
        .fillna(0.0)
        .abs()
        > 1e-6
    )
    raw_breach = drift["raw_all_sleeve_breach"].fillna("")
    metrics = {
        "hedge_breach_group_days": int(drift["breach"].ne("").sum()),
        "hedge_too_long_group_days": int(drift["breach"].eq("too_long").sum()),
        "hedge_too_short_group_days": int(drift["breach"].eq("too_short").sum()),
        "max_abs_hedge_net_pct": (
            float(drift["hedge_net_pct"].abs().max()) if not drift.empty else np.nan
        ),
        "orphan_pair_days": int((etf_open ^ und_open).sum()),
        "missing_delta_rows": stock_missing_rows,
        "missing_delta_group_days": int(len(stock_missing_groups)),
        "raw_all_sleeve_hedge_breach_group_days": int(
            raw_breach.ne("").sum()
        ),
        "raw_all_sleeve_hedge_too_long_group_days": int(
            raw_breach.eq("too_long").sum()
        ),
        "raw_all_sleeve_hedge_too_short_group_days": int(
            raw_breach.eq("too_short").sum()
        ),
        "raw_all_sleeve_max_abs_hedge_net_pct": (
            float(drift["raw_all_sleeve_hedge_net_pct"].abs().max())
            if not drift.empty
            else np.nan
        ),
        "raw_all_sleeve_missing_delta_rows": raw_missing_rows,
        "raw_all_sleeve_missing_delta_group_days": int(
            len(raw_missing_groups)
        ),
    }
    return drift, metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def select_sensitivity_defaults(
    sensitivity: pd.DataFrame,
    *,
    legacy_breach_group_days: int,
    expected_b4_cadence_rebals: int | None = None,
    turnover_target_usd: float = 10_000_000.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Gate fidelity/turnover, then minimize hedge risk before turnover."""
    ranked = sensitivity.copy()
    blockers = pd.to_numeric(
        ranked.get(
            "explicit_hedge_blocker_count",
            pd.Series(0, index=ranked.index),
        ),
        errors="coerce",
    ).fillna(0)
    ranked["post_blocker_hedge_breaches"] = (
        pd.to_numeric(
            ranked["hedge_breach_group_days"], errors="coerce"
        ).fillna(np.inf)
        - blockers
    ).clip(lower=0)
    missing_delta_rows = pd.to_numeric(
        ranked.get(
            "missing_delta_rows",
            pd.Series(0, index=ranked.index),
        ),
        errors="coerce",
    ).fillna(np.inf)
    missing_delta_groups = pd.to_numeric(
        ranked.get(
            "missing_delta_group_days",
            pd.Series(0, index=ranked.index),
        ),
        errors="coerce",
    ).fillna(np.inf)
    ranked["hedge_safety_pass"] = (
        (pd.to_numeric(ranked["orphan_pair_days"], errors="coerce").fillna(np.inf) == 0)
        & (missing_delta_rows == 0)
        & (missing_delta_groups == 0)
    )
    ranked["deployment_pass"] = (
        pd.to_numeric(
            ranked["median_deployed_desired_gross_ratio"], errors="coerce"
        ).fillna(-np.inf)
        >= 0.80
    ) & (
        pd.to_numeric(
            ranked["ending_deployed_desired_gross_ratio"], errors="coerce"
        ).fillna(-np.inf)
        >= 0.75
    )
    cadence = pd.to_numeric(
        ranked["n_b4_cadence_rebals"], errors="coerce"
    ).fillna(0).astype(int)
    positive_cadence = cadence[cadence > 0]
    expected_cadence = (
        int(expected_b4_cadence_rebals)
        if expected_b4_cadence_rebals is not None
        else (
            int(positive_cadence.mode().iloc[0])
            if not positive_cadence.empty
            else 0
        )
    )
    ranked["b4_cadence_pass"] = (
        (cadence > 0) & (cadence == expected_cadence)
    )
    ranked["hard_gates_pass"] = (
        ranked["hedge_safety_pass"]
        & ranked["deployment_pass"]
        & ranked["b4_cadence_pass"]
    )
    ranked["turnover_target_pass"] = (
        pd.to_numeric(ranked["turnover_usd"], errors="coerce").fillna(np.inf)
        <= float(turnover_target_usd)
    )
    ranked["hard_gates_pass"] &= ranked["turnover_target_pass"]
    ranked["p10_deployment_preferred"] = (
        pd.to_numeric(
            ranked.get(
                "p10_deployed_desired_gross_ratio",
                pd.Series(np.nan, index=ranked.index),
            ),
            errors="coerce",
        ).fillna(-np.inf)
        >= 0.50
    )
    stability_group = ["max_daily_turnover_pct"]
    if "target_blend_alpha" in ranked.columns:
        stability_group.insert(0, "target_blend_alpha")
    passing_count_by_cap = ranked.groupby(stability_group)[
        "hard_gates_pass"
    ].transform("sum")
    ranked["stable_region_pass"] = passing_count_by_cap >= 2
    eligible = ranked[ranked["hard_gates_pass"]].copy()
    if eligible.empty:
        raise ValueError("no hedge-safe sensitivity candidate passed all hard gates")
    if "max_abs_hedge_net_pct" not in eligible.columns:
        eligible["max_abs_hedge_net_pct"] = np.inf
    for column in (
        "target_blend_alpha",
        "max_daily_turnover_pct",
        "remaining_gap_rate",
    ):
        if column not in eligible.columns:
            eligible[column] = 0.0
    eligible = eligible.sort_values(
        [
            "hedge_breach_group_days",
            "max_abs_hedge_net_pct",
            "turnover_usd",
            "txn_cost_usd",
            "stable_region_pass",
            "target_blend_alpha",
            "max_daily_turnover_pct",
            "remaining_gap_rate",
        ],
        ascending=[True, True, True, True, False, True, True, True],
        kind="stable",
    )
    selected_idx = eligible.index[0]
    ranked["selected"] = ranked.index == selected_idx
    return ranked.loc[selected_idx], ranked


def run_calibration(
    *,
    run_date: str,
    start: str,
    plans_dir: Path,
    outdir: Path,
    price_panel_min_days: int = 20,
    config_path: Path | None = None,
    reuse_sensitivity: bool = False,
    extend_sensitivity: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all three arms and write isolated calibration artifacts."""
    cfg = load_config(config_path or REPO / "config" / "strategy_config.yml")
    knobs = rebalance_knobs(cfg)
    budgets = sleeve_budgets_usd(cfg)
    capital = float(cfg["strategy"]["capital_usd"])
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(run_date).normalize()
    timeline = load_cached_plan_timeline(plans_dir, start=start_ts, end=end_ts)
    if not timeline:
        raise FileNotFoundError(f"no cached plans in {plans_dir} for {start}..{run_date}")
    panel = load_price_panel(run_date, min_days=int(price_panel_min_days))
    if not panel:
        raise RuntimeError(f"price panel unavailable for {run_date}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "run_date": run_date,
        "start": start,
        "plans_dir": str(Path(plans_dir)),
        "price_panel_min_days": int(price_panel_min_days),
        "n_plans": len(timeline),
        "n_price_pairs": len(panel),
        "arms": {},
    }
    common = dict(
        budgets=budgets,
        capital_usd=capital,
        start=start_ts,
        slippage_bps=float(knobs["slippage_bps"]),
        commission_per_share=float(knobs["commission_per_share"]),
        margin_rate_annual=float(knobs["margin_rate_annual"]),
        financing_daycount=float(knobs["financing_daycount"]),
        short_proceeds_credit_annual=float(knobs["short_proceeds_credit_annual"]),
        execution_lag_sessions=int(knobs["execution_lag_sessions"]),
        target_notional_mode=str(knobs["target_notional_mode"]),
        scale_sleeves_to_budget=bool(knobs.get("scale_sleeves_to_budget", True)),
        enter_band_pct=float(knobs["enter_band_pct"]),
        exit_band_pct=float(knobs["exit_band_pct"]),
        min_trade_usd=float(knobs["min_trade_usd"]),
        check_freq="W-FRI",
        retarget_on_plan_change=False,
        pre_archive_policy="cash",
        b4_execution=str(knobs.get("b4_execution", "cadence")),
        apply_delist_flatten=bool(knobs.get("apply_delist_flatten", True)),
        use_borrow_history=bool(knobs.get("use_borrow_history", True)),
        same_run_churn_enabled=bool(knobs.get("same_run_churn_enabled", True)),
        purgatory_model_zero_policy=str(
            knobs.get("purgatory_model_zero_policy", "hold")
        ),
        b4_membership_clock=str(knobs.get("b4_membership_clock", "operator_5d")),
        operator_check_days=int(knobs.get("operator_check_days", 5)),
        b4_apply_resize_bands=bool(knobs.get("b4_apply_resize_bands", True)),
        b4_ratchet_execution_guard=bool(
            knobs.get("b4_ratchet_execution_guard", True)
        ),
        b4_allow_inverse_cover=bool(knobs.get("b4_allow_inverse_cover", True)),
        b4_empty_plan_policy=str(knobs.get("b4_empty_plan_policy", "hold")),
        net_shared_underlyings=bool(knobs.get("net_shared_underlyings", True)),
        confirmation_count=int(knobs.get("confirmation_count", 2)),
        entry_ramp_sessions=int(knobs.get("entry_ramp_sessions", 5)),
        reduction_ramp_sessions=int(knobs.get("reduction_ramp_sessions", 3)),
        remaining_gap_rate=float(knobs.get("remaining_gap_rate", 0.25)),
        target_blend_alpha=float(knobs.get("target_blend_alpha", 0.25)),
        hedge_reserve_frac=float(knobs.get("hedge_reserve_frac", 0.20)),
        adv_participation_pct=float(knobs.get("adv_participation_pct", 0.10)),
        sleeve_gross_ema_alpha=float(knobs.get("sleeve_gross_ema_alpha", 0.35)),
        max_leg_step_pct=float(knobs.get("max_leg_step_pct", 0.25)),
        pair_gross_ramp_pct=float(knobs.get("pair_gross_ramp_pct", 0.25)),
        max_daily_turnover_pct=float(knobs.get("max_daily_turnover_pct", 0.15)),
        legacy_max_daily_turnover_pct=float(
            knobs.get("legacy_max_daily_turnover_pct", 0.15)
        ),
        establish_budget_frac=float(knobs.get("establish_budget_frac", 0.50)),
        resize_age_boost_days=int(knobs.get("resize_age_boost_days", 5)),
        hedge_long_trigger_net_pct=float(
            knobs.get("hedge_long_trigger_net_pct", 0.04)
        ),
        hedge_long_target_net_pct=float(
            knobs.get("hedge_long_target_net_pct", 0.01)
        ),
        hedge_short_trigger_net_pct=float(
            knobs.get("hedge_short_trigger_net_pct", 0.01)
        ),
        hedge_short_target_net_pct=float(
            knobs.get("hedge_short_target_net_pct", 0.00)
        ),
    )

    def run_arm(
        arm: str,
        mode: str,
        *,
        overrides: dict[str, Any] | None = None,
        write_artifacts: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        print(f"[calibration] running {arm} mode={mode}", flush=True)
        arm_common = dict(common)
        arm_common.update(overrides or {})
        nav, audit, meta, pair_stats, daily = simulate_book_from_plan_timeline(
            timeline,
            panel,
            turnover_pace_mode=mode,
            turnover_pace_enabled=mode != "off",
            **arm_common,
        )
        pair_daily = meta.pop("pair_daily", pd.DataFrame())
        pending = meta.pop("pending_target_audit", pd.DataFrame())
        drift, hedge_metrics = hedge_drift_diagnostics(
            pair_daily,
            timeline,
            long_trigger=float(knobs.get("hedge_long_trigger_net_pct", 0.04)),
            short_trigger=float(knobs.get("hedge_short_trigger_net_pct", 0.01)),
        )
        blocked = (
            pending["block_reason"].dropna().astype(str).value_counts().to_dict()
            if isinstance(pending, pd.DataFrame)
            and not pending.empty
            and "block_reason" in pending.columns
            else {}
        )
        deferred = (
            pd.to_numeric(pending.get("deferred_turnover_usd"), errors="coerce")
            if isinstance(pending, pd.DataFrame) and not pending.empty
            else pd.Series(dtype=float)
        )
        deferred_rows = (
            pending.loc[deferred.fillna(0.0) > 1e-9]
            if isinstance(pending, pd.DataFrame) and not pending.empty
            else pd.DataFrame()
        )
        desired = pd.to_numeric(
            daily.get(
                "confirmed_desired_gross_usd",
                pd.Series(np.nan, index=daily.index),
            ),
            errors="coerce",
        )
        deployed_ratio = pd.to_numeric(
            daily.get(
                "deployed_desired_gross_ratio",
                pd.Series(np.nan, index=daily.index),
            ),
            errors="coerce",
        )
        deployment_mask = desired.gt(1e-9) & deployed_ratio.notna()
        representative_ratios = deployed_ratio.loc[deployment_mask]
        explicit_hedge_blocker_count = int(
            sum(
                int(blocked.get(reason, 0) or 0)
                for reason in (
                    "no_normal_pair",
                    "delta_unavailable",
                    "locate_unavailable",
                    "hedge_infeasible",
                )
            )
        )
        row = {
            "arm": arm,
            "turnover_pace_mode": mode,
            "turnover_usd": float(meta.get("turnover_usd", 0.0) or 0.0),
            "turnover_l1": float(meta.get("turnover_l1", 0.0) or 0.0),
            "txn_cost_usd": (
                float(pd.to_numeric(daily["daily_txn_cost"], errors="coerce").sum())
                if not daily.empty and "daily_txn_cost" in daily.columns
                else 0.0
            ),
            "start_usd": float(meta.get("start_usd", capital)),
            "end_usd": float(meta.get("end_usd", np.nan)),
            "total_return": (
                float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) > 1 else np.nan
            ),
            "cagr": float(meta.get("cagr", np.nan)),
            "vol": float(meta.get("vol", np.nan)),
            "sharpe": float(meta.get("sharpe", np.nan)),
            "maxdd": float(meta.get("maxdd", np.nan)),
            **hedge_metrics,
            "median_deployed_desired_gross_ratio": (
                float(representative_ratios.median())
                if not representative_ratios.empty
                else np.nan
            ),
            "p10_deployed_desired_gross_ratio": (
                float(representative_ratios.quantile(0.10))
                if not representative_ratios.empty
                else np.nan
            ),
            "ending_deployed_desired_gross_ratio": (
                float(representative_ratios.iloc[-1])
                if not representative_ratios.empty
                else np.nan
            ),
            "deployment_observation_days": int(len(representative_ratios)),
            "n_b4_cadence_rebals": int(meta.get("n_b4_cadence_rebals", 0) or 0),
            "explicit_hedge_blocker_count": explicit_hedge_blocker_count,
            "avoided_round_trip_usd": float(
                meta.get("avoided_round_trip_usd", 0.0) or 0.0
            ),
            "risk_override_turnover_usd": float(
                meta.get("risk_override_turnover_usd", 0.0) or 0.0
            ),
            "same_run_churn_enabled": bool(meta.get("same_run_churn_enabled", True)),
            "max_deferred_age": (
                int(pd.to_numeric(deferred_rows["target_age"], errors="coerce").max())
                if not deferred_rows.empty
                else 0
            ),
            "p95_deferred_age": (
                float(
                    pd.to_numeric(
                        deferred_rows["target_age"], errors="coerce"
                    ).quantile(0.95)
                )
                if not deferred_rows.empty
                else 0.0
            ),
            "deferred_turnover_usd": float(deferred.fillna(0.0).sum()),
            "blocked_reason_counts_json": json.dumps(blocked, sort_keys=True),
        }
        if write_artifacts:
            arm_dir = outdir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            nav.rename("nav").to_csv(arm_dir / "daily_nav.csv")
            audit.to_csv(arm_dir / "rebalance_audit.csv", index=False)
            daily.to_csv(arm_dir / "daily_diagnostics.csv", index=False)
            pair_stats.to_csv(arm_dir / "pair_stats.csv", index=False)
            pair_daily.to_csv(arm_dir / "pair_daily_pnl.csv", index=False)
            pending.to_csv(arm_dir / "pending_target_audit.csv", index=False)
            drift.to_csv(arm_dir / "hedge_drift_daily.csv", index=False)
            (arm_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, default=_json_safe), encoding="utf-8"
            )
        detail = {
            **{k: _json_safe(v) for k, v in row.items()},
            "blocked_reason_counts": blocked,
        }
        return row, detail

    # Baselines run once. The sensitivity grid reuses the same plans and panel
    # and varies hedge-safe controls only.
    baseline_rows: dict[str, dict[str, Any]] = {}
    for arm, mode in ARMS[:2]:
        row, detail = run_arm(arm, mode, write_artifacts=True)
        baseline_rows[arm] = row
        details["arms"][arm] = detail

    sensitivity_path = outdir / "sensitivity.csv"
    blend_alphas = (0.15, 0.25, 0.27, 0.30, 0.35, 0.50, 1.00)
    grid_points = [
        (blend_alpha, turn_pct, gap_rate)
        for blend_alpha in blend_alphas
        for turn_pct in (0.12, 0.15)
        for gap_rate in (0.20, 0.25)
    ]
    # Local stability check around the only sub-$10m hard-gate boundary.
    grid_points.extend(
        [(0.27, 0.15, 0.21), (0.27, 0.15, 0.22)]
    )
    if reuse_sensitivity:
        if not sensitivity_path.exists():
            raise FileNotFoundError(
                f"cannot reuse missing sensitivity grid: {sensitivity_path}"
            )
        sensitivity = pd.read_csv(sensitivity_path)
    else:
        existing = (
            pd.read_csv(sensitivity_path)
            if extend_sensitivity and sensitivity_path.exists()
            else pd.DataFrame()
        )
        completed = (
            {
                (
                    round(float(r["target_blend_alpha"]), 6),
                    round(float(r["max_daily_turnover_pct"]), 6),
                    round(float(r["remaining_gap_rate"]), 6),
                )
                for _, r in existing.iterrows()
            }
            if not existing.empty
            else set()
        )
        sensitivity_rows: list[dict[str, Any]] = existing.to_dict("records")
        for blend_alpha, turn_pct, gap_rate in grid_points:
            key = (
                round(blend_alpha, 6),
                round(turn_pct, 6),
                round(gap_rate, 6),
            )
            if key in completed:
                continue
            grid_arm = (
                f"hedge_safe_a{blend_alpha:.2f}_"
                f"t{turn_pct:.2f}_r{gap_rate:.2f}"
            )
            row, _ = run_arm(
                grid_arm,
                "hedge_safe_v1",
                overrides={
                    "target_blend_alpha": blend_alpha,
                    "max_daily_turnover_pct": turn_pct,
                    "remaining_gap_rate": gap_rate,
                },
                write_artifacts=False,
            )
            sensitivity_rows.append(
                {
                    **row,
                    "target_blend_alpha": blend_alpha,
                    "max_daily_turnover_pct": turn_pct,
                    "remaining_gap_rate": gap_rate,
                }
            )
        sensitivity = pd.DataFrame(sensitivity_rows)
    selected, sensitivity = select_sensitivity_defaults(
        sensitivity,
        legacy_breach_group_days=int(
            baseline_rows["current_pacing_legacy"]["hedge_breach_group_days"]
        ),
        expected_b4_cadence_rebals=int(
            baseline_rows["current_pacing_legacy"]["n_b4_cadence_rebals"]
        ),
    )
    sensitivity.to_csv(sensitivity_path, index=False)
    selected_overrides = {
        "target_blend_alpha": float(selected["target_blend_alpha"]),
        "max_daily_turnover_pct": float(selected["max_daily_turnover_pct"]),
        "remaining_gap_rate": float(selected["remaining_gap_rate"]),
    }
    safe_row, safe_detail = run_arm(
        "hedge_safe_v1",
        "hedge_safe_v1",
        overrides=selected_overrides,
        write_artifacts=True,
    )
    details["arms"]["hedge_safe_v1"] = safe_detail
    details["sensitivity"] = {
        "selection_rule": (
            "Require zero missing Delta rows/groups, zero orphan pair-days, median "
            "deployment >=80%, ending deployment >=75%, the baseline nonzero B4 cadence "
            "count, and turnover <=$10m; then rank lexicographically by stock-residual "
            "breach group-days, maximum absolute stock-residual drift, turnover, cost, "
            "and stability without PnL."
        ),
        "selected": {k: _json_safe(v) for k, v in selected_overrides.items()},
        "selected_turnover_target_pass": bool(selected["turnover_target_pass"]),
        "selected_hedge_safety_pass": bool(selected["hedge_safety_pass"]),
        "selected_deployment_pass": bool(selected["deployment_pass"]),
        "selected_b4_cadence_pass": bool(selected["b4_cadence_pass"]),
        "selected_stable_region_pass": bool(selected["stable_region_pass"]),
        "selected_p10_deployment_preferred": bool(
            selected["p10_deployment_preferred"]
        ),
    }
    rows = [
        baseline_rows["full_chase_off"],
        baseline_rows["current_pacing_legacy"],
        safe_row,
    ]

    comparison = pd.DataFrame(rows)
    by_arm = comparison.set_index("arm")
    safe = by_arm.loc["hedge_safe_v1"]
    off = by_arm.loc["full_chase_off"]
    legacy = by_arm.loc["current_pacing_legacy"]
    details["acceptance"] = {
        "turnover_below_full_chase": bool(
            safe["turnover_usd"] < off["turnover_usd"]
        ),
        "txn_cost_below_full_chase": bool(
            safe["txn_cost_usd"] < off["txn_cost_usd"]
        ),
        "hedge_breaches_not_above_legacy": bool(
            safe["hedge_breach_group_days"] <= legacy["hedge_breach_group_days"]
        ),
        "no_orphan_pair_days": bool(safe["orphan_pair_days"] == 0),
        "turnover_at_or_below_10m": bool(safe["turnover_usd"] <= 10_000_000.0),
        "post_blocker_hedge_gate_pass": bool(
            max(
                0,
                safe["hedge_breach_group_days"]
                - safe["explicit_hedge_blocker_count"],
            )
            <= 0.25 * legacy["hedge_breach_group_days"]
        ),
        "position_delta_complete": bool(
            safe["missing_delta_rows"] == 0
            and safe["missing_delta_group_days"] == 0
        ),
        "deployment_gate_pass": bool(
            safe["median_deployed_desired_gross_ratio"] >= 0.80
            and safe["ending_deployed_desired_gross_ratio"] >= 0.75
        ),
        "p10_deployment_preferred": bool(
            safe["p10_deployed_desired_gross_ratio"] >= 0.50
        ),
        "b4_cadence_matches_legacy": bool(
            safe["n_b4_cadence_rebals"]
            == legacy["n_b4_cadence_rebals"]
            and safe["n_b4_cadence_rebals"] > 0
        ),
    }
    comparison.to_csv(outdir / "comparison.csv", index=False)
    (outdir / "comparison.json").write_text(
        json.dumps(details, indent=2, default=_json_safe), encoding="utf-8"
    )
    print(comparison.to_string(index=False), flush=True)
    print(f"[calibration] wrote {outdir}", flush=True)
    return comparison, details


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-date", required=True)
    p.add_argument("--start", required=True)
    p.add_argument(
        "--plans-dir",
        type=Path,
        default=REPO / "notebooks" / "output" / "production_actual_bt" / "plans",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=REPO
        / "notebooks"
        / "output"
        / "production_actual_bt"
        / "hedge_safe_calibration",
    )
    p.add_argument("--price-panel-min-days", type=int, default=20)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--reuse-sensitivity",
        action="store_true",
        help="Reuse sensitivity.csv and rerun only final off/legacy/selected arms.",
    )
    p.add_argument(
        "--extend-sensitivity",
        action="store_true",
        help="Reuse completed grid points and run only newly configured points.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_calibration(
        run_date=args.run_date,
        start=args.start,
        plans_dir=args.plans_dir,
        outdir=args.outdir,
        price_panel_min_days=args.price_panel_min_days,
        config_path=args.config,
        reuse_sensitivity=args.reuse_sensitivity,
        extend_sensitivity=args.extend_sensitivity,
    )
