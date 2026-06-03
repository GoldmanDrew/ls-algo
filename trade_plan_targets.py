"""Shared trade-plan target column helpers (no IB / execution dependencies)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from generate_trade_plan import run_dir


def resolve_target_basis_columns(plan: pd.DataFrame, target_basis: str) -> tuple[str, str]:
    """Return the ``(long_col, short_col)`` to use for target sizing.

    - ``optimal`` (default): ``optimal_long_usd`` / ``optimal_short_usd`` if present in
      ``proposed_trades.csv`` (or merged from ``optimal_targets.csv``). Falls back to
      executable when columns are missing (legacy CSVs).
    - ``executable`` (legacy): ``long_usd`` / ``short_usd``.
    - ``max``: row-level max(|optimal|, |executable|), signed by sleeve direction. Implemented
      by callers (this helper just resolves which absolute columns to read).
    """
    basis = str(target_basis or "optimal").strip().lower()
    if basis == "executable":
        return ("long_usd", "short_usd")
    if "optimal_long_usd" in plan.columns and "optimal_short_usd" in plan.columns:
        opt_l = pd.to_numeric(plan["optimal_long_usd"], errors="coerce").abs().fillna(0.0).sum()
        opt_s = pd.to_numeric(plan["optimal_short_usd"], errors="coerce").abs().fillna(0.0).sum()
        if (opt_l + opt_s) > 1e-9:
            return ("optimal_long_usd", "optimal_short_usd")
    return ("long_usd", "short_usd")


def maybe_merge_optimal_targets(
    plan: pd.DataFrame,
    run_date: str,
    *,
    runs_root: Path | None = None,
) -> pd.DataFrame:
    """Merge ``optimal_targets.csv`` into ``plan`` when available.

    Lets callers use ``optimal_*`` even if the in-hand ``proposed_trades.csv`` predates the
    dual-pipeline output.
    """
    if plan is None or plan.empty:
        return plan
    if "optimal_long_usd" in plan.columns and "optimal_short_usd" in plan.columns:
        return plan
    if runs_root is not None:
        optimal_path = Path(runs_root) / run_date / "optimal_targets.csv"
    else:
        optimal_path = run_dir(run_date) / "optimal_targets.csv"
    if not optimal_path.exists():
        return plan
    try:
        opt = pd.read_csv(optimal_path)
    except Exception:
        return plan
    if opt.empty or "ETF" not in opt.columns or "Underlying" not in opt.columns:
        return plan
    keep_cols = [
        c
        for c in opt.columns
        if c.startswith("optimal_")
        or c in ("ETF", "Underlying", "binding_cap", "liquidity_gap_usd")
    ]
    opt = opt[keep_cols].copy()
    return plan.merge(opt, on=["ETF", "Underlying"], how="left", suffixes=("", "_opt"))


# Back-compat aliases used by harvest / tests
_resolve_target_basis_columns = resolve_target_basis_columns
_maybe_merge_optimal_targets = maybe_merge_optimal_targets
