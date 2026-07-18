"""Contract checks for Bucket-4 planning artifacts after ``generate_trade_plan``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "runs"


def validate_purgatory_inverse_sleeves(plan: pd.DataFrame) -> None:
    """Fail closed when a purgatory inverse pair has lost execution ownership."""
    if plan is None or plan.empty or "purgatory" not in plan.columns:
        return
    delta = pd.to_numeric(
        plan.get("Delta", pd.Series(float("nan"), index=plan.index)),
        errors="coerce",
    )
    purgatory = plan["purgatory"].fillna(False).astype(bool)
    sleeve = plan.get("sleeve", pd.Series("", index=plan.index))
    missing_sleeve = sleeve.fillna("").astype(str).str.strip().isin({"", "nan", "None"})
    bad = plan.loc[purgatory & delta.lt(0) & missing_sleeve]
    if bad.empty:
        return
    pairs = ", ".join(
        f"{row.ETF}/{row.Underlying}"
        for row in bad[["ETF", "Underlying"]].itertuples(index=False)
    )
    raise RuntimeError(
        "B4 plan contract failed: purgatory inverse pair(s) missing sleeve identity: "
        + pairs
    )


def verify_b4_gtp_artifacts(
    run_date: str,
    *,
    proposed_path: Path | None = None,
    runs_root: Path | None = None,
) -> None:
    """Raise ``RuntimeError`` when opt2/ratchet outputs are missing."""
    root = runs_root or RUNS
    run_dir = root / str(run_date)
    cadence = run_dir / "b4_hedge_cadence"
    errors: list[str] = []

    if not (cadence / "b4_cadence_explain.csv").is_file():
        errors.append(f"missing {cadence / 'b4_cadence_explain.csv'}")
    if not (cadence / "b4_ratchet_targets.csv").is_file():
        errors.append(f"missing {cadence / 'b4_ratchet_targets.csv'}")

    p = proposed_path or (run_dir / "proposed_trades.csv")
    if not p.is_file():
        p = REPO / "data" / "proposed_trades.csv"
    if p.is_file():
        plan = pd.read_csv(p)
        cols = set(plan.columns)
        for need in ("ratchet_released", "ratchet_trim_lambda"):
            if need not in cols:
                errors.append(f"proposed_trades missing column {need}")
        try:
            validate_purgatory_inverse_sleeves(plan)
        except RuntimeError as exc:
            errors.append(str(exc).removeprefix("B4 plan contract failed: "))
    else:
        errors.append(f"missing proposed_trades at {p}")

    if errors:
        raise RuntimeError("B4 plan contract failed: " + "; ".join(errors))
