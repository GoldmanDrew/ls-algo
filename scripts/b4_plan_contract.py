"""Contract checks for Bucket-4 planning artifacts after ``generate_trade_plan``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "runs"


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
        cols = set(pd.read_csv(p, nrows=0).columns)
        for need in ("ratchet_released", "ratchet_trim_lambda"):
            if need not in cols:
                errors.append(f"proposed_trades missing column {need}")
    else:
        errors.append(f"missing proposed_trades at {p}")

    if errors:
        raise RuntimeError("B4 plan contract failed: " + "; ".join(errors))
