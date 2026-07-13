"""Full production GTP sizing for historical replay (isolated state).

Runs today's ``generate_trade_plan`` sleeve stack (B1/B2 + B4
opt2 → crash → smooth → ratchet) against an arbitrary screened DataFrame
without touching live ``data/*_state.json`` or overwriting live run archives.
"""

from __future__ import annotations

import copy
import io
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _norm_sym(x: Any) -> str:
    return str(x).strip().upper().replace(".", "-")


def remap_cfg_state_paths(cfg: dict[str, Any], state_root: Path) -> dict[str, Any]:
    """Return a deep-copied cfg with all GTP state / output paths under ``state_root``."""
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    out = copy.deepcopy(cfg)
    paths = dict(out.get("paths") or {})
    paths["screened_csv"] = str(root / "etf_screened_today.csv")
    paths["proposed_trades_csv"] = str(root / "proposed_trades.csv")
    paths["core_leveraged_decay_state_json"] = str(root / "core_leveraged_decay_state.json")
    paths["liquidity_gap_state_json"] = str(root / "liquidity_gap_state.json")
    out["paths"] = paths

    sleeves = dict((out.get("portfolio") or {}).get("sleeves") or {})
    b4 = dict(sleeves.get("inverse_decay_bucket4") or {})
    rules = dict(b4.get("rules") or {})
    opt2 = dict(rules.get("bucket4_weekly_opt2") or {})

    crash = dict(opt2.get("crash_budget") or {})
    crash["l_state_json"] = str(root / "b4_crash_l_state.json")
    opt2["crash_budget"] = crash

    ws = dict(opt2.get("weight_smoothing") or {})
    ws["state_json"] = str(root / "b4_weight_ema_state.json")
    opt2["weight_smoothing"] = ws

    hcp = dict(opt2.get("hedge_cadence_policy") or {})
    hcp["cadence_state_json"] = str(root / "b4_cadence_state.json")
    opt2["hedge_cadence_policy"] = hcp
    rules["bucket4_weekly_opt2"] = opt2

    ratchet = dict(rules.get("ratchet") or {})
    ratchet["state_json"] = str(root / "b4_inverse_ratchet_state.json")
    rules["ratchet"] = ratchet

    life = dict(rules.get("pair_lifecycle") or {})
    life["state_json"] = str(root / "b4_pair_lifecycle_state.json")
    rules["pair_lifecycle"] = life

    gov = dict(rules.get("governor") or {})
    gov["state_json"] = str(root / "b4_governor_state.json")
    rules["governor"] = gov

    b4["rules"] = rules
    sleeves["inverse_decay_bucket4"] = b4
    port = dict(out.get("portfolio") or {})
    port["sleeves"] = sleeves
    out["portfolio"] = port
    return out


def held_from_plan(plan: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Build ratchet-style held legs from a sized plan (assume prior plan filled)."""
    if plan is None or plan.empty:
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    df = plan.copy()
    if "sleeve" in df.columns:
        df = df[
            df["sleeve"]
            .astype(str)
            .str.lower()
            .isin({"inverse_decay_bucket4", "bucket4", "bucket_4"})
        ]
    for _, r in df.iterrows():
        etf = _norm_sym(r.get("ETF"))
        und = _norm_sym(r.get("Underlying"))
        if not etf or not und:
            continue
        inv = abs(float(pd.to_numeric(r.get("etf_target_usd", r.get("short_usd")), errors="coerce") or 0.0))
        und_s = abs(
            float(pd.to_numeric(r.get("underlying_target_usd", r.get("long_usd")), errors="coerce") or 0.0)
        )
        if inv <= 0.0 and und_s <= 0.0:
            continue
        out[(etf, und)] = {
            "inverse_etf_short_usd": inv,
            "underlying_short_usd": und_s,
        }
    return out


def _seed_empty_state_files(state_root: Path) -> None:
    root = Path(state_root)
    defaults = {
        "core_leveraged_decay_state.json": "{}",
        "b4_crash_l_state.json": '{"stage":"crash_l","weight_by_pair":{}}',
        "b4_weight_ema_state.json": '{"stage":"post_cap_scale","weight_by_pair":{},"own_risk_by_pair":{}}',
        "b4_inverse_ratchet_state.json": '{"inverse_short_usd_by_pair":{}}',
        "b4_cadence_state.json": "{}",
        "b4_pair_lifecycle_state.json": "{}",
        "b4_governor_state.json": "{}",
        "liquidity_gap_state.json": "{}",
    }
    for name, payload in defaults.items():
        p = root / name
        if not p.exists():
            p.write_text(payload, encoding="utf-8")


@contextmanager
def _quiet_stdio(quiet: bool):
    if not quiet:
        yield
        return
    buf = io.StringIO()
    with redirect_stdout(buf):
        yield


def size_book_from_screened(
    screened: pd.DataFrame,
    run_date: str,
    cfg: dict[str, Any],
    *,
    state_root: Path | str,
    held_inverse_short_by_pair: dict[tuple[str, str], dict[str, float]] | None = None,
    quiet: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full production GTP sizing into an isolated ``state_root``.

    Returns ``(proposed_trades_df, diag)``. Does not overwrite live production
    state under repo ``data/*.json`` or live ``data/runs/*/proposed_trades.csv``.
    """
    import generate_trade_plan as gtp

    root = Path(state_root)
    if not root.is_absolute():
        root = (REPO / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _seed_empty_state_files(root)

    cfg_iso = remap_cfg_state_paths(cfg, root)
    screened = screened.copy()
    if "Delta" not in screened.columns and "Beta" in screened.columns:
        screened["Delta"] = screened["Beta"]
    screened_path = Path(cfg_iso["paths"]["screened_csv"])
    screened.to_csv(screened_path, index=False)

    held = held_inverse_short_by_pair or {}

    def _fake_held(rd: str, *, runs_root: Path | None = None):  # noqa: ARG001
        return dict(held)

    def _fake_run_dir(rd: str) -> Path:
        p = root / "runs" / str(rd)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _noop_verify(*_a, **_k):
        return None

    diag: dict[str, Any] = {
        "run_date": str(run_date),
        "state_root": str(root),
        "n_screened": int(len(screened)),
        "n_held_pairs": int(len(held)),
    }

    with _quiet_stdio(quiet), patch.object(gtp, "load_config", return_value=cfg_iso), patch.object(
        gtp, "run_dir", _fake_run_dir
    ), patch(
        "scripts.b4_reconstruct_state.held_inverse_short_by_pair",
        _fake_held,
    ), patch(
        "scripts.b4_plan_contract.verify_b4_gtp_artifacts",
        _noop_verify,
    ), patch.object(sys, "argv", ["generate_trade_plan.py", "--run-date", str(run_date)]):
        gtp.main()

    proposed_path = root / "runs" / str(run_date) / "proposed_trades.csv"
    latest = Path(cfg_iso["paths"]["proposed_trades_csv"])
    if not proposed_path.is_file() and latest.is_file():
        proposed_path = latest
    if not proposed_path.is_file():
        raise RuntimeError(f"GTP sizing produced no proposed_trades at {proposed_path}")

    plan = pd.read_csv(proposed_path)
    diag["n_plan_rows"] = int(len(plan))
    if "gross_target_usd" in plan.columns:
        g = pd.to_numeric(plan["gross_target_usd"], errors="coerce").fillna(0.0)
        diag["gross_sum"] = float(g.sum())
        if "sleeve" in plan.columns:
            for sleeve, sub in plan.groupby(plan["sleeve"].astype(str)):
                diag[f"gross_{sleeve}"] = float(
                    pd.to_numeric(sub["gross_target_usd"], errors="coerce").fillna(0.0).sum()
                )
    return plan, diag
